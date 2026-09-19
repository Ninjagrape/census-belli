"""
Agent spec loading and override merging.

Each pipeline stage is configured by ``agents/<stage>.yaml``. This module is
the single place that reads them, so that stage code, the orchestrator and
the tooling commands all see the same spec with the same overrides applied.

Override precedence, lowest to highest:

1. the spec file itself
2. a config file passed with ``--config``
3. individual ``--set key=value`` overrides on the command line

Overrides land in the spec's ``params`` section unless the key is dotted, in
which case it addresses a nested path from the spec root. This keeps the
common case short (``--set mcmc_samples=5000``) while still allowing
``--set retry_policy.max_stage_retries=0``.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, cast

import structlog
import yaml

logger = structlog.get_logger()

__all__ = [
    "AGENTS_DIR",
    "DEFAULT_ENV_FILE",
    "SpecError",
    "available_stages",
    "load_agent_spec",
    "load_env",
    "merge_overrides",
    "parse_set_overrides",
    "spec_path",
]

AGENTS_DIR = Path("agents")
DEFAULT_ENV_FILE = Path(".env")


def load_env(path: Path | str = DEFAULT_ENV_FILE, *, override: bool = False) -> list[str]:
    """Load environment variables from a ``.env`` file, if one exists.

    Nothing in this project read ``.env`` before this function existed. The
    file was written and documented in ``.env.example``, but consumed only by
    a docker-compose setup that is not used on the development machine, so
    ``DATABASE_URL`` and the API keys had to be exported by hand -- which
    surprised two sessions running and left the integration suite skipping
    silently rather than failing loudly.

    An already-exported variable wins by default. A shell that set
    ``DATABASE_URL`` deliberately, or a CI runner injecting a secret, must not
    be quietly overridden by a file left in the working tree.

    Args:
        path: The env file to read. A missing file is not an error: the
            deployed case is real environment variables and no file at all.
        override: Let file values replace variables already in the
            environment. Off by default, for the reason above.

    Returns:
        The names of the variables this call set, for logging. **Never the
        values** -- this file holds real API keys, and the project's rule is
        that a secret is confirmed as set, never printed.
    """
    env_path = Path(path)
    if not env_path.is_file():
        logger.debug("env_file_absent", path=str(env_path))
        return []

    try:
        from dotenv import dotenv_values
    except ImportError:
        logger.warning(
            "dotenv_not_installed",
            path=str(env_path),
            hint="pip install python-dotenv, or export the variables yourself",
        )
        return []

    try:
        values = dotenv_values(env_path, encoding=_env_encoding(env_path))
    except (OSError, UnicodeDecodeError) as exc:
        # An unreadable .env must not take the run down. The variables may
        # well be exported already, and a stack trace from inside dotenv
        # tells the reader nothing about which file or why.
        logger.warning(
            "env_file_unreadable",
            path=str(env_path),
            error=f"{type(exc).__name__}: {exc}",
            hint="re-save it as UTF-8, or export the variables yourself",
        )
        return []

    applied: list[str] = []
    for key, value in values.items():
        if value is None or (key in os.environ and not override):
            continue
        os.environ[key] = value
        applied.append(key)

    logger.info("env_file_loaded", path=str(env_path), variables=sorted(applied))
    return applied


def _env_encoding(path: Path) -> str:
    """Guess an env file's encoding from its byte-order mark.

    PowerShell's ``>`` redirection and ``Set-Content`` without
    ``-Encoding utf8`` write UTF-16-LE with a BOM, and the ``.env`` on this
    project's development machine is exactly that. ``python-dotenv`` assumes
    UTF-8 and raises ``UnicodeDecodeError`` on the first byte, which reads as
    "dotenv is broken" rather than "this file is UTF-16".

    Args:
        path: The env file.

    Returns:
        An encoding name for ``dotenv_values``. Falls back to UTF-8, which is
        what the file should be, when there is no BOM or it cannot be read.
    """
    try:
        head = path.read_bytes()[:4]
    except OSError:
        return "utf-8"

    # "utf-16" rather than "utf-16-le": the explicit-endianness codecs keep
    # the BOM as a character, which then rides along on the first key name and
    # produces a variable called "﻿GEMINI_API_KEY" that nothing reads.
    if head.startswith(b"\xff\xfe\x00\x00"):
        return "utf-32"
    if head.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    if head.startswith(b"\xef\xbb\xbf"):
        # utf-8-sig strips the BOM; plain utf-8 would leave it on the first key.
        return "utf-8-sig"
    return "utf-8"

# Keys every spec must define. A spec missing these fails later in a less
# obvious place, so it is worth catching at load time.
_REQUIRED_KEYS = ("stage", "description")


class SpecError(ValueError):
    """Raised when an agent spec is missing, malformed, or inconsistent."""


def spec_path(stage: str, agents_dir: Path | str = AGENTS_DIR) -> Path:
    """Return the path to a stage's spec file.

    Args:
        stage: The stage name.
        agents_dir: Directory holding the specs.

    Returns:
        The path, which is not guaranteed to exist.
    """
    return Path(agents_dir) / f"{stage}.yaml"


def available_stages(agents_dir: Path | str = AGENTS_DIR) -> list[str]:
    """List the stages that have a spec on disk.

    Args:
        agents_dir: Directory holding the specs.

    Returns:
        Sorted stage names.
    """
    return sorted(p.stem for p in Path(agents_dir).glob("*.yaml"))


def load_agent_spec(
    stage: str,
    overrides: dict[str, Any] | None = None,
    agents_dir: Path | str = AGENTS_DIR,
) -> dict[str, Any]:
    """Load a stage's agent spec, with overrides applied.

    Args:
        stage: The stage name, matching ``agents/<stage>.yaml``.
        overrides: Values to merge in. Bare keys target ``params``; dotted
            keys address a path from the spec root.
        agents_dir: Directory holding the specs.

    Returns:
        The spec as a dict.

    Raises:
        SpecError: If the file is missing, is not a mapping, omits a required
            key, or names a stage inconsistent with its filename.
    """
    path = spec_path(stage, agents_dir)
    if not path.exists():
        known = available_stages(agents_dir)
        raise SpecError(f"Agent spec not found: {path}. Specs on disk: {known}")

    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SpecError(f"{path} is not valid YAML: {exc}") from exc

    if not isinstance(loaded, dict):
        raise SpecError(f"{path} must contain a mapping, got {type(loaded).__name__}")

    missing = [k for k in _REQUIRED_KEYS if k not in loaded]
    if missing:
        raise SpecError(f"{path} is missing required key(s): {missing}")

    if loaded["stage"] != stage:
        raise SpecError(
            f"{path} declares stage {loaded['stage']!r} but is named {stage!r}. "
            "The orchestrator addresses stages by filename, so these must agree."
        )

    # yaml.safe_load is typed Any; the isinstance check above established
    # this is a mapping, so name that for the type checker.
    spec = cast("dict[str, Any]", loaded)

    if overrides:
        spec = merge_overrides(spec, overrides)

    logger.debug(
        "agent_spec_loaded",
        stage=stage,
        checks=len(spec.get("quality_checks") or []),
        overridden=sorted(overrides) if overrides else [],
    )
    return spec


def merge_overrides(spec: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge overrides into a copy of a spec.

    A bare key targets ``params``, since that is what almost every override
    adjusts. A dotted key addresses a path from the spec root, so
    ``retry_policy.max_stage_retries`` reaches outside params when needed.

    The input spec is not mutated: stages may be re-run with different
    overrides within one process, and a mutated spec would leak between runs.

    Args:
        spec: The loaded spec.
        overrides: Values to merge in.

    Returns:
        A new spec with overrides applied.

    Raises:
        SpecError: If a dotted path traverses a non-mapping value.
    """
    merged = copy.deepcopy(spec)

    for key, value in overrides.items():
        if "." in key:
            _set_path(merged, key.split("."), value, original=key)
        else:
            params = merged.setdefault("params", {})
            if not isinstance(params, dict):
                raise SpecError(
                    f"Cannot apply override {key!r}: spec's params is "
                    f"{type(params).__name__}, not a mapping"
                )
            _merge_value(params, key, value)

    return merged


def _merge_value(target: dict[str, Any], key: str, value: Any) -> None:
    """Set one key, recursing when both sides are mappings.

    Args:
        target: The mapping to write into.
        key: The key to set.
        value: The value to set.
    """
    existing = target.get(key)
    if isinstance(existing, dict) and isinstance(value, dict):
        for sub_key, sub_value in value.items():
            _merge_value(existing, sub_key, sub_value)
    else:
        target[key] = value


def _set_path(root: dict[str, Any], path: list[str], value: Any, original: str) -> None:
    """Set a value at a dotted path, creating intermediate mappings.

    Args:
        root: The spec to write into.
        path: Path segments.
        value: The value to set.
        original: The original dotted key, for error messages.

    Raises:
        SpecError: If a segment traverses a non-mapping value.
    """
    node: dict[str, Any] = root
    for segment in path[:-1]:
        nxt = node.setdefault(segment, {})
        if not isinstance(nxt, dict):
            raise SpecError(
                f"Cannot apply override {original!r}: {segment!r} is "
                f"{type(nxt).__name__}, not a mapping"
            )
        node = nxt
    _merge_value(node, path[-1], value)


def parse_set_overrides(pairs: list[str]) -> dict[str, Any]:
    """Parse ``key=value`` strings from the command line.

    Values are parsed as YAML scalars, so ``true``, ``3``, ``0.9`` and
    ``[a, b]`` arrive as the types the spec would have used. A value that
    does not parse is kept as a string, which is what a user typing a bare
    word almost always means.

    Args:
        pairs: Strings of the form ``key=value``.

    Returns:
        A mapping suitable for merge_overrides.

    Raises:
        SpecError: If an entry has no '=' or an empty key.
    """
    parsed: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SpecError(f"Override {pair!r} is not of the form key=value")
        key, _, raw = pair.partition("=")
        key = key.strip()
        if not key:
            raise SpecError(f"Override {pair!r} has an empty key")
        try:
            value: Any = yaml.safe_load(raw)
        except yaml.YAMLError:
            value = raw
        parsed[key] = value
    return parsed
