"""
Seed configuration for the crawl stage.

``config/sources_seed.yaml`` names the battle-list pages and the SPARQL
queries the crawl starts from. Changing the corpus is meant to be a config
edit, so this module validates that file rather than letting a typo surface
hours into a run as an empty result set.

A missing or malformed seed file is a bug in the run, not a property of the
data, so every problem here raises.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog
import yaml

__all__ = [
    "DEFAULT_SEED_PATH",
    "SeedConfigError",
    "Seeds",
    "load_seeds",
]

logger = structlog.get_logger()

DEFAULT_SEED_PATH = Path("config/sources_seed.yaml")


class SeedConfigError(ValueError):
    """Raised when the seed config is missing, malformed or empty."""


@dataclass(frozen=True)
class Seeds:
    """The crawl's entry points.

    Attributes:
        battle_lists: Wikipedia list-page URLs to discover battles from.
        wikidata_queries: Named SPARQL queries, keyed by query name.
        battle_articles: Battle article URLs named directly, bypassing
            discovery. A curated corpus -- a pilot run, or a fixed evaluation
            set whose membership must not drift when someone edits a Wikipedia
            list page -- names its battles here, and may name no list pages at
            all.
    """

    battle_lists: tuple[str, ...]
    wikidata_queries: dict[str, str]
    battle_articles: tuple[str, ...] = ()


def load_seeds(path: Path | str = DEFAULT_SEED_PATH) -> Seeds:
    """Load and validate the seed configuration.

    Args:
        path: Path to the seed YAML.

    Returns:
        The validated seeds.

    Raises:
        SeedConfigError: If the file is missing, is not a mapping, names no
            battle lists, or contains a malformed query.
    """
    seed_path = Path(path)
    if not seed_path.exists():
        raise SeedConfigError(f"Seed config not found: {seed_path}")

    try:
        loaded: Any = yaml.safe_load(seed_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SeedConfigError(f"{seed_path} is not valid YAML: {exc}") from exc

    if not isinstance(loaded, dict):
        raise SeedConfigError(
            f"{seed_path} must contain a mapping, got {type(loaded).__name__}"
        )

    battle_lists = _url_list(
        loaded.get("wikipedia_battle_lists"), "wikipedia_battle_lists", seed_path
    )
    battle_articles = _url_list(
        loaded.get("wikipedia_battle_articles"), "wikipedia_battle_articles", seed_path
    )
    if not battle_lists and not battle_articles:
        raise SeedConfigError(
            f"{seed_path} must define a non-empty wikipedia_battle_lists or "
            "wikipedia_battle_articles list; the crawl has no entry point without one"
        )

    raw_queries = loaded.get("wikidata_queries") or {}
    if not isinstance(raw_queries, dict):
        raise SeedConfigError(
            f"{seed_path}: wikidata_queries must be a mapping of name to SPARQL, "
            f"got {type(raw_queries).__name__}"
        )

    queries: dict[str, str] = {}
    for name, query in raw_queries.items():
        if not isinstance(query, str) or not query.strip():
            raise SeedConfigError(f"{seed_path}: SPARQL query {name!r} is empty or not a string")
        queries[str(name)] = query

    logger.info(
        "seeds_loaded",
        path=str(seed_path),
        lists=len(battle_lists),
        articles=len(battle_articles),
        queries=len(queries),
    )
    return Seeds(
        battle_lists=tuple(battle_lists),
        wikidata_queries=queries,
        battle_articles=tuple(battle_articles),
    )


def _url_list(raw: Any, key: str, seed_path: Path) -> list[str]:
    """Validate one of the seed file's URL lists.

    Args:
        raw: The raw value under ``key``, which may be absent.
        key: The key's name, for error messages.
        seed_path: The file being read, for error messages.

    Returns:
        The URLs, deduplicated, in file order. Empty if the key is absent.

    Raises:
        SeedConfigError: If the value is not a list, or holds anything that is
            not an absolute URL. A relative URL here would fail hours into a
            run rather than at load time, which is the whole point of
            validating the file.
    """
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise SeedConfigError(f"{seed_path}: {key} must be a list, got {type(raw).__name__}")

    urls: list[str] = []
    for entry in raw:
        if not isinstance(entry, str) or not entry.startswith(("http://", "https://")):
            raise SeedConfigError(f"{seed_path}: {key} entry {entry!r} is not an absolute URL")
        if entry not in urls:
            urls.append(entry)
    return urls
