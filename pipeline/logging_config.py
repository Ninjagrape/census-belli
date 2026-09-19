"""
Structured logging configuration.

Every module in this project calls ``structlog.get_logger()``, but until this
module existed nothing ever called ``structlog.configure()``. That is not an
error -- structlog falls back to a default -- which is exactly the problem:
the pipeline logged through an unconfigured library, so the output format,
the level and whether anything was written at all were accidents of the
default rather than decisions.

Named ``logging_config`` rather than ``logging`` on purpose. A
``pipeline/logging.py`` is importable as ``pipeline.logging``, which is
harmless in itself, but it shadows the standard library's name for anyone
reading the imports and invites confusion the day something under
``pipeline/`` needs both. The awkward name costs nothing.

Two renderers, chosen by where the output is going:

- a **console** renderer for a terminal, with key-value pairs aligned and
  coloured, which is what a person watching a crawl wants;
- a **JSON** renderer otherwise, which is what a log collector wants and what
  makes a long pipeline run greppable afterwards.

The stage runners emit one event per battle at debug and one per batch at
info, so the level is the difference between a readable run and forty
thousand lines. It defaults to INFO and is overridden with ``LOG_LEVEL``.

``configure()`` is idempotent and safe to call from anywhere. It is called
from :func:`pipeline.orchestrator.main` and from ``pipeline.db``'s CLI, so
every entry point configures logging before it logs. Importing a library
module does *not* configure it: deciding the logging setup of a process you
do not own is the library mistake this module exists to avoid.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Final

import structlog

__all__ = ["DEFAULT_LEVEL", "configure", "is_configured"]

DEFAULT_LEVEL: Final = "INFO"

_LEVELS: Final[dict[str, int]] = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}

_configured = False


def is_configured() -> bool:
    """Report whether :func:`configure` has run in this process.

    Returns:
        True once logging has been configured.
    """
    return _configured


def _resolve_level(level: str | int | None) -> int:
    """Turn a level name, number or None into a numeric logging level.

    Args:
        level: An explicit level, or None to read ``LOG_LEVEL`` and fall back
            to :data:`DEFAULT_LEVEL`.

    Returns:
        A numeric logging level. An unrecognised name falls back to the
        default rather than raising: a typo in an environment variable should
        not stop a pipeline run, it should produce logs at the usual level.
        The caller warns about it once logging is up.
    """
    if isinstance(level, int):
        return level

    name = (level or os.environ.get("LOG_LEVEL") or DEFAULT_LEVEL).strip().upper()
    return _LEVELS.get(name, _LEVELS[DEFAULT_LEVEL])


def configure(
    *,
    level: str | int | None = None,
    json_logs: bool | None = None,
    force: bool = False,
) -> None:
    """Configure structlog and the standard library's logging for this process.

    Args:
        level: Minimum level to emit. Defaults to ``LOG_LEVEL``, then INFO.
        json_logs: Render JSON rather than console output. Defaults to JSON
            whenever stderr is not a terminal, so a piped or redirected run is
            machine-readable and an interactive one is readable.
        force: Reconfigure even if this has already run. Without it a second
            call is a no-op, so a stage cannot quietly restyle the logs of the
            run that imported it.
    """
    global _configured
    if _configured and not force:
        return

    requested = level if isinstance(level, str) else os.environ.get("LOG_LEVEL") or ""
    requested = requested.strip().upper()
    resolved_level = _resolve_level(level)

    if json_logs is None:
        json_logs = not sys.stderr.isatty()

    shared: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        # Render exceptions the same way in both formats, so a traceback in a
        # collected log is not a different shape from one in a terminal.
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    structlog.configure(
        processors=[*shared, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(resolved_level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    # Third-party libraries log through the standard library, and httpx in
    # particular logs every request at INFO. A crawl of thousands of pages
    # would otherwise bury one line of ours under twenty of theirs.
    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=resolved_level)
    for noisy in ("httpx", "httpcore", "urllib3", "sqlalchemy.engine"):
        logging.getLogger(noisy).setLevel(max(resolved_level, logging.WARNING))

    _configured = True

    if requested and requested not in _LEVELS:
        structlog.get_logger().warning(
            "log_level_unrecognised",
            requested=requested,
            using=logging.getLevelName(resolved_level),
            valid=sorted(_LEVELS),
        )
