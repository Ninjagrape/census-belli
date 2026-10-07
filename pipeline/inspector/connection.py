"""Read-only database access for the inspector dashboard.

The dashboard must never be able to change pipeline state, so the guarantee is
enforced by the server rather than by the discipline of whoever writes a query:
every connection opens with ``default_transaction_read_only=on``, and Postgres
rejects any write with a ``ReadOnlySqlTransaction`` error.
"""

from __future__ import annotations

import os
from typing import Final

import structlog
from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import make_url

logger = structlog.get_logger()

__all__ = ["default_inspect_url", "read_only_engine", "redact_url"]

_READ_ONLY_OPTIONS: Final[str] = "-c default_transaction_read_only=on"
_UNPARSEABLE: Final[str] = "<unparseable url>"
# A dead host must fail fast rather than freeze every dashboard page.
_CONNECT_TIMEOUT_SECONDS: Final[int] = 5


def redact_url(url: str) -> str:
    """Mask the password in a database URL so it is safe to display.

    Args:
        url: A database URL, possibly containing credentials.

    Returns:
        The URL with any password masked, or a placeholder if it cannot be parsed.
    """
    try:
        return make_url(url).render_as_string(hide_password=True)
    except Exception:
        return _UNPARSEABLE


def default_inspect_url() -> str:
    """Resolve the dashboard's default database URL.

    ``INSPECT_DATABASE_URL`` wins so the dashboard can point at a replica or a
    reporting role, then ``DATABASE_URL`` after loading ``.env``.

    Returns:
        The URL, or an empty string when neither is configured.
    """
    from pipeline.config import load_env

    load_env()
    return os.environ.get("INSPECT_DATABASE_URL") or os.environ.get("DATABASE_URL") or ""


def read_only_engine(url: str) -> Engine:
    """Create an engine whose every session is read-only.

    Args:
        url: A SQLAlchemy database URL.

    Returns:
        An Engine that cannot write.
    """
    logger.debug("inspector_engine_created", url=redact_url(url))
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_recycle=1800,
        connect_args={"options": _READ_ONLY_OPTIONS, "connect_timeout": _CONNECT_TIMEOUT_SECONDS},
    )
