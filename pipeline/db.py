"""
Database access for the General WAR pipeline.

Every stage reaches Postgres through this module, so that connection
handling, schema application and URL resolution live in one place rather
than being re-derived per stage.

Following the project's error-handling split: a *configuration* problem (no
DATABASE_URL, an unreadable schema file) raises immediately, because it is a
bug in the run rather than a property of the data. Connection and query
failures propagate to the caller, which knows whether the record should be
flagged for review or the stage aborted.

Usage:
    from pipeline.db import apply_schema, get_connection, get_engine

    with get_connection() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM battles")).scalar()

CLI:
    python -m pipeline.db --apply-schema
    python -m pipeline.db --check
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import structlog
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import Connection

logger = structlog.get_logger()

__all__ = [
    "DEFAULT_SCHEMA_PATH",
    "DatabaseConfigError",
    "apply_schema",
    "database_url",
    "get_connection",
    "get_engine",
    "schema_is_applied",
]

DEFAULT_SCHEMA_PATH = Path("config/schema.sql")

# Tables that must exist for the schema to count as applied. Checking several
# rather than one avoids treating a half-applied schema as complete.
_SENTINEL_TABLES = ("battles", "battle_commanders", "generals", "llm_calls")


class DatabaseConfigError(RuntimeError):
    """Raised when the database is misconfigured, as opposed to unreachable."""


def database_url(url: str | None = None) -> str:
    """Resolve the database URL.

    Args:
        url: An explicit URL, which wins over the environment. Used by tests.

    Returns:
        A SQLAlchemy URL string.

    Raises:
        DatabaseConfigError: If no URL is available anywhere.
    """
    resolved = url or os.environ.get("DATABASE_URL")
    if not resolved:
        raise DatabaseConfigError(
            "DATABASE_URL is not set. Copy .env.example to .env and fill it in, "
            "or pass an explicit url. Expected form: "
            "postgresql+psycopg://user:password@localhost:5432/general_war"
        )
    return resolved


def _redact(url: str) -> str:
    """Strip the password from a URL so it is safe to log.

    Args:
        url: A database URL, possibly containing credentials.

    Returns:
        The URL with any password masked.
    """
    try:
        from sqlalchemy.engine import make_url

        return make_url(url).render_as_string(hide_password=True)
    except Exception:
        # Never let logging a URL be the thing that breaks a run.
        return "<unparseable url>"


def get_engine(url: str | None = None, **kwargs: Any) -> Engine:
    """Create a SQLAlchemy Core engine.

    Args:
        url: Explicit database URL; falls back to DATABASE_URL.
        **kwargs: Passed through to create_engine.

    Returns:
        A configured Engine.

    Raises:
        DatabaseConfigError: If no URL is available.
    """
    resolved = database_url(url)

    options: dict[str, Any] = {
        # The pipeline is long-running and largely sequential; pre-ping guards
        # against Postgres closing an idle connection mid-stage.
        "pool_pre_ping": True,
        "pool_recycle": 1800,
    }
    options.update(kwargs)

    logger.debug("engine_created", url=_redact(resolved))
    return create_engine(resolved, **options)


@contextmanager
def get_connection(
    url: str | None = None,
    engine: Engine | None = None,
) -> Iterator[Connection]:
    """Open a connection, committing on clean exit and rolling back on error.

    Args:
        url: Explicit database URL; falls back to DATABASE_URL.
        engine: An existing engine to use instead of creating one. When given,
            the engine is left open for the caller to dispose.

    Yields:
        An open Connection.
    """
    owned = engine is None
    eng = engine or get_engine(url)
    conn = eng.connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
        if owned:
            eng.dispose()


def schema_is_applied(conn: Connection) -> bool:
    """Report whether the core tables already exist.

    Args:
        conn: An open connection.

    Returns:
        True if every sentinel table is present.
    """
    found = conn.execute(
        text(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name = ANY(:names)"
        ),
        {"names": list(_SENTINEL_TABLES)},
    ).scalar()
    return int(found or 0) == len(_SENTINEL_TABLES)


def apply_schema(
    engine: Engine,
    schema_path: Path | str = DEFAULT_SCHEMA_PATH,
    drop_existing: bool = False,
) -> bool:
    """Execute config/schema.sql against the database.

    The schema uses bare CREATE TYPE and CREATE TABLE, so re-applying it over
    an existing database would fail partway and leave it half-migrated. This
    checks first and no-ops instead, which keeps the call safe to repeat --
    the same idempotence the pipeline stages are expected to have.

    Args:
        engine: The engine to apply against.
        schema_path: Path to the SQL file.
        drop_existing: Drop and recreate the public schema first. Destroys all
            data; intended for `make db-reset` and test fixtures only.

    Returns:
        True if the schema was applied, False if it was already present.

    Raises:
        DatabaseConfigError: If the schema file is missing or empty.
    """
    path = Path(schema_path)
    if not path.exists():
        raise DatabaseConfigError(f"Schema file not found: {path}")

    sql = path.read_text(encoding="utf-8")
    if not sql.strip():
        raise DatabaseConfigError(f"Schema file is empty: {path}")

    with engine.connect() as conn:
        if drop_existing:
            logger.warning("dropping_public_schema", url=_redact(str(engine.url)))
            conn.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
            conn.execute(text("CREATE SCHEMA public"))
            conn.commit()
        elif schema_is_applied(conn):
            logger.info("schema_already_applied", path=str(path))
            return False

        # The file goes to psycopg as one script via a raw cursor with no
        # parameter argument at all. text() would choke on the ':' in casts
        # and range types, and exec_driver_sql still passes an empty parameter
        # set, which makes psycopg parse '%' as a placeholder -- schema.sql
        # has five, all in '95% CI' comments.
        #
        # The commit goes to the driver connection rather than to conn: the
        # raw cursor's work is invisible to SQLAlchemy's transaction tracking,
        # so after the drop_existing branch has already committed, conn.commit()
        # finds no active transaction and silently does nothing, and the DDL is
        # then rolled back on close.
        driver_conn = conn.connection.driver_connection
        if driver_conn is None:  # pragma: no cover - a live connection always has one
            raise DatabaseConfigError(
                "Connection exposes no driver connection; cannot apply the schema."
            )
        raw_cursor = driver_conn.cursor()
        try:
            raw_cursor.execute(sql)
        finally:
            raw_cursor.close()
        driver_conn.commit()

    logger.info("schema_applied", path=str(path), bytes=len(sql))
    return True


def main() -> None:
    """CLI entry point for schema management."""
    parser = argparse.ArgumentParser(description="General WAR database utilities")
    parser.add_argument("--apply-schema", action="store_true", help="Apply config/schema.sql")
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop the public schema first (DESTRUCTIVE, implies --apply-schema)",
    )
    parser.add_argument(
        "--check", action="store_true", help="Report connectivity and schema state"
    )
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA_PATH), help="Path to schema SQL")
    args = parser.parse_args()

    if not (args.apply_schema or args.check or args.drop_existing):
        parser.print_help()
        return

    try:
        engine = get_engine()
    except DatabaseConfigError as exc:
        logger.error("database_config_error", error=str(exc))
        sys.exit(2)

    try:
        if args.apply_schema or args.drop_existing:
            created = apply_schema(engine, args.schema, drop_existing=args.drop_existing)
            logger.info("apply_schema_result", created=created)

        if args.check:
            with engine.connect() as conn:
                version = conn.execute(text("SELECT version()")).scalar()
                applied = schema_is_applied(conn)
            logger.info("database_check", server=version, schema_applied=applied)
    except Exception as exc:
        logger.error("database_command_failed", error=str(exc))
        sys.exit(1)
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
