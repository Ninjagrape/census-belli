"""baseline schema from config/schema.sql

Revision ID: a1b2c3d4e5f6
Revises:
Create Date: 2026-09-18

The baseline is not transcribed into SQLAlchemy operations. It executes
``config/schema.sql`` directly, because that file is the authored source of
truth for the schema (CLAUDE.md: "The base schema is config/schema.sql; all
subsequent changes are Alembic migrations"). Transcribing it would create a
second definition free to drift from the first.

Every migration after this one is written by hand as ops.

Applying this to a database that already has the schema -- from
``make db-schema`` or ``pipeline.db.apply_schema`` -- is a no-op, so an
existing development database can be brought under Alembic control by
running the upgrade, or with ``alembic stamp head``.
"""

from __future__ import annotations

from pathlib import Path

import sqlalchemy as sa

from alembic import context, op

revision: str = "a1b2c3d4e5f6"
down_revision: str | None = None
branch_labels: str | None = None
depends_on: str | None = None

# Resolved relative to this file rather than the working directory, so the
# migration behaves the same however alembic is invoked.
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "config" / "schema.sql"

# Present in every complete application of the schema.
SENTINEL_TABLES = ("battles", "battle_commanders", "generals", "llm_calls")


def _schema_present(conn: sa.Connection) -> bool:
    """Report whether the baseline tables already exist.

    Args:
        conn: An open connection.

    Returns:
        True if every sentinel table is present.
    """
    found = conn.execute(
        sa.text(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name = ANY(:names)"
        ),
        {"names": list(SENTINEL_TABLES)},
    ).scalar()
    return int(found or 0) == len(SENTINEL_TABLES)


def _schema_sql() -> str:
    """Read the base schema.

    Returns:
        The contents of config/schema.sql.

    Raises:
        RuntimeError: If the file is missing or empty.
    """
    if not SCHEMA_PATH.exists():
        raise RuntimeError(f"Base schema not found at {SCHEMA_PATH}")

    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    if not sql.strip():
        raise RuntimeError(f"Base schema is empty: {SCHEMA_PATH}")
    return sql


def upgrade() -> None:
    """Create the base schema."""
    if context.is_offline_mode():
        # `alembic upgrade head --sql` has no connection to interrogate, so
        # the presence check cannot run. Emit the schema unconditionally:
        # the point of offline mode is to produce the DDL for review, and a
        # silently empty script would be worse than a redundant one.
        op.execute(_schema_sql())
        return

    conn = op.get_bind()

    if _schema_present(conn):
        # Already applied out of band by make db-schema. Recording the
        # revision is all that remains, which Alembic does for us.
        return

    # The file is a script of many statements; exec_driver_sql passes it
    # through rather than treating it as one parameterised statement.
    conn.exec_driver_sql(_schema_sql())


def downgrade() -> None:
    """Drop everything this migration created.

    The baseline creates enums, tables and views across the whole public
    schema, so unwinding it statement by statement would be long and easy to
    get wrong. Dropping the schema is both simpler and honest about what it
    does: it destroys every table and all crawled data.
    """
    op.execute("DROP SCHEMA IF EXISTS public CASCADE")
    op.execute("CREATE SCHEMA public")
