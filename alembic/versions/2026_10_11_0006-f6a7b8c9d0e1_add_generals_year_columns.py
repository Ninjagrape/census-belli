"""add astronomical-year projections of generals.born and generals.died

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
Create Date: 2026-10-11

The resolve stage now writes generals.born and generals.died from the linked
Wikidata item. Those are DATE columns and a good part of the corpus is BC, which
datetime.date cannot hold, so any Python reader of a BC row would raise. These
two generated columns give the integer, astronomical year, the same device
battles.year_astronomical uses, and are what code should read.

Mirrored in config/schema.sql, because the integration fixtures build their
database from that file and never run Alembic. Every statement is guarded so
this is a no-op on a database created from the updated baseline.
"""

from __future__ import annotations

from alembic import op

revision = "f6a7b8c9d0e1"
down_revision = "e5f6a7b8c9d0"
branch_labels = None
depends_on = None

_PROJECTION = (
    "CASE WHEN EXTRACT(YEAR FROM {col}) < 0 "
    "THEN EXTRACT(YEAR FROM {col})::int + 1 "
    "ELSE EXTRACT(YEAR FROM {col})::int END"
)


def upgrade() -> None:
    """Add the generated year columns."""
    for column in ("born", "died"):
        op.execute(
            f"ALTER TABLE generals ADD COLUMN IF NOT EXISTS {column}_year_astronomical INT "
            f"GENERATED ALWAYS AS ({_PROJECTION.format(col=column)}) STORED"
        )


def downgrade() -> None:
    """Drop the generated year columns."""
    for column in ("born", "died"):
        op.execute(f"ALTER TABLE generals DROP COLUMN IF EXISTS {column}_year_astronomical")
