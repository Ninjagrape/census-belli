"""add battles.battle_type_source

Revision ID: c9d0e1f2a3b4
Revises: b8c9d0e1f2a3
Create Date: 2026-10-16

Records which stage set battles.battle_type ('extract' or 'classify'). Classify
used to leave that only in a missing_data_log note, which extract deletes on
every re-run, so a rule-typed battle then looked source-set and the campaign
rule never reached it. NULL means no recorded provenance (type still 'unknown',
or a row from before this column).

The same column is in config/schema.sql, because the integration fixtures build
their database from that file and never run Alembic. Every statement is guarded
so this is a no-op on a database created from the updated baseline.
"""

from __future__ import annotations

from alembic import op

revision = "c9d0e1f2a3b4"
down_revision = "b8c9d0e1f2a3"
branch_labels = None
depends_on = None

_CONSTRAINT = "battles_battle_type_source_check"


def upgrade() -> None:
    """Add the nullable provenance column and its CHECK constraint."""
    op.execute("ALTER TABLE battles ADD COLUMN IF NOT EXISTS battle_type_source TEXT")
    op.execute(
        f"""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = '{_CONSTRAINT}' AND conrelid = 'battles'::regclass
            ) THEN
                ALTER TABLE battles ADD CONSTRAINT {_CONSTRAINT}
                    CHECK (battle_type_source IN ('extract', 'classify'));
            END IF;
        END
        $$
        """
    )


def downgrade() -> None:
    """Drop the column; its constraint goes with it."""
    op.execute("ALTER TABLE battles DROP COLUMN IF EXISTS battle_type_source")
