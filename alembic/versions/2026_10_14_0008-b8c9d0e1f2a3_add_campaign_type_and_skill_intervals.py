"""add campaign battle_type, infobox_order, date_circa and WAR interval columns

Revision ID: b8c9d0e1f2a3
Revises: a7b8c9d0e1f2
Create Date: 2026-10-14

'campaign' marks a war or multi-month operation in the battle corpus, which the
model excludes. infobox_order records each commander's position in the infobox
list for a side. date_circa flags hedged source dates, with date_start a best
reading. war_ci_lo, war_ci_hi and p_above_next_rank hold the pooled WAR interval
and the probability of out-ranking the next general.

The same changes are in config/schema.sql, because the integration fixtures
build their database from that file and never run Alembic. Every statement is
guarded so this is a no-op on a database created from the updated baseline.
Postgres cannot drop an enum value, so the downgrade leaves 'campaign' in place.
"""

from __future__ import annotations

from alembic import op

revision = "b8c9d0e1f2a3"
down_revision = "a7b8c9d0e1f2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add the campaign battle_type and the new nullable columns."""
    with op.get_context().autocommit_block():
        op.execute("ALTER TYPE battle_type ADD VALUE IF NOT EXISTS 'campaign' BEFORE 'unknown'")
    op.execute(
        """
        ALTER TABLE battles
            ADD COLUMN IF NOT EXISTS date_circa BOOLEAN NOT NULL DEFAULT FALSE
        """
    )
    op.execute(
        """
        ALTER TABLE battle_commanders
            ADD COLUMN IF NOT EXISTS infobox_order INT
        """
    )
    op.execute(
        """
        ALTER TABLE general_skill_estimates
            ADD COLUMN IF NOT EXISTS war_ci_lo DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS war_ci_hi DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS p_above_next_rank DOUBLE PRECISION
        """
    )


def downgrade() -> None:
    """Drop the new columns. The 'campaign' enum value stays, as Postgres cannot drop it."""
    op.execute("ALTER TABLE general_skill_estimates DROP COLUMN IF EXISTS p_above_next_rank")
    op.execute("ALTER TABLE general_skill_estimates DROP COLUMN IF EXISTS war_ci_hi")
    op.execute("ALTER TABLE general_skill_estimates DROP COLUMN IF EXISTS war_ci_lo")
    op.execute("ALTER TABLE battle_commanders DROP COLUMN IF EXISTS infobox_order")
    op.execute("ALTER TABLE battles DROP COLUMN IF EXISTS date_circa")
    # Postgres cannot drop an enum value, so battle_type keeps 'campaign' on downgrade.
