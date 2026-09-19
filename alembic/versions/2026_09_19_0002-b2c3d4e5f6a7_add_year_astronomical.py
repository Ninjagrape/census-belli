"""add battles.year_astronomical

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-09-19

Adds a readable integer year to battles, because date_start is unreadable
from Python for any BC battle (datetime.date has MINYEAR == 1) and the corpus
is heavily ancient.

The column is also added to config/schema.sql rather than here alone. The
integration fixtures build their database with apply_schema(), which executes
that file directly and never runs Alembic, so a migration-only change would be
invisible to every test. Both paths therefore carry it, and the ALTER below is
guarded with IF NOT EXISTS so it is a no-op on a database created from the
updated baseline and a real change on one created from the original.
"""

from __future__ import annotations

from alembic import op

revision = "b2c3d4e5f6a7"
down_revision = "a1b2c3d4e5f6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add the generated astronomical-year column and its index."""
    # Astronomical numbering, not Postgres's: EXTRACT reports 31 BC as -31,
    # astronomical numbering as -30, because it has a year zero and Postgres
    # does not. Only the astronomical form subtracts correctly across the
    # BC/AD boundary, which is what a time trend or era covariate needs.
    op.execute(
        """
        ALTER TABLE battles ADD COLUMN IF NOT EXISTS year_astronomical INT
        GENERATED ALWAYS AS (
            CASE WHEN EXTRACT(YEAR FROM date_start) < 0
                 THEN EXTRACT(YEAR FROM date_start)::int + 1
                 ELSE EXTRACT(YEAR FROM date_start)::int END
        ) STORED
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_battles_year ON battles (year_astronomical)")


def downgrade() -> None:
    """Drop the column and its index."""
    op.execute("DROP INDEX IF EXISTS idx_battles_year")
    op.execute("ALTER TABLE battles DROP COLUMN IF EXISTS year_astronomical")
