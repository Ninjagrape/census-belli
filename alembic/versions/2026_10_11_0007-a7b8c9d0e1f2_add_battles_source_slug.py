"""add battles.source_slug so same-name battles stay distinct

Revision ID: a7b8c9d0e1f2
Revises: f6a7b8c9d0e1
Create Date: 2026-10-11

Extract and resolve looked battles up by name with LIMIT 1, so two articles
that share a display name collapsed onto one row. source_slug holds the crawl
stage's article filename stem, which carries a digest of the URL and is
therefore unique per article. It is NULL on rows written before this column
existed; extract claims such a row by name the first time it sees the battle
again.

The same column is in config/schema.sql, because the integration fixtures
build their database from that file and never run Alembic. Every statement is
guarded so this is a no-op on a database created from the updated baseline.
"""

from __future__ import annotations

from alembic import op

revision = "a7b8c9d0e1f2"
down_revision = "f6a7b8c9d0e1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add battles.source_slug with a unique constraint."""
    op.execute("ALTER TABLE battles ADD COLUMN IF NOT EXISTS source_slug TEXT")
    op.execute(
        """
        DO $$ BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint WHERE conname = 'battles_source_slug_key'
            ) THEN
                ALTER TABLE battles ADD CONSTRAINT battles_source_slug_key UNIQUE (source_slug);
            END IF;
        END $$;
        """
    )


def downgrade() -> None:
    """Drop battles.source_slug."""
    op.execute("ALTER TABLE battles DROP CONSTRAINT IF EXISTS battles_source_slug_key")
    op.execute("ALTER TABLE battles DROP COLUMN IF EXISTS source_slug")
