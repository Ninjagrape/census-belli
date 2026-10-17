"""allow battles.battle_type_source = 'manual'

Revision ID: d0e1f2a3b4c5
Revises: c9d0e1f2a3b4
Create Date: 2026-10-17

A battle_type override from config/overrides.yaml is applied by classify and
recorded as 'manual', so a later run neither rewrites it as a rule-typed
battle nor mistakes it for an extract-set type.

The same change is in config/schema.sql, because the integration fixtures build
their database from that file and never run Alembic. The statements are
guarded so they are a no-op on a database created from the updated baseline.
"""

from __future__ import annotations

from alembic import op

revision = "d0e1f2a3b4c5"
down_revision = "c9d0e1f2a3b4"
branch_labels = None
depends_on = None

_CONSTRAINT = "battles_battle_type_source_check"


def upgrade() -> None:
    """Widen the CHECK constraint to allow 'manual'."""
    op.execute(f"ALTER TABLE battles DROP CONSTRAINT IF EXISTS {_CONSTRAINT}")
    op.execute(
        f"""
        ALTER TABLE battles ADD CONSTRAINT {_CONSTRAINT}
            CHECK (battle_type_source IN ('extract', 'classify', 'manual'))
        """
    )


def downgrade() -> None:
    """Restore the two-value CHECK constraint."""
    # NULL is the documented "no recorded provenance" state and is allowed by
    # the old check; 'classify' would wrongly let classify rewrite a type that
    # a person set.
    op.execute("UPDATE battles SET battle_type_source = NULL WHERE battle_type_source = 'manual'")
    op.execute(f"ALTER TABLE battles DROP CONSTRAINT IF EXISTS {_CONSTRAINT}")
    op.execute(
        f"""
        ALTER TABLE battles ADD CONSTRAINT {_CONSTRAINT}
            CHECK (battle_type_source IN ('extract', 'classify'))
        """
    )
