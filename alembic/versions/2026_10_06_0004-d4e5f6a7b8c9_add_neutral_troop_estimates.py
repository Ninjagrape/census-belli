"""add regime-neutral troop estimates and is_defender to battle_sides

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-10-06

The reconcile stage now fits troops a second time with the unlabelled claim
regime read as an ordinary modern count, so a force ratio is not tilted by the
unlabelled-regime prior (handover 20.10). Its per-side output lives in five new
columns on battle_sides. is_defender is added unpopulated: NULL means unknown,
and the classify stage fills it later.

The same columns are in config/schema.sql, because the integration fixtures
build their database from that file and never run Alembic. Every statement is
guarded so this is a no-op on a database created from the updated baseline.
The SQL carries no percent sign, which psycopg would read as a placeholder.
"""

from __future__ import annotations

from alembic import op

revision = "d4e5f6a7b8c9"
down_revision = "c3d4e5f6a7b8"
branch_labels = None
depends_on = None

_COLUMNS: tuple[str, ...] = (
    "est_troops_neutral",
    "est_troops_neutral_lo",
    "est_troops_neutral_hi",
    "est_troops_neutral_log_mean",
    "est_troops_neutral_log_sd",
    "est_troops_neutral_run_id",
    "is_defender",
)


def upgrade() -> None:
    """Add the neutral estimate columns and is_defender."""
    op.execute(
        """
        ALTER TABLE battle_sides
            ADD COLUMN IF NOT EXISTS est_troops_neutral          DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS est_troops_neutral_lo       DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS est_troops_neutral_hi       DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS est_troops_neutral_log_mean DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS est_troops_neutral_log_sd   DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS est_troops_neutral_run_id   INT,
            ADD COLUMN IF NOT EXISTS is_defender                 BOOLEAN
        """
    )
    op.execute(
        """
        DO $$ BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = 'battle_sides_est_troops_neutral_run_fk'
            ) THEN
                ALTER TABLE battle_sides ADD CONSTRAINT battle_sides_est_troops_neutral_run_fk
                    FOREIGN KEY (est_troops_neutral_run_id) REFERENCES model_runs(run_id);
            END IF;
        END $$;
        """
    )


def downgrade() -> None:
    """Drop the neutral estimate columns and is_defender."""
    op.execute(
        "ALTER TABLE battle_sides DROP CONSTRAINT IF EXISTS battle_sides_est_troops_neutral_run_fk"
    )
    drops = ", ".join(f"DROP COLUMN IF EXISTS {c}" for c in _COLUMNS)
    op.execute(f"ALTER TABLE battle_sides {drops}")
