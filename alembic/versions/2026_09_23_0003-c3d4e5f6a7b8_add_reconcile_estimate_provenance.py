"""add reconcile fitted parameters and estimate provenance

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-09-23

The reconcile stage fits a hierarchical source-disagreement model and needs
somewhere to put what it learns. The baseline schema had two scalar columns
(sources.troop_bias_mu/_sd) and nothing else: no per-source precision, no
casualty bias, and no provenance at all on battle_sides, which carries only
created_at. Without provenance an estimate from an earlier fit is
indistinguishable from a fresh one -- and re-running extract rewrites the
troop_reports underneath without invalidating anything downstream.

Also corrects troop_reports.scope's default from 'engaged' to 'unknown'. The
Python writer has always passed an explicit value so the default was dead
code, but the two disagreed, and the permissive one asserts troops-on-the-
field, which is precisely what an unreadable scope is not evidence of.

The same changes are made in config/schema.sql rather than here alone. The
integration fixtures build their database with apply_schema(), which executes
that file directly and never runs Alembic, so a migration-only change would be
invisible to every test. Both paths therefore carry it, and every statement
below is guarded so it is a no-op on a database created from the updated
baseline and a real change on one created from the original.
"""

from __future__ import annotations

from alembic import op

revision = "c3d4e5f6a7b8"
down_revision = "b2c3d4e5f6a7"
branch_labels = None
depends_on = None

# (constraint name, table, column). Kept as a module literal so the DDL below
# interpolates nothing that came from outside this file.
_RUN_FOREIGN_KEYS: tuple[tuple[str, str, str], ...] = (
    ("sources_bias_run_fk", "sources", "bias_run_id"),
    ("battle_sides_est_troops_run_fk", "battle_sides", "est_troops_run_id"),
    ("battle_sides_est_casualties_run_fk", "battle_sides", "est_casualties_run_id"),
)

_SOURCE_COLUMNS: tuple[str, ...] = (
    "troop_sigma_mu",
    "troop_sigma_sd",
    "casualty_bias_mu",
    "casualty_bias_sd",
    "casualty_sigma_mu",
    "casualty_sigma_sd",
    "bias_run_id",
    "bias_n_troop_reports",
    "bias_n_casualty_reports",
    "bias_updated_at",
)

_SIDE_COLUMNS: tuple[str, ...] = (
    "est_troops_run_id",
    "est_troops_n_reports",
    "est_troops_n_sources",
    "est_troops_method",
    "est_troops_updated_at",
    "est_casualties_run_id",
    "est_casualties_n_reports",
    "est_casualties_n_sources",
    "est_casualties_method",
    "est_casualties_updated_at",
)


def upgrade() -> None:
    """Add the fitted-parameter and provenance columns, and their keys."""
    op.execute(
        """
        ALTER TABLE sources
            ADD COLUMN IF NOT EXISTS troop_sigma_mu          DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS troop_sigma_sd          DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS casualty_bias_mu        DOUBLE PRECISION DEFAULT 0.0,
            ADD COLUMN IF NOT EXISTS casualty_bias_sd        DOUBLE PRECISION DEFAULT 1.0,
            ADD COLUMN IF NOT EXISTS casualty_sigma_mu       DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS casualty_sigma_sd       DOUBLE PRECISION,
            ADD COLUMN IF NOT EXISTS bias_run_id             INT,
            ADD COLUMN IF NOT EXISTS bias_n_troop_reports    INT DEFAULT 0,
            ADD COLUMN IF NOT EXISTS bias_n_casualty_reports INT DEFAULT 0,
            ADD COLUMN IF NOT EXISTS bias_updated_at         TIMESTAMPTZ
        """
    )
    op.execute(
        """
        ALTER TABLE battle_sides
            ADD COLUMN IF NOT EXISTS est_troops_run_id         INT,
            ADD COLUMN IF NOT EXISTS est_troops_n_reports      INT,
            ADD COLUMN IF NOT EXISTS est_troops_n_sources      INT,
            ADD COLUMN IF NOT EXISTS est_troops_method         TEXT,
            ADD COLUMN IF NOT EXISTS est_troops_updated_at     TIMESTAMPTZ,
            ADD COLUMN IF NOT EXISTS est_casualties_run_id     INT,
            ADD COLUMN IF NOT EXISTS est_casualties_n_reports  INT,
            ADD COLUMN IF NOT EXISTS est_casualties_n_sources  INT,
            ADD COLUMN IF NOT EXISTS est_casualties_method     TEXT,
            ADD COLUMN IF NOT EXISTS est_casualties_updated_at TIMESTAMPTZ
        """
    )

    op.execute("ALTER TABLE troop_reports ALTER COLUMN scope SET DEFAULT 'unknown'")

    # Postgres 15 has no ADD CONSTRAINT IF NOT EXISTS for table constraints, so
    # each key is guarded by a catalogue lookup. Without this the migration
    # fails on a database already built from the updated config/schema.sql,
    # which is exactly the path CI's upgrade/downgrade/upgrade job takes.
    for name, table, column in _RUN_FOREIGN_KEYS:
        op.execute(
            f"""
            DO $$ BEGIN
                IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = '{name}') THEN
                    ALTER TABLE {table} ADD CONSTRAINT {name}
                        FOREIGN KEY ({column}) REFERENCES model_runs(run_id);
                END IF;
            END $$;
            """
        )

    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_sides_est_troops_run "
        "ON battle_sides (est_troops_run_id)"
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_sources_bias_run ON sources (bias_run_id)")
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_model_runs_type ON model_runs (model_type, run_id DESC)"
    )


def downgrade() -> None:
    """Drop the columns, keys and indexes, and restore the old scope default."""
    op.execute("DROP INDEX IF EXISTS idx_model_runs_type")
    op.execute("DROP INDEX IF EXISTS idx_sources_bias_run")
    op.execute("DROP INDEX IF EXISTS idx_sides_est_troops_run")

    for name, table, _column in _RUN_FOREIGN_KEYS:
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {name}")

    op.execute("ALTER TABLE troop_reports ALTER COLUMN scope SET DEFAULT 'engaged'")

    side_drops = ", ".join(f"DROP COLUMN IF EXISTS {c}" for c in _SIDE_COLUMNS)
    op.execute(f"ALTER TABLE battle_sides {side_drops}")

    source_drops = ", ".join(f"DROP COLUMN IF EXISTS {c}" for c in _SOURCE_COLUMNS)
    op.execute(f"ALTER TABLE sources {source_drops}")
