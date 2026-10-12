"""correct the win_prob and imputation_method column comments

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-10-11

``battle_war_details.win_prob_model`` and ``win_prob_baseline`` hold the side's
expected score, ``P(win) + 0.5 P(indecisive)``, not P(win) (gap review A15).
``missing_data_log.imputation_method`` listed ``mice_pmm``, which the joint
Bayesian imputer replaced. The text is also in config/schema.sql, which the
integration fixtures build from. COMMENT ON is idempotent, so this is a no-op
on a database created from the updated baseline.
"""

from __future__ import annotations

from alembic import op

revision = "e5f6a7b8c9d0"
down_revision = "d4e5f6a7b8c9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Set the corrected column comments."""
    op.execute(
        "COMMENT ON COLUMN battle_war_details.win_prob_model IS "
        "'expected score P(win)+0.5*P(indecisive) under the fitted model, not P(win)'"
    )
    op.execute(
        "COMMENT ON COLUMN battle_war_details.win_prob_baseline IS "
        "'expected score with a replacement-level general in this slot'"
    )
    op.execute(
        "COMMENT ON COLUMN missing_data_log.imputation_method IS "
        "'e.g. bayes_joint_v1_neutral_log, bayes_joint_v1_published_log, era_polity_prior, manual'"
    )


def downgrade() -> None:
    """Restore the earlier comments."""
    op.execute("COMMENT ON COLUMN battle_war_details.win_prob_model IS NULL")
    op.execute("COMMENT ON COLUMN battle_war_details.win_prob_baseline IS NULL")
    op.execute("COMMENT ON COLUMN missing_data_log.imputation_method IS NULL")
