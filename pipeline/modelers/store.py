"""Database writes for the model stage.

Every statement inserts rows for a run id this invocation created; nothing
here updates or deletes another run's rows, so earlier rankings stay intact.
The one UPDATE stamps ``imputation_set`` on the run row just inserted, because
:func:`pipeline.reconcilers.store.start_model_run` has no such argument.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import structlog
from sqlalchemy import text

from pipeline.reconcilers import store as run_store

__all__ = [
    "SkillEstimateRow",
    "WarDetailRow",
    "complete_run",
    "start_run",
    "write_skill_estimates",
    "write_war_details",
]

logger = structlog.get_logger()

_SET_IMPUTATION = text("UPDATE model_runs SET imputation_set = :n WHERE run_id = :run_id")

_INSERT_ESTIMATE = text(
    """
    INSERT INTO general_skill_estimates (
        run_id, general_id, skill_mean, skill_median, skill_sd,
        skill_ci_lo, skill_ci_hi, skill_ci90_lo, skill_ci90_hi,
        total_war, war_per_battle, war_sd, war_ci_lo, war_ci_hi,
        p_above_next_rank, n_battles, rank
    ) VALUES (
        :run_id, :general_id, :skill_mean, :skill_median, :skill_sd,
        :skill_ci_lo, :skill_ci_hi, :skill_ci90_lo, :skill_ci90_hi,
        :total_war, :war_per_battle, :war_sd, :war_ci_lo, :war_ci_hi,
        :p_above_next_rank, :n_battles, :rank
    )
    """
)

_INSERT_DETAIL = text(
    """
    INSERT INTO battle_war_details (
        run_id, battle_id, general_id, war_contribution,
        win_prob_model, win_prob_baseline, actual_outcome, attribution_weight
    ) VALUES (
        :run_id, :battle_id, :general_id, :war_contribution,
        :win_prob_model, :win_prob_baseline, :actual_outcome, :attribution_weight
    )
    """
)


@dataclass(frozen=True)
class SkillEstimateRow:
    """One ``general_skill_estimates`` row (without the run id)."""

    general_id: int
    skill_mean: float
    skill_median: float
    skill_sd: float
    skill_ci_lo: float
    skill_ci_hi: float
    skill_ci90_lo: float
    skill_ci90_hi: float
    total_war: float
    war_per_battle: float
    war_sd: float
    war_ci_lo: float | None
    war_ci_hi: float | None
    p_above_next_rank: float | None
    n_battles: int
    rank: int


@dataclass(frozen=True)
class WarDetailRow:
    """One ``battle_war_details`` row (without the run id).

    ``win_prob_model`` and ``win_prob_baseline`` hold the side's EXPECTED
    SCORE, ``P(win) + 0.5 P(indecisive)``, despite the column names.
    """

    battle_id: int
    general_id: int
    war_contribution: float
    win_prob_model: float
    win_prob_baseline: float
    actual_outcome: float
    attribution_weight: float


def start_run(
    conn: Any,
    *,
    model_type: str,
    run_name: str,
    config: dict[str, Any],
    imputation_set: int | None = None,
) -> int:
    """Insert a ``model_runs`` row and return its id.

    Args:
        conn: An open database connection.
        model_type: ``bradley_terry_round``, ``bradley_terry_joint_latent`` or
            ``bradley_terry_hierarchical``.
        run_name: Human-readable label.
        config: JSON-serialisable run configuration.
        imputation_set: The round index for a per-round fit, else None.

    Returns:
        The new run id.
    """
    run_id = run_store.start_model_run(
        conn, model_type=model_type, run_name=run_name, config=config
    )
    if imputation_set is not None:
        conn.execute(_SET_IMPUTATION, {"n": imputation_set, "run_id": run_id})
    return run_id


def complete_run(conn: Any, run_id: int, diagnostics: dict[str, Any]) -> None:
    """Stamp a run complete with its diagnostics.

    Args:
        conn: An open database connection.
        run_id: The row from :func:`start_run`.
        diagnostics: JSON-serialisable payload.
    """
    run_store.complete_model_run(conn, run_id, diagnostics)


def write_skill_estimates(conn: Any, run_id: int, rows: Sequence[SkillEstimateRow]) -> int:
    """Insert per-general posterior summaries for a run.

    Args:
        conn: An open database connection.
        run_id: The run the rows belong to.
        rows: One row per credited general.

    Returns:
        Rows written.
    """
    if rows:
        conn.execute(_INSERT_ESTIMATE, [{"run_id": run_id, **asdict(r)} for r in rows])
    logger.info("model_estimates_written", run_id=run_id, rows=len(rows))
    return len(rows)


def write_war_details(conn: Any, run_id: int, rows: Sequence[WarDetailRow]) -> int:
    """Insert per-battle WAR contributions for a run.

    Args:
        conn: An open database connection.
        run_id: The run the rows belong to.
        rows: One row per (battle, credited general).

    Returns:
        Rows written.
    """
    if rows:
        conn.execute(_INSERT_DETAIL, [{"run_id": run_id, **asdict(r)} for r in rows])
    logger.info("model_war_details_written", run_id=run_id, rows=len(rows))
    return len(rows)
