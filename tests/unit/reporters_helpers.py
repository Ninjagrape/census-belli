"""Builders shared by the report stage's unit tests. No database."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pipeline.reporters.config import ReportParams
from pipeline.reporters.records import (
    BattleDetail,
    EvaluationInfo,
    GeneralEstimate,
    ModelGate,
    RubinInfo,
    VariantSummary,
)
from pipeline.reporters.view import ReportView, build_view

PASSING_GATES = (ModelGate("exclusion_concentration", True, "warning", ""),)


def estimate(
    general_id: int,
    name: str = "",
    *,
    rank: int | None = None,
    n_battles: int = 5,
    total_war: float | None = 1.0,
    war_sd: float | None = 0.1,
    skill_mean: float = 0.5,
) -> GeneralEstimate:
    """A stored estimate with a 95% skill interval of mean +/- 1."""
    return GeneralEstimate(
        general_id=general_id,
        name=name or f"General {general_id}",
        wikidata_id=f"Q{general_id}",
        skill_mean=skill_mean,
        skill_median=skill_mean,
        skill_sd=0.5,
        skill_ci_lo=skill_mean - 1.0,
        skill_ci_hi=skill_mean + 1.0,
        skill_ci90_lo=skill_mean - 0.8,
        skill_ci90_hi=skill_mean + 0.8,
        total_war=total_war,
        war_per_battle=None if total_war is None else total_war / max(n_battles, 1),
        war_sd=war_sd,
        n_battles=n_battles,
        rank=rank,
    )


def battle(general_id: int, battle_id: int, expected_war: float) -> BattleDetail:
    """A per-battle row."""
    return BattleDetail(
        general_id=general_id,
        battle_id=battle_id,
        battle_name=f"Battle {battle_id}",
        year=1800 + battle_id,
        polity="Somewhere",
        side_outcome="victory",
        command_role="overall_commander",
        attribution_weight=1.0,
        expected_war=expected_war,
        expected_score_model=0.6,
        expected_score_replacement=0.5,
        actual_outcome=1.0,
    )


def evaluation(directory: Path, run_id: int = 7) -> EvaluationInfo:
    """A current evaluation report with one gate and two variants."""
    return EvaluationInfo(
        model_run_id=run_id,
        directory=directory,
        disclosures=(
            "Listing order predicts the winner in this corpus.",
            "Technology is not modelled. A commander from a period with a large technological "
            "edge is compared on the same scale as one without it.",
        ),
        gates=({"name": "held_out_accuracy", "passed": True, "message": "accuracy 0.698"},),
        variants=(
            VariantSummary("observed_only", "ok", None, 0.91, 12),
            VariantSummary("equal_attribution", "ok", None, 0.83, 10),
        ),
        cv_accuracy=0.7,
        cv_brier=0.22,
        cv_n=53,
        sensitivity_ranks={"observed_only": {1: 2, 2: 1}},
    )


def params(**overrides: object) -> ReportParams:
    """Default parameters with overrides."""
    return ReportParams.from_mapping({"output_root": "unused", **overrides})


def view(
    estimates: Sequence[GeneralEstimate],
    *,
    run_id: int = 7,
    evaluation_info: EvaluationInfo | None = None,
    evaluation_reason: str = "",
    gates: Sequence[ModelGate] | None = PASSING_GATES,
    report_params: ReportParams | None = None,
    battles: dict[int, tuple[BattleDetail, ...]] | None = None,
) -> ReportView:
    """A view over synthetic estimates."""
    return build_view(
        report_params or params(),
        run_id=run_id,
        estimates=estimates,
        battles=battles or {},
        counts={"battles_in_scope": 10, "battles_included": 6},
        rubin={e.general_id: RubinInfo(effective_battles=float(e.n_battles)) for e in estimates},
        model_gates=gates,
        evaluation=evaluation_info,
        evaluation_reason=evaluation_reason,
    )
