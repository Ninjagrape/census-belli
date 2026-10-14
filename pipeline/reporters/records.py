"""Immutable records passed between the report stage's loaders and renderers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "BattleDetail",
    "EvaluationInfo",
    "GeneralEstimate",
    "ModelGate",
    "RubinInfo",
    "VariantSummary",
]


@dataclass(frozen=True)
class GeneralEstimate:
    """One ``general_skill_estimates`` row joined to its general.

    ``rank`` is the model stage's stored rank by ``total_war``; this stage never
    recomputes it.
    """

    general_id: int
    name: str
    wikidata_id: str | None
    skill_mean: float
    skill_median: float | None
    skill_sd: float
    skill_ci_lo: float
    skill_ci_hi: float
    skill_ci90_lo: float | None
    skill_ci90_hi: float | None
    total_war: float | None
    war_per_battle: float | None
    war_sd: float | None
    n_battles: int
    rank: int | None


@dataclass(frozen=True)
class BattleDetail:
    """One ``battle_war_details`` row with its battle and side context.

    ``expected_score_*`` are the stored ``win_prob_*`` columns, which hold
    P(win) + 0.5 * P(indecisive), not P(win).
    """

    general_id: int
    battle_id: int
    battle_name: str
    year: int | None
    polity: str | None
    side_outcome: str | None
    command_role: str | None
    attribution_weight: float | None
    expected_war: float
    expected_score_model: float | None
    expected_score_replacement: float | None
    actual_outcome: float | None


@dataclass(frozen=True)
class RubinInfo:
    """Per-general columns of the model stage's ``rubin_per_general.csv``."""

    component_id: int | None = None
    effective_battles: float | None = None
    n_uncredited_battles: int | None = None
    first_year: float | None = None
    last_year: float | None = None


@dataclass(frozen=True)
class ModelGate:
    """One gate from the model stage's ``gate_report.json``."""

    name: str
    passed: bool
    severity: str
    message: str


@dataclass(frozen=True)
class VariantSummary:
    """A sensitivity variant's status and agreement with the baseline refit."""

    name: str
    status: str
    message: str | None
    spearman_rho: float | None
    top_overlap: int | None


@dataclass(frozen=True)
class EvaluationInfo:
    """What the report reads from a current ``evaluation_report.json``.

    Attributes:
        model_run_id: The model run the report validated.
        directory: ``{evaluation_root}/run{N}``.
        disclosures: The evaluate stage's own disclosures.
        gates: The evaluate stage's gates, as written.
        variants: Sensitivity variants, in report order.
        cv_accuracy: Held-out accuracy of the imputed arm, if recorded.
        cv_brier: Held-out Brier score of the imputed arm, if recorded.
        cv_n: Held-out battles behind those two.
        figure_files: Existing evaluation PNGs, by file name.
        sensitivity_ranks: ``variant -> {general_id: rank}`` from the
            sensitivity CSVs, as the evaluate stage ranked them.
    """

    model_run_id: int
    directory: Path
    disclosures: tuple[str, ...] = ()
    gates: tuple[Mapping[str, Any], ...] = ()
    variants: tuple[VariantSummary, ...] = ()
    cv_accuracy: float | None = None
    cv_brier: float | None = None
    cv_n: int | None = None
    figure_files: tuple[Path, ...] = ()
    sensitivity_ranks: Mapping[str, Mapping[int, int]] = field(default_factory=dict)
