"""The assembled, renderer-ready view of one model run."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pipeline.reporters.config import ReportParams
from pipeline.reporters.disclosures import (
    INTERVAL_METHOD_NOTE,
    STATIC_DISCLOSURES,
    interval_method_note,
    merge_disclosures,
    p_above_next_note,
    run_disclosures,
)
from pipeline.reporters.records import (
    BattleDetail,
    EvaluationInfo,
    GeneralEstimate,
    ModelGate,
    PooledDraws,
    RubinInfo,
)
from pipeline.reporters.status import provisional_reasons
from pipeline.reporters.tables import RankedGeneral, split_generals, top_headline, z_for_level

__all__ = ["ReportView", "build_view"]


@dataclass(frozen=True)
class ReportView:
    """Everything the renderers need, with nothing left to compute.

    Attributes:
        model_run_id: The reported run.
        params: The validated stage parameters.
        z: Multiplier of the fallback normal-approximation WAR interval.
        uses_normal_approx: Whether any headline interval is that fallback
            (a run from before the model stage stored WAR intervals).
        provisional: Whether the outputs carry a provisional banner.
        reasons: One entry per reason the outputs are provisional.
        headline: Every ranked general (at least ``min_battles``), stored order.
        under_determined: Generals listed without a rank.
        battles: Per-battle rows by general id.
        disclosures: The merged, deduplicated disclosures.
        evaluation: The current evaluation report, if one was found.
        evaluation_reason: Why there is none, else empty.
    """

    model_run_id: int
    params: ReportParams
    z: float
    uses_normal_approx: bool
    provisional: bool
    reasons: tuple[str, ...]
    headline: tuple[RankedGeneral, ...]
    under_determined: tuple[RankedGeneral, ...]
    battles: Mapping[int, tuple[BattleDetail, ...]]
    disclosures: tuple[str, ...]
    evaluation: EvaluationInfo | None
    evaluation_reason: str

    @property
    def top(self) -> tuple[RankedGeneral, ...]:
        """The leading ``top_n`` headline rows."""
        return top_headline(self.headline, self.params.top_n)


def _ranks_in_gaps(headline: Sequence[RankedGeneral], under: Sequence[RankedGeneral]) -> int:
    """Unlisted generals whose stored rank falls inside the headline's rank range."""
    ranks = [g.rank for g in headline if g.rank is not None]
    if not ranks:
        return 0
    lo, hi = min(ranks), max(ranks)
    return sum(1 for g in under if g.estimate.rank is not None and lo <= g.estimate.rank <= hi)


def build_view(
    params: ReportParams,
    *,
    run_id: int,
    estimates: Sequence[GeneralEstimate],
    battles: Mapping[int, tuple[BattleDetail, ...]],
    counts: Mapping[str, Any],
    rubin: Mapping[int, RubinInfo],
    model_gates: Sequence[ModelGate] | None,
    evaluation: EvaluationInfo | None,
    evaluation_reason: str,
    evaluation_severities: Mapping[str, str] | None = None,
    pooled_draws: PooledDraws | None = None,
) -> ReportView:
    """Combine the loaded inputs into one view, applying the split and the flags.

    Args:
        params: The validated parameters.
        run_id: The reported model run.
        estimates: Every stored estimate of the run.
        battles: Per-battle rows by general.
        counts: ``model_runs.diagnostics.counts``.
        rubin: Per-general model-stage columns.
        model_gates: The run's model gates, or None when unreadable.
        evaluation: The current evaluation info, or None.
        evaluation_reason: Why there is no usable evaluation, else empty.
        evaluation_severities: Evaluate gate severities from the spec.
        pooled_draws: The run's pooled draws, used to recompute P(above next)
            over the headline ordering; None keeps the stored column.

    Returns:
        The view.
    """
    z = z_for_level(params.ci_level)
    headline, under = split_generals(
        estimates,
        rubin,
        min_battles=params.min_battles,
        z=z,
        overlap_threshold=params.overlap_threshold,
        show_overlap=params.show_rank_overlap,
        pooled=pooled_draws,
    )
    reasons = provisional_reasons(
        model_gates,
        evaluation_reason,
        require_passing_diagnostics=params.require_passing_diagnostics,
        evaluation_gates=evaluation.gates if evaluation else (),
        evaluation_severities=evaluation_severities,
    )
    if pooled_draws is None:
        p_source = "stored"
    elif pooled_draws.skill_unavailable:
        p_source = "skill_unavailable"
    else:
        p_source = pooled_draws.basis
    uses_approx = any(g.war_interval_approx for g in headline)
    static = tuple(
        interval_method_note(uses_normal_approx=uses_approx) if s == INTERVAL_METHOD_NOTE else s
        for s in STATIC_DISCLOSURES
    ) + (p_above_next_note(p_source),)
    disclosures = merge_disclosures(
        static,
        evaluation.disclosures if evaluation else (),
        run_disclosures(
            counts,
            min_battles=params.min_battles,
            n_under_determined=len(under),
            n_ranks_in_gaps=_ranks_in_gaps(headline, under),
        ),
    )
    return ReportView(
        model_run_id=run_id,
        params=params,
        z=z,
        uses_normal_approx=uses_approx,
        provisional=bool(reasons),
        reasons=reasons,
        headline=headline,
        under_determined=under,
        battles=battles,
        disclosures=disclosures,
        evaluation=evaluation,
        evaluation_reason=evaluation_reason,
    )
