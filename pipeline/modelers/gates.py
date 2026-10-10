"""Assembling the model stage's dataset-level gates.

The individual checks live in :mod:`pipeline.modelers.checks`; this module adds
the two the design names that need the stage's own context
(``top_generals_cross_component`` and ``exclusions_reported``) and runs them
all against one pooled fit. Every gate is a warning; the report stage decides
which ones block publication.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pipeline.modelers.checks import (
    CheckResult,
    PriorPredictive,
    circularity_check,
    exclusion_concentration,
    extreme_ratio_dependence,
    fmi_top_generals,
    implausible_career_span,
    own_minus_opponent,
    posterior_predictive_category_check,
    prior_predictive_sensible,
    top_indices,
)
from pipeline.modelers.components import ComponentResult
from pipeline.modelers.config import GateParams
from pipeline.modelers.model import PosteriorDraws
from pipeline.modelers.records import ModelDesign
from pipeline.modelers.summarise import PooledFit

__all__ = ["exclusions_reported", "gate_dict", "run_gates", "top_generals_cross_component"]

WARNING = "warning"


def gate_dict(result: CheckResult) -> dict[str, Any]:
    """Render a result in the shape ``gate_report.json`` carries.

    Args:
        result: A gate result.

    Returns:
        ``name``, ``passed``, ``severity``, ``actual`` (the message, which is
        what the quality runner shows), ``message`` and ``details``.
    """
    return {
        "name": result.name,
        "passed": result.passed,
        "severity": result.severity,
        "actual": result.message,
        "threshold": "",
        "message": result.message,
        "details": result.details,
    }


def top_generals_cross_component(
    *,
    general_ids: np.ndarray,
    component_id: np.ndarray,
    top_idx: np.ndarray,
) -> CheckResult:
    """Count top generals outside the largest connected component.

    Skills in different components are tied together by the shared prior
    alone, so a top-ranked general there is ranked on that assumption.

    Args:
        general_ids: Raw ids.
        component_id: Component per general (0 is the largest).
        top_idx: Indices of the top generals.

    Returns:
        A warning when any top general sits outside component 0.
    """
    outside = [int(general_ids[i]) for i in top_idx if component_id[i] != 0]
    return CheckResult(
        name="top_generals_cross_component",
        passed=not outside,
        severity=WARNING,
        message=(
            "every top general is in the largest component"
            if not outside
            else f"{len(outside)} of the top {len(top_idx)} generals sit outside the largest "
            "component; their level relative to it rests on the shared prior alone"
        ),
        details={"count": len(outside), "general_ids": outside},
    )


def exclusions_reported(design: ModelDesign, max_excluded_share: float) -> CheckResult:
    """Warn when most battles in scope never entered the model.

    Args:
        design: The design (its counters hold the breakdown).
        max_excluded_share: Largest tolerated excluded share.

    Returns:
        A warning result carrying the full breakdown.
    """
    counts = design.counts
    scope = counts.battles_in_scope
    share = counts.battles_excluded / scope if scope else 0.0
    return CheckResult(
        name="exclusions_reported",
        passed=share <= max_excluded_share,
        severity=WARNING,
        message=(
            f"{counts.battles_excluded} of {scope} battles in scope were excluded "
            f"({share:.0%}); breakdown in details"
        ),
        details={"excluded_share": share, **counts.as_dict()},
    )


def run_gates(
    *,
    design: ModelDesign,
    components: ComponentResult,
    spans: np.ndarray,
    pooled: PooledFit,
    joint: PooledFit | None,
    prior_pp: PriorPredictive,
    ppc_draws: PosteriorDraws,
    ppc_x_diff: np.ndarray,
    params: GateParams,
) -> list[CheckResult]:
    """Run every dataset-level gate against the pooled fit.

    Args:
        design: The design.
        components: Comparison-graph components.
        spans: Career spans from ``career_spans``.
        pooled: The pooled summaries.
        joint: The joint-latent comparator, or None when it did not run or
            did not converge.
        prior_pp: Prior predictive summaries.
        ppc_draws: Thinned draws of the representative round.
        ppc_x_diff: That round's ``x_A - x_B``.
        params: Gate thresholds.

    Returns:
        Results in the order the spec lists them.
    """
    ids = design.general_ids
    top = top_indices(ids, pooled.score, params.top_n)
    fmi = np.maximum(pooled.skill_fmi, pooled.war_fmi)
    n_uncredited = design.n_uncredited_battles

    results = [
        fmi_top_generals(general_ids=ids, fmi=fmi, top_idx=top, threshold=params.fmi_threshold)
    ]
    if joint is None:
        results.append(
            CheckResult(
                "circularity_check",
                False,
                WARNING,
                "the joint latent fit did not run or did not converge, so the circularity "
                "comparison is unavailable",
                {},
            )
        )
    else:
        results.append(
            circularity_check(
                general_ids=ids,
                beta_main=pooled.beta_force_mean,
                beta_joint_mean=joint.beta_force_mean,
                beta_joint_sd=joint.beta_force_sd,
                skill_main=pooled.skill_mean,
                skill_joint=joint.skill_mean,
                score_main=pooled.score,
                score_joint=joint.score,
                battles_per_general=design.n_included_battles,
                slope_sd_factor=params.circ_slope_sd_factor,
                min_spearman=params.circ_min_spearman,
                min_battles=params.circ_min_battles,
                top_n=params.top_n,
                min_overlap=params.circ_min_overlap,
            )
        )
    results.extend(
        [
            exclusion_concentration(
                general_ids=ids,
                n_included=design.n_included_battles,
                n_uncredited=n_uncredited,
                top_idx=top,
                counts=design.counts,
                min_credited_share=params.min_credited_share,
                max_default_split_share=params.max_default_split_share,
            ),
            extreme_ratio_dependence(
                general_ids=ids,
                entry_general=design.cmd_general,
                entry_xwar=pooled.entry_xwar,
                entry_log_ratio=own_minus_opponent(design, pooled.x_diff),
                top_idx=top,
                log_ratio_cutoff=params.extreme_log_ratio_cutoff,
                max_share=params.extreme_max_share,
            ),
            implausible_career_span(
                general_ids=ids, spans=spans, max_span_years=params.max_career_span_years
            ),
            prior_predictive_sensible(
                prior_pp,
                max_category=params.prior_max_category,
                max_median_dev=params.prior_max_median_dev,
            ),
            top_generals_cross_component(
                general_ids=ids, component_id=components.component_id, top_idx=top
            ),
            exclusions_reported(design, params.max_excluded_share),
            posterior_predictive_category_check(
                design,
                ppc_draws,
                ppc_x_diff,
                n_sim=params.ppc_sims,
                level=params.ppc_level,
            ),
        ]
    )
    return results
