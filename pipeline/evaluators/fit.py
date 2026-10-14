"""Fitting a set of imputed rounds and pooling them, without writing anything.

The evaluate stage refits the model many times (folds, sensitivity variants).
This is the shared fit-and-pool path: it mirrors the model stage's per-round
sequence (build, sample with retry, thin, reduce) but holds results in memory.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog

from pipeline.modelers.config import ModelParams
from pipeline.modelers.fitting import FitOutcome, fit_with_retry
from pipeline.modelers.model import (
    PosteriorDraws,
    build_joint_latent_model,
    build_model,
    extract_draws,
)
from pipeline.modelers.pool import combine_rounds
from pipeline.modelers.priors import ModelPriors
from pipeline.modelers.records import ModelDesign
from pipeline.modelers.summarise import (
    PooledFit,
    RoundReduction,
    pool_reductions,
    reduce_draws,
    thin_draws,
)

__all__ = ["JointFit", "PooledResult", "RoundFit", "fit_joint", "fit_pooled", "repool"]

logger = structlog.get_logger()


@dataclass(frozen=True)
class RoundFit:
    """One imputed round's fit.

    Attributes:
        index: Position in the ``x_diffs`` list.
        outcome: The accepted (or last) sampling attempt.
        draws: Draws thinned to ``draws_kept_per_round``; None if the round
            did not converge.
        reduction: WAR reductions of those draws; None if not converged.
        x_diff: The round's fixed ``x_A - x_B``.
    """

    index: int
    outcome: FitOutcome
    draws: PosteriorDraws | None
    reduction: RoundReduction | None
    x_diff: np.ndarray


@dataclass(frozen=True)
class PooledResult:
    """All rounds of one configuration, pooled when every round converged.

    Attributes:
        rounds: One :class:`RoundFit` per round.
        converged: Whether every round met the write thresholds.
        worst: The :func:`pipeline.modelers.pool.combine_rounds` output.
        pooled: The pooled summaries, None unless ``converged``.
    """

    rounds: list[RoundFit]
    converged: bool
    worst: dict[str, Any]
    pooled: PooledFit | None


def fit_pooled(
    design: ModelDesign,
    x_diffs: Sequence[np.ndarray],
    mp: ModelParams,
    *,
    seed: int,
    label: str,
    loglik_weight: np.ndarray | None = None,
    priors: ModelPriors | None = None,
    replacement_percentile: float | None = None,
) -> PooledResult:
    """Fit every round, then pool if all converged.

    Args:
        design: The design.
        x_diffs: One fixed ``x_A - x_B`` vector per imputed round.
        mp: Run parameters.
        seed: Base seed; round ``i`` uses ``seed + i``.
        label: Name for log lines.
        loglik_weight: Optional per-battle likelihood weights (0 holds a
            battle out of training).
        priors: Prior override; defaults to ``mp.priors``.
        replacement_percentile: Override; defaults to ``mp.replacement_percentile``.

    Returns:
        The per-round fits and the pooled summary.
    """
    use_priors = mp.priors if priors is None else priors
    percentile = (
        mp.replacement_percentile if replacement_percentile is None else replacement_percentile
    )
    rounds: list[RoundFit] = []
    for i, x_diff in enumerate(x_diffs):
        model = build_model(design, x_diff, use_priors, loglik_weight=loglik_weight)
        outcome = fit_with_retry(model, mp, seed=seed + i, label=f"{label}_round_{i}")
        draws: PosteriorDraws | None = None
        reduction: RoundReduction | None = None
        if outcome.converged:
            draws = thin_draws(extract_draws(outcome.idata, design), mp.draws_kept_per_round)
            reduction = reduce_draws(design, draws, x_diff, percentile)
        rounds.append(RoundFit(i, outcome, draws, reduction, x_diff))
    worst = combine_rounds(
        [r.outcome.worst for r in rounds],
        max_rhat=mp.max_rhat,
        min_ess_bulk=mp.min_ess_bulk,
        max_divergences=mp.max_divergences,
    )
    converged = worst["converged"] == "true"
    pooled = None
    if converged:
        pooled = pool_reductions(
            design, [r.reduction for r in rounds if r.reduction is not None], mp.rank_by
        )
    logger.info("evaluator_fit_pooled", fit=label, rounds=len(rounds), converged=converged)
    return PooledResult(rounds=rounds, converged=converged, worst=worst, pooled=pooled)


def repool(
    design: ModelDesign,
    rounds: Sequence[RoundFit],
    *,
    replacement_percentile: float,
    rank_by: str,
) -> PooledFit:
    """Re-reduce kept draws under a different WAR setting, without refitting.

    Args:
        design: The design.
        rounds: Fitted rounds, each with kept draws.
        replacement_percentile: Replacement-level percentile.
        rank_by: ``total_war`` or ``skill_mean``.

    Returns:
        The pooled summaries.

    Raises:
        ValueError: If there are no rounds or any round has no draws.
    """
    if not rounds or any(r.draws is None for r in rounds):
        raise ValueError("repool needs at least one round and draws for every round")
    reductions = [
        reduce_draws(design, r.draws, r.x_diff, replacement_percentile)
        for r in rounds
        if r.draws is not None
    ]
    return pool_reductions(design, reductions, rank_by)


@dataclass(frozen=True)
class JointFit:
    """One joint latent-troops fit.

    Attributes:
        converged: Whether it met the write thresholds.
        worst: ``{max_rhat, min_ess_bulk, divergences}`` of the accepted attempt.
        draws: Thinned draws carrying their own per-draw ``x_diff``; None if
            the fit did not converge.
    """

    converged: bool
    worst: dict[str, Any]
    draws: PosteriorDraws | None


def fit_joint(
    design: ModelDesign,
    obs_mean: np.ndarray,
    obs_sd: np.ndarray,
    mp: ModelParams,
    *,
    seed: int,
    label: str,
    loglik_weight: np.ndarray | None = None,
) -> JointFit:
    """Fit the joint latent-troops model once, with the write rule's retry.

    Args:
        design: The design.
        obs_mean: Observed log-troop mean per side index, NaN where missing.
        obs_sd: Observed log-troop sd per side index, NaN where missing.
        mp: Run parameters; ``joint_sigma_w`` sets the within-battle sd.
        seed: Seed of the first attempt.
        label: Name for log lines.
        loglik_weight: Optional per-battle likelihood weights (0 holds a
            battle's outcome out of training).

    Returns:
        The fit and, if it converged, its thinned draws.
    """
    model = build_joint_latent_model(
        design, obs_mean, obs_sd, mp.joint_sigma_w, mp.priors, loglik_weight=loglik_weight
    )
    outcome = fit_with_retry(model, mp, seed=seed, label=label)
    draws: PosteriorDraws | None = None
    if outcome.converged:
        draws = thin_draws(extract_draws(outcome.idata, design), mp.draws_kept_per_round)
    logger.info("evaluator_fit_joint", fit=label, converged=outcome.converged)
    return JointFit(converged=outcome.converged, worst=outcome.worst, draws=draws)
