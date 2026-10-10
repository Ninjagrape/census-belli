"""Reducing each round's posterior and pooling the reductions.

A round's full draws (thousands, times sides) are too large to hold for twenty
rounds, so each round is thinned to ``keep`` evenly spaced draws, WAR is
computed once on those, and only the per-general and per-entry reductions
survive (design section 7, R3, R6). Pooling then uses :mod:`pipeline.modelers.pool`.

The decomposition the report gate checks, ``total_war == sum(war_contribution)``,
holds by construction here because both are means over the same pooled draws of
quantities that are linear in each other; :func:`assert_decomposition` verifies it
in code before anything is written.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from pipeline.modelers.model import PosteriorDraws
from pipeline.modelers.pool import mixture, rank_descending, rubin, widen_interval
from pipeline.modelers.records import ModelDesign
from pipeline.modelers.war import centre_skills, compute_war

__all__ = [
    "DECOMPOSITION_TOLERANCE",
    "PooledFit",
    "RoundReduction",
    "assert_decomposition",
    "pool_reductions",
    "reduce_draws",
    "summarise_single",
    "thin_draws",
]

DECOMPOSITION_TOLERANCE = 1e-6
_CI95 = 0.95
_CI90 = 0.90


@dataclass(frozen=True)
class RoundReduction:
    """What survives of one fit after WAR is computed.

    Attributes:
        skills_c: Per-draw-centred skills ``(K, G)``.
        total_xwar: Total expected WAR per general per draw ``(K, G)``.
        beta_force: ``(K,)``.
        sigma_skill: ``(K,)``.
        entry_xwar: Posterior-mean xWAR per commander entry ``(E,)``.
        entry_realised: Posterior-mean realised WAR per entry ``(E,)``.
        e_model: Posterior-mean expected score per side ``(S,)``.
        e_rep: Posterior-mean replacement expected score per side ``(S,)``.
        x_diff: ``x_A - x_B`` per battle ``(B,)`` (posterior mean for the
            joint fit, the round's value otherwise).
        total_realised: Posterior-mean realised WAR per general ``(G,)``.
    """

    skills_c: np.ndarray
    total_xwar: np.ndarray
    beta_force: np.ndarray
    sigma_skill: np.ndarray
    entry_xwar: np.ndarray
    entry_realised: np.ndarray
    e_model: np.ndarray
    e_rep: np.ndarray
    x_diff: np.ndarray
    total_realised: np.ndarray


@dataclass(frozen=True)
class PooledFit:
    """Per-general posterior summaries of a pooled (or single) fit.

    ``*_fmi`` are zero for a single fit. ``sd`` is ``sqrt(T)`` for a pooled
    fit and the ordinary posterior sd for a single one.
    """

    skill_mean: np.ndarray
    skill_median: np.ndarray
    skill_sd: np.ndarray
    ci95_lo: np.ndarray
    ci95_hi: np.ndarray
    ci90_lo: np.ndarray
    ci90_hi: np.ndarray
    skill_within: np.ndarray
    skill_between: np.ndarray
    skill_fmi: np.ndarray
    war_mean: np.ndarray
    war_sd: np.ndarray
    war_fmi: np.ndarray
    realised_mean: np.ndarray
    entry_xwar: np.ndarray
    entry_realised: np.ndarray
    e_model: np.ndarray
    e_rep: np.ndarray
    x_diff: np.ndarray
    skill_draws: np.ndarray
    beta_force_mean: float
    beta_force_sd: float
    beta_force_fmi: float
    score: np.ndarray
    rank: np.ndarray


def thin_draws(draws: PosteriorDraws, keep: int) -> PosteriorDraws:
    """Keep ``keep`` evenly spaced draws.

    Args:
        draws: Flattened draws.
        keep: How many to keep; at most the number available.

    Returns:
        The thinned draws.
    """
    n = draws.n_draws
    picks = np.linspace(0, n - 1, min(keep, n)).round().astype(int)
    return PosteriorDraws(
        skills=draws.skills[picks],
        side_scores=draws.side_scores[picks],
        beta_force=draws.beta_force[picks],
        beta_def=draws.beta_def[picks],
        c1=draws.c1[picks],
        c2=draws.c2[picks],
        sigma_skill=draws.sigma_skill[picks],
        x_diff=None if draws.x_diff is None else draws.x_diff[picks],
    )


def reduce_draws(
    design: ModelDesign,
    draws: PosteriorDraws,
    x_diff: np.ndarray | None,
    replacement_percentile: float,
) -> RoundReduction:
    """Compute WAR on already-thinned draws and keep only the reductions.

    Args:
        design: The design.
        draws: Thinned draws of one fit.
        x_diff: The round's fixed ``x_A - x_B``, or None for the joint fit
            (whose draws carry their own).
        replacement_percentile: Replacement-level percentile.

    Returns:
        The reductions.

    Raises:
        ValueError: If neither ``x_diff`` nor the draws provide one.
    """
    effective = draws.x_diff if x_diff is None else x_diff
    if effective is None:
        raise ValueError("a joint-latent fit's draws must carry x_diff")
    war = compute_war(
        design,
        draws.skills,
        draws.side_scores,
        effective,
        draws.beta_force,
        draws.beta_def,
        draws.c1,
        draws.c2,
        replacement_percentile,
    )
    mean_x = effective if effective.ndim == 1 else effective.mean(axis=0)
    return RoundReduction(
        skills_c=centre_skills(draws.skills),
        total_xwar=war.total_xwar,
        beta_force=draws.beta_force,
        sigma_skill=draws.sigma_skill,
        entry_xwar=war.xwar_entry.mean(axis=0),
        entry_realised=war.realised_entry.mean(axis=0),
        e_model=war.e_model.mean(axis=0),
        e_rep=war.e_rep.mean(axis=0),
        x_diff=np.asarray(mean_x, dtype=float),
        total_realised=war.total_realised.mean(axis=0),
    )


def _mean_of(arrays: Sequence[np.ndarray]) -> np.ndarray:
    """Elementwise mean over rounds (every round contributes equally)."""
    return np.asarray(np.mean(np.stack(arrays), axis=0))


def _score(rank_by: str, skill_mean: np.ndarray, war_mean: np.ndarray) -> np.ndarray:
    return skill_mean if rank_by == "skill_mean" else war_mean


def pool_reductions(
    design: ModelDesign, reductions: Sequence[RoundReduction], rank_by: str
) -> PooledFit:
    """Pool rounds: mixture draws, Rubin variance, widened intervals, ranks.

    Args:
        design: The design.
        reductions: One per imputed round.
        rank_by: ``total_war`` or ``skill_mean``.

    Returns:
        The pooled summaries.
    """
    skills_by_round = [r.skills_c for r in reductions]
    war_by_round = [r.total_xwar for r in reductions]
    beta_by_round = [r.beta_force[:, None] for r in reductions]
    pooled = mixture(skills_by_round)
    skill_rubin = rubin(skills_by_round)
    war_rubin = rubin(war_by_round)
    beta_rubin = rubin(beta_by_round)
    lo95, hi95 = widen_interval(pooled, skill_rubin.total, _CI95)
    lo90, hi90 = widen_interval(pooled, skill_rubin.total, _CI90)
    skill_mean = pooled.mean(axis=0)
    war_mean = war_rubin.mean
    score = _score(rank_by, skill_mean, war_mean)
    return PooledFit(
        skill_mean=skill_mean,
        skill_median=np.median(pooled, axis=0),
        skill_sd=skill_rubin.sd,
        ci95_lo=lo95,
        ci95_hi=hi95,
        ci90_lo=lo90,
        ci90_hi=hi90,
        skill_within=skill_rubin.within,
        skill_between=skill_rubin.between,
        skill_fmi=skill_rubin.fmi,
        war_mean=war_mean,
        war_sd=war_rubin.sd,
        war_fmi=war_rubin.fmi,
        realised_mean=_mean_of([r.total_realised for r in reductions]),
        entry_xwar=_mean_of([r.entry_xwar for r in reductions]),
        entry_realised=_mean_of([r.entry_realised for r in reductions]),
        e_model=_mean_of([r.e_model for r in reductions]),
        e_rep=_mean_of([r.e_rep for r in reductions]),
        x_diff=np.median(np.stack([r.x_diff for r in reductions]), axis=0),
        skill_draws=pooled,
        beta_force_mean=float(beta_rubin.mean[0]),
        beta_force_sd=float(beta_rubin.sd[0]),
        beta_force_fmi=float(beta_rubin.fmi[0]),
        score=score,
        rank=rank_descending(design.general_ids, score),
    )


def summarise_single(design: ModelDesign, reduction: RoundReduction, rank_by: str) -> PooledFit:
    """The same summaries for a single fit (the joint latent comparator).

    Args:
        design: The design.
        reduction: The fit's reduction.
        rank_by: ``total_war`` or ``skill_mean``.

    Returns:
        Summaries with plain posterior sds, zero FMI and ordinary quantile
        intervals.
    """
    skills = reduction.skills_c
    war = reduction.total_xwar
    q = np.percentile(skills, [2.5, 97.5, 5.0, 95.0], axis=0)
    skill_mean = skills.mean(axis=0)
    war_mean = war.mean(axis=0)
    score = _score(rank_by, skill_mean, war_mean)
    zeros = np.zeros_like(skill_mean)
    return PooledFit(
        skill_mean=skill_mean,
        skill_median=np.median(skills, axis=0),
        skill_sd=skills.std(axis=0, ddof=1),
        ci95_lo=q[0],
        ci95_hi=q[1],
        ci90_lo=q[2],
        ci90_hi=q[3],
        skill_within=skills.var(axis=0, ddof=1),
        skill_between=zeros,
        skill_fmi=zeros,
        war_mean=war_mean,
        war_sd=war.std(axis=0, ddof=1),
        war_fmi=zeros,
        realised_mean=reduction.total_realised,
        entry_xwar=reduction.entry_xwar,
        entry_realised=reduction.entry_realised,
        e_model=reduction.e_model,
        e_rep=reduction.e_rep,
        x_diff=reduction.x_diff,
        skill_draws=skills,
        beta_force_mean=float(reduction.beta_force.mean()),
        beta_force_sd=float(reduction.beta_force.std(ddof=1)),
        beta_force_fmi=0.0,
        score=score,
        rank=rank_descending(design.general_ids, score),
    )


def assert_decomposition(design: ModelDesign, fit: PooledFit) -> float:
    """Check that each general's total WAR equals the sum of their entries.

    Args:
        design: The design.
        fit: The pooled summaries.

    Returns:
        The largest absolute discrepancy.

    Raises:
        RuntimeError: If it exceeds :data:`DECOMPOSITION_TOLERANCE`; the
            report gate ``war_decomposition_reconciles`` would fail on the
            written rows.
    """
    sums = np.zeros(design.n_generals)
    np.add.at(sums, design.cmd_general, fit.entry_xwar)
    worst = float(np.max(np.abs(sums - fit.war_mean))) if sums.size else 0.0
    if worst > DECOMPOSITION_TOLERANCE:
        raise RuntimeError(
            f"total_war differs from the sum of war_contribution by {worst:.3g}: "
            "the per-entry and per-general means are not over the same draws"
        )
    return worst
