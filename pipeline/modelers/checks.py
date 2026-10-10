"""Pure dataset-level gates for the model stage.

Every function returns a :class:`CheckResult`; none reads the database or
decides what a failure means for the run (the stage maps ``severity`` to its
gate behaviour). Thresholds are keyword arguments with the design's defaults.

Statistical choices the design left open, all conservative:

- A check that cannot be evaluated for lack of data (fewer than three
  generals to correlate) fails rather than passing silently.
- ``top-50`` sets are ranked on the score the caller passes (the stage passes
  total xWAR), ties broken by ``general_id``.
- The posterior predictive check covers only battles with an exact outcome;
  an interval-censored battle has no single observed level to compare.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
from scipy.stats import spearmanr

from pipeline.modelers.model import PosteriorDraws, assemble_side_scores
from pipeline.modelers.ordinal import category_probs, linear_predictor, ordinal_probs
from pipeline.modelers.pool import rank_descending
from pipeline.modelers.priors import ModelPriors
from pipeline.modelers.records import ModelCounts, ModelDesign

__all__ = [
    "CheckResult",
    "PriorPredictive",
    "circularity_check",
    "exclusion_concentration",
    "extreme_ratio_dependence",
    "fmi_top_generals",
    "implausible_career_span",
    "own_minus_opponent",
    "posterior_predictive_category_check",
    "prior_predictive",
    "prior_predictive_sensible",
    "top_indices",
]

WARNING: Final[str] = "warning"
CATEGORY_NAMES: Final[tuple[str, ...]] = (
    "decisive_defeat",
    "defeat",
    "indecisive",
    "victory",
    "decisive_victory",
)
_MIN_FOR_CORRELATION: Final[int] = 3


@dataclass(frozen=True)
class CheckResult:
    """The outcome of one gate.

    Attributes:
        name: Gate name, as agents/model.yaml lists it.
        passed: Whether the gate holds.
        severity: ``'warning'`` or ``'error'``.
        message: One line for the log and the report.
        details: Numbers behind the verdict, JSON-serialisable.
    """

    name: str
    passed: bool
    severity: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


def top_indices(general_ids: np.ndarray, score: np.ndarray, n: int = 50) -> np.ndarray:
    """Indices of the ``n`` best generals by score, best first, ties by id.

    Args:
        general_ids: Raw ids.
        score: Higher is better.
        n: How many to take.

    Returns:
        Index array of length ``min(n, len(score))``.
    """
    ranks = rank_descending(general_ids, score)
    return np.asarray(np.argsort(ranks)[:n])


def circularity_check(
    *,
    general_ids: np.ndarray,
    beta_main: float,
    beta_joint_mean: float,
    beta_joint_sd: float,
    skill_main: np.ndarray,
    skill_joint: np.ndarray,
    score_main: np.ndarray,
    score_joint: np.ndarray,
    battles_per_general: np.ndarray,
    slope_sd_factor: float = 0.5,
    min_spearman: float = 0.9,
    min_battles: int = 3,
    top_n: int = 50,
    min_overlap: int = 40,
) -> CheckResult:
    """Compare the imputed-data fit with the joint latent-troops fit (R2).

    Passes when the pooled and joint ``beta_force`` differ by at most
    ``slope_sd_factor`` joint sds, skill means correlate (Spearman) at least
    ``min_spearman`` over generals with ``min_battles`` or more battles, and
    the top-``n`` sets share at least ``min(min_overlap, 0.8 n)`` generals.
    This does not test the shared-error limitation (handover-log 21.7).

    Args:
        general_ids: Raw ids, aligned with every per-general array.
        beta_main: Pooled ``beta_force`` mean.
        beta_joint_mean: Joint-fit ``beta_force`` mean.
        beta_joint_sd: Joint-fit ``beta_force`` sd.
        skill_main: Pooled skill means.
        skill_joint: Joint-fit skill means.
        score_main: Ranking score from the pooled fit (higher is better).
        score_joint: Ranking score from the joint fit.
        battles_per_general: Included battle count per general.
        slope_sd_factor: Allowed slope gap in joint sds.
        min_spearman: Required rank correlation of skill means.
        min_battles: Minimum battles for the correlation.
        top_n: Size of the top set.
        min_overlap: Required top-set overlap when ``top_n`` generals exist.

    Returns:
        A warning-severity result.
    """
    slope_gap = abs(beta_main - beta_joint_mean)
    slope_ok = slope_gap <= slope_sd_factor * beta_joint_sd

    keep = battles_per_general >= min_battles
    if int(keep.sum()) >= _MIN_FOR_CORRELATION:
        rho = float(spearmanr(skill_main[keep], skill_joint[keep]).statistic)
    else:
        rho = float("nan")
    rho_ok = bool(np.isfinite(rho) and rho >= min_spearman)

    n_top = min(top_n, int(general_ids.shape[0]))
    top_main = top_indices(general_ids, score_main, n_top)
    top_joint = top_indices(general_ids, score_joint, n_top)
    overlap = len(set(top_main.tolist()) & set(top_joint.tolist()))
    required = min(float(min_overlap), 0.8 * n_top)
    overlap_ok = overlap >= required

    rank_main = rank_descending(general_ids, score_main)
    rank_joint = rank_descending(general_ids, score_joint)
    shifts = np.abs(rank_main[top_main] - rank_joint[top_main])
    passed = bool(slope_ok and rho_ok and overlap_ok)
    parts = [
        f"beta_force gap {slope_gap:.3f} (limit {slope_sd_factor * beta_joint_sd:.3f})",
        f"skill Spearman {rho:.3f} (min {min_spearman})",
        f"top-{n_top} overlap {overlap} (min {required:.0f})",
    ]
    return CheckResult(
        name="circularity_check",
        passed=passed,
        severity=WARNING,
        message=("ok: " if passed else "joint latent fit disagrees: ") + "; ".join(parts),
        details={
            "beta_force_main": beta_main,
            "beta_force_joint": beta_joint_mean,
            "beta_force_joint_sd": beta_joint_sd,
            "spearman": rho,
            "n_correlated": int(keep.sum()),
            "top_overlap": overlap,
            "top_overlap_required": required,
            "top_rank_shift_max": int(shifts.max()) if shifts.size else 0,
            "top_rank_shift_median": float(np.median(shifts)) if shifts.size else 0.0,
            "limitation": "does not test the shared-error limitation (handover-log 21.7)",
        },
    )


def own_minus_opponent(design: ModelDesign, x_diff: np.ndarray) -> np.ndarray:
    """Log troops of a commander's own side minus the opponent's, per entry.

    Args:
        design: The design.
        x_diff: ``x_A - x_B`` per battle (use the median across rounds).

    Returns:
        Array aligned with ``design.cmd_side``.
    """
    return np.asarray(design.side_sign[design.cmd_side] * x_diff[design.cmd_side // 2])


def extreme_ratio_dependence(
    *,
    general_ids: np.ndarray,
    entry_general: np.ndarray,
    entry_xwar: np.ndarray,
    entry_log_ratio: np.ndarray,
    top_idx: np.ndarray,
    log_ratio_cutoff: float = -2.0,
    max_share: float = 0.5,
) -> CheckResult:
    """Flag top generals whose xWAR comes mostly from heavily outnumbered wins (R4).

    Args:
        general_ids: Raw ids.
        entry_general: General index per commander entry.
        entry_xwar: Posterior-mean xWAR per entry.
        entry_log_ratio: Own-minus-opponent log troops per entry.
        top_idx: General indices to examine.
        log_ratio_cutoff: Entries below this count as extreme.
        max_share: Largest tolerated share of absolute xWAR from them.

    Returns:
        A warning-severity result listing offenders.
    """
    offenders: list[dict[str, float]] = []
    extreme = entry_log_ratio < log_ratio_cutoff
    for idx in top_idx:
        mine = entry_general == idx
        total = float(np.abs(entry_xwar[mine]).sum())
        if total <= 0:
            continue
        share = float(np.abs(entry_xwar[mine & extreme]).sum() / total)
        if share > max_share:
            offenders.append({"general_id": float(general_ids[idx]), "share": share})
    return CheckResult(
        name="extreme_ratio_dependence",
        passed=not offenders,
        severity=WARNING,
        message=(
            "no top general depends mostly on extreme force ratios"
            if not offenders
            else f"{len(offenders)} top general(s) earn over {max_share:.0%} of |xWAR| "
            f"against odds worse than exp({log_ratio_cutoff}); one linear slope for all eras"
        ),
        details={"offenders": offenders, "log_ratio_cutoff": log_ratio_cutoff},
    )


def exclusion_concentration(
    *,
    general_ids: np.ndarray,
    n_included: np.ndarray,
    n_uncredited: np.ndarray,
    top_idx: np.ndarray,
    counts: ModelCounts,
    min_credited_share: float = 0.5,
    max_default_split_share: float = 0.3,
) -> CheckResult:
    """Whether the ranking leans on generals whose battles mostly go uncredited (R1).

    Fires when any top general has under ``min_credited_share`` of its
    battles credited, or more than ``max_default_split_share`` of all
    multi-commander sides are ``default_split``. The report must treat this
    as blocking.

    Args:
        general_ids: Raw ids.
        n_included: Credited included battles per general.
        n_uncredited: Uncredited or excluded battles per general.
        top_idx: General indices to examine.
        counts: The design's counters.
        min_credited_share: Smallest tolerated credited share.
        max_default_split_share: Largest tolerated default-split share.

    Returns:
        A warning-severity result.
    """
    total = (n_included + n_uncredited).astype(float)
    share = np.divide(n_included, total, out=np.ones_like(total), where=total > 0)
    low = [int(general_ids[i]) for i in top_idx if share[i] < min_credited_share]
    multi = counts.sides_multi_commander_all
    split_share = counts.sides_default_split_all / multi if multi else 0.0
    passed = not low and split_share <= max_default_split_share
    return CheckResult(
        name="exclusion_concentration",
        passed=passed,
        severity=WARNING,
        message=(
            "credit is not concentrated in excluded battles"
            if passed
            else f"{len(low)} top general(s) under {min_credited_share:.0%} credited; "
            f"default_split share of multi-commander sides {split_share:.0%}"
        ),
        details={
            "low_credit_generals": low,
            "default_split_share": split_share,
            "min_credited_share_seen": float(share[top_idx].min()) if len(top_idx) else 1.0,
        },
    )


def implausible_career_span(
    *, general_ids: np.ndarray, spans: np.ndarray, max_span_years: float = 60.0
) -> CheckResult:
    """Generals whose credited battles span more years than a career allows (R9).

    Args:
        general_ids: Raw ids.
        spans: ``(G, 3)`` from :func:`pipeline.modelers.components.career_spans`.
        max_span_years: Longest plausible career.

    Returns:
        A warning-severity result listing offenders (likely a merged identity).
    """
    span = spans[:, 2]
    bad = np.flatnonzero(np.nan_to_num(span, nan=0.0) > max_span_years)
    return CheckResult(
        name="implausible_career_span",
        passed=bad.size == 0,
        severity=WARNING,
        message=(
            "no career exceeds the span limit"
            if bad.size == 0
            else f"{bad.size} general(s) span over {max_span_years:.0f} years (merged identities?)"
        ),
        details={
            "generals": [
                {"general_id": int(general_ids[i]), "span_years": float(span[i])} for i in bad
            ]
        },
    )


def fmi_top_generals(
    *, general_ids: np.ndarray, fmi: np.ndarray, top_idx: np.ndarray, threshold: float = 0.3
) -> CheckResult:
    """Whether imputation uncertainty dominates any top general's estimate (R6).

    Args:
        general_ids: Raw ids.
        fmi: Rubin FMI per general.
        top_idx: General indices to examine.
        threshold: Largest tolerated FMI.

    Returns:
        A warning-severity result.
    """
    high = [int(general_ids[i]) for i in top_idx if fmi[i] > threshold]
    return CheckResult(
        name="fmi_top_generals",
        passed=not high,
        severity=WARNING,
        message=(
            "imputation uncertainty is modest for the top generals"
            if not high
            else f"{len(high)} top general(s) have FMI over {threshold}; raise n_imputations "
            "to 40 (FMI is a lower bound while the shared-error limitation stands)"
        ),
        details={
            "high_fmi_generals": high,
            "max_fmi": float(fmi[top_idx].max()) if len(top_idx) else 0.0,
        },
    )


@dataclass(frozen=True)
class PriorPredictive:
    """Summaries of outcomes implied by the priors alone.

    Attributes:
        category_freq: Mean probability of each of the five levels over draws
            and battles.
        median_abs_win_dev: Median ``|expected score - 0.5|`` (expected score is
            ``P(win) + 0.5 P(indecisive)``) for random named pairings.
        skill_gap_p99: 99th percentile of ``|skill_i - skill_j|`` over pairings.
        win_prob_quantiles: 5, 25, 50, 75, 95 percentiles of ``P(win)``
            (information only).
    """

    category_freq: np.ndarray
    median_abs_win_dev: float
    skill_gap_p99: float
    win_prob_quantiles: np.ndarray


def prior_predictive(
    design: ModelDesign,
    x_diff: np.ndarray,
    priors: ModelPriors,
    *,
    n_draws: int = 1000,
    n_pairs: int = 500,
    seed: int = 0,
) -> PriorPredictive:
    """Sample the priors on the real design and summarise the implied outcomes (R10).

    Args:
        design: The design.
        x_diff: ``x_A - x_B`` per battle.
        priors: Hyperparameters.
        n_draws: Prior draws.
        n_pairs: Random general pairings per draw for the win-probability summary.
        seed: Generator seed.

    Returns:
        The summaries.
    """
    rng = np.random.default_rng(seed)
    sigma = np.abs(rng.normal(0.0, priors.skill_prior_sd, n_draws))
    skills = sigma[:, None] * rng.normal(size=(n_draws, design.n_generals))
    n_latent = design.latent_sides.size
    z_side = rng.normal(size=(n_draws, n_latent)) if n_latent else None
    scores = assemble_side_scores(
        design,
        skills,
        sigma,
        z_side,
        rng.normal(0.0, priors.mu_unknown_sd, n_draws),
        rng.normal(0.0, priors.mu_unattributed_sd, n_draws),
    )
    beta_force = rng.normal(priors.beta_force_mu, priors.beta_force_sd, n_draws)
    beta_def = (
        rng.normal(priors.beta_def_mu, priors.beta_def_sd, n_draws)
        if design.defender_term_used
        else np.zeros(n_draws)
    )
    c1 = np.exp(rng.normal(priors.log_c1_mu, priors.log_c1_sd, n_draws))
    c2 = c1 + np.exp(rng.normal(priors.log_gap_mu, priors.log_gap_sd, n_draws))

    eta = linear_predictor(
        design.side_a, design.side_b, scores, x_diff, design.defender_diff, beta_force, beta_def
    )
    freq = np.zeros(5)
    for start in range(0, n_draws, 100):
        chunk = slice(start, start + 100)
        freq += category_probs(eta[chunk], c1[chunk, None], c2[chunk, None]).mean(axis=1).sum(0)
    freq /= n_draws

    first = rng.integers(0, design.n_generals, (n_draws, n_pairs))
    second = rng.integers(0, design.n_generals, (n_draws, n_pairs))
    rows = np.arange(n_draws)[:, None]
    gap = skills[rows, first] - skills[rows, second]
    p_win, p_indecisive, _ = ordinal_probs(gap, c1[:, None], c2[:, None])
    # Expected score is exactly 0.5 at a zero gap; P(win) alone is
    # expit(-c1), about 0.38, which would count an even pairing as lopsided.
    expected = p_win + 0.5 * p_indecisive
    return PriorPredictive(
        category_freq=freq,
        median_abs_win_dev=float(np.median(np.abs(expected - 0.5))),
        skill_gap_p99=float(np.percentile(np.abs(gap), 99)),
        win_prob_quantiles=np.percentile(p_win, [5, 25, 50, 75, 95]),
    )


def prior_predictive_sensible(
    pp: PriorPredictive, *, max_category: float = 0.4, max_median_dev: float = 0.35
) -> CheckResult:
    """Warn when the priors alone imply an implausible tournament (R10).

    Args:
        pp: Output of :func:`prior_predictive`.
        max_category: Largest tolerated prior frequency of any one level.
        max_median_dev: Largest tolerated median ``|expected score - 0.5|``.

    Returns:
        A warning-severity result.
    """
    worst = int(np.argmax(pp.category_freq))
    passed = bool(
        pp.category_freq[worst] <= max_category and pp.median_abs_win_dev <= max_median_dev
    )
    return CheckResult(
        name="prior_predictive_sensible",
        passed=passed,
        severity=WARNING,
        message=(
            "priors imply a plausible spread of outcomes"
            if passed
            else f"priors put {pp.category_freq[worst]:.0%} on {CATEGORY_NAMES[worst]} and the "
            f"median |expected score-0.5| is {pp.median_abs_win_dev:.2f}"
        ),
        details={
            "category_freq": dict(zip(CATEGORY_NAMES, pp.category_freq.tolist(), strict=True)),
            "median_abs_win_dev": pp.median_abs_win_dev,
            "skill_gap_p99": pp.skill_gap_p99,
            "win_prob_quantiles": pp.win_prob_quantiles.tolist(),
        },
    )


def posterior_predictive_category_check(
    design: ModelDesign,
    draws: PosteriorDraws,
    x_diff: np.ndarray,
    *,
    n_sim: int = 500,
    level: float = 0.95,
    seed: int = 0,
) -> CheckResult:
    """Do simulated outcome frequencies cover the observed ones? (R12)

    Simulates the exact-outcome battles from the posterior and compares each
    level's observed frequency with its central ``level`` predictive interval.
    Run it on the median round.

    Args:
        design: The design.
        draws: One round's flattened draws.
        x_diff: ``(n_battles,)`` fixed difference, or ``(T, n_battles)``.
        n_sim: Number of posterior draws to simulate from (evenly spaced).
        level: Predictive interval mass.
        seed: Generator seed.

    Returns:
        A warning-severity result listing levels outside their interval.
    """
    exact = design.k_lo == design.k_hi
    n_exact = int(exact.sum())
    if n_exact == 0:
        return CheckResult(
            "posterior_predictive_category_check", False, WARNING, "no exact-outcome battles", {}
        )
    rng = np.random.default_rng(seed)
    picks = np.linspace(0, draws.n_draws - 1, min(n_sim, draws.n_draws)).round().astype(int)
    x = x_diff if x_diff.ndim == 1 else x_diff[picks]
    eta = linear_predictor(
        design.side_a,
        design.side_b,
        draws.side_scores[picks],
        x,
        design.defender_diff,
        draws.beta_force[picks],
        draws.beta_def[picks],
    )[:, exact]
    probs = category_probs(eta, draws.c1[picks, None], draws.c2[picks, None])
    cumulative = np.cumsum(probs, axis=-1)
    u = rng.random(eta.shape)[..., None]
    simulated = np.minimum((u > cumulative).sum(axis=-1), 4)
    sim_freq = np.stack([(simulated == k).mean(axis=1) for k in range(5)], axis=1)
    observed = np.bincount(design.k_lo[exact], minlength=5) / n_exact
    tail = (1.0 - level) / 2.0 * 100.0
    lo, hi = np.percentile(sim_freq, [tail, 100.0 - tail], axis=0)
    outside = [CATEGORY_NAMES[k] for k in range(5) if not lo[k] <= observed[k] <= hi[k]]
    return CheckResult(
        name="posterior_predictive_category_check",
        passed=not outside,
        severity=WARNING,
        message=(
            "observed outcome frequencies are inside the predictive intervals"
            if not outside
            else f"observed frequency outside the {level:.0%} predictive interval for: "
            + ", ".join(outside)
        ),
        details={
            "observed": dict(zip(CATEGORY_NAMES, observed.tolist(), strict=True)),
            "lower": lo.tolist(),
            "upper": hi.tolist(),
            "n_exact_battles": n_exact,
        },
    )
