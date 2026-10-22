"""Quality gates for the imputed datasets, as pure functions of the rounds.

The stage computes these before writing anything and records them in
``gate_report.json``; ``pipeline.quality``'s ``imputation_gate`` handler reads
that file. ``model_convergence`` is not here: it reads ``model_runs`` like
every other sampling stage's gate.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.stats import ks_2samp

__all__ = [
    "GateResult",
    "distributional_check",
    "imputation_completeness",
    "imputed_ratio_plausible",
    "imputed_values_plausible",
    "kept_pair_ratio_slope",
    "kept_side_slope",
    "run_gates",
]


@dataclass(frozen=True)
class GateResult:
    """One gate's verdict, JSON-serialisable."""

    name: str
    passed: bool
    severity: str
    actual: str
    threshold: str
    message: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Render for ``gate_report.json``."""
        return asdict(self)


def imputation_completeness(values: np.ndarray, *, n_sides: int, n_rounds: int) -> GateResult:
    """Every in-scope side has a finite value in every round.

    Args:
        values: ``(rounds, sides)`` log troop counts.
        n_sides: Sides that must be present.
        n_rounds: Rounds that must be present.

    Returns:
        The verdict.
    """
    shape_ok = values.ndim == 2 and values.shape == (n_rounds, n_sides)
    bad = int(np.size(values) - np.isfinite(values).sum()) if values.size else 0
    passed = shape_ok and bad == 0
    actual = f"shape={tuple(values.shape)}, non_finite={bad}"
    return GateResult(
        name="imputation_completeness",
        passed=passed,
        severity="error",
        actual=actual,
        threshold=f"shape=({n_rounds}, {n_sides}), non_finite=0",
        message="" if passed else "a side is missing from a round, or a value is not finite",
    )


def _battle_ranges(values: np.ndarray, battle_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per battle and round, the spread between its largest and smallest side.

    Args:
        values: ``(rounds, sides)`` log troop counts.
        battle_ids: Battle id per side.

    Returns:
        ``(ranges, counts)``: ``ranges`` is ``(rounds, battles)``, ``counts``
        the number of sides in each battle.
    """
    _, inverse, counts = np.unique(battle_ids, return_inverse=True, return_counts=True)
    n_battles = len(counts)
    high = np.full((values.shape[0], n_battles), -np.inf)
    low = np.full((values.shape[0], n_battles), np.inf)
    for r in range(values.shape[0]):
        np.maximum.at(high[r], inverse, values[r])
        np.minimum.at(low[r], inverse, values[r])
    return high - low, counts


def imputed_ratio_plausible(
    values: np.ndarray,
    battle_ids: np.ndarray,
    is_imputed: np.ndarray,
    *,
    limit: float,
) -> GateResult:
    """No battle with an imputed side has an implausible median log force ratio.

    Only battles where at least one side is truly missing are checked: a
    battle whose sides were both observed carries reconcile's figures, and an
    extreme ratio there is a data question for the reconcile and classify
    gates, not something imputation did. The statistic is the median across
    rounds of the battle's spread, not the worst round: a single posterior
    tail draw is the model honestly expressing uncertainty, whereas a median
    beyond the limit means the typical completed dataset is implausible.
    Values are never clipped.

    For the usual two-sided battle the spread is exactly the absolute log
    ratio; for a battle with more sides it is the widest pair.

    Args:
        values: ``(rounds, sides)`` log troop counts.
        battle_ids: Battle id per side.
        is_imputed: Per side, True if the side was truly missing.
        limit: Exclusive bound on the median spread.

    Returns:
        The verdict. The message lists the offending battle ids.
    """
    ranges, counts = _battle_ranges(values, battle_ids)
    unique_ids = np.unique(battle_ids)
    has_imputed = np.isin(unique_ids, battle_ids[is_imputed])
    checked = (counts >= 2) & has_imputed
    if not checked.any():
        return GateResult(
            "imputed_ratio_plausible",
            True,
            "error",
            "no multi-sided battles with an imputed side",
            f"median |log ratio| < {limit}",
        )
    medians = np.median(ranges[:, checked], axis=0)
    offenders = [int(b) for b in unique_ids[checked][medians >= limit]]
    passed = not offenders
    return GateResult(
        name="imputed_ratio_plausible",
        passed=passed,
        severity="error",
        actual=(
            f"battles checked={int(checked.sum())}, "
            f"max median |log ratio|={float(medians.max()):.3f}, "
            f"battles at or above limit={len(offenders)}"
        ),
        threshold=f"median |log ratio| < {limit}",
        message=""
        if passed
        else (
            "the median force ratio across rounds exceeds the plausibility limit "
            f"for battle_ids {offenders}"
        ),
    )


def imputed_values_plausible(
    values: np.ndarray,
    is_imputed: np.ndarray,
    *,
    scale: str,
    lower: float,
    upper: float,
) -> GateResult:
    """Imputed troop counts lie in a believable range (published scale only).

    The neutral scale is a ratio scale, not troop counts, so an absolute range
    means nothing there and the gate passes as not applicable.

    Args:
        values: ``(rounds, sides)`` log troop counts.
        is_imputed: Per side, True if imputed.
        scale: ``'neutral'`` or ``'published'``.
        lower: Smallest believable count.
        upper: Largest believable count.

    Returns:
        The verdict.
    """
    threshold = f"{lower:g} <= troops <= {upper:g}"
    if scale != "published":
        return GateResult(
            "imputed_values_plausible",
            True,
            "error",
            "not applicable on the neutral scale",
            threshold,
        )
    imputed = values[:, is_imputed]
    if imputed.size == 0:
        return GateResult("imputed_values_plausible", True, "error", "no imputed sides", threshold)
    low, high = math.log(lower), math.log(upper)
    n_bad = int(((imputed < low) | (imputed > high)).sum())
    passed = n_bad == 0
    return GateResult(
        name="imputed_values_plausible",
        passed=passed,
        severity="error",
        actual=f"out of range={n_bad} of {imputed.size}",
        threshold=threshold,
        message="" if passed else "imputed troop counts fall outside the believable range",
    )


def distributional_check(
    values: np.ndarray,
    is_imputed: np.ndarray,
    era_of_side: np.ndarray,
    *,
    alpha: float,
    min_n: int,
) -> GateResult:
    """Within each era, imputed values should resemble observed ones.

    A two-sample KS test per era and round; an era fails when its median
    p-value across rounds is below ``alpha``. Under MAR this is expected to
    fire whenever missingness depends on covariates: if obscure battles are
    both small and poorly documented, the imputed sides *should* differ from
    the observed ones. It is a warning for a human to read, not a defect.

    Args:
        values: ``(rounds, sides)`` log troop counts.
        is_imputed: Per side, True if imputed.
        era_of_side: Era name per side.
        alpha: Significance level.
        min_n: Eras with fewer observed or imputed sides than this are skipped.

    Returns:
        The verdict, severity warning.
    """
    failing: list[str] = []
    tested = 0
    for era in sorted({str(e) for e in era_of_side}):
        in_era = era_of_side == era
        obs, imp = in_era & ~is_imputed, in_era & is_imputed
        if obs.sum() < min_n or imp.sum() < min_n:
            continue
        tested += 1
        p_values = [
            float(ks_2samp(values[r, imp], values[r, obs]).pvalue) for r in range(values.shape[0])
        ]
        if float(np.median(p_values)) < alpha:
            failing.append(f"{era} (median p={np.median(p_values):.4f})")
    passed = not failing
    return GateResult(
        name="distributional_check",
        passed=passed,
        severity="warning",
        actual=f"eras tested={tested}, differing={failing or 'none'}",
        threshold=f"median KS p >= {alpha}",
        message=""
        if passed
        else (
            "Imputed and observed distributions differ within an era. This is expected "
            "to fire under MAR when missingness depends on covariates (small, obscure "
            "battles are both smaller and less documented); read it as information, "
            "not as a fault."
        ),
    )


def _skipped(name: str, threshold: str, reason: str) -> GateResult:
    """A recovery gate that cannot be judged: passes, and says why."""
    return GateResult(name, True, "error", f"skipped: {reason}", threshold)


def kept_pair_ratio_slope(
    values: np.ndarray,
    battle_ids: np.ndarray,
    report: np.ndarray,
    is_observed: np.ndarray,
    *,
    minimum: float,
    min_n: int,
) -> GateResult:
    """Imputed force ratios track the reported ones where both sides carry a report.

    Over battles with exactly two observed (point-reported) sides, the slope
    through the origin of the across-round mean imputed log ratio on the
    reported log ratio (the sides' likelihood means, which are the reports with
    reconcile's offsets removed). Through the origin because the log ratio is
    antisymmetric: which side is "A" (the lower row) is arbitrary. The model
    stage regresses outcome on this ratio, so a slope under 1 attenuates the
    force effect. Before 2026-10-22 it was 0.43 on the corpus.

    Args:
        values: ``(rounds, sides)`` log troop counts.
        battle_ids: Battle id per side.
        report: Per side, the reported (likelihood mean) log value; NaN if none.
        is_observed: Per side, True if it carried a likelihood.
        minimum: The slope must be at least this.
        min_n: Fewest battles for the gate to apply; below it the gate passes
            and says it skipped.

    Returns:
        The verdict, severity error.
    """
    threshold = f">= {minimum:g}"
    mean = values.mean(axis=0)
    xs: list[float] = []
    ys: list[float] = []
    for battle in np.unique(battle_ids):
        rows = np.flatnonzero((battle_ids == battle) & is_observed)
        if len(rows) != 2:
            continue
        a, b = rows
        xs.append(float(report[a] - report[b]))
        ys.append(float(mean[a] - mean[b]))
    if len(xs) < min_n:
        return _skipped("kept_pair_ratio_slope", threshold, f"{len(xs)} pairs, need {min_n}")
    denominator = float(np.sum(np.square(xs)))
    if denominator <= 0.0:
        return _skipped("kept_pair_ratio_slope", threshold, "reported ratios are all zero")
    slope = float(np.sum(np.multiply(xs, ys)) / denominator)
    passed = slope >= minimum
    return GateResult(
        name="kept_pair_ratio_slope",
        passed=passed,
        severity="error",
        actual=f"slope={slope:.3f}, battles={len(xs)}",
        threshold=threshold,
        message=""
        if passed
        else "the imputed force ratio is compressed relative to the reported one",
    )


def kept_side_slope(
    values: np.ndarray,
    report: np.ndarray,
    is_observed: np.ndarray,
    *,
    minimum: float,
    min_n: int,
) -> GateResult:
    """Observed sides keep their reported level: imputed against reported, OLS slope.

    Args:
        values: ``(rounds, sides)`` log troop counts.
        report: Per side, the reported (likelihood mean) log value; NaN if none.
        is_observed: Per side, True if it carried a likelihood.
        minimum: The slope must be at least this.
        min_n: Fewest sides for the gate to apply.

    Returns:
        The verdict, severity error.
    """
    threshold = f">= {minimum:g}"
    n = int(is_observed.sum())
    if n < min_n:
        return _skipped("kept_side_slope", threshold, f"{n} observed sides, need {min_n}")
    x = report[is_observed]
    y = values.mean(axis=0)[is_observed]
    if float(np.var(x)) <= 0.0:
        return _skipped("kept_side_slope", threshold, "reported values do not vary")
    slope = float(np.polyfit(x, y, 1)[0])
    passed = slope >= minimum
    return GateResult(
        name="kept_side_slope",
        passed=passed,
        severity="error",
        actual=f"slope={slope:.3f}, sides={n}",
        threshold=threshold,
        message="" if passed else "observed sides are pulled away from their reports",
    )


def run_gates(
    values: np.ndarray,
    *,
    is_imputed: np.ndarray,
    battle_ids: np.ndarray,
    era_of_side: np.ndarray,
    scale: str,
    n_rounds: int,
    ratio_limit: float,
    value_min: float,
    value_max: float,
    ks_alpha: float,
    ks_min_n: int,
    report: np.ndarray | None = None,
    is_observed: np.ndarray | None = None,
    min_pair_slope: float = 0.8,
    min_side_slope: float = 0.9,
    slope_min_n: int = 10,
) -> list[GateResult]:
    """Compute every dataset-level gate.

    Args:
        values: ``(rounds, sides)`` log troop counts.
        is_imputed: Per side, True if imputed.
        battle_ids: Battle id per side.
        era_of_side: Era name per side.
        scale: ``'neutral'`` or ``'published'``.
        n_rounds: Rounds that should exist.
        ratio_limit: See :func:`imputed_ratio_plausible`.
        value_min: See :func:`imputed_values_plausible`.
        value_max: See :func:`imputed_values_plausible`.
        ks_alpha: See :func:`distributional_check`.
        ks_min_n: See :func:`distributional_check`.
        report: Per side, the reported log value (NaN where unobserved). The two
            slope gates are added only when this and ``is_observed`` are given.
        is_observed: Per side, True if it carried a likelihood.
        min_pair_slope: See :func:`kept_pair_ratio_slope`.
        min_side_slope: See :func:`kept_side_slope`.
        slope_min_n: Fewest pairs or sides for a slope gate to apply.

    Returns:
        The gate results, in spec order.
    """
    complete = imputation_completeness(values, n_sides=len(battle_ids), n_rounds=n_rounds)
    finite = np.where(np.isfinite(values), values, 0.0)
    results = [
        complete,
        imputed_ratio_plausible(finite, battle_ids, is_imputed, limit=ratio_limit),
        imputed_values_plausible(finite, is_imputed, scale=scale, lower=value_min, upper=value_max),
        distributional_check(finite, is_imputed, era_of_side, alpha=ks_alpha, min_n=ks_min_n),
    ]
    if report is not None and is_observed is not None:
        results.append(
            kept_pair_ratio_slope(
                finite, battle_ids, report, is_observed, minimum=min_pair_slope, min_n=slope_min_n
            )
        )
        results.append(
            kept_side_slope(finite, report, is_observed, minimum=min_side_slope, min_n=slope_min_n)
        )
    return results
