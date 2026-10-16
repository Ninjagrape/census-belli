"""Turn one quantity's posterior into database-ready records.

The model samples ``mu_side`` on the log scale; nothing downstream should ever
see a log-scale number, and nothing upstream of this module should ever
average on the natural scale. Both rules matter for the same reason:

**Point estimate is ``exp(median(mu_side))``, never ``mean(exp(mu_side))``.**
For a log-normal posterior with scale ``sd``, the mean of the exponential
overstates the median by a factor of ``exp(sd**2 / 2)`` -- about +32% at
``sd = 0.74``, which is roughly what a singleton-report side's posterior
looks like. Median commutes with a monotonic transform (``exp`` is one), so
``exp(median(x)) == median(exp(x))`` and either order is correct; the mean
does not commute, and that is the whole of the bug this module exists to
avoid re-introducing.

## Reading the posterior by coordinate label, not by position

Every lookup here (a side, a source key, a regime, a source type) resolves
through the fit's own coordinate labels rather than assuming the design's
label order matches the trace. ``Design.coords()`` is what ``build_model``
passed to ``pm.Model``, so in the real pipeline the two always agree -- but
resolving by label rather than trusting that agreement is what keeps a test
built from a hand-written ``InferenceData`` (see
``tests/unit/test_reconcile_summarise.py``) exercising the same code path a
real fit runs, instead of a parallel one that only looks similar.

## What the diagnostics payload is for

``model_runs.diagnostics`` is read back by
:func:`pipeline.quality._check_diagnostics_json`, which looks for a ``worst``
object carrying ``max_rhat``, ``min_ess_bulk`` and ``divergences``. Convergence
alone is not enough to trust an estimate on this model, though: see
``pipeline/reconcilers/model.py``'s module docstring and ``docs/handover-log.md`` §19.9
for why a perfectly converged run can still be reporting an entirely-prior
number when a corpus carries no observations of the anchor regime.
``identified_level`` (``m0 + offset_bar``) and the prior-to-posterior
``contraction`` map exist so that failure mode is visible from a run log
without re-deriving it by hand each time.

## Arviz stays out of the module's import line

Same reason as ``model.py``: ``pipeline/orchestrator.py`` imports every stage
module and checks it before running anything, and arviz pulls in xarray (and,
through it, zarr), which is not a cost every orchestrator invocation should
pay. Arviz is imported inside the one function that needs it.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import structlog

from pipeline.reconcilers.design import Design

# Reaching into model.py's private constants rather than re-declaring them:
# these are judgement calls documented once, in model.py, and a second copy
# here would be free to drift from the prior the model actually sampled.
from pipeline.reconcilers.model import (
    _LEVEL_PRIOR_SD,
    _UNLABELLED_WEIGHT,
    InflationPrior,
    ModelPriors,
    _ratio_level_mask,
    _ratio_pairs,
    _unknown_level_mask,
    _unlabelled_mask,
)
from pipeline.reconcilers.records import Quantity, Report, SideEstimate, SourceBias, SourceKey

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    import arviz as az

__all__ = [
    "DEFAULT_MAX_QUANTILE_MCSE_FRAC",
    "LOW_PRECISION_MCSE_MULTIPLE",
    "SummaryResult",
    "reports_honoured",
    "singleton_slope",
    "spread_ratio",
    "summarise_fit",
]

logger = structlog.get_logger()

# A contamination share above this means the faithful component is carrying too
# little of the data; reported, not gated.
_PI_CONTAM_WARN = 0.3

# The write gate's default limits on a side's quantile MCSE: as a share of its
# interval width, and for the median as an absolute error on the log scale.
DEFAULT_MAX_QUANTILE_MCSE_FRAC = 0.05
DEFAULT_MAX_MEDIAN_MCSE_LOG = 0.05

# A side that fails the MCSE limits is written with each interval bound pushed
# out by this many of that bound's own quantile MCSEs, in log units.
LOW_PRECISION_MCSE_MULTIPLE = 2.0


@dataclass(frozen=True)
class SummaryResult:
    """One quantity's finished output: what the reconcile stage writes.

    Attributes:
        side_estimates: One per side the design covers.
        source_biases: One per ``(source_key, source_id)`` pair backing the
            fit -- a single fitted bias fanned out across every ``source_id``
            row that shares its :class:`~pipeline.reconcilers.records.SourceKey`.
        diagnostics: JSON-serialisable only: plain ``float``/``int``/``str``,
            lists and dicts, with every non-finite float already replaced by
            ``None``. This is what gets written to ``model_runs.diagnostics``
            and read back by
            :func:`pipeline.quality._check_diagnostics_json`.
    """

    side_estimates: list[SideEstimate]
    source_biases: list[SourceBias]
    diagnostics: dict[str, Any]


def summarise_fit(
    idata: az.InferenceData,
    design: Design,
    reports: list[Report],
    *,
    run_id: int,
    ci_mass: float,
    inflation: InflationPrior,
    priors: ModelPriors,
    computed_at: datetime,
    max_quantile_mcse_frac: float = DEFAULT_MAX_QUANTILE_MCSE_FRAC,
    max_median_mcse_log: float = DEFAULT_MAX_MEDIAN_MCSE_LOG,
) -> SummaryResult:
    """Turn a sampled posterior into the records the reconcile stage stores.

    Args:
        idata: The posterior from :func:`pipeline.reconcilers.model.sample`,
            fit against ``design``.
        design: The design the model was built from.
        reports: The reports used for this quantity's design. May include
            reports of the other quantity too (the caller's convenience); this
            function filters to ``design.quantity`` itself so a caller does
            not have to.
        run_id: The ``model_runs.run_id`` this fit was recorded under.
        ci_mass: The credible interval's total mass, e.g. ``0.95``. The
            interval runs from the ``(1 - ci_mass) / 2`` to the
            ``1 - (1 - ci_mass) / 2`` quantile of ``mu_side``.
        inflation: The fitted claim-regime prior the model was built with.
        priors: The hierarchy scales the model was built with.
        computed_at: The timestamp to stamp every record with.
        max_quantile_mcse_frac: The write gate's limit on a side's quantile
            MCSE as a share of its interval width; see :func:`_convergence`.
        max_median_mcse_log: The write gate's absolute limit on the MCSE of a
            side's median, in log units.

    Returns:
        The side estimates, source biases, and diagnostics payload.

    Raises:
        ValueError: If no report of ``design.quantity`` was supplied. The
            design cannot have been built from an empty set (see
            :func:`pipeline.reconcilers.design.build_design`), so this means
            the caller passed a ``reports`` list that has gone out of sync
            with the design it is meant to describe.
    """
    rows = [r for r in reports if r.quantity == design.quantity]
    if not rows:
        raise ValueError(
            f"No {design.quantity} reports were supplied to summarise a fit "
            f"whose design has {design.n_obs} {design.quantity} observations; "
            "reports and design have gone out of sync."
        )

    reports_by_side: dict[int, list[Report]] = defaultdict(list)
    for report in rows:
        reports_by_side[report.side_id].append(report)

    side_estimates = _side_estimates(idata, design, reports_by_side, run_id, ci_mass, computed_at)
    source_biases = _source_biases(idata, design, rows, run_id, computed_at)

    worst, worst_variable, divergences = _convergence(
        idata,
        ci_mass=ci_mass,
        max_quantile_mcse_frac=max_quantile_mcse_frac,
        max_median_mcse_log=max_median_mcse_log,
    )
    side_estimates = _widen_low_precision(
        side_estimates, widening_margins(idata, ci_mass, worst["mcse_failed_side_ids"])
    )
    log_estimates = _log_estimates(side_estimates)
    diagnostics: dict[str, Any] = {
        "quantity": design.quantity,
        "n_obs": design.n_obs,
        "n_sides": design.n_sides,
        "n_sources": design.n_sources,
        "n_source_keys_own_effect": design.n_source_keys_own_effect,
        "n_source_keys_pooled": design.n_source_keys_pooled,
        "reports_honoured": reports_honoured(idata, design, ci_mass=ci_mass),
        "s0": _scalar_stats(idata, "s0"),
        # Type-level observation scale: sqrt(sigma_sq_type) per source type, the
        # sigma a pooled key uses.
        "sigma_type": _sigma_type_stats(idata),
        # Do the estimates spread as widely as the reports? ~0.005 on the pilot
        # corpus where every side came back at the same level; see spread_ratio.
        "spread_ratio": spread_ratio(design, log_estimates),
        # The rhetorical weight applied to every unlabelled row in *this* run,
        # not a fitted quantity -- see model.py's module docstring for why the
        # gold set cannot supply one. Carried here so a published estimate is
        # traceable to the judgement call behind it.
        "unlabelled_weight": _UNLABELLED_WEIGHT,
        "singleton_slope": singleton_slope(design, log_estimates),
        # Round-tripped through JSON because the YAML loader parses
        # `fitted_on: 2026-09-24` as a date, which json.dumps rejects. The
        # payload promises plain JSON; a store that papers over it with
        # default=str only moved the crash to the report writer.
        "inflation_provenance": json.loads(json.dumps(inflation.provenance, default=str)),
        # Centre of the era column, over the dated unlabelled rows only (the rows
        # a_era applies to), so the column the model saw is reproducible.
        "era_mean": design.era_mean,
        "era_centre_rows": design.era_centre_rows,
        "worst": worst,
        "worst_variable": worst_variable,
        "n_sides_mcse_fail": worst["n_sides_mcse_fail"],
        "n_sides_low_precision": worst["n_sides_mcse_fail"],
        "mcse_failed_side_ids": worst["mcse_failed_side_ids"],
        "mcse_failed_limits": worst["mcse_failed_limits"],
        "worst_side": worst["worst_side"],
        "offset_bar": _scalar_stats(idata, "offset_bar"),
        "m0": _scalar_stats(idata, "m0"),
        "corpus_level": _scalar_stats(idata, "corpus_level"),
        "identified_level": _identified_level_stats(idata),
        "n_identifying_sides": n_identifying_sides(design, inflation.anchor_regime),
        "identified_width": identified_width_stats(idata, design, side_estimates, ci_mass),
        "contraction": _contraction(idata, design, priors, inflation),
        "bound_excess": _bound_excess_stats(idata, priors),
    }
    diagnostics.update(_contamination_diagnostics(idata, design, priors.contamination_sd))

    logger.info(
        "reconcile_summary_built",
        quantity=design.quantity,
        run_id=run_id,
        n_side_estimates=len(side_estimates),
        n_source_biases=len(source_biases),
        max_rhat=worst["max_rhat"],
        min_ess_bulk=worst["min_ess_bulk"],
        divergences=divergences,
    )

    return SummaryResult(
        side_estimates=side_estimates, source_biases=source_biases, diagnostics=diagnostics
    )


def widening_margins(
    idata: az.InferenceData, ci_mass: float, side_ids: list[int]
) -> dict[int, tuple[float, float]]:
    """How far to push each published bound out for sides with noisy quantiles.

    Args:
        idata: The fit's posterior.
        ci_mass: The published credible interval's total mass.
        side_ids: The sides that failed the quantile MCSE limits.

    Returns:
        ``side_id`` to ``(lower margin, upper margin)`` in log units: the
        multiple :data:`LOW_PRECISION_MCSE_MULTIPLE` of the MCSE of that bound's
        quantile. A non-finite MCSE (a single chain) gives a zero margin. Empty
        when ``side_ids`` is, without computing anything.
    """
    import arviz as az  # noqa: PLC0415 - deliberately deferred; see module docstring

    if not side_ids or "mu_side" not in idata.posterior:
        return {}
    lower_q = (1.0 - ci_mass) / 2.0
    labels = _coord_labels(idata, "side")
    errors: list[np.ndarray] = []
    for q in (lower_q, 1.0 - lower_q):
        mcse = az.mcse(idata, var_names=["mu_side"], method="quantile", prob=q)["mu_side"]
        errors.append(np.asarray(mcse.values, dtype=float))
    margins: dict[int, tuple[float, float]] = {}
    for side_id in side_ids:
        if str(side_id) not in labels:
            continue
        j = labels.index(str(side_id))
        lo_err, hi_err = (float(e[j]) if math.isfinite(float(e[j])) else 0.0 for e in errors)
        margins[side_id] = (
            LOW_PRECISION_MCSE_MULTIPLE * lo_err,
            LOW_PRECISION_MCSE_MULTIPLE * hi_err,
        )
    return margins


def widen_bounds(lo: float, hi: float, margin: tuple[float, float]) -> tuple[float, float]:
    """Push an interval's bounds out in log space by the given margins.

    Args:
        lo: Lower bound, on the natural scale.
        hi: Upper bound, on the natural scale.
        margin: ``(lower, upper)`` log-unit margins from :func:`widening_margins`.

    Returns:
        ``(exp(log(lo) - lower), exp(log(hi) + upper))``.
    """
    return math.exp(math.log(lo) - margin[0]), math.exp(math.log(hi) + margin[1])


def _widen_low_precision(
    side_estimates: list[SideEstimate], margins: dict[int, tuple[float, float]]
) -> list[SideEstimate]:
    """Widen the interval of every side that has a margin; the point estimate stays."""
    widened: list[SideEstimate] = []
    for estimate in side_estimates:
        margin = margins.get(estimate.side_id)
        if margin is None or estimate.lo is None or estimate.hi is None:
            widened.append(estimate)
            continue
        lo, hi = widen_bounds(estimate.lo, estimate.hi, margin)
        widened.append(replace(estimate, lo=lo, hi=hi))
    return widened


def _log_estimates(side_estimates: list[SideEstimate]) -> dict[int, float]:
    """Log of each side's published estimate, counting the ones that cannot be logged.

    Args:
        side_estimates: The published estimates.

    Returns:
        ``side_id`` to ``log(value)``, leaving out a side whose value is zero or
        negative (a side with no estimate is not counted); those are counted
        and logged rather than dropped silently.
    """
    positive = {
        e.side_id: math.log(e.value) for e in side_estimates if e.value is not None and e.value > 0
    }
    n_dropped = sum(e.value is not None and e.value <= 0 for e in side_estimates)
    if n_dropped:
        logger.warning("reconcile_estimates_without_a_log", n_dropped=n_dropped)
    return positive


def spread_ratio(design: Design, log_estimates: dict[int, float]) -> float | None:
    """How much of the reports' between-side spread the estimates keep.

    The sd across sides of the log estimate over the sd across sides of the
    mean observed log value. A collapsed fit (every side pulled to one level,
    as on the pilot corpus where this was about 0.005) reports clean
    convergence, so only a statistic that compares output with input can see
    it. Over 1 is possible, since estimates are shrunk toward bias-adjusted
    values rather than the raw reports; far under 1 is the failure.

    Only sides with at least one point observation enter either sd, so a side
    known only through bounds cannot move the denominator.

    Args:
        design: The design the fit used.
        log_estimates: ``side_id`` to the log of the published estimate.

    Returns:
        The ratio, or None when fewer than two sides qualify or the observed
        values do not vary.
    """
    point_rows = design.point_rows
    if point_rows.size == 0:
        return None
    sums: dict[int, float] = defaultdict(float)
    n_rows: dict[int, int] = defaultdict(int)
    for row in point_rows:
        side_id = design.side_ids[int(design.side_index[row])]
        sums[side_id] += float(design.log_y[row])
        n_rows[side_id] += 1
    sides = [s for s in sums if s in log_estimates]
    if len(sides) < 2:
        return None
    observed = np.array([sums[s] / n_rows[s] for s in sides])
    estimated = np.array([log_estimates[s] for s in sides])
    observed_sd = float(np.std(observed))
    if observed_sd <= 0.0:
        return None
    return _finite(float(np.std(estimated)) / observed_sd)


def singleton_slope(design: Design, log_estimates: dict[int, float]) -> float | None:
    """OLS slope of the log estimate on the report, over singleton sides.

    A slope near 1 means a lone report is trusted at face value up to the
    bias terms; near 0 means every singleton has been shrunk to the corpus
    mean, which the spread ratio can miss when multi-report sides keep their
    spread.

    Args:
        design: The design the fit used.
        log_estimates: ``side_id`` to the log of the published estimate.

    Returns:
        The slope, or None when fewer than three singleton sides exist or
        their reports do not vary.
    """
    counts = np.bincount(design.side_index, minlength=design.n_sides)
    xs: list[float] = []
    ys: list[float] = []
    for row in design.point_rows:
        position = int(design.side_index[row])
        side_id = design.side_ids[position]
        if counts[position] != 1 or side_id not in log_estimates:
            continue
        xs.append(float(design.log_y[row]))
        ys.append(log_estimates[side_id])
    if len(xs) < 3 or float(np.var(xs)) <= 0.0:
        return None
    return _finite(float(np.polyfit(xs, ys, 1)[0]))


def _log_normal_pdf(y: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Log density of a Normal, elementwise.

    Args:
        y: Observations.
        mu: Means.
        sd: Standard deviations.

    Returns:
        The log density.
    """
    z = (y - mu) / sd
    return cast(np.ndarray, -0.5 * z * z - np.log(sd) - 0.5 * math.log(2.0 * math.pi))


def _row_eta_and_var(
    draw: Callable[[str], np.ndarray], design: Design, rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Linear predictor and variance of the given rows, per draw.

    The same algebra as ``build_model``'s ``eta`` and ``sd ** 2``; a unit test
    compiles the model's own and checks this against it.

    Args:
        draw: Name to an array of shape ``(n_draws, ...)`` of that posterior
            variable.
        design: The design the fit used.
        rows: Row indices into the design.

    Returns:
        ``(eta, var)``, each of shape ``(n_draws, len(rows))``.
    """
    eta = (
        draw("mu_side")[:, design.side_index[rows]]
        + draw("beta_source")[:, design.source_index[rows]]
        + draw("g_regime")[:, design.regime_index[rows]]
        + draw("delta_level")[:, design.level_index[rows]]
        + draw("a_era")[:, None] * (design.era * _unlabelled_mask(design))[rows]
    )
    ratio = _ratio_level_mask(design)
    if ratio.any():
        pair_of_row, _ = _ratio_pairs(design)
        offsets = draw("sigma_ratio")[:, None] * draw("z_ratio")[:, pair_of_row[rows]]
        eta = eta + offsets * ratio[rows]
    var = (
        draw("sigma_source")[:, design.source_index[rows]] ** 2
        + draw("sigma_estimate")[:, None] ** 2 * design.is_estimate.astype(float)[rows]
        + draw("sigma_round")[:, None] ** 2 * design.roundness[rows]
        + design.range_var[rows]
    )
    unknown = _unknown_level_mask(design)
    if unknown.any():
        var = var + draw("sigma_unknown")[:, None] ** 2 * unknown[rows]
    return eta, var


def _contamination_diagnostics(
    idata: az.InferenceData, design: Design, contamination_sd: float, *, max_draws: int = 500
) -> dict[str, Any]:
    """Posterior contamination probability of each point row.

    Rebuilds each point row's linear predictor and scale from the posterior
    (see :func:`_row_eta_and_var`), then for each draw takes
    ``pi * N_wide / ((1 - pi) * N_faithful + pi * N_wide)``, averaged over
    draws. Thinned to ``max_draws`` so a long chain does not allocate a
    draws-by-rows matrix.

    Args:
        idata: The fit's posterior.
        design: The design the fit used.
        contamination_sd: The fixed log-unit sd added in the wide component.
        max_draws: The most draws to evaluate.

    Returns:
        ``pi_contam``, ``n_rows_contaminated``, ``contaminated_rows`` (every
        row with probability above 0.5, keyed by side and lineage) and
        ``top_contaminated`` (the ten most likely), or an empty dict when the
        fit has no ``pi_contam`` (another likelihood family, or a hand-built
        posterior).
    """
    if "pi_contam" not in idata.posterior or design.point_rows.size == 0:
        return {}
    pi = _pooled(idata, "pi_contam")
    stats: dict[str, Any] = {
        "mean": _finite(float(np.mean(pi))),
        "q05": _finite(float(np.quantile(pi, 0.05))),
        "q50": _finite(float(np.quantile(pi, 0.5))),
        "q95": _finite(float(np.quantile(pi, 0.95))),
    }
    if stats["q50"] is not None and stats["q50"] > _PI_CONTAM_WARN:
        logger.warning(
            "reconcile_contamination_share_is_high",
            pi_contam_q50=stats["q50"],
            threshold=_PI_CONTAM_WARN,
            hint="the faithful component may be too narrow for this corpus",
        )
    keep = np.unique(np.linspace(0, pi.shape[0] - 1, min(max_draws, pi.shape[0])).astype(int))
    pi = pi[keep]
    rows = design.point_rows

    def draw(name: str) -> np.ndarray:
        thinned: np.ndarray = _pooled(idata, name)[keep]
        return thinned

    eta, var = _row_eta_and_var(draw, design, rows)
    y = design.log_y[rows][None, :]
    log_faithful = np.log1p(-pi)[:, None] + _log_normal_pdf(y, eta, np.sqrt(var))
    log_wide = np.log(pi)[:, None] + _log_normal_pdf(y, eta, np.sqrt(var + contamination_sd**2))
    prob = np.exp(log_wide - np.logaddexp(log_faithful, log_wide)).mean(axis=0)

    def describe(i: int) -> dict[str, Any]:
        row = rows[i]
        return {
            "side_id": int(design.side_ids[int(design.side_index[row])]),
            "lineage_id": int(design.lineage_ids[int(design.lineage_index[row])]),
            "value": float(math.exp(design.log_y[row])),
            "prob": _finite(float(prob[i])),
        }

    flagged = np.flatnonzero(prob > 0.5)
    order = np.argsort(-prob)[:10]
    return {
        "pi_contam": stats,
        "n_rows_contaminated": int(flagged.size),
        "contaminated_rows": [describe(int(i)) for i in flagged[np.argsort(-prob[flagged])]],
        "top_contaminated": [describe(int(i)) for i in order],
    }


def _sigma_type_stats(idata: az.InferenceData) -> dict[str, float | None]:
    """Posterior mean of ``sqrt(sigma_sq_type)`` for each source type.

    Args:
        idata: The fit's posterior.

    Returns:
        Source type to mean observation sd on the log scale.
    """
    values = np.sqrt(_pooled(idata, "sigma_sq_type"))
    labels = _coord_labels(idata, "source_type")
    return {label: _finite(float(values[:, i].mean())) for i, label in enumerate(labels)}


def reports_honoured(
    idata: az.InferenceData, design: Design, *, ci_mass: float = 0.95
) -> dict[str, Any]:
    """Check each side's estimate against the reports that agree with each other.

    For every side with at least two point reports within a factor of two of
    each other, the credible interval of a row's modelled location (``mu_side``
    plus the source, regime, level and era terms the model adds to that row)
    must contain that row's reported value for at least one of the side's rows.
    The location is the row's own, so a labelled ancient claim is not penalised
    for legitimate deflation. A side whose consistent reports all sit outside
    its interval has been pulled away from evidence that agreed with itself,
    which is what a prior too tight for a large side does.

    Args:
        idata: The fit's posterior.
        design: The design the fit used.
        ci_mass: Total mass of the credible interval.

    Returns:
        ``n_checked``, ``n_violations``, ``violating_side_ids`` and
        ``within_side_mad`` (median absolute deviation of the log point reports
        from their side's median, over sides with two or more point reports).
    """
    mu = _pooled(idata, "mu_side")
    side_labels = _coord_labels(idata, "side")
    beta = _pooled(idata, "beta_source")
    g_regime = _pooled(idata, "g_regime")
    delta = _pooled(idata, "delta_level")
    a_era = _pooled(idata, "a_era")
    era = design.era.copy()
    if "unlabelled" in design.regimes:
        era = era * (design.regime_index == design.regimes.index("unlabelled"))
    else:
        era = era * 0.0

    point = set(design.point_rows.tolist())
    rows_of_side: dict[int, list[int]] = defaultdict(list)
    for row in sorted(point):
        rows_of_side[int(design.side_index[row])].append(row)

    lower_q = (1.0 - ci_mass) / 2.0
    checked = 0
    violations: list[int] = []
    deviations: list[float] = []
    for side_pos, rows in rows_of_side.items():
        if len(rows) < 2:
            continue
        logs = np.sort(design.log_y[rows])
        deviations.extend(np.abs(logs - np.median(logs)).tolist())
        if not bool(np.any(np.diff(logs) <= math.log(2.0))):
            continue
        checked += 1
        side_id = design.side_ids[side_pos]
        column = side_labels.index(str(side_id))
        contained = False
        for row in rows:
            eta = (
                mu[:, column]
                + beta[:, int(design.source_index[row])]
                + g_regime[:, int(design.regime_index[row])]
                + delta[:, int(design.level_index[row])]
                + a_era * era[row]
            )
            lo, hi = np.quantile(eta, [lower_q, 1.0 - lower_q])
            if lo <= design.log_y[row] <= hi:
                contained = True
                break
        if not contained:
            violations.append(int(side_id))

    return {
        "n_checked": checked,
        "n_violations": len(violations),
        "violating_side_ids": sorted(violations),
        "within_side_mad": _finite(float(np.median(deviations))) if deviations else None,
    }


# ─── Posterior access ─────────────────────────────────────────────────────────


def _pooled(idata: az.InferenceData, name: str) -> np.ndarray:
    """Flatten one posterior variable's chain and draw axes into one sample axis.

    Every summary in this module treats a draw from any chain as
    interchangeable with a draw from any other, so the two leading axes
    arviz always gives a posterior variable are pooled before anything is
    computed from it.

    Args:
        idata: The fit's posterior.
        name: A variable name in the ``posterior`` group.

    Returns:
        An array of shape ``(n_chains * n_draws, *extra_dims)``.
    """
    values = np.asarray(idata.posterior[name].values)
    return values.reshape(values.shape[0] * values.shape[1], *values.shape[2:])


def _coord_labels(idata: az.InferenceData, dim: str) -> list[str]:
    """Read one posterior dimension's coordinate labels, as strings.

    Args:
        idata: The fit's posterior.
        dim: A coordinate name, e.g. ``"side"`` or ``"source_key"``.

    Returns:
        The labels in trace order, stringified so an integer side id and a
        string regime name resolve through the same lookup.
    """
    return [str(label) for label in idata.posterior.coords[dim].values]


def _finite(value: float) -> float | None:
    """Replace a non-finite float with ``None``.

    ``json.dumps`` happily emits ``NaN`` and ``Infinity`` as bare tokens that
    are not valid JSON, which would only surface as a rejected write to a
    ``jsonb`` column downstream. Every float that reaches the diagnostics
    payload passes through here first.

    Args:
        value: A computed statistic.

    Returns:
        The value unchanged if finite, else ``None``.
    """
    return value if math.isfinite(value) else None


# ─── Side estimates ────────────────────────────────────────────────────────────


def _side_estimates(
    idata: az.InferenceData,
    design: Design,
    reports_by_side: dict[int, list[Report]],
    run_id: int,
    ci_mass: float,
    computed_at: datetime,
) -> list[SideEstimate]:
    """Build one :class:`SideEstimate` per side in the design.

    Args:
        idata: The fit's posterior.
        design: The design.
        reports_by_side: This quantity's reports, grouped by ``side_id``.
        run_id: The run these estimates belong to.
        ci_mass: The credible interval's total mass.
        computed_at: The timestamp to stamp every record with.

    Returns:
        One estimate per ``design.side_ids`` entry.
    """
    mu = _pooled(idata, "mu_side")
    side_labels = _coord_labels(idata, "side")
    lower_q = (1.0 - ci_mass) / 2.0
    upper_q = 1.0 - lower_q
    quantity = cast(Quantity, design.quantity)

    upper_bound_rows = set(design.censor_lower_at.tolist())
    lower_bound_rows = set(design.censor_upper_at.tolist())
    estimates: list[SideEstimate] = []
    for design_position, side_id in enumerate(design.side_ids):
        position = side_labels.index(str(side_id))
        samples = mu[:, position]
        # Median first, exponentiate after: see the module docstring for why
        # this is not interchangeable with exponentiating first and averaging.
        median = float(np.median(samples))
        lo_log = float(np.quantile(samples, lower_q))
        hi_log = float(np.quantile(samples, upper_q))

        # Judged on the rows that entered the fit, not on the loaded reports:
        # the cross-regime merge and the lineage collapse drop restatements, so
        # counting loaded reports called a side "source_disagreement" when the
        # model saw one claim.
        # Zero-valued rows are dropped from the likelihood, so they are not
        # evidence either; the counts below follow the method on the same rows.
        on_side = design.side_index == design_position
        on_side[design.zero_rows] = False
        fitted_rows = np.flatnonzero(on_side)
        n_lineages = len({int(design.lineage_index[row]) for row in fitted_rows})
        method = "source_disagreement" if n_lineages >= 2 else "single_report_debiased"
        # Every report bounds the side from the same direction ("up to X" and
        # nothing else): the estimate is the hierarchical prior truncated at X,
        # not evidence of where below X the truth lies. Flagged so downstream
        # stages can down-weight it (docs/handover-log.md 20.10, MEDIUM 4).
        row_set = set(fitted_rows.tolist())
        if row_set and (row_set <= upper_bound_rows or row_set <= lower_bound_rows):
            method = "bound_only"

        estimates.append(
            SideEstimate(
                side_id=side_id,
                quantity=quantity,
                value=math.exp(median),
                lo=math.exp(lo_log),
                hi=math.exp(hi_log),
                run_id=run_id,
                n_reports=len(fitted_rows),
                n_sources=len({int(design.source_index[row]) for row in fitted_rows}),
                method=method,
                updated_at=computed_at,
            )
        )
    return estimates


# ─── Source biases ─────────────────────────────────────────────────────────────


def _source_biases(
    idata: az.InferenceData,
    design: Design,
    rows: list[Report],
    run_id: int,
    computed_at: datetime,
) -> list[SourceBias]:
    """Fan a fitted bias out to every ``source_id`` sharing its key.

    One :class:`~pipeline.reconcilers.records.SourceKey` can back several
    ``sources.source_id`` rows (see that class's docstring for why), and the
    model fits one bias per key. Every ``source_id`` sharing a key gets an
    identical :class:`SourceBias` -- same ``bias_mu``, ``bias_sd``,
    ``sigma_mu``, ``sigma_sd`` and ``n_reports`` -- because the fit cannot and
    should not tell them apart.

    ``n_reports`` on each row is the key's total report count, not that count
    divided across its ``source_id`` rows: it answers "how much evidence
    backs this bias", and by construction (``design.source_keys`` is built
    only from keys that appear in the rows) that count is never zero.

    Args:
        idata: The fit's posterior.
        design: The design.
        rows: This quantity's reports.
        run_id: The run these biases belong to.
        computed_at: The timestamp to stamp every record with.

    Returns:
        One :class:`SourceBias` per ``(source_key, source_id)`` pair.
    """
    beta = _pooled(idata, "beta_source")
    sigma = _pooled(idata, "sigma_source")
    key_labels = _coord_labels(idata, "source_key")
    quantity = cast(Quantity, design.quantity)
    # With one source type b_type is a one-level ZeroSumNormal, identically zero,
    # so a pooled key's "bias" is a constant and not a measurement; it is stored
    # as NULL. With several types a pooled key carries its type's real posterior.
    # Its sigma is the type-level scale shared by every pooled key of that type.
    single_type = len(design.source_types) == 1

    reports_by_key: dict[SourceKey, list[Report]] = defaultdict(list)
    for report in rows:
        reports_by_key[report.source_key].append(report)

    biases: list[SourceBias] = []
    for key in design.source_keys:
        position = key_labels.index(f"{key.source_type}|{key.url}")
        bias_samples = beta[:, position]
        sigma_samples = sigma[:, position]
        no_bias = single_type and not bool(
            design.source_has_own_effect[design.source_keys.index(key)]
        )
        bias_mu = None if no_bias else float(np.mean(bias_samples))
        bias_sd = None if no_bias else float(np.std(bias_samples))
        sigma_mu = float(np.mean(sigma_samples))
        sigma_sd = float(np.std(sigma_samples))

        key_reports = reports_by_key.get(key, [])
        n_reports = len(key_reports)
        for source_id in sorted({r.source_id for r in key_reports}):
            biases.append(
                SourceBias(
                    source_id=source_id,
                    quantity=quantity,
                    bias_mu=bias_mu,
                    bias_sd=bias_sd,
                    sigma_mu=sigma_mu,
                    sigma_sd=sigma_sd,
                    run_id=run_id,
                    n_reports=n_reports,
                    updated_at=computed_at,
                )
            )
    return biases


# ─── Convergence diagnostics ────────────────────────────────────────────────────


def _convergence(
    idata: az.InferenceData,
    *,
    ci_mass: float = 0.95,
    max_quantile_mcse_frac: float = DEFAULT_MAX_QUANTILE_MCSE_FRAC,
    max_median_mcse_log: float = DEFAULT_MAX_MEDIAN_MCSE_LOG,
) -> tuple[dict[str, Any], dict[str, str | None], int]:
    """Roll up rhat, bulk ESS, divergences and per-side quantile error.

    Run over arviz's default variable selection for the ``posterior`` group,
    which is every free and deterministic variable the model registered --
    the non-centred ``z_*`` offsets included, since a ridge in one of those
    is exactly the failure mode that sampled cleanly on ``mu_side`` while
    ``ess(m0) = 5.3`` in the case §19.8 of ``docs/handover-log.md`` records. Restricting
    to a hand-picked variable list would have hidden that case by
    construction.

    The gate is split by what a variable publishes. Global variables (no
    ``side`` dim) keep the rhat and bulk-ESS limits. Side variables
    (``mu_side``, ``z_side``) need rhat only, plus a Monte Carlo error bound on
    the published quantities. A side's quantities are judged on the precision
    of the quantiles the stage writes, which is what the gate exists to
    protect, rather than on bulk ESS: with hundreds of side variables the
    minimum ESS fails by multiplicity alone, and it does so on exactly the
    sides whose reports are only weakly separated, where the posterior mixes
    slowly but its quantiles are still estimated well relative to the
    interval they publish. What the stage writes is the lower, 50% and upper
    quantile of ``mu_side`` (the lower and upper from ``ci_mass``), so the
    check is that the MCSE of each is at most ``max_quantile_mcse_frac`` of
    that side's interval width, and that the median's MCSE is at most
    ``max_median_mcse_log`` in absolute log units.

    Args:
        idata: The fit's posterior, plus ``sample_stats`` for divergences.
        ci_mass: The published credible interval's total mass.
        max_quantile_mcse_frac: The largest allowed quantile MCSE as a share
            of the side's interval width.
        max_median_mcse_log: The largest allowed MCSE of a side's median, in
            log units. A narrow interval can pass the relative limit while the
            median is still off by more than anyone would read from the value.

    Returns:
        A ``(worst, worst_variable, divergences)`` triple. ``worst`` maps
        ``"max_rhat"`` (every variable), ``"min_ess_bulk"`` (global variables
        only), ``"min_ess_bulk_side"`` (informational), ``"divergences"``,
        ``"n_sides_mcse_fail"`` and ``"worst_side"`` to their values, with
        ``None`` where every candidate value was non-finite (e.g. a
        single-chain fit, where rhat is undefined). ``worst_variable`` names
        which variable produced the extreme rhat and ESS.
    """
    import arviz as az  # noqa: PLC0415 - deliberately deferred; see module docstring

    rhat_ds = az.rhat(idata)
    ess_ds = az.ess(idata, method="bulk")

    max_rhat = float("-inf")
    worst_rhat_var: str | None = None
    for var_name, data_array in rhat_ds.data_vars.items():
        values = np.asarray(data_array.values, dtype=float).ravel()
        values = values[~np.isnan(values)]
        if values.size == 0:
            continue
        candidate = float(values.max())
        if candidate > max_rhat:
            max_rhat = candidate
            worst_rhat_var = str(var_name)

    min_ess = float("inf")
    worst_ess_var: str | None = None
    min_ess_side = float("inf")
    for var_name, data_array in ess_ds.data_vars.items():
        values = np.asarray(data_array.values, dtype=float).ravel()
        values = values[~np.isnan(values)]
        if values.size == 0:
            continue
        candidate = float(values.min())
        if "side" in data_array.dims or "ratio_pair" in data_array.dims:
            min_ess_side = min(min_ess_side, candidate)
        elif candidate < min_ess:
            min_ess = candidate
            worst_ess_var = str(var_name)

    # A run with no sample_stats group has recorded no divergence information
    # at all, which is distinct from having recorded zero; that distinction is
    # not representable in an int, and a fit this module has ever been asked
    # to summarise always carries sample_stats, so 0 is the practical default
    # rather than a claim that none occurred.
    divergences = 0
    if "sample_stats" in idata.groups() and "diverging" in idata.sample_stats:
        divergences = int(np.asarray(idata.sample_stats["diverging"].values).sum())

    n_fail, worst_side, failed_sides = _side_quantile_mcse(
        idata, ci_mass, max_quantile_mcse_frac, max_median_mcse_log
    )
    worst: dict[str, Any] = {
        "max_rhat": _finite(max_rhat) if worst_rhat_var is not None else None,
        "min_ess_bulk": _finite(min_ess) if worst_ess_var is not None else None,
        "min_ess_bulk_side": _finite(min_ess_side) if min_ess_side != float("inf") else None,
        "divergences": divergences,
        "n_sides_mcse_fail": n_fail,
        "worst_side": worst_side,
        "mcse_failed_side_ids": sorted(int(label) for label in failed_sides),
        "mcse_failed_limits": failed_sides,
    }
    worst_variable: dict[str, str | None] = {"rhat": worst_rhat_var, "ess_bulk": worst_ess_var}
    return worst, worst_variable, divergences


def _side_quantile_mcse(
    idata: az.InferenceData,
    ci_mass: float,
    max_frac: float,
    max_median_log: float,
) -> tuple[int, dict[str, Any] | None, dict[str, str]]:
    """Find the sides whose published quantiles are too noisy to write.

    Args:
        idata: The fit's posterior.
        ci_mass: The published credible interval's total mass; the quantiles
            checked are ``((1 - c) / 2, 0.5, (1 + c) / 2)``.
        max_frac: The largest allowed quantile MCSE as a share of the side's
            interval width.
        max_median_log: The largest allowed MCSE of the median, in log units.

    Returns:
        ``(n_fail, worst)``: the number of sides where any published quantile
        has MCSE above ``max_frac`` times the interval width, or the median's
        MCSE exceeds ``max_median_log``; and the side with the largest ratio
        to its limit as ``{"side", "quantile", "mcse_frac", "mcse", "width",
        "limit"}`` (None when the posterior has no ``mu_side``). ``limit`` is
        ``"relative"`` or ``"absolute"``, whichever the worst side is furthest
        past (or closest to), plus a mapping from each failing side's label to
        the limit it is furthest past (``"relative"`` or ``"absolute"``). A
        non-finite MCSE (a single chain) is skipped, as a non-finite rhat is.
    """
    import arviz as az  # noqa: PLC0415 - deliberately deferred; see module docstring

    if "mu_side" not in idata.posterior:
        return 0, None, {}
    lower_q = (1.0 - ci_mass) / 2.0
    quantiles = (lower_q, 0.5, 1.0 - lower_q)
    mu = _pooled(idata, "mu_side")
    lo, hi = np.quantile(mu, [quantiles[0], quantiles[2]], axis=0)
    width = np.maximum(hi - lo, np.finfo(float).tiny)
    labels = _coord_labels(idata, "side")
    errors = np.full((len(quantiles), mu.shape[1]), np.nan)
    for i, q in enumerate(quantiles):
        mcse = az.mcse(idata, var_names=["mu_side"], method="quantile", prob=q)["mu_side"]
        errors[i] = np.asarray(mcse.values, dtype=float)
    relative = errors / width
    # Each limit becomes a ratio to its own bound, so one argmax ranks sides by
    # how far past either limit they are. Only the median has an absolute limit.
    relative_ratio = np.where(np.isnan(relative), -np.inf, relative / max_frac)
    absolute_ratio = np.full_like(relative, -np.inf)
    median_ratio = errors[1] / max_median_log
    absolute_ratio[1] = np.where(np.isnan(median_ratio), -np.inf, median_ratio)
    combined = np.maximum(relative_ratio, absolute_ratio)
    per_side = combined.max(axis=0)
    n_fail = int(np.sum(per_side > 1.0))
    failed: dict[str, str] = {}
    for j in np.flatnonzero(per_side > 1.0):
        k = int(np.argmax(combined[:, j]))
        failed[labels[int(j)]] = (
            "absolute" if absolute_ratio[k, j] > relative_ratio[k, j] else "relative"
        )
    if not np.isfinite(per_side).any():
        return n_fail, None, failed
    side = int(np.argmax(per_side))
    q_index = int(np.argmax(combined[:, side]))
    absolute_wins = absolute_ratio[q_index, side] > relative_ratio[q_index, side]
    worst_side = {
        "side": labels[side],
        "quantile": quantiles[q_index],
        "mcse_frac": _finite(float(relative[q_index, side])),
        "mcse": _finite(float(errors[q_index, side])),
        "width": _finite(float(width[side])),
        "limit": "absolute" if absolute_wins else "relative",
    }
    return n_fail, worst_side, failed


def _scalar_stats(idata: az.InferenceData, name: str) -> dict[str, float | None]:
    """Posterior mean and sd of a scalar (no extra dims) variable.

    Args:
        idata: The fit's posterior.
        name: A scalar variable name in the ``posterior`` group.

    Returns:
        ``{"mean": ..., "sd": ...}``, either possibly ``None`` if non-finite.
    """
    samples = _pooled(idata, name)
    return {"mean": _finite(float(np.mean(samples))), "sd": _finite(float(np.std(samples)))}


def _identified_level_stats(idata: az.InferenceData) -> dict[str, float | None]:
    """Posterior mean and sd of ``m0 + offset_bar``, the quantity the data identify.

    ``m0`` alone is not identified on an all-unlabelled corpus (see
    ``model.py``'s module docstring and ``docs/handover-log.md`` §19.9); this is the
    corresponding rotated quantity that is.

    Args:
        idata: The fit's posterior.

    Returns:
        ``{"mean": ..., "sd": ...}``.
    """
    combined = _pooled(idata, "m0") + _pooled(idata, "offset_bar")
    return {"mean": _finite(float(np.mean(combined))), "sd": _finite(float(np.std(combined)))}


def n_identifying_sides(design: Design, anchor_regime: str) -> dict[str, int]:
    """Count, per non-anchor regime, the sides that tie it to the anchor's component.

    ``g_regime`` is identified relative to the anchor only through a chain of
    sides that carry reports in more than one regime. Two reports in a single
    regime identify the observation scale, not the regime's bias. So take the
    bipartite graph of regimes and sides (a side is linked to every regime it
    has a fitted report in) and, for each non-anchor regime ``r``, count the
    sides that have a report in ``r`` **and** a report in some other regime
    inside the anchor's connected component. A regime in a component without
    the anchor gets 0 whatever its report count, and so does every regime when
    nothing reports in the anchor (an all-unlabelled corpus), which is the
    honest answer: there is then no contrast at all.

    Args:
        design: The design; rows reporting exactly zero are not evidence and
            are skipped.
        anchor_regime: The regime pinned to zero bias.

    Returns:
        A mapping from each non-anchor regime in ``design.regimes`` to its count.
    """
    zero = set(design.zero_rows.tolist())
    regimes_of_side: dict[int, set[int]] = defaultdict(set)
    for row, (side, regime) in enumerate(zip(design.side_index, design.regime_index, strict=True)):
        if row not in zero:
            regimes_of_side[int(side)].add(int(regime))

    # Regimes joined through a shared side form one component.
    parent = list(range(len(design.regimes)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for regimes in regimes_of_side.values():
        first, *rest = sorted(regimes)
        for other in rest:
            parent[find(other)] = find(first)

    anchor = design.regimes.index(anchor_regime)
    in_anchor_component = {r for r in range(len(design.regimes)) if find(r) == find(anchor)}

    result: dict[str, int] = {}
    for r, name in enumerate(design.regimes):
        if r == anchor:
            continue
        result[name] = sum(
            1
            for regimes in regimes_of_side.values()
            if r in regimes and (regimes - {r}) & in_anchor_component
        )
    return result


def _width_summary(widths: list[float]) -> dict[str, float | int | None]:
    """Summarise a group of interval widths.

    Args:
        widths: Log-scale widths, ``LN(hi / lo)``.

    Returns:
        ``n`` and the ``min``, ``median`` and ``max`` (None when the group is
        empty).
    """
    if not widths:
        return {"n": 0, "min": None, "median": None, "max": None}
    return {
        "n": len(widths),
        "min": _finite(min(widths)),
        "median": _finite(float(np.median(widths))),
        "max": _finite(max(widths)),
    }


def _row_precision(idata: az.InferenceData, design: Design, *, max_draws: int = 500) -> np.ndarray:
    """Posterior-mean precision, ``E[1 / variance]``, of each row's noise.

    The variance is the model's own (:func:`_row_eta_and_var`) without the
    casualty ratio terms, which now sit in the location. Uses whichever of the
    noise pieces the posterior holds.

    Args:
        idata: The fit's posterior.
        design: The design.
        max_draws: Draws used, thinned evenly.

    Returns:
        ``(n_obs,)`` weights; all ones when the posterior has no source scale.
    """
    if "sigma_source" not in idata.posterior.data_vars:
        return np.ones(design.n_obs)
    sigma = _pooled(idata, "sigma_source")
    keep = np.unique(np.linspace(0, sigma.shape[0] - 1, min(max_draws, sigma.shape[0])).astype(int))
    var = sigma[keep][:, design.source_index] ** 2 + design.range_var[None, :]
    columns = {
        "sigma_estimate": design.is_estimate.astype(float),
        "sigma_round": design.roundness,
        "sigma_unknown": _unknown_level_mask(design),
    }
    for name, column in columns.items():
        if name in idata.posterior.data_vars:
            var = var + _pooled(idata, name)[keep][:, None] ** 2 * column[None, :]
    weight: np.ndarray = (1.0 / np.maximum(var, 1e-12)).mean(axis=0)
    return weight


def _side_offsets(idata: az.InferenceData, design: Design) -> np.ndarray:
    """Per-side precision-weighted mean of the row offsets added to ``mu_side``, draw by draw.

    The row offset is ``g_regime[r] + delta_level[k] + a_era * era * unlabelled``
    (``model.py``'s docstring), so a side's reports sit at ``mu_side`` plus the
    precision-weighted mean of that over its own rows. Falls back to the corpus ``offset_bar`` for
    every side when the posterior lacks the pieces.

    Args:
        idata: The fit's posterior.
        design: The design, for the row arrays and vocabularies.

    Returns:
        ``(n_pooled, n_sides)``, columns in ``design.side_ids`` order.
    """
    names = set(idata.posterior.data_vars)
    if not {"g_regime", "delta_level", "a_era"} <= names:
        return _pooled(idata, "offset_bar")[:, None]
    g_labels = _coord_labels(idata, "regime")
    g_regime = _pooled(idata, "g_regime")[:, [g_labels.index(str(r)) for r in design.regimes]]
    l_labels = _coord_labels(idata, "level")
    delta = _pooled(idata, "delta_level")[:, [l_labels.index(str(k)) for k in design.levels]]
    unlabelled = (
        design.regime_index == design.regimes.index("unlabelled")
        if "unlabelled" in design.regimes
        else np.zeros(design.n_obs, dtype=bool)
    )
    era_row = design.era * unlabelled
    regime_weight = np.zeros((design.n_sides, len(design.regimes)))
    level_weight = np.zeros((design.n_sides, len(design.levels)))
    era_weight = np.zeros(design.n_sides)
    # A precise row says more about where the side sits than a vague one, so
    # the side's offset is the precision-weighted mean of its rows' offsets,
    # the same weighting the likelihood gives them.
    row_weight = _row_precision(idata, design)
    totals = np.bincount(design.side_index, weights=row_weight, minlength=design.n_sides)
    totals[totals == 0] = 1.0
    np.add.at(regime_weight, (design.side_index, design.regime_index), row_weight)
    np.add.at(level_weight, (design.side_index, design.level_index), row_weight)
    np.add.at(era_weight, design.side_index, row_weight * era_row)
    regime_weight /= totals[:, None]
    level_weight /= totals[:, None]
    era_weight /= totals
    offsets: np.ndarray = (
        g_regime @ regime_weight.T
        + delta @ level_weight.T
        + _pooled(idata, "a_era")[:, None] * era_weight[None, :]
    )
    return offsets


def identified_width_stats(
    idata: az.InferenceData,
    design: Design,
    side_estimates: list[SideEstimate],
    ci_mass: float,
) -> dict[str, dict[str, float | int | None]]:
    """Interval widths on the identified scale, split singleton against multi-claim.

    The published interval carries the shared prior offset (``offset_bar``),
    which floors its width near 3.7 on a mostly-unlabelled corpus and so can
    never fail the narrow-singleton gate. ``mu_side`` plus the side's own
    row-weighted offset (:func:`_side_offsets`) is the quantity the data
    identify, and its width reflects the evidence alone. The corpus
    ``offset_bar`` would be wrong for a side whose rows are mostly labelled: it
    would add the unlabelled shift the side never carried. Widths are
    ``LN(hi / lo)`` of that sum, drawn jointly so the correlation between
    ``mu_side`` and the offset is kept.

    Args:
        idata: The fit's posterior.
        design: The design, for the side order.
        side_estimates: One per design side, in ``design.side_ids`` order; their
            ``method`` decides the group (``source_disagreement`` is
            multi-claim, everything else is singleton).
        ci_mass: The credible interval's total mass.

    Returns:
        ``{"singleton": ..., "multi_claim": ...}``, each the output of
        :func:`_width_summary`.
    """
    mu = _pooled(idata, "mu_side")
    labels = _coord_labels(idata, "side")
    mu = mu[:, [labels.index(str(side_id)) for side_id in design.side_ids]]
    identified = mu + _side_offsets(idata, design)
    lower_q = (1.0 - ci_mass) / 2.0
    quantiles = np.quantile(identified, [lower_q, 1.0 - lower_q], axis=0)
    widths = quantiles[1] - quantiles[0]

    singleton: list[float] = []
    multi_claim: list[float] = []
    for width, estimate in zip(widths, side_estimates, strict=True):
        group = multi_claim if estimate.method == "source_disagreement" else singleton
        group.append(float(width))
    return {"singleton": _width_summary(singleton), "multi_claim": _width_summary(multi_claim)}


def _bound_excess_stats(
    idata: az.InferenceData, priors: ModelPriors
) -> dict[str, dict[str, float | None]]:
    """Posterior of the ExGaussian bound excess rates, per direction.

    Each rate is shared by every side and source, so it is a hyperparameter
    that a reviewer should see move (or not) from its prior: on a corpus with
    few bounds the contraction is near zero and the excess is the prior.

    Args:
        idata: The fit's posterior.
        priors: Supplies ``bound_excess_prior_sd``.

    Returns:
        ``{"upper": ..., "lower": ...}`` for the rates the model built, each
        with posterior ``mean``, ``sd`` and ``contraction``. Empty under
        censored bounds.
    """
    # sd of HalfNormal(sigma) is sigma * sqrt(1 - 2 / pi).
    prior_sd = priors.bound_excess_prior_sd * math.sqrt(1.0 - 2.0 / math.pi)
    stats: dict[str, dict[str, float | None]] = {}
    for direction in ("upper", "lower"):
        name = f"lambda_{direction}"
        if name not in idata.posterior.data_vars:
            continue
        samples = _pooled(idata, name)
        posterior_sd = float(np.std(samples))
        stats[direction] = {
            "mean": _finite(float(np.mean(samples))),
            "sd": _finite(posterior_sd),
            "contraction": _contraction_value(posterior_sd, prior_sd),
        }
    return stats


def _contraction_value(posterior_sd: float, prior_sd: float) -> float | None:
    """How much a posterior sd has shrunk relative to its prior sd.

    ``1 - posterior_sd / prior_sd``: 0 means the data contributed nothing
    beyond the prior, close to 1 means the data pinned the parameter down
    tightly. Not meaningful when the prior itself has no spread.

    Args:
        posterior_sd: The fitted sd.
        prior_sd: The prior's sd for the same parameter.

    Returns:
        The contraction, or ``None`` when ``prior_sd`` is non-positive or
        either input is non-finite.
    """
    if prior_sd <= 0 or not math.isfinite(posterior_sd) or not math.isfinite(prior_sd):
        return None
    return _finite(1.0 - posterior_sd / prior_sd)


def _contraction(
    idata: az.InferenceData,
    design: Design,
    priors: ModelPriors,
    inflation: InflationPrior,
) -> dict[str, Any]:
    """Prior-to-posterior contraction for the hierarchy's shared-scale parameters.

    These are the parameters where the prior is doing real work on a sparse
    or all-unlabelled corpus (see ``model.py``'s and this module's docstrings)
    and where a reviewer would want to know, from the run log alone, whether
    the posterior moved at all.

    Args:
        idata: The fit's posterior.
        design: The design, for the regime and source-type vocabularies.
        priors: The hierarchy scales the model was built with.
        inflation: The fitted claim-regime prior the model was built with.

    Returns:
        A dict with ``"g_free"`` (per non-anchor regime), ``"a_era"``,
        ``"corpus_level"`` and ``"b_type"`` (per source type) entries.
    """
    free_regimes = [r for r in design.regimes if r != inflation.anchor_regime]

    g_free: dict[str, float | None] = {}
    if free_regimes:
        samples = _pooled(idata, "g_free")
        free_labels = _coord_labels(idata, "regime_free")
        for regime in free_regimes:
            position = free_labels.index(regime)
            _, prior_sd = inflation.moments_for(regime)
            g_free[regime] = _contraction_value(float(np.std(samples[:, position])), prior_sd)

    a_era_sd = float(np.std(_pooled(idata, "a_era")))
    level_sd = float(np.std(_pooled(idata, "corpus_level")))

    b_type_samples = _pooled(idata, "b_type")
    type_labels = _coord_labels(idata, "source_type")
    n_types = len(type_labels)
    # A ZeroSumNormal's n entries are exchangeable by construction, so every
    # entry shares one marginal prior sd: sigma * sqrt((n-1)/n). At n=1 the
    # single entry is forced to exactly zero and has no prior spread to
    # contract from, which _contraction_value reports as None rather than a
    # division by zero.
    prior_marginal_sd = (
        priors.bias_prior_sd * math.sqrt((n_types - 1) / n_types) if n_types > 1 else 0.0
    )
    b_type: dict[str, float | None] = {
        type_name: _contraction_value(float(np.std(b_type_samples[:, i])), prior_marginal_sd)
        for i, type_name in enumerate(type_labels)
    }

    return {
        "g_free": g_free,
        "a_era": _contraction_value(a_era_sd, inflation.era_sd),
        "corpus_level": _contraction_value(level_sd, _LEVEL_PRIOR_SD),
        "b_type": b_type,
    }
