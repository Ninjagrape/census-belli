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
from dataclasses import dataclass
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
)
from pipeline.reconcilers.records import Quantity, Report, SideEstimate, SourceBias, SourceKey

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    import arviz as az

__all__ = ["SummaryResult", "reports_honoured", "spread_ratio", "summarise_fit"]

logger = structlog.get_logger()


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

    worst, worst_variable, divergences = _convergence(idata)
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
        "spread_ratio": spread_ratio(
            design,
            {e.side_id: math.log(e.value) for e in side_estimates if e.value},
        ),
        # The rhetorical weight applied to every unlabelled row in *this* run,
        # not a fitted quantity -- see model.py's module docstring for why the
        # gold set cannot supply one. Carried here so a published estimate is
        # traceable to the judgement call behind it.
        "unlabelled_weight": _UNLABELLED_WEIGHT,
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
        "offset_bar": _scalar_stats(idata, "offset_bar"),
        "m0": _scalar_stats(idata, "m0"),
        "corpus_level": _scalar_stats(idata, "corpus_level"),
        "identified_level": _identified_level_stats(idata),
        "n_identifying_sides": _n_identifying_sides(rows, design.regimes),
        "contraction": _contraction(idata, design, priors, inflation),
    }

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
) -> tuple[dict[str, float | int | None], dict[str, str | None], int]:
    """Roll up rhat, bulk ESS and divergences across every sampled variable.

    Run over arviz's default variable selection for the ``posterior`` group,
    which is every free and deterministic variable the model registered --
    the non-centred ``z_*`` offsets included, since a ridge in one of those
    is exactly the failure mode that sampled cleanly on ``mu_side`` while
    ``ess(m0) = 5.3`` in the case §19.8 of ``docs/handover-log.md`` records. Restricting
    to a hand-picked variable list would have hidden that case by
    construction. Deterministics are included too: they are cheap here (no
    extra sampling, just a reduction over an already-materialised array), and
    a redundant fully-determined entry contributes rhat/ESS values that get
    silently skipped below rather than corrupting the max/min.

    Args:
        idata: The fit's posterior, plus ``sample_stats`` for divergences.

    Returns:
        A ``(worst, worst_variable, divergences)`` triple. ``worst`` maps
        ``"max_rhat"``, ``"min_ess_bulk"`` and ``"divergences"`` to their
        values, with ``None`` where every candidate value was non-finite
        (e.g. a single-chain fit, where rhat is undefined). ``worst_variable``
        names which variable produced the extreme rhat and ESS.
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
    for var_name, data_array in ess_ds.data_vars.items():
        values = np.asarray(data_array.values, dtype=float).ravel()
        values = values[~np.isnan(values)]
        if values.size == 0:
            continue
        candidate = float(values.min())
        if candidate < min_ess:
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

    worst: dict[str, float | int | None] = {
        "max_rhat": _finite(max_rhat) if worst_rhat_var is not None else None,
        "min_ess_bulk": _finite(min_ess) if worst_ess_var is not None else None,
        "divergences": divergences,
    }
    worst_variable: dict[str, str | None] = {"rhat": worst_rhat_var, "ess_bulk": worst_ess_var}
    return worst, worst_variable, divergences


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


def _n_identifying_sides(rows: list[Report], regimes: tuple[str, ...]) -> dict[str, int]:
    """Count, per regime, how many sides have at least two reports carrying it.

    A regime fitted from a single report per side everywhere is not actually
    contrasted against anything within a side; this is the count that says
    whether ``g_regime`` for a given regime is backed by real within-side
    disagreement or is only ever seen once per side.

    Args:
        rows: This quantity's reports.
        regimes: The full regime vocabulary from the design, so every regime
            gets an entry even at zero.

    Returns:
        A mapping from regime name to the number of qualifying sides.
    """
    counts_by_side_regime: dict[tuple[int, str], int] = defaultdict(int)
    for report in rows:
        counts_by_side_regime[(report.side_id, report.claim_regime)] += 1

    result = {regime: 0 for regime in regimes}
    for (_side_id, regime), n in counts_by_side_regime.items():
        # A regime can be absent from the design's vocabulary when every one of
        # its rows was merged away as a duplicate of a labelled row.
        if n >= 2 and regime in result:
            result[regime] += 1
    return result


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
