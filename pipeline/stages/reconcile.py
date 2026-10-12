"""
The reconcile stage: conflicting troop and casualty reports to single estimates.

Everything upstream of this module -- the loader, the design matrix, the
hierarchical model, the posterior summary -- has existed since 2026-09-24
(docs/handover-log.md §19) and has been exercised only against synthetic data or a
fixture database. This module is what finally calls all of it in sequence
against a real connection and commits the first estimate this project has
ever published. There is no fallback path that skips the database: reconcile
produces nothing except database writes, so a stage with no connection has
nothing to do and fails loudly rather than reporting a hollow success.

## The write sequence

Troops and casualties are fit as two independent models against two
different report subsets (see ``pipeline/reconcilers/design.py``), but they
share **one** ``model_runs`` row per stage invocation rather than one each.
That is not the more obvious design -- each quantity has its own posterior --
and it is deliberate, not an economy:

``agents/reconcile.yaml``'s ``model_convergence`` gate reads the *newest*
completed run of ``model_type='source_disagreement'`` (there is one model
type, not one per quantity), and ``estimates_are_not_stale`` compares every
side's ``est_troops_run_id`` against ``MAX(run_id)`` for that same model
type. Two rows per invocation would mean whichever quantity is fit second
is the only one either gate can ever see: the other quantity's convergence
goes ungated, and its estimates read permanently stale against a `MAX(run_id)`
that belongs to a different quantity's fit entirely. One row per invocation,
carrying both quantities' diagnostics, is what makes both gates measure what
they claim to.

For a stage invocation with at least one usable report of either quantity:

1. Insert one ``model_runs`` row (``started_at`` set, ``completed_at`` NULL).
   The run id is needed *before* sampling: it is what
   :func:`~pipeline.reconcilers.summarise.summarise_fit` stamps every
   estimate and bias with, for both quantities alike.
2. For each quantity that has at least one usable report: build and sample
   its model, summarise the posterior into
   :class:`~pipeline.reconcilers.records.SideEstimate` and
   :class:`~pipeline.reconcilers.records.SourceBias` records, and UPDATE
   every estimated side and every fitted source -- all under the one run id
   from step 1.
   Troops is fitted twice: the published fit, then a regime-neutral fit on
   the same design (``build_model(neutral=True)``) whose estimates serve force
   ratios. Both belong to the one run id, and the neutral diagnostics sit in
   their own ``troops_neutral`` block, outside the rolled-up ``"worst"``, so
   ``model_convergence`` keeps meaning the published fits and the separate
   ``neutral_fit_converged`` gate speaks for the neutral one. Neutral
   estimates and ``neutral_draws_run{run_id}.npz`` are written only when it
   passes the same write thresholds; there is no fallback to published ones.
3. Stamp the ``model_runs`` row complete, with a diagnostics payload holding
   each fitted quantity's own diagnostics plus a rolled-up ``"worst"``
   object (the max rhat, min ESS and summed divergences across whichever
   quantities were fit) -- see :func:`_combined_worst`. That ``"worst"``
   object is what ``pipeline.quality``'s ``diagnostics_json`` handler reads.

If step 2 raises for either quantity, the row from step 1 is left with
``completed_at`` NULL -- which is what makes ``model_run_recorded`` correctly
fail rather than pass on a run that never finished -- and the exception
propagates; whatever the *other* quantity already wrote to ``battle_sides``
or ``sources`` stays written, but the run itself is never marked complete, so
nothing downstream can point at it as a finished fit. A stage invocation
with no usable reports of either quantity gets no ``model_runs`` row at all:
an empty fit would sample nothing but its own priors and report clean
convergence on no evidence.

## Owning the connection

``pipeline/orchestrator.py``'s ``run_stage`` calls ``runner_module.run(spec)``
with no ``context`` at all, so ``context is None`` is not a hypothetical path
-- it is what a real pipeline run does. When the stage is not handed an open
connection it opens its own via :func:`pipeline.db.get_connection`, which
commits on a clean exit and rolls back on an exception (see ``crawl.py`` and
``resolve.py`` for the same pattern).

That matters for step 1 above. On a connection this stage owns, the started
row is committed immediately after the insert, *before* either quantity is
sampled, so a later exception's rollback (scoped to whatever has happened
since that commit) cannot also erase the row that a reader needs to see the
run attempted and failed. On a connection the caller supplied (a test's
transaction fixture, or a future orchestrator that manages its own commit
boundary), this stage never calls ``commit()`` or ``rollback()`` at all --
the caller owns that decision, exactly as ``resolve.py`` and ``crawl.py``
already do for a connection they did not open.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, cast

import structlog

from pipeline.db import get_connection
from pipeline.reconcilers import store
from pipeline.reconcilers.design import (
    DEFAULT_MIN_BATTLES_PER_SOURCE_EFFECT,
    Design,
    build_design,
)
from pipeline.reconcilers.load import (
    DEFAULT_MIN_PLAUSIBLE_TROOPS,
    classify_sides,
    load_reports,
    write_unfillable,
)
from pipeline.reconcilers.model import (
    _UNLABELLED_WEIGHT,
    DEFAULT_PRIORS_PATH,
    InflationPrior,
    ModelPriors,
    SamplerSettings,
    build_model,
    load_inflation_prior,
    sample,
)
from pipeline.reconcilers.neutral import summarise_neutral, write_neutral_draws
from pipeline.reconcilers.records import Quantity, ReconcileCounts, Report
from pipeline.reconcilers.report import build_report, write_report
from pipeline.reconcilers.summarise import summarise_fit
from pipeline.stages.base import StageContext

__all__ = ["run"]

logger = structlog.get_logger()

# Read by the orchestrator: this stage commits its own run row before sampling
# so a failed fit stays inspectable, which needs a connection it controls.
OWNS_DB_CONNECTION = True

PROCESSED_ROOT: Final[Path] = Path("data/processed")

# Fit order only; both quantities share one model_runs row (see the module
# docstring), so there is no "which one is latest" question any more.
_QUANTITIES: Final[tuple[str, ...]] = ("troops", "casualties")


def _combined_worst(per_quantity: dict[str, dict[str, Any]]) -> dict[str, float | int | None]:
    """Roll up the worst rhat, worst ESS and total divergences across quantities.

    Each quantity's own diagnostics dict (from
    :func:`~pipeline.reconcilers.summarise.summarise_fit`) already carries a
    ``"worst"`` object of this shape for its own posterior; this combines
    those across however many quantities were actually fit this run, which is
    the object ``pipeline.quality``'s ``diagnostics_json`` handler reads back
    for ``model_convergence``.

    Args:
        per_quantity: Each fitted quantity's diagnostics payload, keyed by
            quantity. Only quantities that were actually fit appear here.

    Returns:
        ``{"max_rhat": ..., "min_ess_bulk": ..., "divergences": ...}``, with
        ``None`` for a metric no quantity's payload carried a finite value
        for at all (e.g. a single-chain fit, where rhat is undefined).
    """
    max_rhats: list[float] = []
    min_esses: list[float] = []
    divergence_counts: list[int] = []
    for diagnostics in per_quantity.values():
        worst = diagnostics.get("worst") or {}
        if worst.get("max_rhat") is not None:
            max_rhats.append(float(worst["max_rhat"]))
        if worst.get("min_ess_bulk") is not None:
            min_esses.append(float(worst["min_ess_bulk"]))
        if worst.get("divergences") is not None:
            divergence_counts.append(int(worst["divergences"]))

    return {
        "max_rhat": max(max_rhats) if max_rhats else None,
        "min_ess_bulk": min(min_esses) if min_esses else None,
        "divergences": sum(divergence_counts) if divergence_counts else None,
    }


def _write_thresholds(params: dict[str, Any]) -> dict[str, float]:
    """Read the limits a fit must clear before its estimates are written.

    Args:
        params: The spec's ``params`` mapping.

    Returns:
        ``max_rhat``, ``min_ess_bulk`` and ``max_divergences``. Recorded in the
        run's config so ``neutral_fit_converged`` judges by what the run used.
    """
    return {
        "max_rhat": float(params.get("write_max_rhat", 1.05)),
        "min_ess_bulk": float(params.get("write_min_ess_bulk", 400)),
        "max_divergences": float(params.get("write_max_divergences", 0)),
    }


def _passes_write_thresholds(worst: dict[str, Any], params: dict[str, Any]) -> bool:
    """Whether a fit's worst diagnostics clear the thresholds for writing.

    Args:
        worst: A diagnostics payload's ``"worst"`` object.
        params: The spec's ``params`` mapping, for the ``write_*`` thresholds.

    Returns:
        True when rhat, bulk ESS and divergences are all acceptable.
    """
    return not _failed_thresholds(worst, params)


def _failed_thresholds(worst: dict[str, Any], params: dict[str, Any]) -> list[str]:
    """Name each write threshold a fit's worst diagnostics breach.

    Args:
        worst: A diagnostics payload's ``"worst"`` object.
        params: The spec's ``params`` mapping, for the ``write_*`` thresholds.

    Returns:
        Short descriptions such as ``"max_rhat 1.2 > 1.05"``; empty when the fit
        clears every threshold.
    """
    limits = _write_thresholds(params)
    max_rhat, min_ess, max_div = (
        limits["max_rhat"],
        limits["min_ess_bulk"],
        int(limits["max_divergences"]),
    )
    rhat = worst.get("max_rhat") or 0.0
    ess = worst.get("min_ess_bulk") or 0.0
    divergences = worst.get("divergences") or 0
    failed: list[str] = []
    # NaN compares False against every threshold, so a diverged fit with a NaN
    # diagnostic would otherwise sail through the gate.
    if not math.isfinite(rhat) or rhat > max_rhat:
        failed.append(f"max_rhat {rhat:.3g} > {max_rhat:g}")
    if not math.isfinite(ess) or ess < min_ess:
        failed.append(f"min_ess_bulk {ess:.0f} < {min_ess:g}")
    if divergences > max_div:
        failed.append(f"divergences {divergences} > {max_div}")
    return failed


def _fit_neutral_troops(
    conn: Any,
    *,
    design: Design,
    priors: ModelPriors,
    reports: list[Report],
    params: dict[str, Any],
    ci_mass: float,
    inflation: InflationPrior,
    run_id: int,
    processed_root: Path,
) -> dict[str, Any]:
    """Fit the regime-neutral troops model and write it if it converged.

    Args:
        conn: An open database connection.
        design: The design the published troops fit used.
        priors: The troops hierarchy scales.
        reports: Every loaded report, for the side-to-battle mapping.
        params: The spec's ``params`` mapping.
        ci_mass: The credible interval's total mass.
        inflation: The fitted claim-regime inflation prior.
        run_id: The shared ``model_runs`` id.
        processed_root: Where the draws file goes.

    Returns:
        The ``troops_neutral`` diagnostics block, including whether the fit was
        written.

    Raises:
        Exception: Whatever sampling raises, re-raised after logging.
    """
    model = build_model(design, priors=priors, inflation=inflation, neutral=True)
    try:
        idata = sample(model, SamplerSettings.from_params(params))
    except Exception:
        logger.error("reconcile_neutral_sampling_failed", run_id=run_id)
        raise

    battle_of_side = {r.side_id: r.battle_id for r in reports if r.quantity == "troops"}
    summary = summarise_neutral(idata, design, battle_of_side, run_id=run_id, ci_mass=ci_mass)
    worst = summary.diagnostics["worst"]
    written = _passes_write_thresholds(worst, params)
    summary.diagnostics["written"] = written
    if not written:
        logger.error(
            "reconcile_neutral_fit_not_written",
            run_id=run_id,
            worst=worst,
            hint="no neutral estimates or draws file; see neutral_fit_converged",
        )
        return summary.diagnostics

    store.write_neutral_estimates(conn, summary.estimates)
    store.clear_unmodelled_neutral(conn, side_ids=set(design.side_ids))
    write_neutral_draws(summary, processed_root=processed_root, run_id=run_id)
    return summary.diagnostics


def _fit_quantity(
    conn: Any,
    *,
    quantity: str,
    reports: list[Report],
    params: dict[str, Any],
    ancient_cutoff_year: int,
    ci_mass: float,
    inflation: InflationPrior,
    run_id: int,
    computed_at: datetime,
    counts: ReconcileCounts,
    processed_root: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Fit one quantity's model and write its estimates, under a shared run id.

    Troops also gets the regime-neutral second fit, on the same design.

    Args:
        conn: An open database connection.
        quantity: ``"troops"`` or ``"casualties"``.
        reports: Every usable report from :func:`~pipeline.reconcilers.load.load_reports`,
            both quantities together -- :func:`~pipeline.reconcilers.design.build_design`
            and :func:`~pipeline.reconcilers.summarise.summarise_fit` each filter
            to their own quantity internally.
        params: The spec's ``params`` mapping.
        ancient_cutoff_year: The era fallback's cutoff year.
        ci_mass: The credible interval's total mass.
        inflation: The fitted claim-regime inflation prior, shared across
            both quantities.
        run_id: The ``model_runs`` row both quantities this invocation fits
            are recorded under.
        computed_at: The timestamp to stamp every record with.
        counts: Counters to accumulate ``sides_estimated`` and
            ``sources_updated`` into.
        processed_root: Where the neutral draws file is written.

    Returns:
        This quantity's diagnostics payload from
        :func:`~pipeline.reconcilers.summarise.summarise_fit`, and the neutral
        fit's diagnostics block (None for casualties).

    Raises:
        Exception: Whatever :func:`~pipeline.reconcilers.model.sample` raises.
            Re-raised after logging; the caller is responsible for leaving
            the shared ``model_runs`` row incomplete rather than completing
            it on this quantity's behalf.
    """
    priors = ModelPriors.from_params(params, quantity=quantity)
    design = build_design(
        reports,
        quantity=quantity,
        ancient_cutoff_year=ancient_cutoff_year,
        counts=counts,
        min_battles_per_source_effect=int(
            params.get("min_battles_per_source_effect", DEFAULT_MIN_BATTLES_PER_SOURCE_EFFECT)
        ),
    )
    model = build_model(design, priors=priors, inflation=inflation)

    try:
        idata = sample(model, SamplerSettings.from_params(params))
    except Exception:
        logger.error(
            "reconcile_sampling_failed",
            quantity=quantity,
            run_id=run_id,
            hint="model_runs row left incomplete; model_run_recorded will fail",
        )
        raise

    summary = summarise_fit(
        idata,
        design,
        reports,
        run_id=run_id,
        ci_mass=ci_mass,
        inflation=inflation,
        priors=priors,
        computed_at=computed_at,
    )

    # Write only a converged fit. The model_convergence gate runs after the
    # stage, so writing first let an unconverged run overwrite the last good
    # estimates before anything could object. The run row and its diagnostics
    # are still recorded, so the gate fails loudly; the old estimates stand.
    worst = summary.diagnostics.get("worst", {})
    if _passes_write_thresholds(worst, params):
        counts.sides_estimated += store.write_side_estimates(conn, summary.side_estimates)
        counts.sources_updated += store.write_source_biases(conn, summary.source_biases)
        store.clear_unmodelled(
            conn,
            cast(Quantity, quantity),
            side_ids=set(design.side_ids),
            source_ids={r.source_id for r in reports if r.quantity == quantity},
        )
    else:
        logger.error(
            "reconcile_fit_not_written",
            quantity=quantity,
            run_id=run_id,
            worst=worst,
            hint="estimates left at their previous values; see model_convergence",
        )
        store.log_withheld_sides(
            conn,
            cast(Quantity, quantity),
            side_ids=set(design.side_ids),
            run_id=run_id,
            reason="; ".join(_failed_thresholds(worst, params)),
        )

    # Judged and written on its own diagnostics, whatever happened to the
    # published fit above.
    neutral_diagnostics = None
    if quantity == "troops":
        neutral_diagnostics = _fit_neutral_troops(
            conn,
            design=design,
            priors=priors,
            reports=reports,
            params=params,
            ci_mass=ci_mass,
            inflation=inflation,
            run_id=run_id,
            processed_root=processed_root,
        )
    return summary.diagnostics, neutral_diagnostics


def _run_body(
    conn: Any,
    *,
    params: dict[str, Any],
    processed_root: Path,
    ancient_cutoff_year: int,
    ci_mass: float,
    model_type: str,
    inflation_path: Path,
    owns_connection: bool,
) -> None:
    """Run the whole stage against an open connection.

    Args:
        conn: An open database connection.
        params: The spec's ``params`` mapping.
        processed_root: Where to write ``reconciliation_report.json``.
        ancient_cutoff_year: The era fallback's cutoff year.
        ci_mass: The credible interval's total mass.
        model_type: ``model_runs.model_type`` for every run this stage
            writes.
        inflation_path: Where to read the fitted claim-regime inflation
            prior from.
        owns_connection: Whether this stage opened ``conn`` itself. See the
            module docstring.
    """
    computed_at = datetime.now(UTC)
    counts = ReconcileCounts()

    min_plausible_troops = float(params.get("min_plausible_troops", DEFAULT_MIN_PLAUSIBLE_TROOPS))
    reports = load_reports(
        conn,
        ancient_cutoff_year=ancient_cutoff_year,
        counts=counts,
        min_plausible_troops=min_plausible_troops,
    )

    # Unfillable sides are a fact about the loaded reports, not about
    # whether a model ever gets fit from them, so this runs regardless of
    # what happens to either quantity's fit below.
    _modellable, unfillable = classify_sides(
        reports, conn, min_plausible_troops=min_plausible_troops
    )
    write_unfillable(conn, unfillable, counts)
    if owns_connection:
        conn.commit()

    inflation = load_inflation_prior(inflation_path)

    quantities_with_reports = [q for q in _QUANTITIES if any(r.quantity == q for r in reports)]

    run_ids: dict[str, int | None] = {quantity: None for quantity in _QUANTITIES}
    diagnostics_by_quantity: dict[str, dict[str, Any] | None] = {
        quantity: None for quantity in _QUANTITIES
    }

    if not quantities_with_reports:
        logger.warning(
            "reconcile_no_usable_reports",
            hint="no usable troops or casualties reports at all; writing nothing",
        )
    else:
        sampler_settings = SamplerSettings.from_params(params)
        config_payload: dict[str, Any] = {
            "quantities": quantities_with_reports,
            "sampler": asdict(sampler_settings),
            "inflation": asdict(inflation),
            "ancient_cutoff_year": ancient_cutoff_year,
            "ci_mass": ci_mass,
            "write_thresholds": _write_thresholds(params),
            "priors": {
                quantity: asdict(ModelPriors.from_params(params, quantity=quantity))
                for quantity in quantities_with_reports
            },
        }
        run_name = (
            f"reconcile_{'_'.join(quantities_with_reports)}"
            f"_{sampler_settings.random_seed}_{computed_at:%Y%m%dT%H%M%SZ}"
        )
        run_id = store.start_model_run(
            conn, model_type=model_type, run_name=run_name, config=config_payload
        )
        if owns_connection:
            # Durable before either quantity is sampled. See the module
            # docstring: a sampling failure below re-raises, and this row
            # must survive that as an inspectable incomplete run rather than
            # vanish with whatever get_connection() rolls back on the way
            # out.
            conn.commit()

        per_quantity_diagnostics: dict[str, dict[str, Any]] = {}
        neutral_blocks: dict[str, dict[str, Any]] = {}
        for quantity in quantities_with_reports:
            per_quantity_diagnostics[quantity], neutral_block = _fit_quantity(
                conn,
                quantity=quantity,
                reports=reports,
                params=params,
                ancient_cutoff_year=ancient_cutoff_year,
                ci_mass=ci_mass,
                inflation=inflation,
                run_id=run_id,
                computed_at=computed_at,
                counts=counts,
                processed_root=processed_root,
            )
            if neutral_block is not None:
                neutral_blocks["troops_neutral"] = neutral_block
            # Both quantities point at the one shared run id: see the module
            # docstring for why splitting this across two model_runs rows
            # would leave model_convergence and estimates_are_not_stale
            # unable to see both quantities at once.
            run_ids[quantity] = run_id
            diagnostics_by_quantity[quantity] = per_quantity_diagnostics[quantity]

        diagnostics = {
            "worst": _combined_worst(per_quantity_diagnostics),
            **per_quantity_diagnostics,
            **neutral_blocks,
        }
        store.complete_model_run(conn, run_id, diagnostics)
        if owns_connection:
            conn.commit()

    report = build_report(
        counts=counts,
        reports=reports,
        diagnostics_by_quantity=diagnostics_by_quantity,
        run_ids=run_ids,
        unlabelled_weight=_UNLABELLED_WEIGHT,
        computed_at=computed_at,
    )
    write_report(report, processed_root=processed_root)

    logger.info("reconcile_complete", run_ids=run_ids, **counts.as_dict())


def run(spec: dict[str, Any], context: StageContext | None = None) -> None:
    """Run the reconcile stage.

    Args:
        spec: The loaded ``agents/reconcile.yaml``, with overrides applied.
        context: Shared services. Without a database connection the stage
            still opens its own (see the module docstring) -- there is no
            path through this stage that does not touch the database, because
            every one of its outputs is a database write.
    """
    ctx = context or StageContext()
    params: dict[str, Any] = spec.get("params") or {}

    processed_root = Path(str(params.get("processed_root", PROCESSED_ROOT)))
    ancient_cutoff_year = int(params.get("ancient_cutoff_year", 500))
    ci_mass = float(params.get("ci_mass", 0.95))
    model_type = str(params.get("model_type", "source_disagreement"))
    inflation_path = Path(str(params.get("inflation_priors", DEFAULT_PRIORS_PATH)))

    if ctx.db_conn is not None:
        _run_body(
            ctx.db_conn,
            params=params,
            processed_root=processed_root,
            ancient_cutoff_year=ancient_cutoff_year,
            ci_mass=ci_mass,
            model_type=model_type,
            inflation_path=inflation_path,
            owns_connection=False,
        )
        return

    with get_connection() as conn:
        _run_body(
            conn,
            params=params,
            processed_root=processed_root,
            ancient_cutoff_year=ancient_cutoff_year,
            ci_mass=ci_mass,
            model_type=model_type,
            inflation_path=inflation_path,
            owns_connection=True,
        )
