"""
The impute stage: multiple imputation of missing log troop totals.

One invocation imputes one *scale* (``params.scale``): ``neutral`` (primary,
what the model stage reads) or ``published`` (sensitivity). Run it once per
scale; outputs go to ``data/imputed/{scale}/round_{n}/``.

## What it does

1. Builds a likelihood for each battle's observed sides from reconcile: on the
   neutral scale from the posterior draws in
   ``data/processed/reconcile/neutral_draws_run{run_id}.npz`` with
   reconcile's own side prior divided out; on the published scale from
   ``est_troops_total`` and its interval, diagonally.
2. Fits the marginalised joint model (``pipeline/imputers/model.py``).
3. Writes only if the fit converged (the same rule reconcile uses): one
   posterior draw per round, every side redrawn from the conditional Gaussian
   (overimputation), files per round, and the gate report. Only the neutral
   scale also writes ``missing_data_log`` rows; published is a sensitivity run
   and never touches that table. A fit that does not converge writes no rounds and no log rows, but
   its ``model_runs`` row and diagnostics are still recorded so
   ``model_convergence`` fails loudly.

## Connection ownership

Same pattern as ``pipeline/stages/reconcile.py``: handed a connection the stage
never commits or rolls back; opening its own, it commits the started
``model_runs`` row before sampling so a failed fit leaves an inspectable
incomplete run.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from pipeline.db import get_connection
from pipeline.imputers.conditional import RoundSet, draw_rounds, extract_posterior
from pipeline.imputers.design import ImputeDesign, build_design
from pipeline.imputers.gates import GateResult, run_gates
from pipeline.imputers.likelihood import (
    find_latest_draws,
    likelihoods_from_draws,
    likelihoods_from_published,
    load_neutral_draws,
)
from pipeline.imputers.load import (
    latest_written_neutral_run_id,
    load_side_rows,
    published_run_id,
    subset_sides,
)
from pipeline.imputers.model import build_model
from pipeline.imputers.outputs import config_hash, write_gate_report, write_rounds
from pipeline.imputers.priors import ImputeParams, ImputePriors, load_priors
from pipeline.imputers.records import BattleLikelihood, ImputeCounts, SideRow
from pipeline.imputers.store import write_imputed_log
from pipeline.imputers.summarise import (
    build_diagnostics,
    convergence,
    is_converged,
    summarise_imputed,
)
from pipeline.reconcilers import store as run_store
from pipeline.reconcilers.model import SamplerSettings, sample
from pipeline.stages.base import StageContext

__all__ = ["run"]

logger = structlog.get_logger()

# Read by the orchestrator: this stage commits its own run row before sampling
# so a failed fit stays inspectable, which needs a connection it controls.
OWNS_DB_CONNECTION = True


def _build_likelihoods(
    conn: Any, ip: ImputeParams, sides: list[SideRow], counts: ImputeCounts
) -> tuple[list[BattleLikelihood], int | None]:
    """Build the per-battle likelihoods for the configured scale.

    Args:
        conn: An open database connection.
        ip: Run parameters.
        sides: Every in-scope side.
        counts: Counters to update.

    Returns:
        The likelihoods and the reconcile run id they came from.

    Raises:
        FileNotFoundError: On the neutral scale, if reconcile has written no
            draws file. There is no fallback to published estimates: that
            would silently change what the output means.
        ValueError: On the neutral scale, if the draws file's run id is not
            the latest ``source_disagreement`` run whose ``troops_neutral``
            diagnostics say it was written.
    """
    if ip.scale == "published":
        return (
            likelihoods_from_published(sides, default_sd=ip.published_default_sd),
            published_run_id(conn),
        )

    path = Path(ip.draws_path) if ip.draws_path else find_latest_draws(Path(ip.reconcile_root))
    if path is None or not path.is_file():
        raise FileNotFoundError(
            f"no neutral_draws_run*.npz under {ip.reconcile_root}: run reconcile first "
            "(its neutral fit must converge to write one)"
        )
    neutral = load_neutral_draws(path)
    expected = latest_written_neutral_run_id(conn)
    if expected is None or neutral.run_id != expected:
        raise ValueError(
            f"{path.name} comes from reconcile run {neutral.run_id}, but the latest reconcile "
            f"run that wrote a neutral fit is {expected}: the draws file is stale or from a "
            "different database. Re-run reconcile, or point reconcile_root/draws_path at "
            "the matching file."
        )
    logger.info("impute_neutral_draws_loaded", path=str(path), run_id=neutral.run_id)
    liks, stats = likelihoods_from_draws(
        neutral, {s.side_id for s in sides}, min_precision=ip.min_lik_precision
    )
    counts.sides_stale_in_draws = stats.stale_sides
    counts.sides_weak_as_missing = stats.weak_sides
    counts.battles_diagonal_fallback = stats.diagonal_battles
    return liks, neutral.run_id


def _fit_and_write(
    conn: Any,
    *,
    ip: ImputeParams,
    design: ImputeDesign,
    run_id: int,
    reconcile_run_id: int | None,
    config_digest: str,
    counts: ImputeCounts,
) -> dict[str, Any]:
    """Sample the model, then write the rounds if the fit converged.

    Args:
        conn: An open database connection.
        ip: Run parameters.
        design: The model design.
        run_id: This invocation's ``model_runs`` id.
        reconcile_run_id: The reconcile run behind the likelihood.
        config_digest: Hash of the run configuration, for the manifests.
        counts: Counters to update.

    Returns:
        The ``model_runs.diagnostics`` payload.

    Raises:
        Exception: Whatever sampling raises, after logging; the caller leaves
            the ``model_runs`` row incomplete.
    """
    root = Path(ip.imputed_root)
    model = build_model(design)
    settings = SamplerSettings(
        draws=ip.draws,
        tune=ip.tune,
        chains=ip.chains,
        cores=ip.cores,
        target_accept=ip.target_accept,
        random_seed=ip.random_seed,
        nuts_sampler=ip.nuts_sampler,
    )
    try:
        idata = sample(model, settings)
    except Exception:
        logger.error("impute_sampling_failed", run_id=run_id, scale=ip.scale)
        raise

    worst = convergence(idata)
    converged = is_converged(
        worst,
        max_rhat=ip.write_max_rhat,
        min_ess_bulk=ip.write_min_ess_bulk,
        max_divergences=ip.write_max_divergences,
    )
    diagnostics = build_diagnostics(
        worst,
        idata,
        scale=ip.scale,
        design=design,
        n_rounds=ip.n_imputations,
        reconcile_run_id=reconcile_run_id,
        written=converged,
    )
    if not converged:
        logger.error(
            "impute_fit_not_written",
            run_id=run_id,
            worst=worst,
            hint="no rounds or log rows written; see model_convergence",
        )
        write_gate_report(root, scale=ip.scale, run_id=run_id, written=False, gates=[])
        return diagnostics

    rounds = draw_rounds(
        design,
        extract_posterior(idata, design),
        n_rounds=ip.n_imputations,
        seed=ip.random_seed,
        mnar_delta=ip.mnar_delta,
    )
    gates = _gates(ip, design, rounds)
    write_rounds(
        root,
        scale=ip.scale,
        rounds=rounds,
        design=design,
        run_id=run_id,
        seed=ip.random_seed,
        reconcile_run_id=reconcile_run_id,
        config_digest=config_digest,
    )
    if ip.scale == "neutral":
        write_imputed_log(
            conn,
            summarise_imputed(rounds, design),
            scale=ip.scale,
            n_rounds=ip.n_imputations,
            counts=counts,
        )
    write_gate_report(root, scale=ip.scale, run_id=run_id, written=True, gates=gates)
    counts.sides_imputed = int(rounds.is_imputed.sum())
    return diagnostics


def _gates(ip: ImputeParams, design: ImputeDesign, rounds: RoundSet) -> list[GateResult]:
    """Compute the dataset-level gates for a set of rounds."""
    return run_gates(
        rounds.values,
        is_imputed=rounds.is_imputed,
        battle_ids=design.battle_ids,
        era_of_side=np.array(design.eras)[design.era_index],
        scale=ip.scale,
        n_rounds=ip.n_imputations,
        ratio_limit=ip.ratio_limit,
        value_min=ip.value_min,
        value_max=ip.value_max,
        ks_alpha=ip.ks_alpha,
        ks_min_n=ip.ks_min_n,
    )


def _run_body(
    conn: Any,
    *,
    params: dict[str, Any],
    ctx: StageContext,
    owns_connection: bool,
) -> None:
    """Run the whole stage against an open connection.

    Args:
        conn: An open database connection.
        params: The spec's ``params`` mapping.
        ctx: Stage context (``limit`` and ``dry_run`` are honoured).
        owns_connection: Whether this stage opened ``conn`` itself.
    """
    ip = ImputeParams.from_params(params)
    priors: ImputePriors = load_priors(ip.priors_path)
    counts = ImputeCounts()

    sides = subset_sides(load_side_rows(conn), ctx.limit)
    counts.sides_in_scope = len(sides)
    if not sides:
        logger.warning("impute_no_sides", hint="no battle_sides rows; writing nothing")
        return

    likelihoods, reconcile_run_id = _build_likelihoods(conn, ip, sides, counts)
    design = build_design(sides, likelihoods, priors)
    counts.sides_observed = design.n_observed
    if design.n_observed == 0:
        logger.warning("impute_no_observed_sides", hint="nothing to condition on; writing nothing")
        return
    if ctx.dry_run:
        logger.info("impute_dry_run", scale=ip.scale, **counts.as_dict())
        return

    config_payload: dict[str, Any] = {
        "scale": ip.scale,
        "params": asdict(ip),
        "priors": {
            "era_bins": priors.era_bins,
            "era_priors": {k: asdict(v) for k, v in priors.era_priors.items()},
            "hyper": priors.hyper,
        },
        "reconcile_run_id": reconcile_run_id,
    }
    started = datetime.now(UTC)
    run_id = run_store.start_model_run(
        conn,
        model_type=ip.model_type,
        run_name=f"impute_{ip.scale}_{ip.random_seed}_{started:%Y%m%dT%H%M%SZ}",
        config=config_payload,
    )
    if owns_connection:
        # Durable before sampling, so a failed fit leaves an inspectable
        # incomplete run (same reason as reconcile).
        conn.commit()

    diagnostics = _fit_and_write(
        conn,
        ip=ip,
        design=design,
        run_id=run_id,
        reconcile_run_id=reconcile_run_id,
        config_digest=config_hash(config_payload),
        counts=counts,
    )
    run_store.complete_model_run(conn, run_id, diagnostics)
    if owns_connection:
        conn.commit()
    logger.info("impute_complete", run_id=run_id, scale=ip.scale, **counts.as_dict())


def run(spec: dict[str, Any], context: StageContext | None = None) -> None:
    """Run the impute stage.

    Args:
        spec: The loaded ``agents/impute.yaml``, with overrides applied.
        context: Shared services. Without a connection the stage opens its own:
            every output is derived from the database.
    """
    ctx = context or StageContext()
    params: dict[str, Any] = spec.get("params") or {}

    if ctx.db_conn is not None:
        _run_body(ctx.db_conn, params=params, ctx=ctx, owns_connection=False)
        return

    with get_connection() as conn:
        _run_body(conn, params=params, ctx=ctx, owns_connection=True)
