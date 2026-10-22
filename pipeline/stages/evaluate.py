"""
The evaluate stage: validate the latest converged primary model run.

Writes files only. No ``model_runs`` or ``general_skill_estimates`` rows, so
``MAX(run_id)`` over ``model_runs`` still names the published run. The
statistical building blocks live in ``pipeline/evaluators/``; this module is
the stage around them. ``agents/evaluate.yaml`` is the contract.

## Flow

1. Find the target run and rebuild its design exactly as the model stage did,
   refusing if the database has moved on since.
2. Cross-validate by battle, predicting held-out battles without the imputed
   force ratio (impute conditions on the outcome).
3. Refit a baseline and every sensitivity variant on shared rounds and seeds.
4. Compare the ranking with the named checks and external lists, run the
   posterior predictive checks, draw figures.
5. Evaluate the gates and write the report.

No connection is committed or rolled back: every statement here is a SELECT.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from pipeline.db import get_connection
from pipeline.evaluators.config import EvaluateParams, load_eval_model_params
from pipeline.evaluators.cv import run_cross_validation
from pipeline.evaluators.experts import (
    ExpertConfig,
    compare_reference,
    load_expert_config,
    named_checks_gate,
    run_named_checks,
)
from pipeline.evaluators.plots import write_figures
from pipeline.evaluators.ppc import posterior_predictive_report
from pipeline.evaluators.report import build_gates, build_report, gate_severities, write_outputs
from pipeline.evaluators.sensitivity import (
    STATUS_OK,
    RankTable,
    SensitivityInputs,
    SensitivityResult,
    run_sensitivity,
)
from pipeline.evaluators.target import (
    PublishedRun,
    load_target_run,
    measured_log_troops,
    round_x_diffs,
)
from pipeline.evaluators.variants import NotEvaluableError, parse_variants, read_imputed_side_ids
from pipeline.imputers.likelihood import NeutralDraws, load_neutral_draws
from pipeline.imputers.load import latest_written_neutral_run_id
from pipeline.modelers.config import ModelParams
from pipeline.modelers.design import build_design
from pipeline.modelers.load import (
    ImputedRound,
    check_round_provenance,
    fill_camp_log_troops,
    has_multi_side_battle,
    load_battle_quality,
    load_battle_types,
    load_side_inputs,
    read_rounds,
    subset_sides,
)
from pipeline.modelers.outputs import run_directory
from pipeline.modelers.records import ModelDesign, SideInput
from pipeline.reconcilers.neutral import neutral_draws_path
from pipeline.stages.base import StageContext

__all__ = ["run"]

logger = structlog.get_logger()

# Offset of the posterior predictive simulation seed from the base seed.
PPC_SEED_OFFSET = 9_000
# Cutoff for the share-in-top-N metric of unordered external lists.
REFERENCE_TOP_N = 50


@dataclass(frozen=True)
class _Prepared:
    """The target run and the inputs rebuilt from the database and disk."""

    published: PublishedRun
    sides: list[SideInput]
    design: ModelDesign
    rounds: list[ImputedRound]
    x_diffs: list[np.ndarray]
    measured: tuple[np.ndarray, np.ndarray]


@contextmanager
def _timed(timings: dict[str, float], section: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[section] = time.perf_counter() - start


def _prepare(conn: Any, mp: ModelParams, ctx: StageContext) -> _Prepared | None:
    """Load the target run and rebuild the design; None if evaluation must not proceed."""
    published = load_target_run(conn)
    if published is None:
        logger.error(
            "evaluate_no_target_run",
            hint="no converged bradley_terry_hierarchical run with estimates written; "
            "run the model stage first",
        )
        return None
    sides = subset_sides(load_side_inputs(conn), ctx.limit)
    # The model stage fills a camp member's missing troops from the imputed
    # rounds before building its design; do the same so the designs match.
    rounds: list[ImputedRound] | None = None
    if has_multi_side_battle(sides):
        rounds = read_rounds(Path(mp.imputed_root), mp.scale, mp.max_rounds)
        sides = fill_camp_log_troops(sides, rounds)
    design = build_design(sides, mp.design)
    if set(design.general_ids.tolist()) != set(published.general_ids.tolist()):
        logger.error(
            "evaluate_design_mismatch",
            model_run_id=published.run_id,
            design_generals=design.n_generals,
            published_generals=int(published.general_ids.size),
            hint="the database changed since the model ran (or --limit differs); "
            "re-run the model stage",
        )
        return None
    if rounds is None:
        rounds = read_rounds(Path(mp.imputed_root), mp.scale, mp.max_rounds)
    check_round_provenance(rounds, published.imputation_run_id)
    return _Prepared(
        published,
        sides,
        design,
        rounds,
        round_x_diffs(design, rounds),
        measured_log_troops(conn, design),
    )


def _load_neutral(conn: Any, reconcile_root: Path) -> NeutralDraws:
    """Reconcile's newest written neutral draws, for the g_unlabelled variants.

    Raises:
        NotEvaluableError: If no run wrote a neutral fit, or its file is missing
            or unreadable. The variants report that instead of failing.
    """
    run_id = latest_written_neutral_run_id(conn)
    if run_id is None:
        raise NotEvaluableError("no reconcile run has written a neutral fit in this database")
    path = reconcile_root / neutral_draws_path(Path(), run_id).name
    if not path.is_file():
        raise NotEvaluableError(f"neutral draws file {path} not found")
    try:
        return load_neutral_draws(path, require_likelihood=False)
    except (OSError, ValueError, KeyError) as exc:
        raise NotEvaluableError(f"cannot read {path}: {type(exc).__name__}: {exc}") from exc


def _expert_section(
    params: EvaluateParams, published: PublishedRun
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Named checks, reference comparisons and the named-checks gate."""
    try:
        config: ExpertConfig = load_expert_config(Path(params.expert_rankings))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # A broken targets file fails its gate; it must not cost the rest of the report.
        message = f"cannot load {params.expert_rankings}: {type(exc).__name__}: {exc}"
        logger.error("evaluate_expert_config_failed", error=message)
        gate = {
            "name": "named_checks",
            "passed": False,
            "actual": None,
            "threshold": "all evaluable checks pass and none not_evaluable",
            "message": message,
        }
        return [], [], gate
    rank_by_qid = published.rank_by_qid()
    n_ranked = int(np.isfinite(published.rank).sum())
    named = run_named_checks(config, rank_by_qid, n_ranked)
    references = compare_reference(config, rank_by_qid, n_ranked, top_n=REFERENCE_TOP_N)
    return named, references, named_checks_gate(named)


def _ppc_section(
    conn: Any,
    prep: _Prepared,
    sens: SensitivityResult,
    mp: ModelParams,
    params: EvaluateParams,
) -> dict[str, Any]:
    """Posterior predictive checks on the baseline refit's middle round."""
    baseline = sens.baseline
    if baseline is None or sens.baseline_table is None:
        return {"skipped": "baseline refit unavailable"}
    rnd = baseline.rounds[len(baseline.rounds) // 2]
    if rnd.draws is None:
        return {"skipped": "representative round has no draws"}
    try:
        return posterior_predictive_report(
            prep.design,
            rnd.draws,
            rnd.x_diff,
            battle_types=load_battle_types(conn),
            score=sens.baseline_table.score,
            names=dict(zip(prep.published.general_ids.tolist(), prep.published.names, strict=True)),
            top_n=params.top_n,
            n_sim=mp.gates.ppc_sims,
            level=mp.gates.ppc_level,
            seed=params.seed + PPC_SEED_OFFSET,
        )
    except Exception as exc:
        # Report-only section: a failure is recorded, not fatal.
        message = f"{type(exc).__name__}: {exc}"
        logger.error("evaluate_ppc_failed", error=message)
        return {"error": message}


def _figure_inputs(
    sens: SensitivityResult, published: PublishedRun, top_n: int
) -> tuple[list[int], dict[str, RankTable]]:
    """Baseline top generals and the rankings the stability plot compares."""
    if sens.baseline_table is None:
        return [], {}
    base = sens.baseline_table
    top = [int(g) for g in base.ids[np.argsort(base.rank)][:top_n]]
    tables = {"baseline": base}
    tables.update(
        {v.spec.name: v.table for v in sens.variants if v.status == STATUS_OK and v.table}
    )
    tables["published"] = RankTable.of(published.general_ids, published.score, published.skill_mean)
    return top, tables


def _settings(mp: ModelParams, ctx: StageContext) -> dict[str, Any]:
    return {
        "sampler": {
            "draws": mp.sampler.draws,
            "tune": mp.sampler.tune,
            "chains": mp.sampler.chains,
            "target_accept": mp.sampler.target_accept,
        },
        "scale": mp.scale,
        "rank_by": mp.rank_by,
        "replacement_percentile": mp.replacement_percentile,
        "write_thresholds": {
            "max_rhat": mp.max_rhat,
            "min_ess_bulk": mp.min_ess_bulk,
            "max_divergences": mp.max_divergences,
        },
        "limit": ctx.limit,
    }


def _counts(prep: _Prepared) -> dict[str, Any]:
    return {
        **prep.design.counts.as_dict(),
        "n_battles": prep.design.n_battles,
        "n_generals": prep.design.n_generals,
        "n_rounds_read": len(prep.rounds),
    }


def _evaluate(
    conn: Any,
    prep: _Prepared,
    mp: ModelParams,
    params: EvaluateParams,
    ctx: StageContext,
    severities: Mapping[str, str],
) -> None:
    """Run every section and write the files."""
    timings: dict[str, float] = {}
    design, published = prep.design, prep.published
    with _timed(timings, "cv"):
        cv = run_cross_validation(design, prep.x_diffs, mp, params, prep.measured)
    inputs = SensitivityInputs(
        design,
        prep.sides,
        prep.rounds,
        prep.x_diffs,
        mp,
        params,
        prep.measured,
        lambda: load_battle_quality(conn),
        published,
        lambda: _load_neutral(conn, Path(params.reconcile_root)),
        lambda: read_imputed_side_ids(
            Path(mp.imputed_root),
            mp.scale,
            [r.index for r in prep.rounds[: params.sensitivity_rounds]],
        ),
    )
    with _timed(timings, "sensitivity"):
        sens = run_sensitivity(inputs, parse_variants(params.sensitivity_variants))
    with _timed(timings, "expert_comparison"):
        named, references, named_gate = _expert_section(params, published)
    with _timed(timings, "ppc"):
        ppc = _ppc_section(conn, prep, sens, mp, params)
    directory = run_directory(Path(params.output_root), published.run_id)
    top_ids, tables = _figure_inputs(sens, published, params.top_n)
    with _timed(timings, "figures"):
        figures = write_figures(
            directory / "figures",
            calibration=cv.metrics["binary"]["calibration_favourite"],
            top_ids=top_ids,
            names=dict(zip(published.general_ids.tolist(), published.names, strict=True)),
            rank_tables=tables,
            published=published,
        )
    gates = build_gates(cv, sens, named_gate, params, severities)
    report = build_report(
        published=published,
        params=params,
        settings=_settings(mp, ctx),
        counts=_counts(prep),
        created_at=datetime.now(UTC).isoformat(),
        cv=cv,
        sens=sens,
        named_checks=named,
        references=references,
        ppc=ppc,
        gates=gates,
        figures=[str(p) for p in figures],
        timings=timings,
    )
    write_outputs(
        directory,
        Path(params.report_path),
        report,
        cv=cv,
        battle_ids=design.battle_ids,
        k_lo=design.k_lo,
        k_hi=design.k_hi,
        sens=sens,
        published=published,
    )
    for gate in gates:
        if not gate["passed"]:
            logger.warning("evaluate_gate_failed", gate=gate["name"], message=gate["message"])
    logger.info(
        "evaluate_complete",
        model_run_id=published.run_id,
        directory=str(directory),
        gates_passed=sum(bool(g["passed"]) for g in gates),
        gates=len(gates),
    )


def _run_body(
    conn: Any,
    spec_params: dict[str, Any],
    ctx: StageContext,
    severities: Mapping[str, str] | None = None,
) -> None:
    params = EvaluateParams.from_mapping(spec_params)
    mp = load_eval_model_params(params)
    prep = _prepare(conn, mp, ctx)
    if prep is None:
        return
    logger.info(
        "evaluate_inputs_ready",
        model_run_id=prep.published.run_id,
        imputation_run_id=prep.published.imputation_run_id,
        n_battles=prep.design.n_battles,
        n_generals=prep.design.n_generals,
        n_rounds=len(prep.rounds),
    )
    if ctx.dry_run:
        logger.info("evaluate_dry_run", hint="stopping before any fit; nothing written")
        return
    _evaluate(conn, prep, mp, params, ctx, severities or {})


def run(spec: dict[str, Any], context: StageContext | None = None) -> None:
    """Run the evaluate stage.

    Args:
        spec: The loaded ``agents/evaluate.yaml``, with overrides applied.
        context: Shared services. Without a connection the stage opens its own.
    """
    ctx = context or StageContext()
    params: dict[str, Any] = spec.get("params") or {}
    severities = gate_severities(spec.get("quality_checks") or [])
    if ctx.db_conn is not None:
        _run_body(ctx.db_conn, params, ctx, severities)
        return
    with get_connection() as conn:
        _run_body(conn, params, ctx, severities)
