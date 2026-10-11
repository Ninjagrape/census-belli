"""
The model stage: a Bayesian ordinal Bradley-Terry model of commander skill,
fitted once per imputed round and pooled, with Wins Above Replacement.

The statistical core is ``pipeline/modelers/``; this module is the stage around
it: loading, the run loop, database writes, files and gates. See
``config/model_default.yaml`` for every parameter and ``agents/model.yaml`` for
the model and its departures from the original spec.

## Flow

1. Load sides and commanders, build the design (exclusion rules and
   counters), read the imputed rounds and refuse them unless they come from the
   latest neutral impute run.
2. Prior predictive check on the real design.
3. One fit per round (``bradley_terry_round`` run rows, one retry with a new
   seed and a higher ``target_accept``), WAR computed once per round and
   reduced.
4. One joint latent-troops fit (``bradley_terry_joint_latent``), the
   circularity comparator.
5. Pool (mixture draws, Rubin variance), run the gates.
6. Write rows only if every round converged; the pooled
   ``bradley_terry_hierarchical`` row is created last so it carries the
   highest ``run_id`` that report's gates read. If a round failed, nothing but
   run rows and files is written and that row says ``converged = 'false'``.

## Connection ownership

As in ``pipeline/stages/impute.py``: handed a connection the stage never
commits or rolls back; opening its own, it commits each started ``model_runs``
row before sampling so a failed fit leaves an inspectable incomplete run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from pipeline.db import get_connection
from pipeline.modelers import outputs, store
from pipeline.modelers.checks import CheckResult, PriorPredictive, prior_predictive
from pipeline.modelers.components import (
    ComponentResult,
    career_spans,
    comparison_edges,
    components,
    find_bridges,
)
from pipeline.modelers.config import ModelParams, load_model_params
from pipeline.modelers.design import build_design, x_difference
from pipeline.modelers.fitting import FitOutcome, fit_with_retry
from pipeline.modelers.gates import gate_dict, run_gates
from pipeline.modelers.load import (
    ImputedRound,
    check_round_provenance,
    latest_written_imputation_run_id,
    load_battle_types,
    load_general_names,
    load_log_troop_measurements,
    load_side_inputs,
    read_rounds,
    subset_sides,
)
from pipeline.modelers.model import (
    PosteriorDraws,
    build_joint_latent_model,
    build_model,
    extract_draws,
)
from pipeline.modelers.pool import combine_rounds
from pipeline.modelers.records import ModelDesign
from pipeline.modelers.summarise import (
    PooledFit,
    RoundReduction,
    assert_decomposition,
    pool_reductions,
    reduce_draws,
    summarise_single,
    thin_draws,
)
from pipeline.stages.base import StageContext

__all__ = ["run"]

logger = structlog.get_logger()

# Read by the orchestrator: this stage commits its own run row before sampling
# so a failed fit stays inspectable, which needs a connection it controls.
OWNS_DB_CONNECTION = True

# Offset of the joint fit's seed from the base seed; rounds use base + index.
JOINT_SEED_OFFSET = 50_000
ROUND_MODEL_TYPE = "bradley_terry_round"
JOINT_MODEL_TYPE = "bradley_terry_joint_latent"
POOLED_MODEL_TYPE = "bradley_terry_hierarchical"


@dataclass
class _RoundResult:
    """One round's run row and, when it converged, its reduction."""

    index: int
    run_id: int
    outcome: FitOutcome
    reduction: RoundReduction | None
    summary: dict[str, Any]


@dataclass
class _JointResult:
    """The joint latent fit's run row and, when it converged, its summaries."""

    run_id: int
    converged: bool
    fit: PooledFit | None
    summary: dict[str, Any]


def _fit_summary(outcome: FitOutcome, draws: PosteriorDraws | None) -> dict[str, Any]:
    """The diagnostics payload shared by round and joint run rows."""
    summary: dict[str, Any] = {
        "worst": outcome.worst,
        "converged": "true" if outcome.converged else "false",
        "attempts": outcome.attempts,
        "retried": len(outcome.attempts) > 1,
    }
    if draws is not None:
        summary["beta_force_mean"] = float(draws.beta_force.mean())
        summary["beta_force_sd"] = float(draws.beta_force.std(ddof=1))
        summary["sigma_skill_mean"] = float(draws.sigma_skill.mean())
    return summary


def _begin(conn: Any, owns_connection: bool, **kwargs: Any) -> int:
    """Insert a run row, durable before sampling when the stage owns the connection."""
    run_id = store.start_run(conn, **kwargs)
    if owns_connection:
        conn.commit()
    return run_id


def _fit_rounds(
    conn: Any,
    *,
    owns_connection: bool,
    mp: ModelParams,
    design: ModelDesign,
    rounds: list[ImputedRound],
    x_diffs: list[np.ndarray],
    imputation_run_id: int,
) -> tuple[list[_RoundResult], tuple[PosteriorDraws, np.ndarray] | None]:
    """Fit every round and reduce the ones that converged.

    Returns:
        The per-round results and the thinned draws of the representative
        round (the middle one by position; rounds are exchangeable draws, so
        any round is as representative as any other) with its ``x_diff``,
        for the posterior predictive check. None if that round failed.
    """
    results: list[_RoundResult] = []
    representative = len(rounds) // 2
    kept: tuple[PosteriorDraws, np.ndarray] | None = None
    for position, (rnd, x_diff) in enumerate(zip(rounds, x_diffs, strict=True)):
        run_id = _begin(
            conn,
            owns_connection,
            model_type=ROUND_MODEL_TYPE,
            run_name=f"bt_round_{rnd.index}_imputation{imputation_run_id}",
            config={"round": rnd.index, "imputation_run_id": imputation_run_id},
            imputation_set=rnd.index,
        )
        model = build_model(design, x_diff, mp.priors)
        outcome = fit_with_retry(
            model, mp, seed=mp.sampler.random_seed + rnd.index, label=f"round_{rnd.index}"
        )
        reduction: RoundReduction | None = None
        draws: PosteriorDraws | None = None
        if outcome.converged:
            draws = thin_draws(extract_draws(outcome.idata, design), mp.draws_kept_per_round)
            reduction = reduce_draws(design, draws, x_diff, mp.replacement_percentile)
            if position == representative:
                kept = (draws, x_diff)
        summary = {"round": rnd.index, **_fit_summary(outcome, draws)}
        store.complete_run(conn, run_id, summary)
        if owns_connection:
            conn.commit()
        logger.info(
            "model_round_fitted", round=rnd.index, run_id=run_id, converged=outcome.converged
        )
        results.append(_RoundResult(rnd.index, run_id, outcome, reduction, summary))
    return results, kept


def _joint_inputs(conn: Any, design: ModelDesign) -> tuple[np.ndarray, np.ndarray]:
    """Measured log-troop mean and sd per side index (NaN where unmeasured)."""
    measured = load_log_troop_measurements(conn)
    obs_mean = np.full(design.n_sides, np.nan)
    obs_sd = np.full(design.n_sides, np.nan)
    for i, side_id in enumerate(design.side_ids):
        if int(side_id) in measured:
            obs_mean[i], obs_sd[i] = measured[int(side_id)]
    return obs_mean, obs_sd


def _fit_joint(
    conn: Any,
    *,
    owns_connection: bool,
    mp: ModelParams,
    design: ModelDesign,
    imputation_run_id: int,
) -> _JointResult:
    """Fit the joint latent-troops comparator (R2)."""
    model = build_joint_latent_model(
        design, *_joint_inputs(conn, design), mp.joint_sigma_w, mp.priors
    )
    run_id = _begin(
        conn,
        owns_connection,
        model_type=JOINT_MODEL_TYPE,
        run_name=f"bt_joint_latent_imputation{imputation_run_id}",
        config={"imputation_run_id": imputation_run_id, "sigma_w": mp.joint_sigma_w},
    )
    outcome = fit_with_retry(
        model, mp, seed=mp.sampler.random_seed + JOINT_SEED_OFFSET, label="joint"
    )
    fit: PooledFit | None = None
    draws: PosteriorDraws | None = None
    if outcome.converged:
        draws = thin_draws(extract_draws(outcome.idata, design), mp.draws_kept_per_round)
        reduction = reduce_draws(design, draws, None, mp.replacement_percentile)
        fit = summarise_single(design, reduction, mp.rank_by)
    summary = _fit_summary(outcome, draws)
    store.complete_run(conn, run_id, summary)
    if owns_connection:
        conn.commit()
    return _JointResult(run_id, outcome.converged, fit, summary)


def _type_counts(design: ModelDesign, types: dict[int, str | None]) -> dict[str, int]:
    """Included battles of the types that share the land-battle parameters."""
    included = [types.get(int(b)) for b in design.battle_ids]
    return {
        "battles_included_naval": sum(t == "naval" for t in included),
        "battles_included_aerial": sum(t == "aerial" for t in included),
    }


def _try_fit_joint(
    conn: Any,
    *,
    owns_connection: bool,
    mp: ModelParams,
    design: ModelDesign,
    imputation_run_id: int,
) -> _JointResult | None:
    """Fit the comparator, degrading to None if it raises.

    Every round has converged by now; losing the whole ranking to an error in
    a secondary check would be the wrong trade. The failure is logged and the
    circularity gate reports that the comparison did not run.
    """
    try:
        return _fit_joint(
            conn,
            owns_connection=owns_connection,
            mp=mp,
            design=design,
            imputation_run_id=imputation_run_id,
        )
    except Exception as exc:
        logger.error(
            "model_joint_fit_failed",
            error=str(exc),
            error_type=type(exc).__name__,
            hint="continuing without the circularity comparator",
        )
        if owns_connection:
            conn.rollback()
        return None


def _estimate_rows(design: ModelDesign, fit: PooledFit) -> list[store.SkillEstimateRow]:
    """One ``general_skill_estimates`` row per credited general."""
    per_battle = fit.war_mean / np.maximum(design.n_included_battles, 1)
    return [
        store.SkillEstimateRow(
            general_id=int(design.general_ids[g]),
            skill_mean=float(fit.skill_mean[g]),
            skill_median=float(fit.skill_median[g]),
            skill_sd=float(fit.skill_sd[g]),
            skill_ci_lo=float(fit.ci95_lo[g]),
            skill_ci_hi=float(fit.ci95_hi[g]),
            skill_ci90_lo=float(fit.ci90_lo[g]),
            skill_ci90_hi=float(fit.ci90_hi[g]),
            total_war=float(fit.war_mean[g]),
            war_per_battle=float(per_battle[g]),
            war_sd=float(fit.war_sd[g]),
            n_battles=int(design.n_included_battles[g]),
            rank=int(fit.rank[g]),
        )
        for g in range(design.n_generals)
    ]


def _detail_rows(design: ModelDesign, fit: PooledFit) -> list[store.WarDetailRow]:
    """One ``battle_war_details`` row per commander entry."""
    y = np.empty(design.n_sides)
    y[design.side_a] = design.y_a
    y[design.side_b] = 1.0 - design.y_a
    return [
        store.WarDetailRow(
            battle_id=int(design.battle_ids[side // 2]),
            general_id=int(design.general_ids[general]),
            war_contribution=float(fit.entry_xwar[e]),
            win_prob_model=float(fit.e_model[side]),
            win_prob_baseline=float(fit.e_rep[side]),
            actual_outcome=float(y[side]),
            attribution_weight=float(design.cmd_weight[e]),
        )
        for e, (side, general) in enumerate(zip(design.cmd_side, design.cmd_general, strict=True))
    ]


def _components_payload(
    design: ModelDesign, comp: ComponentResult, spans: np.ndarray
) -> dict[str, Any]:
    """The component listing: sizes, bridge edges, per-general component."""
    ids = design.general_ids
    bridges = find_bridges(design.n_generals, comparison_edges(design))
    return {
        "n_components": comp.n_components,
        "sizes": comp.sizes,
        "largest_share": comp.largest_share,
        "component_of_general": {
            int(g): int(c) for g, c in zip(ids, comp.component_id, strict=True)
        },
        "bridge_edges": [[int(ids[u]), int(ids[v])] for u, v in bridges],
        "career_spans": {
            int(g): {"first": s[0], "last": s[1], "span": s[2]}
            for g, s in zip(ids, spans, strict=True)
        },
    }


def _per_general_columns(
    design: ModelDesign,
    fit: PooledFit,
    comp: ComponentResult,
    spans: np.ndarray,
    names: dict[int, str],
) -> dict[str, Any]:
    """The Rubin table per general, as CSV columns."""
    ids = [int(g) for g in design.general_ids]
    return {
        "general_id": ids,
        "name": [names.get(g, "") for g in ids],
        "component_id": comp.component_id,
        "n_battles": design.n_included_battles,
        "effective_battles": design.effective_battles,
        "n_uncredited_battles": design.n_uncredited_battles,
        "n_excluded_multi_side": design.n_excluded_multi_side,
        "first_year": spans[:, 0],
        "last_year": spans[:, 1],
        "span_years": spans[:, 2],
        "skill_mean": fit.skill_mean,
        "skill_median": fit.skill_median,
        "skill_sd_sqrt_T": fit.skill_sd,
        "skill_ci95_lo": fit.ci95_lo,
        "skill_ci95_hi": fit.ci95_hi,
        "skill_within": fit.skill_within,
        "skill_between": fit.skill_between,
        "skill_fmi": fit.skill_fmi,
        "total_xwar": fit.war_mean,
        "xwar_sd_sqrt_T": fit.war_sd,
        "xwar_fmi": fit.war_fmi,
        "realised_war_mean": fit.realised_mean,
        "rank": fit.rank,
    }


def _draws_arrays(design: ModelDesign, fit: PooledFit) -> dict[str, np.ndarray]:
    """What evaluate needs from the pooled fit."""
    return {
        "skill_draws": fit.skill_draws.astype(np.float32),
        "general_ids": design.general_ids,
        "entry_xwar_mean": fit.entry_xwar,
        "entry_realised_war_mean": fit.entry_realised,
        "cmd_side": design.cmd_side,
        "cmd_general": design.cmd_general,
        "cmd_weight": design.cmd_weight,
        "battle_ids": design.battle_ids,
        "side_ids": design.side_ids,
        "e_model_mean": fit.e_model,
        "e_rep_mean": fit.e_rep,
        "x_diff_median": fit.x_diff,
    }


def _rubin_summary(fit: PooledFit) -> dict[str, Any]:
    """Headline Rubin numbers for the pooled diagnostics."""
    return {
        "skill_fmi_max": float(fit.skill_fmi.max()) if fit.skill_fmi.size else 0.0,
        "skill_fmi_median": float(np.median(fit.skill_fmi)) if fit.skill_fmi.size else 0.0,
        "xwar_fmi_max": float(fit.war_fmi.max()) if fit.war_fmi.size else 0.0,
        "beta_force": {
            "mean": fit.beta_force_mean,
            "sd_sqrt_T": fit.beta_force_sd,
            "fmi": fit.beta_force_fmi,
        },
    }


def _prior_summary(pp: PriorPredictive) -> dict[str, Any]:
    return {
        "category_freq": pp.category_freq,
        "median_abs_win_dev": pp.median_abs_win_dev,
        "skill_gap_p99": pp.skill_gap_p99,
        "win_prob_quantiles": pp.win_prob_quantiles,
    }


def _pooled_diagnostics(
    *,
    design: ModelDesign,
    comp: ComponentResult,
    combined: dict[str, Any],
    rounds: list[_RoundResult],
    imputation_run_id: int,
    joint: _JointResult | None,
    pp: PriorPredictive,
    pooled: PooledFit | None,
    gates: list[CheckResult],
    decomposition: float | None,
    directory: Path,
    type_counts: dict[str, int],
) -> dict[str, Any]:
    """The pooled ``model_runs.diagnostics`` payload."""
    circ = next((g for g in gates if g.name == "circularity_check"), None)
    return {
        "worst": combined["worst"],
        "converged": combined["converged"],
        "rounds_failed": combined["rounds_failed"],
        "rounds": [r.summary for r in rounds],
        "n_rounds": len(rounds),
        "estimates_written": pooled is not None,
        "imputation_run_id": imputation_run_id,
        "joint_run_id": None if joint is None else joint.run_id,
        "joint_converged": None if joint is None else joint.converged,
        "counts": {**design.counts.as_dict(), **type_counts},
        "components": {
            "n_components": comp.n_components,
            "largest_share": comp.largest_share,
            "sizes_top10": comp.sizes[:10],
        },
        "defender_term": "used" if design.defender_term_used else "omitted_no_data",
        "prior_predictive": _prior_summary(pp),
        "rubin": None if pooled is None else _rubin_summary(pooled),
        "circularity": None if circ is None else circ.details,
        "gates": {g.name: g.passed for g in gates},
        "decomposition_max_abs_diff": decomposition,
        "diagnostics_dir": str(directory),
    }


def _write_pooled(
    conn: Any,
    *,
    run_id: int,
    design: ModelDesign,
    pooled: PooledFit,
    joint: _JointResult | None,
) -> None:
    """Insert the pooled run's estimates and details, and the joint run's estimates."""
    store.write_skill_estimates(conn, run_id, _estimate_rows(design, pooled))
    store.write_war_details(conn, run_id, _detail_rows(design, pooled))
    if joint is not None and joint.fit is not None:
        store.write_skill_estimates(conn, joint.run_id, _estimate_rows(design, joint.fit))


def _round_x_diffs(design: ModelDesign, rounds: list[ImputedRound]) -> list[np.ndarray]:
    """``x_A - x_B`` per round, naming the round when a side is missing."""
    out: list[np.ndarray] = []
    for rnd in rounds:
        try:
            out.append(x_difference(design, rnd.log_troops))
        except ValueError as exc:
            raise ValueError(
                f"imputed round {rnd.index} is incomplete: {exc}. Rounds are complete by "
                "construction, so the files do not match this database; re-run impute."
            ) from exc
    return out


def _run_body(
    conn: Any,
    *,
    spec_params: dict[str, Any],
    ctx: StageContext,
    owns_connection: bool,
) -> None:
    """Run the whole stage against an open connection.

    Args:
        conn: An open database connection.
        spec_params: The spec's ``params`` mapping.
        ctx: Stage context (``limit`` and ``dry_run`` are honoured).
        owns_connection: Whether this stage opened ``conn`` itself.
    """
    mp = load_model_params(spec_params)
    sides = subset_sides(load_side_inputs(conn), ctx.limit)
    if not sides:
        logger.warning("model_no_sides", hint="no battle_sides rows; writing nothing")
        return
    design = build_design(sides)
    logger.info("model_design_built", n_generals=design.n_generals, **design.counts.as_dict())
    if design.n_battles == 0:
        logger.warning(
            "model_no_included_battles", hint="every battle was excluded; writing nothing"
        )
        return

    rounds = read_rounds(Path(mp.imputed_root), mp.scale, mp.max_rounds)
    imputation_run_id = check_round_provenance(
        rounds, latest_written_imputation_run_id(conn, mp.scale)
    )
    x_diffs = _round_x_diffs(design, rounds)
    if ctx.dry_run:
        logger.info("model_dry_run", n_rounds=len(rounds), imputation_run_id=imputation_run_id)
        return

    pp = prior_predictive(
        design,
        np.median(np.stack(x_diffs), axis=0),
        mp.priors,
        n_draws=mp.prior_draws,
        n_pairs=mp.prior_pairs,
        seed=mp.prior_seed,
    )
    results, kept = _fit_rounds(
        conn,
        owns_connection=owns_connection,
        mp=mp,
        design=design,
        rounds=rounds,
        x_diffs=x_diffs,
        imputation_run_id=imputation_run_id,
    )
    combined = combine_rounds(
        [r.outcome.worst for r in results],
        max_rhat=mp.max_rhat,
        min_ess_bulk=mp.min_ess_bulk,
        max_divergences=mp.max_divergences,
    )
    all_converged = combined["converged"] == "true"
    comp = components(design)
    spans = career_spans(design)

    joint: _JointResult | None = None
    pooled: PooledFit | None = None
    gates: list[CheckResult] = []
    decomposition: float | None = None
    if all_converged:
        pooled = pool_reductions(
            design, [r.reduction for r in results if r.reduction is not None], mp.rank_by
        )
        decomposition = assert_decomposition(design, pooled)
        if mp.joint_latent:
            joint = _try_fit_joint(
                conn,
                owns_connection=owns_connection,
                mp=mp,
                design=design,
                imputation_run_id=imputation_run_id,
            )
        assert kept is not None  # the representative round converged with the rest
        gates = run_gates(
            design=design,
            components=comp,
            spans=spans,
            pooled=pooled,
            joint=None if joint is None else joint.fit,
            prior_pp=pp,
            ppc_draws=kept[0],
            ppc_x_diff=kept[1],
            params=mp.gates,
        )
        for gate in gates:
            if not gate.passed:
                logger.warning("model_gate_failed", gate=gate.name, message=gate.message)
    else:
        logger.error(
            "model_not_written",
            rounds_failed=combined["rounds_failed"],
            worst=combined["worst"],
            hint="no estimates written; see the convergence gate",
        )

    _finish(
        conn,
        owns_connection=owns_connection,
        mp=mp,
        design=design,
        comp=comp,
        spans=spans,
        combined=combined,
        results=results,
        imputation_run_id=imputation_run_id,
        joint=joint,
        pp=pp,
        pooled=pooled,
        gates=gates,
        decomposition=decomposition,
        type_counts=_type_counts(design, load_battle_types(conn)),
    )


def _finish(
    conn: Any,
    *,
    owns_connection: bool,
    mp: ModelParams,
    design: ModelDesign,
    comp: ComponentResult,
    spans: np.ndarray,
    combined: dict[str, Any],
    results: list[_RoundResult],
    imputation_run_id: int,
    joint: _JointResult | None,
    pp: PriorPredictive,
    pooled: PooledFit | None,
    gates: list[CheckResult],
    decomposition: float | None,
    type_counts: dict[str, int],
) -> None:
    """Create the pooled run row last, write rows and files, then complete it."""
    config = {
        **mp.raw,
        "pooled_round_run_ids": [r.run_id for r in results],
        "joint_run_id": None if joint is None else joint.run_id,
        "imputation_run_id": imputation_run_id,
    }
    pooled_id = _begin(
        conn,
        owns_connection,
        model_type=POOLED_MODEL_TYPE,
        run_name=f"bt_pooled_imputation{imputation_run_id}",
        config=config,
    )
    directory = outputs.run_directory(Path(mp.diagnostics_root), pooled_id)
    outputs.write_json(directory / "rounds_summary.json", [r.summary for r in results])
    if pooled is not None:
        _write_pooled(conn, run_id=pooled_id, design=design, pooled=pooled, joint=joint)
        names = load_general_names(conn)
        outputs.write_columns_csv(
            directory / "rubin_per_general.csv",
            _per_general_columns(design, pooled, comp, spans, names),
        )
        outputs.write_json(directory / "components.json", _components_payload(design, comp, spans))
        outputs.write_draws_npz(directory / "pooled_draws.npz", _draws_arrays(design, pooled))
    outputs.write_gate_report(
        directory,
        run_id=pooled_id,
        estimates_written=pooled is not None,
        gates=[gate_dict(g) for g in gates],
    )
    store.complete_run(
        conn,
        pooled_id,
        _pooled_diagnostics(
            design=design,
            comp=comp,
            combined=combined,
            rounds=results,
            imputation_run_id=imputation_run_id,
            joint=joint,
            pp=pp,
            pooled=pooled,
            gates=gates,
            decomposition=decomposition,
            directory=directory,
            type_counts=type_counts,
        ),
    )
    if owns_connection:
        conn.commit()
    logger.info(
        "model_complete",
        run_id=pooled_id,
        converged=combined["converged"],
        n_rounds=len(results),
        n_generals=design.n_generals,
    )


def run(spec: dict[str, Any], context: StageContext | None = None) -> None:
    """Run the model stage.

    Args:
        spec: The loaded ``agents/model.yaml``, with overrides applied.
        context: Shared services. Without a connection the stage opens its own:
            every input is derived from the database.
    """
    ctx = context or StageContext()
    params: dict[str, Any] = spec.get("params") or {}

    if ctx.db_conn is not None:
        _run_body(ctx.db_conn, spec_params=params, ctx=ctx, owns_connection=False)
        return

    with get_connection() as conn:
        _run_body(conn, spec_params=params, ctx=ctx, owns_connection=True)
