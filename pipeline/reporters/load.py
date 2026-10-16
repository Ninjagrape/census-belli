"""Read-only loaders: the database rows and the model and evaluate files of one run."""

from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import structlog
import yaml
from sqlalchemy import text
from sqlalchemy.engine import Connection

from pipeline.quality import LATEST_EVALUABLE_RUN
from pipeline.reporters.records import (
    BattleDetail,
    EvaluationInfo,
    GeneralEstimate,
    ModelGate,
    PooledDraws,
    RubinInfo,
    VariantSummary,
)

__all__ = [
    "count_headline_eligible",
    "load_battle_details",
    "load_estimates",
    "load_evaluate_severities",
    "load_evaluation",
    "load_model_counts",
    "load_model_gates",
    "load_pooled_draws",
    "load_rank_by",
    "load_rubin",
    "parse_evaluation",
    "select_run",
]

logger = structlog.get_logger()

EVALUATION_REPORT_NAME = "evaluation_report.json"
_PNG_NAME = re.compile(r"[A-Za-z0-9_.-]+\.png")
_VARIANT_NAME = re.compile(r"[A-Za-z0-9_-]+")

_ESTIMATES = text(
    "SELECT e.general_id, g.canonical_name, g.wikidata_id, e.skill_mean, e.skill_median, "
    "e.skill_sd, e.skill_ci_lo, e.skill_ci_hi, e.skill_ci90_lo, e.skill_ci90_hi, "
    "e.total_war, e.war_per_battle, e.war_sd, e.n_battles, e.rank, "
    "e.war_ci_lo, e.war_ci_hi, e.p_above_next_rank "
    "FROM general_skill_estimates e JOIN generals g ON g.general_id = e.general_id "
    "WHERE e.run_id = :run_id ORDER BY e.rank NULLS LAST, e.general_id"
)
_DETAILS = text(
    "SELECT d.general_id, d.battle_id, b.name AS battle_name, b.year_astronomical, "
    "s.polity, CAST(s.outcome AS TEXT) AS side_outcome, "
    "CAST(bc.command_role AS TEXT) AS command_role, d.attribution_weight, "
    "d.war_contribution, d.win_prob_model, d.win_prob_baseline, d.actual_outcome "
    "FROM battle_war_details d JOIN battles b ON b.battle_id = d.battle_id "
    "LEFT JOIN battle_commanders bc "
    "ON bc.battle_id = d.battle_id AND bc.general_id = d.general_id "
    "LEFT JOIN battle_sides s ON s.side_id = bc.side_id "
    "WHERE d.run_id = :run_id "
    "ORDER BY d.general_id, b.year_astronomical, d.battle_id, bc.bc_id"
)
_COUNTS = text("SELECT diagnostics -> 'counts' FROM model_runs WHERE run_id = :run_id")
_RANK_BY = text("SELECT config ->> 'rank_by' FROM model_runs WHERE run_id = :run_id")
_ELIGIBLE = text(
    "SELECT COUNT(*) FROM general_skill_estimates "
    "WHERE run_id = :run_id AND n_battles >= :min_battles"
)


def _num(value: Any) -> float | None:
    """A finite float, or None for missing and non-finite values."""
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _int(value: Any) -> int | None:
    number = _num(value)
    return None if number is None else int(number)


def select_run(conn: Connection) -> int | None:
    """The newest converged run that wrote estimates, as evaluate and the gates pick it.

    Args:
        conn: An open connection.

    Returns:
        The run id, or None when no run is evaluable.
    """
    run_id = conn.execute(LATEST_EVALUABLE_RUN).scalar()
    return None if run_id is None else int(run_id)


def load_estimates(conn: Connection, run_id: int) -> tuple[GeneralEstimate, ...]:
    """Every general's stored estimate for the run, in stored rank order.

    Args:
        conn: An open connection.
        run_id: The model run.

    Returns:
        The estimates, ``NULL`` ranks last.
    """
    rows = conn.execute(_ESTIMATES, {"run_id": run_id}).mappings().all()
    return tuple(
        GeneralEstimate(
            general_id=int(r["general_id"]),
            name=str(r["canonical_name"]),
            wikidata_id=r["wikidata_id"],
            skill_mean=float(r["skill_mean"]),
            skill_median=_num(r["skill_median"]),
            skill_sd=float(r["skill_sd"]),
            skill_ci_lo=float(r["skill_ci_lo"]),
            skill_ci_hi=float(r["skill_ci_hi"]),
            skill_ci90_lo=_num(r["skill_ci90_lo"]),
            skill_ci90_hi=_num(r["skill_ci90_hi"]),
            total_war=_num(r["total_war"]),
            war_per_battle=_num(r["war_per_battle"]),
            war_sd=_num(r["war_sd"]),
            n_battles=int(r["n_battles"]),
            rank=_int(r["rank"]),
            war_ci_lo=_num(r["war_ci_lo"]),
            war_ci_hi=_num(r["war_ci_hi"]),
            p_above_next_rank=_num(r["p_above_next_rank"]),
        )
        for r in rows
    )


def load_battle_details(conn: Connection, run_id: int) -> dict[int, tuple[BattleDetail, ...]]:
    """Per-battle WAR rows by general, with battle and side context.

    A general holding two commander rows in one battle would join twice; the
    first is kept so the contributions still sum to the stored total.

    Args:
        conn: An open connection.
        run_id: The model run.

    Returns:
        ``general_id -> battles``, oldest first.
    """
    seen: set[tuple[int, int]] = set()
    grouped: dict[int, list[BattleDetail]] = {}
    for r in conn.execute(_DETAILS, {"run_id": run_id}).mappings().all():
        key = (int(r["general_id"]), int(r["battle_id"]))
        if key in seen:
            continue
        seen.add(key)
        grouped.setdefault(key[0], []).append(
            BattleDetail(
                general_id=key[0],
                battle_id=key[1],
                battle_name=str(r["battle_name"]),
                year=_int(r["year_astronomical"]),
                polity=r["polity"],
                side_outcome=r["side_outcome"],
                command_role=r["command_role"],
                attribution_weight=_num(r["attribution_weight"]),
                expected_war=float(r["war_contribution"]),
                expected_score_model=_num(r["win_prob_model"]),
                expected_score_replacement=_num(r["win_prob_baseline"]),
                actual_outcome=_num(r["actual_outcome"]),
            )
        )
    return {g: tuple(rows) for g, rows in grouped.items()}


def load_model_counts(conn: Connection, run_id: int) -> dict[str, Any]:
    """The design counts the model stage recorded in ``model_runs.diagnostics``.

    Args:
        conn: An open connection.
        run_id: The model run.

    Returns:
        The counts, empty when none were recorded.
    """
    raw = conn.execute(_COUNTS, {"run_id": run_id}).scalar()
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            raw = None
    return dict(raw) if isinstance(raw, dict) else {}


def count_headline_eligible(conn: Connection, run_id: int, min_battles: int) -> int:
    """Generals with at least ``min_battles`` credited battles, counted in SQL.

    Args:
        conn: An open connection.
        run_id: The model run.
        min_battles: The headline threshold.

    Returns:
        The count, which the export must equal.
    """
    value = conn.execute(_ELIGIBLE, {"run_id": run_id, "min_battles": min_battles}).scalar()
    return int(value or 0)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("report_input_unreadable", path=str(path), error=type(exc).__name__)
        return None


def load_model_gates(diagnostics_root: Path, run_id: int) -> tuple[ModelGate, ...] | None:
    """The model stage's gates for the run.

    Args:
        diagnostics_root: Root of the model stage's ``run{N}`` directories.
        run_id: The model run.

    Returns:
        The gates, or None when ``gate_report.json`` is missing or unreadable.
    """
    report = _read_json(diagnostics_root / f"run{run_id}" / "gate_report.json")
    gates = report.get("gates") if isinstance(report, dict) else None
    if not isinstance(gates, list):
        return None
    return tuple(
        ModelGate(
            name=str(g.get("name", "")),
            passed=bool(g.get("passed")),
            severity=str(g.get("severity", "")),
            message=str(g.get("message", "")),
        )
        for g in gates
        if isinstance(g, dict)
    )


def load_rank_by(conn: Connection, run_id: int) -> str:
    """The ``rank_by`` the run was fitted with (``total_war`` when not recorded).

    Args:
        conn: An open connection.
        run_id: The model run.

    Returns:
        ``total_war`` or ``skill_mean``.
    """
    value = conn.execute(_RANK_BY, {"run_id": run_id}).scalar()
    return "skill_mean" if value == "skill_mean" else "total_war"


def load_pooled_draws(
    diagnostics_root: Path, run_id: int, rank_by: str = "total_war"
) -> PooledDraws | None:
    """The run's ``pooled_draws.npz``, as per-general draw columns.

    Args:
        diagnostics_root: Root of the model stage's ``run{N}`` directories.
        run_id: The model run.
        rank_by: The run's ranking basis; ``skill_mean`` selects skill draws.

    Returns:
        The draws, or None when the file is missing or unusable.
    """
    path = diagnostics_root / f"run{run_id}" / "pooled_draws.npz"
    if not path.is_file():
        return None
    try:
        with np.load(path) as npz:
            ids = np.asarray(npz["general_ids"])
            war = np.asarray(npz["war_draws"], dtype=float)
            skill = np.asarray(npz["skill_draws"], dtype=float) if "skill_draws" in npz else None
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("report_draws_unreadable", path=str(path), error=type(exc).__name__)
        return None
    want_skill = rank_by == "skill_mean"
    use_skill = want_skill and skill is not None
    matrix = skill if use_skill and skill is not None else war
    if matrix.ndim != 2 or matrix.shape[1] != len(ids):
        logger.warning("report_draws_misshapen", path=str(path), shape=list(matrix.shape))
        return None
    return PooledDraws(
        draws={int(g): matrix[:, i] for i, g in enumerate(ids)},
        basis="skill" if use_skill else "war",
        skill_unavailable=want_skill and not use_skill,
    )


def load_rubin(diagnostics_root: Path, run_id: int) -> dict[int, RubinInfo]:
    """Per-general columns of ``rubin_per_general.csv``.

    Args:
        diagnostics_root: Root of the model stage's ``run{N}`` directories.
        run_id: The model run.

    Returns:
        ``general_id -> info``; empty when the file is missing.
    """
    path = diagnostics_root / f"run{run_id}" / "rubin_per_general.csv"
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        logger.warning("report_rubin_missing", path=str(path))
        return {}
    out: dict[int, RubinInfo] = {}
    for r in rows:
        general_id = _int(r.get("general_id"))
        if general_id is None:
            continue
        out[general_id] = RubinInfo(
            component_id=_int(r.get("component_id")),
            effective_battles=_num(r.get("effective_battles")),
            n_uncredited_battles=_int(r.get("n_uncredited_battles")),
            first_year=_num(r.get("first_year")),
            last_year=_num(r.get("last_year")),
        )
    return out


def _variant_summaries(sensitivity: Mapping[str, Any]) -> tuple[VariantSummary, ...]:
    out: list[VariantSummary] = []
    for v in sensitivity.get("variants") or []:
        if not isinstance(v, dict):
            continue
        score = (v.get("vs_baseline") or {}).get("score") or {}
        out.append(
            VariantSummary(
                name=str(v.get("name", "")),
                status=str(v.get("status", "")),
                message=v.get("message"),
                spearman_rho=_num(score.get("spearman_rho")),
                top_overlap=_int(score.get("top_overlap")),
            )
        )
    return tuple(out)


def _figure_files(report: Mapping[str, Any], directory: Path) -> tuple[Path, ...]:
    """Evaluation PNGs that exist, resolved from the file name alone.

    The report stores Windows-separated paths from whichever machine ran
    evaluate, so only the name is trusted and the directory is rebuilt here.
    """
    found: list[Path] = []
    for raw in report.get("figures") or []:
        name = PureWindowsPath(str(raw)).name
        path = directory / "figures" / name
        if _PNG_NAME.fullmatch(name) and path.is_file():
            found.append(path)
    return tuple(found)


def _sensitivity_ranks(directory: Path) -> dict[str, dict[int, int]]:
    ranks: dict[str, dict[int, int]] = {}
    for path in sorted((directory / "sensitivity").glob("*.csv")):
        if not _VARIANT_NAME.fullmatch(path.stem):
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            table: dict[int, int] = {}
            for r in csv.DictReader(handle):
                general_id, rank = _int(r.get("general_id")), _int(r.get("rank"))
                if general_id is not None and rank is not None:
                    table[general_id] = rank
        ranks[path.stem] = table
    return ranks


def parse_evaluation(report: Mapping[str, Any], directory: Path) -> EvaluationInfo:
    """Pick out what the report needs from a parsed ``evaluation_report.json``.

    Args:
        report: The parsed JSON.
        directory: ``{evaluation_root}/run{N}``, for the figures and CSVs.

    Returns:
        The evaluation info.
    """
    binary = ((report.get("cv") or {}).get("imputed") or {}).get("binary") or {}
    gates: Sequence[Any] = report.get("gates") or []
    disclosures: Sequence[Any] = report.get("disclosures") or []
    return EvaluationInfo(
        model_run_id=int(report["model_run_id"]),
        directory=directory,
        disclosures=tuple(str(d) for d in disclosures),
        gates=tuple(g for g in gates if isinstance(g, dict)),
        variants=_variant_summaries(report.get("sensitivity") or {}),
        cv_accuracy=_num(binary.get("accuracy")),
        cv_brier=_num(binary.get("brier")),
        cv_n=_int(binary.get("n")),
        figure_files=_figure_files(report, directory),
        sensitivity_ranks=_sensitivity_ranks(directory),
    )


def load_evaluate_severities(spec_path: Path) -> dict[str, str]:
    """Severity of each evaluate gate, from the evaluate spec's ``evaluation_gate`` checks.

    A check's gate is its ``params.gate``, else its ``name``. An unreadable spec
    gives no severities, so every gate falls back to blocking.

    Args:
        spec_path: Path to ``agents/evaluate.yaml``.

    Returns:
        ``gate name -> severity``; empty when the spec is unreadable.
    """
    try:
        spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("evaluate_spec_unreadable", path=str(spec_path), error=type(exc).__name__)
        return {}
    checks = spec.get("quality_checks") if isinstance(spec, dict) else None
    if not isinstance(checks, list):
        logger.warning("evaluate_spec_has_no_checks", path=str(spec_path))
        return {}
    out: dict[str, str] = {}
    for check in checks:
        if not isinstance(check, dict) or check.get("method") != "evaluation_gate":
            continue
        params = check.get("params")
        gate = params.get("gate") if isinstance(params, dict) else None
        name = str(gate or check.get("name") or "")
        severity = check.get("severity")
        if name and isinstance(severity, str):
            out[name] = severity
    return out


def load_evaluation(evaluation_root: Path, run_id: int) -> tuple[EvaluationInfo | None, str]:
    """The evaluation report for the run, or why there is none.

    Looks for ``run{N}/evaluation_report.json`` and then the latest copy beside
    the root. A report whose ``model_run_id`` is not ``run_id`` is stale and is
    not read any further.

    Args:
        evaluation_root: Root of the evaluate stage's ``run{N}`` directories.
        run_id: The model run being reported.

    Returns:
        The info and an empty reason, or None and the reason it is unusable.
    """
    directory = evaluation_root / f"run{run_id}"
    reason = f"no evaluation report for model run {run_id}; run the evaluate stage"
    for path in (
        directory / EVALUATION_REPORT_NAME,
        evaluation_root.parent / EVALUATION_REPORT_NAME,
    ):
        report = _read_json(path) if path.is_file() else None
        if not isinstance(report, dict):
            continue
        evaluated = _int(report.get("model_run_id"))
        if evaluated == run_id:
            try:
                return parse_evaluation(report, directory), ""
            except (KeyError, TypeError, ValueError, AttributeError, OSError, csv.Error) as exc:
                logger.warning("report_evaluation_malformed", path=str(path), error=str(exc))
                reason = (
                    f"evaluation report for model run {run_id} is malformed "
                    f"({type(exc).__name__}); re-run evaluate"
                )
                continue
        reason = (
            f"stale evaluation report: evaluated run {evaluated}, reported run {run_id}; "
            "re-run evaluate"
        )
    return None, reason
