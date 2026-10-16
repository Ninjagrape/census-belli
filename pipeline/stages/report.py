"""Report stage: files for people to read, from one model run and its evaluation.

Presentational. It reads the evaluable model run and evaluate's files, writes
files only, never writes to the database and computes no inference. A missing
or stale evaluation report, or a failed blocking model gate, does not stop it:
the outputs are written and labelled provisional, and the spec's gates decide
whether the run may be published.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog
from sqlalchemy.engine import Connection

from pipeline.db import get_connection
from pipeline.reporters import load
from pipeline.reporters.atomic import remove_if_present, write_text_atomic
from pipeline.reporters.config import ReportParams
from pipeline.reporters.explorer import render_explorer
from pipeline.reporters.exports import (
    build_payload,
    read_csv_rows,
    write_rankings_csv,
    write_rankings_json,
)
from pipeline.reporters.figures import copy_evaluation_figures, write_forest_plot
from pipeline.reporters.manifest import (
    build_manifest,
    evaluation_current_gate,
    export_matches_db_gate,
    write_manifest,
)
from pipeline.reporters.markdown import render_markdown
from pipeline.reporters.view import ReportView, build_view
from pipeline.stages.base import StageContext

__all__ = ["run"]

logger = structlog.get_logger()

FILES = {
    "csv": "rankings.csv",
    "json": "rankings.json",
    "explorer": "explorer.html",
    "report": "report.md",
}


def _build(conn: Connection, params: ReportParams, run_id: int) -> ReportView:
    diagnostics_root = Path(params.diagnostics_root)
    evaluation, reason = load.load_evaluation(Path(params.evaluation_root), run_id)
    rank_by = load.load_rank_by(conn, run_id)
    return build_view(
        params,
        run_id=run_id,
        estimates=load.load_estimates(conn, run_id),
        battles=load.load_battle_details(conn, run_id),
        counts=load.load_model_counts(conn, run_id),
        rubin=load.load_rubin(diagnostics_root, run_id),
        model_gates=load.load_model_gates(diagnostics_root, run_id),
        evaluation=evaluation,
        evaluation_reason=reason,
        evaluation_severities=load.load_evaluate_severities(Path(params.evaluate_spec_path)),
        pooled_draws=load.load_pooled_draws(diagnostics_root, run_id, rank_by),
    )


def _write_all(view: ReportView, db_headline_rows: int) -> dict[str, Any]:
    out = Path(view.params.output_root)
    figures_dir = out / "figures"
    forest = write_forest_plot(view.top, figures_dir, view.params.ci_level)
    figure_names = [
        *([forest] if forest else []),
        *copy_evaluation_figures(view.evaluation, figures_dir),
    ]

    csv_rows = write_rankings_csv(out / FILES["csv"], view)
    payload = build_payload(view)
    write_rankings_json(out / FILES["json"], payload)
    explorer_payload = build_payload(view, limit=view.params.explorer_max_generals)
    write_text_atomic(out / FILES["explorer"], render_explorer(explorer_payload))
    write_text_atomic(out / FILES["report"], render_markdown(view, figure_names))

    csv_headline = sum(r["headline"] == "true" for r in read_csv_rows(out / FILES["csv"]))
    gates = [
        export_matches_db_gate(csv_headline_rows=csv_headline, db_headline_rows=db_headline_rows),
        evaluation_current_gate(
            evaluation_reason=view.evaluation_reason,
            evaluated_run_id=view.evaluation.model_run_id if view.evaluation else None,
        ),
    ]
    manifest = build_manifest(
        view,
        csv_rows=csv_rows,
        json_generals=len(payload["generals"]),
        gates=gates,
        files={**FILES, **{f"figure_{i}": n for i, n in enumerate(figure_names)}},
    )
    write_manifest(Path(view.params.report_path), manifest)
    return manifest


def _clear_manifest(params: ReportParams) -> None:
    """Remove any manifest from an earlier run before this run writes or fails.

    The spec's gates read the manifest, so a stale one would vouch for outputs
    this run did not produce; with it gone they fail instead.
    """
    for path in {Path(params.output_root) / "manifest.json", Path(params.report_path)}:
        if remove_if_present(path):
            logger.warning("report_stale_manifest_removed", path=str(path))


def _run_body(conn: Connection, spec_params: dict[str, Any], ctx: StageContext) -> None:
    params = ReportParams.from_mapping(spec_params)
    run_id = load.select_run(conn)
    if run_id is None:
        logger.error(
            "report_no_evaluable_run",
            hint="no converged model run wrote estimates; nothing is written",
        )
        if not ctx.dry_run:
            _clear_manifest(params)
        return
    if ctx.dry_run:
        logger.info("report_dry_run", model_run_id=run_id, hint="nothing written")
        return
    _clear_manifest(params)
    view = _build(conn, params, run_id)
    manifest = _write_all(view, load.count_headline_eligible(conn, run_id, params.min_battles))
    for reason in view.reasons:
        logger.warning("report_provisional", reason=reason)
    for gate in manifest["gates"]:
        if not gate["passed"]:
            logger.warning("report_gate_failed", gate=gate["name"], message=gate["message"])
    logger.info(
        "report_complete",
        model_run_id=run_id,
        provisional=view.provisional,
        output_root=params.output_root,
        **manifest["counts"],
    )


def run(spec: dict[str, Any], context: StageContext | None = None) -> None:
    """Run the report stage.

    Args:
        spec: The loaded ``agents/report.yaml``, with overrides applied.
        context: Shared services. Without a connection the stage opens its own.
    """
    ctx = context or StageContext()
    params: dict[str, Any] = spec.get("params") or {}
    if ctx.db_conn is not None:
        _run_body(ctx.db_conn, params, ctx)
        return
    with get_connection() as conn:
        _run_body(conn, params, ctx)
