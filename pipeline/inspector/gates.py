"""Run a stage's declared quality gates on a read-only connection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final, cast

import structlog
from sqlalchemy.engine import Connection

from pipeline.config import load_agent_spec
from pipeline.inspector.frames import Frame, pd
from pipeline.quality import QualityRunner

logger = structlog.get_logger()

__all__ = ["GATE_COLUMNS", "GATED_STAGES", "run_all_gates", "run_stage_gates"]

GATED_STAGES: Final[tuple[str, ...]] = (
    "crawl",
    "extract",
    "resolve",
    "reconcile",
    "classify",
    "impute",
)
GATE_COLUMNS: Final[list[str]] = [
    "stage",
    "name",
    "passed",
    "severity",
    "actual",
    "threshold",
    "message",
]
_AGENTS_DIR: Final = Path(__file__).resolve().parents[2] / "agents"


def run_stage_gates(conn: Connection, stage: str) -> Frame:
    """Run one stage's ``quality_checks``.

    Never raises: a missing spec or a failing runner becomes a single failing
    row so one broken stage cannot hide the others.

    Args:
        conn: An open (read-only) connection.
        stage: The stage name, matching ``agents/<stage>.yaml``.

    Returns:
        One row per check with columns :data:`GATE_COLUMNS`.
    """
    try:
        spec = load_agent_spec(stage, agents_dir=_AGENTS_DIR)
        checks = cast("list[dict[str, Any]]", spec.get("quality_checks") or [])
        results = QualityRunner(cast("Any", conn)).run_all(checks)
    except Exception as exc:
        logger.warning("inspector_gate_error", stage=stage, error=str(exc))
        _rollback_quietly(conn)
        return _error_row(stage, exc)

    rows = [
        {
            "stage": stage,
            "name": r.name,
            "passed": r.passed,
            "severity": r.severity.value,
            "actual": "" if r.actual_value is None else str(r.actual_value),
            "threshold": r.threshold,
            "message": r.message,
        }
        for r in results
    ]
    return pd.DataFrame(rows, columns=GATE_COLUMNS)


def run_all_gates(conn: Connection, stages: tuple[str, ...] = GATED_STAGES) -> Frame:
    """Run the gates of several stages and stack the results.

    Args:
        conn: An open (read-only) connection.
        stages: Stages to run, in order.

    Returns:
        All rows with columns :data:`GATE_COLUMNS`.
    """
    frames = [run_stage_gates(conn, stage) for stage in stages]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=GATE_COLUMNS)


def _rollback_quietly(conn: Connection) -> None:
    """Clear an aborted transaction so later stages can still query."""
    try:
        conn.rollback()
    except Exception as exc:
        logger.debug("inspector_rollback_failed", error=str(exc))


def _error_row(stage: str, exc: Exception) -> Frame:
    """A one-row frame reporting that a stage's gates could not run."""
    row = {
        "stage": stage,
        "name": "<gates could not run>",
        "passed": False,
        "severity": "error",
        "actual": "",
        "threshold": "",
        "message": f"{type(exc).__name__}: {exc}",
    }
    return pd.DataFrame([row], columns=GATE_COLUMNS)
