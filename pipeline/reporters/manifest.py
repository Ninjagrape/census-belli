"""``manifest.json``: what was written, and the gates the spec reads back."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from pipeline.reporters.atomic import write_text_atomic
from pipeline.reporters.view import ReportView

__all__ = [
    "EVALUATION_CURRENT",
    "EXPORT_MATCHES_DB",
    "build_manifest",
    "evaluation_current_gate",
    "export_matches_db_gate",
    "write_manifest",
]

SCHEMA_VERSION: Final[int] = 1
EXPORT_MATCHES_DB: Final[str] = "export_matches_db"
EVALUATION_CURRENT: Final[str] = "evaluation_current"


def _gate(name: str, passed: bool, actual: Any, threshold: str, message: str) -> dict[str, Any]:
    """A gate in the model and evaluate stages' format, at error severity."""
    return {
        "name": name,
        "passed": bool(passed),
        "severity": "error",
        "threshold": threshold,
        "actual": actual,
        "message": message,
    }


def export_matches_db_gate(*, csv_headline_rows: int, db_headline_rows: int) -> dict[str, Any]:
    """The CSV's headline rows must equal the database's count.

    Args:
        csv_headline_rows: Rows with ``headline`` true, read back from the CSV.
        db_headline_rows: ``general_skill_estimates`` rows of the run with at
            least ``min_battles`` credited battles, counted in SQL.

    Returns:
        The gate dict.
    """
    passed = csv_headline_rows == db_headline_rows and db_headline_rows > 0
    message = (
        f"rankings.csv holds {csv_headline_rows} headline rows; the database has "
        f"{db_headline_rows} generals at or above the battle threshold"
    )
    return _gate(
        EXPORT_MATCHES_DB,
        passed,
        {"csv_headline_rows": csv_headline_rows, "db_headline_rows": db_headline_rows},
        "csv headline rows == database count, and > 0",
        message,
    )


def evaluation_current_gate(
    *, evaluation_reason: str, evaluated_run_id: int | None
) -> dict[str, Any]:
    """The evaluation report read must describe the reported run.

    Args:
        evaluation_reason: Empty when a current report was read, else why not.
        evaluated_run_id: The run the report read validated, if any.

    Returns:
        The gate dict.
    """
    passed = not evaluation_reason
    message = (
        "the evaluation report describes the reported model run" if passed else evaluation_reason
    )
    return _gate(
        EVALUATION_CURRENT,
        passed,
        {"evaluated_run_id": evaluated_run_id},
        "an evaluation report for the reported model run",
        message,
    )


def build_manifest(
    view: ReportView,
    *,
    csv_rows: int,
    json_generals: int,
    gates: list[dict[str, Any]],
    files: Mapping[str, str],
) -> dict[str, Any]:
    """Assemble the manifest.

    Args:
        view: The report view.
        csv_rows: Data rows in ``rankings.csv``.
        json_generals: Generals in ``rankings.json``.
        gates: The stage-side gates.
        files: Role to file name of each output written.

    Returns:
        The manifest dictionary; ``model_run_id`` is what ``evaluation_gate``
        compares against the newest evaluable run.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "model_run_id": view.model_run_id,
        "evaluation_run_id": view.evaluation.model_run_id if view.evaluation else None,
        "provisional": view.provisional,
        "reasons": list(view.reasons),
        "counts": {
            "headline": len(view.headline),
            "under_determined": len(view.under_determined),
            "csv_rows": csv_rows,
            "json_generals": json_generals,
        },
        "files": dict(files),
        "gates": gates,
    }


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> Path:
    """Write the manifest, creating parents.

    Args:
        path: Destination file.
        manifest: Output of :func:`build_manifest`.

    Returns:
        The path.
    """
    return write_text_atomic(path, json.dumps(manifest, indent=2, allow_nan=False))
