"""``manifest.json``: what was written, and the gates the spec reads back."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from pipeline.reporters.atomic import write_text_atomic
from pipeline.reporters.coverage_check import differences
from pipeline.reporters.view import ReportView

__all__ = [
    "EVALUATION_CURRENT",
    "COVERAGE_FIGURES_WRITTEN",
    "DISCLOSURES_MATCH_DB",
    "EXPORT_MATCHES_DB",
    "build_manifest",
    "coverage_figures_gate",
    "disclosures_match_db_gate",
    "evaluation_current_gate",
    "export_matches_db_gate",
    "write_manifest",
]

SCHEMA_VERSION: Final[int] = 1
EXPORT_MATCHES_DB: Final[str] = "export_matches_db"
EVALUATION_CURRENT: Final[str] = "evaluation_current"
DISCLOSURES_MATCH_DB: Final[str] = "disclosures_match_db"
COVERAGE_FIGURES_WRITTEN: Final[str] = "coverage_figures_written"


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


def disclosures_match_db_gate(
    *,
    fresh: Sequence[str],
    reported: Sequence[str],
    in_json: Sequence[str],
    in_markdown: Sequence[str],
    used_counts: Mapping[str, int | None] | None = None,
    recounted: Mapping[str, int | None] | None = None,
) -> dict[str, Any]:
    """Every number in the coverage disclosures must equal a fresh recomputation.

    The sentences carry their numbers, so comparing whole sentences compares the
    numbers: a sentence from a database that has since changed, or one lost on
    the way to a file, shows up as a difference.

    Args:
        fresh: The coverage sentences recomputed from the database just now.
        reported: The coverage sentences the view carried when it rendered.
        in_json: Disclosures read back from ``rankings.json``.
        in_markdown: Disclosures read back from ``report.md``, unescaped.
        used_counts: Counts the sentences were built from (credited sides, rule_single
            sides, headline generals, credited battles).
        recounted: The same counts from the gate's own raw SQL
            (:func:`pipeline.reporters.coverage_check.recount`). Both default to
            "not checked".

    Returns:
        The gate dict. Passes with nothing to check only when both sides are empty;
        fails when a recount differs, or when the run has credited battles but no
        coverage sentence was produced.
    """
    stale = [s for s in reported if s not in fresh]
    missing_json = [s for s in fresh if s not in in_json]
    missing_md = [s for s in fresh if s not in in_markdown]
    count_diffs = (
        {} if used_counts is None or recounted is None else differences(used_counts, recounted)
    )
    silent = bool((recounted or {}).get("credited_battles")) and not fresh
    passed = (
        tuple(fresh) == tuple(reported)
        and not missing_json
        and not missing_md
        and not count_diffs
        and not silent
    )
    message = (
        f"coverage sentences match a fresh database recomputation (checked: {len(fresh)})"
        if passed
        else (
            f"coverage sentences differing from the database: {len(stale)}; "
            f"missing from rankings.json: {len(missing_json)}; "
            f"missing from report.md: {len(missing_md)}; "
            f"counts differing from an independent SQL recount (used, recounted): {count_diffs}; "
            f"no coverage sentences despite credited battles: {silent}"
        )
    )
    return _gate(
        DISCLOSURES_MATCH_DB,
        passed,
        {
            "fresh": len(fresh),
            "stale": stale,
            "missing_json": len(missing_json),
            "missing_markdown": len(missing_md),
            "count_differences": {k: list(v) for k, v in count_diffs.items()},
            "recounted": dict(recounted or {}),
            "no_sentences_despite_battles": silent,
        },
        "every coverage sentence equals its recomputation and reaches both outputs",
        message,
    )


def coverage_figures_gate(
    *,
    expected: Sequence[str],
    on_disk: Sequence[str],
    in_markdown: Sequence[str],
    explorer_has_tab: bool,
    written_panels: Sequence[str] | None = None,
    explorer_panels: Sequence[str] | None = None,
    explorer_hosts: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Each coverage figure whose data existed was written, embedded, and the explorer has its tab.

    Args:
        expected: Figure names for the panels that had data.
        on_disk: Those names found in the figures directory after writing.
        in_markdown: Those names that ``report.md`` references.
        explorer_has_tab: Whether ``explorer.html`` contains the coverage tab.
        written_panels: Panel keys that were built for the report. With the next two,
            checks the explorer against them.
        explorer_panels: Panel keys of the ``coverage`` object embedded in the explorer.
        explorer_hosts: Panel keys for which the explorer page has a chart host element.

    Returns:
        The gate dict. A report with no coverage data at all passes on the
        figures but still needs the explorer tab.
    """
    not_written = [n for n in expected if n not in on_disk]
    not_embedded = [n for n in expected if n not in in_markdown]
    panels_off = (
        sorted(set(written_panels or ()) ^ set(explorer_panels or ()))
        if written_panels is not None and explorer_panels is not None
        else []
    )
    no_host = (
        [k for k in written_panels if k not in set(explorer_hosts or ())]
        if written_panels is not None and explorer_hosts is not None
        else []
    )
    passed = not not_written and not not_embedded and explorer_has_tab
    passed = passed and not panels_off and not no_host
    message = (
        f"coverage figures written and embedded (checked: {len(expected)}); explorer tab present"
        if passed
        else (
            f"coverage figures not written: {not_written}; not referenced in report.md: "
            f"{not_embedded}; explorer coverage tab present: {explorer_has_tab}; "
            f"explorer payload panels differing from the written ones: {panels_off}; "
            f"panels without a chart host in the explorer: {no_host}"
        )
    )
    return _gate(
        COVERAGE_FIGURES_WRITTEN,
        passed,
        {
            "expected": list(expected),
            "not_written": not_written,
            "not_embedded": not_embedded,
            "explorer_has_tab": explorer_has_tab,
            "explorer_payload_mismatch": panels_off,
            "explorer_missing_hosts": no_host,
        },
        "every figure with data is written and embedded, and the explorer has the coverage tab",
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
