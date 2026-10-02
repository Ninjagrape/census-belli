"""Build and write the reconcile stage's summary report.

``data/processed/reconciliation_report.json`` is the human-facing record of
one run: what got read, what got fitted, and how well it converged, without
needing a database connection to inspect it. It is deliberately not the same
thing as ``model_runs.diagnostics`` -- that JSONB column is what
``pipeline.quality``'s ``diagnostics_json`` handler reads back for the
``model_convergence`` gate, and is scoped to one run's own fit. This report
covers the whole stage invocation: both quantities, the sides that could not
be estimated at all, and the regime mix the loader saw, in one file a human
can open.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import structlog

from pipeline.reconcilers.records import ReconcileCounts, Report

__all__ = ["REPORT_FILENAME", "ReconcileReport", "build_report", "write_report"]

logger = structlog.get_logger()

REPORT_FILENAME: Final[str] = "reconciliation_report.json"


@dataclass(frozen=True)
class ReconcileReport:
    """The full report payload, as written to disk.

    Attributes:
        computed_at: When this report was built, ISO-8601.
        counts: :class:`~pipeline.reconcilers.records.ReconcileCounts`,
            rendered.
        run_ids: ``model_runs.run_id`` per quantity that was fitted this run,
            or ``None`` for a quantity that had no usable reports and so was
            skipped -- see ``pipeline/stages/reconcile.py``'s module
            docstring for why a skipped quantity writes no run row at all.
        diagnostics: Each fitted quantity's diagnostics payload (the same
            dict written to that quantity's ``model_runs.diagnostics``), or
            ``None`` for a skipped quantity.
        regime_counts: How many reports of each quantity carried each claim
            regime. This is the measurement handover.md keeps citing (about
            2% of real reports carry a readable regime marker) rendered per
            run rather than only in a one-off script.
        unlabelled_weight: The rhetorical weight ``model.py`` applied to every
            unlabelled report this run -- a judgement call, not a fitted
            quantity (handover.md §19.9), and worth carrying on every report
            so a published figure is traceable to it.
    """

    computed_at: str
    counts: dict[str, int]
    run_ids: dict[str, int | None]
    diagnostics: dict[str, dict[str, Any] | None]
    regime_counts: dict[str, dict[str, int]]
    unlabelled_weight: float

    def as_dict(self) -> dict[str, Any]:
        """Render as a JSON-serialisable mapping.

        Returns:
            The report's fields, in a plain dict.
        """
        return {
            "computed_at": self.computed_at,
            "counts": self.counts,
            "run_ids": self.run_ids,
            "diagnostics": self.diagnostics,
            "regime_counts": self.regime_counts,
            "unlabelled_weight": self.unlabelled_weight,
        }


def _regime_counts(reports: list[Report]) -> dict[str, dict[str, int]]:
    """Count reports by quantity and claim regime.

    Args:
        reports: Every report the loader produced this run, troops and
            casualties together.

    Returns:
        ``{quantity: {regime: count}}``, quantities and regimes present in
        ``reports`` only.
    """
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    for report in reports:
        counters[report.quantity][report.claim_regime] += 1
    return {quantity: dict(counter) for quantity, counter in counters.items()}


def build_report(
    *,
    counts: ReconcileCounts,
    reports: list[Report],
    diagnostics_by_quantity: dict[str, dict[str, Any] | None],
    run_ids: dict[str, int | None],
    unlabelled_weight: float,
    computed_at: datetime,
) -> ReconcileReport:
    """Assemble one run's report from what the stage did.

    Args:
        counts: The run's accumulated counters.
        reports: Every report the loader produced this run.
        diagnostics_by_quantity: Each fitted quantity's diagnostics payload,
            keyed by quantity; a skipped quantity maps to ``None``.
        run_ids: Each quantity's ``model_runs.run_id``, or ``None`` for a
            skipped quantity.
        unlabelled_weight: The rhetorical weight applied to unlabelled
            reports this run.
        computed_at: When this run finished.

    Returns:
        The report.
    """
    return ReconcileReport(
        computed_at=computed_at.isoformat(),
        counts=counts.as_dict(),
        run_ids=run_ids,
        diagnostics=diagnostics_by_quantity,
        regime_counts=_regime_counts(reports),
        unlabelled_weight=unlabelled_weight,
    )


def write_report(report: ReconcileReport, *, processed_root: Path) -> Path:
    """Write the report to ``<processed_root>/reconciliation_report.json``.

    Args:
        report: The report to write.
        processed_root: The directory to write into -- ``params.processed_root``
            from the agent spec in a real run, and always a ``tmp_path`` in a
            test, never the project's real ``data/processed``.

    Returns:
        The path written.
    """
    processed_root.mkdir(parents=True, exist_ok=True)
    path = processed_root / REPORT_FILENAME
    path.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
    logger.info("reconcile_report_written", path=str(path))
    return path
