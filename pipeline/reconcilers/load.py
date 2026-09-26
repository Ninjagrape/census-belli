"""Read troop and casualty reports into :class:`Report` records.

This is the database-facing half of the reconcile stage. Everything in
:mod:`pipeline.reconcilers.records` and :mod:`pipeline.reconcilers.claims` is
pure -- it takes :class:`~pipeline.reconcilers.records.Report` instances and
returns plain values -- and this module is what builds those instances from
real rows, and what decides which sides the model downstream can even attempt.

Three things here are load-bearing.

**`battles.year_astronomical`, never `battles.date_start`.** `datetime.date`
has ``MINYEAR == 1``, so a BC ``date_start`` cannot be constructed or decoded
from Python at all, and this corpus is heavily ancient (handover.md §5.1).
`year_astronomical` is the generated, readable projection, joined in on every
report so :func:`pipeline.reconcilers.claims.era_flag` has something to work
with.

**Exclusions are counted, never silently dropped.** `battle_sides.est_troops_total`
is personnel only: it has no per-branch column, so a cavalry count or a ship
count has nowhere honest to go, and a non-positive number is not a
measurement of anything. Each exclusion increments a dedicated
:class:`~pipeline.reconcilers.records.ReconcileCounts` field rather than just
vanishing from the row count, because a corpus that quietly lost half its
troop reports would still pass every gate that only checks the reports that
made it through.

**A re-run must converge without deleting the extract stage's rows.**
`missing_data_log` under ``field_name='troop_total'`` is written by two
stages: extract, for a side with *no* reports at all, and this module, for a
side that has reports but none usable. Both use the same field name, because
``agents/classify.yaml``'s ``missing_data_all_logged`` gate reads that one
name. Clearing before a re-run must therefore be scoped to rows this module
itself wrote -- identified by the ``'reconcile: '`` prefix on ``notes`` -- or
a reconcile re-run would erase extract's half of the log every time.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Final

import structlog
from sqlalchemy import text

from pipeline.reconcilers.claims import assign_lineages, classify_regime, roundness
from pipeline.reconcilers.records import Quantity, ReconcileCounts, Report, SourceKey

__all__ = [
    "UnfillableSide",
    "classify_sides",
    "load_reports",
    "write_unfillable",
]

logger = structlog.get_logger()

# est_troops_total is personnel only (config/schema.sql), and there is no
# per-branch estimate column to put a non-total figure in. A single infobox
# strength field ("40,000 cavalry and 1,000,000 infantry") extracts as two
# troop_reports rows, and each would be a wrong observation of the side's
# total if it were let through.
_TOTAL_BRANCH: Final[str] = "total"

# Ship counts carry no unit compatible with a head count -- a real infobox
# reads "250-400 galleys" -- so a naval row is excluded and counted on its
# own, not folded into the general non-total-branch count. It is also
# non-total, so it must never increment both counters (see load_reports).
_NAVAL_BRANCH: Final[str] = "naval"

_FIELD_NAME_BY_QUANTITY: Final[dict[Quantity, str]] = {
    "troops": "troop_total",
    "casualties": "casualties",
}

# Load-bearing prefix: agents/classify.yaml's missing_data_all_logged gate
# requires field_name='troop_total' rows, and extract writes those too (for a
# side with no reports at all). Only rows carrying this prefix are ours to
# clear on a re-run.
_RECONCILE_NOTE_PREFIX: Final[str] = "reconcile: "

_LOAD_TROOP_REPORTS = text(
    """
    SELECT
        tr.report_id, tr.side_id, bs.battle_id, tr.source_id,
        s.source_type, s.url,
        tr.branch, tr.reported_value, tr.scope,
        tr.is_estimate, tr.is_upper_bound, tr.is_lower_bound,
        tr.extracted_context, b.year_astronomical
    FROM troop_reports tr
    JOIN battle_sides bs ON bs.side_id = tr.side_id
    JOIN battles b       ON b.battle_id = bs.battle_id
    JOIN sources s       ON s.source_id = tr.source_id
    """
)

_LOAD_CASUALTY_REPORTS = text(
    """
    SELECT
        cr.report_id, cr.side_id, bs.battle_id, cr.source_id,
        s.source_type, s.url,
        cr.casualty_type, cr.reported_value, cr.is_estimate,
        cr.extracted_context, b.year_astronomical
    FROM casualty_reports cr
    JOIN battle_sides bs ON bs.side_id = cr.side_id
    JOIN battles b       ON b.battle_id = bs.battle_id
    JOIN sources s       ON s.source_id = cr.source_id
    """
)

# Every troop_reports row, not just the usable ones -- classify_sides needs
# to see what a side's un-usable rows actually were, to say why.
_LOAD_TROOP_SIDE_ROWS = text(
    """
    SELECT tr.side_id, bs.battle_id, tr.branch, tr.reported_value
    FROM troop_reports tr
    JOIN battle_sides bs ON bs.side_id = tr.side_id
    """
)

_CLEAR_RECONCILE_LOG = text(
    "DELETE FROM missing_data_log "
    "WHERE field_name = :field_name AND was_imputed = FALSE "
    "AND notes LIKE 'reconcile: %'"
)
_INSERT_RECONCILE_LOG = text(
    """
    INSERT INTO missing_data_log (battle_id, side_id, field_name, missingness_class, notes)
    VALUES (:battle_id, :side_id, :field_name, CAST(:missingness AS missingness_class), :notes)
    """
)


@dataclass(frozen=True)
class UnfillableSide:
    """A side with rows in ``troop_reports`` but not one this stage can use.

    Distinct from a side with *no* reports at all, which the extract stage
    already logs to ``missing_data_log`` (handover.md, agents/reconcile.yaml).
    This is the gap one stage further in: rows exist, and every one of them
    failed an exclusion rule.

    Attributes:
        side_id: The ``battle_sides.side_id`` this concerns.
        battle_id: Its battle, so :func:`write_unfillable` can insert a
            ``missing_data_log`` row without a second lookup.
        quantity: ``"troops"`` or ``"casualties"`` -- which estimate this side
            cannot get. Only ``"troops"`` sides are produced today; casualty
            reports have no exclusion rule, so a side with any casualty row at
            all is always modellable for casualties (see :func:`classify_sides`).
        reason: A short, human-readable explanation, one of
            ``"only non-total branch reports"``, ``"only naval counts"``, or
            ``"no positive value"``.
    """

    side_id: int
    battle_id: int
    quantity: Quantity
    reason: str


def _classify_troop_report(
    branch: str, reported_value: float, counts: ReconcileCounts
) -> bool:
    """Apply the troop-report exclusion rules and count the outcome.

    Args:
        branch: The report's ``troop_branch`` value.
        reported_value: The report's raw value.
        counts: Counters to accumulate the exclusion into.

    Returns:
        True if the report is usable, False if it was excluded (and counted).
    """
    if branch == _NAVAL_BRANCH:
        # Naval rows are also non-total; count them once, as naval, never
        # again as a generic non-total exclusion.
        counts.excluded_naval += 1
        return False
    if branch != _TOTAL_BRANCH:
        counts.excluded_non_total_branch += 1
        return False
    if reported_value <= 0:
        counts.excluded_non_positive += 1
        return False
    return True


def _build_troop_reports(conn: Any, counts: ReconcileCounts) -> list[Report]:
    """Load, filter and classify every troop report.

    Args:
        conn: An open database connection.
        counts: Counters to accumulate into.

    Returns:
        Usable troop :class:`Report` instances, with ``claim_regime`` and
        ``roundness`` set but ``lineage_id`` still None.
    """
    reports: list[Report] = []
    for (
        report_id,
        side_id,
        battle_id,
        source_id,
        source_type,
        url,
        branch,
        reported_value,
        scope,
        is_estimate,
        is_upper_bound,
        is_lower_bound,
        extracted_context,
        year_astronomical,
    ) in conn.execute(_LOAD_TROOP_REPORTS).fetchall():
        counts.reports_read += 1
        value = float(reported_value)
        if not _classify_troop_report(str(branch), value, counts):
            continue

        regime = classify_regime(extracted_context, value=value)
        if regime == "unlabelled":
            counts.regime_unlabelled += 1

        reports.append(
            Report(
                report_id=int(report_id),
                side_id=int(side_id),
                battle_id=int(battle_id),
                source_id=int(source_id),
                source_key=SourceKey.from_row(source_type, url),
                source_type=source_type,
                quantity="troops",
                reported_value=value,
                branch=branch,
                scope=scope or "unknown",
                is_estimate=bool(is_estimate),
                is_upper_bound=bool(is_upper_bound),
                is_lower_bound=bool(is_lower_bound),
                extracted_context=extracted_context or "",
                year_astronomical=None if year_astronomical is None else int(year_astronomical),
                claim_regime=regime,
                roundness=roundness(value),
            )
        )
        counts.troop_reports_used += 1
    return reports


def _build_casualty_reports(conn: Any, counts: ReconcileCounts) -> list[Report]:
    """Load and classify every casualty report.

    Every ``casualty_reports`` row is kept: unlike troops, a zero is a
    meaningful observation ("this side took no casualties"), not the absence
    of one, and there is no branch or naval concept to exclude on.

    Args:
        conn: An open database connection.
        counts: Counters to accumulate into.

    Returns:
        Every casualty report as a :class:`Report`, with ``claim_regime`` and
        ``roundness`` set but ``lineage_id`` still None.
    """
    reports: list[Report] = []
    for (
        report_id,
        side_id,
        battle_id,
        source_id,
        source_type,
        url,
        casualty_type,
        reported_value,
        is_estimate,
        extracted_context,
        year_astronomical,
    ) in conn.execute(_LOAD_CASUALTY_REPORTS).fetchall():
        counts.reports_read += 1
        value = float(reported_value)

        regime = classify_regime(extracted_context, value=value)
        if regime == "unlabelled":
            counts.regime_unlabelled += 1

        reports.append(
            Report(
                report_id=int(report_id),
                side_id=int(side_id),
                battle_id=int(battle_id),
                source_id=int(source_id),
                source_key=SourceKey.from_row(source_type, url),
                source_type=source_type,
                quantity="casualties",
                reported_value=value,
                casualty_type=casualty_type,
                is_estimate=bool(is_estimate),
                extracted_context=extracted_context or "",
                year_astronomical=None if year_astronomical is None else int(year_astronomical),
                claim_regime=regime,
                roundness=roundness(value),
            )
        )
        counts.casualty_reports_used += 1
    return reports


def _with_lineages(reports: list[Report]) -> list[Report]:
    """Assign and write back lineage ids for one quantity's reports.

    :func:`~pipeline.reconcilers.claims.assign_lineages` keys its grouping by
    ``report_id``, which is only unique *within* one quantity --
    ``troop_reports`` and ``casualty_reports`` each have their own
    ``SERIAL`` sequence, so calling it once across both would silently
    collide report_id 1 of one table with report_id 1 of the other. Each
    quantity is therefore assigned separately.

    Args:
        reports: Every usable report of one quantity.

    Returns:
        The same reports, replaced with their assigned ``lineage_id``.
    """
    lineage_of_report = assign_lineages(reports)
    return [replace(report, lineage_id=lineage_of_report[report.report_id]) for report in reports]


def load_reports(conn: Any, *, ancient_cutoff_year: int, counts: ReconcileCounts) -> list[Report]:
    """Load every usable troop and casualty report as :class:`Report` records.

    ``ancient_cutoff_year`` is accepted for symmetry with the rest of the
    reconcile stage's config and is not applied here: this function is the
    deterministic loading and classification step,
    :func:`~pipeline.reconcilers.claims.era_flag` is what actually reads the
    cutoff, and it is called by the model-building step against each
    report's ``year_astronomical``, not here.

    Args:
        conn: An open database connection.
        ancient_cutoff_year: The astronomical year below which a battle
            counts as ancient. See the note above for why this function does
            not use it directly.
        counts: Counters to accumulate into. Populates ``reports_read``,
            ``troop_reports_used``, ``casualty_reports_used``,
            ``excluded_non_total_branch``, ``excluded_naval``,
            ``excluded_non_positive``, ``regime_unlabelled`` and ``lineages``.

    Returns:
        Every usable report, troops and casualties together, each with
        ``claim_regime``, ``roundness`` and ``lineage_id`` populated.
    """
    del ancient_cutoff_year  # see docstring

    troop_reports = _with_lineages(_build_troop_reports(conn, counts))
    casualty_reports = _with_lineages(_build_casualty_reports(conn, counts))

    counts.lineages = len({r.lineage_id for r in troop_reports}) + len(
        {r.lineage_id for r in casualty_reports}
    )

    logger.info(
        "reconcile_reports_loaded",
        troop_reports_used=counts.troop_reports_used,
        casualty_reports_used=counts.casualty_reports_used,
        excluded_non_total_branch=counts.excluded_non_total_branch,
        excluded_naval=counts.excluded_naval,
        excluded_non_positive=counts.excluded_non_positive,
    )

    return [*troop_reports, *casualty_reports]


def _unfillable_reason(rows: Sequence[tuple[str, float]]) -> str:
    """Explain why a side's troop reports produced no usable report.

    Checked in this order because "only naval counts" is the more specific,
    more actionable diagnosis: every row being naval is a stronger and rarer
    fact than every row merely being non-total.

    Args:
        rows: Every ``(branch, reported_value)`` pair for one side.

    Returns:
        One of ``"only naval counts"``, ``"only non-total branch reports"``,
        or ``"no positive value"``.
    """
    if all(branch == _NAVAL_BRANCH for branch, _ in rows):
        return "only naval counts"
    if all(branch != _TOTAL_BRANCH for branch, _ in rows):
        return "only non-total branch reports"
    return "no positive value"


def classify_sides(reports: Sequence[Report], conn: Any) -> tuple[set[int], list[UnfillableSide]]:
    """Decide which sides can be modelled, and explain the ones that cannot.

    A side is modellable if it has at least one usable report of any
    quantity, which ``reports`` -- already filtered by :func:`load_reports`
    -- directly gives us. The harder half is the sides that are *not* in
    there: a side with troop_reports rows all of which were excluded reads
    identically, from ``reports`` alone, to a side with no rows at all. Only
    a second look at the raw rows, via ``conn``, can tell those apart and say
    why.

    Casualty reports have no exclusion rule (see
    :func:`_build_casualty_reports`), so a side can never be unfillable for
    casualties under today's rules; only troop sides are produced here.

    Args:
        reports: Every usable report, as returned by :func:`load_reports`.
        conn: An open database connection, used to re-read the raw
            ``troop_reports`` rows for sides that produced no usable report.

    Returns:
        A tuple of (modellable side ids, unfillable sides with reasons).
    """
    modellable = {report.side_id for report in reports}

    rows_by_side: dict[int, list[tuple[str, float]]] = defaultdict(list)
    battle_by_side: dict[int, int] = {}
    for side_id, battle_id, branch, reported_value in conn.execute(
        _LOAD_TROOP_SIDE_ROWS
    ).fetchall():
        side_id = int(side_id)
        battle_by_side[side_id] = int(battle_id)
        rows_by_side[side_id].append((str(branch), float(reported_value)))

    unfillable: list[UnfillableSide] = []
    for side_id, rows in rows_by_side.items():
        if side_id in modellable:
            continue
        unfillable.append(
            UnfillableSide(
                side_id=side_id,
                battle_id=battle_by_side[side_id],
                quantity="troops",
                reason=_unfillable_reason(rows),
            )
        )

    return modellable, unfillable


def write_unfillable(
    conn: Any,
    unfillable: Sequence[UnfillableSide],
    counts: ReconcileCounts,
    *,
    clear_first: bool = True,
) -> None:
    """Record every unfillable side to ``missing_data_log``.

    Args:
        conn: An open database connection.
        unfillable: Sides to record, from :func:`classify_sides`.
        counts: Counters to accumulate into. Increments ``sides_unfillable``.
        clear_first: Whether to delete this stage's previous rows before
            writing. True on a normal run, so a re-run converges rather than
            accumulating duplicates. Scoped to rows carrying the
            ``'reconcile: '`` notes prefix, so it cannot delete the rows the
            extract stage writes under the same ``field_name`` for a side
            with no reports at all -- see the module docstring.

    Note:
        The clearing DELETE runs once per distinct ``field_name`` actually
        present in ``unfillable`` plus every field name this stage can ever
        write, so a re-run that fixes every casualty-side gap (there are
        none today, but the field exists) still clears a stale casualties row
        left by an earlier run even when this run produces none.
    """
    if clear_first:
        for field_name in _FIELD_NAME_BY_QUANTITY.values():
            conn.execute(_CLEAR_RECONCILE_LOG, {"field_name": field_name})

    for side in unfillable:
        conn.execute(
            _INSERT_RECONCILE_LOG,
            {
                "battle_id": side.battle_id,
                "side_id": side.side_id,
                "field_name": _FIELD_NAME_BY_QUANTITY[side.quantity],
                "missingness": "unclassified",
                "notes": f"{_RECONCILE_NOTE_PREFIX}{side.reason}",
            },
        )
        counts.sides_unfillable += 1
