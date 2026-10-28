"""Read troop and casualty reports into :class:`Report` records.

This is the database-facing half of the reconcile stage. Everything in
:mod:`pipeline.reconcilers.records` and :mod:`pipeline.reconcilers.claims` is
pure -- it takes :class:`~pipeline.reconcilers.records.Report` instances and
returns plain values -- and this module is what builds those instances from
real rows, and what decides which sides the model downstream can even attempt.

Three things here are load-bearing.

**`battles.year_astronomical`, never `battles.date_start`.** `datetime.date`
has ``MINYEAR == 1``, so a BC ``date_start`` cannot be constructed or decoded
from Python at all, and this corpus is heavily ancient (docs/handover-log.md §5.1).
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

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Final

import structlog
from sqlalchemy import text

from pipeline.reconcilers.claims import assign_lineages, classify_regime, roundness
from pipeline.reconcilers.plausibility import (
    exclusion_reason,
    filter_troop_reports,
    is_override_report,
)
from pipeline.reconcilers.records import Quantity, ReconcileCounts, Report, SourceKey

__all__ = [
    "DEFAULT_MIN_PLAUSIBLE_TROOPS",
    "SPAN_ERROR_LOG_RATIO",
    "SPAN_WARN_LOG_RATIO",
    "SpanFinding",
    "UnfillableSide",
    "classify_sides",
    "load_reports",
    "troop_span_findings",
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
_POPULATION_SCOPE: Final[str] = "population"
_MODERN_REGIME: Final[str] = "modern_scholarly"

# The default for params.min_plausible_troops. No battle side fields fewer than
# a hundred combatants in this corpus's sources; below that the figure is a day
# number, a count of something else or a fragment of a date.
DEFAULT_MIN_PLAUSIBLE_TROOPS: Final[float] = 100.0

# Span guard bounds (params.troop_span_warn_log / troop_span_error_log), in log
# units: ln(max / min) of a side's kept point totals. e (ratio 2.72) is more
# disagreement than one claim regime's noise explains; e squared (7.4) is a
# scope or unit mix-up until a curator says otherwise.
SPAN_WARN_LOG_RATIO: Final[float] = 1.0
SPAN_ERROR_LOG_RATIO: Final[float] = 2.0

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
    SELECT tr.side_id, bs.battle_id, tr.branch, tr.reported_value, tr.extracted_context,
           tr.scope
    FROM troop_reports tr
    JOIN battle_sides bs ON bs.side_id = tr.side_id
    """
)

_LOAD_CASUALTY_SIDE_ROWS = text(
    """
    SELECT DISTINCT cr.side_id, bs.battle_id
    FROM casualty_reports cr
    JOIN battle_sides bs ON bs.side_id = cr.side_id
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
    already logs to ``missing_data_log`` (docs/handover-log.md, agents/reconcile.yaml).
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
            ``"only non-total branch reports"``, ``"only naval counts"``,
            ``"only implausible or non-personnel figures"`` or
            ``"no positive value"``.
    """

    side_id: int
    battle_id: int
    quantity: Quantity
    reason: str


def _classify_troop_report(
    branch: str, reported_value: float, scope: str | None, counts: ReconcileCounts
) -> bool:
    """Apply the troop-report exclusion rules and count the outcome.

    Args:
        branch: The report's ``troop_branch`` value.
        reported_value: The report's raw value.
        scope: The report's ``scope``. ``population`` counts people (a nation
            on the move, its non-combatants), not an army, and is not a level
            of the force-size hierarchy the scope offsets model.
        counts: Counters to accumulate the exclusion into.

    Returns:
        True if the report is usable, False if it was excluded (and counted).
    """
    if branch == _NAVAL_BRANCH:
        # Naval rows are also non-total; count them once, as naval, never
        # again as a generic non-total exclusion.
        counts.excluded_naval += 1
        return False
    if scope == _POPULATION_SCOPE:
        # Counted before the branch rule: a "368,000 people" row is stored as a
        # component, and calling it a non-total would hide why it was dropped.
        counts.excluded_population_scope += 1
        return False
    if branch != _TOTAL_BRANCH:
        counts.excluded_non_total_branch += 1
        return False
    if reported_value <= 0:
        counts.excluded_non_positive += 1
        return False
    return True


def _sides_with_modern_estimate(reports: list[Report]) -> set[int]:
    """Sides holding a modern point estimate or a modern lower and upper pair.

    A lone modern bound ("up to 20,000 (modern estimates)") is a ceiling, not
    an estimate, and must not displace a point figure. A pair counts when its
    two bounds share one context, i.e. one stated range.

    Args:
        reports: Troop reports with ``claim_regime`` set.

    Returns:
        The ids of the sides on which a modern estimate stands.
    """
    sides: set[int] = set()
    lowers: set[tuple[int, str]] = set()
    uppers: set[tuple[int, str]] = set()
    for report in reports:
        if report.claim_regime != _MODERN_REGIME:
            continue
        key = (report.side_id, report.extracted_context)
        if report.is_lower_bound:
            lowers.add(key)
        if report.is_upper_bound:
            uppers.add(key)
        if not report.is_lower_bound and not report.is_upper_bound:
            sides.add(report.side_id)
    sides.update(side_id for side_id, _ in lowers & uppers)
    return sides


def drop_superseded_by_modern(reports: list[Report], counts: ReconcileCounts) -> list[Report]:
    """Drop every non-modern total on a side that has a modern-scholarly estimate.

    A modern estimate is the model's zero-bias anchor, and the reason an
    infobox prints one beside an ancient figure is that the ancient figure is
    the inflated claim. Fitting both lets the ancient number drag the side's
    estimate back up, so on such a side the other totals (ancient, chronicle,
    unlabelled, and a derived lower-bound sum alike) are excluded and counted.
    The rows stay in ``troop_reports``.

    Args:
        reports: Troop reports that already passed the branch, scope and
            plausibility rules, each with ``claim_regime`` set.
        counts: Counters; increments ``excluded_superseded_by_modern``.

    Returns:
        The reports in their original order, minus the superseded ones.
    """
    modern_sides = _sides_with_modern_estimate(reports)
    kept: list[Report] = []
    for report in reports:
        if report.side_id in modern_sides and report.claim_regime != _MODERN_REGIME:
            counts.excluded_superseded_by_modern += 1
            continue
        kept.append(report)
    return kept


@dataclass(frozen=True)
class SpanFinding:
    """A side whose kept point totals disagree by more than the guard allows.

    Attributes:
        side_id: The ``battle_sides.side_id``.
        battle_id: Its battle.
        min_value: The smallest kept point total.
        max_value: The largest kept point total.
        log_span: ``ln(max_value / min_value)``.
        severity: ``"warning"`` above the warn bound, ``"error"`` above the
            error bound.
    """

    side_id: int
    battle_id: int
    min_value: float
    max_value: float
    log_span: float
    severity: str


def troop_span_findings(
    reports: Sequence[Report],
    *,
    warn_log_span: float = SPAN_WARN_LOG_RATIO,
    error_log_span: float = SPAN_ERROR_LOG_RATIO,
) -> list[SpanFinding]:
    """Flag sides whose kept troop point totals span more than the bounds.

    Runs after every load filter, supersede included, so it judges what the
    model will see. Only point rows count: a bound is one-sided by design. A
    side holding a ``troop_strength`` override row is skipped, because a
    curator has settled it.

    Args:
        reports: The loaded reports; casualty rows are ignored.
        warn_log_span: Flag a side when ``ln(max / min)`` exceeds this.
        error_log_span: Make it an error when it exceeds this.

    Returns:
        One finding per flagged side, in side id order.
    """
    points: dict[int, list[Report]] = defaultdict(list)
    overridden: set[int] = set()
    for report in reports:
        if report.quantity != "troops":
            continue
        # Keyed on the override url, not the manual_entry source type: other hand
        # entries have not had their disagreement settled by a curator.
        if is_override_report(report):
            overridden.add(report.side_id)
        if not report.is_lower_bound and not report.is_upper_bound:
            points[report.side_id].append(report)

    findings: list[SpanFinding] = []
    for side_id in sorted(points.keys() - overridden):
        rows = points[side_id]
        low = min(r.reported_value for r in rows)
        high = max(r.reported_value for r in rows)
        span = math.log(high / low)
        if span <= warn_log_span:
            continue
        findings.append(
            SpanFinding(
                side_id=side_id,
                battle_id=rows[0].battle_id,
                min_value=low,
                max_value=high,
                log_span=span,
                severity="error" if span > error_log_span else "warning",
            )
        )
    return findings


_LOAD_SIDE_NAMES = text(
    "SELECT bs.side_id, COALESCE(b.source_slug, b.name), bs.side_label "
    "FROM battle_sides bs JOIN battles b ON b.battle_id = bs.battle_id "
    "WHERE bs.side_id = ANY(CAST(:ids AS INT[]))"
)


def _record_span_findings(conn: Any, findings: list[SpanFinding], counts: ReconcileCounts) -> None:
    """Log each finding, count it and keep it, named, for the stage diagnostics."""
    if not findings:
        return
    names = {
        int(side_id): (str(slug), str(label))
        for side_id, slug, label in conn.execute(
            _LOAD_SIDE_NAMES, {"ids": [f.side_id for f in findings]}
        ).fetchall()
    }
    for finding in findings:
        battle, side = names.get(finding.side_id, ("", ""))
        log = logger.error if finding.severity == "error" else logger.warning
        log(
            "reconcile_troop_span",
            battle=battle,
            side=side,
            severity=finding.severity,
            min=finding.min_value,
            max=finding.max_value,
        )
        if finding.severity == "error":
            counts.troop_span_errors += 1
        else:
            counts.troop_span_warnings += 1
        counts.span_findings.append(
            {
                "battle": battle,
                "side": side,
                "severity": finding.severity,
                "min": finding.min_value,
                "max": finding.max_value,
                "log_span": finding.log_span,
            }
        )


def _build_troop_reports(
    conn: Any, counts: ReconcileCounts, *, min_plausible_troops: float
) -> list[Report]:
    """Load, filter and classify every troop report.

    Args:
        conn: An open database connection.
        counts: Counters to accumulate into.
        min_plausible_troops: Floor below which a value is not an army's
            strength; see :mod:`pipeline.reconcilers.plausibility`.

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
        if not _classify_troop_report(str(branch), value, scope, counts):
            continue

        regime = classify_regime(extracted_context, value=value)

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

    # After the per-row rules and the component rule, so every counter below
    # describes the reports the model will actually see.
    reports = filter_troop_reports(reports, counts, min_plausible_troops=min_plausible_troops)
    reports = drop_superseded_by_modern(reports, counts)
    counts.troop_reports_used += len(reports)
    counts.regime_unlabelled += sum(1 for r in reports if r.claim_regime == "unlabelled")
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


def load_reports(
    conn: Any,
    *,
    ancient_cutoff_year: int,
    counts: ReconcileCounts,
    min_plausible_troops: float = DEFAULT_MIN_PLAUSIBLE_TROOPS,
    span_warn_log: float = SPAN_WARN_LOG_RATIO,
    span_error_log: float = SPAN_ERROR_LOG_RATIO,
) -> list[Report]:
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
            ``excluded_population_scope``, ``excluded_superseded_by_modern``,
            ``excluded_non_positive``, the four plausibility counters
            (``excluded_implausible_small``, ``excluded_non_personnel``,
            ``excluded_date_context``, ``excluded_component_of_stated_total``),
            ``regime_unlabelled``, ``lineages`` and the span guard's
            ``troop_span_warnings``, ``troop_span_errors`` and ``span_findings``.
        min_plausible_troops: Troop values below this are excluded as not an
            army's strength (``params.min_plausible_troops``).
        span_warn_log: Span guard warning bound (``params.troop_span_warn_log``).
        span_error_log: Span guard error bound (``params.troop_span_error_log``).

    Returns:
        Every usable report, troops and casualties together, each with
        ``claim_regime``, ``roundness`` and ``lineage_id`` populated.
    """
    del ancient_cutoff_year  # see docstring

    kept_troops = _build_troop_reports(conn, counts, min_plausible_troops=min_plausible_troops)
    _record_span_findings(
        conn,
        troop_span_findings(
            kept_troops, warn_log_span=span_warn_log, error_log_span=span_error_log
        ),
        counts,
    )
    troop_reports = _with_lineages(kept_troops)
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
        excluded_population_scope=counts.excluded_population_scope,
        excluded_superseded_by_modern=counts.excluded_superseded_by_modern,
        excluded_non_positive=counts.excluded_non_positive,
        excluded_implausible_small=counts.excluded_implausible_small,
        excluded_non_personnel=counts.excluded_non_personnel,
        excluded_date_context=counts.excluded_date_context,
        excluded_component_of_stated_total=counts.excluded_component_of_stated_total,
    )

    return [*troop_reports, *casualty_reports]


def _unfillable_reason(
    rows: Sequence[tuple[str, float, str, str]], *, min_plausible_troops: float
) -> str:
    """Explain why a side's troop reports produced no usable report.

    Checked in this order because "only naval counts" is the more specific,
    more actionable diagnosis: every row being naval is a stronger and rarer
    fact than every row merely being non-total.

    Args:
        rows: Every ``(branch, reported_value, extracted_context, scope)``
            tuple for one side.
        min_plausible_troops: The plausibility floor the loader applied.

    Returns:
        One of ``"only naval counts"``, ``"only non-total branch reports"``,
        ``"only population-scope totals"``,
        ``"only implausible or non-personnel figures"`` (positive totals that
        every plausibility rule rejected), or ``"no positive value"``.
    """
    if all(branch == _NAVAL_BRANCH for branch, _, _, _ in rows):
        return "only naval counts"
    if all(branch != _TOTAL_BRANCH for branch, _, _, _ in rows):
        return "only non-total branch reports"
    if all(scope == _POPULATION_SCOPE for branch, _, _, scope in rows if branch == _TOTAL_BRANCH):
        return "only population-scope totals"
    if any(
        branch == _TOTAL_BRANCH
        and value > 0
        and exclusion_reason(value, context, min_plausible_troops=min_plausible_troops) is not None
        for branch, value, context, _ in rows
    ):
        return "only implausible or non-personnel figures"
    return "no positive value"


def classify_sides(
    reports: Sequence[Report],
    conn: Any,
    *,
    min_plausible_troops: float = DEFAULT_MIN_PLAUSIBLE_TROOPS,
) -> tuple[set[int], list[UnfillableSide]]:
    """Decide which sides can be modelled, and explain the ones that cannot.

    Each quantity is judged on its own rows. A side whose troop rows were all
    filtered is unfillable for troops even when it has casualty reports: judging
    "modellable" across quantities left such sides unlogged and, on a re-run,
    holding the troop estimate an earlier fit had given them. ``reports`` --
    already filtered by :func:`load_reports` -- says which sides are modellable;
    a side with raw rows that is *not* in there reads identically to a side with
    no rows at all unless the raw rows are re-read via ``conn``, which also
    gives the reason.

    Casualty reports have no exclusion rule (see
    :func:`_build_casualty_reports`), so a casualty side is unfillable only if
    its rows vanished between the two reads; the check is kept so the log
    stays honest if a rule is added.

    Args:
        reports: Every usable report, as returned by :func:`load_reports`.
        conn: An open database connection, used to re-read the raw
            ``troop_reports`` rows for sides that produced no usable report.
        min_plausible_troops: The floor the loader applied, so the reason given
            for a side agrees with why its rows were dropped.

    Returns:
        A tuple of (side ids modellable for troops, unfillable sides of both
        quantities with reasons).
    """
    modellable = {report.side_id for report in reports if report.quantity == "troops"}
    casualty_modellable = {report.side_id for report in reports if report.quantity == "casualties"}

    rows_by_side: dict[int, list[tuple[str, float, str, str]]] = defaultdict(list)
    battle_by_side: dict[int, int] = {}
    for side_id, battle_id, branch, reported_value, context, scope in conn.execute(
        _LOAD_TROOP_SIDE_ROWS
    ).fetchall():
        side_id = int(side_id)
        battle_by_side[side_id] = int(battle_id)
        rows_by_side[side_id].append(
            (str(branch), float(reported_value), context or "", str(scope))
        )

    unfillable: list[UnfillableSide] = []
    for side_id, rows in rows_by_side.items():
        if side_id in modellable:
            continue
        unfillable.append(
            UnfillableSide(
                side_id=side_id,
                battle_id=battle_by_side[side_id],
                quantity="troops",
                reason=_unfillable_reason(rows, min_plausible_troops=min_plausible_troops),
            )
        )

    for side_id, battle_id in conn.execute(_LOAD_CASUALTY_SIDE_ROWS).fetchall():
        if int(side_id) not in casualty_modellable:
            unfillable.append(
                UnfillableSide(
                    side_id=int(side_id),
                    battle_id=int(battle_id),
                    quantity="casualties",
                    reason="no usable casualty report",
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
