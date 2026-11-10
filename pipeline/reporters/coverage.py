"""Data-coverage numbers behind the report's disclosures, queried at report time.

Every figure is a read-only SQLAlchemy Core query over the reported run's
credited battles (the battles that have ``battle_war_details`` rows for it).
A quantity that cannot be determined, for example because a table is empty,
comes back as ``None`` or an empty tuple and is logged; the sentences built
from it are simply left out, and the stage never fails on it.

Nothing here is hard-coded from a particular run. The era cut points are
the only constants, and the sentences that use them name them.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Final

import structlog
from sqlalchemy import text
from sqlalchemy.engine import Connection
from sqlalchemy.sql.elements import TextClause

from pipeline.reporters.coverage_series import (
    CREDITED_BATTLES,
    ONE_COMMANDER_ROW,
    ComponentMarker,
    CreditSource,
    ReportedPoint,
    TimelineBin,
    guarded,
    load_credit_source,
    load_reported_vs_reconciled,
    load_sigma_skill,
    load_timeline,
)
from pipeline.reporters.records import RubinInfo
from pipeline.reporters.tables import RankedGeneral

__all__ = [
    "REPORT_ERAS",
    "TOP_FOR_COMPONENTS",
    "CoverageStats",
    "EraCount",
    "FameBin",
    "TroopEra",
    "era_label",
    "load_coverage",
    "with_component_counts",
    "with_component_sizes",
]

logger = structlog.get_logger()

TOP_FOR_COMPONENTS: Final[int] = 50
CENTURY: Final[int] = 100
TOP_MARKERS: Final[int] = 15
# Report eras for the fame-bias and troop-provenance disclosures: (first year of the era, label),
# astronomical years as on battles.year_astronomical. They are not the reconcile model's era
# bins (ERA_BIN_LABELS: 500/1500/1800/1900), which exist for bias priors, not for readers.
REPORT_ERAS: Final[tuple[tuple[int, str], ...]] = (
    (500, "medieval"),
    (1500, "early modern"),
    (1789, "Revolutionary and Napoleonic"),
    (1816, "later nineteenth century"),
    (1914, "twentieth century and later"),
)
ANCIENT: Final[str] = "ancient"
ERA_ORDER: Final[tuple[str, ...]] = (ANCIENT, *(label for _, label in REPORT_ERAS))
# (lowest credited-battle count, label); the last bin is open-ended.
FAME_BINS: Final[tuple[tuple[int, str], ...]] = (
    (10, "10+"),
    (5, "5-9"),
    (3, "3-4"),
    (2, "2"),
    (1, "1"),
)

_CREDITED = CREDITED_BATTLES

_RULE_SINGLE = text(
    "SELECT COUNT(DISTINCT bc.side_id) FILTER (WHERE bc.attribution_method = 'rule_single'), "
    "COUNT(DISTINCT bc.side_id) "
    f"FROM battle_war_details d JOIN {ONE_COMMANDER_ROW} bc "
    "ON bc.battle_id = d.battle_id AND bc.general_id = d.general_id "
    "WHERE d.run_id = :run_id"
)
_N_COMPONENTS = text(
    "SELECT CAST(diagnostics -> 'components' ->> 'n_components' AS INT) "
    "FROM model_runs WHERE run_id = :run_id"
)
_EXTRACT_OK = text("SELECT COUNT(*) FROM llm_calls WHERE stage = 'extract' AND status = 'ok'")
_CENTURIES = text(
    "SELECT CAST(FLOOR(b.year_astronomical / 100.0) * 100 AS INT) AS century, COUNT(*) "
    f"FROM battles b WHERE b.battle_id IN ({_CREDITED}) AND b.year_astronomical IS NOT NULL "
    "GROUP BY 1 ORDER BY 1"
)
_YEAR_SPAN = text(
    f"SELECT MIN(b.year_astronomical), MAX(b.year_astronomical) FROM battles b "
    f"WHERE b.battle_id IN ({_CREDITED}) AND b.year_astronomical IS NOT NULL"
)
_FIRST_YEARS = text(
    "SELECT MIN(b.year_astronomical) FROM general_skill_estimates e "
    "JOIN battle_war_details d ON d.run_id = e.run_id AND d.general_id = e.general_id "
    "JOIN battles b ON b.battle_id = d.battle_id "
    "WHERE e.run_id = :run_id AND e.n_battles >= :min_battles "
    "AND b.year_astronomical IS NOT NULL GROUP BY e.general_id"
)
_FAME = text(
    "SELECT CASE WHEN e.n_battles >= 10 THEN 10 WHEN e.n_battles >= 5 THEN 5 "
    "WHEN e.n_battles >= 3 THEN 3 WHEN e.n_battles >= 2 THEN 2 ELSE 1 END AS bin, "
    "COUNT(*), COUNT(*) FILTER (WHERE d.actual_outcome >= 1.0) "
    "FROM general_skill_estimates e "
    "JOIN battle_war_details d ON d.run_id = e.run_id AND d.general_id = e.general_id "
    f"JOIN {ONE_COMMANDER_ROW} bc ON bc.battle_id = d.battle_id AND bc.general_id = d.general_id "
    "WHERE e.run_id = :run_id AND bc.hierarchy_rank = 0 AND d.actual_outcome IS NOT NULL "
    "GROUP BY 1"
)
_TROOPS = text(
    "SELECT b.year_astronomical, s.est_troops_method, s.est_troops_total IS NOT NULL "
    "FROM battle_sides s JOIN battles b ON b.battle_id = s.battle_id "
    f"WHERE s.battle_id IN ({_CREDITED}) AND b.year_astronomical IS NOT NULL"
)
_GENERAL_COUNTS = text(
    "SELECT COUNT(*), COUNT(*) FILTER (WHERE n_battles >= :min_battles) "
    "FROM general_skill_estimates WHERE run_id = :run_id"
)


@dataclass(frozen=True)
class EraCount:
    """Generals whose first credited battle falls in one era."""

    era: str
    n: int


@dataclass(frozen=True)
class FameBin:
    """Credited battles of top commanders (hierarchy rank 0) by their general's battle count.

    Attributes:
        label: Human label of the bin, for example ``5 or more``.
        n_credited: Credited battles in the bin where the general led the side.
        n_wins: Of those, battles the side won outright (actual outcome 1).
    """

    label: str
    n_credited: int
    n_wins: int


@dataclass(frozen=True)
class TroopEra:
    """Troop-figure provenance for the sides of credited battles in one era.

    Attributes:
        era: The era label.
        multi_source: Sides with a total reconciled from two or more source lineages.
        single_report: Sides with a total resting on one report.
        imputed: Sides with no reconciled total, filled by imputation.
        other: Sides with a total whose reconcile method is neither of the two above.
        bound_only: Sides whose total is only an upper or lower bound ("up to" or "at least").
    """

    era: str
    multi_source: int
    single_report: int
    imputed: int
    other: int = 0
    bound_only: int = 0

    @property
    def total(self) -> int:
        """All sides in the era."""
        return self.multi_source + self.single_report + self.imputed + self.other + self.bound_only

    @property
    def weak(self) -> int:
        """Sides without a multi-source total."""
        return self.single_report + self.imputed + self.bound_only


@dataclass(frozen=True)
class CoverageStats:
    """The numbers the coverage disclosures quote; ``None`` or empty means undetermined."""

    rule_single_sides: int | None = None
    credited_sides: int | None = None
    n_components: int | None = None
    top_outside_largest: int | None = None
    top_component_known: int | None = None
    extract_answers: int | None = None
    corpus_battles: int | None = None
    first_century: int | None = None
    last_century: int | None = None
    first_year: int | None = None
    last_year: int | None = None
    empty_spans: tuple[tuple[int, int], ...] = ()
    n_headline: int | None = None
    n_generals_total: int | None = None
    headline_by_era: tuple[EraCount, ...] = ()
    fame_bins: tuple[FameBin, ...] = ()
    troop_eras: tuple[TroopEra, ...] = ()
    timeline: tuple[TimelineBin, ...] = ()
    credit_source: tuple[CreditSource, ...] = ()
    sigma_skill: float | None = None
    reported_vs_reconciled: tuple[ReportedPoint, ...] = ()
    component_sizes: tuple[int, ...] = ()
    top_markers: tuple[ComponentMarker, ...] = ()


def era_label(year: int) -> str:
    """The era an astronomical year falls in.

    Args:
        year: Astronomical year.

    Returns:
        One of :data:`ERA_ORDER`.
    """
    label = ANCIENT
    for start, name in REPORT_ERAS:
        if year >= start:
            label = name
    return label


def _empty_spans(counts: dict[int, int]) -> tuple[tuple[int, int], ...]:
    """Runs of centuries between the first and last occupied one that hold no battle."""
    if not counts:
        return ()
    spans: list[tuple[int, int]] = []
    run_start: int | None = None
    for century in range(min(counts), max(counts) + CENTURY, CENTURY):
        if counts.get(century, 0) == 0:
            run_start = century if run_start is None else run_start
        elif run_start is not None:
            spans.append((run_start, century - 1))
            run_start = None
    return tuple(spans)


def _pair(conn: Connection, stmt: TextClause, **params: int) -> tuple[int | None, int | None]:
    row = conn.execute(stmt, params).one_or_none()
    if row is None:
        return None, None
    return (None if row[0] is None else int(row[0])), (None if row[1] is None else int(row[1]))


def _century_stats(
    conn: Connection, run_id: int
) -> tuple[int | None, int | None, int | None, tuple[tuple[int, int], ...]]:
    counts = {int(c): int(n) for c, n in conn.execute(_CENTURIES, {"run_id": run_id}).all()}
    if not counts:
        logger.warning("report_coverage_unavailable", item="corpus_timeline")
        return None, None, None, ()
    return sum(counts.values()), min(counts), max(counts), _empty_spans(counts)


def _headline_eras(conn: Connection, run_id: int, min_battles: int) -> tuple[EraCount, ...]:
    rows = conn.execute(_FIRST_YEARS, {"run_id": run_id, "min_battles": min_battles}).all()
    tally = Counter(era_label(int(r[0])) for r in rows if r[0] is not None)
    if not tally:
        logger.warning("report_coverage_unavailable", item="headline_by_era")
    return tuple(EraCount(era, tally[era]) for era in ERA_ORDER if tally[era])


def _fame_bins(conn: Connection, run_id: int) -> tuple[FameBin, ...]:
    found = {int(r[0]): (int(r[1]), int(r[2])) for r in conn.execute(_FAME, {"run_id": run_id})}
    if not found:
        logger.warning("report_coverage_unavailable", item="fame_bias")
    return tuple(
        FameBin(label, *found[low]) for low, label in FAME_BINS if low in found and found[low][0]
    )


MULTI_SOURCE_METHOD: Final[str] = "source_disagreement"
SINGLE_REPORT_METHOD: Final[str] = "single_report_debiased"
BOUND_ONLY_METHOD: Final[str] = "bound_only"


def _troop_slot(method: str | None, has_total: bool) -> int:
    """Bucket of one side: 0 multi-source, 1 single report, 2 imputed, 3 other, 4 bound only."""
    if not has_total:
        return 2
    if method == MULTI_SOURCE_METHOD:
        return 0
    if method == SINGLE_REPORT_METHOD:
        return 1
    if method == BOUND_ONLY_METHOD:
        return 4
    return 3


def _troop_eras(conn: Connection, run_id: int) -> tuple[TroopEra, ...]:
    tally: dict[str, list[int]] = {}
    unknown: Counter[str] = Counter()
    for year, method, has_total in conn.execute(_TROOPS, {"run_id": run_id}).all():
        slot = _troop_slot(method, bool(has_total))
        if slot == 3:
            unknown[str(method)] += 1
        tally.setdefault(era_label(int(year)), [0, 0, 0, 0, 0])[slot] += 1
    if not tally:
        logger.warning("report_coverage_unavailable", item="troop_provenance")
    if unknown:
        logger.warning("report_troop_method_unrecognised", methods=dict(unknown))
    return tuple(TroopEra(era, *tally[era]) for era in ERA_ORDER if era in tally)


def _year_span(conn: Connection, run_id: int) -> tuple[int | None, int | None]:
    lo, hi = conn.execute(_YEAR_SPAN, {"run_id": run_id}).one()
    return (None if lo is None else int(lo)), (None if hi is None else int(hi))


_NO_PAIR: tuple[int | None, int | None] = (None, None)
_NO_CENTURIES: tuple[int | None, int | None, int | None, tuple[tuple[int, int], ...]] = (
    None,
    None,
    None,
    (),
)


def load_coverage(conn: Connection, run_id: int, *, min_battles: int) -> CoverageStats:
    """Query every coverage number for the reported run.

    Args:
        conn: An open connection; only SELECTs are issued.
        run_id: The reported model run.
        min_battles: The headline threshold.

    Returns:
        The stats. Component counts over the top generals come from the model
        stage's files, not the database; add them with :func:`with_component_counts`.
    """
    rule_single, credited = guarded(
        conn, "rule_single", lambda: _pair(conn, _RULE_SINGLE, run_id=run_id), _NO_PAIR
    )
    if not credited:
        logger.warning("report_coverage_unavailable", item="rule_single")
        rule_single = credited = None
    n_components = guarded(
        conn, "components", lambda: conn.execute(_N_COMPONENTS, {"run_id": run_id}).scalar(), None
    )
    if n_components is None:
        logger.warning("report_coverage_unavailable", item="components")
    n_total, n_headline = guarded(
        conn,
        "general_counts",
        lambda: _pair(conn, _GENERAL_COUNTS, run_id=run_id, min_battles=min_battles),
        _NO_PAIR,
    )
    battles, first, last, spans = guarded(
        conn, "corpus_centuries", lambda: _century_stats(conn, run_id), _NO_CENTURIES
    )
    first_year, last_year = guarded(
        conn, "corpus_year_span", lambda: _year_span(conn, run_id), _NO_PAIR
    )
    return CoverageStats(
        rule_single_sides=rule_single,
        credited_sides=credited,
        n_components=None if n_components is None else int(n_components),
        extract_answers=guarded(
            conn, "extract_answers", lambda: int(conn.execute(_EXTRACT_OK).scalar_one()), None
        ),
        corpus_battles=battles,
        first_century=first,
        last_century=last,
        first_year=first_year,
        last_year=last_year,
        empty_spans=spans,
        n_headline=n_headline,
        n_generals_total=n_total,
        headline_by_era=guarded(
            conn, "headline_by_era", lambda: _headline_eras(conn, run_id, min_battles), ()
        ),
        fame_bins=guarded(conn, "fame_bias", lambda: _fame_bins(conn, run_id), ()),
        troop_eras=guarded(conn, "troop_provenance", lambda: _troop_eras(conn, run_id), ()),
        timeline=guarded(conn, "timeline", lambda: load_timeline(conn, run_id), ()),
        credit_source=guarded(
            conn,
            "credit_source",
            lambda: load_credit_source(conn, run_id, min_battles=min_battles),
            (),
        ),
        sigma_skill=guarded(conn, "sigma_skill", lambda: load_sigma_skill(conn, run_id), None),
        reported_vs_reconciled=guarded(
            conn, "reported_vs_reconciled", lambda: load_reported_vs_reconciled(conn, run_id), ()
        ),
    )


def with_component_counts(
    stats: CoverageStats, headline: Sequence[RankedGeneral], *, top: int = TOP_FOR_COMPONENTS
) -> CoverageStats:
    """Add how many of the leading headline generals sit outside the largest component.

    Component 0 is the largest, as the model stage numbers them. Generals with
    no stored component id (no ``rubin_per_general.csv``) are not counted.

    Args:
        stats: The stats from :func:`load_coverage`.
        headline: Headline generals in rank order.
        top: How many leading generals to look at.

    Returns:
        A copy with the two component fields set, or unchanged when none is known.
    """
    ids = [g.rubin.component_id for g in headline[:top] if g.rubin.component_id is not None]
    if not ids:
        logger.warning("report_coverage_unavailable", item="top_components")
        return stats
    return replace(
        stats, top_outside_largest=sum(1 for c in ids if c != 0), top_component_known=len(ids)
    )


def with_component_sizes(
    stats: CoverageStats,
    rubin: Mapping[int, RubinInfo],
    headline: Sequence[RankedGeneral],
    *,
    top: int = TOP_MARKERS,
) -> CoverageStats:
    """Add component sizes and the leading generals' components for the components panel.

    Sizes count the generals of the model stage's ``rubin_per_general.csv`` per
    component id, so they cover every credited general.

    Args:
        stats: The stats so far.
        rubin: Per-general model-stage columns by general id.
        headline: Headline generals in rank order.
        top: How many leading generals to mark.

    Returns:
        A copy with ``component_sizes`` (index = component id) and ``top_markers``
        set, or unchanged when no component ids are stored.
    """
    tally = Counter(r.component_id for r in rubin.values() if r.component_id is not None)
    if not tally:
        logger.warning("report_coverage_unavailable", item="component_sizes")
        return stats
    sizes = tuple(tally.get(i, 0) for i in range(max(tally) + 1))
    markers = tuple(
        ComponentMarker(g.estimate.name, g.headline_rank or 0, g.rubin.component_id)
        for g in headline[:top]
        if g.rubin.component_id is not None
    )
    return replace(stats, component_sizes=sizes, top_markers=markers)
