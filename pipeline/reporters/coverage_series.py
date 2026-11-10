"""Series behind the coverage figures, queried at report time.

Same contract as :mod:`pipeline.reporters.coverage`: read-only SQLAlchemy Core,
parameterised, and a quantity that cannot be determined comes back empty or
``None`` with a structlog warning rather than an exception.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, TypeVar

import structlog
from sqlalchemy import text
from sqlalchemy.engine import Connection
from sqlalchemy.exc import SQLAlchemyError

__all__ = [
    "CREDITED_BATTLES",
    "ONE_COMMANDER_ROW",
    "MAX_SCATTER_POINTS",
    "TIMELINE_BIN",
    "ComponentMarker",
    "CreditSource",
    "ReportedPoint",
    "TimelineBin",
    "guarded",
    "load_credit_source",
    "load_reported_vs_reconciled",
    "load_sigma_skill",
    "load_timeline",
]

logger = structlog.get_logger()

TIMELINE_BIN: Final[int] = 25
# Keeps the explorer's embedded data small; the sample is a fixed stride, so reruns agree.
MAX_SCATTER_POINTS: Final[int] = 2000

CREDITED_BATTLES: Final[str] = (
    "SELECT DISTINCT battle_id FROM battle_war_details WHERE run_id = :run_id"
)

# One battle_commanders row per (battle, general): battle_war_details has no side, so a general
# listed on both sides of a battle would otherwise join twice and be counted twice. The row with
# the larger attribution weight wins, then the lower side id, so reruns agree.
ONE_COMMANDER_ROW: Final[str] = (
    "(SELECT DISTINCT ON (battle_id, general_id) battle_id, general_id, side_id, hierarchy_rank, "
    "attribution_method, attribution_weight FROM battle_commanders "
    "ORDER BY battle_id, general_id, attribution_weight DESC NULLS LAST, side_id)"
)

_T = TypeVar("_T")


def guarded(conn: Connection, query: str, load: Callable[[], _T], default: _T) -> _T:
    """Run one coverage query in a savepoint; a database error leaves ``default`` and a log line.

    The savepoint keeps the surrounding Postgres transaction usable after a failed statement,
    so the next quantity can still be queried.

    Args:
        conn: An open connection.
        query: Name of the quantity, for the log.
        load: Issues the statement(s) and returns the quantity.
        default: What to return when the database raises.

    Returns:
        ``load()``, or ``default`` after a SQLAlchemyError.
    """
    try:
        with conn.begin_nested():
            return load()
    except SQLAlchemyError as exc:
        logger.warning("report_coverage_unavailable", query=query, error=str(exc))
        return default


_TIMELINE = text(
    f"SELECT CAST(FLOOR(b.year_astronomical / {TIMELINE_BIN}.0) * {TIMELINE_BIN} AS INT), COUNT(*) "
    f"FROM battles b WHERE b.battle_id IN ({CREDITED_BATTLES}) "
    "AND b.year_astronomical IS NOT NULL GROUP BY 1 ORDER BY 1"
)
_CREDIT_SOURCE = text(
    "SELECT d.general_id, COALESCE(bc.attribution_method, 'unknown'), "
    "SUM(COALESCE(d.attribution_weight, bc.attribution_weight, 0.0)) "
    f"FROM battle_war_details d JOIN {ONE_COMMANDER_ROW} bc "
    "ON bc.battle_id = d.battle_id AND bc.general_id = d.general_id "
    "WHERE d.run_id = :run_id AND d.general_id IN ("
    "SELECT general_id FROM general_skill_estimates "
    "WHERE run_id = :run_id AND n_battles >= :min_battles) "
    "GROUP BY 1, 2 ORDER BY 1, 2"
)
# The pooled run row stores the mean of the group skill scale; the per-round rows carry it too.
_SIGMA = text(
    "SELECT CAST(COALESCE(diagnostics ->> 'sigma_skill_mean', "
    "diagnostics -> 'rounds' -> 0 ->> 'sigma_skill_mean') AS DOUBLE PRECISION) "
    "FROM model_runs WHERE run_id = :run_id"
)
_REPORTED = text(
    "SELECT b.year_astronomical, s.est_troops_total, AVG(r.reported_value) "
    "FROM battle_sides s JOIN battles b ON b.battle_id = s.battle_id "
    "JOIN troop_reports r ON r.side_id = s.side_id AND r.branch = 'total' "
    f"WHERE s.battle_id IN ({CREDITED_BATTLES}) AND b.year_astronomical IS NOT NULL "
    "AND s.est_troops_total > 0 AND r.reported_value > 0 "
    "GROUP BY s.side_id, b.year_astronomical, s.est_troops_total ORDER BY s.side_id"
)


@dataclass(frozen=True)
class TimelineBin:
    """Credited battles in one 25-year bin.

    Attributes:
        start: First astronomical year of the bin.
        n: Credited battles in it.
    """

    start: int
    n: int


@dataclass(frozen=True)
class CreditSource:
    """Credited weight of one general that came from one attribution method."""

    general_id: int
    method: str
    weight: float


@dataclass(frozen=True)
class ReportedPoint:
    """One side's mean reported troop total against its reconciled total."""

    year: int
    reported: float
    reconciled: float


@dataclass(frozen=True)
class ComponentMarker:
    """A leading general placed on the bar of its comparison component."""

    name: str
    rank: int
    component_id: int


def load_timeline(conn: Connection, run_id: int) -> tuple[TimelineBin, ...]:
    """Credited battles per 25-year bin.

    Args:
        conn: An open connection.
        run_id: The reported model run.

    Returns:
        The occupied bins in year order; empty when no credited battle is dated.
    """
    rows = conn.execute(_TIMELINE, {"run_id": run_id}).all()
    if not rows:
        logger.warning("report_coverage_unavailable", item="timeline_bins")
    return tuple(TimelineBin(int(start), int(n)) for start, n in rows)


def load_credit_source(
    conn: Connection, run_id: int, *, min_battles: int
) -> tuple[CreditSource, ...]:
    """Credited weight per headline general and attribution method.

    Args:
        conn: An open connection.
        run_id: The reported model run.
        min_battles: The headline threshold.

    Returns:
        One row per general and method present; empty when nothing is credited.
    """
    rows = conn.execute(_CREDIT_SOURCE, {"run_id": run_id, "min_battles": min_battles}).all()
    if not rows:
        logger.warning("report_coverage_unavailable", item="credit_source")
    return tuple(CreditSource(int(g), str(m), float(w)) for g, m, w in rows)


def load_sigma_skill(conn: Connection, run_id: int) -> float | None:
    """The model's skill scale, needed for the contraction 1 - (sd / sigma)^2.

    Args:
        conn: An open connection.
        run_id: The reported model run.

    Returns:
        The stored mean of ``sigma_skill``, or None when the run did not store it.
    """
    value = conn.execute(_SIGMA, {"run_id": run_id}).scalar()
    if value is None or float(value) <= 0:
        logger.warning("report_coverage_unavailable", item="sigma_skill")
        return None
    return float(value)


def load_reported_vs_reconciled(conn: Connection, run_id: int) -> tuple[ReportedPoint, ...]:
    """Mean reported total against reconciled total, one point per side.

    Args:
        conn: An open connection.
        run_id: The reported model run.

    Returns:
        Up to :data:`MAX_SCATTER_POINTS` points, thinned by a fixed stride.
    """
    rows = conn.execute(_REPORTED, {"run_id": run_id}).all()
    if not rows:
        logger.warning("report_coverage_unavailable", item="reported_vs_reconciled")
        return ()
    stride = -(-len(rows) // MAX_SCATTER_POINTS)
    return tuple(ReportedPoint(int(y), float(rep), float(rec)) for y, rec, rep in rows[::stride])


def credit_by_general(rows: tuple[CreditSource, ...]) -> dict[int, dict[str, float]]:
    """Group credit rows as ``general_id -> {method: weight}``."""
    out: dict[int, dict[str, float]] = defaultdict(dict)
    for row in rows:
        out[row.general_id][row.method] = out[row.general_id].get(row.method, 0.0) + row.weight
    return dict(out)
