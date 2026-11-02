"""Campaign-scale battles are kept out of reconcile, and their sides are logged."""

from __future__ import annotations

from typing import Any

from pipeline.reconcilers.load import (
    CAMPAIGN_UNFILLABLE_REASON,
    _build_casualty_reports,
    _build_troop_reports,
    classify_sides,
    load_reports,
)
from pipeline.reconcilers.records import ReconcileCounts


class _Rows:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self._rows = rows

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._rows


class _FakeConn:
    """Answers each load query by the table it reads, in the loader's column order."""

    def __init__(self, troops: list[tuple[Any, ...]], casualties: list[tuple[Any, ...]]) -> None:
        self._by_marker = {
            "FROM troop_reports tr JOIN": troops,
            "FROM casualty_reports cr JOIN": casualties,
        }
        self.troop_side_rows: list[tuple[Any, ...]] = []
        self.casualty_side_rows: list[tuple[Any, ...]] = []

    def execute(self, statement: Any) -> _Rows:
        sql = " ".join(str(statement).split())
        if "SELECT DISTINCT cr.side_id" in sql:
            return _Rows(self.casualty_side_rows)
        if "SELECT tr.side_id" in sql:
            return _Rows(self.troop_side_rows)
        for marker, rows in self._by_marker.items():
            if marker in sql:
                return _Rows(rows)
        return _Rows([])


def _troop(report_id: int, side_id: int, battle_id: int, battle_type: str) -> tuple[Any, ...]:
    return (
        report_id, side_id, battle_id, 9, "wikipedia", f"https://x/{report_id}", "total",
        50_000.0, "unknown", False, False, False, "", 1800, battle_type,
    )  # fmt: skip


def _casualty(report_id: int, side_id: int, battle_id: int, battle_type: str) -> tuple[Any, ...]:
    return (
        report_id, side_id, battle_id, 9, "wikipedia", f"https://x/c{report_id}", "total",
        1_000.0, False, "", 1800, battle_type,
    )  # fmt: skip


def _conn() -> _FakeConn:
    return _FakeConn(
        troops=[_troop(1, 10, 1, "campaign"), _troop(2, 20, 2, "field")],
        casualties=[_casualty(1, 10, 1, "campaign"), _casualty(2, 20, 2, "field")],
    )


def test_campaign_troop_and_casualty_reports_are_excluded_and_counted() -> None:
    counts = ReconcileCounts()

    reports = load_reports(_conn(), ancient_cutoff_year=500, counts=counts)

    assert {(r.quantity, r.side_id) for r in reports} == {("troops", 20), ("casualties", 20)}
    assert counts.excluded_campaign == 2
    assert counts.as_dict()["excluded_campaign"] == 2


def test_a_non_campaign_battle_is_untouched() -> None:
    counts = ReconcileCounts()

    troops = _build_troop_reports(_conn(), counts, min_plausible_troops=100.0)

    assert [r.side_id for r in troops] == [20]
    assert counts.troop_reports_used == 1


def test_exclusion_off_keeps_campaign_reports() -> None:
    counts = ReconcileCounts()
    conn = _conn()

    troops = _build_troop_reports(
        conn, counts, min_plausible_troops=100.0, exclude_campaigns=False
    )
    casualties = _build_casualty_reports(conn, counts, exclude_campaigns=False)

    assert {r.side_id for r in troops} == {10, 20}
    assert {r.side_id for r in casualties} == {10, 20}
    assert counts.excluded_campaign == 0


def _sides_conn() -> _FakeConn:
    conn = _conn()
    conn.troop_side_rows = [
        (10, 1, "total", 50_000.0, "", "unknown", "campaign"),
        (20, 2, "total", 50_000.0, "", "unknown", "field"),
    ]
    conn.casualty_side_rows = [(10, 1, "campaign"), (20, 2, "field")]
    return conn


def test_campaign_sides_are_unfillable_with_a_distinct_reason_for_both_quantities() -> None:
    conn = _sides_conn()
    reports = load_reports(conn, ancient_cutoff_year=500, counts=ReconcileCounts())

    modellable, unfillable = classify_sides(reports, conn)

    assert modellable == {20}
    assert {(u.side_id, u.quantity, u.reason) for u in unfillable} == {
        (10, "troops", CAMPAIGN_UNFILLABLE_REASON),
        (10, "casualties", CAMPAIGN_UNFILLABLE_REASON),
    }


def test_exclusion_off_leaves_no_campaign_unfillable_sides() -> None:
    conn = _sides_conn()
    reports = load_reports(
        conn, ancient_cutoff_year=500, counts=ReconcileCounts(), exclude_campaigns=False
    )

    _, unfillable = classify_sides(reports, conn, exclude_campaigns=False)

    assert unfillable == []
