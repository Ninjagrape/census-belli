"""A side whose only troop figures name one branch counts that figure as its total."""

from __future__ import annotations

from typing import Any

from pipeline.reconcilers.load import DEFAULT_MIN_PLAUSIBLE_TROOPS, _build_troop_reports
from pipeline.reconcilers.records import ReconcileCounts


class _Rows:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self._rows = rows

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._rows


class _FakeConn:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self._rows = rows

    def execute(self, _statement: Any) -> _Rows:
        return _Rows(self._rows)


def _row(
    report_id: int,
    branch: str,
    value: float,
    *,
    side_id: int = 7,
    url: str = "https://example/x",
    source_id: int = 9,
) -> tuple[Any, ...]:
    return (
        report_id, side_id, 3, source_id, "wikipedia_infobox", url, branch, value, "unknown",
        True, False, False, "", 1864, "field",
    )  # fmt: skip


def _load(rows: list[tuple[Any, ...]]) -> tuple[list[Any], ReconcileCounts]:
    counts = ReconcileCounts()
    reports = _build_troop_reports(
        _FakeConn(rows), counts, min_plausible_troops=DEFAULT_MIN_PLAUSIBLE_TROOPS
    )
    return reports, counts


def test_a_lone_cavalry_figure_is_promoted_to_the_total() -> None:
    reports, counts = _load([_row(1, "cavalry", 3500.0)])

    assert [(r.report_id, r.reported_value, r.branch) for r in reports] == [(1, 3500.0, "total")]
    assert counts.promoted_lone_branch == 1
    assert counts.excluded_non_total_branch == 0
    assert counts.troop_reports_used == 1


def test_several_reports_of_the_same_single_branch_are_all_promoted() -> None:
    rows = [_row(1, "infantry", 3000.0), _row(2, "infantry", 4000.0, source_id=10)]
    reports, counts = _load(rows)

    assert len(reports) == 2
    assert counts.promoted_lone_branch == 2


def test_two_distinct_branches_are_not_promoted() -> None:
    reports, counts = _load([_row(1, "infantry", 3000.0), _row(2, "cavalry", 500.0)])

    assert reports == []
    assert counts.promoted_lone_branch == 0
    assert counts.excluded_non_total_branch == 2


def test_no_promotion_when_the_side_has_a_total_report() -> None:
    reports, counts = _load([_row(1, "total", 9000.0), _row(2, "cavalry", 500.0)])

    assert [r.report_id for r in reports] == [1]
    assert counts.promoted_lone_branch == 0
    assert counts.excluded_non_total_branch == 1


def test_a_total_on_another_side_does_not_block_promotion() -> None:
    reports, counts = _load([_row(1, "total", 9000.0, side_id=8), _row(2, "cavalry", 500.0)])

    assert sorted(r.report_id for r in reports) == [1, 2]
    assert counts.promoted_lone_branch == 1


def test_naval_is_never_promoted() -> None:
    reports, counts = _load([_row(1, "naval", 5000.0)])

    assert reports == []
    assert counts.promoted_lone_branch == 0
    assert counts.excluded_naval == 1


def test_naval_beside_one_land_branch_does_not_block_that_branch() -> None:
    reports, counts = _load([_row(1, "naval", 5000.0), _row(2, "infantry", 3000.0)])

    assert [r.report_id for r in reports] == [2]
    assert counts.promoted_lone_branch == 1


def test_equipment_and_catch_all_branches_are_never_promoted_on_their_own() -> None:
    for branch in ("artillery", "armour", "air", "other"):
        reports, counts = _load([_row(1, branch, 900.0)])

        assert reports == [], branch
        assert counts.promoted_lone_branch == 0, branch
        assert counts.excluded_non_total_branch == 1, branch


def test_a_promoted_value_below_the_floor_is_still_dropped() -> None:
    reports, counts = _load([_row(1, "infantry", 85.0)])

    assert reports == []
    assert counts.promoted_lone_branch == 1
    assert counts.excluded_implausible_small == 1
    assert counts.troop_reports_used == 0


def test_a_non_positive_single_branch_row_is_not_promoted() -> None:
    reports, counts = _load([_row(1, "infantry", 0.0)])

    assert reports == []
    assert counts.promoted_lone_branch == 0
    assert counts.excluded_non_total_branch == 1


def test_a_land_branch_beside_an_equipment_count_is_promoted() -> None:
    for branch in ("artillery", "armour", "air"):
        reports, counts = _load([_row(1, "cavalry", 3500.0), _row(2, branch, 8.0)])

        assert [(r.report_id, r.branch) for r in reports] == [(1, "total")], branch
        assert counts.promoted_lone_branch == 1, branch
        assert counts.excluded_non_total_branch == 1, branch


def test_a_land_branch_beside_an_other_row_is_not_promoted() -> None:
    reports, counts = _load([_row(1, "cavalry", 3500.0), _row(2, "other", 900.0)])

    assert reports == []
    assert counts.promoted_lone_branch == 0
    assert counts.excluded_non_total_branch == 2


def test_promoted_rows_carry_the_lone_branch_flag_and_totals_do_not() -> None:
    reports, _ = _load([_row(1, "total", 9000.0, side_id=8), _row(2, "cavalry", 3500.0)])

    assert {r.report_id: r.lone_branch for r in reports} == {1: False, 2: True}


def test_a_side_promoted_then_floored_is_called_implausible_not_non_total() -> None:
    from pipeline.reconcilers.load import _unfillable_reason

    floor = DEFAULT_MIN_PLAUSIBLE_TROOPS
    promoted_floored = [("infantry", 85.0, "", "unknown")]
    two_branches = [("infantry", 3000.0, "", "unknown"), ("cavalry", 50.0, "", "unknown")]
    other_blocks = [("infantry", 85.0, "", "unknown"), ("other", 40.0, "", "unknown")]

    assert (
        _unfillable_reason(promoted_floored, min_plausible_troops=floor)
        == "only implausible or non-personnel figures"
    )
    assert (
        _unfillable_reason(two_branches, min_plausible_troops=floor)
        == "only non-total branch reports"
    )
    assert (
        _unfillable_reason(other_blocks, min_plausible_troops=floor)
        == "only non-total branch reports"
    )
