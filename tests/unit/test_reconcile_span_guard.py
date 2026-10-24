"""The span guard: a side whose kept point totals disagree by more than the bounds."""

from __future__ import annotations

import math

import pytest

from pipeline.reconcilers.load import (
    SPAN_ERROR_LOG_RATIO,
    SPAN_WARN_LOG_RATIO,
    troop_span_findings,
)
from pipeline.reconcilers.records import ReconcileCounts, Report, SourceKey


def _report(
    report_id: int,
    value: float,
    *,
    side_id: int = 1,
    source_type: str = "wikipedia_infobox",
    lower: bool = False,
    upper: bool = False,
    quantity: str = "troops",
) -> Report:
    return Report(
        report_id=report_id,
        side_id=side_id,
        battle_id=side_id,
        source_id=1,
        source_key=SourceKey.from_row(source_type, ""),
        source_type=source_type,
        quantity=quantity,  # type: ignore[arg-type]
        reported_value=value,
        branch="total",
        claim_regime="unlabelled",
        is_lower_bound=lower,
        is_upper_bound=upper,
    )


def test_the_thresholds_are_one_and_two_in_log_units() -> None:
    assert SPAN_WARN_LOG_RATIO == 1.0
    assert SPAN_ERROR_LOG_RATIO == 2.0


def test_a_side_inside_the_warning_bound_raises_nothing() -> None:
    reports = [_report(1, 10_000.0), _report(2, 20_000.0)]

    assert troop_span_findings(reports) == []


def test_a_span_exactly_at_the_warning_bound_is_not_flagged() -> None:
    reports = [_report(1, 1.0), _report(2, math.e)]

    assert troop_span_findings(reports) == []


def test_a_span_just_over_the_warning_bound_warns_with_min_and_max() -> None:
    reports = [_report(1, 10_000.0), _report(2, 28_000.0), _report(3, 15_000.0)]

    (finding,) = troop_span_findings(reports)

    assert finding.severity == "warning"
    assert (finding.side_id, finding.min_value, finding.max_value) == (1, 10_000.0, 28_000.0)
    assert finding.log_span == pytest.approx(math.log(2.8))


def test_a_span_exactly_at_the_error_bound_is_a_warning_not_an_error() -> None:
    reports = [_report(1, 1.0), _report(2, math.e**2)]

    (finding,) = troop_span_findings(reports)

    assert finding.severity == "warning"


def test_a_span_over_the_error_bound_is_an_error() -> None:
    reports = [_report(1, 10_000.0), _report(2, 100_000.0)]

    (finding,) = troop_span_findings(reports)

    assert finding.severity == "error"


def test_sides_are_judged_separately() -> None:
    reports = [
        _report(1, 10_000.0, side_id=1),
        _report(2, 100_000.0, side_id=1),
        _report(3, 10_000.0, side_id=2),
        _report(4, 11_000.0, side_id=2),
    ]

    assert [f.side_id for f in troop_span_findings(reports)] == [1]


def test_bounds_and_casualty_rows_do_not_count_toward_the_span() -> None:
    reports = [
        _report(1, 10_000.0),
        _report(2, 500_000.0, upper=True),
        _report(3, 100.0, lower=True),
        _report(4, 900_000.0, quantity="casualties"),
    ]

    assert troop_span_findings(reports) == []


def test_a_side_with_a_manual_override_row_is_skipped() -> None:
    reports = [
        _report(1, 10_000.0),
        _report(2, 100_000.0),
        _report(3, 50_000.0, source_type="manual_entry"),
    ]

    assert troop_span_findings(reports) == []


def test_a_range_override_skips_the_guard_too() -> None:
    reports = [
        _report(1, 10_000.0),
        _report(2, 100_000.0),
        _report(3, 40_000.0, source_type="manual_entry", lower=True),
        _report(4, 60_000.0, source_type="manual_entry", upper=True),
    ]

    assert troop_span_findings(reports) == []


def test_the_override_skips_only_its_own_side() -> None:
    reports = [
        _report(1, 10_000.0, side_id=1),
        _report(2, 100_000.0, side_id=1),
        _report(3, 50_000.0, side_id=1, source_type="manual_entry"),
        _report(4, 10_000.0, side_id=2),
        _report(5, 100_000.0, side_id=2),
    ]

    assert [f.side_id for f in troop_span_findings(reports)] == [2]


def test_thresholds_can_be_overridden() -> None:
    reports = [_report(1, 10_000.0), _report(2, 12_000.0)]

    (finding,) = troop_span_findings(reports, warn_log_span=0.1, error_log_span=0.15)

    assert finding.severity == "error"


def test_the_counters_are_reported() -> None:
    counts = ReconcileCounts(troop_span_warnings=2, troop_span_errors=1)

    reported = counts.as_dict()

    assert reported["troop_span_warnings"] == 2
    assert reported["troop_span_errors"] == 1
