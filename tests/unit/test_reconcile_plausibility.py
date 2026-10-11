"""Unit tests for the load-time plausibility rules on troop reports.

Every case here is a context string read off the real corpus. The rules exist
because the extractor defaults ``branch`` to ``total``, so a day number from a
date label, a ship count and a component of a stated total all arrive as if
they were a side's whole army.
"""

from __future__ import annotations

import pytest

from pipeline.reconcilers.plausibility import (
    drop_components_of_stated_totals,
    exclusion_reason,
    is_stated_total,
)
from pipeline.reconcilers.records import ReconcileCounts, Report, SourceKey

MIN_TROOPS = 100


@pytest.mark.parametrize(
    ("value", "context"),
    [
        (86_400.0, "Total: 86,400"),
        (47_000.0, "47,000"),
        (1_000_000.0, "250,000–1,000,000 (ancient sources)"),
        (400.0, "400 Heruls"),
        (150.0, "150 archers"),
        (91_000.0, "91,000 infantrymen"),
        (7_232.0, "7,232 gunners"),
        (50_000.0, "Wellington's army: approx. 50,000 soldiers"),
        (28_000.0, "13 March: 28,000"),
        (5_000.0, ""),
    ],
)
def test_real_personnel_totals_are_not_excluded(value: float, context: str) -> None:
    assert exclusion_reason(value, context, min_plausible_troops=MIN_TROOPS) is None


@pytest.mark.parametrize("value", [1.0, 13.0, 50.0, 99.0])
def test_a_value_below_the_floor_is_implausibly_small(value: float) -> None:
    assert (
        exclusion_reason(value, "a plain figure", min_plausible_troops=MIN_TROOPS)
        == "implausible_small"
    )


def test_the_floor_itself_is_kept() -> None:
    assert exclusion_reason(100.0, "100 men", min_plausible_troops=MIN_TROOPS) is None


@pytest.mark.parametrize(
    ("value", "context"),
    [
        (250.0, "250–400 galleys"),
        (400.0, "250–400 galleys"),
        (50.0, "30–50 transports"),
        (4_200.0, "4,200 submarines"),
        (2_000.0, "2 Panokseons, 2,000 Panokseons"),
        (1_400.0, "1,400 helicopters"),
        (3_253.0, "3,253 tanks"),
        (9_966.0, "9,966 guns and mortars"),
        (2_792.0, "2,792 to 3,549 aircraft"),
        (3_549.0, "2,792 to 3,549 aircraft"),
        (4.0, "Luftflotte 4"),
        (600.0, "600 AIRCRAFT, 1,600 by mid-September"),
        (4_600.0, "4,600 fishing boats"),
        (120.0, "120 warships"),
        (252.0, "• 246–252 guns"),
        (200.0, "85–200 war elephants"),
        (150.0, "150 surface-to-air missile batteries (62 in the front line)"),
    ],
)
def test_a_figure_in_front_of_an_equipment_word_is_non_personnel(
    value: float, context: str
) -> None:
    assert exclusion_reason(value, context, min_plausible_troops=MIN_TROOPS) == "non_personnel"


@pytest.mark.parametrize(
    ("value", "context"),
    [
        (7_232.0, "7,232 gunners"),
        (9_000.0, "9,000 shipwrights"),
        (5_000.0, "tankers 5,000"),
        (8_050.0, "8,050 gunners and engineers"),
        # The pilot corpus's real totals that sit beside a gun count: the figure
        # belongs to the men, not to the guns.
        (30_000.0, "30,000 men, 100 cannon"),
        (22_000.0, "Originally: 22,000 men, 15 cannon"),
        (9_000.0, "Prussia: 9,000, 16 guns Total: 23,000 –27,000, 106 guns"),
        (5_000.0, "3,700–5,000, 24 guns"),
        (50_000.0, "• 156 guns Blücher's army: approx. 50,000 soldiers"),
        (20_000.0, "20,000 men on ships"),
    ],
)
def test_a_personnel_figure_beside_an_equipment_count_is_kept(value: float, context: str) -> None:
    assert exclusion_reason(value, context, min_plausible_troops=MIN_TROOPS) is None


@pytest.mark.parametrize(
    ("value", "context"),
    [
        (13.0, "13 March:"),
        (17.0, "16–17 October:"),
        (16.0, "16-17 October:"),
        (1.0, "On 1 March 1944 :"),
        (1941.0, "As of 1 October 1941:"),
        (2.0, "2 January:"),
        (24.0, "24 December:"),
    ],
)
def test_a_date_label_is_not_a_troop_count(value: float, context: str) -> None:
    # Checked before the small-value floor so the count says why: a 1941 here
    # clears the floor and is still a year, not an army.
    assert exclusion_reason(value, context, min_plausible_troops=MIN_TROOPS) == "date_context"


def test_a_date_label_followed_by_a_figure_is_not_a_date_context() -> None:
    assert exclusion_reason(28_000.0, "13 March: 28,000", min_plausible_troops=MIN_TROOPS) is None


@pytest.mark.parametrize(
    ("context", "expected"),
    [
        ("Total: 86,400", True),
        ("Total : 118,000–120,000", True),
        ("total: c. 50,000", True),
        ("• TOTAL: 5,000", True),
        ("30,000 allies", False),
        ("78,000 in total", False),
        ("Totalling 5,000", False),
        ("", False),
    ],
)
def test_a_stated_total_is_marked_at_the_start_of_the_context(context: str, expected: bool) -> None:
    assert is_stated_total(context) is expected


def _report(
    report_id: int,
    *,
    side_id: int = 1,
    url: str = "https://example/cannae",
    value: float = 1_000.0,
    context: str = "",
) -> Report:
    return Report(
        report_id=report_id,
        side_id=side_id,
        battle_id=side_id,
        source_id=report_id,
        source_key=SourceKey("wikipedia_infobox", url),
        source_type="wikipedia_infobox",
        quantity="troops",
        reported_value=value,
        extracted_context=context,
        lineage_id=report_id,
    )


def test_components_of_a_stated_total_are_dropped_from_the_same_source_and_side() -> None:
    reports = [
        _report(1, value=86_400.0, context="Total: 86,400"),
        _report(2, value=25_000.0, context="25,000 Romans"),
        _report(3, value=30_000.0, context="30,000 allies"),
        _report(4, value=10_000.0, context="10,000 allied garrison"),
    ]
    counts = ReconcileCounts()

    kept = drop_components_of_stated_totals(reports, counts)

    assert [r.report_id for r in kept] == [1]
    assert counts.excluded_component_of_stated_total == 3


def test_both_ends_of_a_stated_total_range_are_kept() -> None:
    reports = [
        _report(1, value=120_000.0, context="Total : 118,000–120,000"),
        _report(2, value=118_000.0, context="Total : 118,000–120,000"),
        _report(3, value=91_000.0, context="91,000 infantrymen"),
    ]
    counts = ReconcileCounts()

    kept = drop_components_of_stated_totals(reports, counts)

    assert [r.report_id for r in kept] == [1, 2]
    assert counts.excluded_component_of_stated_total == 1


def test_a_side_without_a_stated_total_keeps_every_row() -> None:
    reports = [_report(1, context="25,000 Romans"), _report(2, context="30,000 allies")]
    counts = ReconcileCounts()

    assert len(drop_components_of_stated_totals(reports, counts)) == 2
    assert counts.excluded_component_of_stated_total == 0


def test_a_stated_total_does_not_reach_across_sources_or_sides() -> None:
    reports = [
        _report(1, value=86_400.0, context="Total: 86,400"),
        _report(2, value=25_000.0, context="25,000 Romans", url="https://example/other"),
        _report(3, value=30_000.0, context="30,000 allies", side_id=2),
    ]
    counts = ReconcileCounts()

    kept = drop_components_of_stated_totals(reports, counts)

    assert [r.report_id for r in kept] == [1, 2, 3]
    assert counts.excluded_component_of_stated_total == 0
