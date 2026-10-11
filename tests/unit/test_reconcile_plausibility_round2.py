"""Second-round plausibility rules: more equipment words, crews, bracketed sub-figures."""

from __future__ import annotations

import pytest

from pipeline.reconcilers.plausibility import (
    drop_orphan_parenthetical_rows,
    exclusion_reason,
    filter_troop_reports,
)
from pipeline.reconcilers.records import ReconcileCounts, Report, SourceKey

MIN_TROOPS = 100


@pytest.mark.parametrize(
    ("value", "context"),
    [
        (4_000.0, "4,000 horses"),
        (300.0, "300 mortars"),
        (1_500.0, "1,500 camels"),
    ],
)
def test_more_equipment_and_animals_are_non_personnel(value: float, context: str) -> None:
    assert exclusion_reason(value, context, min_plausible_troops=MIN_TROOPS) == "non_personnel"


@pytest.mark.parametrize(
    ("value", "context"),
    [
        (2_000.0, "2,000 tank crews"),
        (3_000.0, "3,000 ship's crew"),
        (5_000.0, "5,000 ship crews"),
        (4_000.0, "4,000 sailors on ships"),
        (6_000.0, "6,000 marines"),
        (7_000.0, "7,000 men on ships"),
    ],
)
def test_crews_and_men_block_the_equipment_rule(value: float, context: str) -> None:
    assert exclusion_reason(value, context, min_plausible_troops=MIN_TROOPS) is None


@pytest.mark.parametrize(
    ("value", "context"),
    [
        (1_426_352.0, "1,910,361 (including 1,426,352 combat soldiers)"),
        (3_000.0, "23,000 troops (3,000 crossed)"),
        (60_000.0, "90,000 men (60,000 Italian, 30,000 German)"),
        (30_000.0, "90,000 men (60,000 Italian, 30,000 German)"),
        (65_000.0, "80,000 (65,000 engaged )"),
    ],
)
def test_a_bracketed_figure_beside_a_larger_one_is_a_subfigure(value: float, context: str) -> None:
    assert (
        exclusion_reason(value, context, min_plausible_troops=MIN_TROOPS)
        == "parenthetical_subfigure"
    )


@pytest.mark.parametrize(
    ("value", "context"),
    [
        (90_000.0, "90,000 men (60,000 Italian, 30,000 German)"),
        # The bracketed figure is the larger one, so it is not a sub-figure.
        (80_000.0, "25,000 (80,000 reinforcements under von Daun never engaged)"),
        (23_000.0, "23,000 troops (3,000 crossed)"),
    ],
)
def test_the_outer_or_larger_figure_is_kept(value: float, context: str) -> None:
    assert exclusion_reason(value, context, min_plausible_troops=MIN_TROOPS) is None


def _report(report_id: int, value: float, context: str, url: str = "https://example/a") -> Report:
    return Report(
        report_id=report_id,
        side_id=1,
        battle_id=1,
        source_id=report_id,
        source_key=SourceKey("wikipedia_infobox", url),
        source_type="wikipedia_infobox",
        quantity="troops",
        reported_value=value,
        extracted_context=context,
        lineage_id=report_id,
    )


def test_a_bracket_only_context_is_dropped_when_the_source_states_a_larger_figure() -> None:
    counts = ReconcileCounts()
    reports = [_report(1, 23_000.0, "23,000 troops"), _report(2, 3_000.0, "(3,000 crossed)")]

    kept = drop_orphan_parenthetical_rows(reports, counts)

    assert [r.report_id for r in kept] == [1]
    assert counts.excluded_parenthetical_subfigure == 1


def test_a_lone_bracket_only_context_is_kept() -> None:
    counts = ReconcileCounts()

    kept = drop_orphan_parenthetical_rows([_report(1, 3_000.0, "(3,000 crossed)")], counts)

    assert len(kept) == 1
    assert counts.excluded_parenthetical_subfigure == 0


def test_the_filter_counts_subfigures_under_their_own_reason() -> None:
    counts = ReconcileCounts()
    reports = [
        _report(1, 1_910_361.0, "1,910,361 (including 1,426,352 combat soldiers)"),
        _report(2, 1_426_352.0, "1,910,361 (including 1,426,352 combat soldiers)"),
    ]

    kept = filter_troop_reports(reports, counts, min_plausible_troops=MIN_TROOPS)

    assert [r.report_id for r in kept] == [1]
    assert counts.excluded_parenthetical_subfigure == 1
