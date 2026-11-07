"""A side with a modern-scholarly total drops every other total as superseded."""

from __future__ import annotations

from pipeline.reconcilers.load import (
    _unfillable_reason,
    drop_superseded_by_modern,
    drop_superseded_by_override,
)
from pipeline.reconcilers.records import ReconcileCounts, Report, SourceKey


def _report(
    report_id: int,
    value: float,
    regime: str,
    *,
    side_id: int = 1,
    context: str = "",
    lower: bool = False,
    upper: bool = False,
    url: str = "",
) -> Report:
    return Report(
        report_id=report_id,
        side_id=side_id,
        battle_id=1,
        source_id=1,
        source_key=SourceKey.from_row(
            "manual_entry" if url else "wikipedia_infobox", url
        ),
        source_type="manual_entry" if url else "wikipedia_infobox",
        quantity="troops",
        reported_value=value,
        branch="total",
        claim_regime=regime,
        extracted_context=context,
        is_lower_bound=lower,
        is_upper_bound=upper,
    )


def test_every_non_modern_total_on_a_side_with_a_modern_total_is_dropped_and_counted() -> None:
    counts = ReconcileCounts()
    reports = [
        _report(1, 70_000.0, "ancient_claim"),
        _report(2, 7_000.0, "modern_scholarly"),
        _report(3, 50_000.0, "unlabelled"),
        _report(4, 60_000.0, "chronicle"),
        _report(5, 51_000.0, "unlabelled", context="derived: sum of 2 component figures"),
    ]

    kept = drop_superseded_by_modern(reports, counts)

    assert [r.report_id for r in kept] == [2]
    assert counts.excluded_superseded_by_modern == 4


def test_a_side_without_a_modern_total_keeps_everything() -> None:
    counts = ReconcileCounts()
    reports = [_report(1, 70_000.0, "ancient_claim"), _report(2, 50_000.0, "unlabelled")]

    assert drop_superseded_by_modern(reports, counts) == reports
    assert counts.excluded_superseded_by_modern == 0


def test_supersession_is_per_side() -> None:
    counts = ReconcileCounts()
    reports = [
        _report(1, 70_000.0, "ancient_claim", side_id=1),
        _report(2, 7_000.0, "modern_scholarly", side_id=1),
        _report(3, 18_000.0, "ancient_claim", side_id=2),
    ]

    kept = drop_superseded_by_modern(reports, counts)

    assert [r.report_id for r in kept] == [2, 3]
    assert counts.excluded_superseded_by_modern == 1


def test_the_new_counters_are_reported() -> None:
    counts = ReconcileCounts(excluded_superseded_by_modern=3, excluded_population_scope=2)
    reported = counts.as_dict()
    assert reported["excluded_superseded_by_modern"] == 3
    assert reported["excluded_population_scope"] == 2


def test_a_lone_modern_bound_does_not_supersede_the_other_totals() -> None:
    counts = ReconcileCounts()
    reports = [
        _report(1, 70_000.0, "ancient_claim"),
        _report(
            2, 20_000.0, "modern_scholarly", upper=True, context="up to 20,000 (modern estimates)"
        ),
    ]

    assert drop_superseded_by_modern(reports, counts) == reports
    assert counts.excluded_superseded_by_modern == 0


def test_a_modern_lower_and_upper_pair_with_one_context_supersedes() -> None:
    counts = ReconcileCounts()
    context = "15,000-20,000 (modern estimates)"
    reports = [
        _report(1, 70_000.0, "ancient_claim"),
        _report(2, 15_000.0, "modern_scholarly", lower=True, context=context),
        _report(3, 20_000.0, "modern_scholarly", upper=True, context=context),
    ]

    kept = drop_superseded_by_modern(reports, counts)

    assert [r.report_id for r in kept] == [2, 3]
    assert counts.excluded_superseded_by_modern == 1


def test_modern_bounds_from_different_contexts_do_not_pair() -> None:
    counts = ReconcileCounts()
    reports = [
        _report(1, 70_000.0, "ancient_claim"),
        _report(2, 15_000.0, "modern_scholarly", lower=True, context="at least 15,000 (modern)"),
        _report(3, 20_000.0, "modern_scholarly", upper=True, context="up to 20,000 (modern)"),
    ]

    assert drop_superseded_by_modern(reports, counts) == reports


def test_a_side_whose_only_totals_are_population_scope_says_so() -> None:
    rows = [("total", 368_000.0, "368,000 people", "population"), ("other", 5.0, "x", "engaged")]
    assert _unfillable_reason(rows, min_plausible_troops=100.0) == "only population-scope totals"


_OVERRIDE_URL = "override:ov-0103"


def test_an_override_row_supersedes_a_modern_infobox_pair_on_its_side() -> None:
    counts = ReconcileCounts()
    infobox = "60,000-100,000 (modern estimates)"
    override = "10,000-30,000 (modern estimates)"
    reports = [
        _report(1, 60_000.0, "modern_scholarly", lower=True, context=infobox),
        _report(2, 100_000.0, "modern_scholarly", upper=True, context=infobox),
        _report(3, 10_000.0, "modern_scholarly", lower=True, context=override, url=_OVERRIDE_URL),
        _report(4, 30_000.0, "modern_scholarly", upper=True, context=override, url=_OVERRIDE_URL),
        _report(5, 5_000.0, "modern_scholarly", side_id=2, context="5,000 (modern estimates)"),
    ]

    kept = drop_superseded_by_override(reports, counts)

    assert [r.report_id for r in kept] == [3, 4, 5]
    assert counts.excluded_superseded_by_override == 2


def test_override_supersession_drops_points_and_bounds_of_any_regime() -> None:
    counts = ReconcileCounts()
    reports = [
        _report(1, 70_000.0, "unlabelled"),
        _report(2, 20_000.0, "chronicle", upper=True, context="up to 20,000"),
        _report(3, 15_000.0, "modern_scholarly", url=_OVERRIDE_URL),
    ]

    kept = drop_superseded_by_override(reports, counts)

    assert [r.report_id for r in kept] == [3]
    assert counts.excluded_superseded_by_override == 2


def test_a_side_without_an_override_is_untouched_by_override_supersession() -> None:
    counts = ReconcileCounts()
    reports = [_report(1, 70_000.0, "modern_scholarly"), _report(2, 50_000.0, "modern_scholarly")]

    assert drop_superseded_by_override(reports, counts) == reports
    assert counts.excluded_superseded_by_override == 0


def test_a_non_modern_override_row_survives_the_modern_supersede_and_wins() -> None:
    counts = ReconcileCounts()
    reports = [
        _report(1, 77_460.0, "modern_scholarly", context="60,000 (modern estimates)"),
        _report(2, 17_000.0, "unlabelled", url=_OVERRIDE_URL),
    ]

    after_modern = drop_superseded_by_modern(reports, counts)
    assert [r.report_id for r in after_modern] == [1, 2]
    kept = drop_superseded_by_override(after_modern, counts)

    assert [r.report_id for r in kept] == [2]
    assert counts.excluded_superseded_by_modern == 0
    assert counts.excluded_superseded_by_override == 1


def test_the_override_counter_is_reported() -> None:
    assert ReconcileCounts(excluded_superseded_by_override=4).as_dict()[
        "excluded_superseded_by_override"
    ] == 4
