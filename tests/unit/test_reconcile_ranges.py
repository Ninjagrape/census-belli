"""A stated range loaded as two bound rows becomes one point row."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from pipeline.reconcilers.claims import assign_lineages, merge_range_pairs
from pipeline.reconcilers.design import build_design
from pipeline.reconcilers.records import ReconcileCounts, Report, SourceKey

_KEY = SourceKey("wikipedia_infobox", "https://example/battle")


def _row(
    report_id: int,
    value: float,
    *,
    lower: bool = False,
    upper: bool = False,
    context: str = "914,000-1,067,500",
    side_id: int = 1,
    key: SourceKey = _KEY,
    quantity: str = "troops",
    branch: str = "total",
    casualty_type: str = "total",
    scope: str = "unknown",
) -> Report:
    return Report(
        report_id=report_id,
        side_id=side_id,
        battle_id=side_id,
        source_id=1,
        source_key=key,
        source_type=key.source_type,
        quantity=quantity,  # type: ignore[arg-type]
        branch=branch,
        casualty_type=casualty_type,
        scope=scope,
        reported_value=value,
        is_lower_bound=lower,
        is_upper_bound=upper,
        extracted_context=context,
        lineage_id=report_id,
    )


def _pair() -> list[Report]:
    return [_row(1, 914_000.0, lower=True), _row(2, 1_067_500.0, upper=True)]


def test_a_range_pair_becomes_one_point_row_at_the_log_midpoint() -> None:
    merged, n, _ = merge_range_pairs(_pair())

    assert n == 1 and len(merged) == 1
    row = merged[0]
    assert not row.is_lower_bound and not row.is_upper_bound
    assert math.isclose(math.log(row.reported_value), 0.5 * math.log(914_000.0 * 1_067_500.0))
    assert math.isclose(row.range_var, (math.log(1_067_500.0 / 914_000.0) / 4.0) ** 2)


def test_the_design_carries_range_var_and_counts_the_merge() -> None:
    counts = ReconcileCounts()

    design = build_design(_pair(), quantity="troops", ancient_cutoff_year=500, counts=counts)

    assert design.n_obs == 1 and list(design.point_rows) == [0]
    assert design.censor_lower_at.size == design.censor_upper_at.size == 0
    assert design.range_var[0] > 0.0
    assert counts.ranges_merged == 1
    assert counts.as_dict()["ranges_merged"] == 1


def test_merging_can_be_disabled_and_the_pair_stays_censored() -> None:
    design = build_design(_pair(), quantity="troops", ancient_cutoff_year=500, merge_ranges=False)

    assert design.n_obs == 2
    assert design.point_rows.size == 0
    assert np.all(design.range_var == 0.0)


def test_a_lone_bound_stays_censored_with_no_range_variance() -> None:
    design = build_design(
        [_row(1, 914_000.0, lower=True)], quantity="troops", ancient_cutoff_year=500
    )

    assert design.censor_upper_at.size == 1
    assert design.point_rows.size == 0
    assert design.range_var[0] == 0.0


def test_a_pair_with_different_contexts_is_not_merged() -> None:
    rows = [_row(1, 914_000.0, lower=True), _row(2, 1_067_500.0, upper=True, context="other")]

    merged, n, _ = merge_range_pairs(rows)

    assert n == 0 and merged == rows


def test_a_pair_on_different_sides_or_sources_is_not_merged() -> None:
    other = SourceKey("wikipedia_infobox", "https://example/other")

    assert (
        merge_range_pairs([_row(1, 9.0e5, lower=True), _row(2, 1.0e6, upper=True, side_id=2)])[1]
        == 0
    )
    assert (
        merge_range_pairs([_row(1, 9.0e5, lower=True), _row(2, 1.0e6, upper=True, key=other)])[1]
        == 0
    )


def test_an_inverted_range_is_not_merged() -> None:
    assert merge_range_pairs([_row(1, 2.0e6, lower=True), _row(2, 1.0e6, upper=True)])[1] == 0


def test_point_rows_pass_through_unchanged() -> None:
    point = _row(3, 5_000.0, context="5,000")

    merged, n, _ = merge_range_pairs([*_pair(), point])

    assert n == 1 and merged[-1] is point


def test_two_ranges_for_different_casualty_types_in_one_segment_merge_separately() -> None:
    context = "killed 1,000-2,000, wounded 3,000-5,000"
    rows = [
        _row(
            1, 1_000.0, lower=True, context=context, quantity="casualties", casualty_type="killed"
        ),
        _row(
            2, 2_000.0, upper=True, context=context, quantity="casualties", casualty_type="killed"
        ),
        _row(
            3, 3_000.0, lower=True, context=context, quantity="casualties", casualty_type="wounded"
        ),
        _row(
            4, 5_000.0, upper=True, context=context, quantity="casualties", casualty_type="wounded"
        ),
    ]

    merged, n, _ = merge_range_pairs(rows)

    assert n == 2 and len(merged) == 2
    by_id = {r.report_id: r for r in merged}
    assert by_id[1].casualty_type == "killed" and by_id[3].casualty_type == "wounded"
    assert math.isclose(by_id[1].reported_value, math.sqrt(1_000.0 * 2_000.0))
    assert math.isclose(by_id[3].reported_value, math.sqrt(3_000.0 * 5_000.0))


def test_a_lower_and_an_upper_bound_for_different_branches_are_not_merged() -> None:
    context = "infantry at least 10,000, cavalry up to 20,000"
    rows = [
        _row(1, 10_000.0, lower=True, context=context, branch="infantry"),
        _row(2, 20_000.0, upper=True, context=context, branch="cavalry"),
    ]

    merged, n, _ = merge_range_pairs(rows)

    assert n == 0 and merged == rows


def test_a_degenerate_range_becomes_one_point_row_with_no_extra_variance() -> None:
    rows = [_row(1, 5_000.0, lower=True), _row(2, 5_000.0, upper=True)]

    merged, n, n_degenerate = merge_range_pairs(rows)

    assert n == 1 and n_degenerate == 1 and len(merged) == 1
    assert merged[0].range_var == 0.0
    assert math.isclose(merged[0].reported_value, 5_000.0)
    assert not merged[0].is_lower_bound and not merged[0].is_upper_bound


def test_the_degenerate_count_reaches_the_reconcile_counts() -> None:
    counts = ReconcileCounts()
    rows = [_row(1, 5_000.0, lower=True), _row(2, 5_000.0, upper=True)]

    build_design(rows, quantity="troops", ancient_cutoff_year=500, counts=counts)

    assert counts.ranges_merged == 1 and counts.ranges_degenerate == 1
    assert counts.as_dict()["ranges_degenerate"] == 1


def test_a_wikidata_copy_of_the_upper_bound_does_not_double_count_the_range() -> None:
    wikidata = SourceKey("wikidata", "https://wikidata/Q1")
    rows = [*_pair(), _row(3, 1_067_500.0, context="", key=wikidata)]
    lineage = assign_lineages(rows)
    rows = [replace(r, lineage_id=lineage[r.report_id]) for r in rows]

    design = build_design(rows, quantity="troops", ancient_cutoff_year=500)

    assert design.n_obs == 1
    assert list(design.point_rows) == [0]
    assert design.range_var[0] > 0.0
    assert math.isclose(
        float(np.exp(design.log_y[0])), math.sqrt(914_000.0 * 1_067_500.0), rel_tol=1e-9
    )
