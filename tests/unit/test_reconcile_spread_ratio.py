"""The collapse statistic: do the estimates spread as widely as the reports do?

On the pilot corpus every side came out near 35,000 whatever its reports said,
with a clean convergence, so no sampler diagnostic could see it. This ratio is
what can.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pipeline.reconcilers.design import build_design
from pipeline.reconcilers.records import Report, SourceKey
from pipeline.reconcilers.summarise import spread_ratio


def _report(report_id: int, side_id: int, value: float, *, upper: bool = False) -> Report:
    return Report(
        report_id=report_id,
        side_id=side_id,
        battle_id=side_id,
        source_id=report_id,
        source_key=SourceKey("wikipedia_infobox", f"https://example/{side_id}"),
        source_type="wikipedia_infobox",
        quantity="troops",
        reported_value=value,
        is_upper_bound=upper,
        year_astronomical=1800,
        lineage_id=report_id,
    )


def _design(values: dict[int, float]):  # type: ignore[no-untyped-def]
    reports = [_report(i + 1, side, value) for i, (side, value) in enumerate(values.items())]
    return build_design(reports, quantity="troops", ancient_cutoff_year=500)


def test_estimates_that_track_the_reports_give_a_ratio_of_one() -> None:
    values = {1: 1_000.0, 2: 10_000.0, 3: 100_000.0}
    design = _design(values)

    ratio = spread_ratio(design, {s: math.log(v) for s, v in values.items()})

    assert ratio == pytest.approx(1.0)


def test_estimates_squashed_to_one_level_give_a_ratio_near_zero() -> None:
    values = {1: 1_000.0, 2: 10_000.0, 3: 2_500_000.0}
    design = _design(values)

    ratio = spread_ratio(design, {1: 10.5, 2: 10.51, 3: 10.49})

    assert ratio is not None and ratio < 0.01


def test_a_halved_spread_gives_a_ratio_of_one_half() -> None:
    values = {1: 1_000.0, 2: 10_000.0, 3: 100_000.0}
    design = _design(values)
    centre = float(np.mean([math.log(v) for v in values.values()]))

    halved = {s: centre + 0.5 * (math.log(v) - centre) for s, v in values.items()}

    ratio = spread_ratio(design, halved)

    assert ratio == pytest.approx(0.5)


def test_a_single_side_has_no_spread_to_compare() -> None:
    design = _design({1: 5_000.0})

    assert spread_ratio(design, {1: 8.5}) is None


def test_sides_with_only_bound_observations_are_left_out_of_both_spreads() -> None:
    reports = [
        _report(1, 1, 1_000.0),
        _report(2, 2, 100_000.0),
        _report(3, 3, 5.0e6, upper=True),
    ]
    design = build_design(reports, quantity="troops", ancient_cutoff_year=500)

    ratio = spread_ratio(design, {1: math.log(1_000.0), 2: math.log(100_000.0), 3: 3.0})

    assert ratio == pytest.approx(1.0)
