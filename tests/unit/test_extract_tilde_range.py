"""A tilde between two numbers is a range; before one number it means approximately."""

from __future__ import annotations

import pytest

from pipeline.extractors.infobox import parse_infobox
from pipeline.extractors.textnorm import parse_quantities


@pytest.mark.parametrize(
    ("text", "bounds"),
    [
        ("500~1,400", (500.0, 1400.0)),
        ("3,000~5,000", (3000.0, 5000.0)),
        ("3,000 ~ 5,000", (3000.0, 5000.0)),
        ("3,000～5,000", (3000.0, 5000.0)),
        ("3,000〜5,000", (3000.0, 5000.0)),
    ],
)
def test_tilde_between_numbers_is_a_range(text: str, bounds: tuple[float, float]) -> None:
    quantities = parse_quantities(text)

    assert [q.value for q in quantities] == list(bounds)
    assert quantities[0].is_lower_bound and not quantities[0].is_upper_bound
    assert quantities[1].is_upper_bound and not quantities[1].is_lower_bound
    assert not any(q.is_estimate for q in quantities)


@pytest.mark.parametrize("text", ["~60,000", "~ 5,000"])
def test_leading_tilde_stays_a_single_estimate(text: str) -> None:
    (quantity,) = parse_quantities(text)

    assert quantity.is_estimate is True
    assert not quantity.is_lower_bound and not quantity.is_upper_bound


def test_approximate_dash_range_is_still_a_range() -> None:
    quantities = parse_quantities("~60,000–100,000")

    assert [q.value for q in quantities] == [60000.0, 100000.0]
    assert quantities[0].is_lower_bound and quantities[1].is_upper_bound
    assert all(q.is_estimate for q in quantities)


def test_infobox_tilde_ranges_become_bounded_reports() -> None:
    wikitext = (
        "{{Infobox military conflict\n| conflict = Battle of Yonan\n"
        "| combatant1 = A\n| combatant2 = B\n"
        "| strength1 = 500~1,400\n| strength2 = 3,000~5,000\n}}\n"
    )
    extraction = parse_infobox(wikitext)

    assert extraction is not None
    for side, bounds in zip(extraction.sides, [(500.0, 1400.0), (3000.0, 5000.0)], strict=True):
        assert [r.reported_value for r in side.troop_reports] == list(bounds)
        assert side.troop_reports[0].is_lower_bound
        assert side.troop_reports[1].is_upper_bound
