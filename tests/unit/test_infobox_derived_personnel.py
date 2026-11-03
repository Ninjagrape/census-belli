"""A derived infobox total sums only figures that count people.

Pins the corpus3 audit cases (Okinawa kamikazes, Croix d'Orade engineers) and
the equipment vocabulary that keeps horses, aircraft and batteries out of a
headcount.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.extractors.infobox import parse_infobox
from pipeline.extractors.textnorm import is_equipment_count, parse_quantities

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "extract"
DERIVED = "infobox:derived_total"


def _side_reports(wikitext: str, side: int = 0) -> list:
    extraction = parse_infobox(wikitext, source_title="Fixture")
    assert extraction is not None
    return extraction.sides[side].troop_reports


def _derived(reports: list) -> float | None:
    found = [r.reported_value for r in reports if r.page_or_section == DERIVED]
    assert len(found) <= 1
    return found[0] if found else None


def _strength(text: str) -> list:
    wikitext = (
        "{{Infobox military conflict\n| conflict=B\n"
        "| combatant1={{flagcountry|France}}\n| combatant2={{flagcountry|Spain}}\n"
        f"| strength1={text}\n| strength2=1,000\n}}}}\n"
    )
    return _side_reports(wikitext)


def _fixture(stem: str, side: int) -> list:
    return _side_reports((FIXTURES / f"{stem}.wikitext").read_text(encoding="utf-8"), side)


def test_okinawa_kamikazes_are_not_summed_as_men() -> None:
    assert _derived(_fixture("okinawa_strength", 1)) == 116_000


def test_croix_dorade_engineers_and_sappers_join_the_arms() -> None:
    assert _derived(_fixture("croix_dorade_strength", 0)) == 950
    assert _derived(_fixture("croix_dorade_strength", 1)) == 1830


def test_stored_branches_are_unchanged() -> None:
    uk = [(r.branch, r.reported_value) for r in _fixture("croix_dorade_strength", 0)]
    assert ("other", 150.0) in uk and ("other", 200.0) in uk
    fr = [(r.branch, r.reported_value) for r in _fixture("croix_dorade_strength", 1)]
    assert ("other", 30.0) in fr and ("other", 50.0) in fr
    jp = [(r.branch, r.reported_value) for r in _fixture("okinawa_strength", 1)]
    assert jp[:3] == [("other", 76_000.0), ("other", 40_000.0), ("other", 850.0)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("5,000 Spaniards<br>3,000 Dutch<br>120 horses", 8_000),
        ("5,000 Spaniards<br>3,000 Dutch<br>600 helicopters", 8_000),
        ("5,000 Spaniards<br>3,000 Dutch<br>200 batteries", 8_000),
        ("10,000 infantry<br>8,000 cavalry<br>850 Japanese Navy and Army aircraft", 18_000),
        ("10,000 infantry<br>8,000 cavalry<br>400 machine-guns", 18_000),
        ("10,000 infantry<br>8,000 cavalry<br>12 batteries", 18_000),
        ("10,000 infantry<br>8,000 cavalry<br>6 helicopters", 18_000),
    ],
)
def test_equipment_is_never_summed(text: str, expected: float) -> None:
    assert _derived(_strength(text)) == expected


def test_men_and_guns_is_personnel() -> None:
    (quantity,) = parse_quantities("3,000 men and guns")
    assert not is_equipment_count(quantity)


def test_small_personnel_count_is_summable() -> None:
    assert _derived(_strength("5,000 infantry<br>50 cavalry")) == 5_050


def test_small_unqualified_figure_stays_unsafe() -> None:
    assert _derived(_strength("5,000 Spaniards<br>3,000 Dutch<br>60 Greeks")) == 8_000
    # A bare figure with no noun is an orphan, which blocks any derived sum.
    assert _derived(_strength("5,000 Spaniards<br>3,000 Dutch<br>60")) is None


def test_contingents_versus_arms_is_unchanged() -> None:
    reports = _strength("25,000 Romans<br>10,000 allies<br>8,000 cavalry")
    assert _derived(reports) == 35_000
