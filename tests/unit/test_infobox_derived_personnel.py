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


@pytest.mark.parametrize(
    "text",
    [
        "5,000 French infantry supported by artillery batteries",
        "7,000 cavalry mounted on horses",
        "4,000 light infantry transported by ships",
        "8,000 infantry and cannon",
        "1,200 Zouaves attached to the artillery batteries",
        "3,000 men and 40 guns",
        "900 dragoons with horses",
        "2,000 Hussars and lancers",
        "60 artillery",
    ],
)
def test_head_noun_personnel_is_not_equipment(text: str) -> None:
    (quantity, *_) = parse_quantities(text)
    assert not is_equipment_count(quantity)


@pytest.mark.parametrize(
    "text",
    [
        "40 guns",
        "120 tanks",
        "300 aircraft",
        "12 ships",
        "60 artillery pieces",
        "2,000 horses",
        "15 batteries",
        "850 Japanese Navy and Army aircraft",
        "30 light cruisers",
    ],
)
def test_head_noun_equipment_stays_equipment(text: str) -> None:
    (quantity, *_) = parse_quantities(text)
    assert is_equipment_count(quantity)


def test_arm_with_supporting_equipment_is_summed() -> None:
    text = "5,000 infantry supported by artillery batteries<br>2,000 cavalry"
    assert _derived(_strength(text)) == 7_000


def _branches(text: str) -> list[tuple[float, str]]:
    return [(r.reported_value, r.branch) for r in _strength(text)]


@pytest.mark.parametrize(
    "text",
    [
        "15 ships of the line<br>6 frigates<br>4 brigs<br>1 aviso",
        "14 ships of the line<br>1 frigate<br>1 sloop<br>1 fireship",
        "5 ships of the line<br>1 sloop<br>6 bomb ketches",
        "2 schooners<br>1 steamship",
        "37th Infantry Regiment",
        "1 lugger",
    ],
)
def test_ship_or_unit_counts_never_become_the_side_total(text: str) -> None:
    assert not [v for v, b in _branches(text) if b == "total"]


@pytest.mark.parametrize("text", ["1 aviso", "1 fireship", "6 bomb ketches", "30 merchantmen"])
def test_more_vessel_words_are_equipment(text: str) -> None:
    (quantity, *_) = parse_quantities(text)
    assert is_equipment_count(quantity)


def test_lone_small_personnel_figure_is_still_promoted() -> None:
    assert (50.0, "total") in _branches("50 French troops")


def test_actium_style_infantry_on_ships_keeps_arms_and_no_tiny_total() -> None:
    text = "250 galleys<br>16,000 infantry on ships<br>3,000 archers"
    assert (3_000.0, "total") not in _branches(text)
