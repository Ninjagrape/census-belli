"""Infobox heading lines ("Seventh Coalition:") are neither sides nor people.

Fixtures are minimal and were written from the corpus3 audit (handover-log
36/37); they are not fetched from Wikipedia.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.extractors import infobox
from pipeline.extractors.infobox import parse_infobox
from pipeline.extractors.records import SideExtraction, SourceExtraction

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "extract"


def _parse(stem: str) -> SourceExtraction:
    """Parse a heading fixture's infobox.

    Args:
        stem: Fixture file stem.

    Returns:
        The extraction, asserted to be non-empty.
    """
    wikitext = (FIXTURES / f"{stem}.wikitext").read_text(encoding="utf-8")
    extraction = parse_infobox(wikitext, source_ref="test", source_title=stem)
    assert extraction is not None
    return extraction


HEADING_FIXTURES = [
    "quatre_bras_headings",
    "okinawa_headings",
    "timor_headings",
    "la_suffel_headings",
    "tomaszow_headings",
]


@pytest.mark.parametrize("stem", HEADING_FIXTURES)
def test_no_heading_survives_anywhere(stem: str) -> None:
    """No label, alias, polity or commander name ends with a colon."""
    for side in _parse(stem).sides:
        texts = [side.label, *side.aliases, side.polity or ""]
        texts += [mention.name for mention in side.commanders]
        texts += [mention.side_label for mention in side.commanders]
        assert [t for t in texts if t.rstrip().endswith(":")] == []


@pytest.mark.parametrize(
    ("stem", "index", "label", "aliases"),
    [
        (
            "quatre_bras_headings",
            1,
            "United Kingdom of Great Britain and Ireland",
            ["Netherlands", "Brunswick", "Nassau"],
        ),
        ("okinawa_headings", 0, "United States", ["United Kingdom"]),
        ("timor_headings", 0, "Australia", ["Netherlands", "United Kingdom"]),
        ("la_suffel_headings", 1, "Austrian Empire", ["Württemberg"]),
    ],
)
def test_side_label_is_first_real_polity(
    stem: str, index: int, label: str, aliases: list[str]
) -> None:
    """The label is the first non-heading item; later items stay on the same side."""
    side = _parse(stem).sides[index]
    assert side.label == label
    assert side.polity == label
    assert side.aliases == aliases


def test_headings_do_not_add_sides() -> None:
    """A heading line stays inside its side rather than opening a new one."""
    assert [len(_parse(stem).sides) for stem in HEADING_FIXTURES] == [2] * 5


def test_commander_heading_skipped_and_label_follows_side() -> None:
    """Quatre Bras commanders are the two people, tagged with the real label."""
    side = _parse("quatre_bras_headings").sides[1]
    assert [(m.name, m.infobox_order) for m in side.commanders] == [
        ("Duke of Wellington", 0),
        ("William, Prince of Orange", 1),
    ]
    assert {m.side_label for m in side.commanders} == {side.label}


def test_commander_order_contiguous_over_people() -> None:
    """Tomaszow phase headings are dropped and the order has no gaps."""
    side = _parse("tomaszow_headings").sides[0]
    assert [(m.name, m.infobox_order) for m in side.commanders] == [
        ("Name A", 0),
        ("Name B", 1),
    ]


def test_all_headings_falls_back_to_first_heading_without_colon() -> None:
    """A side made only of headings is better named by one than by "side N"."""
    wikitext = (
        "{{Infobox military conflict\n| conflict = X\n"
        "| combatant1 = '''Allies:'''<br />'''Axis:'''\n| combatant2 = [[France]]\n}}"
    )
    extraction = parse_infobox(wikitext, source_ref="t", source_title="X")
    assert extraction is not None
    first: SideExtraction = extraction.sides[0]
    assert first.label == "Allies"
    assert first.polity == "Allies"
    assert first.aliases == []


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        ("Seventh Coalition:", True),
        ("Ground forces: ", True),
        ("United States", False),
        ("Prussia: 1806", False),
    ],
)
def test_is_heading(item: str, expected: bool) -> None:
    """Only a trailing colon marks a heading."""
    assert infobox._is_heading(item) is expected
