"""En-dash templates, "with <Name>" detachments and flag size parameters.

Each wikitext fragment is copied from a corpus 3 article (Glasgow, Roan's Tan
Yard, Fort Davidson, McDowell, Corbach, Siege of Tournai).
"""

from __future__ import annotations

import pytest

from pipeline.extractors.article import build_passages, clean_article_text
from pipeline.extractors.infobox import parse_infobox
from pipeline.extractors.textnorm import (
    classify_figure,
    parse_quantities,
    strip_markup,
)


def _infobox(**fields: str) -> str:
    body = "\n".join(f"| {name} = {value}" for name, value in fields.items())
    return f"{{{{Infobox military conflict\n| conflict = Test\n{body}\n}}}}\nBody."


def _troop_values(wikitext: str) -> list[list[float]]:
    extraction = parse_infobox(wikitext)
    assert extraction is not None
    return [[r.reported_value for r in side.troop_reports] for side in extraction.sides]


# ─── {{endash}} on the infobox path only ─────────────────────────────────────


@pytest.mark.parametrize(
    ("template", "raw"),
    [
        ("endash", "550{{endash}}800"),
        ("en dash", "550{{en dash}}800"),
        ("nd", "550{{nd}}800"),
        ("spnd", "550{{spnd}}800"),
        ("spaced ndash", "550{{spaced ndash}}800"),
        ("spaced en dash", "550{{spaced en dash}}800"),
    ],
)
def test_infobox_text_keeps_dash_templates(template: str, raw: str) -> None:
    text = strip_markup(raw, infobox=True)
    assert "–" in text, template
    assert text.replace(" ", "") == "550–800"


def test_default_strip_markup_still_drops_endash() -> None:
    assert strip_markup("550{{endash}}800") == "550800"


@pytest.mark.parametrize(
    ("field", "value", "bounds"),
    [
        ("strength1", "550{{endash}}800", (550, 800)),
        ("strength1", "{{Circa}} 450{{endash}}500", (450, 500)),
        ("casualties1", "4{{endash}}27", (4, 27)),
        ("casualties2", "80{{endash}}128", (80, 128)),
        ("casualties1", "500{{endash}}1,000", (500, 1000)),
        ("strength1", "{{circa}} 500{{endash}}532", (500, 532)),
    ],
)
def test_infobox_range_bounds_survive_endash(
    field: str, value: str, bounds: tuple[int, int]
) -> None:
    extraction = parse_infobox(_infobox(combatant1="A", combatant2="B", **{field: value}))
    assert extraction is not None
    side = extraction.sides[int(field[-1]) - 1]
    reports = side.troop_reports if field.startswith("strength") else side.casualty_reports
    assert {r.reported_value for r in reports} >= set(bounds)
    assert not any(r.reported_value in (5001, 550800, 450500, 427, 80128) for r in reports)


def test_endash_between_ref_and_sfn_pieces() -> None:
    value = "550<ref>x</ref>{{endash}}{{sfn|Smith|2001|p=3}}800"
    assert strip_markup(value, infobox=True) == "550–800"


def test_passage_text_is_unchanged_by_endash() -> None:
    raw = "The garrison numbered 550{{endash}}800 men.{{sfn|X|2001}} It held.\n"
    assert clean_article_text(raw) == "The garrison numbered 550800 men. It held."
    passages = build_passages("Test", raw)
    assert all("–" not in p.text for p in passages)


# ─── "with <Name>" detachments ───────────────────────────────────────────────


def _classify(text: str) -> str:
    quantities = parse_quantities(text)
    assert quantities, text
    return classify_figure(quantities[-1], first_in_field=False)


@pytest.mark.parametrize(
    "text",
    [
        "400 to 500 with Luckner",
        "800 led by Prince Frederick",
        "600 commanded by Wurmb",
    ],
)
def test_named_detachment_is_component(text: str) -> None:
    assert _classify(text) == "component"


@pytest.mark.parametrize(
    "text",
    ["30,000 with 20 guns", "20,000 with 40 guns", "12,000 with artillery"],
)
def test_with_equipment_is_not_a_component(text: str) -> None:
    # The guns make these arm-labelled today; the point is that "with" followed
    # by a number or a common noun never reads as a named detachment.
    quantity = parse_quantities(text)[0]
    assert classify_figure(quantity, first_in_field=True) != "component"


def test_with_plain_noun_stays_total() -> None:
    quantity = parse_quantities("5,000 with supplies")[0]
    assert classify_figure(quantity, first_in_field=True) == "total"


def test_under_behaviour_unchanged() -> None:
    assert _classify("15,000 under Ruechel") == "component"
    assert (
        classify_figure(parse_quantities("20,000 under arms")[0], first_in_field=True) == "total"
    )


# ─── Flag template size parameter ────────────────────────────────────────────


def test_flagcountry_size_param_is_not_the_country() -> None:
    wikitext = _infobox(
        combatant1="{{flagcountry|size=20px|Kingdom of France}}",
        combatant2="{{flagcountry|size=20px|Dutch Republic}}",
        strength1="72,000",
        strength2="~9,000",
    )
    extraction = parse_infobox(wikitext)
    assert extraction is not None
    assert [s.label for s in extraction.sides] == ["Kingdom of France", "Dutch Republic"]


def test_flag_bare_px_value_is_skipped() -> None:
    assert strip_markup("{{flag|20px|Spain}}", infobox=True) == "Spain"
    assert strip_markup("{{flag|Spain|size=20px}}", infobox=True) == "Spain"


def test_flag_passage_text_unchanged() -> None:
    assert strip_markup("{{flag|Spain|size=20px}}") == "Spain"
