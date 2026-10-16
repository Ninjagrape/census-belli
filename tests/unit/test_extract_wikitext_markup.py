"""Wikitext markup cleaning, from patterns found on the real corpus.

Crawl began saving ``.wikitext`` files on 2026-10-15, so the wikitext parser
met real infoboxes for the first time then. Each fixture below is a field
copied from one of those articles, trimmed to the pattern under test. The
rendered-HTML parse of the same article is the reference: the wikitext path
must yield at least what it did.
"""

from __future__ import annotations

import pytest

from pipeline.extractors.infobox import parse_infobox, parse_infobox_html
from pipeline.extractors.textnorm import strip_markup


def _infobox(**fields: str) -> str:
    body = "\n".join(f"| {name} = {value}" for name, value in fields.items())
    return f"{{{{Infobox military conflict\n| conflict = Test\n{body}\n}}}}\nBody."


# ─── File and image links ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("[[File:Flag of the Lebanese Forces.svg|23px]] [[Elie Hobeika]]", "Elie Hobeika"),
        ("[[File:Flag of Palestine.svg|23px|border]] [[Yasser Arafat]]", "Yasser Arafat"),
        ("[[File:Flag.svg|border|25px]] [[Robert E. Lee]]", "Robert E. Lee"),
        ("[[Image:Flag.svg|20px|link=]] [[A. P. Hill]]", "A. P. Hill"),
        ("[[file:Flag.svg|border|18px]] [[Lloyd Tilghman]]", "Lloyd Tilghman"),
        ("[[File:Flag.svg|20px|alt=[[Union]] flag]] [[XVIII Corps]]", "XVIII Corps"),
    ],
)
def test_file_links_are_dropped_entirely(raw: str, expected: str) -> None:
    assert strip_markup(raw) == expected


def test_ordinary_links_keep_their_display_text() -> None:
    assert strip_markup("[[Charles-Pierre Augereau|Pierre Augereau]]") == "Pierre Augereau"


def test_file_link_inside_a_list_template_does_not_leak() -> None:
    raw = "{{ubl|[[File:Flag.svg|23px|border]] [[Saad Sayel]]{{KIA}}|[[File:F.svg|23px]] [[Ali]]}}"
    assert strip_markup(raw) == "Saad Sayel (KIA)\nAli"


def test_file_link_in_commander_field_resolves_to_clean_name() -> None:
    wikitext = _infobox(
        commander1="[[File:Flag of the Army of Northern Virginia.svg|border|25px]] "
        "[[Robert E. Lee]]<br />[[File:Flag.svg|border|25px]] [[A.P. Hill]]{{KIA}}",
        strength1="[[File:Flag.svg|border|25px]] [[Army of the Potomac]] 100,000",
    )
    extraction = parse_infobox(wikitext)
    assert extraction is not None
    names = [(m.name, m.note) for s in extraction.sides for m in s.commanders]
    assert names == [("Robert E. Lee", ""), ("A.P. Hill", "KIA")]
    # "25px" is not a headcount.
    values = [r.reported_value for s in extraction.sides for r in s.troop_reports]
    assert values == [100000.0]


# ─── Flag templates ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw",
    [
        "{{flagicon|Israel|tsahal}} Yitzhak Rabin",
        "{{flagicon image|Flag of Druze.svg|size=20px}} Yitzhak Rabin",
        "{{flagdeco|First French Republic}} Yitzhak Rabin",
        "{{symbol|Haganah Symbol.svg}} Yitzhak Rabin",
    ],
)
def test_icon_only_templates_render_nothing(raw: str) -> None:
    assert strip_markup(raw) == "Yitzhak Rabin"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("{{flag|Israel|size=22px}}", "Israel"),
        ("{{flagcountry|Kingdom of Egypt}}", "Kingdom of Egypt"),
        ("{{flag|France}} 20,000", "France 20,000"),
    ],
)
def test_flag_templates_render_the_country_name(raw: str, expected: str) -> None:
    assert strip_markup(raw) == expected


def test_combatant_label_is_clean() -> None:
    wikitext = _infobox(
        combatant1="'''{{flag|Israel|size=22px}}'''",
        combatant2=(
            "[[File:Flag of the Lebanese Forces.svg|23px]] "
            "[[Lebanese Forces (militia)|Lebanese Forces]]"
        ),
    )
    extraction = parse_infobox(wikitext)
    assert extraction is not None
    assert [s.label for s in extraction.sides] == ["Israel", "Lebanese Forces"]


# ─── List and wrapper templates (the troop-report drop) ──────────────────────


@pytest.mark.parametrize(
    "template",
    ["Indented plainlist", "unbulletedlist", "ublist", "ubli", "bulletedlist", "blist", "ubl"],
)
def test_list_templates_yield_one_commander_per_item(template: str) -> None:
    wikitext = _infobox(
        commander1=f"{{{{{template}|[[Napoleon Bonaparte]]|[[Jean Lannes]]{{{{WIA}}}}"
        "|[[André Masséna]]}}"
    )
    extraction = parse_infobox(wikitext)
    assert extraction is not None
    names = [(m.name, m.note) for s in extraction.sides for m in s.commanders]
    assert names == [("Napoleon Bonaparte", ""), ("Jean Lannes", "WIA"), ("André Masséna", "")]


def test_indented_plainlist_with_bullets_and_markers() -> None:
    wikitext = _infobox(
        commander1="{{Indented plainlist|\n* '''[[Napoleon Bonaparte]]'''\n"
        "* [[Jean-Baptiste Muiron]]{{KIA}}\n"
        "* {{ill|Jean Gilles André Robert|fr|Jean Gilles André Robert}}{{DOW}}\n}}"
    )
    extraction = parse_infobox(wikitext)
    assert extraction is not None
    names = [(m.name, m.note) for s in extraction.sides for m in s.commanders]
    assert names == [
        ("Napoleon Bonaparte", ""),
        ("Jean-Baptiste Muiron", "KIA"),
        ("Jean Gilles André Robert", "DOW"),
    ]


def test_nested_blist_strength_keeps_every_figure() -> None:
    wikitext = _infobox(
        strength1="'''Total: 86,400'''{{blist\n | 15,000 light infantry"
        "{{blist\n  | [[Velites]]\n }}"
        "\n | 55,000 infantry{{blist\n  | 25,000 [[Roman people|Romans]]\n"
        "  | 30,000 [[Socii|allies]]\n }}\n | 6,400 cavalry\n}}"
    )
    extraction = parse_infobox(wikitext)
    assert extraction is not None
    values = {r.reported_value for s in extraction.sides for r in s.troop_reports}
    assert {86400.0, 15000.0, 55000.0, 25000.0, 30000.0, 6400.0} <= values


def test_ublist_with_refs_and_breaks_keeps_every_estimate() -> None:
    wikitext = _infobox(
        strength2='{{ublist|80,000<ref>Carey, p. 124</ref><br />50,000<ref name="S">Sverdrup.'
        "</ref><br />25,000<ref>{{citation |last=Markó |title=Great Honours}}</ref>}}"
    )
    extraction = parse_infobox(wikitext)
    assert extraction is not None
    values = sorted(r.reported_value for s in extraction.sides for r in s.troop_reports)
    assert values == [25000.0, 50000.0, 80000.0]


def test_center_wrapper_keeps_strength_and_drops_flagdeco() -> None:
    wikitext = _infobox(
        strength1="{{center|Overall:<br/>{{flagdeco|First French Empire}} 48,000.<br/>"
        "<strong>Craonne:</strong><br/>{{flagdeco|First French Empire}} 30,000; 102 guns."
        "{{sfn|Pigeard|2002|pp=648-649}}}}"
    )
    extraction = parse_infobox(wikitext)
    assert extraction is not None
    values = {r.reported_value for s in extraction.sides for r in s.troop_reports}
    assert {48000.0, 30000.0} <= values


def test_bdl_wrap_parameter_keeps_definition_list_figures() -> None:
    wikitext = _infobox(
        combatant1="Rome",
        casualties1="'''67,500-80,000'''{{bdl|wrap=\n; [[Livy]]\n: 48,200 killed\n"
        ": 19,300 captured\n; [[Polybius]]\n: 70,000 killed\n}}",
    )
    extraction = parse_infobox(wikitext)
    assert extraction is not None
    values = {r.reported_value for s in extraction.sides for r in s.casualty_reports}
    assert {48200.0, 19300.0, 70000.0} <= values


def test_collapsible_list_title_is_not_a_commander() -> None:
    wikitext = _infobox(
        commander1="[[Napoleon|Napoleon I]] {{Collapsible list|bullets=yes|title=''... list''\n"
        "| [[Louis-Alexandre Berthier]]\n| [[Michel Ney]]}}"
    )
    extraction = parse_infobox(wikitext)
    assert extraction is not None
    names = [m.name for s in extraction.sides for m in s.commanders]
    assert "yes" not in names
    assert names[-2:] == ["Louis-Alexandre Berthier", "Michel Ney"]


def test_approximately_template_marks_an_estimate() -> None:
    wikitext = _infobox(strength1="{{approximately|109,500}}")
    extraction = parse_infobox(wikitext)
    assert extraction is not None
    report = extraction.sides[0].troop_reports[0]
    assert report.reported_value == 109500.0
    assert report.is_estimate


def test_ndash_and_bullet_templates_render_glyphs() -> None:
    assert strip_markup("2,000{{ndash}}3,000") == "2,000–3,000"
    assert strip_markup("{{*}}37,654 regulars") == "•37,654 regulars"


# ─── Parity with the rendered-HTML path ──────────────────────────────────────


def test_wikitext_matches_html_on_a_list_templated_infobox() -> None:
    wikitext = _infobox(
        combatant1="{{flag|France}}",
        combatant2="{{flag|Austria}}",
        commander1="{{Indented plainlist|\n* [[Napoleon Bonaparte]]\n* [[André Masséna]]\n}}",
        commander2="{{Indented plainlist|\n* [[József Alvinczi]]\n}}",
        strength1="{{ublist|20,000 infantry|1,000 cavalry}}",
        strength2="24,000",
    )
    html = (
        "<table class='infobox'>"
        "<tr><th colspan='2'>Belligerents</th></tr><tr><td>France</td><td>Austria</td></tr>"
        "<tr><th colspan='2'>Commanders and leaders</th></tr>"
        "<tr><td>Napoleon Bonaparte<br/>André Masséna</td><td>József Alvinczi</td></tr>"
        "<tr><th colspan='2'>Strength</th></tr>"
        "<tr><td>20,000 infantry<br/>1,000 cavalry</td><td>24,000</td></tr></table>"
    )
    from_wikitext = parse_infobox(wikitext)
    from_html = parse_infobox_html(html)
    assert from_wikitext is not None
    assert from_html is not None

    def shape(extraction):  # type: ignore[no-untyped-def]
        return [
            (
                [m.name for m in side.commanders],
                sorted((r.reported_value, r.branch) for r in side.troop_reports),
            )
            for side in extraction.sides
        ]

    assert shape(from_wikitext) == shape(from_html)
