"""Regime-labelled totals, scope wording, phases and arm/man wording in infobox strengths.

Every strength string below is copied from the wikitext Wikipedia served for
the named battle (``data/raw/battles_html``), because earlier fixtures that
encoded the parser's own assumptions hid a systematic bug (see
docs/handover/04-the-lesson.md). Each case asserts the branch, scope and bound
flags of every figure on the side, not just the one a change is about.
"""

from __future__ import annotations

import pytest

from pipeline.extractors.infobox import parse_infobox
from pipeline.extractors.records import SCOPES, TroopReport
from pipeline.extractors.textnorm import (
    classify_figure,
    infer_scope,
    parse_quantities,
    resolve_scope,
)

Row = tuple[float, str, str, str]  # value, branch, scope, bounds ("", "L", "U")


def _bounds(report: TroopReport) -> str:
    return ("L" if report.is_lower_bound else "") + ("U" if report.is_upper_bound else "")


def _sides(strength1: str, strength2: str) -> tuple[list[TroopReport], list[TroopReport]]:
    wikitext = (
        "{{Infobox military conflict\n| conflict = Test\n| combatant1 = [[A]]\n"
        f"| combatant2 = [[B]]\n| strength1 = {strength1}\n| strength2 = {strength2}\n"
        "}}\n"
    )
    extraction = parse_infobox(wikitext, source_title="Test")
    assert extraction is not None
    by_label = {side.label: side.troop_reports for side in extraction.sides}
    return by_label["A"], by_label["B"]


def _rows(reports: list[TroopReport]) -> list[Row]:
    return [(r.reported_value, r.branch, r.scope, _bounds(r)) for r in reports]


# --- Decision A: labelled modern estimates are totals ---------------------------


def test_hazir_modern_estimate_is_a_total_beside_the_muslim_source_total() -> None:
    _, byzantine = _sides(
        "17,000 (Muslim source){{sfn|Crawford|2013|p=149}}",
        "70,000 (Muslim source){{sfn|Crawford|2013|p=149}} "
        "~7,000 (Modern estimate){{sfn|Crawford|2013|p=149}}",
    )
    assert _rows(byzantine) == [
        (70_000.0, "total", "engaged", ""),
        (7_000.0, "total", "engaged", ""),
    ]


def test_chains_modern_range_keeps_its_bounds_as_totals() -> None:
    _, sasanian = _sides("18,000", "80,000 (Islamic estimates) 15,000–20,000 (modern estimates)")
    assert sorted(_rows(sasanian)) == sorted(
        [
            (15_000.0, "total", "engaged", "L"),
            (20_000.0, "total", "engaged", "U"),
            (80_000.0, "total", "engaged", ""),
        ]
    )


def test_ullais_modern_estimation_is_a_total() -> None:
    _, sasanian = _sides(
        '18,000<ref name="Ikram1"/>',
        "70,000 (Arab estimates)<br/>30,000{{sfn|Crawford|2013|p=105}} (modern estimation)",
    )
    assert _rows(sasanian) == [
        (70_000.0, "total", "engaged", ""),
        (30_000.0, "total", "engaged", ""),
    ]


def test_beneventum_epirus_modern_estimate_is_a_total_and_elephants_are_not() -> None:
    rome, epirus = _sides(
        "20,000-25,000 (modern estimates)",
        "20,000 (Plutarch)<br>~40,000 (modern estimates)<br>At least 10 elephants",
    )
    assert _rows(rome) == [
        (20_000.0, "total", "engaged", "L"),
        (25_000.0, "total", "engaged", "U"),
    ]
    assert _rows(epirus) == [
        (20_000.0, "total", "engaged", ""),
        (40_000.0, "total", "engaged", ""),
        (10.0, "other", "engaged", "L"),
    ]


def test_persian_gate_modern_range_is_total_and_picked_fighters_are_men() -> None:
    macedonian, achaemenid = _sides(
        '17,000 picked fighters<ref name = "iranica">x</ref>',
        "\n* 40,000 infantry and 700 cavalry ([[Arrian]])\n"
        "* 700–2,000 (modern estimates)<ref name = iranica />",
    )
    assert _rows(macedonian) == [(17_000.0, "total", "engaged", "")]
    assert sorted(_rows(achaemenid)) == sorted(
        [
            (40_000.0, "infantry", "engaged", ""),
            (700.0, "cavalry", "engaged", ""),
            (700.0, "total", "engaged", "L"),
            (2_000.0, "total", "engaged", "U"),
        ]
    )


_ISSUS_PERSIAN = (
    "'''Modern estimate: 108,000–158,000'''<ref>x</ref><br>\n"
    "'''Ancient sources: 250,000–600,000'''<br>\n"
    "* 30,000–80,000 light infantry{{efn|Armed with Babylonian spears}}\n"
    "* 11,000 cavalry\n"
    "* 10,000 [[Persian Immortals]]\n"
    "* 40,000 infantry Armenians and 7,000 Armenian cavalry\n"
    "* 10,000 Greek mercenaries"
)


def test_issus_modern_estimate_is_the_side_total_and_no_lower_bound_total_is_derived() -> None:
    _, persian = _sides("'''Total: {{Circa|37,000}}'''", _ISSUS_PERSIAN)
    rows = _rows(persian)
    assert (108_000.0, "total", "engaged", "L") in rows
    assert (158_000.0, "total", "engaged", "U") in rows
    assert (250_000.0, "other", "engaged", "L") in rows
    assert (600_000.0, "other", "engaged", "U") in rows
    assert not [r for r in persian if r.page_or_section == "infobox:derived_total"]


# --- Decision D (scope) and E (under <Name>) ------------------------------------

_BIBRACTE_ROMAN = (
    "'''Caesar:'''\n40,000 troops:<ref name=Del1>x</ref>\n"
    "* 25,000–30,000 [[Legionary|legionaries]]\n"
    "* unknown number [[Auxiliaries (Roman military)|auxiliaries]]\n"
    "* unknown number allied infantry\n* 4,000 allied cavalry\n"
    "<br/>'''Engaged:''' 30,000 troops<ref name=Del1/><br>4 legions and auxiliaries"
)
_BIBRACTE_HELVETII = (
    "'''Tribes:'''<br>368,000 people:<br>90,000 warriors<br>278,000 non-combatants\n----\n"
    "'''Ancient historians:'''\n* [[Gaius Asinius Pollio|Pollio]]: 157,000 people\n"
    "* [[Orosius]]: 157,000 people\n* [[Livy]]: 157,000 people\n"
    "'''Modern historians:'''\n* [[Hans Delbrück|Delbrück]]: 12,000 warriors\n"
    "* Gilliver: 50,000 warriors"
)


def test_bibracte_helvetii_people_and_non_combatants_are_population_scope() -> None:
    roman, helvetii = _sides(_BIBRACTE_ROMAN, _BIBRACTE_HELVETII)
    assert _rows(roman) == [
        (40_000.0, "total", "engaged", ""),
        (25_000.0, "infantry", "engaged", "L"),
        (30_000.0, "infantry", "engaged", "U"),
        (4_000.0, "cavalry", "engaged", ""),
        (30_000.0, "other", "engaged", ""),
    ]
    assert _rows(helvetii) == [
        (368_000.0, "other", "population", ""),
        (90_000.0, "total", "engaged", ""),
        (278_000.0, "other", "population", ""),
        (157_000.0, "other", "population", ""),
        (157_000.0, "other", "population", ""),
        (157_000.0, "other", "population", ""),
        (12_000.0, "other", "engaged", ""),
        (50_000.0, "other", "engaged", ""),
    ]


_JENA_FRANCE = (
    "40,000 deployed to do battle, 96,000 total <small>(Jena)</small>"
    "{{sfn|Connelly|2006|pages=98–101}}<br>26,000 <small>(Auerstedt)</small>"
    "{{sfn|Connelly|2006|page=101}}<br>'''Total engaged''':<br>66,000"
)
_JENA_PRUSSIA = (
    "{{tree list}}\n* 46,000 <small>(Jena)</small>{{sfn|Chandler|1966|p=488}}\n"
    "** 40,000 directly under Hohenlohe, with 33,000 of them engaged\n"
    "** 15,000 under Rüchel, with 13,000 of them engaged\n{{tree list/end}}\n"
    "64,000 <small>(Auerstedt)</small>{{sfn|Chandler|1966|p=488}}"
    "<br>'''Total engaged''':<br>110,000"
)


def test_jena_deployed_is_engaged_and_the_larger_total_is_available() -> None:
    france, _ = _sides(_JENA_FRANCE, "1")
    assert _rows(france) == [
        (40_000.0, "total", "engaged", ""),
        (96_000.0, "total", "available", ""),
        (26_000.0, "other", "engaged", ""),
        (66_000.0, "other", "engaged", ""),
    ]


def test_jena_prussia_figure_under_a_named_commander_is_a_component() -> None:
    _, prussia = _sides("1", _JENA_PRUSSIA)
    assert _rows(prussia) == [
        (46_000.0, "total", "engaged", ""),
        (40_000.0, "other", "engaged", ""),
        (33_000.0, "other", "engaged", "U"),
        (15_000.0, "other", "engaged", ""),
        (13_000.0, "other", "engaged", "U"),
        (64_000.0, "other", "engaged", ""),
        (110_000.0, "other", "engaged", ""),
    ]


# --- Decision D (phase headings): Leipzig ---------------------------------------

_LEIPZIG_RUSSIA = (
    "'''16–17 October:'''<br />257,000{{sfn|Chandler|1966|p=922 ''(the strength of "
    "Schwarzenberg and Blücher)''}}<br />1,145 guns<ref name=Millar>x</ref>"
    "{{efn|The Army of the North was not on the battlefield.}}<br />\n"
    "'''18–19 October:'''<br />359,000<ref name=Millar/><br />1,500 guns"
    "{{sfn|Chandler|1966|p=932}}<br>Total in service:\n"
    "* {{flagicon|Russian Empire}} 150,000\n* {{flagicon|Austrian Empire}} 115,000\n"
    "* {{flagicon|Kingdom of Prussia|1803}} 90,000\n* {{flagicon|Sweden|naval}} 30,000\n"
    "* {{flagicon|Mecklenburg-Schwerin}} 6,149{{sfn|Hofschröer|2012|p=27}}"
)
_LEIPZIG_FRANCE = (
    "'''16–17 October:'''<br />177,000{{sfn|Chandler|1966|p=922}}<br />700 guns"
    "{{sfn|napoleonguide|2010}}<br />\n'''18–19 October:'''<br />195,000"
    "{{sfn|Chandler|1966|p=922}}<br />900 guns{{sfn|Chandler|1966|p=932}}"
    "<br>Total in service:\n* {{flagicon|First French Empire}} 160,000\n"
    "* {{flagicon|Napoleonic Italy}} 9,000\n* {{flagicon|Baden|1806}} 5,000"
)


def test_leipzig_largest_phase_figure_is_the_total_and_the_other_phase_is_a_component() -> None:
    russia, france = _sides(_LEIPZIG_RUSSIA, _LEIPZIG_FRANCE)
    assert _rows(russia) == [
        (257_000.0, "other", "engaged", ""),
        (1_145.0, "artillery", "engaged", ""),
        (359_000.0, "total", "engaged", ""),
        (1_500.0, "artillery", "engaged", ""),
        (150_000.0, "other", "engaged", ""),
        (115_000.0, "other", "engaged", ""),
        (90_000.0, "other", "engaged", ""),
        (30_000.0, "other", "engaged", ""),
        (6_149.0, "other", "engaged", ""),
    ]
    assert _rows(france) == [
        (177_000.0, "other", "engaged", ""),
        (700.0, "artillery", "engaged", ""),
        (195_000.0, "total", "engaged", ""),
        (900.0, "artillery", "engaged", ""),
        (160_000.0, "other", "engaged", ""),
        (9_000.0, "other", "engaged", ""),
        (5_000.0, "other", "engaged", ""),
    ]


def test_leipzig_phase_heading_is_kept_in_the_context_of_each_figure() -> None:
    russia, _ = _sides(_LEIPZIG_RUSSIA, "1")
    by_value = {r.reported_value: r.extracted_context for r in russia}
    assert "16–17 October" in by_value[257_000.0]
    assert "18–19 October" in by_value[359_000.0]
    assert "18–19 October" in by_value[1_500.0]


def test_phase_headings_in_words_are_attached_to_the_figures_that_follow() -> None:
    quantities = parse_quantities(
        "Initial strength:\n30,000\nOn day 2:\n45,000", default_scope="engaged"
    )
    assert [(q.value, q.phase) for q in quantities] == [
        (30_000.0, "Initial strength"),
        (45_000.0, "On day 2"),
    ]


def test_a_single_phase_is_not_demoted() -> None:
    side, _ = _sides("'''16–17 October:'''<br />257,000", "1")
    assert _rows(side) == [(257_000.0, "total", "engaged", "")]


# --- Scope wording ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("40,000 deployed to do battle", "engaged"),
        ("368,000 people:", "population"),
        ("278,000 non-combatants", "population"),
        ("the population of the city was 20,000", "population"),
        ("60,000 including non-combatants", "population"),
        ("9,000 civilians", "population"),
        ("30,000 men were engaged in the battle", "engaged"),
    ],
)
def test_infer_scope_reads_deployed_and_population_wording(text: str, expected: str) -> None:
    assert infer_scope(text) == expected


def test_population_is_a_scope_vocabulary_member_and_survives_resolve_scope() -> None:
    assert "population" in SCOPES
    assert resolve_scope("engaged", "278,000 non-combatants", "engaged") == "population"


def test_a_lone_total_keeps_the_field_default_scope() -> None:
    (quantity,) = parse_quantities("Total: 37,000", default_scope="engaged")
    assert quantity.scope == "engaged"


def test_a_total_beside_a_deployed_figure_is_available_but_the_deployed_one_is_not() -> None:
    deployed, total = parse_quantities(
        "40,000 deployed to do battle, 96,000 total", default_scope="engaged"
    )
    assert (deployed.scope, total.scope) == ("engaged", "available")


def test_a_population_word_scopes_only_its_own_figure() -> None:
    soldiers, civilians = parse_quantities(
        "30,000 men and 10,000 civilians", default_scope="engaged"
    )
    assert (soldiers.scope, civilians.scope) == ("engaged", "population")


# --- Decisions E and F at the function level -------------------------------------


def test_a_figure_under_a_named_commander_is_a_component_not_a_total() -> None:
    quantity = parse_quantities("15,000 under Rüchel, with 13,000 of them engaged")[0]
    assert classify_figure(quantity, first_in_field=True) == "component"


def test_a_figure_under_arms_is_still_a_side_figure() -> None:
    (quantity,) = parse_quantities("20,000 under arms")
    assert classify_figure(quantity, first_in_field=True) == "total"


def test_fighters_are_men_unless_the_field_speaks_of_aircraft() -> None:
    (men,) = parse_quantities("17,000 picked fighters")
    assert men.branch == "total"
    planes = parse_quantities("300 fighters\n200 bombers")
    assert [q.branch for q in planes] == ["air", "air"]
    (jets,) = parse_quantities("150 fighter aircraft")
    assert jets.branch == "air"


@pytest.mark.parametrize(
    "text",
    ["155,900–200,000 Polish People's Army", "2,000 civilian levies", "5,000 Peoples' Militia"],
)
def test_names_and_armed_civilians_are_not_population(text: str) -> None:
    assert infer_scope(text) == "unknown"


# --- Population wording is judged per figure, never from the whole clause ----------


@pytest.mark.parametrize(
    "sentence",
    [
        "The army of 30,000 men was drawn from the Boeotian people.",
        "Athens fielded 10,000 hoplites out of a population of 40,000.",
        "Caesar led 30,000 men; the civilians of the town were spared.",
        "20,000 Roman citizens and peoples allies",
    ],
)
def test_prose_with_population_words_beside_an_army_stays_engaged(sentence: str) -> None:
    assert resolve_scope("engaged", sentence, "engaged") == "engaged"


def test_a_prose_figure_counting_people_stays_population_when_declared() -> None:
    assert resolve_scope("population", "368,000 people", "engaged") == "population"


def test_extract_schema_accepts_the_population_scope() -> None:
    from pipeline.config import load_agent_spec

    spec = load_agent_spec("extract")
    schema = spec["prompt"]["output_schema"]
    scope = schema["properties"]["troop_reports"]["items"]["properties"]["scope"]
    assert "population" in scope["enum"]


@pytest.mark.parametrize(
    "text",
    [
        "15,000 men under Rüchel",
        "15,000 troops under Blücher",
        "15,000 soldiers under Rüchel",
        "15,000 under the command of Blücher",
        "15,000 men under command of Blücher",
        "15,000 men under the command of Blücher",
    ],
)
def test_a_figure_under_a_named_commander_is_a_component_with_a_head_noun(text: str) -> None:
    quantity = parse_quantities(text)[0]
    assert classify_figure(quantity, first_in_field=True) == "component"


@pytest.mark.parametrize("text", ["20,000 men under arms", "15,000 men", "15,000 men underway"])
def test_under_name_leaves_arms_and_bare_first_line_figures_alone(text: str) -> None:
    quantity = parse_quantities(text)[0]
    assert classify_figure(quantity, first_in_field=True) == "total"


# --- Phases: totals, key normalisation, per-field grouping -----------------------


def test_a_total_figure_after_phases_keeps_the_total_and_the_phases_become_components() -> None:
    side, _ = _sides(
        "16 October:<br />300,000<br />18 October:<br />100,000<br />Total: 250,000", "1"
    )
    assert [(r.reported_value, r.branch) for r in side] == [
        (300_000.0, "other"),
        (100_000.0, "other"),
        (250_000.0, "total"),
    ]


def test_a_total_worded_figure_inherits_no_phase() -> None:
    quantities = parse_quantities("16 October:\n300,000\nTotal: 250,000", default_scope="engaged")
    assert [(q.value, q.phase) for q in quantities] == [(300_000.0, "16 October"), (250_000.0, "")]


def test_phase_keys_ignore_a_leading_on_dash_style_and_spacing() -> None:
    side, _ = _sides("On 1 March 1944:<br />50,000<br />1  March 1944:<br />40,000", "1")
    assert [(r.reported_value, r.branch) for r in side] == [
        (50_000.0, "total"),
        (40_000.0, "total"),
    ]
    dashed, _ = _sides("16–17 October:<br />50,000<br />16-17 October:<br />40,000", "1")
    assert [r.branch for r in dashed] == ["total", "total"]


def test_phases_are_compared_within_a_strength_field_not_across_fields() -> None:
    wikitext = (
        "{{Infobox military conflict\n| conflict = Test\n| combatant1 = [[A]]\n"
        "| combatant2 = [[B]]\n| strength1 = 16 October:<br />50,000\n"
        "| forces1 = 18 October:<br />60,000\n| strength2 = 1\n}}\n"
    )
    extraction = parse_infobox(wikitext, source_title="Test")
    assert extraction is not None
    side = {s.label: s.troop_reports for s in extraction.sides}["A"]
    assert sorted((r.reported_value, r.branch) for r in side) == [
        (50_000.0, "total"),
        (60_000.0, "total"),
    ]
