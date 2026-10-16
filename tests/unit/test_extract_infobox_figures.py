"""Branch labelling of infobox troop figures, outcome matching and spacing.

The strength strings below are copied from the rendered infoboxes in the
real pilot corpus (Cannae, Waterloo, Chaeronea, Lodi, Leipzig). An earlier
set of fixtures encoded the parser's own assumptions and hid a systematic
bug, so the wording here is the wording Wikipedia served.
"""

from __future__ import annotations

import pytest

from pipeline.extractors.infobox import _parse_result, parse_infobox_html
from pipeline.extractors.merger import merge_battle
from pipeline.extractors.records import (
    BattleFacts,
    CommanderMention,
    Provenance,
    SideExtraction,
    SourceExtraction,
    TroopReport,
)
from pipeline.extractors.textnorm import parse_quantities, tidy_spacing

_INFOBOX_PROV = Provenance(source_type="wikipedia_infobox", extraction_method="infobox_parser")


def _cell(lines: str) -> str:
    return lines.replace("\n", "<br/>")


def _html(
    *,
    sides: tuple[str, str],
    strengths: tuple[str, str],
    commanders: tuple[str, str] = ("", ""),
    result: str = "",
) -> str:
    rows = []
    if result:
        rows.append(f"<tr><th>Result</th><td>{result}</td></tr>")
    rows.append("<tr><th colspan='2'>Belligerents</th></tr>")
    rows.append(f"<tr><td>{_cell(sides[0])}</td><td>{_cell(sides[1])}</td></tr>")
    if any(commanders):
        rows.append("<tr><th colspan='2'>Commanders and leaders</th></tr>")
        rows.append(f"<tr><td>{_cell(commanders[0])}</td><td>{_cell(commanders[1])}</td></tr>")
    rows.append("<tr><th colspan='2'>Strength</th></tr>")
    rows.append(f"<tr><td>{_cell(strengths[0])}</td><td>{_cell(strengths[1])}</td></tr>")
    return f"<html><body><table class='infobox'>{''.join(rows)}</table></body></html>"


def _reports(html: str) -> dict[str, list[TroopReport]]:
    extraction = parse_infobox_html(html, source_ref="t", source_title="t")
    assert extraction is not None
    return {side.label: side.troop_reports for side in extraction.sides}


CANNAE_ROMAN = (
    "Total: 86,400\n15,000 light infantry\n55,000 infantry\n25,000 Romans\n30,000 allies\n"
    "6,400 cavalry\n2,400 Roman cavalry\n4,000 allied cavalry\n10,000 allied garrison"
)
CANNAE_CARTHAGE = (
    "Total: 50,000\n8,000 light infantry\n6,000 skirmishers\n2,000 Balearic slingers\n"
    "32,000 infantry\n21,000 Gauls\n3,000 Hispanics\n8,000 Libyans\n10,000 cavalry"
)
WATERLOO_FRANCE = (
    "72,000–73,000\n48,950 to 50,600 infantrymen\n14,390 to 15,765 cavalrymen\n"
    "8,050 gunners and engineers\n7,232 gunners\n• 246–252 guns"
)
WATERLOO_COALITION = (
    "Total : 118,000–120,000\n91,000 infantrymen\n21,500 cavalrymen\n7,500 gunners\n"
    "• 282–288 guns\nWellington's army: approx. 68,000 soldiers\n31,000\n17,000\n11,000\n"
    "6,000\n3,000\n• 156 guns Blücher's army: approx. 50,000 soldiers\n• 126 guns"
)


def test_contingents_are_components_not_totals() -> None:
    reports = _reports(
        _html(sides=("Roman Republic", "Carthage"), strengths=(CANNAE_ROMAN, CANNAE_CARTHAGE))
    )

    roman = reports["Roman Republic"]
    assert [r.reported_value for r in roman if r.branch == "total"] == [86_400.0]
    by_context = {r.extracted_context: r.branch for r in roman}
    assert by_context["25,000 Romans"] == "other"
    assert by_context["30,000 allies"] == "other"
    assert by_context["10,000 allied garrison"] == "other"
    assert by_context["55,000 infantry"] == "infantry"

    carthage = {r.extracted_context: r.branch for r in reports["Carthage"]}
    assert carthage["21,000 Gauls"] == "other"
    assert carthage["6,000 skirmishers"] == "infantry"
    assert carthage["2,000 Balearic slingers"] == "infantry"


def test_arm_words_with_a_men_suffix_name_the_arm() -> None:
    reports = _reports(
        _html(sides=("France", "United Kingdom"), strengths=(WATERLOO_FRANCE, WATERLOO_COALITION))
    )

    france = reports["France"]
    totals = [r for r in france if r.branch == "total"]
    assert sorted(r.reported_value for r in totals) == [72_000.0, 73_000.0]
    assert {r.extracted_context: r.branch for r in france}["7,232 gunners"] == "artillery"
    assert {r.extracted_context: r.branch for r in france}[
        "48,950 to 50,600 infantrymen"
    ] == "infantry"


def test_bare_figures_after_the_first_line_are_components() -> None:
    reports = _reports(
        _html(sides=("France", "United Kingdom"), strengths=(WATERLOO_FRANCE, WATERLOO_COALITION))
    )

    coalition = reports["United Kingdom"]
    totals = sorted(r.reported_value for r in coalition if r.branch == "total")
    assert totals == [118_000.0, 120_000.0]
    others = sorted(r.reported_value for r in coalition if r.branch == "other")
    assert others == [3_000.0, 6_000.0, 11_000.0, 17_000.0, 31_000.0, 50_000.0, 68_000.0]


def test_a_side_without_a_total_gets_one_derived_lower_bound_from_its_arms() -> None:
    reports = _reports(
        _html(
            sides=("Macedon", "Athens"),
            strengths=("30,000 infantry\n2,000 cavalry", "30,000 men"),
        )
    )

    macedon = reports["Macedon"]
    derived = [r for r in macedon if r.branch == "total"]
    assert len(derived) == 1
    assert derived[0].reported_value == 32_000.0
    assert derived[0].is_lower_bound
    assert derived[0].extracted_context.startswith("derived")
    assert derived[0].page_or_section == "infobox:derived_total"


def test_guns_are_not_summed_into_a_derived_total() -> None:
    reports = _reports(
        _html(
            sides=("French Republic", "Austria"),
            strengths=("15,500 infantry\n2,000 cavalry\n30 guns", "10,000 men"),
        )
    )

    (derived,) = [r for r in reports["French Republic"] if r.branch == "total"]
    assert derived.reported_value == 17_500.0


def test_bare_headings_block_a_derived_total() -> None:
    reports = _reports(
        _html(
            sides=("Russia", "France"),
            strengths=("16–17 October:\n257,000\n18–19 October:\n359,000\n150,000", "100,000"),
        )
    )

    assert not any(r.page_or_section == "infobox:derived_total" for r in reports["Russia"])


def test_a_lone_nationality_figure_is_the_sides_total() -> None:
    reports = _reports(_html(sides=("France", "Spain"), strengths=("50,000 French", "20,000 men")))

    assert [r.branch for r in reports["France"]] == ["total"]


def test_date_headings_in_a_strength_field_are_not_forces() -> None:
    assert parse_quantities("16–17 October:") == []
    assert parse_quantities("7 October:") == []


def test_legion_counts_with_an_adjective_are_not_headcounts() -> None:
    values = [q.value for q in parse_quantities("4 Roman legions, approximately 40,000")]
    assert values == [40_000.0]


def test_tidy_spacing_removes_html_joining_artifacts() -> None:
    assert tidy_spacing("Octavian 's forces") == "Octavian's forces"
    assert tidy_spacing("Volcae , a tribe of Gauls") == "Volcae, a tribe of Gauls"
    assert tidy_spacing("Hezbollah ( Islamic Jihad Organisation , after 1982)") == (
        "Hezbollah (Islamic Jihad Organisation, after 1982)"
    )
    assert tidy_spacing("Total : 5") == "Total: 5"


def test_rendered_side_labels_have_no_stray_spaces() -> None:
    html = _html(
        sides=("<a>Octavian</a> <a>'s</a> forces", "<a>Antony</a> 's forces"),
        strengths=("1,000", "2,000"),
    )
    extraction = parse_infobox_html(html, source_ref="t", source_title="t")
    assert extraction is not None
    assert [s.label for s in extraction.sides] == ["Octavian's forces", "Antony's forces"]


def test_the_infobox_column_decides_which_side_holds_a_commander() -> None:
    """Austerlitz lists Russia and Austria in one column; commanders follow it."""
    html = _html(
        sides=("French Empire\nKingdom of Italy", "Russian Empire\nAustrian Empire"),
        strengths=("65,000–75,000", "73,000–89,000"),
        commanders=("Napoleon I", "Alexander I\nMikhail Kutuzov\nFrancis II"),
    )
    extraction = parse_infobox_html(html, source_ref="t", source_title="t")
    assert extraction is not None

    allies = extraction.sides[1]
    assert allies.label == "Russian Empire"
    assert allies.aliases == ["Austrian Empire"]
    assert [c.name for c in allies.commanders] == ["Alexander I", "Mikhail Kutuzov", "Francis II"]


@pytest.mark.parametrize(
    ("text", "victor", "level"),
    [
        ("Carthaginian victory (see § Aftermath )", "Carthaginian", "victory"),
        (
            "Victory for Octavian, renewed alliance between Octavian and Antony",
            "Octavian",
            "victory",
        ),
        ("Decisive Prussian victory", "Prussian", "decisive_victory"),
        ("Roman pyrrhic victory", "Roman", "pyrrhic_victory"),
        ("Strategic Soviet victory", "Soviet", "victory"),
        ("Decisive victory for Octavian", "Octavian", "decisive_victory"),
        ("Decisive Macedonian victory", "Macedonian", "decisive_victory"),
        ("See § Aftermath", None, None),
    ],
)
def test_result_parsing_strips_cross_references(
    text: str, victor: str | None, level: str | None
) -> None:
    assert _parse_result(text) == (victor, level)


def _sides_outcomes(
    victor: str, labels: list[str], commanders: dict[str, list[str]] | None = None
) -> dict[str, str | None]:
    sides = []
    for label in labels:
        side = SideExtraction(label=label, polity=label)
        for name in (commanders or {}).get(label, []):
            side.commanders.append(
                CommanderMention(side_label=label, name=name, provenance=_INFOBOX_PROV)
            )
        sides.append(side)
    source = SourceExtraction(
        provenance=_INFOBOX_PROV,
        facts=BattleFacts(victor=victor, outcome_level="victory"),
        sides=sides,
    )
    return {s.label: s.outcome for s in merge_battle("x", [source]).sides}


@pytest.mark.parametrize(
    ("victor", "labels", "winner"),
    [
        ("Carthaginian", ["Carthage", "Roman Republic"], "Carthage"),
        ("Macedonian", ["Macedon", "Athens"], "Macedon"),
        ("Macedonian", ["Macedonia", "Achaemenid Empire"], "Macedonia"),
        ("French", ["France", "Austria"], "France"),
        ("British", ["United Kingdom", "Germany"], "United Kingdom"),
        ("Union", ["United States", "Confederate States"], "United States"),
        ("Confederate", ["United States", "CSA (Confederacy)"], "CSA (Confederacy)"),
        ("Persian", ["Athens", "Achaemenid Empire"], "Achaemenid Empire"),
        ("Roman", ["Rome", "Carthage"], "Rome"),
        ("Athenian", ["Athens", "Achaemenid Empire"], "Athens"),
        ("Pompeian", ["Pompeians", "Caesarians"], "Pompeians"),
    ],
)
def test_a_demonym_victor_matches_its_polity(victor: str, labels: list[str], winner: str) -> None:
    outcomes = _sides_outcomes(victor, labels)

    assert outcomes[winner] == "victory"
    assert [o for label, o in outcomes.items() if label != winner] == ["defeat"]


def test_austrian_does_not_match_australia() -> None:
    outcomes = _sides_outcomes("Austrian", ["Australia", "Germany"])

    assert all(o is None for o in outcomes.values())


def test_a_commander_name_victor_matches_the_side_that_commander_served() -> None:
    outcomes = _sides_outcomes(
        "Caesarian",
        ["Populares", "Optimates"],
        {"Populares": ["Julius Caesar"], "Optimates": ["Pompey"]},
    )

    assert outcomes == {"Populares": "victory", "Optimates": "defeat"}


def test_a_compound_victor_whose_parts_name_different_sides_stays_blank() -> None:
    outcomes = _sides_outcomes("Russo-Prussian", ["Prussia", "Russia"])

    assert all(o is None for o in outcomes.values())


def test_a_coalition_victor_stays_blank() -> None:
    outcomes = _sides_outcomes("Coalition", ["France", "United Kingdom"])

    assert all(o is None for o in outcomes.values())
