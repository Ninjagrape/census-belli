"""Splitting one article into several battles with a ``battle_split`` override.

The fixture is the real wikitext of "Battle of Jena-Auerstedt": two battles
fought on 14 October 1806 under one infobox whose commander, strength and
casualty fields carry headed or labelled sub-lists.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from pipeline.extractors import BattleExtraction, SourceExtraction
from pipeline.extractors.infobox import parse_article_infobox_parts
from pipeline.extractors.override_apply import apply_extract_overrides, split_parts_by_slug
from pipeline.extractors.split import SplitPart
from pipeline.overrides import OverrideReport, load_overrides
from pipeline.stages import extract

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "extract"
PARENT = "Battle_of_Jena_Auerstedt-7e1600f782"
JENA = SplitPart(key="jena", name="Battle of Jena", heading="Jena")
AUERSTEDT = SplitPart(key="auerstedt", name="Battle of Auerstedt", heading="Auerstedt")
FRENCH = "French Empire"
PRUSSIA = "Prussia"


def _wikitext() -> str:
    return (FIXTURES / "jena_auerstedt.wikitext").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def parts() -> dict[str, SourceExtraction]:
    found = parse_article_infobox_parts(
        _wikitext(),
        is_html=False,
        source_ref="https://en.wikipedia.org/wiki/Battle_of_Jena%E2%80%93Auerstedt",
        source_title="Battle of Jena-Auerstedt",
        parts=[JENA, AUERSTEDT],
    )
    assert found is not None
    return found


def _side(extraction: SourceExtraction, label: str) -> Any:
    (side,) = [s for s in extraction.sides if s.label == label]
    return side


def _names(extraction: SourceExtraction, label: str) -> list[str]:
    return [c.name for c in _side(extraction, label).commanders]


def _troops(extraction: SourceExtraction, label: str) -> list[tuple[float, str, str]]:
    return [(t.reported_value, t.branch, t.scope) for t in _side(extraction, label).troop_reports]


def _casualties(extraction: SourceExtraction, label: str) -> list[float]:
    return [c.reported_value for c in _side(extraction, label).casualty_reports]


def test_each_part_gets_both_sides_with_the_parents_labels(
    parts: dict[str, SourceExtraction],
) -> None:
    assert set(parts) == {"jena", "auerstedt"}
    for extraction in parts.values():
        assert [s.label for s in extraction.sides] == [FRENCH, PRUSSIA]
        assert _side(extraction, PRUSSIA).aliases == ["Saxony"]


def test_jena_commanders_follow_the_jena_heading(parts: dict[str, SourceExtraction]) -> None:
    jena = parts["jena"]

    assert _names(jena, FRENCH) == [
        "Napoleon I",
        "Jean-Baptiste Bessières",
        "François Joseph Lefebvre",
        "Jean-de-Dieu Soult",
        "Jean Lannes",
        "Michel Ney",
        "Charles-Pierre Augereau",
        "Joachim Murat",
    ]
    assert _names(jena, PRUSSIA) == [
        "Prince of Hohenlohe",
        "Julius von Grawert",
        "Bogislav Friedrich Emanuel von Tauentzien",
        "Ernst von Rüchel",
    ]


def test_auerstedt_commanders_follow_the_auerstedt_heading(
    parts: dict[str, SourceExtraction],
) -> None:
    auerstedt = parts["auerstedt"]

    assert _names(auerstedt, FRENCH) == ["Louis-Nicolas Davout", "Charles-Étienne Gudin"]
    assert _names(auerstedt, PRUSSIA) == [
        "Duke of Brunswick",
        "Frederick William III of Prussia",
        "Friedrich Wilhelm von Schmettau",
        "Gerhard von Scharnhorst",
        "Count of Kalckreuth",
        "Gebhard von Blücher",
    ]


def test_commander_order_restarts_in_each_part(parts: dict[str, SourceExtraction]) -> None:
    order = [c.infobox_order for c in _side(parts["auerstedt"], FRENCH).commanders]

    assert order == [0, 1]


def test_a_part_heading_is_not_a_commander_mention(parts: dict[str, SourceExtraction]) -> None:
    for extraction in parts.values():
        for side in extraction.sides:
            names = {c.name.lower().strip(": ") for c in side.commanders}
            assert not names & {"jena", "auerstedt"}


def test_jena_troops_are_the_jena_labelled_figures(parts: dict[str, SourceExtraction]) -> None:
    jena = parts["jena"]

    assert _troops(jena, FRENCH) == [
        (40000.0, "total", "engaged"),
        (96000.0, "total", "available"),
    ]
    prussia = _troops(jena, PRUSSIA)
    assert prussia[0] == (46000.0, "total", "engaged")
    # The two "under" sub-lines are components of the 46,000, not totals.
    assert {value for value, _, _ in prussia[1:]} == {40000.0, 33000.0, 15000.0, 13000.0}
    assert all(branch == "other" for _, branch, _ in prussia[1:])


def test_auerstedt_troops_are_the_auerstedt_labelled_figures(
    parts: dict[str, SourceExtraction],
) -> None:
    auerstedt = parts["auerstedt"]

    assert _troops(auerstedt, FRENCH) == [(26000.0, "total", "engaged")]
    assert _troops(auerstedt, PRUSSIA) == [(64000.0, "total", "engaged")]


def test_unlabelled_whole_battle_totals_go_to_no_part(
    parts: dict[str, SourceExtraction],
) -> None:
    for extraction in parts.values():
        written = {
            r.reported_value
            for side in extraction.sides
            for r in (*side.troop_reports, *side.casualty_reports)
        }
        assert not written & {66000.0, 110000.0, 12600.0, 41000.0, 1000.0}


def test_casualties_are_routed_by_their_label(parts: dict[str, SourceExtraction]) -> None:
    assert _casualties(parts["jena"], FRENCH) == [5000.0, 6000.0]
    assert _casualties(parts["auerstedt"], FRENCH) == [7052.0, 7100.0]
    assert _casualties(parts["jena"], PRUSSIA) == [26000.0, 27000.0, 7200.0]
    assert _casualties(parts["auerstedt"], PRUSSIA) == [13000.0, 15000.0]


def test_the_infobox_name_is_the_parts_name_and_the_article_level_facts_are_shared(
    parts: dict[str, SourceExtraction],
) -> None:
    for key, extraction in parts.items():
        assert extraction.facts.name == {"jena": JENA.name, "auerstedt": AUERSTEDT.name}[key]
        assert extraction.facts.date_start == "1806-10-14"
        assert extraction.facts.date_precision == "day"
        assert extraction.facts.victor == "French"


def test_an_article_without_an_infobox_yields_none() -> None:
    found = parse_article_infobox_parts(
        "no infobox here",
        is_html=False,
        source_ref="x",
        source_title="x",
        parts=[JENA, AUERSTEDT],
    )

    assert found is None


def test_a_field_with_no_part_labels_writes_nothing_to_any_part() -> None:
    text = (
        "{{Infobox military conflict\n| conflict = Battle of Two\n| date = 1 May 1800\n"
        "| combatant1 = A\n| combatant2 = B\n| strength1 = 10,000\n| strength2 = 20,000\n}}"
    )

    found = parse_article_infobox_parts(
        text, is_html=False, source_ref="x", source_title="x", parts=[JENA, AUERSTEDT]
    )

    assert found is not None
    assert all(not s.troop_reports for e in found.values() for s in e.sides)


def test_rendered_html_infoboxes_route_flat_lines() -> None:
    html = (
        "<html><body><table class='infobox'>"
        "<tr><th colspan='2'>Belligerents</th></tr>"
        "<tr><td>A</td><td>B</td></tr>"
        "<tr><th colspan='2'>Strength</th></tr>"
        "<tr><td>10,000 (Jena)<br/>20,000 (Auerstedt)<br/>30,000</td><td>5,000 (Jena)</td></tr>"
        "</table></body></html>"
    )

    found = parse_article_infobox_parts(
        html, is_html=True, source_ref="x", source_title="x", parts=[JENA, AUERSTEDT]
    )

    assert found is not None
    assert [r.reported_value for r in found["jena"].sides[0].troop_reports] == [10000.0]
    assert [r.reported_value for r in found["auerstedt"].sides[0].troop_reports] == [20000.0]


# ─── The stage: one battle per part ──────────────────────────────────────────


def _raw_battle(tmp_path: Path) -> extract.RawBattle:
    articles = tmp_path / "battles_html"
    articles.mkdir()
    path = articles / f"{PARENT}.wikitext"
    path.write_bytes((FIXTURES / "jena_auerstedt.wikitext").read_bytes())
    return extract.RawBattle(slug=PARENT, article_path=path, title="Battle of Jena-Auerstedt")


def _split(tmp_path: Path) -> list[BattleExtraction]:
    return extract.extract_split_battle(_raw_battle(tmp_path), [JENA, AUERSTEDT])


def test_the_stage_writes_one_record_per_part_keyed_by_a_stable_child_slug(
    tmp_path: Path,
) -> None:
    children = _split(tmp_path)

    assert [c.slug for c in children] == [f"{PARENT}#jena", f"{PARENT}#auerstedt"]
    assert [c.name for c in children] == ["Battle of Jena", "Battle of Auerstedt"]
    assert [c.facts.name for c in children] == ["Battle of Jena", "Battle of Auerstedt"]
    for child in children:
        assert child.facts.wikidata_id is None
        assert child.facts.date_start == "1806-10-14"
        assert any(PARENT in note and "battle_split" in note for note in child.notes)


def test_each_child_inherits_the_parents_result_for_both_sides(tmp_path: Path) -> None:
    for child in _split(tmp_path):
        outcomes = {s.label: s.outcome for s in child.sides}
        assert outcomes[FRENCH] in {"victory", "decisive_victory", "pyrrhic_victory"}
        assert outcomes[PRUSSIA] in {"defeat", "decisive_defeat"}


def test_no_unlabelled_total_reaches_either_child(tmp_path: Path) -> None:
    for child in _split(tmp_path):
        values = {
            r.reported_value
            for side in child.sides
            for r in (*side.troop_reports, *side.casualty_reports)
        }
        assert not values & {66000.0, 110000.0, 12600.0, 41000.0}


def _overrides(tmp_path: Path, *raw: dict[str, Any]) -> Any:
    path = tmp_path / "overrides.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "entries": list(raw)}), encoding="utf-8")
    return load_overrides(path)


SPLIT_ENTRY = {
    "id": "ov-0001",
    "kind": "battle_split",
    "target": {"battle": PARENT},
    "value": [
        {"key": "jena", "name": "Battle of Jena", "heading": "Jena"},
        {"key": "auerstedt", "name": "Battle of Auerstedt", "heading": "Auerstedt"},
    ],
    "source": "Wikipedia, infobox",
    "note": "",
    "reviewed": {"by": "lee", "on": "2026-10-22"},
}


def test_split_parts_are_looked_up_by_the_normalised_parent_slug(tmp_path: Path) -> None:
    overrides = _overrides(tmp_path, SPLIT_ENTRY)

    found = split_parts_by_slug(overrides.for_stage("extract"))

    assert found == {PARENT.casefold(): (JENA, AUERSTEDT)}


def test_a_split_entry_is_applied_when_every_child_exists_and_unmatched_otherwise(
    tmp_path: Path,
) -> None:
    overrides = _overrides(tmp_path, SPLIT_ENTRY)
    report = OverrideReport("extract", overrides, limited=False)

    apply_extract_overrides(_split(tmp_path), overrides.for_stage("extract"), report)
    other = OverrideReport("extract", overrides, limited=False)
    apply_extract_overrides([], overrides.for_stage("extract"), other)

    done = json.loads(report.write(tmp_path / "a").read_text(encoding="utf-8"))
    missed = json.loads(other.write(tmp_path / "b").read_text(encoding="utf-8"))
    assert done["applied"]["battle_split"] == 1
    assert missed["applied"]["battle_split"] == 0
    assert [u["reason"] for u in missed["unmatched"]][0].startswith("battle not found")


def test_an_outcome_override_can_target_a_child_slug(tmp_path: Path) -> None:
    outcome = {
        "id": "ov-0002",
        "kind": "outcome",
        "target": {"battle": f"{PARENT}#auerstedt"},
        "value": {FRENCH: "decisive_victory", PRUSSIA: "decisive_defeat"},
        "source": "Chandler",
        "note": "",
        "reviewed": {"by": "lee", "on": "2026-10-22"},
    }
    overrides = _overrides(tmp_path, SPLIT_ENTRY, outcome)
    report = OverrideReport("extract", overrides, limited=False)
    children = _split(tmp_path)

    apply_extract_overrides(children, overrides.for_stage("extract"), report)

    by_slug = {c.slug: c for c in children}
    sides = {s.label: s.outcome for s in by_slug[f"{PARENT}#auerstedt"].sides}
    assert sides[FRENCH] == "decisive_victory"


def test_a_child_slug_finds_its_parents_article(tmp_path: Path) -> None:
    from pipeline.classifiers.load import find_article_path

    articles = tmp_path / "battles_html"
    articles.mkdir()
    article = articles / f"{PARENT}.wikitext"
    article.write_text("x", encoding="utf-8")

    assert find_article_path(tmp_path, None, f"{PARENT}#jena") == article


@pytest.mark.parametrize(
    "raw",
    [
        "{{plainlist|* [[Napoleon]]\n* [[Ney]]}}",
        "{{plainlist|\n* [[Napoleon]]\n* [[Ney]]}}",
        "{{plainlist|\n* [[Napoleon]]\n* [[Ney]]\n}}",
        "{{plainlist|\n* [[Napoleon]]\n* [[Ney]]\n}}\n",
    ],
)
def test_split_raw_lines_keeps_content_on_a_template_opening_or_closing_line(raw: str) -> None:
    from pipeline.extractors.split import split_raw_lines

    texts = [text for _depth, text in split_raw_lines(raw) if text]

    assert texts == ["Napoleon", "Ney"]


def test_split_raw_lines_still_blanks_a_pure_template_boundary() -> None:
    from pipeline.extractors.split import split_raw_lines

    lines = split_raw_lines("* [[A]]\n{{tree list/end}}\n* [[B]]")

    assert [text for _d, text in lines] == ["A", "", "B"]
