"""
Extract side of the infobox-wikilink signal: ``CommanderMention.link_titles``.

The infobox's own wikilink is the strongest evidence of who a commander is
("G. H. Thomas" links George Henry Thomas), and extract used to throw it away
when it stripped markup. These tests pin what is kept, and that the plain-text
fields (and so the LLM passage text and every mention's name) are unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.extractors.infobox import parse_infobox
from pipeline.extractors.records import CommanderMention, Provenance, to_jsonable
from pipeline.resolvers.mentions import load_mentions


def _commanders(commander1: str) -> list[CommanderMention]:
    wikitext = (
        "{{Infobox military conflict\n"
        "| conflict = Battle of Testing\n"
        "| combatant1 = [[France]]\n"
        "| combatant2 = [[Austria]]\n"
        f"| commander1 = {commander1}\n"
        "| commander2 = [[Archduke Charles]]\n"
        "}}\nBody."
    )
    extraction = parse_infobox(wikitext, source_title="Battle of Testing")
    assert extraction is not None
    return extraction.sides[0].commanders


def _links(commander1: str) -> list[tuple[str, tuple[str, ...]]]:
    return [(m.name, m.link_titles) for m in _commanders(commander1)]


def test_a_piped_link_records_the_article_not_the_display_text() -> None:
    assert _links("[[George Henry Thomas|G. H. Thomas]]") == [
        ("G. H. Thomas", ("George Henry Thomas",))
    ]


def test_a_rank_link_and_a_person_link_are_both_kept_in_order() -> None:
    assert _links("[[Vice admiral]] [[Ernst Schirlitz]]") == [
        ("Vice admiral Ernst Schirlitz", ("Vice admiral", "Ernst Schirlitz"))
    ]


def test_a_link_after_a_flag_template_is_kept() -> None:
    assert _links("{{flagicon|France}} [[Napoleon]]") == [("Napoleon", ("Napoleon",))]


def test_fragment_underscores_and_case_are_normalised() -> None:
    assert _links("[[john_smith (general)#Early life|Gen. Smith]]") == [
        ("Gen. Smith", ("John smith (general)",))
    ]


def test_an_unlinked_name_has_no_link_titles() -> None:
    assert _links("Joachim Murat") == [("Joachim Murat", ())]


def test_file_and_interwiki_targets_are_skipped() -> None:
    links = _links("[[File:Flag.svg|20px]] [[fr:Napoleon Ier|Napoleon]] [[Michel Ney]]")
    assert [titles for _, titles in links] == [("Michel Ney",)]


def test_each_person_in_a_br_separated_field_gets_only_their_own_link() -> None:
    assert _links("[[Michel Ney]]<br />[[Louis-Nicolas Davout|Davout]]<br />Unlinked Man") == [
        ("Michel Ney", ("Michel Ney",)),
        ("Davout", ("Louis-Nicolas Davout",)),
        ("Unlinked Man", ()),
    ]


def test_names_and_order_are_exactly_what_they_were() -> None:
    mentions = _commanders("[[Michel Ney]] (KIA)<br />[[Louis-Nicolas Davout|Davout]]")
    assert [(m.name, m.note, m.infobox_order) for m in mentions] == [
        ("Michel Ney", "KIA", 0),
        ("Davout", "", 1),
    ]
    assert mentions[0].role_evidence.startswith("Michel Ney (KIA)")


def test_link_titles_survive_the_jsonl_round_trip_into_resolve(tmp_path: Path) -> None:
    provenance = Provenance(source_type="wikipedia_infobox", extraction_method="infobox_parser")
    linked = CommanderMention(
        side_label="France",
        name="G. H. Thomas",
        provenance=provenance,
        link_titles=("George Henry Thomas",),
    )
    plain = CommanderMention(side_label="France", name="Joachim Murat", provenance=provenance)
    path = tmp_path / "commanders_raw.jsonl"
    rows = [{"battle_slug": "b", "battle_name": "B", **to_jsonable(m)} for m in (linked, plain)]
    # A row written before the key existed.
    rows.append({"battle_slug": "b", "battle_name": "B", "name": "Old Row", "side_label": "X"})
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    loaded = load_mentions(path, tmp_path / "missing_battles.jsonl")

    assert [m.link_titles for m in loaded] == [("George Henry Thomas",), (), ()]
    first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert first["link_titles"] == ["George Henry Thomas"]
