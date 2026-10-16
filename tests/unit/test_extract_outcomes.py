"""Unit tests for battle outcome assignment (fix R1).

The victor strings and side labels are the ones the real corpus produced, so a
regression here is a regression on a named battle.
"""

from __future__ import annotations

import pytest

from pipeline.extractors import (
    BattleFacts,
    Provenance,
    SideExtraction,
    SourceExtraction,
    merge_battle,
)


def _source(source_type: str, facts: BattleFacts, sides: list[SideExtraction]) -> SourceExtraction:
    return SourceExtraction(
        provenance=Provenance(
            source_type=source_type, extraction_method="test", source_ref=source_type
        ),
        facts=facts,
        sides=sides,
    )


def _outcomes(
    victor: str, sides: list[SideExtraction], level: str = "victory"
) -> dict[str, str | None]:
    source = _source("wikipedia_infobox", BattleFacts(victor=victor, outcome_level=level), sides)
    return {s.label: s.outcome for s in merge_battle("x", [source]).sides}


def _side(label: str, belligerents: int = 1) -> SideExtraction:
    return SideExtraction(label=label, polity=label, belligerents=belligerents)


# ─── Victor and level come from the source that parsed them ──────────────────


@pytest.mark.parametrize(
    ("infobox_victor", "dbpedia_victor", "labels", "winner"),
    [
        ("Union", "Unionvictory", ["United States", "Confederate States"], "United States"),
        ("Macedonian", "Macedonianvictory", ["Macedonia", "Achaemenid Empire"], "Macedonia"),
        ("Athenian", "Greek victory", ["Athens", "Achaemenid Empire"], "Athens"),
    ],
)
def test_the_infobox_victor_outranks_dbpedias_raw_literal(
    infobox_victor: str, dbpedia_victor: str, labels: list[str], winner: str
) -> None:
    merged = merge_battle(
        "x",
        [
            _source(
                "wikipedia_infobox",
                BattleFacts(victor=infobox_victor, outcome_level="victory"),
                [_side(label) for label in labels],
            ),
            _source("dbpedia", BattleFacts(victor=dbpedia_victor), []),
        ],
    )

    assert merged.facts.victor == infobox_victor
    assert {s.label: s.outcome for s in merged.sides}[winner] == "victory"


def test_a_glued_dbpedia_victor_is_usable_when_nothing_better_exists() -> None:
    merged = merge_battle(
        "x",
        [
            _source(
                "wikipedia_infobox",
                BattleFacts(outcome_level="victory"),
                [_side("United States"), _side("Confederate States")],
            ),
            _source("dbpedia", BattleFacts(victor="Unionvictory"), []),
        ],
    )

    assert merged.facts.victor == "Union"
    assert {s.label: s.outcome for s in merged.sides}["United States"] == "victory"


@pytest.mark.parametrize(
    "literal",
    ["Inconclusive", "(see the [[#Result", "*Persians gain control ofEuboea"],
)
def test_a_dbpedia_literal_that_names_no_victor_is_dropped(literal: str) -> None:
    merged = merge_battle("x", [_source("dbpedia", BattleFacts(victor=literal), [])])

    assert merged.facts.victor is None


def test_a_dbpedia_sentence_is_reduced_to_its_victor() -> None:
    merged = merge_battle(
        "x",
        [_source("dbpedia", BattleFacts(victor="Soviet victory and successful encirclement"), [])],
    )

    assert merged.facts.victor == "Soviet"


# ─── Coalition victors ───────────────────────────────────────────────────────


@pytest.mark.parametrize("victor", ["Coalition", "Seventh Coalition", "Allied"])
def test_a_coalition_victor_picks_the_one_multi_belligerent_side(victor: str) -> None:
    outcomes = _outcomes(
        victor,
        [
            _side("First French Empire"),
            _side("United Kingdom of Great Britain and Ireland", belligerents=4),
        ],
    )

    assert outcomes["United Kingdom of Great Britain and Ireland"] == "victory"
    assert outcomes["First French Empire"] == "defeat"


def test_a_coalition_victor_is_ambiguous_when_both_sides_are_coalitions() -> None:
    outcomes = _outcomes(
        "Coalition",
        [_side("Russian Empire", belligerents=3), _side("First French Empire", belligerents=5)],
    )

    assert all(o is None for o in outcomes.values())


def test_a_coalition_victor_is_ambiguous_when_no_side_is_a_coalition() -> None:
    outcomes = _outcomes("Coalition", [_side("France"), _side("United Kingdom")])

    assert all(o is None for o in outcomes.values())


# ─── Compound victors ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("victor", "labels", "winner"),
    [
        ("Prusso-Russian", ["Prussia", "Austria"], "Prussia"),
        ("Russo-Austrian", ["Prussia", "Russian Empire"], "Russian Empire"),
        ("Franco-Polish", ["France", "Spain"], "France"),
        ("Anglo-Ottoman", ["Ottoman Empire", "French Republic"], "Ottoman Empire"),
    ],
)
def test_a_compound_victor_picks_the_one_side_any_part_names(
    victor: str, labels: list[str], winner: str
) -> None:
    outcomes = _outcomes(victor, [_side(label) for label in labels])

    assert outcomes[winner] == "victory"
    assert [o for label, o in outcomes.items() if label != winner] == ["defeat"]


def test_a_compound_victor_naming_both_sides_stays_blank() -> None:
    outcomes = _outcomes("Franco-Spanish", [_side("France"), _side("Spanish Empire")])

    assert all(o is None for o in outcomes.values())


# ─── Demonym table and possessives ───────────────────────────────────────────


def test_a_possessive_victor_matches_the_side_named_for_the_person() -> None:
    outcomes = _outcomes("Octavian's", [_side("Octavian's forces"), _side("Antony's forces")])

    assert outcomes["Octavian's forces"] == "victory"


@pytest.mark.parametrize(
    ("victor", "labels", "winner"),
    [
        ("Korean", ["Toyotomi Japan", "Kingdom of Joseon"], "Kingdom of Joseon"),
        ("Axis", ["Kingdom of Italy", "United Kingdom"], "Kingdom of Italy"),
    ],
)
def test_corpus_demonyms_resolve(victor: str, labels: list[str], winner: str) -> None:
    outcomes = _outcomes(victor, [_side(label) for label in labels])

    assert outcomes[winner] == "victory"
