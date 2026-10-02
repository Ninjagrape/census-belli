"""
Unit tests for battle-type inference.

The one test that matters most here is Actium: handover.md §4.2 records that
the infobox parser used to hardcode ``battle_type="field"`` for every battle
using the generic "Infobox military conflict" template, which is nearly
every battle on Wikipedia including every naval one, and that the fix left
``battle_type`` unset for classify to infer instead. ``infer_battle_type``
is that inference, and the fixture that caught the original bug
(``tests/fixtures/extract/actium_naval.wikitext``) is reused here as the
regression pin: whatever evidence the real infobox parser extracts from it
must still say 'naval', not 'field' and not 'unknown'.
"""

from __future__ import annotations

from pathlib import Path

from pipeline.classifiers.battle_type import extract_categories, infer_battle_type
from pipeline.extractors import clean_article_text, parse_infobox

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "extract"


def _troop_branches_and_strings(wikitext: str, *, source_title: str) -> tuple[list[str], list[str]]:
    """Pull the two evidence lists infer_battle_type wants out of an infobox parse."""
    extraction = parse_infobox(wikitext, source_ref="test", source_title=source_title)
    assert extraction is not None
    branches = [r.branch for side in extraction.sides for r in side.troop_reports]
    strings = [
        r.extracted_context or str(r.reported_value)
        for side in extraction.sides
        for r in side.troop_reports
    ]
    return branches, strings


def test_actium_infers_naval_the_442_regression() -> None:
    wikitext = (FIXTURES / "actium_naval.wikitext").read_text(encoding="utf-8")
    branches, strings = _troop_branches_and_strings(wikitext, source_title="Battle of Actium")
    article_text = clean_article_text(wikitext)
    categories = extract_categories(wikitext)

    decision = infer_battle_type(1, "Battle of Actium", article_text, categories, branches, strings)

    assert decision.battle_type == "naval"
    assert decision.confidence > 0.0


def test_actium_naval_signal_survives_even_without_categories() -> None:
    # The fixture carries no [[Category:...]] lines at all -- the naval
    # signal has to come from the ships1/ships2 troop branch or the "400
    # ships" / "230 ships" vocabulary alone, exactly as it would for a real
    # article using the generic infobox template with no naval category.
    wikitext = (FIXTURES / "actium_naval.wikitext").read_text(encoding="utf-8")
    assert extract_categories(wikitext) == []


def test_siege_of_alesia_is_siege_offensive_and_fortified() -> None:
    wikitext = (FIXTURES / "alesia_siege.wikitext").read_text(encoding="utf-8")
    branches, strings = _troop_branches_and_strings(wikitext, source_title="Siege of Alesia")
    article_text = clean_article_text(wikitext)
    categories = extract_categories(wikitext)

    decision = infer_battle_type(2, "Siege of Alesia", article_text, categories, branches, strings)

    assert decision.battle_type == "siege_offensive"
    assert decision.fortified is True


def test_a_siege_category_alone_is_also_recognised() -> None:
    decision = infer_battle_type(
        3,
        "Battle of Some Fortress",
        "A long and detailed engagement.",
        ["Sieges of the Napoleonic Wars"],
        [],
        [],
    )

    assert decision.battle_type == "siege_offensive"
    assert decision.fortified is True


def test_no_article_read_yields_unknown_not_field() -> None:
    # This is the §4.2 lesson applied in the other direction: defaulting an
    # unread battle to 'field' would be the same class of silent mistake,
    # just relabelled.
    decision = infer_battle_type(4, "Battle of Nowhere", None, [], [], [])

    assert decision.battle_type == "unknown"
    assert decision.confidence == 0.0


def test_land_battle_with_article_and_no_signal_is_field() -> None:
    article_text = (
        "The armies met on open ground outside the town. After several "
        "hours of fighting the attacking force withdrew."
    )
    decision = infer_battle_type(
        5,
        "Battle of Open Ground",
        article_text,
        [],
        ["infantry", "cavalry"],
        ["12,000 infantry", "3,000 cavalry"],
    )

    assert decision.battle_type == "field"


def test_ship_vocabulary_in_strength_text_alone_infers_naval() -> None:
    decision = infer_battle_type(
        6,
        "Battle of the Strait",
        None,
        [],
        [],
        ["a fleet of 40 galleys", "12 frigates"],
    )

    assert decision.battle_type == "naval"


def test_naval_category_is_the_highest_confidence_naval_signal() -> None:
    decision = infer_battle_type(
        7,
        "Battle of the Bay",
        None,
        ["Naval battles of the Pacific War"],
        [],
        [],
    )

    assert decision.battle_type == "naval"
    assert decision.confidence >= 0.9


def test_aerial_category_infers_aerial() -> None:
    decision = infer_battle_type(
        8,
        "Battle of the Sky",
        None,
        ["Air battles of World War II"],
        [],
        [],
    )

    assert decision.battle_type == "aerial"


def test_amphibious_needs_naval_and_land_and_an_explicit_mention() -> None:
    decision = infer_battle_type(
        9,
        "Battle of the Beachhead",
        "The amphibious landing put infantry ashore under naval gunfire support.",
        [],
        ["naval", "infantry"],
        [],
    )

    assert decision.battle_type == "amphibious"
    assert decision.fortified is None


def test_naval_and_land_without_the_word_amphibious_stays_naval() -> None:
    # A judgement call, documented in battle_type.py: two branches of
    # evidence alone also describes an ordinary combined campaign, so this
    # function only calls it 'amphibious' when the text actually says so.
    decision = infer_battle_type(
        10,
        "Battle of Two Fronts",
        "Ships bombarded the coast while infantry advanced inland.",
        [],
        ["naval", "infantry"],
        [],
    )

    assert decision.battle_type == "naval"


def test_extract_categories_reads_names_without_the_category_prefix() -> None:
    wikitext = (
        "Some article text.\n\n"
        "[[Category:Naval battles of the Roman civil wars]]\n"
        "[[Category:31 BC]]\n"
        "[[Category:Battles of the Final War of the Roman Republic|Actium]]\n"
    )

    categories = extract_categories(wikitext)

    assert categories == [
        "Naval battles of the Roman civil wars",
        "31 BC",
        "Battles of the Final War of the Roman Republic",
    ]
