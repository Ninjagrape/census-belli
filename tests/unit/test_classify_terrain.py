"""Unit tests for rule-based terrain tags."""

from __future__ import annotations

import pytest

from pipeline.classifiers.terrain import MAX_TERRAIN_TAGS, TERRAIN_VOCABULARY, infer_terrain


def test_vocabulary_is_the_documented_thirteen() -> None:
    assert set(TERRAIN_VOCABULARY) == {
        "mountainous",
        "river",
        "forested",
        "desert",
        "urban",
        "coastal",
        "plain",
        "marsh",
        "hills",
        "steppe",
        "snow",
        "lake",
        "island",
    }


def test_no_text_gives_no_tags() -> None:
    assert infer_terrain(None) == ([], "")
    assert infer_terrain("   ") == ([], "")


def test_a_mention_in_the_lead_is_enough() -> None:
    tags, evidence = infer_terrain("The armies met on a broad plain north of the town.")

    assert tags == ["plain"]
    assert "plain" in evidence


def test_a_single_late_mention_is_not_enough() -> None:
    text = "The army marched for days. " * 200 + "It crossed a river."

    assert infer_terrain(text) == ([], "")


def test_three_late_mentions_clear_the_bar() -> None:
    text = "The army marched for days. " * 200 + "A river. The river. Beyond the river."

    tags, _evidence = infer_terrain(text)

    assert tags == ["river"]


def test_bare_city_is_not_urban_but_street_fighting_is() -> None:
    assert infer_terrain("The siege of the city began in spring.")[0] == []
    assert infer_terrain("There was house-to-house fighting in the old quarter.")[0] == ["urban"]


@pytest.mark.parametrize(
    ("text", "tag"),
    [
        ("Fought in the mountains of the Alps.", "mountainous"),
        ("Fought in the dense forest.", "forested"),
        ("Fought in the Sahara desert.", "desert"),
        ("Troops stormed the beach on the coast.", "coastal"),
        ("The army was trapped in the marsh.", "marsh"),
    ],
)
def test_each_tag_is_reachable_from_plain_vocabulary(text: str, tag: str) -> None:
    assert tag in infer_terrain(text)[0]


def test_tags_are_capped_and_ordered_by_mention_count() -> None:
    text = (
        "A river, a river, a river, a forest, a forest, a plain, a plain, a plain, "
        "a plain, a marsh, a marsh."
    )

    tags, _evidence = infer_terrain(text)

    assert tags == ["plain", "river", "forested"][:MAX_TERRAIN_TAGS]


@pytest.mark.parametrize(
    ("text", "tag"),
    [
        ("The army seized the ridge above the village.", "hills"),
        ("They held the high ground north of the road.", "hills"),
        ("Fought on rolling hills near the town.", "hills"),
        ("The cavalry crossed the open steppe.", "steppe"),
        ("Fought across wide grassland.", "steppe"),
        ("Fighting in the bocage hedgerows of Normandy.", "forested"),
        ("The battle was fought in Delville Wood.", "forested"),
        ("Fought in deep snow during the winter.", "snow"),
        ("The army crossed the thick ice of the bay.", "snow"),
        ("The armies met on the shore of a lake.", "lake"),
        ("The garrison held the island against the landing.", "island"),
    ],
)
def test_the_new_tags_are_reachable(text: str, tag: str) -> None:
    assert tag in infer_terrain(text)[0]


@pytest.mark.parametrize(
    ("text", "tag"),
    [
        ("Many men began to desert the camp.", "desert"),
        ("The Highland Brigade advanced in line.", "mountainous"),
        ("A Highlander regiment held the left.", "mountainous"),
        ("The mountain artillery opened fire.", "mountainous"),
        ("Two mountain divisions were held back.", "mountainous"),
        ("The mountain troops marched at dawn.", "mountainous"),
        ("Alpine troops and the Alpini guarded the pass.", "mountainous"),
        ("General Ford ordered the advance.", "river"),
        ("Fought at the Plains of Abraham outside Quebec.", "plain"),
        ("The Plains Indians raided the column.", "plain"),
        ("The Coast Guard cutter fired.", "coastal"),
        ("Troops from the Gold Coast arrived.", "coastal"),
        ("The advance bogged down at noon.", "marsh"),
        ("The assault would bog down in the mud.", "marsh"),
    ],
)
def test_terrain_words_inside_other_phrases_do_not_tag(text: str, tag: str) -> None:
    assert tag not in infer_terrain(text)[0]


def test_a_lower_case_ford_and_a_real_desert_still_tag() -> None:
    assert "river" in infer_terrain("The army crossed at the ford.")[0]
    assert "desert" in infer_terrain("They marched across the desert.")[0]
    assert "mountainous" in infer_terrain("The highlands rose behind the camp.")[0]
