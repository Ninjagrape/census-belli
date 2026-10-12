"""Unit tests for the siege defender rule and the raw-article lookup."""

from __future__ import annotations

from pathlib import Path

from pipeline.classifiers.defender import SideLabel, besieged_place, infer_siege_defenders
from pipeline.classifiers.load import find_article_path
from pipeline.crawlers.wikipedia import article_filename


def test_besieged_place_strips_a_trailing_year() -> None:
    assert besieged_place("Siege of Toulon (1793)") == "Toulon"
    assert besieged_place("Siege of Rome (537–538)") == "Rome"
    assert besieged_place("Battle of Marathon") is None


def test_the_side_carrying_the_besieged_place_is_the_defender() -> None:
    # Real sides from the pilot database.
    sides = [
        SideLabel(297, "Supporters of Julius Caesar", "Supporters of Julius Caesar"),
        SideLabel(298, "City of Massilia", "City of Massilia"),
    ]

    assert infer_siege_defenders("Siege of Massilia", sides) == {297: False, 298: True}


def test_a_siege_whose_title_matches_neither_side_is_left_unset() -> None:
    # Acre lists the defenders first, El Arish the attackers: ordering and
    # outcome say nothing, and neither label carries the place name.
    acre = [SideLabel(289, "Ottoman Empire"), SideLabel(290, "French Republic")]
    toulon = [SideLabel(303, "France"), SideLabel(304, "French Royalists")]

    assert infer_siege_defenders("Siege of Acre (1799)", acre) == {}
    assert infer_siege_defenders("Siege of Toulon (1793)", toulon) == {}


def test_a_title_matching_both_sides_is_left_unset() -> None:
    sides = [SideLabel(1, "Rome loyalists"), SideLabel(2, "Rome rebels")]

    assert infer_siege_defenders("Siege of Rome", sides) == {}


def test_the_article_is_found_when_the_stored_url_differs_from_the_crawled_one(
    tmp_path: Path,
) -> None:
    # battles.wikipedia_url comes from Wikidata (http, decoded); crawl hashed
    # the https, percent-encoded URL it fetched. The digests differ.
    crawled = "https://en.wikipedia.org/wiki/Battle_of_Chaeronea_%28338_BC%29"
    stored = "http://en.wikipedia.org/wiki/Battle_of_Chaeronea_(338_BC)"
    directory = tmp_path / "battles_html"
    directory.mkdir()
    article = directory / article_filename(crawled)
    article.write_text("<html></html>", encoding="utf-8")

    assert article_filename(stored) != article.name
    assert find_article_path(tmp_path, stored) == article


def test_two_articles_with_one_title_are_not_guessed_between(tmp_path: Path) -> None:
    directory = tmp_path / "battles_html"
    directory.mkdir()
    (directory / "Battle_of_X-aaaaaaaaaa.html").write_text("a", encoding="utf-8")
    (directory / "Battle_of_X-bbbbbbbbbb.html").write_text("b", encoding="utf-8")

    assert find_article_path(tmp_path, "http://en.wikipedia.org/wiki/Battle_of_X") is None


def test_source_slug_is_preferred_over_the_url_lookup(tmp_path: Path) -> None:
    # The slug is crawl's own file stem, so it finds the article even when
    # the stored URL is missing or hashes to a different digest, and when a
    # title is shared by two files that the title fallback would refuse.
    directory = tmp_path / "battles_html"
    directory.mkdir()
    (directory / "Battle_of_X-aaaaaaaaaa.html").write_text("a", encoding="utf-8")
    wanted = directory / "Battle_of_X-bbbbbbbbbb.html"
    wanted.write_text("b", encoding="utf-8")
    url = "http://en.wikipedia.org/wiki/Battle_of_X"

    assert find_article_path(tmp_path, url, "Battle_of_X-bbbbbbbbbb") == wanted
    assert find_article_path(tmp_path, None, "Battle_of_X-bbbbbbbbbb") == wanted
    assert find_article_path(tmp_path, url, "Battle_of_X-missing") is None
