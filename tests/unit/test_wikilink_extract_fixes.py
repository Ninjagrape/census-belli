"""
Review fixes for the extract side of the infobox-wikilink signal.

A link must be attributed to the commander item it stands in (not to any item
that happens to contain its display text), a link that is not the item's own
name must not identify it, and link targets are decoded the way MediaWiki
decodes them.
"""

from __future__ import annotations

import pytest

from pipeline.extractors.infobox import parse_infobox
from pipeline.extractors.links import _article_title
from pipeline.extractors.records import CommanderMention


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


def test_a_link_does_not_attach_to_an_unlinked_namesake_item() -> None:
    links = _links("[[Charles, Archduke of Austria|Charles]]<br />Charles Smith (Duke)")
    assert links == [("Charles", ("Charles, Archduke of Austria",)), ("Charles Smith (Duke)", ())]


def test_a_rank_link_does_not_attach_to_an_item_that_merely_contains_the_word() -> None:
    links = dict(_links("[[Duke]]<br />Duke of Somewhere<br />[[Michel Ney]]"))
    assert links["Duke of Somewhere"] == ()
    assert links["Michel Ney"] == ("Michel Ney",)


def test_a_one_letter_display_attaches_nowhere() -> None:
    links = _links("[[Pope Leo I|I]] Smith<br />John I of Elsewhere")
    assert all(titles == () for _, titles in links)


def test_a_short_display_that_is_the_whole_name_is_kept() -> None:
    assert _links("[[Michel Ney|Ney]]") == [("Ney", ("Michel Ney",))]


def test_a_link_inside_a_parenthetical_is_not_the_mentions_identity() -> None:
    links = _links("Jean Lannes (son of [[Augustus II the Strong]])")
    assert [titles for _, titles in links] == [()]


def test_a_link_beside_an_unlinked_name_is_not_attached() -> None:
    links = _links("Jean Lannes, aide to [[Augustus II the Strong]]")
    assert [titles for _, titles in links] == [()]


def test_rank_words_and_notes_around_a_linked_name_are_fine() -> None:
    assert _links("General [[Michel Ney]] (KIA)") == [("General Michel Ney", ("Michel Ney",))]


@pytest.mark.parametrize(
    ("target", "title"),
    [
        ("Jean%20Lannes", "Jean Lannes"),
        ("jean_lannes", "Jean lannes"),
        ("Pierre &amp; Marie", "Pierre & Marie"),
        ("O&#39;Neill", "O'Neill"),
        ("Jean&nbsp;Lannes", "Jean Lannes"),
        ("Jos%C3%A9_de_San_Mart%C3%ADn", "José de San Martín"),
        ("Jean Lannes#Early%20life", "Jean Lannes"),
    ],
)
def test_article_titles_are_decoded(target: str, title: str) -> None:
    assert _article_title(target) == title


def test_an_encoded_link_reaches_the_mention_as_a_decoded_title() -> None:
    assert _links("[[Jean%20Lannes|Lannes]]") == [("Lannes", ("Jean Lannes",))]
