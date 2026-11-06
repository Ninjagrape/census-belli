"""
The article titles an infobox commander field wikilinks.

:func:`pipeline.extractors.textnorm.strip_markup` reduces ``[[George Henry
Thomas|G. H. Thomas]]`` to the display text, and the display text is all the
resolve stage used to see. The link target is much better evidence of who is
meant, so it is read here from the raw field, separately. Nothing in this
module changes the plain text any other code reads.
"""

from __future__ import annotations

import html
import re
from collections.abc import Sequence
from typing import Final
from urllib.parse import unquote

import mwparserfromhell
import structlog

from pipeline.extractors.textnorm import strip_markup
from pipeline.resolvers.names import LEADING_TITLES

__all__ = ["CommanderLink", "commander_wikilinks", "titles_in_item"]

logger = structlog.get_logger()

# (article title, display text as strip_markup renders it)
CommanderLink = tuple[str, str]

_NAMESPACES: Final[frozenset[str]] = frozenset(
    {
        "file",
        "image",
        "media",
        "category",
        "template",
        "wikipedia",
        "wp",
        "help",
        "portal",
        "draft",
        "user",
        "talk",
        "special",
        "wikt",
        "wiktionary",
        "commons",
        "wikidata",
        "d",
    }
)
# Interwiki and language prefixes are short lowercase codes ("fr:", "de:", "zh-min-nan:").
_LANGUAGE_PREFIX_RE: Final = re.compile(r"^[a-z]{2,3}(?:-[a-z]+)*$")
_WHITESPACE_RE: Final = re.compile(r"\s+")


def _article_title(target: str) -> str | None:
    """Normalise a link target to an English Wikipedia article title.

    Args:
        target: The text left of the pipe, e.g. ``"john_smith#Early life"``.

    Returns:
        The title with the fragment dropped, underscores as spaces, whitespace
        collapsed and the first letter upper-cased; None for a link that is
        not to an article (a file, category, interwiki link or bare fragment).
    """
    # MediaWiki reads entities and percent-escapes in a target before it looks
    # the page up, so "Jean%20Lannes" and "Jean&nbsp;Lannes" are "Jean Lannes".
    decoded = unquote(html.unescape(target))
    title = decoded.split("#", 1)[0].replace("_", " ")
    title = _WHITESPACE_RE.sub(" ", title).strip()
    if not title:
        return None
    if ":" in title:
        prefix = title.split(":", 1)[0].strip()
        if title.startswith(":") or prefix.casefold() in _NAMESPACES:
            return None
        if _LANGUAGE_PREFIX_RE.match(prefix):
            return None
    return title[0].upper() + title[1:]


def commander_wikilinks(raw: str) -> list[CommanderLink]:
    """List the article links in a raw commander field.

    Args:
        raw: The field's wikitext, before markup was stripped.

    Returns:
        ``(title, display)`` for every wikilink, links inside templates
        included, in document order. Display is what the link contributes to
        the cleaned text, so it can be found again in a cleaned item.
    """
    if "[[" not in raw:
        return []
    try:
        nodes = mwparserfromhell.parse(raw).filter_wikilinks(recursive=True)
    except Exception as exc:
        # One malformed field must not cost the article its commanders.
        logger.warning("infobox_wikilink_parse_failed", error=str(exc))
        return []

    links: list[CommanderLink] = []
    for node in nodes:
        title = _article_title(str(node.title))
        if title is None:
            continue
        shown = node.text if node.text is not None and str(node.text).strip() else node.title
        display = strip_markup(str(shown), infobox=True)
        if display:
            links.append((title, display))
    return links


_PARENTHETICAL_RE: Final = re.compile(r"\([^()]*\)|\[[^\[\]]*\]")
_WORD_RE: Final = re.compile(r"[^\W\d_][\w'’.-]*")
_RANK_WORDS: Final[frozenset[str]] = frozenset(LEADING_TITLES)


def _span_in_parentheses(item: str, start: int, end: int) -> bool:
    """Whether ``item[start:end]`` lies inside a parenthetical or bracket."""
    return any(m.start() <= start and end <= m.end() for m in _PARENTHETICAL_RE.finditer(item))


def _names_the_item(item: str, spans: Sequence[tuple[int, int]]) -> bool:
    """Whether the linked spans are all the name the item holds.

    What is left of the item once the linked spans, parentheticals and rank
    words are removed must hold no capitalised word. "Charles Smith (Duke)"
    keeps "Smith", so a link on "Charles" does not name that item; "Vice
    admiral Ernst Schirlitz" keeps nothing.
    """
    chars = list(item)
    for start, end in spans:
        chars[start:end] = " " * (end - start)
    leftover = _PARENTHETICAL_RE.sub(" ", "".join(chars))
    for word in _WORD_RE.findall(leftover):
        if word[0].isupper() and word.casefold().strip(".") not in _RANK_WORDS:
            return False
    return True


def titles_in_item(links: Sequence[CommanderLink], item: str) -> tuple[str, ...]:
    """Pick the links that belong to one commander item.

    A link belongs to the item when its display text stands in the item as a
    whole word, outside any parenthetical, and the linked text is all the name
    the item holds (see :func:`_names_the_item`). That last test stops
    ``[[Charles, Archduke of Austria|Charles]]`` attaching to an unlinked
    "Charles Smith (Duke)", and stops a link on a relative or superior ("X,
    aide to [[Y]]", "X (son of [[Y]])") naming X. Attribution is by
    word-boundary match plus name coverage, not by position in the raw
    wikitext. When in doubt no link is attached and the name matcher decides.

    Args:
        links: The field's links from :func:`commander_wikilinks`.
        item: One cleaned item of the field (one person).

    Returns:
        Titles whose display text occurs in the item, in order of first
        occurrence in it, without repeats; empty when the item holds a name
        the links do not cover.
    """
    found: list[tuple[int, str]] = []
    spans: list[tuple[int, int]] = []
    for title, display in links:
        pattern = rf"(?<!\w){re.escape(display)}(?!\w)"
        for match in re.finditer(pattern, item):
            if _span_in_parentheses(item, match.start(), match.end()):
                continue
            spans.append((match.start(), match.end()))
            if all(title != seen for _, seen in found):
                found.append((match.start(), title))
            break
    if not found or not _names_the_item(item, spans):
        return ()
    found.sort(key=lambda pair: pair[0])
    return tuple(title for _, title in found)
