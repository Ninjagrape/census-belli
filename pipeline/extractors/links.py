"""
The article titles an infobox commander field wikilinks.

:func:`pipeline.extractors.textnorm.strip_markup` reduces ``[[George Henry
Thomas|G. H. Thomas]]`` to the display text, and the display text is all the
resolve stage used to see. The link target is much better evidence of who is
meant, so it is read here from the raw field, separately. Nothing in this
module changes the plain text any other code reads.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Final

import mwparserfromhell
import structlog

from pipeline.extractors.textnorm import strip_markup

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
    title = target.split("#", 1)[0].replace("_", " ")
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


def titles_in_item(links: Sequence[CommanderLink], item: str) -> tuple[str, ...]:
    """Pick the links that belong to one commander item.

    Args:
        links: The field's links from :func:`commander_wikilinks`.
        item: One cleaned item of the field (one person).

    Returns:
        Titles whose display text occurs in the item, in order of first
        occurrence in it, without repeats.
    """
    found: list[tuple[int, str]] = []
    for title, display in links:
        position = item.find(display)
        if position >= 0 and all(title != seen for _, seen in found):
            found.append((position, title))
    found.sort(key=lambda pair: pair[0])
    return tuple(title for _, title in found)
