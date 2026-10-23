"""
Route the lines of one infobox field to the parts of a split article.

A ``battle_split`` override says one article describes several battles. The
infobox of such an article lists them in the same fields: a commander list
carries a heading line per part (``* '''Jena''':``), and a strength or
casualty line carries a label (``<small>(Jena)</small>``) with indented
sub-lines hanging under it. This module decides, line by line, which part
each line belongs to. It does not read figures; the ordinary infobox parser
runs afterwards on each part's own lines.

Rules:

* Commander field: a line that is only a part heading switches routing to that
  part for the lines after it and is dropped, not a mention. A commander line
  before any heading belongs to no part.
* Strength and casualty fields: a line whose text carries a part's label in
  brackets goes to that part. A deeper-indented line (``**`` under ``*``)
  follows the nearest shallower labelled line. Every other line describes the
  whole and goes to no part.

Lines that go to no part are returned so the caller can log why they were not
written to any child.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import structlog

from pipeline.extractors.textnorm import strip_markup

logger = structlog.get_logger(__name__)

__all__ = [
    "PART_SLUG_SEPARATOR",
    "RoutedField",
    "SplitPart",
    "child_slug",
    "route_commander_lines",
    "route_figure_lines",
    "split_raw_lines",
]

# Joins the parent's slug and a part key. '#' cannot occur in a crawl filename
# stem, so a child slug never collides with an article.
PART_SLUG_SEPARATOR: Final[str] = "#"

_LINE_BREAK_RE: Final = re.compile(r"\r?\n|<br\s*/?>", re.IGNORECASE)
_BULLET_RE: Final = re.compile(r"^\s*([*:#]+)")
_TEMPLATE_OPENER_RE: Final = re.compile(r"\{\{[^{}|\[\]]*\|?")
_BALANCED_TEMPLATE_RE: Final = re.compile(r"\{\{[^{}]*\}\}")
_CLOSER: Final = "}}"
_HEADING_TRIM: Final = " \t:*'–—-"


@dataclass(frozen=True)
class SplitPart:
    """One battle an article is split into.

    Attributes:
        key: Lowercase slug-safe id; the child's slug is ``<parent>#<key>``.
        name: The child battle's name.
        heading: The label the infobox uses for this part.
    """

    key: str
    name: str
    heading: str


@dataclass(frozen=True)
class RoutedField:
    """The lines of one field after routing.

    Attributes:
        by_part: Part key to the cleaned lines routed to it, in order.
        unrouted: Cleaned lines that belong to no part.
    """

    by_part: dict[str, list[str]]
    unrouted: list[str]


def child_slug(parent_slug: str, part: SplitPart) -> str:
    """Name the battle a part becomes.

    Args:
        parent_slug: The article's ``source_slug``.
        part: The part.

    Returns:
        ``<parent_slug>#<part.key>``, stable across runs.
    """
    return f"{parent_slug}{PART_SLUG_SEPARATOR}{part.key}"


def split_raw_lines(raw: str) -> list[tuple[int, str]]:
    """Break a field's wikitext into lines with their list depth.

    Args:
        raw: A raw infobox parameter value (or already cleaned text).

    Returns:
        ``(depth, cleaned text)`` per line; depth is the number of leading
        list markers. A line that cleans to nothing is kept as empty text,
        because a template boundary such as ``{{tree list/end}}`` ends the
        nesting it closes.
    """
    lines: list[tuple[int, str]] = []
    for raw_line in _LINE_BREAK_RE.split(raw):
        line = _drop_unmatched_braces(raw_line)
        marker = _BULLET_RE.match(line)
        depth = len(marker.group(1)) if marker else 0
        cleaned = strip_markup(line).strip()
        if not cleaned and _has_content(line):
            logger.warning("split_line_discarded", line=raw_line[:200])
        lines.append((depth, cleaned))
    return lines


def _drop_unmatched_braces(line: str) -> str:
    """Remove the template opener or closer a multi-line template leaves on a line.

    ``{{plainlist|* [[A]]`` keeps ``* [[A]]``; ``* [[B]]}}`` keeps ``* [[B]]``.
    The opening or closing line of a multi-line template is layout, but the
    content sharing that line must survive.
    """
    opened = line.count("{{")
    closed = line.count(_CLOSER)
    if opened > closed:
        return _TEMPLATE_OPENER_RE.sub("", line, count=1)
    while closed > opened:
        cut = line.rfind(_CLOSER)
        line = line[:cut] + line[cut + len(_CLOSER) :]
        closed -= 1
    return line


def _has_content(line: str) -> bool:
    """Whether a line holds words outside balanced templates."""
    return bool(re.search(r"\w", _BALANCED_TEMPLATE_RE.sub("", line)))


def _normal(text: str) -> str:
    return " ".join(text.casefold().split())


def _label_pattern(part: SplitPart) -> re.Pattern[str]:
    return re.compile(rf"\(\s*{re.escape(part.heading.strip())}\s*\)", re.IGNORECASE)


def route_commander_lines(
    lines: Sequence[tuple[int, str]], parts: Sequence[SplitPart]
) -> RoutedField:
    """Route a commander field by its heading lines.

    Args:
        lines: The field's lines from :func:`split_raw_lines`.
        parts: The article's parts.

    Returns:
        The commander lines per part, headings removed; lines before the first
        heading are unrouted.
    """
    headings = {_normal(p.heading): p.key for p in parts}
    by_part: dict[str, list[str]] = {p.key: [] for p in parts}
    unrouted: list[str] = []
    current: str | None = None
    for _, text in lines:
        if not text:
            continue
        heading_key = headings.get(_normal(text.strip(_HEADING_TRIM)))
        if heading_key is not None:
            current = heading_key
        elif current is None:
            unrouted.append(text)
        else:
            by_part[current].append(text)
    return RoutedField(by_part=by_part, unrouted=unrouted)


def route_figure_lines(lines: Sequence[tuple[int, str]], parts: Sequence[SplitPart]) -> RoutedField:
    """Route a strength or casualty field by the part label on each line.

    Args:
        lines: The field's lines from :func:`split_raw_lines`.
        parts: The article's parts.

    Returns:
        The lines per part; a line labelled with more than one part, or with
        none and not nested under a labelled line, is unrouted.
    """
    patterns = [(p.key, _label_pattern(p)) for p in parts]
    by_part: dict[str, list[str]] = {p.key: [] for p in parts}
    unrouted: list[str] = []
    anchor: tuple[str, int] | None = None
    for depth, text in lines:
        if not text:
            anchor = None
            continue
        named = [key for key, pattern in patterns if pattern.search(text)]
        if len(named) == 1:
            anchor = (named[0], depth)
            by_part[named[0]].append(text)
        elif not named and anchor is not None and depth > anchor[1]:
            by_part[anchor[0]].append(text)
        else:
            anchor = None
            unrouted.append(text)
    return RoutedField(by_part=by_part, unrouted=unrouted)
