"""
Read a rendered infobox back into the fields its template declared.

The crawl stage stores what Wikipedia served, which is HTML. Its infobox
carries the same quantities the ``Infobox military conflict`` template does,
but as a table of human-readable row labels rather than named parameters,
and a side-indexed quantity is spread across a header row plus one cell per
belligerent:

.. code-block:: html

    <tr><th colspan="2">Strength</th></tr>
    <tr><td>86,000 engaged</td><td>about 50,000</td></tr>

This module turns that back into ``strength1`` and ``strength2``, so
:mod:`pipeline.extractors.infobox` has one field mapping to maintain rather
than two.

Line structure inside a cell is load-bearing. Two commanders separated by a
``<br>`` are two people, and ``get_text`` without care welds them into one
name, so breaks and list items are marked before the text is flattened.
"""

from __future__ import annotations

import re
from typing import Final

import structlog
from bs4 import Tag
from bs4.element import NavigableString

__all__ = ["HTML_ROW_LABELS", "find_infobox_table", "html_infobox_fields"]

logger = structlog.get_logger()

# How a rendered infobox's row labels map onto the template field stems.
HTML_ROW_LABELS: Final[dict[str, str]] = {
    "date": "date",
    "location": "place",
    "place": "place",
    "result": "result",
    "territorial changes": "territory",
    "part of": "partof",
    "belligerents": "combatant",
    "combatants": "combatant",
    "commanders and leaders": "commander",
    "commanders": "commander",
    "leaders": "commander",
    "strength": "strength",
    "units involved": "units",
    "casualties and losses": "casualties",
    "casualties": "casualties",
    "losses": "casualties",
    "ships": "ships",
    "aircraft": "aircraft",
    "garrison": "garrison",
}

_PART_OF_RE: Final = re.compile(r"^part of\s+(?:the\s+)?(?P<name>.{3,120})$", re.IGNORECASE)

# Sentinel marking a line break inside a cell, so that two commanders
# separated by <br> do not become one name.
_CELL_BREAK: Final = "␞"


def find_infobox_table(soup: Tag) -> Tag | None:
    """Locate the infobox table on a rendered article.

    Args:
        soup: The parsed document, or any element to search within.

    Returns:
        The first ``<table>`` whose class list mentions ``infobox``, or None.
    """
    table = soup.find(
        "table",
        class_=lambda value: bool(value) and "infobox" in " ".join(str(value).split()).lower(),
    )
    return table if isinstance(table, Tag) else None


def _cell_text(cell: Tag) -> str:
    """Read a rendered table cell, preserving its line structure.

    Args:
        cell: A ``<td>`` or ``<th>`` element. Modified in place; reference
            superscripts and scripts are removed outright.

    Returns:
        Cleaned text with ``<br>`` and list items rendered as newlines.
    """
    for tag in cell.find_all(["sup", "style", "script"]):
        tag.decompose()
    for br in cell.find_all("br"):
        br.replace_with(NavigableString(_CELL_BREAK))
    for item in cell.find_all("li"):
        item.insert_before(NavigableString(_CELL_BREAK))

    text = cell.get_text(" ")
    text = re.sub(rf"\s*{_CELL_BREAK}\s*", "\n", text)
    text = re.sub(r"[ \t ]+", " ", text)
    return "\n".join(line.strip() for line in text.splitlines() if line.strip()).strip()


def _label_stem(label: str) -> str | None:
    """Map a rendered row label onto a template field stem.

    Args:
        label: The row's header text, e.g. ``"Commanders and leaders"``.

    Returns:
        The field stem, or None when the label is not one this parser reads.
    """
    key = re.sub(r"[^a-z ]+", " ", label.lower())
    key = re.sub(r"\s+", " ", key).strip()
    return HTML_ROW_LABELS.get(key)


def html_infobox_fields(table: Tag) -> dict[str, str]:
    """Read a rendered infobox table into template-shaped fields.

    Args:
        table: The ``<table class="infobox">`` element.

    Returns:
        Normalised field name to cleaned value, e.g. ``{"strength1": "86,000
        engaged"}``. Rows whose label this parser does not recognise are
        skipped rather than guessed at.
    """
    fields: dict[str, str] = {}
    pending: str | None = None

    for row in table.find_all("tr"):
        cells = [c for c in row.find_all(["th", "td"], recursive=False) if isinstance(c, Tag)]
        if not cells:
            continue

        headers = [c for c in cells if c.name == "th"]
        values = [c for c in cells if c.name == "td"]

        if headers and values:
            stem = _label_stem(_cell_text(headers[0]))
            pending = None
            if stem is None:
                continue
            if len(values) == 1:
                fields.setdefault(stem, _cell_text(values[0]))
            else:
                _store_indexed(fields, stem, values)
            continue

        if headers and not values:
            # A header row on its own announces a side-indexed quantity whose
            # values are in the row beneath it.
            pending = _label_stem(_cell_text(headers[0]))
            continue

        if values and pending is not None:
            _store_indexed(fields, pending, values)
            pending = None
            continue

        # "Part of the Second Punic War" is rendered as an unlabelled
        # subheader rather than a field, so it is recognised by shape.
        if len(values) == 1 and "partof" not in fields:
            match = _PART_OF_RE.match(_cell_text(values[0]))
            if match:
                fields["partof"] = re.sub(r"\s+", " ", match.group("name")).strip()

    logger.debug("rendered_infobox_fields_read", fields=len(fields))
    return fields


def _store_indexed(fields: dict[str, str], stem: str, cells: list[Tag]) -> None:
    """Record one row of per-side cells as numbered fields.

    Args:
        fields: The field mapping being built, modified in place.
        stem: The field stem the row states, e.g. ``"strength"``.
        cells: The row's value cells, in belligerent order.
    """
    for index, cell in enumerate(cells, start=1):
        text = _cell_text(cell)
        if text:
            fields.setdefault(f"{stem}{index}", text)
