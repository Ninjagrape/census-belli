"""
Map a DBpedia resource onto the battles schema.

DBpedia re-publishes Wikipedia's infoboxes as RDF with the template mess
already resolved, which makes it a useful second opinion on the fields an
infobox parse gets wrong: dates are typed, coordinates are numeric, and the
``dbo:`` ontology properties have been cleaned where the raw ``dbp:`` ones
have not.

**It cannot say who fought on which side, and this mapper does not pretend
otherwise.** The RDF flattens the infobox's ``combatant1``/``combatant2`` and
``commander1``/``commander2`` into single unordered lists, losing the only
thing that assigned them. Measured against the live endpoint on 2026-09-19:

- Actium's ``dbo:combatant`` is ``["Ptolemaic Egypt", "Octavian's forces",
  "Antony's forces"]`` -- three entries for two sides, in no order.
- Trafalgar's is ``["Spain"]`` alone -- one entry, for the battle where a
  British fleet fought a Franco-Spanish one -- and its ``dbo:commander``
  lists Villeneuve, Collingwood, Gravina and Nelson in one flat list with no
  marker for which fleet anyone belonged to.

Splitting that into sides would be invention, and a commander attributed to
the wrong side is the error the model cannot see: it fits one skill parameter
to a career that includes battles the general fought against. So this mapper
returns ``sides=[]`` and puts the belligerents, commanders, strengths and
casualties in ``notes``, exactly as the Wikidata mapper does with
``P710 participant`` and for the same reason. The infobox parser reads
``combatant1``/``combatant2`` from the wikitext directly and keeps the
assignment; that is the source sides should come from.

Dates carry the same BC problem as Wikidata. DBpedia writes 2 September 31 BC
as ``-031-09-02`` -- note three year digits, not four -- which no
``datetime.date`` can hold, so the mapper emits the ``0031-09-02 BC`` literal
PostgreSQL accepts instead.
"""

from __future__ import annotations

import re
from typing import Any, Final
from urllib.parse import unquote

import structlog

from pipeline.extractors.records import BattleFacts, Provenance, SourceExtraction

__all__ = [
    "DBPEDIA_RESOURCE_PREFIX",
    "dbpedia_date_to_literal",
    "map_resource",
    "resource_label",
]

logger = structlog.get_logger()

DBPEDIA_RESOURCE_PREFIX: Final = "http://dbpedia.org/resource/"

_ONTOLOGY: Final = "http://dbpedia.org/ontology/"
_PROPERTY: Final = "http://dbpedia.org/property/"
_GEO: Final = "http://www.w3.org/2003/01/geo/wgs84_pos#"
_RDFS_LABEL: Final = "http://www.w3.org/2000/01/rdf-schema#label"
_FOAF_PRIMARY_TOPIC_OF: Final = "http://xmlns.com/foaf/0.1/isPrimaryTopicOf"

# DBpedia dates are ISO-like but the year is not zero-padded to four digits
# and may be negative: "-031-09-02" is 2 September 31 BC.
_DATE_RE: Final = re.compile(r"^(?P<sign>-?)(?P<year>\d{1,6})-(?P<month>\d{2})-(?P<day>\d{2})$")

# A free-text value longer than this is prose rather than a field value, and
# putting prose in location_name would poison a column the report reads.
_MAX_LOCATION_CHARS: Final = 200

# Notes are provenance, not payload. A battle with fifty commanders would
# otherwise write fifty lines nothing reads.
_MAX_NOTE_ITEMS: Final = 12


def resource_label(uri: str) -> str:
    """Turn a DBpedia resource URI into a readable label.

    Args:
        uri: A resource URI, or any string.

    Returns:
        The percent-decoded final segment with underscores replaced, or the
        input unchanged when it is not a DBpedia resource URI.
    """
    if not uri.startswith(DBPEDIA_RESOURCE_PREFIX):
        return uri
    return unquote(uri[len(DBPEDIA_RESOURCE_PREFIX) :]).replace("_", " ").strip()


def dbpedia_date_to_literal(raw: str) -> tuple[str | None, str | None]:
    """Convert a DBpedia date literal into one PostgreSQL accepts.

    Args:
        raw: The literal value of a date-typed statement, e.g. ``-031-09-02``.

    Returns:
        A (date literal, precision) pair. The literal carries a ``BC`` suffix
        for negative years, which ``datetime.date`` cannot represent and
        PostgreSQL's DATE can. Both are None when the value is unreadable.
    """
    match = _DATE_RE.match(raw.strip())
    if match is None:
        logger.debug("dbpedia_date_unparseable", value=raw[:32])
        return None, None

    year = int(match.group("year"))
    if year == 0:
        # There is no year zero. A source that emits one is not describing a
        # real date, and guessing which side of the boundary it meant would
        # put a silent off-by-one into every date-derived covariate.
        logger.debug("dbpedia_date_year_zero", value=raw[:32])
        return None, None

    month = int(match.group("month"))
    day = int(match.group("day"))
    if not (1 <= month <= 12) or not (1 <= day <= 31):
        logger.debug("dbpedia_date_out_of_range", value=raw[:32])
        return None, None

    literal = f"{year:04d}-{month:02d}-{day:02d}"
    if match.group("sign") == "-":
        literal = f"{literal} BC"
    return literal, "day"


def _resource(payload: dict[str, Any], subject: str | None) -> tuple[str, dict[str, Any]] | None:
    """Locate the battle resource inside a DBpedia JSON payload.

    A ``/data/<Title>.json`` document describes hundreds of resources; the one
    the crawl asked for is keyed by its own URI.

    Args:
        payload: The decoded document.
        subject: The resource URI to look for, when the caller knows it.

    Returns:
        A (uri, predicates) pair, or None when no battle resource is present.
    """
    if subject and isinstance(payload.get(subject), dict):
        return subject, payload[subject]

    # Fall back to the resource carrying the properties a battle has. The
    # document also describes every entity the article links to, so taking
    # the first key would usually take the wrong one.
    best: tuple[str, dict[str, Any]] | None = None
    best_score = 0
    for uri, predicates in payload.items():
        if not uri.startswith(DBPEDIA_RESOURCE_PREFIX) or not isinstance(predicates, dict):
            continue
        score = sum(
            1
            for marker in ("date", "result", "combatant", "commander", "place")
            if f"{_ONTOLOGY}{marker}" in predicates
        )
        if score > best_score:
            best, best_score = (uri, predicates), score

    return best


def _values(
    predicates: dict[str, Any], uri: str, *, want: str | None = None, lang: str | None = None
) -> list[str]:
    """Read the values of one predicate.

    Args:
        predicates: The resource's predicate mapping.
        uri: The full predicate URI.
        want: Keep only values of this RDF type, ``"literal"`` or ``"uri"``.
        lang: Keep only literals in this language, plus untagged ones.

    Returns:
        The values in document order.
    """
    entries = predicates.get(uri)
    if not isinstance(entries, list):
        return []

    found: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if want is not None and entry.get("type") != want:
            continue
        if lang is not None and entry.get("lang") not in (None, lang):
            continue
        value = entry.get("value")
        if value is not None and str(value).strip():
            found.append(str(value).strip())
    return found


def _first_float(predicates: dict[str, Any], uri: str) -> float | None:
    """Read the first numeric value of a predicate.

    Args:
        predicates: The resource's predicate mapping.
        uri: The full predicate URI.

    Returns:
        The value as a float, or None when absent or non-numeric.
    """
    for raw in _values(predicates, uri):
        try:
            return float(raw)
        except ValueError:
            continue
    return None


def _first_date(predicates: dict[str, Any]) -> tuple[str | None, str | None]:
    """Read the battle's date, preferring the cleaned ontology property.

    ``dbp:date`` carries whatever the infobox held, which on Actium includes
    a 2014 citation access date alongside the real one. ``dbo:date`` is the
    extracted, typed value, so it is tried first and the raw property is only
    a fallback.

    Args:
        predicates: The resource's predicate mapping.

    Returns:
        A (date literal, precision) pair, both None when unreadable.
    """
    for uri in (f"{_ONTOLOGY}date", f"{_PROPERTY}date"):
        for raw in _values(predicates, uri, want="literal"):
            literal, precision = dbpedia_date_to_literal(raw)
            if literal is not None:
                return literal, precision
    return None, None


def _location_name(predicates: dict[str, Any]) -> str | None:
    """Read a readable place name.

    Args:
        predicates: The resource's predicate mapping.

    Returns:
        The place as text, or None when absent or implausibly long.
    """
    for raw in _values(predicates, f"{_PROPERTY}place", want="literal", lang="en"):
        if len(raw) <= _MAX_LOCATION_CHARS:
            return raw
    for uri in _values(predicates, f"{_ONTOLOGY}place", want="uri"):
        label = resource_label(uri)
        if label and len(label) <= _MAX_LOCATION_CHARS:
            return label
    return None


def _english_label(predicates: dict[str, Any], uri: str) -> str | None:
    """Read the resource's English label.

    Args:
        predicates: The resource's predicate mapping.
        uri: The resource's own URI, used as a fallback.

    Returns:
        The label, falling back to the URI's final segment.
    """
    entries = predicates.get(_RDFS_LABEL)
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, dict) and entry.get("lang") == "en" and entry.get("value"):
                return str(entry["value"]).strip()
    return resource_label(uri) or None


def _unassigned_notes(predicates: dict[str, Any]) -> list[str]:
    """Record what DBpedia reports but cannot attribute to a side.

    Args:
        predicates: The resource's predicate mapping.

    Returns:
        One note per populated field, each saying plainly that the values are
        unassigned so no downstream stage reads them as side data.
    """
    notes: list[str] = []

    def add(label: str, values: list[str]) -> None:
        if not values:
            return
        shown = values[:_MAX_NOTE_ITEMS]
        suffix = f" (+{len(values) - len(shown)} more)" if len(values) > len(shown) else ""
        notes.append(f"dbpedia {label} (unassigned to sides): {', '.join(shown)}{suffix}")

    combatants = _values(predicates, f"{_ONTOLOGY}combatant", want="literal") or _values(
        predicates, f"{_PROPERTY}combatant", want="literal", lang="en"
    )
    add("combatants", combatants)

    commanders = [
        resource_label(uri) for uri in _values(predicates, f"{_ONTOLOGY}commander", want="uri")
    ]
    add("commanders", [c for c in commanders if c])

    add("strength", _values(predicates, f"{_ONTOLOGY}strength", want="literal"))
    add("casualties", _values(predicates, f"{_PROPERTY}casualties", want="literal", lang="en"))

    return notes


def map_resource(
    payload: dict[str, Any],
    *,
    source_ref: str = "",
    subject: str | None = None,
    fallback_name: str = "",
) -> SourceExtraction | None:
    """Map a crawled DBpedia payload onto schema fields.

    Args:
        payload: A decoded ``https://dbpedia.org/data/<Title>.json`` document.
        source_ref: URL or file path identifying the payload.
        subject: The resource URI to read, when known. Without it the mapper
            picks the resource carrying battle properties, since the document
            describes every entity the article links to.
        fallback_name: Battle name to use when the resource has no English
            label.

    Returns:
        What DBpedia says about the battle, or None when the payload holds no
        battle resource. No sides are produced: the RDF flattens the
        infobox's numbered combatant and commander fields into unordered
        lists, so assigning them to sides would be an invention. They are
        reported in ``notes`` instead.
    """
    located = _resource(payload, subject)
    if located is None:
        logger.info("dbpedia_payload_has_no_resource", source=source_ref)
        return None

    uri, predicates = located

    date_start, precision = _first_date(predicates)
    result = _values(predicates, f"{_ONTOLOGY}result", want="literal") or _values(
        predicates, f"{_PROPERTY}result", want="literal", lang="en"
    )
    part_of = [
        resource_label(value)
        for value in _values(predicates, f"{_ONTOLOGY}isPartOfMilitaryConflict", want="uri")
    ]
    wikipedia_url = next(iter(_values(predicates, _FOAF_PRIMARY_TOPIC_OF, want="uri")), None)

    facts = BattleFacts(
        name=_english_label(predicates, uri) or fallback_name or None,
        wikipedia_url=wikipedia_url,
        date_start=date_start,
        date_precision=precision,
        latitude=_first_float(predicates, f"{_GEO}lat"),
        longitude=_first_float(predicates, f"{_GEO}long"),
        location_name=_location_name(predicates),
        # victor is the battle-level statement of who won. outcome_level is
        # deliberately left unset: it is the model's five-level ordinal, and
        # "Octavian victory" does not say whether that was decisive.
        victor=result[0] if result else None,
        part_of=[p for p in part_of if p],
    )

    return SourceExtraction(
        provenance=Provenance(
            source_type="dbpedia",
            extraction_method="dbpedia_rdf",
            source_ref=source_ref or uri,
            source_title=facts.name or uri,
        ),
        facts=facts,
        sides=[],
        notes=_unassigned_notes(predicates),
    )
