"""
Map a Wikidata entity onto the battles schema.

Wikidata is the most reliable thing this stage reads. Its dates and
coordinates are curated and machine-checkable, so they outrank the infobox,
which outranks anything a model read out of prose.

The awkward part is dates. Wikidata stores times in a form Python's
``date`` cannot hold: ``-0031-09-02`` means 2 September 31 BC, and a
quarter of this project's corpus is BC. PostgreSQL can store those, so the
mapper emits a literal Postgres accepts (``0031-09-02 BC``) together with
the precision, rather than losing the era by forcing it through
``datetime.date``.
"""

from __future__ import annotations

import re
from typing import Any, Final

import structlog

from pipeline.extractors.records import BattleFacts, Provenance, SourceExtraction

__all__ = [
    "PRECISION_NAMES",
    "map_entity",
    "wikidata_time_to_date",
]

logger = structlog.get_logger()

# Wikidata time precision codes. Anything coarser than a century is not
# useful for a battle and is reported as-is for the caller to reject.
PRECISION_NAMES: Final[dict[int, str]] = {
    11: "day",
    10: "month",
    9: "year",
    8: "decade",
    7: "century",
    6: "millennium",
}

_P_POINT_IN_TIME: Final = "P585"
_P_START_TIME: Final = "P580"
_P_END_TIME: Final = "P582"
_P_COORDINATES: Final = "P625"
_P_LOCATION: Final = "P276"
_P_COUNTRY: Final = "P17"
_P_PART_OF: Final = "P361"
_P_PARTICIPANT: Final = "P710"
_P_DEATHS: Final = "P1120"

_TIME_RE: Final = re.compile(
    r"^(?P<sign>[+-])(?P<year>\d{4,11})-(?P<month>\d{2})-(?P<day>\d{2})"
)


def wikidata_time_to_date(value: dict[str, Any]) -> tuple[str | None, str | None]:
    """Convert a Wikidata time value into a Postgres date literal.

    Args:
        value: The ``datavalue.value`` mapping of a time claim, carrying
            ``time`` and ``precision``.

    Returns:
        A (date literal, precision name) pair. The literal carries a ``BC``
        suffix for negative years, which PostgreSQL's DATE type accepts and
        ``datetime.date`` cannot represent. Both members are None when the
        value is unreadable or coarser than a century.
    """
    raw = str(value.get("time") or "")
    match = _TIME_RE.match(raw)
    if match is None:
        logger.debug("wikidata_time_unparseable", time=raw[:32])
        return None, None

    precision_code = value.get("precision")
    precision = (
        PRECISION_NAMES.get(int(precision_code)) if isinstance(precision_code, int) else None
    )
    if precision is None or precision == "millennium":
        return None, precision

    year = int(match.group("year"))
    if year == 0:
        return None, precision

    # Wikidata zeroes the month and day it does not know; Postgres needs a
    # real date, so the start of the period stands in and the precision
    # column says how far to trust it.
    month = max(1, int(match.group("month")))
    day = max(1, int(match.group("day")))
    if precision != "day":
        day = 1
    if precision in ("year", "decade", "century"):
        month = 1

    literal = f"{year:04d}-{month:02d}-{day:02d}"
    if match.group("sign") == "-":
        literal = f"{literal} BC"

    return literal, precision


def _entity(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Locate the entity object inside a crawled Wikidata payload.

    Args:
        payload: Either a ``wbgetentities`` response or a bare entity.

    Returns:
        The entity mapping, or None when the payload holds none.
    """
    entities = payload.get("entities")
    if isinstance(entities, dict) and entities:
        first = next(iter(entities.values()))
        return first if isinstance(first, dict) else None
    if "claims" in payload or "labels" in payload:
        return payload
    return None


def _claims(entity: dict[str, Any], prop: str) -> list[dict[str, Any]]:
    """Return the statements for one property.

    Args:
        entity: A Wikidata entity mapping.
        prop: A property id such as ``"P585"``.

    Returns:
        The statement list, empty when the property is absent.
    """
    claims = entity.get("claims")
    if not isinstance(claims, dict):
        return []
    statements = claims.get(prop)
    return [s for s in statements if isinstance(s, dict)] if isinstance(statements, list) else []


def _main_value(statement: dict[str, Any]) -> Any:
    """Read a statement's main snak value.

    Args:
        statement: One Wikidata statement.

    Returns:
        The ``datavalue.value``, or None when the snak has no value.
    """
    snak = statement.get("mainsnak")
    if not isinstance(snak, dict) or snak.get("snaktype") != "value":
        return None
    datavalue = snak.get("datavalue")
    if not isinstance(datavalue, dict):
        return None
    return datavalue.get("value")


def _first_time(entity: dict[str, Any], prop: str) -> tuple[str | None, str | None]:
    """Read the first time-valued statement for a property.

    Args:
        entity: A Wikidata entity mapping.
        prop: A time-valued property id.

    Returns:
        A (date literal, precision) pair, both None when absent.
    """
    for statement in _claims(entity, prop):
        value = _main_value(statement)
        if isinstance(value, dict):
            return wikidata_time_to_date(value)
    return None, None


def _qids(entity: dict[str, Any], prop: str) -> list[str]:
    """Read the entity ids referenced by a property.

    Args:
        entity: A Wikidata entity mapping.
        prop: An item-valued property id.

    Returns:
        Q-ids in statement order.
    """
    found: list[str] = []
    for statement in _claims(entity, prop):
        value = _main_value(statement)
        if isinstance(value, dict) and value.get("id"):
            found.append(str(value["id"]))
    return found


def _label(entity: dict[str, Any], language: str = "en") -> str | None:
    """Read an entity's label in one language.

    Args:
        entity: A Wikidata entity mapping.
        language: Language code.

    Returns:
        The label, or None when absent.
    """
    labels = entity.get("labels")
    if not isinstance(labels, dict):
        return None
    entry = labels.get(language)
    if isinstance(entry, dict) and entry.get("value"):
        return str(entry["value"])
    return None


def _wikipedia_url(entity: dict[str, Any]) -> str | None:
    """Read the English Wikipedia URL from an entity's sitelinks.

    Args:
        entity: A Wikidata entity mapping.

    Returns:
        The article URL, or None when there is no English sitelink.
    """
    sitelinks = entity.get("sitelinks")
    if not isinstance(sitelinks, dict):
        return None
    entry = sitelinks.get("enwiki")
    if not isinstance(entry, dict):
        return None
    if entry.get("url"):
        return str(entry["url"])
    title = entry.get("title")
    if title:
        return f"https://en.wikipedia.org/wiki/{str(title).replace(' ', '_')}"
    return None


def map_entity(
    payload: dict[str, Any],
    *,
    source_ref: str = "",
    fallback_name: str = "",
) -> SourceExtraction | None:
    """Map a crawled Wikidata payload onto schema fields.

    Args:
        payload: A ``wbgetentities`` response or a bare entity mapping.
        source_ref: URL or file path identifying the payload.
        fallback_name: Battle name to use when the entity has no English label.

    Returns:
        What Wikidata says about the battle, or None when the payload holds
        no entity. No sides are produced: Wikidata's participant statements
        name polities and people without saying which fought which, so
        attributing them to sides here would be an invention.
    """
    entity = _entity(payload)
    if entity is None:
        logger.info("wikidata_payload_has_no_entity", source=source_ref)
        return None

    entity_id = str(entity.get("id") or "") or None

    date_start, precision = _first_time(entity, _P_POINT_IN_TIME)
    if date_start is None:
        date_start, precision = _first_time(entity, _P_START_TIME)
    date_end, _ = _first_time(entity, _P_END_TIME)

    latitude: float | None = None
    longitude: float | None = None
    for statement in _claims(entity, _P_COORDINATES):
        value = _main_value(statement)
        if isinstance(value, dict) and "latitude" in value and "longitude" in value:
            latitude = float(value["latitude"])
            longitude = float(value["longitude"])
            break

    location_qids = _qids(entity, _P_LOCATION) or _qids(entity, _P_COUNTRY)

    facts = BattleFacts(
        name=_label(entity) or fallback_name or None,
        wikidata_id=entity_id,
        wikipedia_url=_wikipedia_url(entity),
        date_start=date_start,
        date_end=date_end,
        date_precision=precision,
        latitude=latitude,
        longitude=longitude,
        location_name=location_qids[0] if location_qids else None,
        part_of=_qids(entity, _P_PART_OF),
    )

    notes: list[str] = []
    participants = _qids(entity, _P_PARTICIPANT)
    if participants:
        notes.append(f"wikidata participants (unassigned to sides): {', '.join(participants)}")
    for statement in _claims(entity, _P_DEATHS):
        value = _main_value(statement)
        if isinstance(value, dict) and value.get("amount"):
            notes.append(f"wikidata total deaths (unassigned to a side): {value['amount']}")

    return SourceExtraction(
        provenance=Provenance(
            source_type="wikidata",
            extraction_method="wikidata_sparql",
            source_ref=source_ref,
            source_title=facts.name or entity_id or "wikidata",
        ),
        facts=facts,
        sides=[],
        notes=notes,
    )
