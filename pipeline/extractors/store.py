"""
Write merged extractions into the database.

The extract stage runs before resolve, so what it can write is limited:
``battles``, ``battle_sides``, ``troop_reports`` and ``casualty_reports``.
Commander mentions stay in ``data/processed/commanders_raw.jsonl`` until
resolve has canonical ``generals`` rows to point ``battle_commanders`` at.

Re-running the stage must not duplicate reports. Each side's reports that
this stage produced are deleted and rewritten, scoped to the extraction
methods the stage owns, so a re-run converges rather than accumulating. Rows
written by a later stage or by hand are left alone.
"""

from __future__ import annotations

from typing import Any, Final

import structlog
from sqlalchemy import text

from pipeline.extractors.dates import sanitise_date_fields
from pipeline.extractors.records import (
    BattleExtraction,
    BattleFacts,
    ExtractionFailure,
    Provenance,
    SideExtraction,
)

__all__ = ["ExtractWriteCounts", "source_id", "write_battle", "write_battle_isolated"]

logger = structlog.get_logger()

# Extraction methods this stage owns. A re-run clears only rows carrying one
# of these, so manual corrections survive.
_OWNED_METHODS: Final[tuple[str, ...]] = (
    "infobox_parser",
    "llm_extraction",
    "wikidata_sparql",
    "dbpedia_rdf",
)

_FIND_SOURCE = text(
    "SELECT source_id FROM sources WHERE source_type = CAST(:source_type AS source_type) "
    "AND COALESCE(url, '') = COALESCE(:url, '') LIMIT 1"
)
_INSERT_SOURCE = text(
    "INSERT INTO sources (source_type, title, url) "
    "VALUES (CAST(:source_type AS source_type), :title, :url) RETURNING source_id"
)

_FIND_BATTLE_BY_QID = text("SELECT battle_id FROM battles WHERE wikidata_id = :wikidata_id")
_FIND_BATTLE_BY_NAME = text("SELECT battle_id FROM battles WHERE name = :name LIMIT 1")

_INSERT_BATTLE = text(
    """
    INSERT INTO battles (
        name, wikidata_id, wikipedia_url, date_start, date_end, date_precision,
        latitude, longitude, location_name, battle_type, terrain, fortified,
        weather, n_sources, needs_review, review_notes
    ) VALUES (
        :name, :wikidata_id, :wikipedia_url,
        CAST(:date_start AS DATE), CAST(:date_end AS DATE), :date_precision,
        :latitude, :longitude, :location_name,
        CAST(COALESCE(:battle_type, 'unknown') AS battle_type), :terrain, :fortified,
        :weather, :n_sources, :needs_review, :review_notes
    )
    RETURNING battle_id
    """
)

_UPDATE_BATTLE = text(
    """
    UPDATE battles SET
        name = :name,
        wikidata_id = COALESCE(:wikidata_id, wikidata_id),
        wikipedia_url = COALESCE(:wikipedia_url, wikipedia_url),
        date_start = COALESCE(CAST(:date_start AS DATE), date_start),
        date_end = COALESCE(CAST(:date_end AS DATE), date_end),
        date_precision = COALESCE(:date_precision, date_precision),
        latitude = COALESCE(:latitude, latitude),
        longitude = COALESCE(:longitude, longitude),
        location_name = COALESCE(:location_name, location_name),
        battle_type = COALESCE(CAST(:battle_type AS battle_type), battle_type),
        terrain = COALESCE(:terrain, terrain),
        fortified = COALESCE(:fortified, fortified),
        weather = COALESCE(:weather, weather),
        n_sources = :n_sources,
        needs_review = :needs_review,
        review_notes = :review_notes,
        updated_at = now()
    WHERE battle_id = :battle_id
    """
)

_FIND_SIDE = text(
    "SELECT side_id FROM battle_sides WHERE battle_id = :battle_id AND side_label = :side_label"
)
_INSERT_SIDE = text(
    "INSERT INTO battle_sides (battle_id, side_label, polity, outcome) "
    "VALUES (:battle_id, :side_label, :polity, CAST(:outcome AS outcome_level)) "
    "RETURNING side_id"
)
_UPDATE_SIDE = text(
    "UPDATE battle_sides SET polity = COALESCE(:polity, polity), "
    "outcome = COALESCE(CAST(:outcome AS outcome_level), outcome) WHERE side_id = :side_id"
)

_CLEAR_TROOPS = text(
    "DELETE FROM troop_reports WHERE side_id = :side_id "
    "AND extraction_method = ANY(CAST(:methods AS extraction_method[]))"
)
_CLEAR_CASUALTIES = text(
    "DELETE FROM casualty_reports WHERE side_id = :side_id "
    "AND extraction_method = ANY(CAST(:methods AS extraction_method[]))"
)

_INSERT_TROOP = text(
    """
    INSERT INTO troop_reports (
        side_id, source_id, branch, reported_value, scope,
        is_estimate, is_upper_bound, is_lower_bound,
        extraction_method, extracted_context, page_or_section
    ) VALUES (
        :side_id, :source_id, CAST(:branch AS troop_branch), :reported_value, :scope,
        :is_estimate, :is_upper_bound, :is_lower_bound,
        CAST(:extraction_method AS extraction_method), :extracted_context, :page_or_section
    )
    """
)

_INSERT_CASUALTY = text(
    """
    INSERT INTO casualty_reports (
        side_id, source_id, casualty_type, reported_value, is_estimate,
        extraction_method, extracted_context
    ) VALUES (
        :side_id, :source_id, :casualty_type, :reported_value, :is_estimate,
        CAST(:extraction_method AS extraction_method), :extracted_context
    )
    """
)

_CLEAR_MISSING = text(
    "DELETE FROM missing_data_log WHERE battle_id = :battle_id AND was_imputed = FALSE"
)
_INSERT_MISSING = text(
    """
    INSERT INTO missing_data_log (battle_id, side_id, field_name, notes)
    VALUES (:battle_id, :side_id, :field_name, :notes)
    """
)


class ExtractWriteCounts:
    """Row counts for one stage run, for the summary log."""

    def __init__(self) -> None:
        self.battles = 0
        self.sides = 0
        self.troop_reports = 0
        self.casualty_reports = 0
        self.missing_rows = 0
        self.failed_battles = 0

    def as_dict(self) -> dict[str, int]:
        """Render the counts for a structlog call.

        Returns:
            A flat mapping of counter name to value.
        """
        return {
            "battles": self.battles,
            "sides": self.sides,
            "troop_reports": self.troop_reports,
            "casualty_reports": self.casualty_reports,
            "missing_rows": self.missing_rows,
            "failed_battles": self.failed_battles,
        }


def source_id(conn: Any, provenance: Provenance, cache: dict[tuple[str, str], int]) -> int:
    """Find or create the ``sources`` row a claim should hang off.

    Args:
        conn: An open database connection.
        provenance: Where the claim came from.
        cache: Per-run memo so one document costs one lookup.

    Returns:
        The source id.
    """
    key = (provenance.source_type, provenance.source_ref)
    if key in cache:
        return cache[key]

    params = {
        "source_type": provenance.source_type,
        "url": provenance.source_ref or None,
        "title": provenance.source_title or provenance.source_ref or provenance.source_type,
    }
    row = conn.execute(_FIND_SOURCE, params).fetchone()
    if row is None:
        row = conn.execute(_INSERT_SOURCE, params).fetchone()

    source_id = int(row[0])
    cache[key] = source_id
    return source_id


def _upsert_battle(conn: Any, battle: BattleExtraction) -> int:
    """Insert or update the ``battles`` row.

    Args:
        conn: An open database connection.
        battle: The merged battle.

    Returns:
        The battle id.
    """
    facts = battle.facts
    # Last line of defence: raw text in a DATE cast aborts the transaction.
    date_start, date_end, date_precision = sanitise_date_fields(
        facts.date_start, facts.date_end, facts.date_precision
    )
    params: dict[str, Any] = {
        "name": battle.name,
        "wikidata_id": facts.wikidata_id,
        "wikipedia_url": facts.wikipedia_url,
        "date_start": date_start,
        "date_end": date_end,
        "date_precision": date_precision,
        "latitude": facts.latitude,
        "longitude": facts.longitude,
        "location_name": facts.location_name,
        "battle_type": facts.battle_type,
        "terrain": facts.terrain or None,
        "fortified": facts.fortified,
        "weather": facts.weather,
        "n_sources": battle.n_sources,
        "needs_review": battle.needs_review,
        "review_notes": battle.review_notes or None,
    }

    existing = None
    if facts.wikidata_id:
        existing = conn.execute(_FIND_BATTLE_BY_QID, {"wikidata_id": facts.wikidata_id}).fetchone()
    if existing is None:
        existing = conn.execute(_FIND_BATTLE_BY_NAME, {"name": battle.name}).fetchone()

    if existing is None:
        row = conn.execute(_INSERT_BATTLE, params).fetchone()
        return int(row[0])

    battle_id = int(existing[0])
    conn.execute(_UPDATE_BATTLE, {**params, "battle_id": battle_id})
    return battle_id


def _upsert_side(conn: Any, battle_id: int, side: SideExtraction) -> int:
    """Insert or update one ``battle_sides`` row.

    Args:
        conn: An open database connection.
        battle_id: The battle this side belongs to.
        side: The merged side.

    Returns:
        The side id.
    """
    params = {
        "battle_id": battle_id,
        "side_label": side.label,
        "polity": side.polity,
        "outcome": side.outcome,
    }
    existing = conn.execute(
        _FIND_SIDE, {"battle_id": battle_id, "side_label": side.label}
    ).fetchone()

    if existing is None:
        row = conn.execute(_INSERT_SIDE, params).fetchone()
        return int(row[0])

    side_id = int(existing[0])
    conn.execute(_UPDATE_SIDE, {**params, "side_id": side_id})
    return side_id


def write_battle(
    conn: Any,
    battle: BattleExtraction,
    counts: ExtractWriteCounts | None = None,
    source_cache: dict[tuple[str, str], int] | None = None,
) -> int:
    """Persist one merged battle and everything hanging off it.

    Args:
        conn: An open database connection.
        battle: The merged battle.
        counts: Optional counters to accumulate into.
        source_cache: Optional per-run memo of source ids.

    Returns:
        The battle id.
    """
    tally = counts or ExtractWriteCounts()
    cache = source_cache if source_cache is not None else {}

    battle_id = _upsert_battle(conn, battle)
    tally.battles += 1

    side_ids: dict[str, int] = {}
    for side in battle.sides:
        side_id = _upsert_side(conn, battle_id, side)
        side_ids[side.label] = side_id
        tally.sides += 1

        conn.execute(_CLEAR_TROOPS, {"side_id": side_id, "methods": list(_OWNED_METHODS)})
        conn.execute(_CLEAR_CASUALTIES, {"side_id": side_id, "methods": list(_OWNED_METHODS)})

        for report in side.troop_reports:
            conn.execute(
                _INSERT_TROOP,
                {
                    "side_id": side_id,
                    "source_id": source_id(conn, report.provenance, cache),
                    "branch": report.branch,
                    "reported_value": report.reported_value,
                    "scope": report.scope,
                    "is_estimate": report.is_estimate,
                    "is_upper_bound": report.is_upper_bound,
                    "is_lower_bound": report.is_lower_bound,
                    "extraction_method": report.provenance.extraction_method,
                    "extracted_context": report.extracted_context or None,
                    "page_or_section": report.page_or_section,
                },
            )
            tally.troop_reports += 1

        for casualty in side.casualty_reports:
            conn.execute(
                _INSERT_CASUALTY,
                {
                    "side_id": side_id,
                    "source_id": source_id(conn, casualty.provenance, cache),
                    "casualty_type": casualty.casualty_type,
                    "reported_value": casualty.reported_value,
                    "is_estimate": casualty.is_estimate,
                    "extraction_method": casualty.provenance.extraction_method,
                    "extracted_context": casualty.extracted_context or None,
                },
            )
            tally.casualty_reports += 1

    conn.execute(_CLEAR_MISSING, {"battle_id": battle_id})
    for missing in battle.missing:
        conn.execute(
            _INSERT_MISSING,
            {
                "battle_id": battle_id,
                "side_id": side_ids.get(missing.side_label) if missing.side_label else None,
                "field_name": missing.field_name,
                "notes": missing.notes or None,
            },
        )
        tally.missing_rows += 1

    return battle_id


def write_battle_isolated(
    conn: Any,
    battle: BattleExtraction,
    counts: ExtractWriteCounts,
    source_cache: dict[tuple[str, str], int],
) -> int | None:
    """Write one battle inside a savepoint so a failure cannot poison the rest.

    On PostgreSQL the first failing statement aborts the whole transaction;
    without a savepoint every later battle raises ``InFailedSqlTransaction``
    and the final commit rolls back, leaving no battles at all. A failed
    battle is rolled back alone, counted, and then written again as a bare
    row flagged ``needs_review`` so it is not silently absent.

    Args:
        conn: An open database connection.
        battle: The merged battle.
        counts: Counters to accumulate into; restored for a failed battle.
        source_cache: Per-run memo of source ids. Entries added by a failed
            battle are dropped, since the rows they point at were rolled back.

    Returns:
        The battle id, or None when the battle (and its review stub) failed.
    """
    before_counts = counts.as_dict()
    before_cache = dict(source_cache)
    try:
        with conn.begin_nested():
            return write_battle(conn, battle, counts, source_cache)
    except Exception as exc:
        logger.error(
            "extract_write_failed",
            slug=battle.slug,
            error=str(exc)[:300],
            error_type=type(exc).__name__,
        )
        _restore(counts, before_counts)
        source_cache.clear()
        source_cache.update(before_cache)
        counts.failed_battles += 1

    return _write_review_stub(conn, battle, counts)


def _restore(counts: ExtractWriteCounts, snapshot: dict[str, int]) -> None:
    """Reset every counter except ``failed_battles`` to a snapshot."""
    for name, value in snapshot.items():
        if name != "failed_battles":
            setattr(counts, name, value)


def _write_review_stub(
    conn: Any, battle: BattleExtraction, counts: ExtractWriteCounts
) -> int | None:
    """Record that a battle could not be written, as a flagged bare row."""
    stub = BattleExtraction(
        slug=battle.slug,
        name=battle.name,
        facts=BattleFacts(
            name=battle.facts.name,
            wikidata_id=battle.facts.wikidata_id,
            wikipedia_url=battle.facts.wikipedia_url,
        ),
        failures=[
            ExtractionFailure(
                battle_name=battle.name,
                source_ref=battle.facts.wikipedia_url or battle.slug,
                status="write_failed",
                error="database rejected the extracted record; details in the run log",
            )
        ],
        n_sources=battle.n_sources,
    )
    try:
        with conn.begin_nested():
            return _upsert_battle(conn, stub)
    except Exception as exc:
        logger.error("extract_review_stub_failed", slug=battle.slug, error=str(exc)[:300])
        return None
