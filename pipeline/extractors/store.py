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
from pipeline.extractors.wars import classify_part_of

__all__ = [
    "ExtractWriteCounts",
    "reserve_battle_id",
    "source_id",
    "write_battle",
    "write_battle_isolated",
]

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
_BATTLE_SLUG = text("SELECT source_slug FROM battles WHERE battle_id = :battle_id")
_FIND_BATTLE_BY_SLUG = text("SELECT battle_id FROM battles WHERE source_slug = :slug")
# Only a row that predates source_slug may be claimed by name. A row that
# already has a slug belongs to a different article that happens to share
# the display name.
_FIND_LEGACY_BATTLE_BY_NAME = text(
    "SELECT battle_id FROM battles WHERE name = :name AND source_slug IS NULL "
    "ORDER BY battle_id LIMIT 1"
)
_RESERVE_BATTLE = text(
    "INSERT INTO battles (name, source_slug) VALUES (:name, :slug) RETURNING battle_id"
)

_INSERT_BATTLE = text(
    """
    INSERT INTO battles (
        name, source_slug, wikidata_id, wikipedia_url, date_start, date_end, date_precision,
        date_circa, latitude, longitude, location_name, battle_type, battle_type_source,
        terrain, fortified, weather, n_sources, needs_review, review_notes
    ) VALUES (
        :name, :source_slug, :wikidata_id, :wikipedia_url,
        CAST(:date_start AS DATE), CAST(:date_end AS DATE), :date_precision,
        :date_circa, :latitude, :longitude, :location_name,
        CAST(COALESCE(:battle_type, 'unknown') AS battle_type),
        CASE WHEN CAST(:battle_type AS TEXT) IS NOT NULL THEN 'extract' END,
        :terrain, :fortified,
        :weather, :n_sources, :needs_review, :review_notes
    )
    RETURNING battle_id
    """
)

_UPDATE_BATTLE = text(
    """
    UPDATE battles SET
        name = :name,
        source_slug = COALESCE(source_slug, :source_slug),
        wikidata_id = COALESCE(:wikidata_id, wikidata_id),
        wikipedia_url = COALESCE(:wikipedia_url, wikipedia_url),
        date_start = COALESCE(CAST(:date_start AS DATE), date_start),
        date_end = COALESCE(CAST(:date_end AS DATE), date_end),
        date_precision = COALESCE(:date_precision, date_precision),
        -- circa describes the date_start it arrived with, so it moves with it.
        date_circa = CASE WHEN CAST(:date_start AS DATE) IS NULL THEN date_circa
                          ELSE :date_circa END,
        latitude = COALESCE(:latitude, latitude),
        longitude = COALESCE(:longitude, longitude),
        location_name = COALESCE(:location_name, location_name),
        battle_type = COALESCE(CAST(:battle_type AS battle_type), battle_type),
        -- Provenance follows the value: only a type this run supplied is extract's.
        battle_type_source = CASE WHEN CAST(:battle_type AS TEXT) IS NOT NULL THEN 'extract'
                                  ELSE battle_type_source END,
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

# wars and campaigns have no UNIQUE name, so the link is select-then-insert.
_FIND_WAR = text("SELECT war_id FROM wars WHERE name = :name ORDER BY war_id LIMIT 1")
_INSERT_WAR = text("INSERT INTO wars (name) VALUES (:name) RETURNING war_id")
_FIND_CAMPAIGN = text(
    "SELECT campaign_id FROM campaigns WHERE name = :name "
    "AND war_id IS NOT DISTINCT FROM CAST(:war_id AS INT) ORDER BY campaign_id LIMIT 1"
)
_INSERT_CAMPAIGN = text(
    "INSERT INTO campaigns (name, war_id) VALUES (:name, CAST(:war_id AS INT)) "
    "RETURNING campaign_id"
)
# COALESCE so a re-run whose sources named nothing never undoes an earlier link.
_LINK_BATTLE = text(
    "UPDATE battles SET war_id = COALESCE(CAST(:war_id AS INT), war_id), "
    "campaign_id = COALESCE(CAST(:campaign_id AS INT), campaign_id) WHERE battle_id = :battle_id"
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

# A side this run no longer produces is removed with everything on it. There
# are no FK cascades, so dependents go first; a commander on a surviving side
# may report to one on a stale side, so that link is cut before the delete.
_FIND_STALE_SIDES = text(
    "SELECT side_id, side_label FROM battle_sides WHERE battle_id = :battle_id "
    "AND NOT (side_id = ANY(CAST(:keep AS INT[]))) ORDER BY side_id"
)
_DETACH_STALE_PARENTS = text(
    "UPDATE battle_commanders SET reports_to_bc_id = NULL WHERE reports_to_bc_id IN "
    "(SELECT bc_id FROM battle_commanders WHERE side_id = ANY(CAST(:ids AS INT[])))"
)
_COUNT_FOREIGN_REPORTS: Final[dict[str, Any]] = {
    table: text(
        f"SELECT COUNT(*) FROM {table} WHERE side_id = ANY(CAST(:ids AS INT[])) "  # noqa: S608
        "AND NOT (COALESCE(CAST(extraction_method AS TEXT), '') = ANY(CAST(:methods AS TEXT[])))"
    )
    for table in ("troop_reports", "casualty_reports")
}
# Extraction methods only the LLM body pass produces. A stale side holding
# such a row may be waiting for that pass to be answered, see
# ``_remove_stale_sides``.
_LLM_METHODS: Final[tuple[str, ...]] = ("llm_extraction",)
_FIND_SIDES_WITH_LLM_ROWS = text(
    "SELECT side_id FROM troop_reports WHERE side_id = ANY(CAST(:ids AS INT[])) "
    "AND CAST(extraction_method AS TEXT) = ANY(CAST(:methods AS TEXT[])) "
    "UNION SELECT side_id FROM casualty_reports WHERE side_id = ANY(CAST(:ids AS INT[])) "
    "AND CAST(extraction_method AS TEXT) = ANY(CAST(:methods AS TEXT[])) "
    "UNION SELECT side_id FROM battle_commanders WHERE side_id = ANY(CAST(:ids AS INT[])) "
    "AND CAST(extraction_method AS TEXT) = ANY(CAST(:methods AS TEXT[]))"
)
# Extract writes no commanders, and only un-imputed missing rows; anything
# else on a stale side belongs to another stage or to a person.
_COUNT_FOREIGN_COMMANDERS = text(
    "SELECT COUNT(*) FROM battle_commanders WHERE side_id = ANY(CAST(:ids AS INT[]))"
)
_COUNT_FOREIGN_MISSING = text(
    "SELECT COUNT(*) FROM missing_data_log WHERE side_id = ANY(CAST(:ids AS INT[])) "
    "AND was_imputed = TRUE"
)
_STALE_DEPENDENT_TABLES: Final[tuple[str, ...]] = (
    "missing_data_log",
    "casualty_reports",
    "troop_reports",
    "battle_commanders",
)
# Table names are fixed literals above, never input; values stay bound.
_DELETE_STALE_DEPENDENTS: Final[dict[str, Any]] = {
    table: text(f"DELETE FROM {table} WHERE side_id = ANY(CAST(:ids AS INT[]))")  # noqa: S608
    for table in _STALE_DEPENDENT_TABLES
}
_DELETE_STALE_SIDES = text("DELETE FROM battle_sides WHERE side_id = ANY(CAST(:ids AS INT[]))")

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
        self.stale_sides_removed = 0
        self.troop_reports = 0
        self.casualty_reports = 0
        self.missing_rows = 0
        self.failed_battles = 0
        # Not in as_dict: a set of ids is not a counter.
        self.written_battle_ids: set[int] = set()

    def as_dict(self) -> dict[str, int]:
        """Render the counts for a structlog call.

        Returns:
            A flat mapping of counter name to value.
        """
        return {
            "battles": self.battles,
            "sides": self.sides,
            "stale_sides_removed": self.stale_sides_removed,
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
        "source_slug": battle.slug or None,
        "wikidata_id": facts.wikidata_id,
        "wikipedia_url": facts.wikipedia_url,
        "date_start": date_start,
        "date_end": date_end,
        "date_precision": date_precision,
        "date_circa": bool(facts.date_circa) and date_start is not None,
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
    if battle.slug:
        existing = conn.execute(_FIND_BATTLE_BY_SLUG, {"slug": battle.slug}).fetchone()
    if existing is None and facts.wikidata_id:
        by_qid = conn.execute(_FIND_BATTLE_BY_QID, {"wikidata_id": facts.wikidata_id}).fetchone()
        if by_qid is not None:
            owner_slug = conn.execute(_BATTLE_SLUG, {"battle_id": int(by_qid[0])}).scalar_one()
            if owner_slug is None or owner_slug == battle.slug:
                existing = by_qid
            else:
                # A different article already owns this entity: adopting its
                # row would overwrite that battle and delete its sides.
                logger.warning(
                    "extract_wikidata_id_owned_by_other_slug",
                    slug=battle.slug,
                    owner_slug=owner_slug,
                    wikidata_id=facts.wikidata_id,
                    owner_battle_id=int(by_qid[0]),
                )
                params["wikidata_id"] = None
    if existing is None:
        existing = conn.execute(_FIND_LEGACY_BATTLE_BY_NAME, {"name": battle.name}).fetchone()

    if existing is None:
        row = conn.execute(_INSERT_BATTLE, params).fetchone()
        return int(row[0])

    battle_id = int(existing[0])
    if facts.wikidata_id:
        owner = conn.execute(_FIND_BATTLE_BY_QID, {"wikidata_id": facts.wikidata_id}).fetchone()
        if owner is not None and int(owner[0]) != battle_id:
            # Two articles resolving to one entity (a redirect, a merged
            # page): the first keeps the QID, the unique constraint would
            # otherwise reject this battle outright.
            logger.warning(
                "extract_wikidata_id_already_owned",
                slug=battle.slug,
                wikidata_id=facts.wikidata_id,
                owner_battle_id=int(owner[0]),
            )
            params["wikidata_id"] = None
    conn.execute(_UPDATE_BATTLE, {**params, "battle_id": battle_id})
    return battle_id


def _find_or_insert(conn: Any, find: Any, insert: Any, params: dict[str, Any]) -> int:
    """Return the id of the row ``find`` matches, inserting it when absent."""
    row = conn.execute(find, params).fetchone()
    if row is None:
        row = conn.execute(insert, params).fetchone()
    return int(row[0])


def _link_war_and_campaign(conn: Any, battle_id: int, battle: BattleExtraction) -> None:
    """Write the battle's war and campaign rows and point the battle at them.

    Args:
        conn: An open database connection.
        battle_id: The battle to link.
        battle: The merged battle, whose ``part_of`` names are classified.
    """
    chosen = classify_part_of(battle.facts.part_of)
    if chosen.war is None and chosen.campaign is None:
        return

    war_id = (
        _find_or_insert(conn, _FIND_WAR, _INSERT_WAR, {"name": chosen.war})
        if chosen.war is not None
        else None
    )
    campaign_id = (
        _find_or_insert(
            conn, _FIND_CAMPAIGN, _INSERT_CAMPAIGN, {"name": chosen.campaign, "war_id": war_id}
        )
        if chosen.campaign is not None
        else None
    )
    conn.execute(
        _LINK_BATTLE, {"battle_id": battle_id, "war_id": war_id, "campaign_id": campaign_id}
    )


def reserve_battle_id(conn: Any, slug: str, name: str) -> int | None:
    """Find or create the bare ``battles`` row for an article.

    The LLM pass runs before the merged record exists, but the batch file it
    writes carries ``battle_id`` in each request's metadata so that imported
    ``llm_calls`` rows are attributed. That needs an id up front. The row is
    completed later by :func:`write_battle`, which finds it by slug.

    Args:
        conn: An open database connection.
        slug: The article's crawl-stage filename stem.
        name: The battle's display name.

    Returns:
        The battle id, or None when the row could not be created. A failure
        costs the attribution only, so the caller carries on without it.
    """
    try:
        with conn.begin_nested():
            row = conn.execute(_FIND_BATTLE_BY_SLUG, {"slug": slug}).fetchone()
            if row is None:
                row = conn.execute(_FIND_LEGACY_BATTLE_BY_NAME, {"name": name}).fetchone()
                if row is not None:
                    conn.execute(
                        text("UPDATE battles SET source_slug = :slug WHERE battle_id = :id"),
                        {"slug": slug, "id": row[0]},
                    )
            if row is None:
                row = conn.execute(_RESERVE_BATTLE, {"name": name, "slug": slug}).fetchone()
            return int(row[0])
    except Exception as exc:
        logger.warning("extract_battle_reserve_failed", slug=slug, error=str(exc)[:200])
        return None


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


def _remove_stale_sides(
    conn: Any,
    battle_id: int,
    kept_side_ids: list[int],
    counts: ExtractWriteCounts,
    *,
    llm_pending: bool = False,
) -> None:
    """Delete the battle's sides that this run did not write, with their rows.

    Side labels change between extract versions, so a relabelled side is a new
    row and the old one lingers with its own commanders and reports. Nothing
    is removed when this run wrote no sides, so a failed parse cannot wipe a
    battle.

    Args:
        conn: An open database connection.
        battle_id: The battle just written.
        kept_side_ids: Ids of every side this run wrote for the battle.
        counts: Counters to accumulate into.
        llm_pending: Whether the record still has LLM passages deferred. Its
            sides then lack whatever only the LLM pass produces, so a stale
            side holding LLM-method rows is kept until that pass is answered.
            Stale sides without such rows (older deterministic runs) still go.
    """
    if not kept_side_ids:
        return

    stale = conn.execute(
        _FIND_STALE_SIDES, {"battle_id": battle_id, "keep": kept_side_ids}
    ).fetchall()
    if llm_pending and stale:
        held = {
            int(row[0])
            for row in conn.execute(
                _FIND_SIDES_WITH_LLM_ROWS,
                {"ids": [int(r[0]) for r in stale], "methods": list(_LLM_METHODS)},
            )
        }
        for row in stale:
            if int(row[0]) in held:
                logger.info(
                    "extract_stale_side_kept_pending_llm",
                    battle_id=battle_id,
                    side_id=int(row[0]),
                    label=str(row[1]),
                )
        stale = [row for row in stale if int(row[0]) not in held]
    if not stale:
        return

    ids = [int(row[0]) for row in stale]
    labels = [str(row[1]) for row in stale]

    # Counted before the delete: rows from other methods (manual corrections),
    # resolve's commanders and imputed missingness go too, because the side
    # they hang off is gone.
    non_extract = {
        table: int(conn.execute(query, {"ids": ids, "methods": list(_OWNED_METHODS)}).scalar_one())
        for table, query in _COUNT_FOREIGN_REPORTS.items()
    }
    non_extract["battle_commanders"] = int(
        conn.execute(_COUNT_FOREIGN_COMMANDERS, {"ids": ids}).scalar_one()
    )
    non_extract["missing_data_log"] = int(
        conn.execute(_COUNT_FOREIGN_MISSING, {"ids": ids}).scalar_one()
    )

    conn.execute(_DETACH_STALE_PARENTS, {"ids": ids})
    deleted = {
        table: conn.execute(_DELETE_STALE_DEPENDENTS[table], {"ids": ids}).rowcount
        for table in _STALE_DEPENDENT_TABLES
    }
    conn.execute(_DELETE_STALE_SIDES, {"ids": ids})

    counts.stale_sides_removed += len(ids)
    log = logger.warning if any(non_extract.values()) else logger.info
    log(
        "extract_stale_sides_removed",
        battle_id=battle_id,
        side_ids=ids,
        side_labels=labels,
        rows_deleted=deleted,
        non_extract_rows_deleted=non_extract,
    )


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
    _link_war_and_campaign(conn, battle_id, battle)
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

    if battle_id in tally.written_battle_ids:
        # Two records resolved to one row this run; each one's sides look
        # stale to the other, so cleanup would delete the first record's.
        logger.warning("extract_battle_written_twice_skipping_cleanup", battle_id=battle_id)
    else:
        _remove_stale_sides(
            conn,
            battle_id,
            sorted(set(side_ids.values())),
            tally,
            llm_pending=bool(battle.deferred_failures),
        )
    tally.written_battle_ids.add(battle_id)

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
