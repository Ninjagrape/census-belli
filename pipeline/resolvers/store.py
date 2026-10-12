"""
Writing resolved identities into the database.

This is the stage that finally puts rows in ``battle_commanders``, which the
extract stage deliberately left empty: its ``general_id`` is NOT NULL, and
before resolve there is no canonical person to point it at.

Three things about the writes are load-bearing.

**A re-run must converge, not accumulate.** ``battle_commanders`` carries a
UNIQUE (battle_id, side_id, general_id), so the insert upserts on it.

**A re-run must not undo the classify stage.** classify refines
``command_role``, ``hierarchy_rank``, ``reports_to_bc_id`` and
``attribution_weight`` on these same rows. Deleting and reinserting -- which
is what the extract writer does for troop reports -- would silently discard
that work, so the conflict clause updates only the columns resolve owns, and
leaves a ``command_role`` that has already moved off ``unknown`` alone.

**An unresolved mention is recorded, not dropped.** It goes to
``missing_data_log`` under ``commander_general_id``. That is what gives the
``resolution_rate`` gate a denominator: without it the gate would divide
written rows by written rows and report 100% however much the stage lost.
"""

from __future__ import annotations

import re
from typing import Any, Final

import structlog
from sqlalchemy import text

from pipeline.extractors.records import Provenance, to_schema_command_role
from pipeline.extractors.store import source_id
from pipeline.resolvers.records import Identity, Mention, ResolveCounts

__all__ = [
    "NON_PERSON_FIELD",
    "UNRESOLVED_FIELD",
    "load_battle_index",
    "load_side_index",
    "log_unresolved",
    "year_to_date_literal",
    "write_identity",
]

logger = structlog.get_logger()

# A bare Wikidata item id. Never a person's name, and the last thing that
# should ever reach generals.canonical_name -- see _publishable_name.
_QID_SHAPE: Final[re.Pattern[str]] = re.compile(r"Q\d+")

# The one ``missing_data_log.field_name`` this stage owns. Resolve clears
# every row carrying it before writing, so nothing else may use it.
UNRESOLVED_FIELD: Final[str] = "commander_general_id"

# Mentions that name an organisation or a place rather than a person. Logged so
# the filter is auditable, but under a field of their own: they are not
# commander mentions the stage failed on, so counting them in the
# ``resolution_rate`` denominator would charge the stage for refusing to invent
# a general. Resolve clears this field too before writing.
NON_PERSON_FIELD: Final[str] = "commander_non_person"

_LOAD_BATTLES = text("SELECT battle_id, name, source_slug FROM battles")
_LOAD_SIDES = text("SELECT side_id, battle_id, side_label FROM battle_sides")

_FIND_GENERAL_BY_QID = text("SELECT general_id FROM generals WHERE wikidata_id = :wikidata_id")
# Only rows with no Q-id are matched by name: a name collision must never
# rewrite an entity Wikidata has already identified.
_FIND_GENERAL_BY_NAME = text(
    "SELECT general_id FROM generals WHERE canonical_name = :canonical_name "
    "AND wikidata_id IS NULL LIMIT 1"
)

_INSERT_GENERAL = text(
    """
    INSERT INTO generals (
        canonical_name, wikidata_id, wikipedia_url, nationality, years_active, notes,
        born, died, era
    ) VALUES (
        :canonical_name, :wikidata_id, :wikipedia_url, :nationality,
        CASE WHEN CAST(:year_lo AS INT) IS NULL THEN NULL
             ELSE int4range(CAST(:year_lo AS INT), CAST(:year_hi AS INT)) END,
        :notes,
        CAST(:born AS DATE), CAST(:died AS DATE), :era
    )
    RETURNING general_id
    """
)

_UPDATE_GENERAL = text(
    """
    UPDATE generals SET
        canonical_name = :canonical_name,
        wikidata_id    = COALESCE(:wikidata_id, wikidata_id),
        wikipedia_url  = COALESCE(:wikipedia_url, wikipedia_url),
        nationality    = COALESCE(:nationality, nationality),
        years_active   = CASE
            WHEN CAST(:year_lo AS INT) IS NULL THEN years_active
            ELSE int4range(
                LEAST(COALESCE(lower(years_active), CAST(:year_lo AS INT)),
                      CAST(:year_lo AS INT)),
                GREATEST(COALESCE(upper(years_active), CAST(:year_hi AS INT)),
                         CAST(:year_hi AS INT))
            )
        END,
        -- COALESCE, like wikidata_id: a re-run that linked nothing new must not
        -- blank a biography an earlier run took from Wikidata.
        born       = COALESCE(CAST(:born AS DATE), born),
        died       = COALESCE(CAST(:died AS DATE), died),
        era        = COALESCE(:era, era),
        notes      = :notes,
        updated_at = now()
    WHERE general_id = :general_id
    """
)

_FIND_ALIASES = text("SELECT alias_name FROM general_aliases WHERE general_id = :general_id")

# generals.canonical_name is rewritten unconditionally on a re-run, so a
# general can be renamed -- which is exactly what the mul fix does to anyone
# previously stored under a bare Q-id. _write_aliases only ever *inserts*, and
# general_aliases has no constraint on is_primary (config/schema.sql), so the
# old primary row would survive beside the new one and the general would have
# two. Clear the flag before writing rather than discover it in the data.
_CLEAR_PRIMARY_ALIAS = text(
    "UPDATE general_aliases SET is_primary = FALSE WHERE general_id = :general_id AND is_primary"
)
_INSERT_ALIAS = text(
    "INSERT INTO general_aliases (general_id, alias_name, source_id, is_primary) "
    "VALUES (:general_id, :alias_name, :source_id, :is_primary)"
)

# A renamed general's new canonical name is often already on file as an
# ordinary alias, so promoting the existing row is the common path, not the
# rare one. Without this the general would end a re-run with no primary at all.
_SET_PRIMARY_ALIAS = text(
    "UPDATE general_aliases SET is_primary = TRUE "
    "WHERE general_id = :general_id AND alias_name = :alias_name"
)

_UPSERT_COMMANDER = text(
    """
    INSERT INTO battle_commanders (
        battle_id, side_id, general_id, command_role, source_id,
        extraction_method, confidence, role_evidence, attribution_method
    ) VALUES (
        :battle_id, :side_id, :general_id, CAST(:command_role AS command_role), :source_id,
        CAST(:extraction_method AS extraction_method), :confidence, :role_evidence, 'equal'
    )
    ON CONFLICT (battle_id, side_id, general_id) DO UPDATE SET
        -- classify refines command_role in place, so a resolve re-run only
        -- fills it where nothing has decided it yet.
        command_role = CASE
            WHEN battle_commanders.command_role = 'unknown' THEN EXCLUDED.command_role
            ELSE battle_commanders.command_role
        END,
        source_id         = EXCLUDED.source_id,
        extraction_method = EXCLUDED.extraction_method,
        confidence        = EXCLUDED.confidence,
        role_evidence     = COALESCE(
            NULLIF(EXCLUDED.role_evidence, ''), battle_commanders.role_evidence
        )
    """
)

_CLEAR_LOGGED = text(
    "DELETE FROM missing_data_log WHERE field_name = :field_name AND was_imputed = FALSE"
)
_INSERT_UNRESOLVED = text(
    """
    INSERT INTO missing_data_log (battle_id, side_id, field_name, missingness_class, notes)
    VALUES (:battle_id, :side_id, :field_name, CAST(:missingness AS missingness_class), :notes)
    """
)


def load_battle_index(conn: Any) -> dict[str, int]:
    """Map battle slugs, and unambiguous battle names, onto their ids.

    One query rather than a lookup per mention: a corpus produces far more
    commander mentions than battles, and the index is read once per row.

    Args:
        conn: An open database connection.

    Returns:
        Every battle's ``source_slug`` to its id, which is unique per
        article. A display name is added as a second key only when exactly
        one battle carries it, because two articles can share a name (two
        battles called "Battle of Luetzen") and picking one would attach a
        commander to the wrong battle. A caller holding a mention should
        try its ``battle_slug`` first; the name key serves rows that
        predate the slug column.
    """
    index: dict[str, int] = {}
    by_name: dict[str, list[int]] = {}
    for battle_id, name, slug in conn.execute(_LOAD_BATTLES).fetchall():
        by_name.setdefault(str(name), []).append(int(battle_id))
        if slug:
            index[str(slug)] = int(battle_id)
    for name, ids in by_name.items():
        if len(ids) == 1:
            index.setdefault(name, ids[0])
    return index


def load_side_index(conn: Any) -> dict[tuple[int, str], int]:
    """Map each (battle id, side label) onto its side id.

    Args:
        conn: An open database connection.

    Returns:
        Side ids keyed by battle and label.
    """
    return {
        (int(battle_id), str(side_label)): int(side_id)
        for side_id, battle_id, side_label in conn.execute(_LOAD_SIDES).fetchall()
    }


def _years_active(identity: Identity) -> tuple[int | None, int | None]:
    """Bound a general's active years by the battles they appear in.

    Args:
        identity: The resolved person.

    Returns:
        The inclusive lower bound and the **exclusive** upper bound
        PostgreSQL's ``int4range`` wants, or (None, None) when no battle of
        theirs is dated. These are years active *in this corpus*, not a
        biography: a commander is bounded by the battles that were crawled.
    """
    years = identity.years
    if not years:
        return None, None
    return min(years), max(years) + 1


def year_to_date_literal(year: int | None) -> str | None:
    """Render an astronomical year as a PostgreSQL date literal.

    ``generals.born`` and ``died`` are DATEs, and Wikidata candidates reach this
    stage as years only, so the date written is 1 January of that year. The
    precision is the year's: nothing reads the month, and the generated
    ``born_year_astronomical`` / ``died_year_astronomical`` columns are the
    ones to use, because ``datetime.date`` cannot hold a BC value.

    Args:
        year: An astronomical year (1 BC is 0), or None.

    Returns:
        ``"1769-01-01"`` or, for 44 BC (astronomical -43), ``"0044-01-01 BC"``.
        None when the year is unknown.
    """
    if year is None:
        return None
    if year >= 1:
        return f"{year:04d}-01-01"
    return f"{1 - year:04d}-01-01 BC"


def _general_params(identity: Identity) -> dict[str, Any]:
    """Build the parameters shared by the insert and the update.

    Args:
        identity: The resolved person.

    Returns:
        Parameters for ``generals``.
    """
    year_lo, year_hi = _years_active(identity)
    return {
        "canonical_name": _publishable_name(identity),
        "wikidata_id": identity.qid,
        "wikipedia_url": identity.wikipedia_url or None,
        "nationality": identity.nationality or None,
        "year_lo": year_lo,
        "year_hi": year_hi,
        "born": year_to_date_literal(identity.birth_year),
        "died": year_to_date_literal(identity.death_year),
        "era": identity.era,
        "notes": f"resolved by {identity.method} (confidence {identity.confidence:.2f})",
    }


def _publishable_name(identity: Identity) -> str:
    """The name to publish for an identity, refusing a bare Wikidata id.

    The candidate lookup already replaces a Q-id-shaped label
    (:func:`pipeline.resolvers.candidates._preferred_name`), so reaching here
    means something upstream produced one by a route nobody predicted -- which
    is how the defect arrived the first time. This is the last boundary before
    the name becomes published data, so it is checked again rather than
    trusted.

    Args:
        identity: The resolved person.

    Returns:
        The canonical name, or the group's display name when the canonical
        name is a bare item id.
    """
    name = (identity.canonical_name or "").strip()
    if not _QID_SHAPE.fullmatch(name):
        return identity.canonical_name

    # The surface forms the corpus actually saw are the honest fallback: they
    # are what a source wrote, rather than anything derived from Wikidata, so
    # they cannot carry the same defect.
    candidates: list[str] = []
    for group in identity.groups:
        candidates.append(group.display_name)
        candidates.extend(group.surface_forms)
    candidates.extend(identity.aliases)
    fallback = next(
        (c.strip() for c in candidates if c and c.strip() and not _QID_SHAPE.fullmatch(c.strip())),
        "",
    )

    logger.error(
        "general_canonical_name_was_a_qid",
        canonical_name=name,
        qid=identity.qid,
        used=fallback or name,
    )
    return fallback or name


def _upsert_general(conn: Any, identity: Identity) -> int:
    """Insert or update the ``generals`` row for one identity.

    Args:
        conn: An open database connection.
        identity: The resolved person.

    Returns:
        The general id.
    """
    params = _general_params(identity)

    existing = None
    if identity.qid:
        existing = conn.execute(_FIND_GENERAL_BY_QID, {"wikidata_id": identity.qid}).fetchone()
    if existing is None:
        existing = conn.execute(
            _FIND_GENERAL_BY_NAME, {"canonical_name": identity.canonical_name}
        ).fetchone()

    if existing is None:
        row = conn.execute(_INSERT_GENERAL, params).fetchone()
        return int(row[0])

    general_id = int(existing[0])
    conn.execute(_UPDATE_GENERAL, {**params, "general_id": general_id})
    return general_id


def _write_aliases(conn: Any, identity: Identity, general_id: int, source: int | None) -> int:
    """Record every surface form this person was published under.

    ``general_aliases`` has no unique constraint, so existing names are read
    back and only new ones inserted; a re-run therefore adds nothing.

    Args:
        conn: An open database connection.
        identity: The resolved person.
        general_id: Their row id.
        source: A ``sources`` id to attribute the aliases to, if any.

    Returns:
        How many alias rows were written.
    """
    known = {str(row[0]) for row in conn.execute(_FIND_ALIASES, {"general_id": general_id})}
    primary = _publishable_name(identity)

    # Exactly one primary, whatever the row already said. A re-run can rename a
    # general, and inserting alone would leave the old primary standing.
    conn.execute(_CLEAR_PRIMARY_ALIAS, {"general_id": general_id})

    written = 0
    for alias in [primary, *identity.aliases]:
        if not alias:
            continue
        if alias in known:
            if alias == primary:
                conn.execute(_SET_PRIMARY_ALIAS, {"general_id": general_id, "alias_name": alias})
            continue
        conn.execute(
            _INSERT_ALIAS,
            {
                "general_id": general_id,
                "alias_name": alias,
                "source_id": source,
                "is_primary": alias == primary,
            },
        )
        known.add(alias)
        written += 1

    return written


def _provenance(mention: Mention) -> Provenance:
    """Rebuild the provenance the extract stage recorded for a mention.

    Args:
        mention: The commander mention.

    Returns:
        The provenance, so the ``sources`` row this claim hangs off is the
        same one the troop and casualty reports from that document use.
    """
    return Provenance(
        source_type=mention.source_type,
        extraction_method=mention.extraction_method,
        source_ref=mention.source_ref,
        source_title=mention.source_title,
    )


def write_identity(
    conn: Any,
    identity: Identity,
    battles: dict[str, int],
    sides: dict[tuple[int, str], int],
    counts: ResolveCounts,
    source_cache: dict[tuple[str, str], int],
    unplaced: list[Mention] | None = None,
) -> int | None:
    """Persist one resolved person and every battle they commanded in.

    Args:
        conn: An open database connection.
        identity: The resolved person, with all their mentions.
        battles: Battle slug and unambiguous name to id, from :func:`load_battle_index`.
        sides: Side id by (battle id, label), from :func:`load_side_index`.
        counts: Counters to accumulate into.
        source_cache: Per-run memo of source ids.
        unplaced: Receives every mention whose battle or side is not in the
            database, so the caller can log it. Without this the mention was
            counted and then forgotten, and the ``resolution_rate`` gate never
            saw it.

    Returns:
        The general id, or None when the identity had no mention that could
        be attached to a battle and side actually present in the database.

    Note:
        A mention whose battle or side is missing from the database is
        counted and logged, not written. It means extract's database write
        did not run, or ran against a different corpus, and inventing the
        battle here would produce a battle with commanders and no troops.
    """
    placements: list[tuple[Mention, int, int, float]] = []
    for group in identity.groups:
        for mention in group.mentions:
            # Slug first: it is unique per article, whereas two battles can share a name.
            battle_id = battles.get(mention.battle_slug) or battles.get(mention.battle_name)
            if battle_id is None:
                counts.battles_not_in_db += 1
                logger.debug("resolve_battle_not_in_database", battle=mention.battle_name)
                if unplaced is not None:
                    unplaced.append(mention)
                continue
            side_id = sides.get((battle_id, mention.side_label))
            if side_id is None:
                counts.sides_not_in_db += 1
                logger.debug(
                    "resolve_side_not_in_database",
                    battle=mention.battle_name,
                    side=mention.side_label,
                )
                if unplaced is not None:
                    unplaced.append(mention)
                continue
            placements.append((mention, battle_id, side_id, group.confidence))

    if not placements:
        return None

    general_id = _upsert_general(conn, identity)
    identity.general_id = general_id
    counts.generals_written += 1

    first_source = source_id(conn, _provenance(placements[0][0]), source_cache)
    counts.aliases_written += _write_aliases(conn, identity, general_id, first_source)

    # One row per battle side, not per mention: two sources naming the same
    # commander on the same side describe one command, and the UNIQUE
    # constraint says so. The mention with the fullest role evidence wins,
    # since that is the text classify will read.
    best: dict[tuple[int, int], tuple[Mention, float]] = {}
    for mention, battle_id, side_id, confidence in placements:
        key = (battle_id, side_id)
        current = best.get(key)
        if current is None or len(mention.role_evidence) > len(current[0].role_evidence):
            best[key] = (mention, confidence)

    for (battle_id, side_id), (mention, confidence) in best.items():
        conn.execute(
            _UPSERT_COMMANDER,
            {
                "battle_id": battle_id,
                "side_id": side_id,
                "general_id": general_id,
                "command_role": to_schema_command_role(mention.apparent_role),
                "source_id": source_id(conn, _provenance(mention), source_cache),
                "extraction_method": mention.extraction_method,
                "confidence": confidence,
                "role_evidence": mention.role_evidence[:2000],
            },
        )
        counts.commanders_written += 1

    return general_id


def _log_mentions(
    conn: Any,
    mentions: list[Mention],
    battles: dict[str, int],
    sides: dict[tuple[int, str], int],
    counts: ResolveCounts,
    *,
    field_name: str,
    missingness: str,
    note: str,
) -> list[Mention]:
    """Insert one ``missing_data_log`` row per mention that has a battle.

    Args:
        conn: An open database connection.
        mentions: The mentions to record.
        battles: Battle name to id.
        sides: Side id by (battle id, label).
        counts: Counters to accumulate into.
        field_name: The ``missing_data_log.field_name`` to write under.
        missingness: A ``missingness_class`` member.
        note: Text put in front of the commander's name in ``notes``.

    Returns:
        The mentions whose battle is not in the database. ``battle_id`` is
        NOT NULL with a foreign key, so there is nowhere in that table to put
        them; the caller writes them to a file instead.
    """
    homeless: list[Mention] = []
    for mention in mentions:
        battle_id = battles.get(mention.battle_name)
        if battle_id is None:
            homeless.append(mention)
            continue

        conn.execute(
            _INSERT_UNRESOLVED,
            {
                "battle_id": battle_id,
                "side_id": sides.get((battle_id, mention.side_label)),
                "field_name": field_name,
                "missingness": missingness,
                "notes": f"{note}: {mention.name[:200]}",
            },
        )
        counts.missing_rows += 1
    return homeless


def log_unresolved(
    conn: Any,
    mentions: list[Mention],
    battles: dict[str, int],
    sides: dict[tuple[int, str], int],
    counts: ResolveCounts,
    *,
    clear_first: bool = True,
    unplaced: list[Mention] | None = None,
    non_person: list[Mention] | None = None,
) -> list[Mention]:
    """Record every commander mention this run did not turn into a row.

    Three kinds, all of which the ``resolution_rate`` gate has to see or it
    reads high:

    - ``mentions``: groups the matcher left unresolved (``UNRESOLVED_FIELD``,
      MNAR: they go unresolved because of how obscure the person is).
    - ``unplaced``: mentions that did resolve to a general but whose side is
      not in the database, so no ``battle_commanders`` row could be written
      (``UNRESOLVED_FIELD``, unclassified: a pipeline fault, not obscurity).
    - ``non_person``: mentions naming an organisation or place
      (``NON_PERSON_FIELD``), logged for audit and deliberately outside the
      gate's denominator.

    Args:
        conn: An open database connection.
        mentions: Every mention belonging to an unresolved group.
        battles: Battle name to id.
        sides: Side id by (battle id, label).
        counts: Counters to accumulate into.
        clear_first: Whether to delete this stage's previous rows. True on a
            normal run, so the log reflects this run rather than the union of
            every run.
        unplaced: Mentions of resolved people that could not be attached.
        non_person: Mentions the non-person filter refused.

    Returns:
        The mentions that could not be logged because their battle is not in
        the database, for the caller to write to a file.

    Note:
        The extract stage clears *all* un-imputed ``missing_data_log`` rows
        for a battle it rewrites (``_CLEAR_MISSING`` in
        :mod:`pipeline.extractors.store` has no field filter), so re-running
        extract after resolve drops these rows while leaving
        ``battle_commanders`` intact. The gate then divides by a denominator
        that has quietly lost its misses. Running resolve again restores
        them.
    """
    if clear_first:
        conn.execute(_CLEAR_LOGGED, {"field_name": UNRESOLVED_FIELD})
        conn.execute(_CLEAR_LOGGED, {"field_name": NON_PERSON_FIELD})

    homeless: list[Mention] = []
    for mention in mentions:
        if mention.battle_name not in battles:
            counts.battles_not_in_db += 1

    homeless.extend(
        _log_mentions(
            conn,
            mentions,
            battles,
            sides,
            counts,
            field_name=UNRESOLVED_FIELD,
            missingness="mnar",
            note="unresolved commander mention",
        )
    )
    homeless.extend(
        _log_mentions(
            conn,
            unplaced or [],
            battles,
            sides,
            counts,
            field_name=UNRESOLVED_FIELD,
            missingness="unclassified",
            note="resolved commander mention with no battle or side row",
        )
    )
    _log_mentions(
        conn,
        non_person or [],
        battles,
        sides,
        counts,
        field_name=NON_PERSON_FIELD,
        missingness="unclassified",
        note="non-person commander mention",
    )
    counts.unplaced_rows += len(homeless)
    return homeless
