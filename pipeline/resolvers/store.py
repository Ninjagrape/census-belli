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
    "UNRESOLVED_FIELD",
    "load_battle_index",
    "load_side_index",
    "log_unresolved",
    "write_identity",
]

logger = structlog.get_logger()

# A bare Wikidata item id. Never a person's name, and the last thing that
# should ever reach generals.canonical_name -- see _publishable_name.
_QID_SHAPE: Final[re.Pattern[str]] = re.compile(r"Q\d+")

# The one ``missing_data_log.field_name`` this stage owns. Resolve clears
# every row carrying it before writing, so nothing else may use it.
UNRESOLVED_FIELD: Final[str] = "commander_general_id"

_LOAD_BATTLES = text("SELECT battle_id, name FROM battles")
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
        canonical_name, wikidata_id, wikipedia_url, nationality, years_active, notes
    ) VALUES (
        :canonical_name, :wikidata_id, :wikipedia_url, :nationality,
        CASE WHEN CAST(:year_lo AS INT) IS NULL THEN NULL
             ELSE int4range(CAST(:year_lo AS INT), CAST(:year_hi AS INT)) END,
        :notes
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
    "UPDATE general_aliases SET is_primary = FALSE "
    "WHERE general_id = :general_id AND is_primary"
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

_CLEAR_UNRESOLVED = text(
    "DELETE FROM missing_data_log WHERE field_name = :field_name AND was_imputed = FALSE"
)
_INSERT_UNRESOLVED = text(
    """
    INSERT INTO missing_data_log (battle_id, side_id, field_name, missingness_class, notes)
    VALUES (:battle_id, :side_id, :field_name, CAST(:missingness AS missingness_class), :notes)
    """
)


def load_battle_index(conn: Any) -> dict[str, int]:
    """Map battle names onto their ids.

    One query rather than a lookup per mention: a corpus produces far more
    commander mentions than battles, and the index is read once per row.

    Args:
        conn: An open database connection.

    Returns:
        Battle name to id. A duplicated name keeps the lowest id, matching
        the ``LIMIT 1`` the extract writer resolves such a collision with.
    """
    index: dict[str, int] = {}
    for battle_id, name in conn.execute(_LOAD_BATTLES).fetchall():
        key = str(name)
        if key not in index or int(battle_id) < index[key]:
            index[key] = int(battle_id)
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
                conn.execute(
                    _SET_PRIMARY_ALIAS, {"general_id": general_id, "alias_name": alias}
                )
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
) -> int | None:
    """Persist one resolved person and every battle they commanded in.

    Args:
        conn: An open database connection.
        identity: The resolved person, with all their mentions.
        battles: Battle name to id, from :func:`load_battle_index`.
        sides: Side id by (battle id, label), from :func:`load_side_index`.
        counts: Counters to accumulate into.
        source_cache: Per-run memo of source ids.

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
            battle_id = battles.get(mention.battle_name)
            if battle_id is None:
                counts.battles_not_in_db += 1
                logger.debug("resolve_battle_not_in_database", battle=mention.battle_name)
                continue
            side_id = sides.get((battle_id, mention.side_label))
            if side_id is None:
                counts.sides_not_in_db += 1
                logger.debug(
                    "resolve_side_not_in_database",
                    battle=mention.battle_name,
                    side=mention.side_label,
                )
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


def log_unresolved(
    conn: Any,
    mentions: list[Mention],
    battles: dict[str, int],
    sides: dict[tuple[int, str], int],
    counts: ResolveCounts,
    *,
    clear_first: bool = True,
) -> None:
    """Record the mentions this run could not turn into a general.

    Args:
        conn: An open database connection.
        mentions: Every mention belonging to an unresolved group.
        battles: Battle name to id.
        sides: Side id by (battle id, label).
        counts: Counters to accumulate into.
        clear_first: Whether to delete this stage's previous rows. True on a
            normal run, so the log reflects this run rather than the union of
            every run.

    Note:
        The extract stage clears *all* un-imputed ``missing_data_log`` rows
        for a battle it rewrites, so re-running extract after resolve drops
        these. Running resolve again restores them, and the pipeline runs the
        stages in that order anyway.
    """
    if clear_first:
        conn.execute(_CLEAR_UNRESOLVED, {"field_name": UNRESOLVED_FIELD})

    for mention in mentions:
        battle_id = battles.get(mention.battle_name)
        if battle_id is None:
            counts.battles_not_in_db += 1
            continue

        conn.execute(
            _INSERT_UNRESOLVED,
            {
                "battle_id": battle_id,
                "side_id": sides.get((battle_id, mention.side_label)),
                "field_name": UNRESOLVED_FIELD,
                # Not random: a mention goes unresolved because of what it
                # says and how obscure its subject is, and obscurity relates
                # to the commander's own record.
                "missingness": "mnar",
                "notes": f"unresolved commander mention: {mention.name[:200]}",
            },
        )
        counts.missing_rows += 1
