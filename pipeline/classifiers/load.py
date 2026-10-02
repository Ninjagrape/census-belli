"""
Database and filesystem reads for the classify stage.

Every pure module in :mod:`pipeline.classifiers` takes plain records and
returns plain records; this module is where those records come from. It
reads ``battle_commanders``/``battle_sides``/``battles`` for the role step,
``troop_reports`` and the raw crawled article for the battle-type step, and
``missing_data_log`` for the missingness step -- and it is the one place that
knows how a ``battles`` row (a database id and a ``wikipedia_url``) maps back
onto the file the crawl stage left on disk for it.

**The battle -> article mapping.** ``battles.wikipedia_url`` is the only
fact a database row carries forward from crawl that the crawl stage's own
filename rule can still be run on. ``pipeline.crawlers.wikipedia
.article_filename`` turns a URL into ``<sanitised-title>-<10-hex-digest>``,
the same computation extract's ``discover_battles`` uses to build its slug
index -- so recomputing it here needs no index and no directory scan beyond
checking, in preference order, which of the suffixes crawl or extract might
have left actually exists (``.wikitext`` preferred over rendered HTML, matching
``pipeline.stages.extract.RawBattle``'s own preference).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import structlog
from sqlalchemy import text

from pipeline.classifiers.missingness import IMPUTE_FIELDS, MissingnessInput
from pipeline.classifiers.records import CommanderRow, SideGroup
from pipeline.crawlers.wikipedia import article_filename

__all__ = [
    "ARTICLE_SUFFIXES",
    "HTML_SUFFIXES",
    "find_article_path",
    "find_unclassified_battle_type_log_ids",
    "load_battle_wikipedia_urls",
    "load_battles_needing_type",
    "load_missingness_inputs",
    "load_side_groups",
    "load_troop_evidence",
    "read_article_text",
]

logger = structlog.get_logger()

# Same preference order as pipeline.stages.extract._ARTICLE_SUFFIXES: a
# wikitext sibling parses better than rendered HTML, when one exists.
ARTICLE_SUFFIXES: Final[tuple[str, ...]] = (".wikitext", ".wiki", ".txt", ".html", ".htm")
HTML_SUFFIXES: Final[frozenset[str]] = frozenset({".html", ".htm"})

_ARTICLE_DIRNAME: Final[str] = "battles_html"

_WINNER_OUTCOMES: Final[frozenset[str]] = frozenset(
    {"decisive_victory", "victory", "pyrrhic_victory"}
)
_LOSER_OUTCOMES: Final[frozenset[str]] = frozenset({"defeat", "decisive_defeat"})


# ─── Role step ───────────────────────────────────────────────────────────────

_LOAD_SIDE_GROUPS = text(
    """
    SELECT
        bc.bc_id, bc.battle_id, bc.side_id, bc.general_id,
        g.canonical_name, bc.command_role, COALESCE(bc.role_evidence, ''),
        bs.side_label, b.name, b.year_astronomical,
        COALESCE(bc.attribution_method, 'equal')
    FROM battle_commanders bc
    JOIN generals g ON g.general_id = bc.general_id
    JOIN battle_sides bs ON bs.side_id = bc.side_id
    JOIN battles b ON b.battle_id = bc.battle_id
    ORDER BY bc.side_id, bc.bc_id
    """
)


def load_side_groups(conn: Any) -> list[SideGroup]:
    """Read every side and its commanders, ready for role classification.

    Args:
        conn: An open database connection.

    Returns:
        One :class:`SideGroup` per ``(battle_id, side_id)``, in ``side_id``
        order, each carrying its commanders in ``bc_id`` order.

        **Listing order.** The extract stage's infobox listing order is not
        persisted anywhere resolve or classify can read -- ``battle_commanders``
        carries no such column, and resolve's mention-to-row write does not
        preserve it either. ``bc_id`` order is the best surviving proxy: rows
        are written in the order resolve processed each identity, which for
        an infobox-derived mention tracks the order it was extracted in. This
        is a documented approximation, not a fact recovered from the data.

        **``apparent_role``.** Extract's own guess at a role
        (``CommanderRow.apparent_role``, a different vocabulary from
        ``command_role``, see ``pipeline.classifiers.records``) is likewise
        not persisted -- resolve maps it straight into ``command_role`` via
        ``to_schema_command_role`` and keeps no separate copy. Every row here
        therefore carries the dataclass default ``"unclear"``; it only ever
        affects a prompt's display text, never a classification decision.
    """
    rows = conn.execute(_LOAD_SIDE_GROUPS).fetchall()

    order_seen: dict[int, int] = {}
    sides: dict[int, dict[str, Any]] = {}
    side_order: list[int] = []

    for (
        bc_id,
        battle_id,
        side_id,
        general_id,
        name,
        command_role,
        role_evidence,
        side_label,
        battle_name,
        year_astronomical,
        attribution_method,
    ) in rows:
        side_id = int(side_id)
        if side_id not in sides:
            sides[side_id] = {
                "battle_id": int(battle_id),
                "side_label": str(side_label),
                "battle_name": str(battle_name),
                "year_astronomical": (
                    int(year_astronomical) if year_astronomical is not None else None
                ),
                "commanders": [],
            }
            side_order.append(side_id)

        listing_order = order_seen.get(side_id, 0)
        order_seen[side_id] = listing_order + 1
        sides[side_id]["commanders"].append(
            CommanderRow(
                bc_id=int(bc_id),
                battle_id=int(battle_id),
                side_id=side_id,
                general_id=int(general_id),
                name=str(name),
                command_role=str(command_role),
                role_evidence=str(role_evidence or ""),
                apparent_role="unclear",
                listing_order=listing_order,
                attribution_method=str(attribution_method),
            )
        )

    return [
        SideGroup(
            battle_id=sides[side_id]["battle_id"],
            side_id=side_id,
            side_label=sides[side_id]["side_label"],
            battle_name=sides[side_id]["battle_name"],
            year_astronomical=sides[side_id]["year_astronomical"],
            commanders=tuple(sides[side_id]["commanders"]),
        )
        for side_id in side_order
    ]


# ─── Battle-type step ────────────────────────────────────────────────────────

_LOAD_BATTLE_URLS = text("SELECT battle_id, wikipedia_url FROM battles")

_LOAD_BATTLES_NEEDING_TYPE = text(
    "SELECT battle_id, name, wikipedia_url FROM battles WHERE battle_type = 'unknown'"
)

_LOAD_TROOP_EVIDENCE = text(
    """
    SELECT tr.branch, COALESCE(tr.extracted_context, '')
    FROM troop_reports tr
    JOIN battle_sides bs ON bs.side_id = tr.side_id
    WHERE bs.battle_id = :battle_id
    """
)


def load_battle_wikipedia_urls(conn: Any) -> dict[int, str | None]:
    """Read every battle's stored Wikipedia URL in one query.

    Args:
        conn: An open database connection.

    Returns:
        Battle id to URL (or None), for :func:`find_article_path` to resolve
        into a raw article file without a query per battle.
    """
    return {int(bid): url for bid, url in conn.execute(_LOAD_BATTLE_URLS).fetchall()}


def load_battles_needing_type(conn: Any) -> list[dict[str, Any]]:
    """List battles whose ``battle_type`` is still 'unknown'.

    Args:
        conn: An open database connection.

    Returns:
        One mapping per battle with ``battle_id``, ``name`` and
        ``wikipedia_url``, the inputs :func:`pipeline.classifiers.battle_type
        .infer_battle_type` and this module's own file lookup need.
    """
    return [
        {"battle_id": int(bid), "name": str(name), "wikipedia_url": url}
        for bid, name, url in conn.execute(_LOAD_BATTLES_NEEDING_TYPE).fetchall()
    ]


def load_troop_evidence(conn: Any, battle_id: int) -> tuple[list[str], list[str]]:
    """Read a battle's troop-report branches and strength text.

    Args:
        conn: An open database connection.
        battle_id: The battle to read evidence for.

    Returns:
        ``(troop_branches, strength_strings)``, exactly the two sequences
        :func:`pipeline.classifiers.battle_type.infer_battle_type` wants for
        its naval/land signal. Reports from every side of the battle are
        pooled, since a single ``ships1`` field on either side is evidence
        the engagement had a naval component regardless of which side it was
        reported for.
    """
    rows = conn.execute(_LOAD_TROOP_EVIDENCE, {"battle_id": battle_id}).fetchall()
    branches = [str(branch) for branch, _context in rows]
    strengths = [str(context) for _branch, context in rows if context]
    return branches, strengths


def find_article_path(raw_root: Path, wikipedia_url: str | None) -> Path | None:
    """Map a battle's stored Wikipedia URL onto its raw article file.

    Args:
        raw_root: The ``data/raw`` directory (or a test fixture root standing
            in for it).
        wikipedia_url: ``battles.wikipedia_url``, or None/empty when the
            battle carries none -- a battle predating extract's write of the
            column, or one added by hand.

    Returns:
        The path crawl (or a fixture) left for this article, preferring a
        wikitext sibling over rendered HTML per :data:`ARTICLE_SUFFIXES`, or
        None when no URL is recorded or no matching file exists on disk.
    """
    if not wikipedia_url:
        return None

    stem = Path(article_filename(wikipedia_url)).stem
    directory = raw_root / _ARTICLE_DIRNAME
    for suffix in ARTICLE_SUFFIXES:
        candidate = directory / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def read_article_text(path: Path) -> str | None:
    """Read a raw article file, treating an unreadable one as absent.

    Args:
        path: A file :func:`find_article_path` returned.

    Returns:
        The file's text, or None on any read failure. A missing or
        unreadable article is a legitimate "no signal" case for
        ``infer_battle_type``, not a reason to stop the stage.
    """
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("classify_article_unreadable", path=str(path), error=str(exc))
        return None


# ─── Missingness step ────────────────────────────────────────────────────────

_FIND_BATTLE_TYPE_LOG_IDS = text(
    """
    SELECT log_id FROM missing_data_log
    WHERE battle_id = ANY(CAST(:battle_ids AS INT[]))
      AND field_name = 'battle_type'
      AND missingness_class = 'unclassified'
    """
)


def find_unclassified_battle_type_log_ids(conn: Any, battle_ids: set[int]) -> set[int]:
    """Find the still-'unclassified' ``battle_type`` row for each battle given.

    Used to keep the missingness pass from reconsidering a row the
    battle-type pass in the same run already decided, without depending on
    the battle-type write having happened first: both passes read the
    database's state as of the start of the run, so without this a battle
    whose type was just inferred would still look 'unclassified' to the
    missingness heuristics, which would then compute a decision for it that
    the later, guarded write silently discards -- correct in the end, but
    double the work and an inflated ``missingness_classified`` count.

    Args:
        conn: An open database connection.
        battle_ids: Battles whose ``battle_type`` was inferred as something
            other than 'unknown' this run.

    Returns:
        The matching ``missing_data_log.log_id`` values. Empty when
        ``battle_ids`` is empty, or none of them has such a row.
    """
    if not battle_ids:
        return set()
    rows = conn.execute(
        _FIND_BATTLE_TYPE_LOG_IDS, {"battle_ids": sorted(battle_ids)}
    ).fetchall()
    return {int(row[0]) for row in rows}


_LOAD_UNCLASSIFIED_ROWS = text(
    """
    SELECT mdl.log_id, mdl.field_name, mdl.missingness_class,
           mdl.battle_id, mdl.side_id, b.year_astronomical
    FROM missing_data_log mdl
    JOIN battles b ON b.battle_id = mdl.battle_id
    WHERE mdl.missingness_class = 'unclassified'
      AND mdl.field_name = ANY(CAST(:fields AS TEXT[]))
    """
)

_LOAD_TROOP_SIDES = text(
    """
    SELECT DISTINCT tr.side_id
    FROM troop_reports tr
    JOIN battle_sides bs ON bs.side_id = tr.side_id
    WHERE bs.battle_id = :battle_id
    """
)

_LOAD_SIDE_OUTCOMES = text(
    "SELECT side_id, outcome FROM battle_sides WHERE battle_id = :battle_id"
)


def _side_outcome_label(outcome: str | None) -> str | None:
    """Map an ``outcome_level`` enum value onto 'winner'/'loser'/None.

    Args:
        outcome: The raw ``battle_sides.outcome`` value.

    Returns:
        'winner' for a victory of any degree, 'loser' for a defeat of any
        degree, and None for 'indecisive', an unset outcome, or a value this
        stage does not recognise -- an indecisive result is not evidence
        either heuristic in ``classify_missingness`` is built to use.
    """
    if outcome in _WINNER_OUTCOMES:
        return "winner"
    if outcome in _LOSER_OUTCOMES:
        return "loser"
    return None


def load_missingness_inputs(conn: Any) -> list[MissingnessInput]:
    """Build a :class:`MissingnessInput` for every row classify should look at.

    Args:
        conn: An open database connection.

    Returns:
        One input per ``missing_data_log`` row that is still 'unclassified'
        and names a field ``pipeline.classifiers.missingness.IMPUTE_FIELDS``
        covers -- rows resolve owns (``commander_general_id``) or any other
        stage's field are never read here, matching
        ``classify_missingness``'s own scope guard.

        Troop presence is read from ``troop_reports`` rather than
        ``battle_sides.est_troops_total``: the reconcile stage, which writes
        the latter, has not necessarily run to completion by the time
        classify does, and a raw report's mere existence is the same signal
        the extract stage's own ``missing_fields()`` uses to decide whether
        ``troop_total`` was missing in the first place (see
        ``pipeline.extractors.merger.missing_fields``).
    """
    rows = conn.execute(_LOAD_UNCLASSIFIED_ROWS, {"fields": sorted(IMPUTE_FIELDS)}).fetchall()
    if not rows:
        return []

    battle_ids = {int(row[3]) for row in rows}
    troop_sides_by_battle: dict[int, set[int]] = {}
    outcome_by_side: dict[int, str | None] = {}
    for battle_id in battle_ids:
        troop_sides_by_battle[battle_id] = {
            int(side_id)
            for (side_id,) in conn.execute(_LOAD_TROOP_SIDES, {"battle_id": battle_id}).fetchall()
        }
        for side_id, outcome in conn.execute(
            _LOAD_SIDE_OUTCOMES, {"battle_id": battle_id}
        ).fetchall():
            outcome_by_side[int(side_id)] = outcome

    inputs: list[MissingnessInput] = []
    for log_id, field_name, current_class, battle_id, side_id, year in rows:
        battle_id = int(battle_id)
        side_id = int(side_id) if side_id is not None else None
        troop_sides = troop_sides_by_battle.get(battle_id, set())

        any_side_has_troops = bool(troop_sides)
        this_side_has_troops = side_id is not None and side_id in troop_sides
        other_side_has_troops = (
            any(sid != side_id for sid in troop_sides)
            if side_id is not None
            else any_side_has_troops
        )
        side_outcome = (
            _side_outcome_label(outcome_by_side.get(side_id)) if side_id is not None else None
        )

        inputs.append(
            MissingnessInput(
                log_id=int(log_id),
                field_name=str(field_name),
                current_class=str(current_class),
                year_astronomical=int(year) if year is not None else None,
                this_side_has_troops=this_side_has_troops,
                other_side_has_troops=other_side_has_troops,
                any_side_has_troops=any_side_has_troops,
                side_outcome=side_outcome,
            )
        )

    return inputs
