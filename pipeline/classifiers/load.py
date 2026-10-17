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

import re
from pathlib import Path
from typing import Any, Final

import structlog
from sqlalchemy import text

from pipeline.classifiers.defender import SideLabel
from pipeline.classifiers.missingness import IMPUTE_FIELDS, MissingnessInput
from pipeline.classifiers.records import CommanderRow, MissingnessContext, SideGroup
from pipeline.crawlers.wikipedia import article_filename

__all__ = [
    "ARTICLE_SUFFIXES",
    "HTML_SUFFIXES",
    "find_article_path",
    "load_battle_source_slugs",
    "find_populated_unclassified_log_ids",
    "find_unclassified_battle_type_log_ids",
    "load_battle_wikipedia_urls",
    "load_battles_needing_terrain",
    "load_battles_needing_type",
    "load_missingness_contexts",
    "load_missingness_inputs",
    "load_side_groups",
    "load_siege_sides",
    "load_troop_evidence",
    "read_article_text",
]

logger = structlog.get_logger()

# Same preference order as pipeline.stages.extract._ARTICLE_SUFFIXES: a
# wikitext sibling parses better than rendered HTML, when one exists.
ARTICLE_SUFFIXES: Final[tuple[str, ...]] = (".wikitext", ".wiki", ".txt", ".html", ".htm")
HTML_SUFFIXES: Final[frozenset[str]] = frozenset({".html", ".htm"})

_ARTICLE_DIRNAME: Final[str] = "battles_html"
_DIGEST_SUFFIX_RE: Final = re.compile(r"-[0-9a-f]{10}$")

_WINNER_OUTCOMES: Final[frozenset[str]] = frozenset(
    {"decisive_victory", "victory", "pyrrhic_victory"}
)
_LOSER_OUTCOMES: Final[frozenset[str]] = frozenset({"defeat", "decisive_defeat"})


# ─── Role step ───────────────────────────────────────────────────────────────

_LOAD_SIDE_GROUPS = text(
    """
    SELECT
        bc.bc_id, bc.battle_id, bc.side_id, bc.general_id,
        g.canonical_name, g.wikidata_id, bc.command_role, COALESCE(bc.role_evidence, ''),
        bs.side_label, b.name, b.year_astronomical,
        COALESCE(bc.attribution_method, 'equal'),
        bc.infobox_order,
        (SELECT COUNT(*) FROM missing_data_log mdl
         WHERE mdl.side_id = bc.side_id AND mdl.field_name = 'commander_general_id')
    FROM battle_commanders bc
    JOIN generals g ON g.general_id = bc.general_id
    JOIN battle_sides bs ON bs.side_id = bc.side_id
    JOIN battles b ON b.battle_id = bc.battle_id
    ORDER BY bc.side_id, bc.infobox_order NULLS LAST, bc.bc_id
    """
)


def load_side_groups(conn: Any) -> list[SideGroup]:
    """Read every side and its commanders, ready for role classification.

    Args:
        conn: An open database connection.

    Returns:
        One :class:`SideGroup` per ``(battle_id, side_id)``, in ``side_id``
        order, each carrying its commanders in infobox order.

        **Listing order.** Commanders are sorted by ``battle_commanders.infobox_order``
        (the infobox position extract and resolve record), rows with no position
        after those that have one, ``bc_id`` breaking ties. ``bc_id`` order alone
        is the old approximation (resolve's insert order) and still applies to
        rows that carry no infobox position; ``CommanderRow.infobox_order`` stays
        None for them so the prompt never presents a guess as the infobox order.

        **Unresolved mentions.** ``SideGroup.unresolved_count`` is the number of
        ``commander_general_id`` rows in ``missing_data_log`` for the side: people
        the infobox listed whom resolve could not identify.

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
        wikidata_id,
        command_role,
        role_evidence,
        side_label,
        battle_name,
        year_astronomical,
        attribution_method,
        infobox_order,
        unresolved_count,
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
                "unresolved_count": int(unresolved_count or 0),
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
                wikidata_id=str(wikidata_id) if wikidata_id else None,
                command_role=str(command_role),
                role_evidence=str(role_evidence or ""),
                apparent_role="unclear",
                listing_order=listing_order,
                attribution_method=str(attribution_method),
                infobox_order=int(infobox_order) if infobox_order is not None else None,
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
            unresolved_count=sides[side_id]["unresolved_count"],
        )
        for side_id in side_order
    ]


# ─── Battle-type step ────────────────────────────────────────────────────────

_LOAD_BATTLE_URLS = text("SELECT battle_id, wikipedia_url FROM battles")

_LOAD_BATTLE_SLUGS = text("SELECT battle_id, source_slug FROM battles")

# Unknown battles, plus battles an earlier classify run typed by rule
# (battle_type_source = 'classify'), campaigns included. The second group is
# re-evaluated every run: the campaign rule may demote a type, and a campaign
# whose rule no longer fires is re-typed. Nothing extract set is ever selected, and
# neither is a hand-set ('manual') type, even a manual 'unknown'. The
# source is a column, not a missing_data_log note, because extract deletes those
# notes on every re-run. date_end - date_start is computed here because a BC date
# cannot be held by datetime.date, and is trusted only at day precision.
_LOAD_BATTLES_NEEDING_TYPE = text(
    """
    SELECT b.battle_id, b.name, b.wikipedia_url, b.source_slug,
           CAST(b.battle_type AS TEXT),
           CASE WHEN b.date_precision = 'day' AND b.date_end IS NOT NULL
                THEN b.date_end - b.date_start END AS span_days,
           b.date_precision
    FROM battles b
    WHERE (b.battle_type = 'unknown' OR b.battle_type_source = 'classify')
      AND b.battle_type_source IS DISTINCT FROM 'manual'
    ORDER BY b.battle_id
    """
)

_LOAD_BATTLES_NEEDING_TERRAIN = text(
    """
    SELECT battle_id, wikipedia_url, source_slug
    FROM battles
    WHERE terrain IS NULL OR cardinality(terrain) = 0
    ORDER BY battle_id
    """
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


def load_battle_source_slugs(conn: Any) -> dict[int, str | None]:
    """Read every battle's crawl ``source_slug`` in one query.

    Args:
        conn: An open database connection.

    Returns:
        Battle id to slug (or None for a row predating the column), for
        :func:`find_article_path`, which prefers it over the URL.
    """
    return {int(bid): slug for bid, slug in conn.execute(_LOAD_BATTLE_SLUGS).fetchall()}


def load_battles_needing_type(conn: Any) -> list[dict[str, Any]]:
    """List battles classify may still type, with what the campaign rule needs.

    Args:
        conn: An open database connection.

    Returns:
        One mapping per battle with ``battle_id``, ``name``,
        ``wikipedia_url``, ``source_slug``, ``current_type``, ``span_days`` and ``date_precision``.
        ``current_type`` is 'unknown' for a battle nothing has typed, or the
        type an earlier classify run chose by rule (possibly 'campaign'); the
        campaign rule may demote the latter and a stale campaign reverts (a
        type extract or a person set is never listed).
        ``span_days`` is None unless both dates are known to the day.
    """
    return [
        {
            "battle_id": int(bid),
            "name": str(name),
            "wikipedia_url": url,
            "source_slug": slug,
            "current_type": str(current_type),
            "span_days": int(span) if span is not None else None,
            "date_precision": precision,
        }
        for bid, name, url, slug, current_type, span, precision in conn.execute(
            _LOAD_BATTLES_NEEDING_TYPE
        ).fetchall()
    ]


def load_battles_needing_terrain(conn: Any) -> list[dict[str, Any]]:
    """List battles whose ``terrain`` is still NULL or empty.

    Args:
        conn: An open database connection.

    Returns:
        One mapping per battle with ``battle_id``, ``wikipedia_url`` and
        ``source_slug``, for the article lookup.
    """
    return [
        {"battle_id": int(bid), "wikipedia_url": url, "source_slug": slug}
        for bid, url, slug in conn.execute(_LOAD_BATTLES_NEEDING_TERRAIN).fetchall()
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


def find_article_path(
    raw_root: Path, wikipedia_url: str | None, source_slug: str | None = None
) -> Path | None:
    """Map a battle onto its raw article file.

    Args:
        raw_root: The ``data/raw`` directory (or a test fixture root standing
            in for it).
        wikipedia_url: ``battles.wikipedia_url``, or None/empty when the
            battle carries none -- a battle predating extract's write of the
            column, or one added by hand.
        source_slug: ``battles.source_slug``, which is the crawl file's stem.
            Preferred when present, because it needs no URL-spelling guesses.

    Returns:
        The path crawl (or a fixture) left for this article, preferring a
        wikitext sibling over rendered HTML per :data:`ARTICLE_SUFFIXES`, or
        None when no URL is recorded or no matching file exists on disk.
    """
    directory = raw_root / _ARTICLE_DIRNAME
    if source_slug:
        for suffix in ARTICLE_SUFFIXES:
            candidate = directory / f"{source_slug}{suffix}"
            if candidate.is_file():
                return candidate
    if not wikipedia_url:
        return None

    stem = Path(article_filename(wikipedia_url)).stem
    for suffix in ARTICLE_SUFFIXES:
        candidate = directory / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate

    # The digest in the filename is a hash of the URL string crawl fetched
    # (https, percent-encoded), while battles.wikipedia_url comes from
    # Wikidata or DBpedia (often http, decoded), so the exact stem misses
    # nearly every article. The sanitised title in front of the digest does
    # not depend on the URL spelling, so match on that and accept only an
    # unambiguous hit.
    title_part = _DIGEST_SUFFIX_RE.sub("", stem)
    for suffix in ARTICLE_SUFFIXES:
        matches = [
            p
            for p in directory.glob(f"{title_part}-*{suffix}")
            if _DIGEST_SUFFIX_RE.sub("", p.stem) == title_part
        ]
        if len(matches) == 1:
            return matches[0]
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
      AND field_name = :field_name
      AND missingness_class = 'unclassified'
    """
)


def find_unclassified_battle_type_log_ids(
    conn: Any, battle_ids: set[int], field_name: str = "battle_type"
) -> set[int]:
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
        battle_ids: Battles whose ``battle_type`` (or ``field_name``) was
            inferred this run.
        field_name: The ``missing_data_log`` field, 'battle_type' or 'terrain'.

    Returns:
        The matching ``missing_data_log.log_id`` values. Empty when
        ``battle_ids`` is empty, or none of them has such a row.
    """
    if not battle_ids:
        return set()
    rows = conn.execute(
        _FIND_BATTLE_TYPE_LOG_IDS, {"battle_ids": sorted(battle_ids), "field_name": field_name}
    ).fetchall()
    return {int(row[0]) for row in rows}


_FIND_POPULATED_LOG_IDS = text(
    """
    SELECT mdl.log_id
    FROM missing_data_log mdl
    JOIN battles b ON b.battle_id = mdl.battle_id
    WHERE mdl.missingness_class = 'unclassified'
      AND mdl.side_id IS NULL
      AND ((mdl.field_name = 'battle_type' AND b.battle_type <> 'unknown')
        OR (mdl.field_name = 'terrain' AND b.terrain IS NOT NULL AND cardinality(b.terrain) > 0)
        OR (mdl.field_name = 'fortified'
            AND (b.fortified IS NOT NULL OR b.battle_id = ANY(CAST(:fortified_ids AS INT[])))))
    """
)


def find_populated_unclassified_log_ids(conn: Any, fortified_battle_ids: set[int]) -> set[int]:
    """Find 'unclassified' covariate rows whose battles column already has a value.

    Extract re-inserts a ``battle_type``, ``terrain`` or ``fortified`` row on
    every run for a field its own sources did not name, even when classify
    filled the column on an earlier run. Such a row is not missing data, and
    must reach neither the missingness heuristics nor the LLM.

    Args:
        conn: An open database connection.
        fortified_battle_ids: Battles whose ``fortified`` this run's
            battle-type pass is about to set. Read before any write, so the
            column is still NULL when this query runs.

    Returns:
        The ``missing_data_log.log_id`` values to exclude from missingness.
    """
    rows = conn.execute(
        _FIND_POPULATED_LOG_IDS, {"fortified_ids": sorted(fortified_battle_ids)}
    ).fetchall()
    return {int(row[0]) for row in rows}


_LOAD_UNCLASSIFIED_ROWS = text(
    """
    SELECT mdl.log_id, mdl.field_name, mdl.missingness_class,
           mdl.battle_id, mdl.side_id, b.year_astronomical,
           bs.polity, bs.side_label
    FROM missing_data_log mdl
    JOIN battles b ON b.battle_id = mdl.battle_id
    LEFT JOIN battle_sides bs ON bs.side_id = mdl.side_id
    WHERE mdl.missingness_class = 'unclassified'
      AND mdl.field_name = ANY(CAST(:fields AS TEXT[]))
    """
)

# Reconcile's verdict, not the raw reports: a side whose reports were all
# excluded (implausible figures, non-personnel counts, branch-only) has
# troop_reports rows but no estimate, and reconcile logs it unfillable. Those
# sides have no usable troop number, so impute's mnar_delta has to reach them.
_LOAD_TROOP_SIDES = text(
    """
    SELECT side_id
    FROM battle_sides
    WHERE battle_id = :battle_id
      AND est_troops_total IS NOT NULL
    """
)

_LOAD_SIDE_OUTCOMES = text("SELECT side_id, outcome FROM battle_sides WHERE battle_id = :battle_id")


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

        Troop presence is read from ``battle_sides.est_troops_total``, the
        reconcile stage's result, which runs before classify. Reading the raw
        ``troop_reports`` instead made a side reconcile had marked unfillable
        look as though it had troops, so no missingness heuristic fired for it.
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
    for log_id, field_name, current_class, battle_id, side_id, year, polity, side_label in rows:
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
                side_polity=str(polity) if polity else None,
                side_label=str(side_label) if side_label else None,
            )
        )

    return inputs


_LOAD_MISSINGNESS_CONTEXTS = text(
    """
    SELECT mdl.log_id, mdl.battle_id, mdl.field_name, b.name, b.year_astronomical,
           w.name, b.location_name, bs.side_label, bs.polity, CAST(bs.outcome AS TEXT),
           COALESCE(bs.est_troops_total IS NOT NULL, FALSE),
           COALESCE(bs.est_casualties IS NOT NULL, FALSE),
           EXISTS (SELECT 1 FROM battle_sides o
                   WHERE o.battle_id = mdl.battle_id
                     AND o.side_id IS DISTINCT FROM mdl.side_id
                     AND o.est_troops_total IS NOT NULL),
           EXISTS (SELECT 1 FROM battle_sides o
                   WHERE o.battle_id = mdl.battle_id
                     AND o.side_id IS DISTINCT FROM mdl.side_id
                     AND o.est_casualties IS NOT NULL)
    FROM missing_data_log mdl
    JOIN battles b ON b.battle_id = mdl.battle_id
    LEFT JOIN wars w ON w.war_id = b.war_id
    LEFT JOIN battle_sides bs ON bs.side_id = mdl.side_id
    WHERE mdl.log_id = ANY(CAST(:log_ids AS INT[]))
    ORDER BY mdl.log_id
    """
)


def load_missingness_contexts(conn: Any, log_ids: list[int]) -> list[MissingnessContext]:
    """Read what the missingness prompt needs for the given log rows.

    Args:
        conn: An open database connection.
        log_ids: The ``missing_data_log`` rows the heuristics left open.

    Returns:
        One context per row, in ``log_id`` order. Only facts the classify
        stage does not rewrite are read, so the same row yields the same
        context before and after a run (see
        :class:`pipeline.classifiers.records.MissingnessContext`).
    """
    if not log_ids:
        return []
    rows = conn.execute(_LOAD_MISSINGNESS_CONTEXTS, {"log_ids": sorted(log_ids)}).fetchall()
    return [
        MissingnessContext(
            log_id=int(log_id),
            battle_id=int(battle_id),
            field_name=str(field_name),
            battle_name=str(battle_name),
            year_astronomical=int(year) if year is not None else None,
            war_name=str(war) if war else None,
            region=str(region) if region else None,
            side_label=str(label) if label else None,
            polity=str(polity) if polity else None,
            outcome=str(outcome) if outcome else None,
            this_side_has_troops=bool(this_troops),
            other_side_has_troops=bool(other_troops),
            this_side_has_casualties=bool(this_cas),
            other_side_has_casualties=bool(other_cas),
        )
        for (
            log_id,
            battle_id,
            field_name,
            battle_name,
            year,
            war,
            region,
            label,
            polity,
            outcome,
            this_troops,
            this_cas,
            other_troops,
            other_cas,
        ) in rows
    ]


# ─── Defender step ───────────────────────────────────────────────────────────

_LOAD_SIEGE_SIDES = text(
    """
    SELECT b.battle_id, b.name, bs.side_id, bs.side_label, bs.polity
    FROM battles b
    JOIN battle_sides bs ON bs.battle_id = b.battle_id
    WHERE b.battle_type = 'siege_offensive'
      AND bs.is_defender IS NULL
    ORDER BY b.battle_id, bs.side_id
    """
)


def load_siege_sides(conn: Any) -> dict[int, tuple[str, list[SideLabel]]]:
    """Read the sides of every siege that still has an unset defender flag.

    Args:
        conn: An open database connection.

    Returns:
        Battle id to ``(battle name, sides)`` for sieges with at least one
        side whose ``is_defender`` is NULL.
    """
    sieges: dict[int, tuple[str, list[SideLabel]]] = {}
    for battle_id, name, side_id, label, polity in conn.execute(_LOAD_SIEGE_SIDES).fetchall():
        entry = sieges.setdefault(int(battle_id), (str(name), []))
        entry[1].append(SideLabel(int(side_id), str(label), str(polity) if polity else None))
    return sieges
