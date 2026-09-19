"""
Reading the extract stage's commander mentions, and grouping them.

``data/processed/commanders_raw.jsonl`` holds one row per person per source
per battle, so a commander who fought forty battles appears at least forty
times, under however many spellings his sources used. Resolving each row
independently would be wasteful and, worse, inconsistent: the same name could
link to one entity in one battle and another in the next.

:func:`group_mentions` collapses them first. Mentions whose normalised keys
overlap join one :class:`~pipeline.resolvers.records.MentionGroup`, which is
resolved once. That is also where alias evidence collects, because every
surface form that joined the group is a name the person was published under.

Battle context comes from ``data/processed/battles.jsonl``, written in the
same run by the same stage. The year is read as an **astronomical** integer,
never as a ``datetime.date``: much of this corpus is BC and ``date`` cannot
hold it (handover §5.1).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Final

import structlog

from pipeline.resolvers.names import clean_surface, is_title_free, name_keys
from pipeline.resolvers.records import BattleContext, Mention, MentionGroup

__all__ = [
    "astronomical_year",
    "group_mentions",
    "load_battle_contexts",
    "load_mentions",
]

logger = structlog.get_logger()

# A Postgres date literal as pipeline.extractors.wikidata_mapper writes it:
# "0031-09-02 BC" for 2 September 31 BC, "1805-12-02" for an AD date.
_DATE_LITERAL_RE: Final = re.compile(
    r"^\s*(?P<year>\d{1,6})-(?P<month>\d{2})-(?P<day>\d{2})(?P<bc>\s*BC)?\s*$",
    re.IGNORECASE,
)


def astronomical_year(date_literal: str | None) -> int | None:
    """Read the astronomical year out of a stored date literal.

    Astronomical numbering has a year zero, so 31 BC is -30, not -31. Only
    that form subtracts correctly across the era boundary, and it is what
    ``battles.year_astronomical`` holds, so the two agree by construction.

    Args:
        date_literal: A date as the extract stage stored it, or None.

    Returns:
        The astronomical year, or None when the literal is absent or
        unparseable.
    """
    if not date_literal:
        return None

    match = _DATE_LITERAL_RE.match(str(date_literal))
    if match is None:
        logger.debug("battle_date_literal_unparseable", value=str(date_literal)[:32])
        return None

    year = int(match.group("year"))
    if match.group("bc"):
        # 1 BC is astronomical year 0, so 31 BC is -30.
        return -(year - 1)
    return year


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON lines file, skipping lines that will not parse.

    Args:
        path: The file to read.

    Returns:
        One mapping per readable line. An absent file yields no rows and a
        warning, because an empty resolve run is a pipeline-ordering problem
        rather than a crash.
    """
    if not path.is_file():
        logger.warning("resolve_input_missing", path=str(path), hint="run the extract stage first")
        return []

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                decoded = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("resolve_input_line_malformed", path=str(path), line=number)
                continue
            if isinstance(decoded, dict):
                rows.append(decoded)
    return rows


def load_battle_contexts(battles_path: Path) -> dict[str, BattleContext]:
    """Build the per-battle context the matcher gates candidates on.

    Args:
        battles_path: ``data/processed/battles.jsonl``.

    Returns:
        Battle slug to context. A battle with no readable date still gets a
        context, with ``year`` None; the matcher treats that as "cannot
        judge" rather than as a match.
    """
    contexts: dict[str, BattleContext] = {}

    for row in _read_jsonl(battles_path):
        slug = str(row.get("slug") or "").strip()
        if not slug:
            continue
        part_of = row.get("part_of")
        war = str(part_of[0]) if isinstance(part_of, list) and part_of else ""
        date_text = str(row.get("date_start") or "")
        contexts[slug] = BattleContext(
            slug=slug,
            name=str(row.get("name") or slug),
            year=astronomical_year(date_text),
            war=war,
            date_text=date_text,
        )

    undated = sum(1 for c in contexts.values() if c.year is None)
    logger.info("battle_contexts_loaded", battles=len(contexts), undated=undated)
    return contexts


def _side_polities(rows: list[dict[str, Any]]) -> dict[tuple[str, str], str]:
    """Map each (battle slug, side label) onto the side's polity.

    Args:
        rows: Rows of ``battles.jsonl``.

    Returns:
        The polity for each side, used as a matching tiebreak.
    """
    polities: dict[tuple[str, str], str] = {}
    for row in rows:
        slug = str(row.get("slug") or "").strip()
        sides = row.get("sides")
        if not slug or not isinstance(sides, list):
            continue
        for side in sides:
            if not isinstance(side, dict):
                continue
            label = str(side.get("label") or "")
            if label:
                polities[(slug, label)] = str(side.get("polity") or "")
    return polities


def load_mentions(commanders_path: Path, battles_path: Path) -> list[Mention]:
    """Read every commander mention, with its battle context attached.

    Args:
        commanders_path: ``data/processed/commanders_raw.jsonl``.
        battles_path: ``data/processed/battles.jsonl``.

    Returns:
        One :class:`~pipeline.resolvers.records.Mention` per readable row.
        Rows naming no battle are dropped, since there is nothing to attach
        the resulting ``battle_commanders`` row to.
    """
    contexts = load_battle_contexts(battles_path)
    polities = _side_polities(_read_jsonl(battles_path))

    mentions: list[Mention] = []
    for row in _read_jsonl(commanders_path):
        slug = str(row.get("battle_slug") or "").strip()
        name = str(row.get("name") or "").strip()
        if not slug or not name:
            continue

        raw_provenance = row.get("provenance")
        provenance = raw_provenance if isinstance(raw_provenance, dict) else {}
        side_label = str(row.get("side_label") or "")

        mentions.append(
            Mention(
                battle_slug=slug,
                battle_name=str(row.get("battle_name") or slug),
                side_label=side_label,
                name=name,
                apparent_role=str(row.get("apparent_role") or "unclear"),
                role_evidence=str(row.get("role_evidence") or ""),
                source_type=str(provenance.get("source_type") or "wikipedia_infobox"),
                extraction_method=str(provenance.get("extraction_method") or "infobox_parser"),
                source_ref=str(provenance.get("source_ref") or ""),
                source_title=str(provenance.get("source_title") or ""),
                mentions=int(row.get("mentions") or 1),
                polity=polities.get((slug, side_label), ""),
                context=contexts.get(slug),
            )
        )

    logger.info("commander_mentions_loaded", mentions=len(mentions))
    return mentions


def _display_name(surface_forms: list[str]) -> str:
    """Choose the surface form that best names a group.

    Args:
        surface_forms: Every cleaned form seen, with repeats.

    Returns:
        A form without a leading rank first of all -- this becomes
        ``generals.canonical_name`` for a new entity, and "Gen. Patton" is a
        worse name to publish than "Patton". Then the most frequent form,
        then the longest, which is usually the fullest, and finally
        alphabetical order so a re-run picks the same name.
    """
    counts = Counter(surface_forms)
    return max(
        counts,
        key=lambda form: (is_title_free(form), counts[form], len(form), form),
    )


def group_mentions(mentions: list[Mention]) -> list[MentionGroup]:
    """Collapse mentions that name the same person into one group.

    Grouping is transitive through shared keys: "Gen. Patton" and "Patton"
    share the title-stripped key, so they join, and a mention of "Patton,
    George" joins through its uninverted key. No similarity scoring happens
    here -- only exact key identity -- because a wrong merge at this stage is
    invisible downstream and would pool two commanders' records into one
    skill estimate.

    Args:
        mentions: Every mention loaded from the processed files.

    Returns:
        One group per distinct person-as-named, in first-seen order.
    """
    groups: list[MentionGroup] = []
    by_key: dict[str, MentionGroup] = {}
    surfaces: dict[int, list[str]] = {}

    for mention in mentions:
        keys = name_keys(mention.name)
        if not keys:
            # A mention that folds to nothing still has to reach the stage so
            # that it is logged as unresolved rather than silently dropped.
            keys = (f"unnameable:{mention.battle_slug}:{mention.name.strip().lower()}",)

        existing = next((by_key[key] for key in keys if key in by_key), None)

        if existing is None:
            group = MentionGroup(key=keys[-1], keys=keys, display_name="")
            groups.append(group)
            surfaces[id(group)] = []
        else:
            group = existing
            merged = list(group.keys)
            merged.extend(key for key in keys if key not in merged)
            group.keys = tuple(merged)

        for key in group.keys:
            by_key.setdefault(key, group)

        group.mentions.append(mention)
        surfaces[id(group)].append(clean_surface(mention.name) or mention.name.strip())

    for group in groups:
        forms = surfaces[id(group)]
        group.display_name = _display_name(forms)
        group.surface_forms = sorted(set(forms))

    logger.info("mentions_grouped", mentions=len(mentions), groups=len(groups))
    return groups
