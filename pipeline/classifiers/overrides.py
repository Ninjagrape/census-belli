"""
Manual overrides owned by the classify stage: ``battle_type`` and ``attribution``.

``config/overrides.yaml`` is applied by the stage that owns each kind, after
that stage's own logic, on every run (see :mod:`pipeline.overrides`). This
module holds the matching and the writes; ``pipeline.stages.classify`` only
wires them in.

**battle_type.** The battle is found by ``battles.source_slug`` (strip +
casefold). The value is written with ``battle_type_source = 'manual'`` before
the rule pass runs, and the rule pass never selects a 'manual' row, so no rule
or campaign check can revert it. Extract may rewrite the type (source
'extract') on a re-run; the next classify run puts the manual value back.

**attribution.** The entry lists the *whole* commander set of one side. It
applies only when that list equals the side's linked commanders exactly: a
partial application would leave weights that do not sum to one, and a missing
or extra general usually means resolve linked someone else than the curator
saw. A mismatch, or a hierarchy :func:`pipeline.classifiers.roles.validate_side`
rejects, makes the entry unmatched with a reason, and nothing is written for
it. Resolve may reset a side whenever it prunes a link, which is why this runs
on every classify run. A side with no linked commanders is not loaded as a
side group at all, so an entry for it is unmatched rather than a crash.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import structlog
from sqlalchemy import text

from pipeline.classifiers.records import CommanderRow, RoleDecision, SideGroup
from pipeline.classifiers.roles import validate_side
from pipeline.overrides import OverrideEntry

__all__ = [
    "MANUAL_METHOD",
    "ManualBattleType",
    "ManualSide",
    "match_attribution",
    "match_battle_types",
    "overridden_side_ids",
    "write_manual_battle_types",
    "write_manual_role_evidence",
]

logger = structlog.get_logger()

MANUAL_METHOD: Final[str] = "manual"
_EVIDENCE_PREFIX: Final[str] = "[manual "
_QID_RE: Final[re.Pattern[str]] = re.compile(r"^[Qq][1-9]\d*$")

_WRITE_BATTLE_TYPE = text(
    """
    UPDATE battles SET
        battle_type = CAST(:battle_type AS battle_type),
        battle_type_source = 'manual'
    WHERE battle_id = :battle_id
    """
)

_WRITE_ROLE_EVIDENCE = text(
    "UPDATE battle_commanders SET role_evidence = :evidence WHERE bc_id = :bc_id"
)


@dataclass(frozen=True)
class ManualBattleType:
    """A battle-type override that found its battle.

    Attributes:
        entry: The override entry.
        battle_id: The matched battle.
        battle_type: The value to write.
    """

    entry: OverrideEntry
    battle_id: int
    battle_type: str


@dataclass(frozen=True)
class ManualSide:
    """An attribution override that matched a side and passed validation.

    Attributes:
        entry: The override entry.
        side: The side it applies to.
        decisions: One decision per commander on the side, in side order.
        evidence: ``bc_id`` to the ``role_evidence`` text to store.
    """

    entry: OverrideEntry
    side: SideGroup
    decisions: list[RoleDecision]
    evidence: dict[int, str]


def _fold(value: str) -> str:
    return value.strip().casefold()


def match_battle_types(
    entries: Iterable[OverrideEntry], slugs: Mapping[int, str | None]
) -> list[tuple[OverrideEntry, ManualBattleType | str]]:
    """Find the battle of every ``battle_type`` entry.

    Args:
        entries: The classify entries; other kinds are skipped.
        slugs: Battle id to ``source_slug``, as ``load_battle_source_slugs`` reads it.

    Returns:
        ``(entry, match)`` per entry, where ``match`` is a
        :class:`ManualBattleType` or the reason there is none (no battle with
        that slug, or the slug is ambiguous after casefolding).
    """
    by_slug: dict[str, list[int]] = {}
    for battle_id, slug in slugs.items():
        if slug:
            by_slug.setdefault(_fold(slug), []).append(battle_id)
    results: list[tuple[OverrideEntry, ManualBattleType | str]] = []
    for entry in entries:
        if entry.kind != "battle_type":
            continue
        ids = by_slug.get(_fold(entry.target["battle"]), [])
        if len(ids) == 1:
            results.append((entry, ManualBattleType(entry, ids[0], str(entry.value))))
        elif ids:
            results.append((entry, f"battle slug {entry.target['battle']!r} is ambiguous"))
        else:
            results.append((entry, f"no battle with source_slug {entry.target['battle']!r}"))
    return results


def _find_commander(key: str, commanders: Sequence[CommanderRow]) -> CommanderRow | None:
    """Match a general key to a side's commander: Q-id exactly, else name casefolded."""
    stripped = key.strip()
    if _QID_RE.match(stripped):
        wanted = stripped.upper()
        found = [c for c in commanders if c.wikidata_id and c.wikidata_id.strip().upper() == wanted]
    else:
        found = [c for c in commanders if _fold(c.name) == _fold(stripped)]
    return found[0] if len(found) == 1 else None


def _evidence_text(entry: OverrideEntry, existing: str) -> str:
    """Prefix the stored evidence with the override citation, replacing an older prefix."""
    base = existing
    if base.startswith(_EVIDENCE_PREFIX):
        base = base.split("\n", 1)[1] if "\n" in base else ""
    head = f"{_EVIDENCE_PREFIX}{entry.id}] {entry.source}"
    return f"{head}\n{base}" if base else head


def _find_sides(
    entry: OverrideEntry, groups: Sequence[SideGroup], slugs: Mapping[int, str | None]
) -> list[SideGroup]:
    battle, label = _fold(entry.target["battle"]), _fold(entry.target["side"])
    return [
        g
        for g in groups
        if _fold(slugs.get(g.battle_id) or "") == battle and _fold(g.side_label) == label
    ]


def match_attribution(
    entry: OverrideEntry, groups: Sequence[SideGroup], slugs: Mapping[int, str | None]
) -> ManualSide | str:
    """Match one attribution entry to a side and build its validated decisions.

    Args:
        entry: An ``attribution`` entry.
        groups: Every side with at least one linked commander.
        slugs: Battle id to ``source_slug``.

    Returns:
        The :class:`ManualSide`, or the reason the entry cannot be applied:
        no such side, a commander set that differs from the side's (naming the
        missing and extra generals), or a hierarchy the role validator rejects.
    """
    sides = _find_sides(entry, groups, slugs)
    if not sides:
        return (
            f"no side {entry.target['side']!r} with linked commanders "
            f"in battle {entry.target['battle']!r}"
        )
    if len(sides) > 1:
        return f"side {entry.target['side']!r} is ambiguous in battle {entry.target['battle']!r}"
    side = sides[0]

    matched: dict[str, CommanderRow] = {}
    extra: list[str] = []
    for item in entry.value:
        commander = _find_commander(str(item["general"]), side.commanders)
        if commander is None or commander.bc_id in {c.bc_id for c in matched.values()}:
            extra.append(str(item["general"]))
        else:
            matched[str(item["general"])] = commander
    linked = {c.bc_id for c in matched.values()}
    missing = [c.name for c in side.commanders if c.bc_id not in linked]
    if missing or extra:
        return (
            "listed generals differ from the side's linked commanders: "
            f"linked but not listed {missing}, listed but not linked {extra}"
        )

    decisions: list[RoleDecision] = []
    for item in entry.value:
        commander = matched[str(item["general"])]
        parent = item["reports_to"]
        parent_commander = None if parent is None else _find_commander(str(parent), side.commanders)
        if parent is not None and parent_commander is None:
            return f"reports_to {parent!r} is not a linked commander of the side"
        decisions.append(
            RoleDecision(
                bc_id=commander.bc_id,
                command_role=str(item["role"]),
                hierarchy_rank=int(item["rank"]),
                reports_to_bc_id=None if parent_commander is None else parent_commander.bc_id,
                attribution_weight=float(item["weight"]),
                attribution_method=MANUAL_METHOD,
                confidence=1.0,
                needs_review=False,
                reason=f"manual override {entry.id}: {entry.source}",
            )
        )
    problem = validate_side(decisions)
    if problem is not None:
        return f"attribution fails the role rules: {problem}"

    by_bc = {c.bc_id: c for c in side.commanders}
    evidence = {d.bc_id: _evidence_text(entry, by_bc[d.bc_id].role_evidence) for d in decisions}
    # Side order, so decisions line up with side.commanders like the rule pass.
    order = {c.bc_id: i for i, c in enumerate(side.commanders)}
    decisions.sort(key=lambda d: order[d.bc_id])
    return ManualSide(entry, side, decisions, evidence)


def overridden_side_ids(
    entries: Iterable[OverrideEntry],
    groups: Sequence[SideGroup],
    slugs: Mapping[int, str | None],
) -> set[int]:
    """Return the sides an attribution override applies to, without reporting.

    Used by the offline request exporter, which must not ask the LLM about a
    side the stage will overwrite.

    Args:
        entries: The classify entries.
        groups: Every side with linked commanders.
        slugs: Battle id to ``source_slug``.

    Returns:
        ``side_id`` of every side whose entry matches and validates.
    """
    ids: set[int] = set()
    for entry in entries:
        if entry.kind != "attribution":
            continue
        match = match_attribution(entry, groups, slugs)
        if isinstance(match, ManualSide):
            ids.add(match.side.side_id)
    return ids


def write_manual_battle_types(conn: Any, matches: Iterable[ManualBattleType]) -> int:
    """Write manual battle types and mark their source 'manual'.

    Args:
        conn: An open database connection.
        matches: Matched ``battle_type`` overrides.

    Returns:
        How many battles were written.
    """
    count = 0
    for match in matches:
        conn.execute(
            _WRITE_BATTLE_TYPE,
            {"battle_id": match.battle_id, "battle_type": match.battle_type},
        )
        count += 1
    return count


def write_manual_role_evidence(conn: Any, sides: Iterable[ManualSide]) -> int:
    """Store the override citation on each overridden commander row.

    Args:
        conn: An open database connection.
        sides: Matched attribution overrides.

    Returns:
        How many rows were written.
    """
    count = 0
    for manual in sides:
        for bc_id, evidence in manual.evidence.items():
            conn.execute(_WRITE_ROLE_EVIDENCE, {"bc_id": bc_id, "evidence": evidence})
            count += 1
    return count
