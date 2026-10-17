"""
Apply the extract stage's manual overrides to the merged battle records.

``config/overrides.yaml`` entries of kind ``outcome``, ``battle_date`` and
``side_polity`` are applied to the in-memory :class:`BattleExtraction` list
after extraction and before anything is written, so ``data/processed/*.jsonl``
(which resolve reads) and the database both carry the correction. The stage
runs this on every run, after its own logic, so a re-run never undoes it.

An entry that cannot be applied is reported unmatched with a reason and
changes nothing: each entry is checked in full before the first mutation.

Removing a ``side_polity`` or ``battle_date`` entry does not revert the stored
value when extract itself supplies none for that field: the database write
COALESCEs polity and ``date_start`` (see ``store.py``), so the old value stays.
Reverting needs a fresh extract into a clean database or a manual reset. An
outcome entry is different, because extract owns ``outcome`` and clears it.

This module is not re-exported from ``pipeline.extractors``: ``pipeline.overrides``
imports ``pipeline.extractors.dates``, so the package importing it back would be
a cycle.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Final

import structlog

from pipeline.extractors.dates import astronomical_year, parse_date_text
from pipeline.extractors.records import BattleExtraction, SideExtraction
from pipeline.overrides import OverrideEntry, OverrideReport

__all__ = [
    "MANUAL_OUTCOME_METHOD",
    "apply_extract_overrides",
    "carries_override",
    "outcome_override_slugs",
]

logger = structlog.get_logger()

# Recorded on BattleExtraction.outcome_method for outcomes an override set.
MANUAL_OUTCOME_METHOD: Final[str] = "manual"

_OUTCOME_GAP_FIELDS: Final[frozenset[str]] = frozenset({"outcome", "outcome_level"})
_DATE_GAP_FIELDS: Final[frozenset[str]] = frozenset({"date_start"})
_UNPARSED_DATE_STATUS: Final[str] = "unparsed_date"
# Disagreement.field_name values each override kind settles, so the battle
# stops being flagged for a conflict the override resolved.
_OUTCOME_DISAGREEMENT_FIELDS: Final[frozenset[str]] = frozenset(
    {"victor", "outcome_level", "outcome"}
)
_DATE_DISAGREEMENT_FIELDS: Final[frozenset[str]] = frozenset(
    {"date_start", "date_end", "date_precision"}
)
_NOTE_PREFIX: Final[str] = "override "
_WIN_LEVELS: Final[frozenset[str]] = frozenset({"decisive_victory", "victory", "pyrrhic_victory"})


def _norm(text: str) -> str:
    """Return the comparison form of a slug or side label (strip + casefold)."""
    return text.strip().casefold()


def outcome_override_slugs(entries: Sequence[OverrideEntry]) -> frozenset[str]:
    """List the battles an outcome override targets, in comparison form.

    The stage uses this before extraction to skip the outcome LLM request for
    a battle whose outcome is already decided by hand: that request would
    defer and halt the pipeline for nothing.

    Args:
        entries: The extract stage's override entries.

    Returns:
        Normalised battle slugs.
    """
    return frozenset(_norm(e.target["battle"]) for e in entries if e.kind == "outcome")


def _sides_by_label(battle: BattleExtraction) -> dict[str, list[SideExtraction]]:
    grouped: dict[str, list[SideExtraction]] = {}
    for side in battle.sides:
        grouped.setdefault(_norm(side.label), []).append(side)
    return grouped


def _decided_level(levels: Sequence[str]) -> str | None:
    """Summarise per-side outcomes as a battle-level outcome, or None if mixed."""
    winners = [level for level in levels if level in _WIN_LEVELS]
    if len(winners) == 1:
        return winners[0]
    if levels and len(set(levels)) == 1 and levels[0] == "indecisive":
        return "indecisive"
    return None


def _apply_outcome(entry: OverrideEntry, battle: BattleExtraction) -> str | None:
    """Set every side's outcome; return an unmatched reason, or None when applied."""
    grouped = _sides_by_label(battle)
    wanted = {_norm(label): level for label, level in entry.value.items()}
    if set(wanted) != set(grouped):
        shown = {_norm(label): label for label in entry.value}
        absent = sorted(shown[k] for k in set(wanted) - set(grouped))
        extra = sorted(grouped[k][0].label for k in set(grouped) - set(wanted))
        return (
            f"side labels differ: override names {absent} which the battle lacks; "
            f"battle has {extra} which the override omits"
        )
    for key, level in wanted.items():
        for side in grouped[key]:
            side.outcome = level
    battle.outcome_method = MANUAL_OUTCOME_METHOD
    # The merge's battle-level grade described the outcome it decided, not this one.
    battle.facts.outcome_level = _decided_level(list(wanted.values()))
    # An outcome gap would tell later stages the outcome is missing, and imputation
    # would then overwrite a decided one.
    battle.missing = [m for m in battle.missing if m.field_name not in _OUTCOME_GAP_FIELDS]
    battle.disagreements = [
        d for d in battle.disagreements if d.field_name not in _OUTCOME_DISAGREEMENT_FIELDS
    ]
    return None


def _date_sort_key(literal: str) -> tuple[int, int, int]:
    """Order date literals chronologically, BC-aware (no datetime.date)."""
    year = astronomical_year(literal)
    if year is None:
        raise ValueError(f"not a date literal: {literal!r}")
    return year, int(literal[5:7]), int(literal[8:10])


def _apply_battle_date(entry: OverrideEntry, battle: BattleExtraction) -> str | None:
    """Set the battle's dates; return an unmatched reason, or None when applied."""
    start = parse_date_text(entry.value["start"])
    if start is None:
        return f"start {entry.value['start']!r} is not a date the extractor can represent"
    if start.date_end is not None:
        return f"start {entry.value['start']!r} names a range; give one date and use end"

    end_literal: str | None = None
    if entry.value["end"] is not None:
        end = parse_date_text(entry.value["end"])
        if end is None:
            return f"end {entry.value['end']!r} is not a date the extractor can represent"
        if end.date_end is not None or end.precision != "day":
            return f"end {entry.value['end']!r} must be one full day (YYYY-MM-DD)"
        end_literal = end.date_start
        if _date_sort_key(end_literal) < _date_sort_key(start.date_start):
            return f"end {end_literal} is before start {start.date_start}"

    facts = battle.facts
    facts.date_start = start.date_start
    facts.date_end = end_literal
    facts.date_precision = entry.value["precision"]
    # Only the entry's own wording ("c. 1812") makes a curated date approximate.
    facts.date_circa = True if start.circa else None
    battle.missing = [m for m in battle.missing if m.field_name not in _DATE_GAP_FIELDS]
    # The infobox date the parser gave up on no longer flags the battle for review.
    battle.failures = [f for f in battle.failures if f.status != _UNPARSED_DATE_STATUS]
    battle.disagreements = [
        d for d in battle.disagreements if d.field_name not in _DATE_DISAGREEMENT_FIELDS
    ]
    return None


def _apply_side_polity(entry: OverrideEntry, battle: BattleExtraction) -> str | None:
    """Set the side's polity; return an unmatched reason, or None when applied."""
    sides = _sides_by_label(battle).get(_norm(entry.target["side"]))
    if sides is None:
        known = sorted(s.label for s in battle.sides)
        return f"side {entry.target['side']!r} not found in battle (sides: {known})"
    for side in sides:
        side.polity = str(entry.value)
    return None


_APPLIERS: Final[dict[str, Callable[[OverrideEntry, BattleExtraction], str | None]]] = {
    "outcome": _apply_outcome,
    "battle_date": _apply_battle_date,
    "side_polity": _apply_side_polity,
}


def carries_override(battle: BattleExtraction) -> bool:
    """Report whether an override was applied to the battle this run.

    Args:
        battle: A merged battle after :func:`apply_extract_overrides`.

    Returns:
        True when an applied-override note is on the record. The note is used
        rather than a new field so ``battles.jsonl`` keeps its schema.
    """
    return any(n.startswith(_NOTE_PREFIX) and " applied (" in n for n in battle.notes)


def apply_extract_overrides(
    battles: Sequence[BattleExtraction],
    entries: Sequence[OverrideEntry],
    report: OverrideReport,
) -> dict[str, OverrideEntry]:
    """Apply the extract override entries to the merged battles, in place.

    Every entry is reported to ``report`` exactly once, applied or unmatched.

    Args:
        battles: This run's merged battle records; modified in place.
        entries: The extract stage's entries, in file order.
        report: The run's report, which the ``overrides_gate`` check reads.

    Returns:
        The applied outcome entries by battle slug, for the outcome provenance
        the database write attaches.
    """
    by_slug: dict[str, BattleExtraction] = {}
    for battle in battles:
        by_slug.setdefault(_norm(battle.slug), battle)

    outcome_entries: dict[str, OverrideEntry] = {}
    for entry in entries:
        target = by_slug.get(_norm(entry.target["battle"]))
        if target is None:
            report.unmatched(entry, "battle not found")
            continue
        reason = _APPLIERS[entry.kind](entry, target)
        if reason is not None:
            report.unmatched(entry, reason)
            continue
        target.notes.append(f"{_NOTE_PREFIX}{entry.id} applied ({entry.kind})")
        report.applied(entry)
        if entry.kind == "outcome":
            outcome_entries[target.slug] = entry
        logger.info("override_applied", id=entry.id, kind=entry.kind, slug=target.slug)
    return outcome_entries
