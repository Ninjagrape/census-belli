"""From raw side rows to the model's aligned arrays.

Implements design sections 1, 2, 4, 5 and revisions R1, R5, R12.

Rules, in the order they are applied to each battle:

1. Exactly two sides, else excluded (``battles_multi_side``; also catches a
   one-side battle, which has no opponent).
2. Orientation: A is the lower ``side_id``. Insertion order carries no
   information, so nothing downstream may depend on it (the symmetric
   cutpoints in :mod:`pipeline.modelers.ordinal` guarantee that).
3. Outcome from A's perspective, mirrored from B when A has none. Both
   present: same class (win / indecisive / loss) but different grade becomes
   an interval over the two grades (``battles_grade_disagree``, R5); different
   class is a conflict (``battles_outcome_conflict``); neither present is
   ``battles_no_outcome``.
4. Side classification: no commander with positive weight is ``unknown``;
   two or more commanders all ``default_split`` is ``unattributed`` (R1);
   anything else is ``named`` with weights renormalised to sum to one (the
   largest pre-renormalisation deviation is recorded, R12). A side mixing
   ``default_split`` with other methods stays named: some credit was
   assigned on evidence, and dropping the default-split members would
   silently move their weight to the others.
5. A battle with no named side (``battles_no_creditable_side``) or with the
   same general on both named sides (``battles_general_both_sides``) is
   excluded.
6. Defender flags: if exactly one side is known the other is its complement;
   coded +1 / -1 / 0 (unknown) and differenced.

An outcome string that is not in :data:`OUTCOME_LEVEL` is treated as missing
and counted in ``outcomes_unrecognised``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from pipeline.modelers.records import (
    DEFAULT_SPLIT,
    KIND_NAMED,
    KIND_UNATTRIBUTED,
    KIND_UNKNOWN,
    ModelCounts,
    ModelDesign,
    SideInput,
)

__all__ = ["OUTCOME_LEVEL", "build_design", "outcome_interval", "score_of_interval", "x_difference"]

# k is the A-perspective ordinal level; pyrrhic is a win because the scale is
# who held the field, not what it cost.
OUTCOME_LEVEL: Final[Mapping[str, int]] = {
    "decisive_defeat": 0,
    "defeat": 1,
    "indecisive": 2,
    "victory": 3,
    "pyrrhic_victory": 3,
    "decisive_victory": 4,
}
_TOP_LEVEL: Final[int] = 4


def _level_class(level: int) -> int:
    """-1 loss, 0 indecisive, +1 win."""
    if level >= 3:
        return 1
    return 0 if level == 2 else -1


def outcome_interval(a_level: int | None, b_level: int | None) -> tuple[int, int] | str:
    """Combine the two sides' reported levels into A-perspective bounds.

    Args:
        a_level: A's own level, or None.
        b_level: B's own level (B's perspective), or None.

    Returns:
        ``(k_lo, k_hi)``, equal when exact; or the exclusion reason,
        ``'no_outcome'`` or ``'conflict'``.
    """
    mirrored = None if b_level is None else _TOP_LEVEL - b_level
    if a_level is None and mirrored is None:
        return "no_outcome"
    if a_level is None:
        assert mirrored is not None
        return mirrored, mirrored
    if mirrored is None:
        return a_level, a_level
    if _level_class(a_level) != _level_class(mirrored):
        return "conflict"
    return min(a_level, mirrored), max(a_level, mirrored)


def score_of_interval(k_lo: int, k_hi: int) -> float:
    """A-perspective 0 / 0.5 / 1 score of an outcome interval.

    Args:
        k_lo: Lowest compatible level.
        k_hi: Highest compatible level (same class as ``k_lo``).

    Returns:
        1.0 for a win, 0.5 for indecisive, 0.0 for a loss.
    """
    if k_lo >= 3:
        return 1.0
    return 0.5 if k_hi == 2 else 0.0


def _defender_codes(a: bool | None, b: bool | None) -> tuple[int, int, bool]:
    """Code the two flags +1 / -1 / 0, filling the complement of a known one."""
    if a is None and b is not None:
        a = not b
    elif b is None and a is not None:
        b = not a

    def code(flag: bool | None) -> int:
        return 0 if flag is None else (1 if flag else -1)

    return code(a), code(b), a is not None


@dataclass(frozen=True)
class _ResolvedSide:
    kind: int
    commanders: tuple[tuple[int, float], ...]  # (general_id, renormalised weight)
    deviation: float


def _classify_side(side: SideInput) -> _ResolvedSide:
    """Decide how a side enters the model and renormalise its weights."""
    merged: dict[int, float] = defaultdict(float)
    for cmd in side.commanders:
        weight = cmd.attribution_weight
        if math.isfinite(weight) and weight > 0.0:
            merged[cmd.general_id] += weight
    total = sum(merged.values())
    if not merged or total <= 0.0:
        return _ResolvedSide(KIND_UNKNOWN, (), 0.0)
    if len(merged) >= 2 and all(c.attribution_method == DEFAULT_SPLIT for c in side.commanders):
        return _ResolvedSide(KIND_UNATTRIBUTED, (), 0.0)
    commanders = tuple((gid, w / total) for gid, w in sorted(merged.items()))
    return _ResolvedSide(KIND_NAMED, commanders, abs(total - 1.0))


def _count_inputs(sides: Sequence[SideInput], counts: ModelCounts) -> None:
    """Fill the all-sides denominators of the exclusion_concentration gate."""
    for side in sides:
        distinct = {c.general_id for c in side.commanders}
        if len(distinct) >= 2:
            counts.sides_multi_commander_all += 1
            if all(c.attribution_method == DEFAULT_SPLIT for c in side.commanders):
                counts.sides_default_split_all += 1


def _level_of(text: str | None, counts: ModelCounts) -> int | None:
    if text is None:
        return None
    level = OUTCOME_LEVEL.get(text)
    if level is None:
        counts.outcomes_unrecognised += 1
    return level


@dataclass(frozen=True)
class _Row:
    """One included battle, before it is flattened into arrays."""

    battle_id: int
    sides: tuple[SideInput, SideInput]
    resolved: tuple[_ResolvedSide, _ResolvedSide]
    k_lo: int
    k_hi: int
    d_a: int
    d_b: int
    known: bool
    year: float


@dataclass
class _Tally:
    """Per-general appearance counts, keyed by raw general id."""

    included: dict[int, int]
    uncredited: dict[int, int]
    excluded_multi: dict[int, int]
    effective: dict[int, float]


def _exclusion_reason(group: list[SideInput], counts: ModelCounts) -> str | _Row:
    """Apply rules 1 to 5 to one battle.

    Returns:
        The name of the ``ModelCounts`` counter to increment when the battle is
        excluded, otherwise the resolved row.
    """
    if len(group) != 2:
        return "battles_multi_side"
    side_a, side_b = group
    interval = outcome_interval(
        _level_of(side_a.outcome, counts), _level_of(side_b.outcome, counts)
    )
    if interval == "no_outcome":
        return "battles_no_outcome"
    if isinstance(interval, str):
        return "battles_outcome_conflict"
    resolved = (_classify_side(side_a), _classify_side(side_b))
    if all(r.kind != KIND_NAMED for r in resolved):
        return "battles_no_creditable_side"
    if all(r.kind == KIND_NAMED for r in resolved) and {g for g, _ in resolved[0].commanders} & {
        g for g, _ in resolved[1].commanders
    }:
        return "battles_general_both_sides"
    d_a, d_b, known = _defender_codes(side_a.is_defender, side_b.is_defender)
    years = [s.year for s in group if s.year is not None]
    return _Row(
        battle_id=side_a.battle_id,
        sides=(side_a, side_b),
        resolved=resolved,
        k_lo=interval[0],
        k_hi=interval[1],
        d_a=d_a,
        d_b=d_b,
        known=known,
        year=float(years[0]) if years else math.nan,
    )


def build_design(sides: Sequence[SideInput]) -> ModelDesign:
    """Apply the inclusion rules and lay the survivors out as arrays.

    Input order is irrelevant: battles are processed in ``battle_id`` order
    and sides within a battle in ``side_id`` order.

    Args:
        sides: Every in-scope ``battle_sides`` row with its commanders.

    Returns:
        The design; ``design.counts`` holds the exclusion counters. With no
        included battle every array is empty.
    """
    counts = ModelCounts()
    _count_inputs(sides, counts)
    by_battle: dict[int, list[SideInput]] = defaultdict(list)
    for side in sides:
        by_battle[side.battle_id].append(side)
    counts.battles_in_scope = len(by_battle)

    tally = _Tally(defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(float))
    rows: list[_Row] = []

    for battle_id in sorted(by_battle):
        group = sorted(by_battle[battle_id], key=lambda s: s.side_id)
        reason = _exclusion_reason(group, counts)
        if isinstance(reason, str):
            setattr(counts, reason, getattr(counts, reason) + 1)
            for gid in {c.general_id for s in group for c in s.commanders}:
                tally.uncredited[gid] += 1
                if reason == "battles_multi_side":
                    tally.excluded_multi[gid] += 1
            continue
        row = reason
        rows.append(row)
        counts.battles_included += 1
        if row.k_lo != row.k_hi:
            counts.battles_grade_disagree += 1
        for side, res in zip(row.sides, row.resolved, strict=True):
            counts.max_weight_deviation = max(counts.max_weight_deviation, res.deviation)
            if res.kind == KIND_NAMED:
                counts.sides_named += 1
                for gid, w in res.commanders:
                    tally.included[gid] += 1
                    tally.effective[gid] += w
            elif res.kind == KIND_UNKNOWN:
                counts.sides_unknown += 1
                # Commanders whose weights were all zero or NULL are on the
                # side but earn nothing: uncredited, like the unattributed case.
                for gid in {c.general_id for c in side.commanders}:
                    tally.uncredited[gid] += 1
            else:
                counts.sides_unattributed += 1
                # Present on the side, credited with nothing.
                for gid in {c.general_id for c in side.commanders}:
                    tally.uncredited[gid] += 1

    return _assemble(rows, counts, tally)


def _assemble(rows: list[_Row], counts: ModelCounts, tally: _Tally) -> ModelDesign:
    """Turn the per-battle records into the aligned arrays."""
    n = len(rows)
    general_ids = np.array(sorted(tally.included), dtype=np.int64)
    index_of = {int(g): i for i, g in enumerate(general_ids)}

    side_ids = np.zeros(2 * n, dtype=np.int64)
    side_kind = np.zeros(2 * n, dtype=np.int64)
    cmd_side: list[int] = []
    cmd_general: list[int] = []
    cmd_weight: list[float] = []

    for b, row in enumerate(rows):
        for offset in (0, 1):
            side_index = 2 * b + offset
            side_ids[side_index] = row.sides[offset].side_id
            side_kind[side_index] = row.resolved[offset].kind
            for gid, w in row.resolved[offset].commanders:
                cmd_side.append(side_index)
                cmd_general.append(index_of[gid])
                cmd_weight.append(w)

    k_lo = np.array([r.k_lo for r in rows], dtype=np.int64)
    k_hi = np.array([r.k_hi for r in rows], dtype=np.int64)
    known = np.array([r.known for r in rows], dtype=bool)

    def per_general(source: Mapping[int, int] | Mapping[int, float], dtype: type) -> np.ndarray:
        return np.array([source.get(int(g), 0) for g in general_ids], dtype=dtype)

    return ModelDesign(
        battle_ids=np.array([r.battle_id for r in rows], dtype=np.int64),
        side_a=2 * np.arange(n, dtype=np.int64),
        side_b=2 * np.arange(n, dtype=np.int64) + 1,
        k_lo=k_lo,
        k_hi=k_hi,
        y_a=np.array([score_of_interval(r.k_lo, r.k_hi) for r in rows], dtype=float),
        defender_diff=np.array([r.d_a - r.d_b for r in rows], dtype=float),
        defender_known=known,
        defender_term_used=bool(known.any()),
        year=np.array([r.year for r in rows], dtype=float),
        side_ids=side_ids,
        side_kind=side_kind,
        side_sign=np.tile(np.array([1.0, -1.0]), n),
        cmd_side=np.array(cmd_side, dtype=np.int64),
        cmd_general=np.array(cmd_general, dtype=np.int64),
        cmd_weight=np.array(cmd_weight, dtype=float),
        general_ids=general_ids,
        n_included_battles=per_general(tally.included, np.int64),
        n_uncredited_battles=per_general(tally.uncredited, np.int64),
        n_excluded_multi_side=per_general(tally.excluded_multi, np.int64),
        effective_battles=per_general(tally.effective, np.float64),
        counts=counts,
    )


def x_difference(design: ModelDesign, log_troops: Mapping[int, float]) -> np.ndarray:
    """``x_A - x_B`` per battle from a mapping of ``side_id`` to log troops.

    Args:
        design: The design.
        log_troops: Log troop count for every side in the design. The caller
            guarantees completeness (an imputed round has a value for every
            in-scope side).

    Returns:
        Array of shape ``(n_battles,)``.

    Raises:
        ValueError: If any side of an included battle has no value.
    """
    missing = [int(s) for s in design.side_ids if int(s) not in log_troops]
    if missing:
        raise ValueError(
            f"log_troops is missing {len(missing)} side(s) of included battles, e.g. {missing[:5]}"
        )
    x = np.array([log_troops[int(s)] for s in design.side_ids], dtype=float)
    return np.asarray(x[design.side_a] - x[design.side_b])
