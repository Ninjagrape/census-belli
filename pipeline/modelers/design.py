"""From raw side rows to the model's aligned arrays.

Implements design sections 1, 2, 4, 5 and revisions R1, R5, R12.

Rules, in the order they are applied to each battle:

0. ``battle_type == 'campaign'`` (a war or multi-month operation that slipped
   into the battle corpus) is excluded (``battles_campaign``) and its
   commanders are not tallied as uncredited: the battle was never a candidate.
1. Exactly two sides, else excluded (``battles_multi_side``; also catches a
   one-side battle, which has no opponent). The exception is a battle with 3+
   sides whose outcomes split cleanly into two camps, every side of one
   victory-like and every side of the other defeat-like: each camp becomes one
   side (``battles_merged_camps``). An indecisive or missing outcome on any
   side leaves the battle excluded, because the camp boundary is then a guess.
2. Orientation: A is the lower ``side_id`` (of a camp: its lowest member).
   Insertion order carries no information, so nothing downstream may depend on
   it (the symmetric cutpoints in :mod:`pipeline.modelers.ordinal` and the
   difference form of every other term guarantee that).
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
   A camp is named when any member is: commanders are the union, each weight
   is the member's own weight times that member's share of the camp's troops
   (reconcile's neutral estimate, a member without one using its mean over the
   imputed rounds; equal shares only if no member has any value). Named
   weights sum to the named share, not to one: the troops of unknown or
   unattributed members stay in the camp as a latent component
   (``side_unknown_share`` / ``side_unattributed_share``) and are credited to
   nobody, so a camp that is 10% named and 90% unknown credits its named
   general with 0.1 of the camp. With no named member the camp is unattributed
   when every member is, else unknown (its shares still record the mix).
   Disclosure: a member's missing troops are filled from the mean imputed
   troops, and the imputed rounds condition on the outcome, so the camp
   credit shares leak a little outcome into the attribution weights.
   Variants that use observed troops only are unaffected.
5. A battle with no named side (``battles_no_creditable_side``) or with the
   same general on both named sides (``battles_general_both_sides``) is
   excluded.
6. Defender flags: d = 1 defender, 0 attacker. The classify prompt defines a
   side's flag relative to the other side, so when exactly one flag is stated
   the other is filled with its complement (``defender_pairs_filled``); both
   stated and opposite counts as ``defender_pairs_both_stated``. Neither
   stated, or both stated alike, contributes 0 (missing means no information,
   not "attacker"). The term exists only when the two counts together reach
   ``min_defender_battles``. A camp is a defender if any member is (so a camp
   defends when any member defends), an attacker if every member is known and
   none defends, else unknown. The model's WAR counterfactual keeps the
   side's defender posture: it neither credits nor discounts attacking or
   defending.
7. Force slope: naval and aerial battles get a water/air offset
   (``delta_water_air``) from the land slope when they number at least
   ``min_battles_per_force_group`` included battles; otherwise one slope
   serves every battle. Amphibious counts as land.

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
    WATER_AIR_TYPES,
    DesignParams,
    ModelCounts,
    ModelDesign,
    SideInput,
)

__all__ = [
    "OUTCOME_LEVEL",
    "build_design",
    "camp_measurements",
    "outcome_interval",
    "score_of_interval",
    "side_log_troops",
    "x_difference",
]

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
_CAMPAIGN: Final[str] = "campaign"
_PAIR_BOTH: Final[str] = "both"
_PAIR_FILLED: Final[str] = "filled"
_PAIR_NONE: Final[str] = "none"


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


def _defender_codes(a: bool | None, b: bool | None) -> tuple[int, int, str]:
    """Code the flags 1 defender / 0 attacker, filling a lone stated flag.

    The classify prompt defines each flag relative to the other side, so one
    stated flag implies the other as its complement. Neither stated is a gap
    (``(0, 0)``), not "attacker".

    Returns:
        ``(d_a, d_b, status)`` with status ``"both"`` (stated and opposite),
        ``"filled"`` (one stated, the other its complement) or ``"none"`` (the
        difference is 0: neither stated, or stated alike).
    """
    if a is None and b is None:
        return 0, 0, _PAIR_NONE
    if b is None:
        return int(bool(a)), 1 - int(bool(a)), _PAIR_FILLED
    if a is None:
        return 1 - int(b), int(b), _PAIR_FILLED
    if a == b:
        return int(a), int(b), _PAIR_NONE
    return int(a), int(b), _PAIR_BOTH


@dataclass(frozen=True)
class _ResolvedSide:
    kind: int
    commanders: tuple[tuple[int, float], ...]  # (general_id, weight)
    deviation: float
    unknown_share: float = 0.0  # score share that is the anonymous mu_unknown latent
    unattributed_share: float = 0.0  # ... and the mu_unattributed latent


def _classify_side(side: SideInput) -> _ResolvedSide:
    """Decide how a side enters the model and renormalise its weights."""
    merged: dict[int, float] = defaultdict(float)
    for cmd in side.commanders:
        weight = cmd.attribution_weight
        if math.isfinite(weight) and weight > 0.0:
            merged[cmd.general_id] += weight
    total = sum(merged.values())
    if not merged or total <= 0.0:
        return _ResolvedSide(KIND_UNKNOWN, (), 0.0, unknown_share=1.0)
    if len(merged) >= 2 and all(c.attribution_method == DEFAULT_SPLIT for c in side.commanders):
        return _ResolvedSide(KIND_UNATTRIBUTED, (), 0.0, unattributed_share=1.0)
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
class _Camp:
    """One side of an included battle: a single side or several merged ones."""

    side_id: int  # the lowest member's id, the camp's stable key
    members: tuple[int, ...]
    resolved: _ResolvedSide
    general_ids: frozenset[int]  # every commander on any member, credited or not


@dataclass(frozen=True)
class _Row:
    """One included battle, before it is flattened into arrays."""

    battle_id: int
    camps: tuple[_Camp, _Camp]
    k_lo: int
    k_hi: int
    d_a: int
    d_b: int
    pair_status: str
    year: float
    battle_type: str | None
    merged: bool


@dataclass
class _Tally:
    """Per-general appearance counts, keyed by raw general id."""

    included: dict[int, int]
    uncredited: dict[int, int]
    excluded_multi: dict[int, int]
    effective: dict[int, float]


def _split_camps(
    group: list[SideInput], counts: ModelCounts
) -> tuple[list[SideInput], list[SideInput], list[int]] | None:
    """Split a 3+ side battle into two camps by outcome, or None if not clean.

    Clean means every side has a recognised outcome that is a win or a loss,
    with at least one of each. The camp holding the lowest ``side_id`` comes
    first.

    Returns:
        ``(camp_a, camp_b, levels)`` with ``levels`` aligned to ``group``.
    """
    maybe_levels = [_level_of(side.outcome, counts) for side in group]
    levels = [lv for lv in maybe_levels if lv is not None]
    if len(levels) != len(group):
        return None
    classes = [_level_class(lv) for lv in levels]
    if 0 in classes or len(set(classes)) != 2:
        return None
    camp_a = [s for s, c in zip(group, classes, strict=True) if c == classes[0]]
    camp_b = [s for s, c in zip(group, classes, strict=True) if c != classes[0]]
    return camp_a, camp_b, levels


def _camp_defender(members: Sequence[SideInput]) -> bool | None:
    """A camp defends if any member does; it attacks only if all are known."""
    flags = [m.is_defender for m in members]
    if any(f is True for f in flags):
        return True
    if all(f is False for f in flags):
        return False
    return None


def _camp_shares(members: Sequence[SideInput]) -> list[float]:
    """Each member's share of the camp's troops.

    Equal shares only when no member has a value; a member without one while
    others have takes the mean of the others' logs (the model stage fills such
    gaps from the imputed rounds before the design is built, so this is a last
    resort rather than the usual path).
    """
    logs = [m.log_troops for m in members]
    finite = [x for x in logs if x is not None and math.isfinite(x)]
    if not finite:
        return [1.0 / len(members)] * len(members)
    fill = sum(finite) / len(finite)
    values = [x if x is not None and math.isfinite(x) else fill for x in logs]
    top = max(values)
    raw = [math.exp(x - top) for x in values]
    total = sum(raw)
    return [r / total for r in raw]


def _merge_camp(members: Sequence[SideInput]) -> _Camp:
    """Combine member sides into one camp (a single member passes through).

    Named members' commanders keep ``share * weight``, so the camp's named
    weights sum to the named share. Unknown and unattributed members' shares
    are carried as latent shares, never given to the named commanders.
    """
    resolved = [_classify_side(m) for m in members]
    member_ids = tuple(sorted(m.side_id for m in members))
    general_ids = frozenset(c.general_id for m in members for c in m.commanders)
    if len(members) == 1:
        return _Camp(member_ids[0], member_ids, resolved[0], general_ids)
    shares = _camp_shares(members)
    weights: dict[int, float] = defaultdict(float)
    unknown = unattributed = 0.0
    for res, share in zip(resolved, shares, strict=True):
        if res.kind == KIND_NAMED:
            for gid, w in res.commanders:
                weights[gid] += share * w
        elif res.kind == KIND_UNKNOWN:
            unknown += share
        else:
            unattributed += share
    deviation = max(r.deviation for r in resolved)
    if weights:
        commanders = tuple(sorted(weights.items()))
        merged = _ResolvedSide(KIND_NAMED, commanders, deviation, unknown, unattributed)
    else:
        kind = KIND_UNATTRIBUTED if unknown == 0.0 else KIND_UNKNOWN
        merged = _ResolvedSide(kind, (), 0.0, unknown, unattributed)
    return _Camp(member_ids[0], member_ids, merged, general_ids)


def _camp_interval(a_levels: Sequence[int], b_levels: Sequence[int]) -> tuple[int, int]:
    """A-perspective bounds from every member's level; classes agree by construction."""
    levels = [*a_levels, *(_TOP_LEVEL - lv for lv in b_levels)]
    return min(levels), max(levels)


def _exclusion_reason(group: list[SideInput], counts: ModelCounts) -> str | _Row:
    """Apply rules 0 to 5 to one battle.

    Returns:
        The name of the ``ModelCounts`` counter to increment when the battle is
        excluded, otherwise the resolved row.
    """
    battle_type = group[0].battle_type
    if battle_type == _CAMPAIGN:
        return "battles_campaign"
    if len(group) < 2:
        return "battles_multi_side"
    merged = len(group) >= 3
    interval: tuple[int, int] | str
    if merged:
        split = _split_camps(group, counts)
        if split is None:
            return "battles_multi_side"
        members_a, members_b, levels = split
        level_of = {s.side_id: lv for s, lv in zip(group, levels, strict=True)}
        interval = _camp_interval(
            [level_of[s.side_id] for s in members_a], [level_of[s.side_id] for s in members_b]
        )
    else:
        members_a, members_b = [group[0]], [group[1]]
        interval = outcome_interval(
            _level_of(group[0].outcome, counts), _level_of(group[1].outcome, counts)
        )
    if interval == "no_outcome":
        return "battles_no_outcome"
    if isinstance(interval, str):
        return "battles_outcome_conflict"
    camp_a, camp_b = _merge_camp(members_a), _merge_camp(members_b)
    resolved = (camp_a.resolved, camp_b.resolved)
    if all(r.kind != KIND_NAMED for r in resolved):
        return "battles_no_creditable_side"
    if all(r.kind == KIND_NAMED for r in resolved) and {g for g, _ in resolved[0].commanders} & {
        g for g, _ in resolved[1].commanders
    }:
        return "battles_general_both_sides"
    d_a, d_b, pair_status = _defender_codes(_camp_defender(members_a), _camp_defender(members_b))
    years = [s.year for s in group if s.year is not None]
    return _Row(
        battle_id=group[0].battle_id,
        camps=(camp_a, camp_b),
        k_lo=interval[0],
        k_hi=interval[1],
        d_a=d_a,
        d_b=d_b,
        pair_status=pair_status,
        year=float(years[0]) if years else math.nan,
        battle_type=battle_type,
        merged=merged,
    )


def _tally_row(row: _Row, counts: ModelCounts, tally: _Tally) -> None:
    """Count an included battle's sides and credit its generals."""
    counts.battles_included += 1
    counts.battles_merged_camps += int(row.merged)
    if row.k_lo != row.k_hi:
        counts.battles_grade_disagree += 1
    for camp in row.camps:
        res = camp.resolved
        counts.max_weight_deviation = max(counts.max_weight_deviation, res.deviation)
        merged_camp = len(camp.members) > 1
        if res.kind == KIND_NAMED:
            counts.sides_named += 1
            credited = 0.0
            for gid, w in res.commanders:
                tally.included[gid] += 1
                tally.effective[gid] += w
                credited += w
            if merged_camp:
                _count_camp_share(counts, credited)
                # Commanders of a merged member that was not itself creditable.
                # An ordinary side's zero-weight commanders are not counted.
                for gid in camp.general_ids - {g for g, _ in res.commanders}:
                    tally.uncredited[gid] += 1
            continue
        if merged_camp:
            _count_camp_share(counts, 0.0)
        if res.kind == KIND_UNKNOWN:
            counts.sides_unknown += 1
        else:
            counts.sides_unattributed += 1
        # Present on the camp, credited with nothing.
        for gid in camp.general_ids:
            tally.uncredited[gid] += 1


def _count_camp_share(counts: ModelCounts, credited: float) -> None:
    """Record the share of a merged camp that its named commanders are credited."""
    counts.camps_merged += 1
    counts.camps_partial_credit += int(credited < 1.0 - 1e-9)
    counts.camp_credited_share_min = min(counts.camp_credited_share_min, credited)
    counts.camp_credited_share_sum += credited


def build_design(sides: Sequence[SideInput], params: DesignParams | None = None) -> ModelDesign:
    """Apply the inclusion rules and lay the survivors out as arrays.

    Input order is irrelevant: battles are processed in ``battle_id`` order
    and sides within a battle in ``side_id`` order.

    Args:
        sides: Every in-scope ``battle_sides`` row with its commanders.
        params: Thresholds for the optional defender and water/air terms;
            defaults when None.

    Returns:
        The design; ``design.counts`` holds the exclusion counters. With no
        included battle every array is empty.
    """
    params = params or DesignParams()
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
            if reason == "battles_campaign":
                continue
            for gid in {c.general_id for s in group for c in s.commanders}:
                tally.uncredited[gid] += 1
                if reason == "battles_multi_side":
                    tally.excluded_multi[gid] += 1
            continue
        rows.append(reason)
        _tally_row(reason, counts, tally)

    return _assemble(rows, counts, tally, params)


def _assemble(
    rows: list[_Row], counts: ModelCounts, tally: _Tally, params: DesignParams
) -> ModelDesign:
    """Turn the per-battle records into the aligned arrays."""
    n = len(rows)
    general_ids = np.array(sorted(tally.included), dtype=np.int64)
    index_of = {int(g): i for i, g in enumerate(general_ids)}

    side_ids = np.zeros(2 * n, dtype=np.int64)
    side_kind = np.zeros(2 * n, dtype=np.int64)
    unknown_share = np.zeros(2 * n)
    unattributed_share = np.zeros(2 * n)
    side_members: list[tuple[int, ...]] = []
    cmd_side: list[int] = []
    cmd_general: list[int] = []
    cmd_weight: list[float] = []

    for b, row in enumerate(rows):
        for offset in (0, 1):
            side_index = 2 * b + offset
            camp = row.camps[offset]
            side_ids[side_index] = camp.side_id
            side_kind[side_index] = camp.resolved.kind
            unknown_share[side_index] = camp.resolved.unknown_share
            unattributed_share[side_index] = camp.resolved.unattributed_share
            side_members.append(camp.members)
            for gid, w in camp.resolved.commanders:
                cmd_side.append(side_index)
                cmd_general.append(index_of[gid])
                cmd_weight.append(w)

    known = np.array([r.pair_status != _PAIR_NONE for r in rows], dtype=bool)
    water_air = np.array([r.battle_type in WATER_AIR_TYPES for r in rows], dtype=bool)
    counts.defender_pairs_both_stated = sum(1 for r in rows if r.pair_status == _PAIR_BOTH)
    counts.defender_pairs_filled = sum(1 for r in rows if r.pair_status == _PAIR_FILLED)
    counts.battles_water_air = int(water_air.sum())
    # An explicit override keeps a sensitivity subset on the main design's term
    # decisions; the thresholds are for the full corpus.
    defender_used = (
        counts.defender_battles_known >= params.min_defender_battles
        if params.defender_term is None
        else params.defender_term
    )
    two_groups = (
        counts.battles_water_air >= params.min_battles_per_force_group
        if params.water_air_group is None
        else params.water_air_group
    )
    d_diff = np.array([r.d_a - r.d_b for r in rows], dtype=float)

    def per_general(source: Mapping[int, int] | Mapping[int, float], dtype: type) -> np.ndarray:
        return np.array([source.get(int(g), 0) for g in general_ids], dtype=dtype)

    return ModelDesign(
        battle_ids=np.array([r.battle_id for r in rows], dtype=np.int64),
        side_a=2 * np.arange(n, dtype=np.int64),
        side_b=2 * np.arange(n, dtype=np.int64) + 1,
        k_lo=np.array([r.k_lo for r in rows], dtype=np.int64),
        k_hi=np.array([r.k_hi for r in rows], dtype=np.int64),
        y_a=np.array([score_of_interval(r.k_lo, r.k_hi) for r in rows], dtype=float),
        # Zeros when the term is dropped, so downstream arithmetic is inert;
        # defender_term_used tells the model not to build a coefficient at all.
        defender_diff=d_diff if defender_used else np.zeros(n),
        defender_known=known,
        defender_term_used=defender_used,
        force_group=(water_air & two_groups).astype(np.int64),
        n_force_groups=2 if two_groups else 1,
        year=np.array([r.year for r in rows], dtype=float),
        side_ids=side_ids,
        side_members=tuple(side_members),
        side_kind=side_kind,
        side_unknown_share=unknown_share,
        side_unattributed_share=unattributed_share,
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


def side_log_troops(design: ModelDesign, log_troops: Mapping[int, float]) -> np.ndarray:
    """Log troops per side index; a camp is the log-sum-exp of its members.

    Args:
        design: The design.
        log_troops: Log troop count for every raw side in the design. The
            caller guarantees completeness (an imputed round has a value for
            every in-scope side).

    Returns:
        Array of shape ``(n_sides,)``.

    Raises:
        ValueError: If any member side of an included battle has no value.
    """
    missing = [m for members in design.side_members for m in members if m not in log_troops]
    if missing:
        raise ValueError(
            f"log_troops is missing {len(missing)} side(s) of included battles, e.g. {missing[:5]}"
        )
    return np.array(
        [
            log_troops[members[0]]
            if len(members) == 1
            else np.logaddexp.reduce([log_troops[m] for m in members])
            for members in design.side_members
        ],
        dtype=float,
    )


def x_difference(design: ModelDesign, log_troops: Mapping[int, float]) -> np.ndarray:
    """``x_A - x_B`` per battle from a mapping of ``side_id`` to log troops.

    Args:
        design: The design.
        log_troops: Log troop count for every side in the design (see
            :func:`side_log_troops`).

    Returns:
        Array of shape ``(n_battles,)``.

    Raises:
        ValueError: If any side of an included battle has no value.
    """
    x = side_log_troops(design, log_troops)
    return np.asarray(x[design.side_a] - x[design.side_b])


def camp_measurements(
    design: ModelDesign, measured: Mapping[int, tuple[float, float]]
) -> tuple[np.ndarray, np.ndarray]:
    """Measured log-troop mean and sd per side index (NaN where unmeasured).

    A merged camp is measured only when every member is: its mean is the
    log-sum-exp of the members' and its sd the delta-method
    ``sqrt(sum(share_i^2 sd_i^2))`` (the derivative of a log-sum-exp with
    respect to member i's log is that member's share). The formula assumes the
    members' errors are independent, which understates the sd where they share
    a source or an estimator. One unmeasured member leaves the camp unmeasured
    rather than understating it.

    Args:
        design: The design.
        measured: ``side_id -> (log_mean, log_sd)`` as loaded.

    Returns:
        ``(obs_mean, obs_sd)``, each of shape ``(n_sides,)``.
    """
    obs_mean = np.full(design.n_sides, np.nan)
    obs_sd = np.full(design.n_sides, np.nan)
    for i, members in enumerate(design.side_members):
        if not all(m in measured for m in members):
            continue
        means = np.array([measured[m][0] for m in members])
        sds = np.array([measured[m][1] for m in members])
        if len(members) == 1:
            obs_mean[i], obs_sd[i] = means[0], sds[0]
            continue
        obs_mean[i] = np.logaddexp.reduce(means)
        if np.isfinite(sds).all():
            shares = np.exp(means - obs_mean[i])
            obs_sd[i] = float(np.sqrt(np.sum((shares * sds) ** 2)))
    return obs_mean, obs_sd
