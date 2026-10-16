"""Plain records for the model stage: raw side rows in, aligned arrays out.

``SideInput`` is what the loader hands over (one row per ``battle_sides`` row
with its commanders). ``ModelDesign`` is what
:func:`pipeline.modelers.design.build_design` produces for the battles that
survive the exclusion rules, laid out so that every per-side quantity is one
vector indexed by a side index and every per-battle quantity one vector
indexed by a battle index.

Layout conventions, relied on everywhere else in the package:

- Battle ``b`` has side A at side index ``2b`` and side B at ``2b + 1``, where
  A is the lower ``side_id``. ``side_a`` and ``side_b`` carry those indices
  explicitly so callers never rely on the arithmetic.
- ``outcome`` is always from A's perspective, five ordered levels 0..4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np

__all__ = [
    "DEFAULT_SPLIT",
    "FORCE_GROUP_LABELS",
    "WATER_AIR_TYPES",
    "DesignParams",
    "KIND_NAMED",
    "KIND_UNATTRIBUTED",
    "KIND_UNKNOWN",
    "CommanderInput",
    "ModelCounts",
    "ModelDesign",
    "SideInput",
]

# attribution_method written by classify when it could only split a side's
# credit evenly among co-commanders. Such a side says nothing about who won it.
DEFAULT_SPLIT: Final[str] = "default_split"

KIND_NAMED: Final[int] = 0
KIND_UNKNOWN: Final[int] = 1
KIND_UNATTRIBUTED: Final[int] = 2

# Battle types whose force ratio plausibly carries a different slope than a
# land battle's (supply, concentration and detection work differently at sea
# and in the air). Every other type, including 'unknown', is land.
WATER_AIR_TYPES: Final[frozenset[str]] = frozenset({"naval", "aerial"})
FORCE_GROUP_LABELS: Final[tuple[str, str]] = ("land", "water_air")


@dataclass(frozen=True)
class DesignParams:
    """Thresholds that decide which optional terms the design carries.

    Attributes:
        min_defender_battles: The defender term is built only when at least this
            many included battles have a known, opposite flag pair.
        min_battles_per_force_group: The water/air force offset is built only
            when at least this many included battles are naval or aerial.
        defender_term: Forces the defender term on or off, ignoring the
            threshold, so a sensitivity subset keeps the main design's term
            decisions instead of re-deciding them on fewer battles. None
            applies the threshold.
        water_air_group: The same override for the water/air force offset.
    """

    min_defender_battles: int = 10
    min_battles_per_force_group: int = 10
    defender_term: bool | None = None
    water_air_group: bool | None = None


@dataclass(frozen=True)
class CommanderInput:
    """One commander credited on a side.

    Attributes:
        general_id: ``generals.general_id``.
        attribution_weight: The weight classify wrote; renormalised per side.
        attribution_method: ``rule_single``, ``rule_roles``, ``llm``,
            ``manual`` or ``default_split``.
        command_role: ``battle_commanders.command_role`` text, or None when
            the loader did not read it.
    """

    general_id: int
    attribution_weight: float
    attribution_method: str
    command_role: str | None = None


@dataclass(frozen=True)
class SideInput:
    """One ``battle_sides`` row with its commanders.

    Attributes:
        side_id: ``battle_sides.side_id``.
        battle_id: ``battle_sides.battle_id``.
        year: ``battles.year_astronomical`` (BC negative) or None.
        outcome: ``outcome_level`` text from this side's perspective, or None.
        is_defender: Whether the side defended, or None when unknown.
        battle_type: ``battles.battle_type`` text, or None when not read.
        log_troops: Reconcile's neutral log troop mean, or None (the model
            stage fills a merged camp member's gap with its mean over the
            imputed rounds). Used only to split a merged camp's credit among
            its member sides; the model's own troop values come from the
            imputed rounds.
        commanders: Every ``battle_commanders`` row for the side.
    """

    side_id: int
    battle_id: int
    year: int | None
    outcome: str | None
    is_defender: bool | None
    commanders: tuple[CommanderInput, ...] = ()
    battle_type: str | None = None
    log_troops: float | None = None


@dataclass
class ModelCounts:
    """Exclusion and classification counters, logged and written to diagnostics."""

    battles_in_scope: int = 0
    battles_included: int = 0
    battles_multi_side: int = 0
    battles_campaign: int = 0
    battles_merged_camps: int = 0
    battles_outcome_conflict: int = 0
    battles_no_outcome: int = 0
    battles_grade_disagree: int = 0
    battles_no_creditable_side: int = 0
    battles_general_both_sides: int = 0
    outcomes_unrecognised: int = 0
    sides_named: int = 0
    sides_unknown: int = 0
    sides_unattributed: int = 0
    # Over every input side, before exclusions: the denominators of the
    # exclusion_concentration gate.
    sides_multi_commander_all: int = 0
    sides_default_split_all: int = 0
    max_weight_deviation: float = 0.0
    # Included battles whose two flags are both stated and opposite, and those
    # where one flag was stated and the other filled with its complement; their
    # sum gates the defender term. Included naval or aerial battles gate the
    # water/air offset.
    defender_pairs_both_stated: int = 0
    defender_pairs_filled: int = 0
    battles_water_air: int = 0
    # Merged camps: how many, how many carry an unnamed share, and the
    # distribution of the share of the camp that named commanders are credited.
    camps_merged: int = 0
    camps_partial_credit: int = 0
    camp_credited_share_min: float = 1.0
    camp_credited_share_sum: float = 0.0

    @property
    def defender_battles_known(self) -> int:
        """Battles with a usable opposite flag pair, stated or filled."""
        return self.defender_pairs_both_stated + self.defender_pairs_filled

    @property
    def camp_credited_share_mean(self) -> float:
        """Mean credited share over merged camps (1.0 with none)."""
        return self.camp_credited_share_sum / self.camps_merged if self.camps_merged else 1.0

    @property
    def battles_excluded(self) -> int:
        """Battles in scope that did not enter the model."""
        return self.battles_in_scope - self.battles_included

    def as_dict(self) -> dict[str, int | float]:
        """Render the counters for a structlog call or a diagnostics payload.

        Returns:
            A flat mapping of counter name to value.
        """
        return {
            "battles_in_scope": self.battles_in_scope,
            "battles_included": self.battles_included,
            "battles_excluded": self.battles_excluded,
            "battles_multi_side": self.battles_multi_side,
            "battles_campaign": self.battles_campaign,
            "battles_merged_camps": self.battles_merged_camps,
            "defender_battles_known": self.defender_battles_known,
            "defender_pairs_both_stated": self.defender_pairs_both_stated,
            "defender_pairs_filled": self.defender_pairs_filled,
            "camps_merged": self.camps_merged,
            "camps_partial_credit": self.camps_partial_credit,
            "camp_credited_share_min": self.camp_credited_share_min,
            "camp_credited_share_mean": self.camp_credited_share_mean,
            "battles_water_air": self.battles_water_air,
            "battles_outcome_conflict": self.battles_outcome_conflict,
            "battles_no_outcome": self.battles_no_outcome,
            "battles_grade_disagree": self.battles_grade_disagree,
            "battles_no_creditable_side": self.battles_no_creditable_side,
            "battles_general_both_sides": self.battles_general_both_sides,
            "outcomes_unrecognised": self.outcomes_unrecognised,
            "sides_named": self.sides_named,
            "sides_unknown": self.sides_unknown,
            "sides_unattributed": self.sides_unattributed,
            "sides_multi_commander_all": self.sides_multi_commander_all,
            "sides_default_split_all": self.sides_default_split_all,
            "max_weight_deviation": self.max_weight_deviation,
        }


@dataclass(frozen=True)
class ModelDesign:
    """Aligned arrays for the included battles.

    Attributes:
        battle_ids: Raw battle id per battle, shape ``(n_battles,)``.
        side_a: Side index of A per battle.
        side_b: Side index of B per battle.
        k_lo: Lowest A-perspective level compatible with the outcome (0..4).
        k_hi: Highest compatible level; equal to ``k_lo`` when exact.
        y_a: A-perspective score, 0, 0.5 or 1.
        defender_diff: ``d_A - d_B`` with d = 1 defender, 0 attacker. When
            exactly one side's flag is stated the other is its complement; with
            both or neither stated the difference is 0 unless they are
            opposite (missing is no information, not "attacker"). All zeros
            when the term is not used.
        defender_known: Whether the pair is opposite, stated or filled.
        defender_term_used: Whether at least ``min_defender_battles`` battles
            are ``defender_known`` (or the params force it); otherwise the
            model has no defender coefficient at all.
        force_group: Per battle, 1 for naval or aerial when the water/air
            offset is built, else 0 (land, amphibious included).
        n_force_groups: 2 when the water/air offset is built, else 1.
        side_unknown_share: Share of each side's score that is an anonymous
            ``mu_unknown`` latent: 1 for an unknown side, the unnamed members'
            troop share for a merged camp, else 0.
        side_unattributed_share: The same for ``mu_unattributed``.
        side_members: Raw ``side_id`` s behind each side index. One entry for
            an ordinary side, several for a camp merged from a 3+ side battle.
        year: Battle year (float, NaN when undated).
        side_ids: Raw side id per side index (a merged camp's lowest member).
        side_kind: ``KIND_NAMED``, ``KIND_UNKNOWN`` or ``KIND_UNATTRIBUTED``
            (a camp with any named member is named).
        side_sign: +1 for an A side, -1 for a B side.
        cmd_side: Side index per commander entry (COO).
        cmd_general: General index per commander entry.
        cmd_weight: Weight per commander entry. Sums to one on an ordinary
            named side and to the named share of a merged camp.
        general_ids: Sorted ids of generals credited on at least one named
            side of an included battle.
        n_included_battles: Included battles in which each general is credited.
        n_uncredited_battles: Battles in which the general appears as a
            commander but earns no credit (unattributed side, or an excluded
            battle).
        n_excluded_multi_side: Excluded 3+ side battles per general (a
            subset of ``n_uncredited_battles``).
        effective_battles: Sum of weights over a general's included battles.
        counts: The exclusion counters.
    """

    battle_ids: np.ndarray
    side_a: np.ndarray
    side_b: np.ndarray
    k_lo: np.ndarray
    k_hi: np.ndarray
    y_a: np.ndarray
    defender_diff: np.ndarray
    defender_known: np.ndarray
    defender_term_used: bool
    force_group: np.ndarray
    n_force_groups: int
    year: np.ndarray
    side_ids: np.ndarray
    side_members: tuple[tuple[int, ...], ...]
    side_kind: np.ndarray
    side_unknown_share: np.ndarray
    side_unattributed_share: np.ndarray
    side_sign: np.ndarray
    cmd_side: np.ndarray
    cmd_general: np.ndarray
    cmd_weight: np.ndarray
    general_ids: np.ndarray
    n_included_battles: np.ndarray
    n_uncredited_battles: np.ndarray
    n_excluded_multi_side: np.ndarray
    effective_battles: np.ndarray
    counts: ModelCounts = field(default_factory=ModelCounts)

    @property
    def n_battles(self) -> int:
        """Number of included battles."""
        return int(self.battle_ids.shape[0])

    @property
    def n_sides(self) -> int:
        """Number of sides (two per battle)."""
        return int(self.side_ids.shape[0])

    @property
    def n_generals(self) -> int:
        """Number of credited generals."""
        return int(self.general_ids.shape[0])

    @property
    def latent_sides(self) -> np.ndarray:
        """Side indices whose score has an anonymous latent component."""
        return np.flatnonzero(self.side_unknown_share + self.side_unattributed_share > 0)

    def commander_padded(self) -> tuple[np.ndarray, np.ndarray]:
        """Commander entries as dense ``(n_sides, K)`` index and weight arrays.

        Padding has index 0 and weight 0, so ``(skill[index] * weight).sum(1)``
        is the named part of every side's score with no sparse algebra.

        Returns:
            ``(index, weight)``; ``K`` is at least 1.
        """
        width = 1
        if self.cmd_side.size:
            width = int(np.bincount(self.cmd_side, minlength=self.n_sides).max())
        index = np.zeros((self.n_sides, width), dtype=np.int64)
        weight = np.zeros((self.n_sides, width), dtype=float)
        filled = np.zeros(self.n_sides, dtype=np.int64)
        for side, general, w in zip(self.cmd_side, self.cmd_general, self.cmd_weight, strict=True):
            slot = filled[side]
            index[side, slot] = general
            weight[side, slot] = w
            filled[side] += 1
        return index, weight
