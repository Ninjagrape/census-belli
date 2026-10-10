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


@dataclass(frozen=True)
class CommanderInput:
    """One commander credited on a side.

    Attributes:
        general_id: ``generals.general_id``.
        attribution_weight: The weight classify wrote; renormalised per side.
        attribution_method: ``rule_single``, ``rule_roles``, ``llm``,
            ``manual`` or ``default_split``.
    """

    general_id: int
    attribution_weight: float
    attribution_method: str


@dataclass(frozen=True)
class SideInput:
    """One ``battle_sides`` row with its commanders.

    Attributes:
        side_id: ``battle_sides.side_id``.
        battle_id: ``battle_sides.battle_id``.
        year: ``battles.year_astronomical`` (BC negative) or None.
        outcome: ``outcome_level`` text from this side's perspective, or None.
        is_defender: Whether the side defended, or None when unknown.
        commanders: Every ``battle_commanders`` row for the side.
    """

    side_id: int
    battle_id: int
    year: int | None
    outcome: str | None
    is_defender: bool | None
    commanders: tuple[CommanderInput, ...] = ()


@dataclass
class ModelCounts:
    """Exclusion and classification counters, logged and written to diagnostics."""

    battles_in_scope: int = 0
    battles_included: int = 0
    battles_multi_side: int = 0
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
        defender_diff: ``d_A - d_B`` with d = +1 defender, -1 attacker, 0
            unknown (after complement fill).
        defender_known: Whether either side's flag was known.
        defender_term_used: Whether any battle has a known flag, i.e.
            whether the model carries a defender coefficient.
        year: Battle year (float, NaN when undated).
        side_ids: Raw side id per side index, shape ``(n_sides,)``.
        side_kind: ``KIND_NAMED``, ``KIND_UNKNOWN`` or ``KIND_UNATTRIBUTED``.
        side_sign: +1 for an A side, -1 for a B side.
        cmd_side: Side index per commander entry (COO).
        cmd_general: General index per commander entry.
        cmd_weight: Renormalised weight per commander entry.
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
    year: np.ndarray
    side_ids: np.ndarray
    side_kind: np.ndarray
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
        """Side indices whose score is a latent rather than a skill sum."""
        return np.flatnonzero(self.side_kind != KIND_NAMED)

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
