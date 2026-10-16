"""Plain records shared across the impute stage.

No numpy-heavy logic and no database access lives here: these are the shapes
the loader produces, the likelihood step refines and the model consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np

__all__ = [
    "SCALES",
    "UNKNOWN",
    "BattleLikelihood",
    "ImputeCounts",
    "SideRow",
]

SCALES: Final[tuple[str, ...]] = ("neutral", "published")

# One spelling for "no value" in every categorical block, so a missing outcome
# and a missing battle type are the same level name across the model.
UNKNOWN: Final[str] = "unknown"


@dataclass(frozen=True)
class SideRow:
    """One in-scope side with every covariate the model reads.

    Attributes:
        side_id: ``battle_sides.side_id``.
        battle_id: ``battle_sides.battle_id``.
        year: ``battles.year_astronomical`` (BC is negative), or None when
            the battle is undated.
        outcome: ``outcome_level`` text, or None.
        battle_type: ``battle_type`` text, or None.
        is_defender: Whether the side defended, or None when unknown.
        polity_key: Wikidata id of the polity, else its string, else
            ``'unknown'``.
        commander_key: The primary commander's key, or ``'unknown'``.
        mnar: Whether this side's troop row in ``missing_data_log`` is
            classified ``mnar``.
        published: ``(estimate, lo, hi)`` from ``battle_sides`` on the
            exp scale, or None when reconcile wrote nothing. Only the
            published scale reads this.
    """

    side_id: int
    battle_id: int
    year: int | None
    outcome: str | None
    battle_type: str | None
    is_defender: bool | None
    polity_key: str
    commander_key: str
    mnar: bool = False
    published: tuple[float, float | None, float | None] | None = None


@dataclass(frozen=True)
class BattleLikelihood:
    """What reconcile says about the observed sides of one battle.

    The measurement model is ``m_lik ~ N(y_obs, inv(precision))``: ``y_obs``
    is the latent log troop count and ``precision`` the likelihood's
    precision, with reconcile's own side prior already divided out on the
    neutral scale.

    Attributes:
        battle_id: The battle.
        side_ids: Observed sides, in the row/column order of the arrays.
        m_lik: Likelihood mean per side, log scale, shape ``(k,)``.
        precision: Likelihood precision, shape ``(k, k)``, positive definite.
        method: ``'joint'`` (full matrix), ``'diagonal'`` (fallback) or
            ``'published'``.
        shift_loading: ``(k, p)`` response of ``m_lik`` to the ``p`` shared
            reconcile parameters, or None when the shared error is not
            propagated. A round that draws theta shifts ``m_lik`` by this
            times the centred theta.
    """

    battle_id: int
    side_ids: tuple[int, ...]
    m_lik: np.ndarray
    precision: np.ndarray
    method: str
    shift_loading: np.ndarray | None = None


@dataclass
class ImputeCounts:
    """Counters for one stage invocation, logged at the end."""

    sides_in_scope: int = 0
    sides_observed: int = 0
    sides_imputed: int = 0
    sides_weak_as_missing: int = 0
    sides_stale_in_draws: int = 0
    battles_diagonal_fallback: int = 0
    log_rows_updated: int = 0
    log_rows_inserted: int = 0
    extra: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, int]:
        """Render the counters for a structlog call.

        Returns:
            A flat mapping of counter name to value.
        """
        return {
            "sides_in_scope": self.sides_in_scope,
            "sides_observed": self.sides_observed,
            "sides_imputed": self.sides_imputed,
            "sides_weak_as_missing": self.sides_weak_as_missing,
            "sides_stale_in_draws": self.sides_stale_in_draws,
            "battles_diagonal_fallback": self.battles_diagonal_fallback,
            "log_rows_updated": self.log_rows_updated,
            "log_rows_inserted": self.log_rows_inserted,
            **self.extra,
        }
