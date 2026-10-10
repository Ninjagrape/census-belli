"""Wins Above Replacement from posterior draws (design section 6, R3, R12).

Per draw ``t``:

1. ``s_rep[t]`` is the ``replacement_percentile`` quantile of the draw's
   skills over the credited generals.
2. For every side, ``E_model`` is its expected score (``P(win) + 0.5 P(indecisive)``)
   at the fitted ``S_side[t]``; ``E_rep`` is the same with the side's whole
   score replaced by ``s_rep[t]`` (weights sum to one, so a replacement side
   has ``S = s_rep``), opponent, forces and defender term unchanged.
3. Expected WAR of a side is ``xWAR = E_model - E_rep``; realised WAR is
   ``y - E_rep`` with ``y`` the 0 / 0.5 / 1 score.
4. Commander ``i``'s contribution is ``w_i`` times the side's value, so a
   side's contributions sum exactly to its value.

All of this works in the sampler's own frame (raw, uncentred skills and the
raw ``S`` of latent sides): centring skills but not the latent sides' scores
would shift a named side against an unknown opponent and corrupt both
``E_model`` and ``E_rep``. :func:`centre_skills` is for reporting only.

Latent (unknown or unattributed) sides get a side value like any other but
no contributions, so a battle against one still gives its named opponent a
WAR that uses the drawn ``S`` of the latent side.

Arrays are per posterior draw, so a caller with many rounds should call
:func:`compute_war` once per round and reduce before pooling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pipeline.modelers.ordinal import expected_score
from pipeline.modelers.records import ModelDesign

__all__ = ["WarDraws", "centre_skills", "compute_war", "replacement_level"]

DEFAULT_REPLACEMENT_PERCENTILE = 40.0


@dataclass(frozen=True)
class WarDraws:
    """WAR quantities for every draw.

    Attributes:
        s_rep: Replacement skill per draw, ``(T,)``.
        e_model: Expected score per side at the fitted scores, ``(T, n_sides)``.
        e_rep: Expected score per side with the side replaced, ``(T, n_sides)``.
        xwar_side: ``e_model - e_rep``, ``(T, n_sides)``.
        realised_side: ``y - e_rep``, ``(T, n_sides)``.
        xwar_entry: Contribution per commander entry, ``(T, n_entries)``,
            aligned with ``design.cmd_side``.
        realised_entry: Realised contribution per entry.
        total_xwar: Sum of ``xwar_entry`` per general, ``(T, n_generals)``.
        total_realised: Sum of ``realised_entry`` per general.
    """

    s_rep: np.ndarray
    e_model: np.ndarray
    e_rep: np.ndarray
    xwar_side: np.ndarray
    realised_side: np.ndarray
    xwar_entry: np.ndarray
    realised_entry: np.ndarray
    total_xwar: np.ndarray
    total_realised: np.ndarray


def replacement_level(
    skills: np.ndarray, percentile: float = DEFAULT_REPLACEMENT_PERCENTILE
) -> np.ndarray:
    """Per-draw replacement skill.

    Args:
        skills: Raw skills, ``(T, n_generals)``.
        percentile: Percentile in 0..100 over the credited generals.

    Returns:
        Array of shape ``(T,)``.
    """
    return np.asarray(np.percentile(skills, percentile, axis=1))


def centre_skills(skills: np.ndarray) -> np.ndarray:
    """Subtract each draw's mean over generals (for reporting, not for WAR).

    Args:
        skills: ``(T, n_generals)``.

    Returns:
        Centred skills of the same shape.
    """
    return np.asarray(skills - skills.mean(axis=1, keepdims=True))


def compute_war(
    design: ModelDesign,
    skills: np.ndarray,
    side_scores: np.ndarray,
    x_diff: np.ndarray,
    beta_force: np.ndarray,
    beta_def: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
    replacement_percentile: float = DEFAULT_REPLACEMENT_PERCENTILE,
) -> WarDraws:
    """Expected and realised WAR for every draw.

    Args:
        design: The design.
        skills: Raw skills ``(T, n_generals)``.
        side_scores: ``S`` per side ``(T, n_sides)`` including latent sides.
        x_diff: ``x_A - x_B``, ``(n_battles,)`` or ``(T, n_battles)``.
        beta_force: ``(T,)``.
        beta_def: ``(T,)``; ignored by a design without defender data (its
            ``defender_diff`` is zero).
        c1: ``(T,)``.
        c2: ``(T,)``.
        replacement_percentile: Percentile defining replacement level.

    Returns:
        The per-draw WAR quantities.
    """
    n_draws = skills.shape[0]
    s_rep = replacement_level(skills, replacement_percentile)
    force = beta_force[:, None] * np.atleast_2d(x_diff)
    defender = beta_def[:, None] * design.defender_diff[None, :]
    score_a = side_scores[:, design.side_a]
    score_b = side_scores[:, design.side_b]
    c1_col, c2_col = c1[:, None], c2[:, None]

    # Eta from each side's own perspective; B's is the negation of A's.
    eta_a = score_a - score_b + force + defender
    eta_a_rep = s_rep[:, None] - score_b + force + defender
    eta_b_rep = s_rep[:, None] - score_a - force - defender

    e_model = np.empty((n_draws, design.n_sides))
    e_rep = np.empty((n_draws, design.n_sides))
    e_model[:, design.side_a] = expected_score(eta_a, c1_col, c2_col)
    e_model[:, design.side_b] = expected_score(-eta_a, c1_col, c2_col)
    e_rep[:, design.side_a] = expected_score(eta_a_rep, c1_col, c2_col)
    e_rep[:, design.side_b] = expected_score(eta_b_rep, c1_col, c2_col)

    y = np.empty(design.n_sides)
    y[design.side_a] = design.y_a
    y[design.side_b] = 1.0 - design.y_a
    xwar_side = e_model - e_rep
    realised_side = y[None, :] - e_rep

    weights = design.cmd_weight[None, :]
    xwar_entry = weights * xwar_side[:, design.cmd_side]
    realised_entry = weights * realised_side[:, design.cmd_side]
    total_xwar = np.zeros((n_draws, design.n_generals))
    total_realised = np.zeros((n_draws, design.n_generals))
    for entry, general in enumerate(design.cmd_general):
        total_xwar[:, general] += xwar_entry[:, entry]
        total_realised[:, general] += realised_entry[:, entry]

    return WarDraws(
        s_rep=s_rep,
        e_model=e_model,
        e_rep=e_rep,
        xwar_side=xwar_side,
        realised_side=realised_side,
        xwar_entry=xwar_entry,
        realised_entry=realised_entry,
        total_xwar=total_xwar,
        total_realised=total_realised,
    )
