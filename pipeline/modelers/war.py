"""Wins Above Replacement from posterior draws (design section 6, R3, R12).

Per draw ``t``:

1. ``s_rep[t]`` is the ``replacement_percentile`` quantile of the draw's
   skills over the credited generals.
2. For every side, ``E_model`` is its expected score (``P(win) + 0.5 P(indecisive)``)
   at the fitted ``S_side[t]``; ``E_rep`` is the same with the side's whole
   score replaced by ``s_rep[t]`` (weights sum to one, so a replacement side
   has ``S = s_rep``), opponent, forces (each battle's own group slope) and
   defender term unchanged. The defender term is part of the side's context,
   not its skill: WAR keeps the side's attacking or defending posture and
   neither credits nor discounts it. A merged camp with an unnamed share
   replaces only its named part: ``S_rep = S - sum_i w_i skill_i + W s_rep``
   with ``W = sum_i w_i``, so the unnamed troops stay as fitted.
3. Expected WAR of a side is ``xWAR = E_model - E_rep``; realised WAR is
   ``y - E_rep`` with ``y`` the 0 / 0.5 / 1 score.
4. Commander ``i``'s contribution is ``w_i / W`` times the side's value, so a
   side's contributions sum exactly to its value; ``W`` is 1 on an ordinary
   side and the named share on a merged camp, so the unnamed part is credited
   to nobody.

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
        beta_force: ``(T,)`` or per-battle ``(T, n_battles)`` (see
            :meth:`pipeline.modelers.model.PosteriorDraws.force_slopes`).
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
    # (T,) for one slope, (T, n_battles) when battles carry their group's slope.
    slope = beta_force[:, None] if beta_force.ndim == 1 else beta_force
    force = slope * np.atleast_2d(x_diff)
    defender = beta_def[:, None] * design.defender_diff[None, :]
    score_a = side_scores[:, design.side_a]
    score_b = side_scores[:, design.side_b]
    c1_col, c2_col = c1[:, None], c2[:, None]
    named_weight = np.bincount(design.cmd_side, weights=design.cmd_weight, minlength=design.n_sides)
    named_part = np.zeros((n_draws, design.n_sides))
    for side, general, weight in zip(
        design.cmd_side, design.cmd_general, design.cmd_weight, strict=True
    ):
        named_part[:, side] += weight * skills[:, general]
    # A side with named commanders swaps out their part only; a wholly latent
    # side has none, so its whole score is replaced as before.
    has_named = named_weight > 0
    rep_score = np.where(
        has_named[None, :],
        side_scores - named_part + named_weight[None, :] * s_rep[:, None],
        s_rep[:, None],
    )
    rep_a = rep_score[:, design.side_a]
    rep_b = rep_score[:, design.side_b]

    # Eta from each side's own perspective; B's is the negation of A's.
    eta_a = score_a - score_b + force + defender
    eta_a_rep = rep_a - score_b + force + defender
    eta_b_rep = rep_b - score_a - force - defender

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

    # Split the side's value among its named commanders by their share of the
    # named part; on an ordinary side that is the weight itself.
    weights = (design.cmd_weight / named_weight[design.cmd_side])[None, :]
    xwar_entry = weights * xwar_side[:, design.cmd_side]
    # Realised = luck (y - E_model) + the expected part. The luck is the whole
    # camp's, so a general credited with a fraction w of it takes w, not the
    # whole of it: a 10%-named general must not absorb the camp's good fortune.
    # The expected part is split by share of the named part, as xwar is. On a
    # fully named side (W = 1) both reduce to the old share-times-realised.
    raw_weights = design.cmd_weight[None, :]
    luck = (y[None, :] - e_model)[:, design.cmd_side]
    realised_entry = raw_weights * luck + weights * xwar_side[:, design.cmd_side]
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
