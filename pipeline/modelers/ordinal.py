"""The ordinal logit in plain numpy, the reference the PyMC model must match.

Five levels 0..4 (decisive defeat .. decisive victory) from one side's
perspective, cumulative logit with symmetric cutpoints::

    c = (-c2, -c1, c1, c2),   P(k <= j) = expit(c_j - eta)

Symmetry gives ``P(k | eta) = P(4 - k | -eta)``: swapping the two sides is
exactly ``eta -> -eta, k -> 4 - k``, which is what makes the likelihood
independent of which side is called A.

The three-way probabilities (win, indecisive, loss) depend only on ``c1``;
``c2`` only splits each win or loss into decisive and ordinary.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from scipy.special import expit

__all__ = [
    "N_LEVELS",
    "PROB_FLOOR",
    "category_probs",
    "cutpoints",
    "expected_score",
    "interval_probability",
    "linear_predictor",
    "log_likelihood",
    "log_likelihood_pointwise",
    "ordinal_probs",
]

N_LEVELS: Final[int] = 5

# Shared with the PyMC likelihood so a rounding-level zero never gives -inf.
PROB_FLOOR: Final[float] = 1e-12


def cutpoints(c1: float, c2: float) -> np.ndarray:
    """The four symmetric cutpoints.

    Args:
        c1: Inner cutpoint, positive.
        c2: Outer cutpoint, greater than ``c1``.

    Returns:
        ``[-c2, -c1, c1, c2]``.
    """
    return np.array([-c2, -c1, c1, c2], dtype=float)


def ordinal_probs(
    eta: np.ndarray | float, c1: float | np.ndarray, c2: float | np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Win, indecisive and loss probabilities.

    ``c2`` is accepted so callers can pass the cutpoints uniformly; it does
    not change the three-way split.

    Args:
        eta: Linear predictor from the side's perspective.
        c1: Inner cutpoint.
        c2: Outer cutpoint.

    Returns:
        ``(P(win), P(indecisive), P(loss))``, with ``P_win(eta) == P_loss(-eta)``
        exactly.
    """
    del c2
    eta_arr = np.asarray(eta, dtype=float)
    p_win = expit(eta_arr - c1)
    p_loss = expit(-c1 - eta_arr)
    p_indecisive = 1.0 - p_win - p_loss
    return p_win, p_indecisive, p_loss


def expected_score(
    eta: np.ndarray | float, c1: float | np.ndarray, c2: float | np.ndarray
) -> np.ndarray:
    """Expected 0 / 0.5 / 1 score, ``P(win) + 0.5 P(indecisive)``.

    Args:
        eta: Linear predictor from the side's perspective.
        c1: Inner cutpoint.
        c2: Outer cutpoint.

    Returns:
        The expected score, in ``[0, 1]``; ``E(-eta) == 1 - E(eta)``.
    """
    p_win, p_indecisive, _ = ordinal_probs(eta, c1, c2)
    return np.asarray(p_win + 0.5 * p_indecisive)


def category_probs(
    eta: np.ndarray | float, c1: float | np.ndarray, c2: float | np.ndarray
) -> np.ndarray:
    """Probabilities of the five levels.

    Args:
        eta: Linear predictor, any shape.
        c1: Inner cutpoint (scalar or broadcastable to ``eta``).
        c2: Outer cutpoint.

    Returns:
        Array of shape ``eta.shape + (5,)``, rows summing to one.
    """
    eta_arr = np.asarray(eta, dtype=float)
    c1_arr = np.broadcast_to(np.asarray(c1, dtype=float), eta_arr.shape)
    c2_arr = np.broadcast_to(np.asarray(c2, dtype=float), eta_arr.shape)
    cdf = np.stack(
        [
            expit(-c2_arr - eta_arr),
            expit(-c1_arr - eta_arr),
            expit(c1_arr - eta_arr),
            expit(c2_arr - eta_arr),
        ],
        axis=-1,
    )
    zeros = np.zeros(eta_arr.shape + (1,))
    ones = np.ones(eta_arr.shape + (1,))
    padded = np.concatenate([zeros, cdf, ones], axis=-1)
    return np.asarray(np.diff(padded, axis=-1))


def interval_probability(
    eta: np.ndarray,
    k_lo: np.ndarray,
    k_hi: np.ndarray,
    c1: float,
    c2: float,
) -> np.ndarray:
    """``P(k_lo <= k <= k_hi)`` per battle; the exact case has ``k_lo == k_hi``.

    Args:
        eta: Linear predictor per battle (A's perspective).
        k_lo: Lowest compatible level.
        k_hi: Highest compatible level.
        c1: Inner cutpoint.
        c2: Outer cutpoint.

    Returns:
        Probability per battle.
    """
    probs = category_probs(eta, c1, c2)
    cumulative = np.concatenate([np.zeros((*probs.shape[:-1], 1)), np.cumsum(probs, axis=-1)], -1)
    rows = np.arange(probs.shape[0])
    return np.asarray(cumulative[rows, k_hi + 1] - cumulative[rows, k_lo], dtype=float)


def log_likelihood_pointwise(
    eta: np.ndarray, k_lo: np.ndarray, k_hi: np.ndarray, c1: float, c2: float
) -> np.ndarray:
    """Log probability of each battle's outcome, floored at ``PROB_FLOOR``.

    Args:
        eta: Linear predictor per battle.
        k_lo: Lowest compatible level.
        k_hi: Highest compatible level.
        c1: Inner cutpoint.
        c2: Outer cutpoint.

    Returns:
        Array of log probabilities.
    """
    return np.asarray(np.log(np.maximum(interval_probability(eta, k_lo, k_hi, c1, c2), PROB_FLOOR)))


def log_likelihood(
    eta: np.ndarray, k_lo: np.ndarray, k_hi: np.ndarray, c1: float, c2: float
) -> float:
    """Total log likelihood over battles.

    Args:
        eta: Linear predictor per battle.
        k_lo: Lowest compatible level.
        k_hi: Highest compatible level.
        c1: Inner cutpoint.
        c2: Outer cutpoint.

    Returns:
        The sum of :func:`log_likelihood_pointwise`.
    """
    return float(log_likelihood_pointwise(eta, k_lo, k_hi, c1, c2).sum())


def linear_predictor(
    side_a: np.ndarray,
    side_b: np.ndarray,
    side_scores: np.ndarray,
    x_diff: np.ndarray,
    defender_diff: np.ndarray,
    beta_force: float | np.ndarray,
    beta_def: float | np.ndarray,
) -> np.ndarray:
    """``eta = S_A - S_B + beta_force x_diff + beta_def d_diff``.

    ``side_scores`` may have a leading draw axis, in which case the result
    does too, and the betas may be per-draw vectors.

    Args:
        side_a: Side index of A per battle.
        side_b: Side index of B per battle.
        side_scores: ``S`` per side, ``(n_sides,)`` or ``(T, n_sides)``.
        x_diff: Log-troop difference per battle, ``(n_battles,)`` or
            ``(T, n_battles)``.
        defender_diff: ``d_A - d_B`` per battle.
        beta_force: Force coefficient, scalar or ``(T,)``.
        beta_def: Defender coefficient, scalar or ``(T,)``.

    Returns:
        ``eta`` with shape ``(n_battles,)`` or ``(T, n_battles)``.
    """
    scores = np.asarray(side_scores, dtype=float)
    force = np.asarray(beta_force, dtype=float)
    defender = np.asarray(beta_def, dtype=float)
    if scores.ndim == 2:
        force = force[:, None] if force.ndim == 1 else force
        defender = defender[:, None] if defender.ndim == 1 else defender
    return np.asarray(
        scores[..., side_a]
        - scores[..., side_b]
        + force * np.asarray(x_diff, dtype=float)
        + defender * defender_diff
    )
