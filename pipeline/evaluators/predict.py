"""Held-out prediction and scoring for the evaluate stage."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from pipeline.modelers.model import PosteriorDraws, joint_x_diff_prior
from pipeline.modelers.ordinal import N_LEVELS, PROB_FLOOR, category_probs, linear_predictor
from pipeline.modelers.records import ModelDesign

__all__ = [
    "binary_metrics",
    "force_only_baseline",
    "heldout_x_diff_draws",
    "log_score",
    "predict_category_probs",
    "ranked_probability_score",
    "wilson_interval",
]

_NAN_INTERVAL = (float("nan"), float("nan"))


def heldout_x_diff_draws(
    design: ModelDesign,
    obs_mean: np.ndarray,
    obs_sd: np.ndarray,
    sigma_w: float,
    n_draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``x_A - x_B`` for held-out battles from the measurement-only prior.

    The impute stage conditions on the outcome, so an imputed ``x_diff`` for a
    held-out battle already encodes its result and would leak it into the
    prediction. Prediction therefore uses the outcome-free prior of
    :func:`pipeline.modelers.model.joint_x_diff_prior`.

    Args:
        design: The design.
        obs_mean: Observed log-troop mean per side index, NaN where missing.
        obs_sd: Observed log-troop sd per side index, NaN where missing.
        sigma_w: Within-battle sd of log troops.
        n_draws: Number of draws.
        rng: Random generator.

    Returns:
        Array of shape ``(n_draws, n_battles)``.
    """
    mu, sd = joint_x_diff_prior(design, obs_mean, obs_sd, sigma_w)
    return np.asarray(mu + sd * rng.standard_normal((n_draws, design.n_battles)))


def predict_category_probs(
    design: ModelDesign,
    draws: PosteriorDraws,
    x_diff: np.ndarray,
    battle_idx: np.ndarray,
) -> np.ndarray:
    """Posterior predictive probabilities of the five levels for chosen battles.

    Args:
        design: The design.
        draws: Posterior draws.
        x_diff: ``(n_battles,)`` or ``(T, n_battles)``.
        battle_idx: Battle indices to predict.

    Returns:
        Array ``(len(battle_idx), 5)`` of A-perspective level probabilities
        (0 decisive defeat .. 4 decisive victory), averaged over draws.
    """
    eta = linear_predictor(
        design.side_a,
        design.side_b,
        draws.side_scores,
        x_diff,
        design.defender_diff,
        draws.beta_force,
        draws.beta_def,
    )
    eta = eta[:, battle_idx]
    probs = category_probs(eta, draws.c1[:, None], draws.c2[:, None])
    return np.asarray(probs.mean(axis=0))


def wilson_interval(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Args:
        successes: Number of successes (half credits allowed).
        n: Number of trials.
        z: Normal quantile.

    Returns:
        ``(lo, hi)``, or ``(nan, nan)`` when ``n == 0``.
    """
    if n == 0:
        return _NAN_INTERVAL
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _binary_split(k_lo: np.ndarray, k_hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Masks of battles whose interval lies wholly in {3,4} and wholly in {0,1}."""
    return np.asarray(k_lo) >= 3, np.asarray(k_hi) <= 1


def _favourite_calibration(p: np.ndarray, y: np.ndarray, n_bins: int) -> list[dict[str, Any]]:
    """Calibration bins over the favourite's probability on [0.5, 1.0].

    Side A is an arbitrary orientation (the lower side_id), so bins over
    ``P(A wins)`` mix model error with how the corpus lists combatants. The
    favourite's frame removes the orientation. A tie at 0.5 counts A as the
    favourite.
    """
    fav_p = np.maximum(p, 1.0 - p)
    fav_won = np.where(p >= 0.5, y, 1.0 - y)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    bin_of = np.minimum(((fav_p - 0.5) * 2.0 * n_bins).astype(int), n_bins - 1)
    out: list[dict[str, Any]] = []
    for b in range(n_bins):
        in_bin = bin_of == b
        count = int(in_bin.sum())
        out.append(
            {
                "lo": float(edges[b]),
                "hi": float(edges[b + 1]),
                "count": count,
                "mean_p": float(fav_p[in_bin].mean()) if count else float("nan"),
                "observed": float(fav_won[in_bin].mean()) if count else float("nan"),
            }
        )
    return out


def binary_metrics(
    probs: np.ndarray, k_lo: np.ndarray, k_hi: np.ndarray, n_bins: int = 5
) -> dict[str, Any]:
    """Accuracy, Brier score and calibration on decided battles.

    The binary set is battles whose observed interval lies wholly in
    ``{3, 4}`` (A won) or ``{0, 1}`` (A lost). ``p`` is ``P(3)+P(4)`` over
    ``P(0)+P(1)+P(3)+P(4)``.

    Args:
        probs: ``(n, 5)`` predicted level probabilities.
        k_lo: Lowest compatible level per battle.
        k_hi: Highest compatible level per battle.
        n_bins: Equal-width calibration bins on ``p``.

    Returns:
        ``{n, accuracy, accuracy_ci, brier, calibration, side_a_win_rate,
        calibration_favourite}``; accuracy counts a prediction of exactly 0.5 as
        half correct. ``calibration`` is a list of ``{lo, hi, count, mean_p,
        observed}`` dicts over ``P(A wins)``; ``calibration_favourite`` has the
        same fields over ``max(p, 1 - p)`` on [0.5, 1.0] with ``observed`` the
        share the favourite won, and is unchanged by swapping A and B.
        ``side_a_win_rate`` is the observed A-win share.
    """
    won, lost = _binary_split(k_lo, k_hi)
    mask = won | lost
    n = int(mask.sum())
    if n == 0:
        return {
            "n": 0,
            "accuracy": float("nan"),
            "accuracy_ci": _NAN_INTERVAL,
            "brier": float("nan"),
            "calibration": [],
            "side_a_win_rate": float("nan"),
            "calibration_favourite": [],
        }
    sub = probs[mask]
    y = won[mask].astype(float)
    win_mass = sub[:, 3] + sub[:, 4]
    decided = win_mass + sub[:, 0] + sub[:, 1]
    p = np.divide(win_mass, decided, out=np.full(n, 0.5), where=decided > 0)
    credit = np.where(p == 0.5, 0.5, ((p > 0.5) == (y == 1.0)).astype(float))
    correct = float(credit.sum())
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_of = np.minimum((p * n_bins).astype(int), n_bins - 1)
    calibration: list[dict[str, Any]] = []
    for b in range(n_bins):
        in_bin = bin_of == b
        count = int(in_bin.sum())
        calibration.append(
            {
                "lo": float(edges[b]),
                "hi": float(edges[b + 1]),
                "count": count,
                "mean_p": float(p[in_bin].mean()) if count else float("nan"),
                "observed": float(y[in_bin].mean()) if count else float("nan"),
            }
        )
    return {
        "n": n,
        "accuracy": correct / n,
        "accuracy_ci": wilson_interval(correct, n),
        "brier": float(np.mean((p - y) ** 2)),
        "calibration": calibration,
        "side_a_win_rate": float(y.mean()),
        "calibration_favourite": _favourite_calibration(p, y, n_bins),
    }


def ranked_probability_score(probs: np.ndarray, k: np.ndarray) -> float:
    """Mean ranked probability score over exact outcomes.

    Args:
        probs: ``(n, 5)`` predicted probabilities.
        k: Observed level per battle.

    Returns:
        Mean over battles of the squared cumulative error summed over the
        categories, divided by ``N_LEVELS - 1``; NaN for no battles.
    """
    if probs.shape[0] == 0:
        return float("nan")
    observed = (np.arange(N_LEVELS)[None, :] >= np.asarray(k)[:, None]).astype(float)
    err = (np.cumsum(probs, axis=1) - observed) ** 2
    return float(np.mean(err.sum(axis=1) / (N_LEVELS - 1)))


def log_score(probs: np.ndarray, k_lo: np.ndarray, k_hi: np.ndarray) -> float:
    """Mean log probability of each battle's observed interval.

    Args:
        probs: ``(n, 5)`` predicted probabilities.
        k_lo: Lowest compatible level.
        k_hi: Highest compatible level.

    Returns:
        Mean of ``log(max(P(k_lo..k_hi), PROB_FLOOR))``; NaN for no battles.
    """
    if probs.shape[0] == 0:
        return float("nan")
    cumulative = np.concatenate([np.zeros((probs.shape[0], 1)), np.cumsum(probs, axis=1)], axis=1)
    rows = np.arange(probs.shape[0])
    mass = cumulative[rows, np.asarray(k_hi) + 1] - cumulative[rows, np.asarray(k_lo)]
    return float(np.mean(np.log(np.maximum(mass, PROB_FLOOR))))


def force_only_baseline(
    x_diff_measured: np.ndarray, k_lo: np.ndarray, k_hi: np.ndarray
) -> dict[str, Any]:
    """Accuracy of "the larger army wins" on decided battles with a measured x_diff.

    Args:
        x_diff_measured: Measured ``x_A - x_B`` per battle, NaN where unmeasured.
        k_lo: Lowest compatible level.
        k_hi: Highest compatible level.

    Returns:
        ``{n, accuracy, accuracy_ci}``; ``x_diff == 0`` earns half credit.
    """
    won, lost = _binary_split(k_lo, k_hi)
    mask = (won | lost) & np.isfinite(x_diff_measured)
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "accuracy": float("nan"), "accuracy_ci": _NAN_INTERVAL}
    x = x_diff_measured[mask]
    credit = np.where(x == 0.0, 0.5, ((x > 0) == won[mask]).astype(float))
    correct = float(credit.sum())
    return {"n": n, "accuracy": correct / n, "accuracy_ci": wilson_interval(correct, n)}
