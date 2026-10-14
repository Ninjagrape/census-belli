"""Cross-validation folds over battles."""

from __future__ import annotations

import numpy as np

__all__ = ["assign_folds", "train_weight"]


def assign_folds(n_battles: int, k: int, seed: int) -> np.ndarray:
    """Assign each battle to one of ``k`` folds by a seeded permutation.

    Args:
        n_battles: Number of battles.
        k: Number of folds, at least 2 and at most ``n_battles``.
        seed: Seed of the permutation.

    Returns:
        Integer fold id per battle, shape ``(n_battles,)``; fold sizes differ
        by at most one.

    Raises:
        ValueError: If ``k < 2`` or ``k > n_battles``.
    """
    if k < 2 or k > n_battles:
        raise ValueError(f"k must be in [2, {n_battles}], got {k}")
    order = np.random.default_rng(seed).permutation(n_battles)
    folds = np.empty(n_battles, dtype=np.int64)
    folds[order] = np.arange(n_battles) % k
    return folds


def train_weight(folds: np.ndarray, held_out: int) -> np.ndarray:
    """Log-likelihood weights that drop one fold from training.

    Args:
        folds: Output of :func:`assign_folds`.
        held_out: The fold to hold out.

    Returns:
        Float array, 1.0 for training battles and 0.0 for the held-out fold.
    """
    return np.where(folds == held_out, 0.0, 1.0)
