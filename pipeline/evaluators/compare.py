"""Rank-list comparison helpers for the sensitivity and expert-ranking checks."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats

__all__ = ["min_required_overlap", "rank_comparison", "ranks_from_scores"]


def ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    """Rank scores so that 1 is the highest, ties broken by input order.

    Args:
        scores: Higher is better. NaN sorts last.

    Returns:
        Integer ranks 1..n, same length as ``scores``.
    """
    values = np.asarray(scores, dtype=float)
    key = np.where(np.isnan(values), -np.inf, values)
    order = np.argsort(-key, kind="stable")
    ranks = np.empty(values.shape[0], dtype=np.int64)
    ranks[order] = np.arange(1, values.shape[0] + 1)
    return ranks


def _top_ids(ids: np.ndarray, score: np.ndarray, n: int) -> set[int]:
    order = np.argsort(ranks_from_scores(score), kind="stable")[:n]
    return {int(i) for i in np.asarray(ids)[order]}


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    # scipy warns and returns NaN on constant input; skip that path.
    if x.shape[0] < 3 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return math.nan
    return float(stats.spearmanr(x, y).statistic)


def rank_comparison(
    ids_a: np.ndarray,
    score_a: np.ndarray,
    ids_b: np.ndarray,
    score_b: np.ndarray,
    *,
    top_n: int = 20,
) -> dict[str, Any]:
    """Compare two scored lists of generals.

    Args:
        ids_a: Ids of list A (unique).
        score_a: Scores of list A, higher is better.
        ids_b: Ids of list B (unique).
        score_b: Scores of list B.
        top_n: Size of the top set compared.

    Returns:
        Mapping with ``spearman_rho`` (NaN under three common generals),
        ``n_common``, ``top_n_effective``, ``top_overlap``, ``jaccard`` and the
        sorted id lists ``only_in_a`` / ``only_in_b``. Top sets are taken over
        each full list, so those two are top-set members absent from the other
        top set.
    """
    ids_a, ids_b = np.asarray(ids_a), np.asarray(ids_b)
    score_a, score_b = np.asarray(score_a, dtype=float), np.asarray(score_b, dtype=float)
    _, ia, ib = np.intersect1d(ids_a, ids_b, return_indices=True)
    rho = _spearman(score_a[ia], score_b[ib])

    effective = min(top_n, ids_a.shape[0], ids_b.shape[0])
    top_a = _top_ids(ids_a, score_a, effective)
    top_b = _top_ids(ids_b, score_b, effective)
    union = top_a | top_b
    return {
        "spearman_rho": rho,
        "n_common": int(ia.shape[0]),
        "top_n_effective": effective,
        "top_overlap": len(top_a & top_b),
        "jaccard": len(top_a & top_b) / len(union) if union else 0.0,
        "only_in_a": sorted(top_a - top_b),
        "only_in_b": sorted(top_b - top_a),
    }


def min_required_overlap(top_n_effective: int, base_top_n: int = 20, base_min: int = 12) -> int:
    """Scale the minimum top-N overlap to a shorter list.

    Args:
        top_n_effective: Size of the top set actually compared.
        base_top_n: The top-N the base threshold was set for.
        base_min: Required overlap at ``base_top_n``.

    Returns:
        ``ceil(base_min * top_n_effective / base_top_n)``.
    """
    return math.ceil(base_min * top_n_effective / base_top_n)
