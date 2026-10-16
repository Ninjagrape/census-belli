"""Rank comparison helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pipeline.evaluators.compare import min_required_overlap, rank_comparison, ranks_from_scores


def test_ranks_highest_first_with_stable_ties() -> None:
    ranks = ranks_from_scores(np.array([1.0, 3.0, 3.0, 2.0]))
    assert ranks.tolist() == [4, 1, 2, 3]


def test_identical_lists_agree_fully() -> None:
    ids = np.arange(30)
    score = np.linspace(0, 1, 30)
    out = rank_comparison(ids, score, ids, score, top_n=10)
    assert out["spearman_rho"] == pytest.approx(1.0)
    assert out["n_common"] == 30
    assert out["top_overlap"] == 10
    assert out["jaccard"] == 1.0
    assert out["only_in_a"] == [] and out["only_in_b"] == []


def test_reversed_lists_have_rho_minus_one() -> None:
    ids = np.arange(10)
    score = np.arange(10.0)
    out = rank_comparison(ids, score, ids, -score, top_n=3)
    assert out["spearman_rho"] == pytest.approx(-1.0)
    assert out["top_overlap"] == 0
    assert out["only_in_a"] == [7, 8, 9]
    assert out["only_in_b"] == [0, 1, 2]


def test_top_sets_use_full_lists_not_common_ids() -> None:
    ids_a = np.array([1, 2, 3, 4])
    ids_b = np.array([2, 3, 4, 99])
    out = rank_comparison(ids_a, np.array([4.0, 3, 2, 1]), ids_b, np.array([1.0, 2, 3, 9]), top_n=2)
    assert out["n_common"] == 3
    assert out["only_in_a"] == [1, 2]
    assert out["only_in_b"] == [4, 99]
    assert out["top_overlap"] == 0


def test_fewer_generals_than_top_n() -> None:
    ids = np.array([5, 6, 7])
    out = rank_comparison(ids, np.array([3.0, 2, 1]), ids, np.array([3.0, 1, 2]), top_n=20)
    assert out["top_n_effective"] == 3
    assert out["top_overlap"] == 3


def test_rho_nan_below_three_common() -> None:
    out = rank_comparison(
        np.array([1, 2]), np.array([1.0, 2.0]), np.array([1, 2]), np.array([2.0, 1.0])
    )
    assert math.isnan(out["spearman_rho"])
    assert out["n_common"] == 2


def test_rho_nan_for_constant_scores() -> None:
    ids = np.arange(5)
    out = rank_comparison(ids, np.zeros(5), ids, np.arange(5.0))
    assert math.isnan(out["spearman_rho"])


@pytest.mark.parametrize(("n", "expected"), [(20, 12), (10, 6), (3, 2), (0, 0), (7, 5)])
def test_min_required_overlap(n: int, expected: int) -> None:
    assert min_required_overlap(n) == expected


def test_max_rank_change_is_taken_over_the_reference_top_set() -> None:
    ids = np.arange(6)
    score_a = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    score_b = np.array([4.0, 5.0, 6.0, 3.0, 2.0, 1.0])  # id 0 falls from rank 1 to 3

    out = rank_comparison(ids, score_a, ids, score_b, top_n=3)

    assert out["max_abs_rank_change_top"] == 2
    assert rank_comparison(ids, score_a, ids, score_a, top_n=3)["max_abs_rank_change_top"] == 0
    assert rank_comparison(ids, score_a, ids + 100, score_a)["max_abs_rank_change_top"] is None
