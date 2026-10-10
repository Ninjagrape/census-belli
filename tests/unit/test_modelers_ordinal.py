"""The numpy ordinal logit."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.modelers.ordinal import (
    category_probs,
    expected_score,
    interval_probability,
    log_likelihood,
    ordinal_probs,
)


def test_probabilities_sum_to_one() -> None:
    eta = np.linspace(-4, 4, 17)
    win, ind, loss = ordinal_probs(eta, 0.5, 2.0)
    assert np.allclose(win + ind + loss, 1.0)
    assert np.allclose(category_probs(eta, 0.5, 2.0).sum(-1), 1.0)
    assert (ind >= 0).all()


def test_win_loss_symmetry() -> None:
    eta = np.linspace(-3, 3, 13)
    win, ind, loss = ordinal_probs(eta, 0.5, 2.0)
    win_n, ind_n, loss_n = ordinal_probs(-eta, 0.5, 2.0)
    assert np.array_equal(win, loss_n)
    assert np.allclose(ind, ind_n)
    assert np.allclose(expected_score(eta, 0.5, 2.0), 1 - expected_score(-eta, 0.5, 2.0))


def test_five_level_symmetry() -> None:
    eta = np.array([-1.3, 0.0, 0.8])
    forward = category_probs(eta, 0.5, 2.0)
    backward = category_probs(-eta, 0.5, 2.0)
    assert np.allclose(forward, backward[:, ::-1])


def test_three_way_matches_five_level() -> None:
    eta = np.array([-1.3, 0.2, 2.0])
    p5 = category_probs(eta, 0.5, 2.0)
    win, ind, loss = ordinal_probs(eta, 0.5, 2.0)
    assert np.allclose(p5[:, 3] + p5[:, 4], win)
    assert np.allclose(p5[:, 2], ind)
    assert np.allclose(p5[:, 0] + p5[:, 1], loss)


def test_interval_probability_is_sum_of_levels() -> None:
    eta = np.array([0.4, -0.7])
    p5 = category_probs(eta, 0.5, 2.0)
    got = interval_probability(eta, np.array([3, 0]), np.array([4, 1]), 0.5, 2.0)
    assert np.allclose(got, [p5[0, 3] + p5[0, 4], p5[1, 0] + p5[1, 1]])


def test_log_likelihood_flip_identity() -> None:
    eta = np.array([0.4, -0.7, 1.1])
    k_lo = np.array([3, 0, 2])
    k_hi = np.array([4, 1, 2])
    ll = log_likelihood(eta, k_lo, k_hi, 0.5, 2.0)
    flipped = log_likelihood(-eta, 4 - k_hi, 4 - k_lo, 0.5, 2.0)
    assert flipped == pytest.approx(ll)
