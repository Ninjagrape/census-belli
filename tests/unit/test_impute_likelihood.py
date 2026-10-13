"""Unit tests for dividing reconcile's side prior out of a posterior."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipeline.imputers.likelihood import (
    divide_prior,
    likelihoods_from_draws,
    likelihoods_from_published,
    load_neutral_draws,
)
from pipeline.imputers.records import SideRow

M0 = 0.5
S0 = 2.0
MIN_PRECISION = 0.5


def _posterior_from_likelihood(
    precision: np.ndarray, m_lik: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The posterior reconcile would report for a known likelihood and prior."""
    n = len(m_lik)
    posterior_precision = precision + np.eye(n) / S0**2
    cov = np.linalg.inv(posterior_precision)
    mu = cov @ (precision @ m_lik + (M0 / S0**2) * np.ones(n))
    return mu, cov


def test_known_posterior_gives_back_the_known_likelihood() -> None:
    precision = np.array([[2.0, 0.5], [0.5, 3.0]])
    m_lik = np.array([1.0, 2.0])
    mu, cov = _posterior_from_likelihood(precision, m_lik)

    result = divide_prior(mu, cov, M0, S0, MIN_PRECISION)

    assert result.method == "joint"
    assert result.keep == (0, 1)
    np.testing.assert_allclose(result.precision, precision, atol=1e-9)
    np.testing.assert_allclose(result.m_lik, m_lik, atol=1e-9)


def test_single_side_is_the_scalar_case() -> None:
    mu, cov = _posterior_from_likelihood(np.array([[4.0]]), np.array([3.0]))

    result = divide_prior(mu, cov, M0, S0, MIN_PRECISION)

    assert result.keep == (0,)
    assert result.precision[0, 0] == pytest.approx(4.0)
    assert result.m_lik[0] == pytest.approx(3.0)


def test_weak_side_becomes_missing_and_the_rest_is_recomputed() -> None:
    precision = np.diag([3.0, 0.2])
    m_lik = np.array([1.0, 5.0])
    mu, cov = _posterior_from_likelihood(precision, m_lik)

    result = divide_prior(mu, cov, M0, S0, MIN_PRECISION)

    assert result.keep == (0,)
    assert result.precision[0, 0] == pytest.approx(3.0)
    assert result.m_lik[0] == pytest.approx(1.0)


def test_every_side_weak_leaves_nothing() -> None:
    precision = np.diag([0.1, 0.2])
    mu, cov = _posterior_from_likelihood(precision, np.array([1.0, 2.0]))

    result = divide_prior(mu, cov, M0, S0, MIN_PRECISION)

    assert result.keep == ()
    assert result.m_lik.size == 0


def test_non_positive_definite_precision_falls_back_to_the_diagonal() -> None:
    # Strongly correlated posterior: the joint likelihood precision has a
    # negative eigenvalue once the prior is subtracted, though each diagonal
    # entry is comfortably above the weak threshold.
    s0 = 1.0
    c = 0.6
    cov = c * np.array([[1.0, 0.99], [0.99, 1.0]])
    mu = np.array([1.0, 1.4])
    full = np.linalg.inv(cov) - np.eye(2) / s0**2
    assert np.linalg.eigvalsh(full).min() < 0
    assert np.diag(full).min() > MIN_PRECISION

    result = divide_prior(mu, cov, M0, s0, MIN_PRECISION)

    assert result.method == "diagonal"
    assert result.keep == (0, 1)
    p = 1.0 / c - 1.0 / s0**2
    np.testing.assert_allclose(result.precision, np.diag([p, p]))
    np.testing.assert_allclose(result.m_lik, (mu / c - M0 / s0**2) / p)


def test_singular_covariance_falls_back_to_the_diagonal() -> None:
    cov = np.array([[0.2, 0.2], [0.2, 0.2]])

    result = divide_prior(np.array([1.0, 1.0]), cov, M0, S0, MIN_PRECISION)

    assert result.method == "diagonal"
    assert result.precision.shape == (2, 2)
    assert result.precision[0, 1] == 0.0


def _write_draws(
    path: Path,
    run_id: int,
    draws: np.ndarray,
    side_ids: list[int],
    fingerprint: dict[str, str] | None = None,
) -> None:
    np.savez(
        path,
        run_id=run_id,
        **(fingerprint or {}),
        side_id=np.array(side_ids, dtype=np.int64),
        battle_id=np.array([(s + 1) // 2 for s in side_ids], dtype=np.int64),
        draws=draws,
        m0_mean=M0,
        s0_mean=S0,
    )


def test_load_rejects_mismatched_shapes(tmp_path: Path) -> None:
    path = tmp_path / "neutral_draws_run1.npz"
    _write_draws(path, 1, np.zeros((5, 3)), [1, 2])

    with pytest.raises(ValueError, match="does not match"):
        load_neutral_draws(path)


def test_load_reads_the_fingerprint_and_tolerates_its_absence(tmp_path: Path) -> None:
    stamped = tmp_path / "neutral_draws_run1.npz"
    _write_draws(stamped, 1, np.zeros((5, 2)), [1, 2], {"database": "db_a", "started_at": "t0"})
    old = tmp_path / "neutral_draws_run2.npz"
    _write_draws(old, 2, np.zeros((5, 2)), [1, 2])

    assert (load_neutral_draws(stamped).database, load_neutral_draws(stamped).started_at) == (
        "db_a",
        "t0",
    )
    assert load_neutral_draws(old).database is None


def test_likelihoods_from_draws_recover_a_known_likelihood(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    precision = np.array([[2.0, 0.0], [0.0, 3.0]])
    mu, cov = _posterior_from_likelihood(precision, np.array([1.0, 2.0]))
    draws = rng.multivariate_normal(mu, cov, size=20000)
    path = tmp_path / "neutral_draws_run5.npz"
    # Sides 1 and 2 share battle 1; side 9 is stale (no longer in scope).
    _write_draws(path, 5, np.column_stack([draws, rng.normal(size=20000)]), [1, 2, 9])
    neutral = load_neutral_draws(path)

    liks, stats = likelihoods_from_draws(neutral, {1, 2}, min_precision=MIN_PRECISION)

    assert stats.stale_sides == 1
    assert len(liks) == 1
    assert liks[0].side_ids == (1, 2)
    np.testing.assert_allclose(liks[0].m_lik, [1.0, 2.0], atol=0.05)
    np.testing.assert_allclose(liks[0].precision, precision, atol=0.1)


def _side(
    side_id: int,
    battle_id: int,
    published: tuple[float, float | None, float | None] | None,
) -> SideRow:
    return SideRow(
        side_id=side_id,
        battle_id=battle_id,
        year=1800,
        outcome=None,
        battle_type=None,
        is_defender=None,
        polity_key="unknown",
        commander_key="unknown",
        published=published,
    )


def test_published_likelihood_is_diagonal_with_sd_from_the_interval() -> None:
    lo, hi = 1000.0, 1000.0 * float(np.exp(3.92))
    sides = [_side(1, 1, (4000.0, lo, hi)), _side(2, 1, None), _side(3, 2, (500.0, None, None))]

    liks = likelihoods_from_published(sides, default_sd=1.5)

    assert [lik.battle_id for lik in liks] == [1, 2]
    assert liks[0].side_ids == (1,)
    assert liks[0].m_lik[0] == pytest.approx(np.log(4000.0))
    assert liks[0].precision[0, 0] == pytest.approx(1.0)
    assert liks[1].precision[0, 0] == pytest.approx(1.0 / 1.5**2)
    assert liks[0].method == "published"
