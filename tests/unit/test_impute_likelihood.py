"""Unit tests for the likelihood impute reads from reconcile's neutral draws file."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipeline.imputers.likelihood import (
    StaleDrawsFileError,
    likelihoods_from_draws,
    likelihoods_from_published,
    load_neutral_draws,
)
from pipeline.imputers.records import SideRow

N_DRAWS = 40


def write_draws(
    path: Path,
    *,
    run_id: int = 1,
    side_ids: tuple[int, ...] = (1, 2, 3),
    battle_ids: tuple[int, ...] = (1, 1, 2),
    lik_mean: tuple[float, ...] = (9.0, 10.0, 11.0),
    lik_sd: tuple[float, ...] = (0.25, 0.5, 0.3),
    has_point: tuple[bool, ...] = (True, True, True),
    loadings: np.ndarray | None = None,
    names: tuple[str, ...] = (),
    theta: dict[str, np.ndarray] | None = None,
    fingerprint: dict[str, str] | None = None,
    with_likelihood: bool = True,
) -> Path:
    """Write a neutral draws file with the exported likelihood (or without it)."""
    n = len(side_ids)
    arrays: dict[str, object] = {
        "run_id": run_id,
        **(fingerprint or {}),
        "side_id": np.array(side_ids, dtype=np.int64),
        "battle_id": np.array(battle_ids, dtype=np.int64),
        "draws": np.zeros((N_DRAWS, n)),
        "m0_mean": 9.0,
        "s0_mean": 1.0,
    }
    if with_likelihood:
        arrays.update(
            lik_mean=np.array(lik_mean, dtype=float),
            lik_sd=np.array(lik_sd, dtype=float),
            lik_has_point=np.array(has_point, dtype=bool),
            lik_loadings=np.zeros((n, len(names))) if loadings is None else loadings,
            lik_loading_names=np.array(names, dtype=np.str_),
        )
    arrays.update(theta or {})
    np.savez(path, **arrays)
    return path


def test_load_reads_the_exported_likelihood(tmp_path: Path) -> None:
    path = write_draws(tmp_path / "neutral_draws_run1.npz")

    loaded = load_neutral_draws(path)

    assert loaded.lik_mean is not None
    np.testing.assert_allclose(loaded.lik_mean, [9.0, 10.0, 11.0])
    assert loaded.lik_has_point is not None
    assert loaded.lik_has_point.dtype == bool
    assert loaded.lik_loading_names == ()


def test_an_old_file_without_the_likelihood_fails_loudly_and_says_what_to_do(
    tmp_path: Path,
) -> None:
    path = write_draws(tmp_path / "neutral_draws_run1.npz", with_likelihood=False)

    with pytest.raises(StaleDrawsFileError, match="Re-run reconcile") as caught:
        load_neutral_draws(path)

    assert "lik_mean" in str(caught.value)
    assert isinstance(caught.value, ValueError)


def test_evaluate_can_still_read_a_file_that_predates_the_likelihood(tmp_path: Path) -> None:
    path = write_draws(tmp_path / "neutral_draws_run1.npz", with_likelihood=False)

    loaded = load_neutral_draws(path, require_likelihood=False)

    assert loaded.lik_mean is None
    with pytest.raises(StaleDrawsFileError):
        likelihoods_from_draws(loaded, {1, 2, 3})


def test_load_rejects_mismatched_shapes(tmp_path: Path) -> None:
    path = write_draws(tmp_path / "neutral_draws_run1.npz", lik_mean=(9.0, 10.0))

    with pytest.raises(ValueError, match="lik_mean"):
        load_neutral_draws(path)


def test_load_rejects_a_point_side_without_a_finite_mean(tmp_path: Path) -> None:
    path = write_draws(tmp_path / "neutral_draws_run1.npz", lik_mean=(9.0, float("nan"), 11.0))

    with pytest.raises(ValueError, match="non-finite mean"):
        load_neutral_draws(path)


def test_load_reads_the_fingerprint(tmp_path: Path) -> None:
    path = write_draws(
        tmp_path / "neutral_draws_run1.npz", fingerprint={"database": "db_a", "started_at": "t0"}
    )

    loaded = load_neutral_draws(path)

    assert (loaded.database, loaded.started_at) == ("db_a", "t0")


def test_every_side_with_a_point_row_gets_a_likelihood_at_its_exported_sd(tmp_path: Path) -> None:
    neutral = load_neutral_draws(write_draws(tmp_path / "neutral_draws_run1.npz"))

    liks, stats = likelihoods_from_draws(neutral, {1, 2, 3})

    assert [lik.battle_id for lik in liks] == [1, 2]
    assert liks[0].side_ids == (1, 2)
    np.testing.assert_allclose(liks[0].m_lik, [9.0, 10.0])
    np.testing.assert_allclose(liks[0].precision, np.diag([1 / 0.25**2, 1 / 0.5**2]))
    assert liks[0].method == "reconcile"
    assert stats.bound_only_side_ids == frozenset()
    assert stats.stale_sides == 0


def test_a_side_far_from_the_corpus_centre_is_never_weak(tmp_path: Path) -> None:
    # The sides the old prior division discarded: the largest and smallest armies.
    path = write_draws(
        tmp_path / "neutral_draws_run1.npz",
        side_ids=(1, 2),
        battle_ids=(1, 1),
        lik_mean=(4.6, 15.5),
        lik_sd=(0.27, 0.27),
        has_point=(True, True),
    )

    liks, _ = likelihoods_from_draws(load_neutral_draws(path), {1, 2})

    assert liks[0].side_ids == (1, 2)
    np.testing.assert_allclose(liks[0].m_lik, [4.6, 15.5])


def test_a_bound_only_side_is_missing_and_counted_by_cause(tmp_path: Path) -> None:
    path = write_draws(
        tmp_path / "neutral_draws_run1.npz",
        lik_mean=(9.0, float("nan"), 11.0),
        lik_sd=(0.25, float("nan"), 0.3),
        has_point=(True, False, True),
    )

    liks, stats = likelihoods_from_draws(load_neutral_draws(path), {1, 2, 3})

    assert liks[0].side_ids == (1,)
    assert stats.bound_only_side_ids == frozenset({2})


def test_a_battle_with_only_bound_sides_has_no_likelihood(tmp_path: Path) -> None:
    path = write_draws(
        tmp_path / "neutral_draws_run1.npz",
        lik_mean=(float("nan"), float("nan"), 11.0),
        lik_sd=(float("nan"), float("nan"), 0.3),
        has_point=(False, False, True),
    )

    liks, stats = likelihoods_from_draws(load_neutral_draws(path), {1, 2, 3})

    assert [lik.battle_id for lik in liks] == [2]
    assert stats.bound_only_side_ids == frozenset({1, 2})


def test_a_stale_side_is_ignored_and_counted(tmp_path: Path) -> None:
    neutral = load_neutral_draws(write_draws(tmp_path / "neutral_draws_run1.npz"))

    liks, stats = likelihoods_from_draws(neutral, {1, 2})

    assert stats.stale_sides == 1
    assert [lik.battle_id for lik in liks] == [1]


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
