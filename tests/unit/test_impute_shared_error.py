"""Unit tests for propagating reconcile's shared parameter error through the rounds."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pipeline.imputers.conditional import PosteriorSamples, draw_rounds, select_theta_indices
from pipeline.imputers.design import build_design
from pipeline.imputers.likelihood import (
    NeutralDraws,
    likelihoods_from_draws,
    shared_error_summary,
)
from pipeline.imputers.outputs import write_gate_report, write_rounds
from pipeline.imputers.priors import load_priors
from pipeline.imputers.records import SideRow

PRIORS = load_priors("config/imputation_priors.yaml")
N_DRAWS = 4000
TRUE_LOADING = -2.0
LIK_SD = 0.3
# Battles 1 and 2 load on g; battle 3 does not.
LOADINGS = {1: TRUE_LOADING, 2: TRUE_LOADING, 3: 0.0}


def _neutral(
    *,
    with_theta: bool = True,
    g_constant: bool = False,
    names: tuple[str, ...] = ("g_unlabelled",),
    seed: int = 0,
) -> NeutralDraws:
    """Two sides per battle with centres ``1 + battle + k`` and an exported loading on g."""
    rng = np.random.default_rng(seed)
    g = np.zeros(N_DRAWS) if g_constant else rng.standard_normal(N_DRAWS)
    side_id: list[int] = []
    battle_id: list[int] = []
    mean: list[float] = []
    loading: list[list[float]] = []
    for battle, slope in LOADINGS.items():
        for k in range(2):
            side_id.append(2 * battle - 1 + k)
            battle_id.append(battle)
            mean.append(1.0 + battle + k)
            loading.append([slope, 0.0][: len(names)])
    n_sides = len(side_id)
    return NeutralDraws(
        run_id=1,
        side_id=np.array(side_id, dtype=np.int64),
        battle_id=np.array(battle_id, dtype=np.int64),
        draws=np.zeros((N_DRAWS, n_sides)),
        m0_mean=9.0,
        s0_mean=1.0,
        g_unlabelled=g if with_theta else None,
        a_era=rng.standard_normal(N_DRAWS) if (with_theta and "a_era" in names) else None,
        lik_mean=np.array(mean),
        lik_sd=np.full(n_sides, LIK_SD),
        lik_has_point=np.ones(n_sides, dtype=bool),
        lik_loadings=np.array(loading, dtype=float).reshape(n_sides, len(names)),
        lik_loading_names=names,
    )


def _sides() -> list[SideRow]:
    return [
        SideRow(
            side_id=2 * b - 1 + k,
            battle_id=b,
            year=1805,
            outcome=None,
            battle_type=None,
            is_defender=None,
            polity_key="unknown",
            commander_key="unknown",
        )
        for b in LOADINGS
        for k in range(2)
    ]


def _posterior(n: int, prior_sd: float = 2.0) -> PosteriorSamples:
    """A loose prior by default, so the era-level shift does not swamp the data."""
    eras = len(PRIORS.era_names)
    return PosteriorSamples(
        a_era=np.full((n, eras), 1.0),
        beta_out=None,
        beta_type=None,
        beta_def=None,
        v_polity=None,
        c_cmd=None,
        tau_p=None,
        tau_c=None,
        tau_b=np.full(n, prior_sd),
        sigma=np.full(n, prior_sd),
        n_chains=1,
        n_draws=n,
    )


def test_the_shared_error_carries_the_exported_loadings_and_the_centred_draws() -> None:
    neutral = _neutral()

    liks, stats = likelihoods_from_draws(neutral, set(neutral.side_id.tolist()))

    assert stats.shared is not None
    assert stats.shared.names == ("g_unlabelled",)
    assert stats.shared.theta_centered.shape == (N_DRAWS, 1)
    assert float(stats.shared.theta_centered.mean()) == pytest.approx(0.0, abs=1e-12)
    assert stats.shared.mean_abs_loading == pytest.approx(4 * abs(TRUE_LOADING) / 6)
    # Four loaded sides at variance 4 against 0.09 of Fisher variance, two unloaded.
    expected = 4 * 4.0 / (4 * 4.0 + 6 * LIK_SD**2)
    assert stats.shared.explained_share == pytest.approx(expected, abs=0.03)
    assert liks[0].shift_loading is not None
    np.testing.assert_allclose(liks[0].shift_loading[:, 0], TRUE_LOADING)
    np.testing.assert_allclose(liks[2].shift_loading[:, 0], 0.0)


def _round_values(neutral: NeutralDraws, *, propagate: bool, n_rounds: int = 300) -> np.ndarray:
    liks, stats = likelihoods_from_draws(
        neutral, set(neutral.side_id.tolist()), propagate_shared=propagate
    )
    design = build_design(_sides(), liks, PRIORS)
    rounds = draw_rounds(
        design,
        _posterior(n_rounds),
        n_rounds=n_rounds,
        seed=11,
        mnar_delta=0.0,
        shared=stats.shared,
    )
    return rounds.values


def test_battles_sharing_a_loading_are_correlated_across_rounds() -> None:
    neutral = _neutral()

    on = _round_values(neutral, propagate=True)
    off = _round_values(neutral, propagate=False)

    # Column 0 is battle 1's first side, 2 battle 2's, 4 battle 3's.
    corr_on = np.corrcoef(on[:, [0, 2, 4]], rowvar=False)
    corr_off = np.corrcoef(off[:, [0, 2, 4]], rowvar=False)
    assert corr_on[0, 1] > 0.9
    assert abs(corr_on[0, 2]) < 0.2
    assert abs(corr_on[1, 2]) < 0.2
    assert abs(corr_off[0, 1]) < 0.2


def test_theta_draw_is_deterministic_per_round_and_leaves_the_main_stream_alone() -> None:
    neutral = _neutral()
    liks, stats = likelihoods_from_draws(neutral, set(neutral.side_id.tolist()))
    design = build_design(_sides(), liks, PRIORS)
    post = _posterior(20)

    first = draw_rounds(design, post, n_rounds=6, seed=3, mnar_delta=0.0, shared=stats.shared)
    again = draw_rounds(design, post, n_rounds=6, seed=3, mnar_delta=0.0, shared=stats.shared)
    other_seed = draw_rounds(design, post, n_rounds=6, seed=4, mnar_delta=0.0, shared=stats.shared)

    assert first.theta_index is not None
    assert len(first.theta_index) == 6
    assert first.theta_index == again.theta_index
    assert first.theta_index != other_seed.theta_index
    np.testing.assert_array_equal(first.values, again.values)


def test_a_file_without_the_parameter_draws_propagates_nothing() -> None:
    neutral = _neutral(with_theta=False)

    liks, stats = likelihoods_from_draws(neutral, set(neutral.side_id.tolist()))
    plain, _ = likelihoods_from_draws(
        neutral, set(neutral.side_id.tolist()), propagate_shared=False
    )

    assert stats.shared is None
    assert stats.shared_reason == "shared_arrays_absent"
    for got, want in zip(liks, plain, strict=True):
        assert got.shift_loading is None
        np.testing.assert_array_equal(got.m_lik, want.m_lik)
        np.testing.assert_array_equal(got.precision, want.precision)
    design = build_design(_sides(), liks, PRIORS)
    off = draw_rounds(design, _posterior(10), n_rounds=4, seed=2, mnar_delta=0.0, shared=None)
    assert off.theta_index is None


def test_a_parameter_that_does_not_vary_falls_back_to_independent_battles() -> None:
    flat = _neutral(g_constant=True)
    disabled = _neutral()

    _, stats = likelihoods_from_draws(flat, set(flat.side_id.tolist()))
    _, off = likelihoods_from_draws(
        disabled, set(disabled.side_id.tolist()), propagate_shared=False
    )

    assert stats.shared is None
    assert stats.shared_reason == "theta_has_no_variance"
    assert off.shared is None
    assert off.shared_reason == "disabled"


def test_summary_and_manifests_record_propagation(tmp_path: Path) -> None:
    neutral = _neutral()
    liks, stats = likelihoods_from_draws(neutral, set(neutral.side_id.tolist()))
    design = build_design(_sides(), liks, PRIORS)
    rounds = draw_rounds(
        design, _posterior(10), n_rounds=3, seed=1, mnar_delta=0.0, shared=stats.shared
    )

    write_rounds(
        tmp_path,
        scale="neutral",
        rounds=rounds,
        design=design,
        run_id=1,
        seed=1,
        reconcile_run_id=1,
        config_digest="x",
    )
    summary = shared_error_summary(stats.shared, stats.shared_reason, rounds.theta_index)
    write_gate_report(
        tmp_path,
        scale="neutral",
        run_id=1,
        written=True,
        gates=[],
        shared_error=summary,
        missing_by_cause={"bound_only": 2, "no_reports": 1, "other": 0},
    )

    manifest = json.loads((tmp_path / "neutral" / "round_2" / "manifest.json").read_text("utf-8"))
    report = json.loads((tmp_path / "neutral" / "gate_report.json").read_text("utf-8"))
    assert manifest["shared_error_propagated"] is True
    assert rounds.theta_index is not None
    assert manifest["theta_draw_index"] == rounds.theta_index[2]
    assert report["shared_error"]["propagated"] is True
    assert report["shared_error"]["theta"] == ["g_unlabelled"]
    assert report["shared_error"]["theta_draw_index"] == list(rounds.theta_index)
    assert 0.5 < report["shared_error"]["explained_variance_share"] <= 1.0
    assert report["missing_by_cause"] == {"bound_only": 2, "no_reports": 1, "other": 0}
    fallback = shared_error_summary(None, "shared_arrays_absent")
    assert fallback == {"propagated": False, "reason": "shared_arrays_absent"}


def test_theta_selection_is_stratified_deterministic_and_permuted() -> None:
    rng = np.random.default_rng(1)
    theta = rng.standard_normal((5000, 2))
    n_rounds = 20

    first = select_theta_indices(theta, n_rounds, seed=3)
    again = select_theta_indices(theta, n_rounds, seed=3)
    other = select_theta_indices(theta, n_rounds, seed=4)

    assert first == again
    assert first != other
    leading = np.sort(theta[first, 0])
    # One draw per twentieth of the distribution: the mean and spread of the picks
    # sit close to the full draws', which independent picks of 20 do not manage.
    assert abs(leading.mean()) < 0.05
    assert leading.std() == pytest.approx(theta[:, 0].std(), rel=0.1)
    assert len(set(first)) == n_rounds
    assert first != sorted(first, key=lambda i: theta[i, 0])
    # Exactly one pick per stratum of the sorted leading component, and not
    # always the stratum midpoint.
    rank = np.empty(len(theta), dtype=int)
    rank[np.argsort(theta[:, 0], kind="stable")] = np.arange(len(theta))
    for picks in (first, other):
        strata = sorted(int(rank[i]) * n_rounds // len(theta) for i in picks)
        assert strata == list(range(n_rounds))
    offsets = {(int(rank[i]) % (len(theta) // n_rounds)) for i in first}
    assert len(offsets) > 1


def test_an_imputed_side_moves_with_theta_as_the_observed_side_in_its_era_does() -> None:
    # Side 1 is observed and loads on g; side 2, same battle and era, has no likelihood.
    # Their difference is what the data and the era prior pin down, so it must not
    # drift with theta across rounds, though each level does.
    rng = np.random.default_rng(12)
    g = rng.standard_normal(N_DRAWS)
    neutral = NeutralDraws(
        run_id=1,
        side_id=np.array([1, 2]),
        battle_id=np.array([1, 1]),
        draws=np.zeros((N_DRAWS, 2)),
        m0_mean=9.0,
        s0_mean=1.0,
        g_unlabelled=g,
        lik_mean=np.array([1.0, np.nan]),
        lik_sd=np.array([LIK_SD, np.nan]),
        lik_has_point=np.array([True, False]),
        lik_loadings=np.array([[0.6], [0.0]]),
        lik_loading_names=("g_unlabelled",),
    )
    liks, stats = likelihoods_from_draws(neutral, {1, 2})
    assert [lik.side_ids for lik in liks] == [(1,)]
    assert stats.bound_only_side_ids == frozenset({2})
    design = build_design(_sides()[:2], liks, PRIORS)
    n_rounds = 400

    rounds = draw_rounds(
        design,
        _posterior(n_rounds, prior_sd=0.5),
        n_rounds=n_rounds,
        seed=5,
        mnar_delta=0.0,
        shared=stats.shared,
    )

    assert stats.shared is not None
    assert rounds.theta_index is not None
    theta = stats.shared.theta_centered[list(rounds.theta_index), 0]
    observed, imputed = rounds.values[:, 0], rounds.values[:, 1]
    slope_level = np.polyfit(theta, observed, 1)[0]
    slope_diff = np.polyfit(theta, observed - imputed, 1)[0]
    assert abs(slope_level) > 0.4
    assert abs(slope_diff) < 0.1
