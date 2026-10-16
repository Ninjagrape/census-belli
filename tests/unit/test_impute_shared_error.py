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
    divide_prior,
    fit_shared_error,
    likelihoods_from_draws,
    shared_error_summary,
)
from pipeline.imputers.outputs import write_gate_report, write_rounds
from pipeline.imputers.priors import load_priors
from pipeline.imputers.records import SideRow

PRIORS = load_priors("config/imputation_priors.yaml")
M0 = 0.5
S0 = 2.0
MIN_PRECISION = 0.5
N_DRAWS = 4000
TRUE_LOADING = 2.0
NOISE_SD = 0.3
# Battles 1 and 2 load on g; battle 3 does not.
LOADINGS = {1: TRUE_LOADING, 2: TRUE_LOADING, 3: 0.0}


def _neutral(
    *,
    with_shared: bool = True,
    g_constant: bool = False,
    a_era_present: bool = False,
    seed: int = 0,
) -> NeutralDraws:
    """Two sides per battle; side draws are ``center + loading * g + noise``."""
    rng = np.random.default_rng(seed)
    g = np.zeros(N_DRAWS) if g_constant else rng.standard_normal(N_DRAWS)
    columns: list[np.ndarray] = []
    side_id: list[int] = []
    battle_id: list[int] = []
    for battle, loading in LOADINGS.items():
        for k in range(2):
            center = 1.0 + battle + k
            columns.append(center + loading * g + NOISE_SD * rng.standard_normal(N_DRAWS))
            side_id.append(2 * battle - 1 + k)
            battle_id.append(battle)
    n_sides = len(columns)
    return NeutralDraws(
        run_id=1,
        side_id=np.array(side_id, dtype=np.int64),
        battle_id=np.array(battle_id, dtype=np.int64),
        draws=np.column_stack(columns),
        m0_mean=M0,
        s0_mean=S0,
        g_unlabelled=g if with_shared else None,
        a_era=(0.5 * g + rng.standard_normal(N_DRAWS)) if (with_shared and a_era_present) else None,
        offset_bar=np.zeros(N_DRAWS) if with_shared else None,
        side_era=np.zeros(n_sides) if with_shared else None,
        side_unlabelled_share=np.ones(n_sides) if with_shared else None,
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


def test_regression_recovers_the_known_loading() -> None:
    neutral = _neutral()

    fit = fit_shared_error(neutral, list(range(6)))

    assert not isinstance(fit, str)
    shared, loadings, residual = fit
    assert shared.names == ("g_unlabelled",)
    np.testing.assert_allclose(loadings[:4, 0], TRUE_LOADING, atol=0.05)
    np.testing.assert_allclose(loadings[4:, 0], 0.0, atol=0.05)
    # Residual keeps each column's mean and loses the shared part.
    np.testing.assert_allclose(residual.mean(axis=0), neutral.draws.mean(axis=0), atol=1e-9)
    np.testing.assert_allclose(residual.std(axis=0), NOISE_SD, atol=0.02)
    assert shared.explained_share == pytest.approx(0.97, abs=0.02)
    assert shared.mean_abs_loading == pytest.approx(4 * TRUE_LOADING / 6, abs=0.05)


def test_likelihood_loading_undoes_the_prior_pull_on_the_posterior_mean() -> None:
    # Data mean m_lik = c + 2 g with precision P; reconcile's posterior mean moves by
    # only 2 P / (P + 1/s0^2) per unit g. The likelihood loading must come back to 2.
    rng = np.random.default_rng(5)
    precision = 3.0
    g = 0.3 * rng.standard_normal(N_DRAWS)
    post_precision = precision + 1.0 / S0**2
    mean = (precision * (1.0 + TRUE_LOADING * g) + M0 / S0**2) / post_precision
    draws = (mean + rng.standard_normal(N_DRAWS) / np.sqrt(post_precision))[:, None]
    neutral = NeutralDraws(
        run_id=1,
        side_id=np.array([1]),
        battle_id=np.array([1]),
        draws=draws,
        m0_mean=M0,
        s0_mean=S0,
        g_unlabelled=g,
    )

    liks, stats = likelihoods_from_draws(neutral, {1}, min_precision=MIN_PRECISION)

    assert stats.shared is not None
    shift = liks[0].shift_loading
    assert shift is not None
    assert shift[0, 0] == pytest.approx(TRUE_LOADING, abs=0.1)
    assert liks[0].precision[0, 0] == pytest.approx(precision, rel=0.1)


def _comoving_neutral(*, with_m0: bool) -> tuple[NeutralDraws, float]:
    """One all-unlabelled side: data mean ``d - g``, prior mean ``M0 - g`` (as ``m0`` is).

    Reconcile's posterior mean is ``(P (d - g) + (M0 - g) / s0^2) / (P + 1/s0^2)``,
    which falls by exactly one per unit g, the same as the data does.
    """
    rng = np.random.default_rng(8)
    precision = 2.0
    g = 0.5 * rng.standard_normal(N_DRAWS)
    post_precision = precision + 1.0 / S0**2
    mean = (precision * (1.0 - g) + (M0 - g) / S0**2) / post_precision
    draws = (mean + rng.standard_normal(N_DRAWS) / np.sqrt(post_precision))[:, None]
    neutral = NeutralDraws(
        run_id=1,
        side_id=np.array([1]),
        battle_id=np.array([1]),
        draws=draws,
        m0_mean=M0,
        s0_mean=S0,
        g_unlabelled=g,
        a_era=None,
        offset_bar=g,
        side_era=np.zeros(1),
        side_unlabelled_share=np.ones(1),
        m0_draws=(M0 - g) if with_m0 else None,
    )
    return neutral, precision


def test_comoving_m0_returns_an_unlabelled_sides_data_loading_as_minus_one() -> None:
    neutral, precision = _comoving_neutral(with_m0=True)

    liks, stats = likelihoods_from_draws(neutral, {1}, min_precision=MIN_PRECISION)

    shift = liks[0].shift_loading
    assert shift is not None and stats.shared is not None
    assert shift[0, 0] == pytest.approx(-1.0, abs=0.06)
    assert stats.shared.m0_loading is not None
    assert stats.shared.m0_loading[0] == pytest.approx(-1.0, abs=0.05)
    assert stats.shared.loading_gap is not None
    assert stats.shared.loading_gap < 0.1
    summary = shared_error_summary(stats.shared, stats.shared_reason)
    assert summary["m0_comoves"] is True
    assert summary["loading_gap_vs_design"] == stats.shared.loading_gap
    assert precision > 0


def test_fixed_m0_formula_overstates_the_loading_and_the_gap_shows_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    neutral, precision = _comoving_neutral(with_m0=False)

    liks, stats = likelihoods_from_draws(neutral, {1}, min_precision=MIN_PRECISION)

    shift = liks[0].shift_loading
    assert shift is not None and stats.shared is not None
    assert stats.shared.m0_loading is None
    assert shift[0, 0] == pytest.approx(-(1.0 + 1.0 / (precision * S0**2)), abs=0.06)
    assert stats.shared.loading_gap is not None
    assert stats.shared.loading_gap > 0.05


def _round_values(neutral: NeutralDraws, *, propagate: bool, n_rounds: int = 300) -> np.ndarray:
    liks, stats = likelihoods_from_draws(
        neutral,
        set(neutral.side_id.tolist()),
        min_precision=MIN_PRECISION,
        propagate_shared=propagate,
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
    liks, stats = likelihoods_from_draws(
        neutral, set(neutral.side_id.tolist()), min_precision=MIN_PRECISION
    )
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


def test_old_file_without_shared_arrays_behaves_as_before(
    caplog: pytest.LogCaptureFixture,
) -> None:
    neutral = _neutral(with_shared=False)

    liks, stats = likelihoods_from_draws(
        neutral, set(neutral.side_id.tolist()), min_precision=MIN_PRECISION
    )
    plain, _ = likelihoods_from_draws(
        neutral,
        set(neutral.side_id.tolist()),
        min_precision=MIN_PRECISION,
        propagate_shared=False,
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


def test_constant_component_is_dropped_and_all_constant_falls_back() -> None:
    mixed = fit_shared_error(_neutral(g_constant=True, a_era_present=True), list(range(6)))
    flat = fit_shared_error(_neutral(g_constant=True), list(range(6)))

    assert not isinstance(mixed, str)
    assert mixed[0].names == ("a_era",)
    assert flat == "theta_has_no_variance"


def test_summary_and_manifests_record_propagation(tmp_path: Path) -> None:
    neutral = _neutral()
    liks, stats = likelihoods_from_draws(
        neutral, set(neutral.side_id.tolist()), min_precision=MIN_PRECISION
    )
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
        tmp_path, scale="neutral", run_id=1, written=True, gates=[], shared_error=summary
    )

    manifest = json.loads((tmp_path / "neutral" / "round_2" / "manifest.json").read_text("utf-8"))
    report = json.loads((tmp_path / "neutral" / "gate_report.json").read_text("utf-8"))
    assert manifest["shared_error_propagated"] is True
    assert manifest["theta_draw_index"] == rounds.theta_index[2] if rounds.theta_index else False
    assert report["shared_error"]["propagated"] is True
    assert report["shared_error"]["theta"] == ["g_unlabelled"]
    assert report["shared_error"]["theta_draw_index"] == list(rounds.theta_index or ())
    assert 0.9 < report["shared_error"]["explained_variance_share"] <= 1.0
    fallback = shared_error_summary(None, "shared_arrays_absent")
    assert fallback == {"propagated": False, "reason": "shared_arrays_absent"}


def test_singular_subset_fallback_keeps_original_side_indices() -> None:
    # Sides 0 and 2 are the same draws (singular together); side 1 is far too diffuse
    # to carry a likelihood. After dropping it the subset is still singular, and the
    # diagonal path's survivors must be reported against the original three sides.
    rng = np.random.default_rng(2)
    base = 1.0 + 0.3 * rng.standard_normal(500)
    diffuse = 5.0 * rng.standard_normal(500)
    block = np.column_stack([base, diffuse, base])

    result = divide_prior(block.mean(axis=0), np.cov(block, rowvar=False), M0, S0, MIN_PRECISION)

    assert result.method == "diagonal"
    assert result.keep == (0, 2)
    assert result.m_lik.shape == (2,)


def test_weak_sides_are_classified_on_the_raw_covariance() -> None:
    # Side 2 carries a loading of 1.5 on g with little noise of its own: weak on its
    # raw posterior, comfortably strong once g is regressed out. Propagation on or
    # off must still drop it.
    rng = np.random.default_rng(6)
    g = rng.standard_normal(N_DRAWS)
    strong = 1.0 + 0.3 * rng.standard_normal(N_DRAWS)
    loaded = 2.0 + 1.5 * g + 0.8 * rng.standard_normal(N_DRAWS)
    neutral = NeutralDraws(
        run_id=1,
        side_id=np.array([1, 2]),
        battle_id=np.array([1, 1]),
        draws=np.column_stack([strong, loaded]),
        m0_mean=M0,
        s0_mean=S0,
        g_unlabelled=g,
        offset_bar=np.zeros(N_DRAWS),
        side_era=np.zeros(2),
        side_unlabelled_share=np.ones(2),
    )

    on, on_stats = likelihoods_from_draws(neutral, {1, 2}, min_precision=MIN_PRECISION)
    off, off_stats = likelihoods_from_draws(
        neutral, {1, 2}, min_precision=MIN_PRECISION, propagate_shared=False
    )

    assert on_stats.shared is not None
    assert on_stats.weak_sides == off_stats.weak_sides == 1
    assert [lik.side_ids for lik in on] == [lik.side_ids for lik in off] == [(1,)]


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
    draws = np.column_stack(
        [1.0 + 0.6 * g + NOISE_SD * rng.standard_normal(N_DRAWS) for _ in range(2)]
    )
    neutral = NeutralDraws(
        run_id=1,
        side_id=np.array([1, 2]),
        battle_id=np.array([1, 1]),
        draws=draws,
        m0_mean=M0,
        s0_mean=S0,
        g_unlabelled=g,
        offset_bar=np.zeros(N_DRAWS),
        side_era=np.zeros(2),
        side_unlabelled_share=np.ones(2),
    )
    liks, stats = likelihoods_from_draws(neutral, {1}, min_precision=MIN_PRECISION)
    assert [lik.side_ids for lik in liks] == [(1,)]
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

    assert stats.shared is not None and rounds.theta_index is not None
    theta = stats.shared.theta_centered[list(rounds.theta_index), 0]
    observed, imputed = rounds.values[:, 0], rounds.values[:, 1]
    slope_level = np.polyfit(theta, observed, 1)[0]
    slope_diff = np.polyfit(theta, observed - imputed, 1)[0]
    assert abs(slope_level) > 0.4
    assert abs(slope_diff) < 0.1
