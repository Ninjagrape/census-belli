"""Folds, held-out prediction metrics and the fit/pool wrapper (sampling stubbed)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from pipeline.evaluators import fit as fit_module
from pipeline.evaluators.fit import fit_pooled, repool
from pipeline.evaluators.folds import assign_folds, train_weight
from pipeline.evaluators.predict import (
    binary_metrics,
    force_only_baseline,
    heldout_x_diff_draws,
    log_score,
    predict_category_probs,
    ranked_probability_score,
    wilson_interval,
)
from pipeline.modelers.config import ModelParams
from pipeline.modelers.design import build_design
from pipeline.modelers.fitting import FitOutcome
from pipeline.modelers.model import PosteriorDraws, assemble_side_scores, joint_x_diff_prior
from pipeline.modelers.ordinal import category_probs
from pipeline.modelers.records import ModelDesign
from tests.unit.modelers_helpers import battle

K = 30


def _design() -> ModelDesign:
    sides = (
        battle(1, (1,), (2,))
        + battle(2, (2,), (3, 4), a_outcome="defeat", b_outcome="victory")
        + battle(3, (1,), (3,), a_outcome="indecisive", b_outcome="indecisive")
        + battle(4, (4,), (1,))
    )
    return build_design(sides)


def _draws(design: ModelDesign, seed: int = 0) -> PosteriorDraws:
    rng = np.random.default_rng(seed)
    skills = rng.normal(0.0, 1.0, (K, design.n_generals))
    sigma = np.ones(K)
    return PosteriorDraws(
        skills=skills,
        side_scores=assemble_side_scores(design, skills, sigma, None, None, None),
        beta_force=rng.normal(0.5, 0.1, K),
        beta_def=np.zeros(K),
        c1=np.full(K, 0.5),
        c2=np.full(K, 2.0),
        sigma_skill=sigma,
    )


def _mp() -> ModelParams:
    return ModelParams.from_mapping({"pooling": {"draws_kept_per_round": 10}})


def test_assign_folds_balanced_seeded_and_validated() -> None:
    folds = assign_folds(23, 5, seed=1)
    sizes = np.bincount(folds, minlength=5)
    assert sizes.max() - sizes.min() <= 1
    assert sizes.sum() == 23
    np.testing.assert_array_equal(folds, assign_folds(23, 5, seed=1))
    assert not np.array_equal(folds, assign_folds(23, 5, seed=2))
    for k in (1, 24):
        with pytest.raises(ValueError, match="k must be"):
            assign_folds(23, k, seed=0)


def test_train_weight_zeroes_only_the_held_out_fold() -> None:
    folds = np.array([0, 1, 2, 1, 0])
    np.testing.assert_array_equal(train_weight(folds, 1), [1.0, 0.0, 1.0, 0.0, 1.0])


def test_heldout_x_diff_draws_follow_the_measurement_prior() -> None:
    design = _design()
    obs_mean = np.full(design.n_sides, np.nan)
    obs_sd = np.full(design.n_sides, np.nan)
    obs_mean[design.side_a[0]], obs_sd[design.side_a[0]] = 1.0, 0.2
    obs_mean[design.side_b[0]], obs_sd[design.side_b[0]] = 0.4, 0.2
    draws = heldout_x_diff_draws(design, obs_mean, obs_sd, 1.0, 20000, np.random.default_rng(0))
    mu, sd = joint_x_diff_prior(design, obs_mean, obs_sd, 1.0)
    assert draws.shape == (20000, design.n_battles)
    np.testing.assert_allclose(draws.mean(axis=0), mu, atol=0.05)
    np.testing.assert_allclose(draws.std(axis=0), sd, rtol=0.05)


def test_predict_category_probs_matches_manual_average() -> None:
    design = _design()
    draws = _draws(design)
    x = np.linspace(-1.0, 1.0, design.n_battles)
    idx = np.array([2, 0])
    probs = predict_category_probs(design, draws, x, idx)
    assert probs.shape == (2, 5)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    eta = (
        draws.side_scores[:, design.side_a]
        - draws.side_scores[:, design.side_b]
        + draws.beta_force[:, None] * x
        + draws.beta_def[:, None] * design.defender_diff
    )
    manual = category_probs(eta[:, idx], draws.c1[:, None], draws.c2[:, None]).mean(axis=0)
    np.testing.assert_allclose(probs, manual)
    per_draw_x = np.tile(x, (K, 1))
    np.testing.assert_allclose(predict_category_probs(design, draws, per_draw_x, idx), probs)


def test_wilson_interval() -> None:
    lo, hi = wilson_interval(8, 10)
    assert 0.4 < lo < 0.8 < hi < 1.0
    assert np.isnan(wilson_interval(0, 0)).all()
    assert wilson_interval(0, 10)[0] == 0.0


def test_binary_metrics_accuracy_brier_and_calibration() -> None:
    probs = np.array(
        [
            [0.1, 0.1, 0.0, 0.4, 0.4],  # p = 0.8, A won
            [0.4, 0.4, 0.0, 0.1, 0.1],  # p = 0.2, A lost
            [0.1, 0.1, 0.0, 0.4, 0.4],  # p = 0.8, but A lost
            [0.2, 0.2, 0.2, 0.2, 0.2],  # p = 0.5, A won: half credit
            [0.2, 0.2, 0.2, 0.2, 0.2],  # indecisive: excluded
        ]
    )
    k_lo = np.array([3, 0, 0, 4, 2])
    k_hi = np.array([4, 1, 1, 4, 2])
    out = binary_metrics(probs, k_lo, k_hi, n_bins=5)
    assert out["n"] == 4
    assert out["accuracy"] == pytest.approx((1 + 1 + 0 + 0.5) / 4)
    assert out["brier"] == pytest.approx(np.mean([0.04, 0.04, 0.64, 0.25]))
    counts = [b["count"] for b in out["calibration"]]
    assert counts == [0, 1, 1, 0, 2]
    assert out["calibration"][4]["observed"] == pytest.approx(0.5)
    assert np.isnan(out["calibration"][0]["mean_p"])
    assert binary_metrics(probs, np.array([2] * 5), np.array([2] * 5))["n"] == 0


def test_binary_metrics_side_a_rate_and_favourite_calibration() -> None:
    probs = np.array(
        [
            [0.1, 0.1, 0.0, 0.4, 0.4],  # p = 0.8, A won: favourite A won
            [0.4, 0.4, 0.0, 0.1, 0.1],  # p = 0.2, A lost: favourite B won
            [0.1, 0.1, 0.0, 0.4, 0.4],  # p = 0.8, A lost: favourite lost
            [0.2, 0.2, 0.2, 0.2, 0.2],  # p = 0.5, A won: favourite is A
            [0.2, 0.2, 0.2, 0.2, 0.2],  # indecisive: excluded
        ]
    )
    k_lo = np.array([3, 0, 0, 4, 2])
    k_hi = np.array([4, 1, 1, 4, 2])
    out = binary_metrics(probs, k_lo, k_hi, n_bins=5)
    assert out["side_a_win_rate"] == pytest.approx(0.5)
    fav = out["calibration_favourite"]
    assert [b["count"] for b in fav] == [1, 0, 0, 3, 0]
    assert fav[0]["lo"] == pytest.approx(0.5) and fav[-1]["hi"] == pytest.approx(1.0)
    assert fav[0]["observed"] == pytest.approx(1.0)
    assert fav[3]["mean_p"] == pytest.approx(0.8)
    assert fav[3]["observed"] == pytest.approx(2 / 3)
    assert np.isnan(fav[1]["mean_p"])


def test_favourite_calibration_is_unchanged_by_swapping_a_and_b() -> None:
    rng = np.random.default_rng(3)
    probs = rng.dirichlet(np.ones(5), size=60)
    k = rng.choice([0, 1, 3, 4], size=60)
    k_lo = k_hi = k
    swapped = binary_metrics(probs[:, ::-1], 4 - k_hi, 4 - k_lo, n_bins=5)
    original = binary_metrics(probs, k_lo, k_hi, n_bins=5)
    assert swapped["side_a_win_rate"] == pytest.approx(1 - original["side_a_win_rate"])
    for a, b in zip(
        original["calibration_favourite"], swapped["calibration_favourite"], strict=True
    ):
        assert a["count"] == b["count"]
        if a["count"]:
            assert a["mean_p"] == pytest.approx(b["mean_p"])
            assert a["observed"] == pytest.approx(b["observed"])
    assert (
        binary_metrics(probs, np.array([2] * 60), np.array([2] * 60))["calibration_favourite"] == []
    )


def test_rps_and_log_score_reference_values() -> None:
    probs = np.array([[0.0, 0.0, 0.0, 0.0, 1.0], [0.2, 0.2, 0.2, 0.2, 0.2]])
    assert ranked_probability_score(probs[:1], np.array([4])) == 0.0
    # Uniform forecast, outcome 0: cumulative errors (0.8, 0.6, 0.4, 0.2) squared, over 4.
    expected = (0.64 + 0.36 + 0.16 + 0.04) / 4
    assert ranked_probability_score(probs[1:], np.array([0])) == pytest.approx(expected)
    assert np.isnan(ranked_probability_score(probs[:0], np.array([], dtype=int)))
    assert log_score(probs[1:], np.array([3]), np.array([4])) == pytest.approx(np.log(0.4))
    assert log_score(probs[:1], np.array([0]), np.array([0])) == pytest.approx(np.log(1e-12))
    assert np.isnan(log_score(probs[:0], np.array([], dtype=int), np.array([], dtype=int)))


def test_force_only_baseline() -> None:
    x = np.array([1.0, -1.0, 0.0, np.nan, 2.0, 0.5])
    k_lo = np.array([3, 0, 4, 4, 0, 2])
    k_hi = np.array([4, 1, 4, 4, 1, 2])
    out = force_only_baseline(x, k_lo, k_hi)
    assert out["n"] == 4
    assert out["accuracy"] == pytest.approx((1 + 1 + 0.5 + 0) / 4)
    assert force_only_baseline(np.full(3, np.nan), k_lo[:3], k_hi[:3])["n"] == 0


def _stub_sampling(
    monkeypatch: pytest.MonkeyPatch, design: ModelDesign, ok: list[bool]
) -> list[Any]:
    seen: list[Any] = []

    def fake_build(_design: ModelDesign, x: np.ndarray, _priors: Any, **kw: Any) -> str:
        seen.append(kw.get("loglik_weight"))
        return "model"

    def fake_fit(_model: Any, _mp: Any, *, seed: int, label: str) -> FitOutcome:
        i = int(label.rsplit("_", 1)[1])
        worst = {"max_rhat": 1.0, "min_ess_bulk": 2000.0, "divergences": 0}
        if not ok[i]:
            worst = {"max_rhat": 1.5, "min_ess_bulk": 10.0, "divergences": 3}
        return FitOutcome(SimpleNamespace(seed=seed), worst, ok[i], [])  # type: ignore[arg-type]

    monkeypatch.setattr(fit_module, "build_model", fake_build)
    monkeypatch.setattr(fit_module, "fit_with_retry", fake_fit)
    monkeypatch.setattr(
        fit_module, "extract_draws", lambda idata, _d: _draws(design, seed=idata.seed)
    )
    return seen


def test_fit_pooled_pools_when_all_rounds_converge(monkeypatch: pytest.MonkeyPatch) -> None:
    design = _design()
    seen = _stub_sampling(monkeypatch, design, [True, True])
    weight = np.array([1.0, 0.0, 1.0, 1.0])
    x_diffs = [np.zeros(design.n_battles), np.ones(design.n_battles)]
    result = fit_pooled(design, x_diffs, _mp(), seed=7, label="t", loglik_weight=weight)

    assert result.converged and result.pooled is not None
    assert [r.index for r in result.rounds] == [0, 1]
    assert all(r.draws is not None and r.draws.n_draws == 10 for r in result.rounds)
    assert all(w is weight for w in seen)
    other = repool(design, result.rounds, replacement_percentile=10.0, rank_by="skill_mean")
    assert other.war_mean.shape == result.pooled.war_mean.shape
    assert not np.allclose(other.war_mean, result.pooled.war_mean)


def test_fit_pooled_withholds_pool_when_a_round_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    design = _design()
    _stub_sampling(monkeypatch, design, [True, False])
    x_diffs = [np.zeros(design.n_battles)] * 2
    result = fit_pooled(design, x_diffs, _mp(), seed=0, label="t")

    assert not result.converged
    assert result.pooled is None
    assert result.worst["rounds_failed"] == [1]
    assert result.rounds[1].draws is None and result.rounds[1].reduction is None
    with pytest.raises(ValueError, match="repool"):
        repool(design, result.rounds, replacement_percentile=40.0, rank_by="total_war")
