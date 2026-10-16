"""Pooling arithmetic: mixture, Rubin's rules, widening, ranking, roll-up."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.modelers.pool import combine_rounds, mixture, rank_descending, rubin, widen_interval

ROUNDS = [np.array([[1.0], [3.0]]), np.array([[2.0], [4.0]]), np.array([[3.0], [5.0]])]


def test_rubin_matches_hand_computation() -> None:
    r = rubin(ROUNDS)
    # Round means 2, 3, 4; within variances all 2; B = var(2, 3, 4) = 1.
    assert r.mean[0] == pytest.approx(3.0)
    assert r.within[0] == pytest.approx(2.0)
    assert r.between[0] == pytest.approx(1.0)
    assert r.total[0] == pytest.approx(2.0 + 4.0 / 3.0)
    assert r.fmi[0] == pytest.approx((4.0 / 3.0) / (10.0 / 3.0))
    assert r.sd[0] == pytest.approx(np.sqrt(10.0 / 3.0))


def test_rubin_single_round_has_no_between_variance() -> None:
    r = rubin([ROUNDS[0]])
    assert r.between[0] == 0.0
    assert r.fmi[0] == 0.0


def test_mixture_takes_equal_draws_from_each_round() -> None:
    pooled = mixture([np.arange(4.0)[:, None], np.arange(10.0, 16.0)[:, None]])
    assert pooled.shape == (8, 1)
    assert pooled[:4, 0].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert pooled[4:, 0].tolist() == [10.0, 12.0, 13.0, 15.0]


def test_interval_is_widened_about_the_mixture_mean() -> None:
    pooled = mixture(ROUNDS)
    r = rubin(ROUNDS)
    # Mixture 1, 3, 2, 4, 3, 5: mean 3, variance 2; factor sqrt((10/3) / 2).
    factor = np.sqrt(5.0 / 3.0)
    lo, hi = widen_interval(pooled, r.total, 0.9)
    raw_lo, raw_hi = np.percentile(pooled[:, 0], [5, 95])
    assert lo[0] == pytest.approx(3.0 + factor * (raw_lo - 3.0))
    assert hi[0] == pytest.approx(3.0 + factor * (raw_hi - 3.0))
    assert hi[0] - lo[0] > raw_hi - raw_lo


def test_zero_variance_column_is_left_alone() -> None:
    pooled = np.full((6, 1), 2.0)
    lo, hi = widen_interval(pooled, np.zeros(1))
    assert lo[0] == hi[0] == 2.0


def test_ranking_breaks_ties_by_general_id() -> None:
    ids = np.array([5, 3, 9, 1])
    values = np.array([1.0, 1.0, 0.0, 2.0])
    assert rank_descending(ids, values).tolist() == [3, 2, 4, 1]


def test_combine_rounds_worst_case_and_flag() -> None:
    good = {"max_rhat": 1.01, "min_ess_bulk": 1500.0, "divergences": 0}
    slow = {"max_rhat": 1.03, "min_ess_bulk": 900.0, "divergences": 0}
    out = combine_rounds([good, slow])
    assert out["worst"] == {"max_rhat": 1.03, "min_ess_bulk": 900.0, "divergences": 0}
    assert out["converged"] == "false"
    assert out["rounds_failed"] == [1]
    ok = combine_rounds([good, {**good, "divergences": 0, "max_rhat": 1.0}])
    assert ok["converged"] == "true"


def test_combine_rounds_sums_divergences_and_treats_none_as_failure() -> None:
    a = {"max_rhat": 1.0, "min_ess_bulk": 2000.0, "divergences": 2}
    b = {"max_rhat": None, "min_ess_bulk": 2000.0, "divergences": 1}
    out = combine_rounds([a, b], max_divergences=5)
    assert out["converged"] == "false"
    assert out["worst"]["max_rhat"] is None
    assert out["worst"]["divergences"] == 3


def _shifted_idata() -> object:
    import arviz as az

    rng = np.random.default_rng(0)
    chains, draws, generals = 4, 400, 5
    noise = rng.normal(size=(chains, draws, generals))
    # Every chain parks the common shift somewhere else: the likelihood cannot
    # see it, so raw skill fails rhat while all differences mix fine.
    shift = np.array([-3.0, -1.0, 1.0, 3.0])[:, None, None]
    return az.from_dict(
        posterior={
            "skill": noise + shift,
            "z_skill": noise,
            "sigma_skill": 1.0 + 0.01 * rng.normal(size=(chains, draws)),
        },
        dims={"skill": ["general"], "z_skill": ["general"]},
    )


def test_round_convergence_gates_on_identified_quantities_and_keeps_raw_skill_for_reference() -> (
    None
):
    from pipeline.modelers.pool import round_convergence

    out = round_convergence(_shifted_idata())  # type: ignore[arg-type]

    assert out["max_rhat"] is not None and out["max_rhat"] < 1.02
    ref = out["reference"]
    assert ref["skill"]["max_rhat"] > 1.5
    assert ref["skill_centred"]["max_rhat"] < 1.02
    assert ref["z_skill"]["max_rhat"] < 1.02


def test_combine_rounds_carries_the_worst_reference_values() -> None:
    from pipeline.modelers.pool import round_convergence

    one = round_convergence(_shifted_idata())  # type: ignore[arg-type]
    out = combine_rounds([one, one])

    assert out["reference"]["skill"]["max_rhat"] == one["reference"]["skill"]["max_rhat"]
