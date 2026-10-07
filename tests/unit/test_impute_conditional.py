"""Unit tests for the design, the model's structure and the conditional draws."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.imputers.conditional import (
    PosteriorSamples,
    apply_mnar_delta,
    conditional_draw,
    draw_rounds,
    select_draw_indices,
    side_means,
)
from pipeline.imputers.design import ImputeDesign, build_design
from pipeline.imputers.model import build_model
from pipeline.imputers.priors import ImputePriors, load_priors
from pipeline.imputers.records import BattleLikelihood, SideRow
from tests.model.impute_sim import (
    ERA_YEARS,
    TRUE_A_ERA,
    TRUE_BETA_OUT,
    TRUE_BETA_TYPE,
    TRUE_SIGMA,
    TRUE_TAU_B,
    TRUE_TAU_C,
    TRUE_TAU_P,
    SimCorpus,
    simulate_corpus,
)

PRIORS = load_priors("config/imputation_priors.yaml")


def _side(
    side_id: int,
    battle_id: int,
    *,
    year: int | None = 1805,
    outcome: str | None = None,
    polity: str = "unknown",
    commander: str = "unknown",
    mnar: bool = False,
) -> SideRow:
    return SideRow(
        side_id=side_id,
        battle_id=battle_id,
        year=year,
        outcome=outcome,
        battle_type=None,
        is_defender=None,
        polity_key=polity,
        commander_key=commander,
        mnar=mnar,
    )


def _lik(
    battle_id: int, side_ids: tuple[int, ...], m: list[float], p: list[float]
) -> BattleLikelihood:
    return BattleLikelihood(battle_id, side_ids, np.array(m), np.diag(p), "joint")


def test_design_groups_battles_by_observed_side_count() -> None:
    sides = [_side(1, 1), _side(2, 1), _side(3, 2), _side(4, 2), _side(5, 3), _side(6, 3)]
    liks = [
        _lik(1, (1, 2), [9.0, 10.0], [4.0, 4.0]),
        _lik(2, (3,), [8.0], [2.0]),
    ]

    design = build_design(sides, liks, PRIORS)

    assert design.n_sides == 6
    assert design.n_observed == 3
    assert {g.k: len(g.positions) for g in design.groups} == {1: 1, 2: 1}
    two = next(g for g in design.groups if g.k == 2)
    np.testing.assert_allclose(two.cov[0], np.diag([0.25, 0.25]))
    assert design.is_observed.tolist() == [True, True, True, False, False, False]
    # Battle 3 has no observed side but still has a block, so it gets imputed.
    assert [b.battle_id for b in design.battles] == [1, 2, 3]
    assert len(design.battles[2].obs_local) == 0


def test_design_rejects_a_likelihood_for_an_unknown_side() -> None:
    with pytest.raises(ValueError, match="not in scope"):
        build_design([_side(1, 1)], [_lik(1, (99,), [1.0], [1.0])], PRIORS)


def test_undated_battles_get_their_own_era() -> None:
    sides = [_side(1, 1, year=-300), _side(2, 2, year=None)]

    design = build_design(sides, [_lik(1, (1,), [9.0], [1.0])], PRIORS)

    assert design.eras[design.era_index[0]] == "ancient"
    assert design.eras[design.era_index[1]] == "undated"


def test_random_levels_seen_only_on_missing_sides_are_not_sampled() -> None:
    sides = [
        _side(1, 1, commander="a"),
        _side(2, 1, commander="b"),
        _side(3, 2, commander="c"),
    ]
    design = build_design(sides, [_lik(1, (1, 2), [9.0, 9.0], [1.0, 1.0])], PRIORS)

    assert design.commander.sampled == ("a", "b")
    assert design.commander.levels == ("a", "b", "c")
    assert design.commander.index.tolist() == [0, 1, -1]


def test_model_uses_data_containers_and_no_variable_shares_a_dimension_name() -> None:
    pytest.importorskip("pymc")
    sides = [
        _side(i, 1 + i // 2, outcome=("victory" if i % 2 else "defeat"), polity=f"Q{i % 3}")
        for i in range(12)
    ]
    liks = [_lik(b, (2 * b - 2, 2 * b - 1), [9.0, 9.5], [1.0, 1.0]) for b in range(1, 5)]
    design = build_design(sides, liks, PRIORS)

    model = build_model(design)

    named = {v.name for v in model.free_RVs + model.deterministics + model.observed_RVs}
    assert not named & set(model.coords), "a variable shares a name with a dimension"
    data_names = {v.name for v in model.data_vars}
    assert {"data_era_index", "data_pos_k2", "data_y_k2", "data_cov_k2"} <= data_names
    assert {"tau_b", "sigma", "a_era", "beta_out"} <= named


def test_model_refuses_a_corpus_with_no_observed_sides() -> None:
    pytest.importorskip("pymc")
    design = build_design([_side(1, 1)], [], PRIORS)

    with pytest.raises(ValueError, match="no observed sides"):
        build_model(design)


def test_select_draw_indices_spreads_over_chains_and_time() -> None:
    indices = select_draw_indices(n_chains=4, n_draws=1000, n_rounds=20)

    assert len(indices) == len(set(indices)) == 20
    assert {i // 1000 for i in indices} == {0, 1, 2, 3}
    chain0 = sorted(i % 1000 for i in indices if i // 1000 == 0)
    assert chain0[0] > 0 and chain0[-1] < 999
    assert np.diff(chain0).min() >= 150


def test_select_draw_indices_handles_fewer_rounds_than_chains() -> None:
    assert select_draw_indices(4, 100, 2) == [50, 150]


def _posterior(n: int, *, a_era: float, tau_b: float, sigma: float) -> PosteriorSamples:
    eras = len(PRIORS.era_names)
    return PosteriorSamples(
        a_era=np.full((n, eras), a_era),
        beta_out=None,
        beta_type=None,
        beta_def=None,
        v_polity=None,
        c_cmd=None,
        tau_p=None,
        tau_c=None,
        tau_b=np.full(n, tau_b),
        sigma=np.full(n, sigma),
        n_chains=1,
        n_draws=n,
    )


def test_conditional_draw_matches_the_analytic_gaussian_posterior() -> None:
    # Two sides, one observed. Prior covariance S = tau^2 J + sigma^2 I.
    tau, sigma, p, m = 0.7, 0.5, 4.0, 11.0
    mu = np.array([10.0, 9.0])
    sides = [_side(1, 1), _side(2, 1)]
    design = build_design(sides, [_lik(1, (1,), [m], [p])], PRIORS)
    prior_cov = tau**2 * np.ones((2, 2)) + sigma**2 * np.eye(2)

    rng = np.random.default_rng(3)
    draws = np.array([conditional_draw(design, mu, tau, sigma, rng) for _ in range(40000)])

    noise_var = prior_cov[0, 0] + 1.0 / p
    gain = prior_cov[1, 0] / noise_var
    expected_mean = [
        mu[0] + prior_cov[0, 0] / noise_var * (m - mu[0]),
        mu[1] + gain * (m - mu[0]),
    ]
    expected_var_missing = prior_cov[1, 1] - prior_cov[1, 0] ** 2 / noise_var
    np.testing.assert_allclose(draws.mean(axis=0), expected_mean, atol=0.02)
    assert draws[:, 1].var() == pytest.approx(expected_var_missing, rel=0.05)
    # Overimputation: the observed side is redrawn, not copied through.
    assert draws[:, 0].std() > 0.1


def test_battle_with_no_observed_side_draws_from_the_prior() -> None:
    design = build_design([_side(1, 1), _side(2, 1)], [], PRIORS)
    tau, sigma = 0.6, 0.4

    rng = np.random.default_rng(4)
    draws = np.array(
        [conditional_draw(design, np.array([9.0, 9.0]), tau, sigma, rng) for _ in range(30000)]
    )

    assert draws.mean() == pytest.approx(9.0, abs=0.02)
    assert draws[:, 0].var() == pytest.approx(tau**2 + sigma**2, rel=0.05)
    assert np.cov(draws.T)[0, 1] == pytest.approx(tau**2, rel=0.08)


def test_two_missing_sides_of_one_unseen_commander_share_an_effect() -> None:
    sides = [
        _side(1, 1, commander="seen_a"),
        _side(2, 1, commander="seen_b"),
        _side(3, 2, commander="ghost"),
        _side(4, 3, commander="ghost"),
    ]
    design = build_design(sides, [_lik(1, (1, 2), [9.0, 9.0], [1.0, 1.0])], PRIORS)
    n = 400
    post = PosteriorSamples(
        a_era=np.zeros((n, len(PRIORS.era_names))),
        beta_out=None,
        beta_type=None,
        beta_def=None,
        v_polity=None,
        c_cmd=np.zeros((n, 2)),
        tau_p=None,
        tau_c=np.full(n, 2.0),
        tau_b=np.full(n, 1.0),
        sigma=np.full(n, 1.0),
        n_chains=1,
        n_draws=n,
    )

    means = [side_means(post, 0, design, np.random.default_rng(i)) for i in range(300)]

    effects = np.array(means)
    assert np.all(effects[:, 2] == effects[:, 3])
    assert effects[:, 2].std() == pytest.approx(2.0, rel=0.15)
    assert np.all(effects[:, 0] == 0.0)


def test_apply_mnar_delta_moves_only_imputed_mnar_sides() -> None:
    values = np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
    imputed = np.array([False, True, True, False])
    mnar = np.array([True, True, False, False])

    out = apply_mnar_delta(values, imputed, mnar, -0.5)

    np.testing.assert_allclose(out[0], [1.0, 1.5, 3.0, 4.0])
    np.testing.assert_allclose(values[0], [1.0, 2.0, 3.0, 4.0])  # input untouched
    np.testing.assert_allclose(apply_mnar_delta(values, imputed, mnar, 0.0), values)


def test_draw_rounds_is_deterministic_and_applies_the_delta() -> None:
    sides = [_side(1, 1, mnar=False), _side(2, 1, mnar=True)]
    design = build_design(sides, [_lik(1, (1,), [9.0], [4.0])], PRIORS)
    post = _posterior(200, a_era=9.0, tau_b=0.5, sigma=0.5)

    first = draw_rounds(design, post, n_rounds=5, seed=7, mnar_delta=0.0)
    again = draw_rounds(design, post, n_rounds=5, seed=7, mnar_delta=0.0)
    shifted = draw_rounds(design, post, n_rounds=5, seed=7, mnar_delta=0.25)

    np.testing.assert_array_equal(first.values, again.values)
    assert first.is_imputed.tolist() == [False, True]
    np.testing.assert_allclose(shifted.values[:, 0], first.values[:, 0])
    np.testing.assert_allclose(shifted.values[:, 1], first.values[:, 1] + 0.25)


def test_priors_object_is_not_mutated_by_design() -> None:
    before: ImputePriors = load_priors("config/imputation_priors.yaml")
    build_design([_side(1, 1)], [], before)

    assert before == load_priors("config/imputation_priors.yaml")


def _oracle_posterior(corpus: SimCorpus, design: ImputeDesign) -> PosteriorSamples:
    """A one-draw 'posterior' sitting exactly on the simulation's true parameters."""
    year_of_era = {
        ERA_YEARS.index(s.year): int(design.era_index[i]) for i, s in enumerate(corpus.sides)
    }
    a_era = np.empty(len(design.eras))
    for sim_era, model_era in year_of_era.items():
        a_era[model_era] = TRUE_A_ERA[sim_era]

    def levels(
        block_levels: tuple[str, ...], truth: dict[str, float] | dict[str, float]
    ) -> np.ndarray:
        return np.array([[truth[name] for name in block_levels]])

    return PosteriorSamples(
        a_era=a_era[None, :],
        beta_out=levels(design.outcome.levels, TRUE_BETA_OUT),
        beta_type=levels(design.battle_type.levels, TRUE_BETA_TYPE),
        beta_def=None,
        v_polity=levels(design.polity.sampled, corpus.polity_effect),
        c_cmd=levels(design.commander.sampled, corpus.commander_effect),
        tau_p=np.array([TRUE_TAU_P]),
        tau_c=np.array([TRUE_TAU_C]),
        tau_b=np.array([TRUE_TAU_B]),
        sigma=np.array([TRUE_SIGMA]),
        n_chains=1,
        n_draws=1,
    )


def _slope(outcome: np.ndarray, ratio: np.ndarray) -> float:
    centred = ratio - ratio.mean()
    return float((centred * (outcome - outcome.mean())).sum() / (centred**2).sum())


def test_oracle_parameters_give_an_unattenuated_outcome_slope() -> None:
    """With the true parameters, no MCMC, the imputed-ratio slope matches the true one.

    Pins the conditional draws and the overimputation logic on their own, so a
    failure here is the draw code and not the sampler. The model-marker test
    does the same with a fitted posterior and a looser band.
    """
    corpus = simulate_corpus(3, n_battles=1500, p_missing=0.3)
    design = build_design(corpus.sides, corpus.likelihoods, PRIORS)
    post = _oracle_posterior(corpus, design)
    row_of = {int(s): i for i, s in enumerate(design.side_ids)}
    battles = sorted(corpus.two_sided)
    outcome = np.array([corpus.outcome_sign_a[b] for b in battles], dtype=float)
    a_rows = np.array([row_of[corpus.two_sided[b][0]] for b in battles])
    b_rows = np.array([row_of[corpus.two_sided[b][1]] for b in battles])
    truth = np.array(
        [
            corpus.true_y[corpus.two_sided[b][0]] - corpus.true_y[corpus.two_sided[b][1]]
            for b in battles
        ]
    )

    ratios = []
    for seed in range(10):
        values = draw_rounds(design, post, n_rounds=1, seed=seed, mnar_delta=0.0).values[0]
        ratios.append(_slope(outcome, values[a_rows] - values[b_rows]) / _slope(outcome, truth))

    assert np.mean(ratios) == pytest.approx(1.0, abs=0.05), ratios
