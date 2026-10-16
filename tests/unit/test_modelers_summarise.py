"""Pooling, reduction and single-fit summaries on hand-built draws."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pipeline.modelers.design import build_design
from pipeline.modelers.model import PosteriorDraws, assemble_side_scores
from pipeline.modelers.records import ModelDesign
from pipeline.modelers.summarise import (
    adjacent_order_probability,
    assert_decomposition,
    pool_reductions,
    reduce_draws,
    summarise_single,
)
from tests.unit.modelers_helpers import battle

K = 40


def _design() -> ModelDesign:
    sides = (
        battle(1, (1,), (2,))
        + battle(2, (2,), (3, 4), a_outcome="defeat", b_outcome="victory")
        + battle(3, (1,), (3,), a_outcome="indecisive", b_outcome="indecisive")
        + battle(4, (4,), (1,))
    )
    return build_design(sides)


def _draws(design: ModelDesign, seed: int) -> PosteriorDraws:
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


def test_pooled_sd_widened_interval_and_decomposition() -> None:
    design = _design()
    x = np.linspace(-1.0, 1.0, design.n_battles)
    reductions = [reduce_draws(design, _draws(design, s), x, 40.0) for s in (1, 2)]

    pooled = pool_reductions(design, reductions, "total_war")

    for column in range(design.n_generals):
        a, b = (r.skills_c[:, column] for r in reductions)
        total = (a.var(ddof=1) + b.var(ddof=1)) / 2 + 1.5 * np.var([a.mean(), b.mean()], ddof=1)
        assert np.isclose(pooled.skill_sd[column], np.sqrt(total))
        mix = np.concatenate([a, b])
        centre = mix.mean()
        factor = np.sqrt(total / mix.var(ddof=1))
        expected_lo = centre + factor * (np.percentile(mix, 2.5) - centre)
        assert np.isclose(pooled.ci95_lo[column], expected_lo)
        wa, wb = (r.total_xwar[:, column] for r in reductions)
        w_total = (wa.var(ddof=1) + wb.var(ddof=1)) / 2 + 1.5 * np.var(
            [wa.mean(), wb.mean()], ddof=1
        )
        assert np.isclose(pooled.war_sd[column], np.sqrt(w_total))
    sums = np.zeros(design.n_generals)
    np.add.at(sums, design.cmd_general, pooled.entry_xwar)
    np.testing.assert_allclose(sums, pooled.war_mean, atol=1e-9)
    assert assert_decomposition(design, pooled) < 1e-9
    assert sorted(pooled.rank.tolist()) == list(range(1, design.n_generals + 1))


def test_single_fit_summary_uses_plain_posterior_moments() -> None:
    design = _design()
    reduction = reduce_draws(design, _draws(design, 3), np.zeros(design.n_battles), 40.0)

    single = summarise_single(design, reduction, "skill_mean")

    np.testing.assert_allclose(single.skill_sd, reduction.skills_c.std(axis=0, ddof=1))
    np.testing.assert_allclose(single.war_sd, reduction.total_xwar.std(axis=0, ddof=1))
    assert not single.skill_fmi.any()
    assert single.rank[np.argmax(single.skill_mean)] == 1


def test_pooled_war_interval_and_adjacent_order_probability() -> None:
    design = _design()
    x = np.linspace(-1.0, 1.0, design.n_battles)
    reductions = [reduce_draws(design, _draws(design, s), x, 40.0) for s in (1, 2)]

    pooled = pool_reductions(design, reductions, "total_war")

    assert pooled.war_draws.shape == (2 * K, design.n_generals)
    assert np.all(pooled.war_ci_lo < pooled.war_mean)
    assert np.all(pooled.war_mean < pooled.war_ci_hi)
    by_rank = np.argsort(pooled.rank)
    for upper, lower in zip(by_rank[:-1], by_rank[1:], strict=True):
        expected = np.mean(pooled.war_draws[:, upper] > pooled.war_draws[:, lower])
        assert np.isclose(pooled.p_above_next_rank[upper], expected)
    assert np.isnan(pooled.p_above_next_rank[by_rank[-1]])


def test_single_fit_war_interval_matches_quantiles() -> None:
    design = _design()
    reduction = reduce_draws(design, _draws(design, 3), np.zeros(design.n_battles), 40.0)

    single = summarise_single(design, reduction, "total_war")

    lo, hi = np.percentile(reduction.total_xwar, [2.5, 97.5], axis=0)
    np.testing.assert_allclose(single.war_ci_lo, lo)
    np.testing.assert_allclose(single.war_ci_hi, hi)
    assert np.isnan(single.p_above_next_rank[np.argmax(single.rank)])


def test_adjacent_order_probability_uses_paired_draws() -> None:
    # Perfectly correlated draws with a constant gap: i beats j in every draw
    # even though the marginal intervals overlap heavily. Independent-draw
    # shuffling would give about 0.6.
    base = np.linspace(-10.0, 10.0, 1000)
    draws = np.column_stack([base + 0.5, base, base - 3.0])
    rank = np.array([1, 2, 3])

    prob = adjacent_order_probability(draws, rank)

    assert prob[0] == 1.0
    assert prob[1] == 1.0
    assert np.isnan(prob[2])

    anti = np.column_stack([base, -base])
    half = adjacent_order_probability(anti, np.array([1, 2]))
    assert np.isclose(half[0], 0.5, atol=0.01)


def test_optional_term_summaries_are_rubin_pooled_with_the_defender_correlation() -> None:
    design = replace(_design(), defender_term_used=True)
    x = np.linspace(-1.0, 1.0, design.n_battles)
    reductions = []
    for seed in (1, 2, 3):
        rng = np.random.default_rng(seed)
        mu = rng.normal(size=K)
        draws = replace(
            _draws(design, seed),
            beta_def=0.5 * mu + 0.01 * rng.normal(size=K),
            mu_unknown=mu,
            delta_water_air=rng.normal(0.2, 0.1, K),
            beta_force_group=np.stack([np.full(K, 0.5), np.full(K, 0.7)], axis=1),
        )
        reductions.append(reduce_draws(design, draws, x, 40.0))
    terms = pool_reductions(design, reductions, "total_war").term_summaries
    assert set(terms) == {"beta_def", "mu_unknown", "delta_water_air", "beta_water_air"}
    assert terms["beta_water_air"]["mean"] == pytest.approx(0.7)
    assert terms["beta_def"]["sd_sqrt_T"] > 0
    assert terms["beta_def"]["corr_with_mu_unknown"] > 0.9
