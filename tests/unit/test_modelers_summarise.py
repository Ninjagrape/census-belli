"""Pooling, reduction and single-fit summaries on hand-built draws."""

from __future__ import annotations

import numpy as np

from pipeline.modelers.design import build_design
from pipeline.modelers.model import PosteriorDraws, assemble_side_scores
from pipeline.modelers.records import ModelDesign
from pipeline.modelers.summarise import (
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
