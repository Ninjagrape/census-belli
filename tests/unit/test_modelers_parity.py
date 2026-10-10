"""The PyMC outcome potential must equal the numpy ordinal reference."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from pipeline.modelers.design import build_design, x_difference
from pipeline.modelers.model import (
    assemble_side_scores,
    build_joint_latent_model,
    build_model,
    joint_x_diff_prior,
)
from pipeline.modelers.ordinal import linear_predictor, log_likelihood
from pipeline.modelers.priors import ModelPriors
from pipeline.modelers.records import KIND_UNATTRIBUTED, KIND_UNKNOWN, CommanderInput, SideInput
from tests.model.model_sim import simulate_tournament


def _awkward_sides() -> tuple[list[SideInput], dict[int, float]]:
    """Tournament with unknown, default-split and grade-disagreement battles."""
    tour = simulate_tournament(
        7, generals_per_era=10, battles_per_era=60, p_unknown=0.1, p_pair=0.2
    )
    rng = np.random.default_rng(0)
    by_battle: dict[int, list[SideInput]] = {}
    for s in tour.sides:
        by_battle.setdefault(s.battle_id, []).append(s)
    out: list[SideInput] = []
    for group in by_battle.values():
        first, second = sorted(group, key=lambda s: s.side_id)
        if len(first.commanders) == 2 and rng.random() < 0.5:
            split = tuple(
                CommanderInput(c.general_id, 0.5, "default_split") for c in first.commanders
            )
            first = replace(first, commanders=split)
        if first.outcome == "victory" and rng.random() < 0.5:
            second = replace(second, outcome="decisive_defeat")
        elif first.outcome is None and second.outcome == "defeat" and rng.random() < 0.5:
            first = replace(first, outcome="decisive_victory")
        out += [first, second]
    return out, tour.log_troops


def test_pymc_potential_matches_numpy_log_likelihood() -> None:
    sides, log_troops = _awkward_sides()
    design = build_design(sides)
    assert (design.side_kind == KIND_UNKNOWN).any()
    assert (design.side_kind == KIND_UNATTRIBUTED).any()
    assert (design.k_lo != design.k_hi).any()
    x = x_difference(design, log_troops)
    model = build_model(design, x, ModelPriors())
    potential = model.compile_fn(
        model.replace_rvs_by_values([model.potentials[0]])[0],
        inputs=model.value_vars,
        on_unused_input="ignore",
    )
    rng = np.random.default_rng(0)
    point = model.initial_point()
    for name in point:
        point[name] = rng.normal(size=point[name].shape) * 0.7

    sigma = np.exp(point["sigma_skill_log__"])
    scores = assemble_side_scores(
        design,
        (sigma * point["z_skill"])[None],
        np.array([sigma]),
        point["z_side"][None],
        np.array([point["mu_unknown"]]),
        np.array([point["mu_unattributed"]]),
    )[0]
    c1 = float(np.exp(point["log_c1"]))
    c2 = c1 + float(np.exp(point["log_gap"]))
    eta = linear_predictor(
        design.side_a, design.side_b, scores, x, design.defender_diff, point["beta_force"], 0.0
    )

    np.testing.assert_allclose(
        float(potential(point)), log_likelihood(eta, design.k_lo, design.k_hi, c1, c2), rtol=1e-8
    )


def test_joint_prior_is_a_difference_and_ignores_absolute_level() -> None:
    sides, _ = _awkward_sides()
    design = build_design(sides)
    mean = np.full(design.n_sides, 9.5)
    sd = np.full(design.n_sides, 0.4)
    for index in (design.side_a[0], design.side_a[3], design.side_b[3]):
        mean[index] = np.nan
        sd[index] = np.nan
    mean[design.side_a[1]] = 11.0

    mu, sigma = joint_x_diff_prior(design, mean, sd, 1.0)
    shifted, _ = joint_x_diff_prior(design, mean + 5.0, sd, 1.0)

    assert mu[0] == 0.0 and np.isclose(sigma[0], np.sqrt(0.4**2 + 2.0))  # one missing
    assert np.isclose(mu[1], 1.5) and np.isclose(sigma[1], np.sqrt(2 * 0.4**2))  # both measured
    assert mu[3] == 0.0 and np.isclose(sigma[3], np.sqrt(2.0))  # both missing
    np.testing.assert_allclose(shifted, mu)  # a common level shift cancels
    model = build_joint_latent_model(design, mean, sd, 1.0, ModelPriors())
    assert "x_diff_latent" in model.named_vars
