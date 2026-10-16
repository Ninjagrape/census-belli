"""The PyMC outcome potential must equal the numpy ordinal reference."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pipeline.modelers.design import build_design, x_difference
from pipeline.modelers.model import (
    assemble_side_scores,
    build_joint_latent_model,
    build_model,
    joint_x_diff_prior,
)
from pipeline.modelers.ordinal import linear_predictor, log_likelihood, log_likelihood_pointwise
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


def _random_point(model: object, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    point = model.initial_point()  # type: ignore[attr-defined]
    for name in point:
        point[name] = rng.normal(size=point[name].shape) * 0.7
    return point  # type: ignore[no-any-return]


def test_loglik_weight_none_leaves_graph_unweighted() -> None:
    sides, log_troops = _awkward_sides()
    design = build_design(sides)
    x = x_difference(design, log_troops)
    plain = build_model(design, x, ModelPriors())
    ones = build_model(design, x, ModelPriors(), loglik_weight=np.ones(design.n_battles))

    assert "data_loglik_weight" not in plain.named_vars
    assert "data_loglik_weight" in ones.named_vars
    assert [v.name for v in plain.free_RVs] == [v.name for v in ones.free_RVs]
    assert [p.name for p in plain.potentials] == [p.name for p in ones.potentials]

    point = _random_point(plain)
    np.testing.assert_allclose(
        float(plain.compile_logp()(point)), float(ones.compile_logp()(point)), rtol=1e-12
    )


def test_loglik_weight_zero_removes_only_that_battles_term() -> None:
    sides, log_troops = _awkward_sides()
    design = build_design(sides)
    x = x_difference(design, log_troops)
    weight = np.ones(design.n_battles)
    weight[:5] = 0.0
    full = build_model(design, x, ModelPriors())
    masked = build_model(design, x, ModelPriors(), loglik_weight=weight)
    point = _random_point(full, seed=3)

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
    pointwise = log_likelihood_pointwise(eta, design.k_lo, design.k_hi, c1, c2)

    diff = float(full.compile_logp()(point)) - float(masked.compile_logp()(point))
    np.testing.assert_allclose(diff, pointwise[:5].sum(), rtol=1e-8)


def test_loglik_weight_is_validated() -> None:
    sides, log_troops = _awkward_sides()
    design = build_design(sides)
    x = x_difference(design, log_troops)
    for bad in (
        np.ones(design.n_battles + 1),
        np.full(design.n_battles, 1.5),
        np.full(design.n_battles, -0.1),
        np.full(design.n_battles, np.nan),
    ):
        with pytest.raises(ValueError, match="loglik_weight"):
            build_model(design, x, ModelPriors(), loglik_weight=bad)


def _joint_inputs(design: object) -> tuple[np.ndarray, np.ndarray]:
    n_sides = design.n_sides  # type: ignore[attr-defined]
    mean = np.full(n_sides, 9.5)
    sd = np.full(n_sides, 0.4)
    mean[::3] = np.nan  # a mix of measured and unmeasured sides
    sd[::3] = np.nan
    return mean, sd


def test_joint_loglik_weight_none_leaves_graph_unweighted() -> None:
    sides, _ = _awkward_sides()
    design = build_design(sides)
    mean, sd = _joint_inputs(design)
    plain = build_joint_latent_model(design, mean, sd, 1.0, ModelPriors())
    ones = build_joint_latent_model(
        design, mean, sd, 1.0, ModelPriors(), loglik_weight=np.ones(design.n_battles)
    )

    assert "data_loglik_weight" not in plain.named_vars
    assert "data_loglik_weight" in ones.named_vars
    assert [v.name for v in plain.free_RVs] == [v.name for v in ones.free_RVs]
    assert [p.name for p in plain.potentials] == [p.name for p in ones.potentials]
    point = _random_point(plain)
    np.testing.assert_allclose(
        float(plain.compile_logp()(point)), float(ones.compile_logp()(point)), rtol=1e-12
    )


def test_joint_loglik_weight_zero_removes_only_that_battles_term() -> None:
    sides, _ = _awkward_sides()
    design = build_design(sides)
    mean, sd = _joint_inputs(design)
    weight = np.ones(design.n_battles)
    weight[:5] = 0.0
    full = build_joint_latent_model(design, mean, sd, 1.0, ModelPriors())
    masked = build_joint_latent_model(design, mean, sd, 1.0, ModelPriors(), loglik_weight=weight)
    point = _random_point(full, seed=3)

    sigma = np.exp(point["sigma_skill_log__"])
    scores = assemble_side_scores(
        design,
        (sigma * point["z_skill"])[None],
        np.array([sigma]),
        point["z_side"][None],
        np.array([point["mu_unknown"]]),
        np.array([point["mu_unattributed"]]),
    )[0]
    mu, x_sd = joint_x_diff_prior(design, mean, sd, 1.0)
    x = mu + x_sd * point["z_x"]
    c1 = float(np.exp(point["log_c1"]))
    c2 = c1 + float(np.exp(point["log_gap"]))
    eta = linear_predictor(
        design.side_a, design.side_b, scores, x, design.defender_diff, point["beta_force"], 0.0
    )
    pointwise = log_likelihood_pointwise(eta, design.k_lo, design.k_hi, c1, c2)

    diff = float(full.compile_logp()(point)) - float(masked.compile_logp()(point))
    np.testing.assert_allclose(diff, pointwise[:5].sum(), rtol=1e-8)
    # The held-out battles keep their latent difference, which then samples its prior.
    assert "x_diff_latent" in masked.named_vars and "z_x" in masked.named_vars


def test_joint_loglik_weight_is_validated() -> None:
    sides, _ = _awkward_sides()
    design = build_design(sides)
    mean, sd = _joint_inputs(design)
    for bad in (
        np.ones(design.n_battles + 1),
        np.full(design.n_battles, 1.5),
        np.full(design.n_battles, -0.1),
        np.full(design.n_battles, np.nan),
    ):
        with pytest.raises(ValueError, match="loglik_weight"):
            build_joint_latent_model(design, mean, sd, 1.0, ModelPriors(), loglik_weight=bad)


def test_group_slopes_and_defender_term_match_the_numpy_reference() -> None:
    sides, log_troops = _awkward_sides()
    rng = np.random.default_rng(3)
    marked: list[SideInput] = []
    flag_of: dict[int, bool] = {}
    type_of: dict[int, str] = {}
    for s in sorted(sides, key=lambda s: s.side_id):
        type_of.setdefault(s.battle_id, "naval" if rng.random() < 0.3 else "field")
        # The first side seen in a battle defends, the second attacks.
        flag = flag_of.setdefault(s.battle_id, True)
        flag_of[s.battle_id] = not flag
        marked.append(replace(s, is_defender=flag, battle_type=type_of[s.battle_id]))
    design = build_design(marked)
    assert design.defender_term_used
    assert design.n_force_groups == 2
    x = x_difference(design, log_troops)
    model = build_model(design, x, ModelPriors())
    assert {"delta_water_air", "beta_force_group", "beta_def"} <= set(model.named_vars)

    potential = model.compile_fn(
        model.replace_rvs_by_values([model.potentials[0]])[0],
        inputs=model.value_vars,
        on_unused_input="ignore",
    )
    point = _random_point(model)
    sigma = np.exp(point["sigma_skill_log__"])
    scores = assemble_side_scores(
        design,
        (sigma * point["z_skill"])[None],
        np.array([sigma]),
        point["z_side"][None],
        np.array([point["mu_unknown"]]),
        np.array([point["mu_unattributed"]]),
    )[0]
    group = np.array([point["beta_force"], point["beta_force"] + point["delta_water_air"]])
    c1 = float(np.exp(point["log_c1"]))
    c2 = c1 + float(np.exp(point["log_gap"]))
    eta = linear_predictor(
        design.side_a,
        design.side_b,
        scores,
        x,
        design.defender_diff,
        group[design.force_group],
        point["beta_def"],
    )
    np.testing.assert_allclose(
        float(potential(point)), log_likelihood(eta, design.k_lo, design.k_hi, c1, c2), rtol=1e-8
    )


def test_single_slope_design_has_no_group_variables_and_no_defender_when_unflagged() -> None:
    sides, log_troops = _awkward_sides()
    design = build_design(sides)
    model = build_model(design, x_difference(design, log_troops), ModelPriors())
    assert design.n_force_groups == 1
    assert not {"delta_water_air", "beta_force_group", "beta_def"} & set(model.named_vars)


def test_partial_camp_potential_matches_numpy_reference() -> None:
    sides, log_troops = _awkward_sides()
    named = [
        CommanderInput(9001, 1.0, "rule_single"),
    ]
    camp = [
        SideInput(
            side_id=900_001,
            battle_id=900_000,
            year=1800,
            outcome="victory",
            is_defender=None,
            commanders=tuple(named),
            log_troops=1.0,
        ),
        SideInput(
            side_id=900_002,
            battle_id=900_000,
            year=1800,
            outcome="victory",
            is_defender=None,
            commanders=(),
            log_troops=2.0,
        ),
        SideInput(
            side_id=900_003,
            battle_id=900_000,
            year=1800,
            outcome="defeat",
            is_defender=None,
            commanders=(CommanderInput(9002, 1.0, "rule_single"),),
            log_troops=1.5,
        ),
    ]
    design = build_design([*sides, *camp])
    assert design.counts.battles_merged_camps == 1
    partial = np.flatnonzero((design.side_unknown_share > 0) & (design.side_kind == 0))
    assert partial.size == 1
    x = x_difference(design, {**log_troops, 900_001: 1.0, 900_002: 2.0, 900_003: 1.5})
    model = build_model(design, x, ModelPriors())
    potential = model.compile_fn(
        model.replace_rvs_by_values([model.potentials[0]])[0],
        inputs=model.value_vars,
        on_unused_input="ignore",
    )
    point = _random_point(model)
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
