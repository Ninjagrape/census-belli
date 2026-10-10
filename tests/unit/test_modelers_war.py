"""WAR arithmetic on hand-built posterior draws."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.modelers.design import build_design
from pipeline.modelers.model import assemble_side_scores
from pipeline.modelers.ordinal import expected_score
from pipeline.modelers.records import KIND_NAMED
from pipeline.modelers.war import centre_skills, compute_war, replacement_level
from tests.unit.modelers_helpers import battle

N_DRAWS = 40


def _setup() -> tuple[object, dict[str, np.ndarray]]:
    sides = (
        battle(1, (1, 2), (3,), method="rule_roles", a_outcome="victory", b_outcome="defeat")
        + battle(2, (), (1,), a_outcome="defeat", b_outcome="victory")
        + battle(3, (4,), (5,), a_outcome="indecisive", b_outcome="indecisive")
    )
    design = build_design(sides)
    rng = np.random.default_rng(0)
    skills = rng.normal(size=(N_DRAWS, design.n_generals))
    scores = assemble_side_scores(
        design,
        skills,
        np.ones(N_DRAWS),
        rng.normal(size=(N_DRAWS, design.latent_sides.size)),
        rng.normal(size=N_DRAWS),
        None,
    )
    pieces = {
        "skills": skills,
        "side_scores": scores,
        "x_diff": rng.normal(size=design.n_battles),
        "beta_force": rng.normal(0.5, 0.1, N_DRAWS),
        "beta_def": np.zeros(N_DRAWS),
        "c1": np.full(N_DRAWS, 0.6),
        "c2": np.full(N_DRAWS, 2.0),
    }
    return design, pieces


def test_contributions_on_a_side_sum_to_the_side_value() -> None:
    design, p = _setup()
    war = compute_war(design, **p)  # type: ignore[arg-type]
    for side_index in range(design.n_sides):
        entries = design.cmd_side == side_index
        if design.side_kind[side_index] != KIND_NAMED:
            assert not entries.any()
            continue
        assert np.allclose(war.xwar_entry[:, entries].sum(axis=1), war.xwar_side[:, side_index])
        assert np.allclose(
            war.realised_entry[:, entries].sum(axis=1), war.realised_side[:, side_index]
        )


def test_totals_are_sums_of_entries() -> None:
    design, p = _setup()
    war = compute_war(design, **p)  # type: ignore[arg-type]
    for g in range(design.n_generals):
        assert np.allclose(war.total_xwar[:, g], war.xwar_entry[:, design.cmd_general == g].sum(1))


def test_xwar_zero_when_side_equals_replacement() -> None:
    sides = battle(1, (1,), (2,)) + battle(2, (3,), (4,)) + battle(3, (5,), (6,))
    design = build_design(sides)
    skills = np.tile(np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]), (3, 1))
    scores = assemble_side_scores(design, skills, np.ones(3), None, None, None)
    war = compute_war(
        design,
        skills,
        scores,
        np.array([0.3, -0.2, 0.0]),
        np.full(3, 0.5),
        np.zeros(3),
        np.full(3, 0.6),
        np.full(3, 2.0),
        replacement_percentile=40.0,
    )
    s_rep = replacement_level(skills, 40.0)
    assert s_rep[0] == pytest.approx(2.0)
    third_general = list(design.general_ids).index(3)
    assert np.allclose(war.total_xwar[:, third_general], 0.0)


def test_replacement_level_and_zero_sum_between_opponents() -> None:
    design, p = _setup()
    war = compute_war(design, **p)  # type: ignore[arg-type]
    assert np.allclose(war.s_rep, np.percentile(p["skills"], 40.0, axis=1))
    a, b = design.side_a, design.side_b
    assert np.allclose(war.e_model[:, a] + war.e_model[:, b], 1.0)


def test_e_rep_matches_manual_computation() -> None:
    design, p = _setup()
    war = compute_war(design, **p)  # type: ignore[arg-type]
    t, battle_index = 3, 0
    a, b = design.side_a[battle_index], design.side_b[battle_index]
    force = p["beta_force"][t] * p["x_diff"][battle_index]
    eta_rep = war.s_rep[t] - p["side_scores"][t, b] + force
    assert war.e_rep[t, a] == pytest.approx(float(expected_score(eta_rep, 0.6, 2.0)))
    eta_rep_b = war.s_rep[t] - p["side_scores"][t, a] - force
    assert war.e_rep[t, b] == pytest.approx(float(expected_score(eta_rep_b, 0.6, 2.0)))


def test_realised_uses_the_observed_score() -> None:
    design, p = _setup()
    war = compute_war(design, **p)  # type: ignore[arg-type]
    assert np.allclose(war.realised_side[:, 0], design.y_a[0] - war.e_rep[:, 0])
    assert np.allclose(war.realised_side[:, 1], 1.0 - design.y_a[0] - war.e_rep[:, 1])


def test_centre_skills_zero_mean_per_draw() -> None:
    skills = np.random.default_rng(1).normal(size=(5, 7)) + 3.0
    assert np.allclose(centre_skills(skills).mean(axis=1), 0.0)
