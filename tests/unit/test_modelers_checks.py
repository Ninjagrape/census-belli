"""The dataset-level gates, on small hand-built inputs."""

from __future__ import annotations

import numpy as np

from pipeline.modelers.checks import (
    circularity_check,
    exclusion_concentration,
    extreme_ratio_dependence,
    fmi_top_generals,
    implausible_career_span,
    own_minus_opponent,
    posterior_predictive_category_check,
    prior_predictive,
    prior_predictive_sensible,
    top_indices,
)
from pipeline.modelers.design import build_design
from pipeline.modelers.model import PosteriorDraws, assemble_side_scores
from pipeline.modelers.priors import ModelPriors
from pipeline.modelers.records import ModelCounts
from tests.unit.modelers_helpers import battle

IDS = np.arange(100, 160)


def test_top_indices_ties_by_id() -> None:
    assert top_indices(np.array([7, 3, 5]), np.array([1.0, 1.0, 0.0]), 2).tolist() == [1, 0]


def _circ(**overrides: object) -> object:
    rng = np.random.default_rng(0)
    skill = rng.normal(size=60)
    args: dict[str, object] = {
        "general_ids": IDS,
        "beta_main": 0.50,
        "beta_joint_mean": 0.52,
        "beta_joint_sd": 0.1,
        "skill_main": skill,
        "skill_joint": skill + rng.normal(0, 0.05, 60),
        "score_main": skill,
        "score_joint": skill,
        "battles_per_general": np.full(60, 5),
    }
    args.update(overrides)
    return circularity_check(**args)  # type: ignore[arg-type]


def test_circularity_passes_when_fits_agree() -> None:
    assert _circ().passed  # type: ignore[attr-defined]


def test_circularity_fails_on_slope_gap() -> None:
    assert not _circ(beta_main=0.9).passed  # type: ignore[attr-defined]


def test_circularity_fails_on_uncorrelated_skills() -> None:
    noise = np.random.default_rng(5).normal(size=60)
    result = _circ(skill_joint=noise, score_joint=noise)
    assert not result.passed  # type: ignore[attr-defined]


def test_circularity_scales_overlap_requirement_for_few_generals() -> None:
    skill = np.arange(10.0)
    result = _circ(
        general_ids=IDS[:10],
        skill_main=skill,
        skill_joint=skill,
        score_main=skill,
        score_joint=skill,
        battles_per_general=np.full(10, 5),
    )
    assert result.passed  # type: ignore[attr-defined]
    assert result.details["top_overlap_required"] == 8.0  # type: ignore[attr-defined]


def test_extreme_ratio_flags_dependent_general() -> None:
    entry_general = np.array([0, 0, 1, 1])
    xwar = np.array([0.3, 0.1, 0.2, 0.2])
    ratio = np.array([-3.0, 0.0, -3.0, 1.0])
    result = extreme_ratio_dependence(
        general_ids=np.array([10, 11]),
        entry_general=entry_general,
        entry_xwar=xwar,
        entry_log_ratio=ratio,
        top_idx=np.array([0, 1]),
    )
    assert not result.passed
    assert [o["general_id"] for o in result.details["offenders"]] == [10.0]


def test_own_minus_opponent_signs() -> None:
    design = build_design(battle(1, (1,), (2,)))
    assert own_minus_opponent(design, np.array([1.5])).tolist() == [1.5, -1.5]


def test_exclusion_concentration() -> None:
    counts = ModelCounts(sides_multi_commander_all=10, sides_default_split_all=2)
    ids = np.array([1, 2])
    ok = exclusion_concentration(
        general_ids=ids,
        n_included=np.array([8, 6]),
        n_uncredited=np.array([1, 4]),
        top_idx=np.array([0, 1]),
        counts=counts,
    )
    assert ok.passed
    bad = exclusion_concentration(
        general_ids=ids,
        n_included=np.array([8, 2]),
        n_uncredited=np.array([1, 8]),
        top_idx=np.array([0, 1]),
        counts=counts,
    )
    assert not bad.passed
    assert bad.details["low_credit_generals"] == [2]
    heavy = exclusion_concentration(
        general_ids=ids,
        n_included=np.array([8, 6]),
        n_uncredited=np.array([1, 4]),
        top_idx=np.array([0, 1]),
        counts=ModelCounts(sides_multi_commander_all=10, sides_default_split_all=5),
    )
    assert not heavy.passed


def test_career_span_and_fmi_gates() -> None:
    spans = np.array([[1800.0, 1815.0, 15.0], [1700.0, 1790.0, 90.0], [np.nan] * 3])
    result = implausible_career_span(general_ids=np.array([1, 2, 3]), spans=spans)
    assert not result.passed
    assert result.details["generals"] == [{"general_id": 2, "span_years": 90.0}]
    fmi = fmi_top_generals(
        general_ids=np.array([1, 2]), fmi=np.array([0.1, 0.5]), top_idx=np.array([0, 1])
    )
    assert not fmi.passed
    assert fmi.details["high_fmi_generals"] == [2]


def _tournament_design() -> tuple[object, np.ndarray]:
    rng = np.random.default_rng(2)
    levels = ["decisive_defeat", "defeat", "indecisive", "victory", "decisive_victory"]
    sides = []
    for b in range(1, 121):
        a, c = rng.choice(10, 2, replace=False) + 1
        k = int(rng.integers(0, 5))
        sides += battle(b, (int(a),), (int(c),), a_outcome=levels[k], b_outcome=levels[4 - k])
    design = build_design(sides)
    return design, rng.normal(size=design.n_battles)


def test_prior_predictive_summaries_are_consistent() -> None:
    design, x = _tournament_design()
    pp = prior_predictive(design, x, ModelPriors(), n_draws=200, seed=1)  # type: ignore[arg-type]
    assert pp.category_freq.sum() == np.float64(pp.category_freq.sum())
    assert abs(pp.category_freq.sum() - 1.0) < 1e-9
    assert prior_predictive_sensible(pp).passed
    wide = prior_predictive(design, x, ModelPriors(skill_prior_sd=8.0), n_draws=200, seed=1)  # type: ignore[arg-type]
    assert not prior_predictive_sensible(wide).passed


def test_posterior_predictive_check_passes_for_the_generating_model() -> None:
    rng = np.random.default_rng(11)
    levels = ["decisive_defeat", "defeat", "indecisive", "victory", "decisive_victory"]
    skills_true = rng.normal(size=10)
    sides = []
    from pipeline.modelers.ordinal import category_probs

    for b in range(1, 801):
        a, c = rng.choice(10, 2, replace=False)
        probs = category_probs(skills_true[a] - skills_true[c], 0.6, 2.0)
        k = int(rng.choice(5, p=probs))
        sides += battle(b, (int(a) + 1,), (int(c) + 1,), a_outcome=levels[k], b_outcome=None)
    design = build_design(sides)
    skills = np.tile(skills_true[design.general_ids - 1], (50, 1))
    draws = PosteriorDraws(
        skills=skills,
        side_scores=assemble_side_scores(design, skills, np.ones(50), None, None, None),
        beta_force=np.zeros(50),
        beta_def=np.zeros(50),
        c1=np.full(50, 0.6),
        c2=np.full(50, 2.0),
        sigma_skill=np.ones(50),
    )
    # Five levels at 95% would reject a correct model about a fifth of the time.
    result = posterior_predictive_category_check(
        design, draws, np.zeros(design.n_battles), level=0.999, seed=1
    )
    assert result.passed, result.message
    # A model with the wrong cutpoints must fail.
    wrong = PosteriorDraws(**{**draws.__dict__, "c1": np.full(50, 2.5), "c2": np.full(50, 4.0)})
    assert not posterior_predictive_category_check(
        design, wrong, np.zeros(design.n_battles), level=0.999, seed=1
    ).passed


def test_prior_predictive_deviation_is_zero_for_equal_skills() -> None:
    """At a zero gap the expected score is exactly 0.5, though P(win) is not."""
    design, x = _tournament_design()
    pp = prior_predictive(design, x, ModelPriors(skill_prior_sd=1e-6), n_draws=200, seed=1)  # type: ignore[arg-type]

    assert pp.median_abs_win_dev < 1e-3
    assert 0.3 < float(pp.win_prob_quantiles[2]) < 0.45  # P(win) = expit(-c1), information only
