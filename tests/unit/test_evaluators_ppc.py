"""Posterior predictive checks by stratum and by general (no sampler)."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.evaluators.ppc import (
    era_bucket,
    posterior_predictive_report,
    simulate_levels,
)
from pipeline.modelers.design import build_design
from pipeline.modelers.model import PosteriorDraws, assemble_side_scores
from pipeline.modelers.records import ModelDesign
from tests.unit.modelers_helpers import battle

K = 60
N_PER_ERA = 12


def _design() -> ModelDesign:
    """Strong general 1 against weak general 2, alternating sides, in four eras.

    General 1 always wins decisively except in the 1900s battles, where the
    recorded outcome is a decisive defeat for 1.
    """
    sides = []
    bid = 0
    for year, lost in ((-300, False), (1000, False), (1700, False), (1900, True), (None, False)):
        for i in range(N_PER_ERA):
            bid += 1
            # Alternating sides makes the A-perspective level mix uninformative, so
            # the misfit era keeps general 1 on side A to make it visible.
            one_first = i % 2 == 0 or lost
            a, b = ((1,), (2,)) if one_first else ((2,), (1,))
            win, lose = "decisive_victory", "decisive_defeat"
            one_wins = not lost
            a_out, b_out = (win, lose) if one_first == one_wins else (lose, win)
            sides += battle(bid, a, b, a_outcome=a_out, b_outcome=b_out, year=year)
    return build_design(sides)


def _draws(design: ModelDesign) -> PosteriorDraws:
    rng = np.random.default_rng(0)
    skills = np.tile([6.0, -6.0], (K, 1)) + rng.normal(0, 0.01, (K, 2))
    sigma = np.ones(K)
    return PosteriorDraws(
        skills=skills,
        side_scores=assemble_side_scores(design, skills, sigma, None, None, None),
        beta_force=np.zeros(K),
        beta_def=np.zeros(K),
        c1=np.full(K, 0.5),
        c2=np.full(K, 2.0),
        sigma_skill=sigma,
    )


@pytest.mark.parametrize(
    ("year", "bucket"),
    [
        (-3000.0, "ancient"),
        (499.0, "ancient"),
        (500.0, "medieval"),
        (1499.0, "medieval"),
        (1500.0, "early_modern"),
        (1799.0, "early_modern"),
        (1800.0, "modern"),
        (float("nan"), "undated"),
    ],
)
def test_era_bucket_boundaries(year: float, bucket: str) -> None:
    assert era_bucket(year) == bucket


def test_simulate_levels_is_seeded_and_in_range() -> None:
    design = _design()
    draws = _draws(design)
    x = np.zeros(design.n_battles)

    first = simulate_levels(design, draws, x, n_sim=25, seed=3)

    assert first.shape == (25, design.n_battles)
    assert first.min() >= 0 and first.max() <= 4
    np.testing.assert_array_equal(first, simulate_levels(design, draws, x, n_sim=25, seed=3))
    assert simulate_levels(design, draws, x, n_sim=500, seed=3).shape[0] == K


def _report(types: dict[int, str | None] | None = None) -> dict[str, object]:
    design = _design()
    names = {int(g): f"general {g}" for g in design.general_ids}
    return posterior_predictive_report(
        design,
        _draws(design),
        np.zeros(design.n_battles),
        battle_types=types or {},
        score=np.array([2.0, 1.0]),
        names=names,
        top_n=2,
        n_sim=K,
        level=0.95,
        seed=1,
    )


def test_report_stratifies_by_era_with_undated_and_flags_the_misfit_era() -> None:
    report = _report()

    era = report["category"]["era"]  # type: ignore[index]
    assert set(era) == {"ancient", "medieval", "early_modern", "modern", "undated"}
    assert all(s["n_exact"] == N_PER_ERA for s in era.values())
    assert era["modern"]["outside"], "the era with reversed outcomes should be flagged"
    assert not era["ancient"]["outside"]
    flagged = report["flagged"]["strata"]  # type: ignore[index]
    assert {"group": "era", "stratum": "modern", "outside": era["modern"]["outside"]} in flagged


def test_report_stratifies_by_battle_type_and_defaults_to_unknown() -> None:
    types: dict[int, str | None] = {b: "naval" for b in range(1, 25)}

    report = _report(types)

    by_type = report["category"]["battle_type"]  # type: ignore[index]
    assert by_type["naval"]["n_exact"] == 24
    assert by_type["unknown"]["n_exact"] == 5 * N_PER_ERA - 24


def test_small_strata_are_reported_but_never_flagged() -> None:
    design = _design()
    types: dict[int, str | None] = {int(b): "land" for b in design.battle_ids}
    types[1] = "siege"

    report = posterior_predictive_report(
        design,
        _draws(design),
        np.zeros(design.n_battles),
        battle_types=types,
        score=np.array([2.0, 1.0]),
        names={},
        top_n=2,
        n_sim=K,
        level=0.95,
        seed=1,
    )

    siege = report["category"]["battle_type"]["siege"]  # type: ignore[index]
    assert siege == {"n_exact": 1, "evaluable": False, "outside": []}


def test_per_general_win_rate_handles_both_sides_and_flags_the_outlier() -> None:
    report = _report()

    rows = {r["general_id"]: r for r in report["per_general"]}  # type: ignore[attr-defined]
    strong, weak = rows[1], rows[2]
    assert strong["n_decided"] == weak["n_decided"] == 5 * N_PER_ERA
    # General 1 wins 48 of 60 and is predicted to win nearly all of them.
    assert strong["observed_win_share"] == pytest.approx(48 / 60)
    assert strong["flagged"] is True
    assert weak["observed_win_share"] == pytest.approx(12 / 60)
    assert weak["flagged"] is True
    assert report["flagged"]["generals"] == ["general 1", "general 2"]  # type: ignore[index]


def test_per_general_skips_generals_without_decided_battles() -> None:
    design = build_design(
        battle(1, (1,), (2,), a_outcome="indecisive", b_outcome="indecisive")
        + battle(2, (3,), (4,))
    )
    skills = np.zeros((K, design.n_generals))
    draws = PosteriorDraws(
        skills=skills,
        side_scores=assemble_side_scores(design, skills, np.ones(K), None, None, None),
        beta_force=np.zeros(K),
        beta_def=np.zeros(K),
        c1=np.full(K, 0.5),
        c2=np.full(K, 2.0),
        sigma_skill=np.ones(K),
    )

    report = posterior_predictive_report(
        design,
        draws,
        np.zeros(design.n_battles),
        battle_types={},
        score=np.arange(design.n_generals, dtype=float),
        names={},
        top_n=design.n_generals,
        n_sim=K,
        level=0.95,
        seed=0,
    )

    assert {r["general_id"] for r in report["per_general"]} == {3, 4}  # type: ignore[attr-defined]
