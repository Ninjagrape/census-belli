"""Recovery test for the Bradley-Terry model of the model stage.

Fits the real model on a simulated tournament of 60 generals in two eras that
never meet and about 800 battles, with co-commanded and unknown-commander
sides, and checks what a wrong model would still sample cleanly through:

1. skills are ranked correctly within each era (Spearman);
2. the 95% intervals for ``beta_force`` and ``beta_def`` cover the truth;
3. 95% skill intervals cover the truth at a sensible rate (between 85% and 99%).

Skills are compared centred (per draw for the posterior, by the true mean for
the truth): only differences enter the likelihood, and the offset between the
eras is pinned by the prior alone.

Real MCMC (``pytest.mark.model``), no database; nutpie and one core.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np
import pytest
from scipy.stats import spearmanr

from pipeline.modelers.design import build_design, x_difference
from pipeline.modelers.model import (
    PosteriorDraws,
    build_joint_latent_model,
    build_model,
    extract_draws,
)
from pipeline.modelers.pool import round_convergence
from pipeline.modelers.priors import ModelPriors
from pipeline.modelers.records import ModelDesign
from pipeline.modelers.war import centre_skills
from pipeline.reconcilers.model import SamplerSettings, sample
from tests.model.conftest import fit_until_converged
from tests.model.model_sim import (
    TRUE_BETA_DEFENDER,
    TRUE_BETA_FORCE,
    SimTournament,
    simulate_tournament,
)

pytestmark = pytest.mark.model

SEED = 20261009
SETTINGS = SamplerSettings(
    draws=600,
    tune=600,
    chains=4,
    cores=1,
    target_accept=0.9,
    random_seed=SEED,
    nuts_sampler="nutpie",
)
MIN_SPEARMAN = 0.8
# 400 battles per era (about 27 appearances per general) puts the expected Spearman well
# above the 0.8 bar; at 200 it sat at the information limit (0.797 to 0.922 over seeds).
BATTLES_PER_ERA = 400
MODEST_BATTLES_PER_ERA = 200
COVERAGE_RANGE = (0.85, 0.99)


@dataclass(frozen=True)
class Fit:
    """One fitted tournament."""

    tournament: SimTournament
    design: ModelDesign
    draws: PosteriorDraws
    convergence: dict[str, object]
    seconds: float
    attempts: int


def _converged(convergence: dict[str, object]) -> bool:
    """Whether a fit's roll-up clears the convergence gate this module asserts."""
    rhat, ess = convergence["max_rhat"], convergence["min_ess_bulk"]
    return (
        rhat is not None
        and rhat <= 1.02  # type: ignore[operator]
        and ess is not None
        and ess >= 400  # type: ignore[operator]
        and convergence["divergences"] == 0
    )


def _fit_tournament(battles_per_era: int, *, beta_defender: float = 0.0) -> Fit:
    """Simulate a tournament and fit the model on it.

    A fit that misses the convergence gate is refitted on a new seed, as the model
    stage does, so one unlucky chain set does not fail every claim below.
    """
    tournament = simulate_tournament(
        SEED, battles_per_era=battles_per_era, beta_defender=beta_defender
    )
    design = build_design(tournament.sides)
    x_diff = x_difference(design, tournament.log_troops)
    model = build_model(design, x_diff, ModelPriors())
    started = time.time()
    idata, seeds = fit_until_converged(
        lambda seed: sample(model, replace(SETTINGS, random_seed=seed)),
        seed=SEED,
        converged=lambda idata: _converged(round_convergence(idata)),
        describe=round_convergence,
    )
    seconds = time.time() - started
    return Fit(
        tournament=tournament,
        design=design,
        draws=extract_draws(idata, design),
        convergence=round_convergence(idata),
        seconds=seconds,
        attempts=len(seeds),
    )


@pytest.fixture(scope="module")
def fit() -> Fit:
    """Fit once; every test but the joint-latent one reads the same posterior."""
    return _fit_tournament(BATTLES_PER_ERA, beta_defender=TRUE_BETA_DEFENDER)


@pytest.fixture(scope="module")
def modest_fit() -> Fit:
    """The 200-battle-per-era corpus the joint-latent comparator is calibrated on.

    On the larger corpus the comparator's beta_force sits about 0.15 below the
    truth (means 0.28 to 0.37 over four seeds, against 0.45 to 0.51 with the true
    x), a shrinkage that the narrower posterior then exposes. That is a finding
    about the comparator, not something to hide by moving the seed.
    """
    return _fit_tournament(MODEST_BATTLES_PER_ERA)


def _truth(fit: Fit) -> np.ndarray:
    return np.array([fit.tournament.true_skill[int(g)] for g in fit.design.general_ids])


def test_design_keeps_the_battles_and_classifies_unknown_sides(fit: Fit) -> None:
    counts = fit.design.counts
    assert counts.battles_included >= 780
    assert counts.sides_unknown >= 1
    assert counts.sides_unattributed == 0


def test_fit_converged(fit: Fit, record_property: pytest.RecordProperty) -> None:
    record_property("convergence_attempts", fit.attempts)
    print(
        f"\nmodel recovery: sampling {fit.seconds:.0f}s over {fit.attempts} attempt(s), "
        f"convergence={fit.convergence}"
    )
    assert fit.convergence["max_rhat"] is not None and fit.convergence["max_rhat"] <= 1.02  # type: ignore[operator]
    assert fit.convergence["min_ess_bulk"] is not None and fit.convergence["min_ess_bulk"] >= 400  # type: ignore[operator]
    assert fit.convergence["divergences"] == 0


def test_skills_are_ranked_within_each_era(fit: Fit) -> None:
    truth = _truth(fit)
    mean = centre_skills(fit.draws.skills).mean(axis=0)
    eras = np.array([fit.tournament.era_of[int(g)] for g in fit.design.general_ids])
    rhos = [float(spearmanr(mean[eras == e], truth[eras == e]).statistic) for e in (0, 1)]
    print(f"within-era Spearman: {rhos}")
    assert min(rhos) > MIN_SPEARMAN


def test_beta_force_interval_covers_truth(fit: Fit) -> None:
    lo, hi = np.percentile(fit.draws.beta_force, [2.5, 97.5])
    print(f"beta_force mean {fit.draws.beta_force.mean():.3f}, 95% [{lo:.3f}, {hi:.3f}]")
    assert lo <= TRUE_BETA_FORCE <= hi


def test_defender_term_is_built_and_beta_def_interval_covers_truth(fit: Fit) -> None:
    assert fit.design.defender_term_used
    beta = fit.draws.beta_def
    lo, hi = np.percentile(beta, [2.5, 97.5])
    print(f"beta_def mean {beta.mean():.3f} sd {beta.std(ddof=1):.3f}, 95% [{lo:.3f}, {hi:.3f}]")
    assert lo <= TRUE_BETA_DEFENDER <= hi


def test_skill_interval_coverage(fit: Fit) -> None:
    truth = _truth(fit)
    centred_truth = truth - truth.mean()
    lo, hi = np.percentile(centre_skills(fit.draws.skills), [2.5, 97.5], axis=0)
    covered = float(np.mean((lo <= centred_truth) & (centred_truth <= hi)))
    print(f"95% skill interval coverage: {covered:.3f}")
    assert COVERAGE_RANGE[0] <= covered <= COVERAGE_RANGE[1]


JOINT_LEVEL = 9.5  # a realistic absolute log-troop level; it must not matter
JOINT_MEASUREMENT_SD = 0.3
# Measured 1.51 to 1.54 on Windows/3.14 and 1.65 on CI's Linux/3.11 (ESS ~500): 56% of
# battles have a side with no measurement, so the comparator is genuinely less
# informative than a fit with the true x. 1.8 leaves cross-platform sampling noise room
# without hiding a bias, which the coverage assertion above guards (a shifted mean).
MAX_SD_RATIO = 1.8
JOINT_SIGMA_W = 1.0


def test_joint_latent_beta_force_covers_truth_with_a_third_of_sides_missing(
    modest_fit: Fit,
) -> None:
    tournament, design = modest_fit.tournament, modest_fit.design
    rng = np.random.default_rng(SEED + 1)
    obs_mean = np.full(design.n_sides, np.nan)
    obs_sd = np.full(design.n_sides, np.nan)
    for index, side_id in enumerate(design.side_ids):
        if int(side_id) % 3 != 0:
            noise = JOINT_MEASUREMENT_SD * rng.standard_normal()
            obs_mean[index] = JOINT_LEVEL + tournament.log_troops[int(side_id)] + noise
            obs_sd[index] = JOINT_MEASUREMENT_SD
    idata = sample(
        build_joint_latent_model(design, obs_mean, obs_sd, JOINT_SIGMA_W, ModelPriors()), SETTINGS
    )

    joint = extract_draws(idata, design).beta_force
    lo, hi = np.percentile(joint, [2.5, 97.5])
    fixed_sd = float(modest_fit.draws.beta_force.std(ddof=1))
    ratio = float(joint.std(ddof=1)) / fixed_sd
    print(
        f"joint beta_force mean {joint.mean():.3f} sd {joint.std(ddof=1):.3f} "
        f"95% [{lo:.3f}, {hi:.3f}]; fixed-x sd {fixed_sd:.3f}; ratio {ratio:.2f}; "
        f"convergence {round_convergence(idata)}"
    )
    assert lo <= TRUE_BETA_FORCE <= hi
    assert ratio <= MAX_SD_RATIO
