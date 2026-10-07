"""Recovery tests for the impute stage's joint model.

Fits the real model on a synthetic corpus with known coefficients and about
30% of sides missing (some battles missing both), then checks three things a
wrong model would still sample, converge and report cleanly through:

1. **Coefficient recovery**: the era levels, the outcome and type contrasts and
   the two variance components come back near truth.
2. **Held-out coverage**: the 90% interval of the imputed values for sides that
   were masked contains their true log troops about 90% of the time.
3. **No attenuation** (the overimputation point): regressing outcome on the
   imputed log force ratio recovers the slope of the true ratio, while
   regressing on the noisy observed values does not. This only holds because
   outcome is a covariate of the imputation model: draws from the posterior
   given the outcome preserve the joint law of (outcome, troops).

Real MCMC (``pytest.mark.model``), no database; nutpie and one core, as in the
reconcile recovery suite.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np
import pytest

from pipeline.imputers.conditional import PosteriorSamples, RoundSet, draw_rounds, extract_posterior
from pipeline.imputers.design import ImputeDesign, build_design
from pipeline.imputers.model import build_model
from pipeline.imputers.priors import load_priors
from pipeline.imputers.summarise import convergence
from pipeline.reconcilers.model import SamplerSettings, sample
from tests.model.impute_sim import (
    TRUE_A_ERA,
    TRUE_BETA_OUT,
    TRUE_BETA_TYPE,
    TRUE_SIGMA,
    TRUE_TAU_B,
    SimCorpus,
    simulate_corpus,
)

pytestmark = pytest.mark.model

SEED = 20261006
SETTINGS = SamplerSettings(
    draws=1000,
    tune=1000,
    chains=4,
    cores=1,
    target_accept=0.9,
    random_seed=SEED,
    nuts_sampler="nutpie",
)
N_COVERAGE_ROUNDS = 400
ERA_MAX_TOL = 0.3
ERA_MEAN_TOL = 0.15


@dataclass(frozen=True)
class Fit:
    """One fitted corpus."""

    corpus: SimCorpus
    design: ImputeDesign
    posterior: dict[str, np.ndarray]
    samples: PosteriorSamples
    worst: dict[str, float | int | None]
    seconds: float
    rounds: RoundSet
    coverage_rounds: RoundSet


@pytest.fixture(scope="module")
def fit() -> Fit:
    """Fit the model once; every test in this module reads the same posterior."""
    corpus = simulate_corpus(SEED, n_battles=300, p_missing=0.3)
    priors = load_priors("config/imputation_priors.yaml")
    design = build_design(corpus.sides, corpus.likelihoods, priors)
    started = time.time()
    idata = sample(build_model(design), SETTINGS)
    seconds = time.time() - started
    samples = extract_posterior(idata, design)
    posterior = {
        name: np.asarray(getattr(samples, name))
        for name in ("a_era", "beta_out", "beta_type", "v_polity", "c_cmd", "tau_b", "sigma")
    }
    return Fit(
        corpus=corpus,
        design=design,
        posterior=posterior,
        samples=samples,
        worst=convergence(idata),
        seconds=seconds,
        rounds=draw_rounds(design, samples, n_rounds=20, seed=SEED, mnar_delta=0.0),
        coverage_rounds=draw_rounds(
            design, samples, n_rounds=N_COVERAGE_ROUNDS, seed=SEED + 1, mnar_delta=0.0
        ),
    )


def _contrast(fit: Fit, block: str, plus: str, minus: str) -> float:
    """Posterior mean of one level minus another in a ZeroSumNormal block."""
    levels = (fit.design.outcome if block == "beta_out" else fit.design.battle_type).levels
    draws = fit.posterior[block]
    return float(draws[:, levels.index(plus)].mean() - draws[:, levels.index(minus)].mean())


def _mean_effect(draws: np.ndarray) -> float:
    """Posterior mean of the average of a random-effect block's sampled levels."""
    return float(draws.mean())


def _slope(outcome: np.ndarray, ratio: np.ndarray) -> float:
    """OLS slope of outcome on a log ratio."""
    centred = ratio - ratio.mean()
    return float((centred * (outcome - outcome.mean())).sum() / (centred**2).sum())


def test_fit_converged_within_the_write_gate(fit: Fit) -> None:
    print(f"\nimpute recovery: sampling {fit.seconds:.0f}s, worst={fit.worst}")
    assert fit.worst["max_rhat"] is not None and fit.worst["max_rhat"] <= 1.01
    assert fit.worst["min_ess_bulk"] is not None and fit.worst["min_ess_bulk"] >= 400
    assert fit.worst["divergences"] == 0


def test_coefficients_are_recovered(fit: Fit) -> None:
    out_err = _contrast(fit, "beta_out", "victory", "defeat") - (
        TRUE_BETA_OUT["victory"] - TRUE_BETA_OUT["defeat"]
    )
    type_err = _contrast(fit, "beta_type", "field", "naval") - (
        TRUE_BETA_TYPE["field"] - TRUE_BETA_TYPE["naval"]
    )
    # The era intercept alone is not identified: it trades off against the mean
    # of the polity and commander effects (a constant shift in all of them is
    # absorbed by the era levels). The identified quantity is the intercept plus
    # the mean of the sampled polity and commander effects, so that is compared
    # against its simulation counterpart, with a tighter tolerance.
    fit_shift = _mean_effect(fit.posterior["v_polity"]) + _mean_effect(fit.posterior["c_cmd"])
    true_shift = np.mean(
        [fit.corpus.polity_effect[k] for k in fit.design.polity.sampled]
    ) + np.mean([fit.corpus.commander_effect[k] for k in fit.design.commander.sampled])
    era_err = fit.posterior["a_era"].mean(axis=0) + fit_shift - (np.array(TRUE_A_ERA) + true_shift)
    tau_b_err = float(fit.posterior["tau_b"].mean()) - TRUE_TAU_B
    sigma_err = float(fit.posterior["sigma"].mean()) - TRUE_SIGMA
    print(
        f"coefficient errors: outcome={out_err:+.3f} type={type_err:+.3f} "
        f"era(max abs)={np.abs(era_err).max():.3f} era(mean abs)={np.abs(era_err).mean():.3f} "
        f"tau_b={tau_b_err:+.3f} sigma={sigma_err:+.3f}"
    )
    assert abs(out_err) < 0.2
    assert abs(type_err) < 0.25
    assert np.abs(era_err).max() < ERA_MAX_TOL
    assert np.abs(era_err).mean() < ERA_MEAN_TOL
    assert abs(tau_b_err) < 0.25
    assert abs(sigma_err) < 0.2


def test_held_out_interval_coverage_is_near_nominal(fit: Fit) -> None:
    rows = np.flatnonzero(fit.coverage_rounds.is_imputed)
    truth = np.array([fit.corpus.true_y[int(fit.design.side_ids[r])] for r in rows])
    lo, hi = np.percentile(fit.coverage_rounds.values[:, rows], [5.0, 95.0], axis=0)
    coverage = float(((truth >= lo) & (truth <= hi)).mean())
    print(f"held-out 90% coverage: {coverage:.3f} over {len(rows)} masked sides")
    assert 0.83 <= coverage <= 0.96


def _two_sided(fit: Fit) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Outcome sign, both side rows and the true log ratio, per two-sided battle."""
    corpus, design = fit.corpus, fit.design
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
    return outcome, a_rows, b_rows, truth


def _mean_slope(outcome: np.ndarray, rounds: RoundSet, a: np.ndarray, b: np.ndarray) -> float:
    """Mean over rounds of the outcome slope on the imputed log ratio."""
    draws = rounds.values
    return float(np.mean([_slope(outcome, draws[r, a] - draws[r, b]) for r in range(len(draws))]))


def test_outcome_regression_is_not_attenuated_by_overimputation(fit: Fit) -> None:
    corpus = fit.corpus
    battles = sorted(corpus.two_sided)
    outcome, a_rows, b_rows, truth = _two_sided(fit)
    slope_true = _slope(outcome, truth)
    # 400 rounds rather than 20: the slope's Monte Carlo noise should not be
    # what the tolerance is absorbing.
    slope_imputed = _mean_slope(outcome, fit.coverage_rounds, a_rows, b_rows)

    both_observed = np.array(
        [
            corpus.two_sided[b][0] not in corpus.is_missing
            and corpus.two_sided[b][1] not in corpus.is_missing
            for b in battles
        ]
    )
    naive = np.array(
        [
            corpus.measured_y[corpus.two_sided[b][0]] - corpus.measured_y[corpus.two_sided[b][1]]
            for b in battles
        ]
    )
    slope_naive = _slope(outcome[both_observed], naive[both_observed])
    slope_true_observed = _slope(outcome[both_observed], truth[both_observed])

    imputed_ratio = slope_imputed / slope_true
    naive_ratio = slope_naive / slope_true_observed
    print(
        f"attenuation: imputed/true={imputed_ratio:.3f} "
        f"(slope {slope_imputed:.3f} vs {slope_true:.3f}); "
        f"naive observed/true={naive_ratio:.3f} on {int(both_observed.sum())} battles"
    )
    # The simulation must actually produce attenuation for the check to mean anything.
    assert naive_ratio < 0.75
    # Loose on purpose: the fitted outcome contrast is shrunk a little by its
    # prior and by noisy likelihoods, and one corpus has sampling noise in both
    # slopes. The tight version (true parameters, no MCMC) is the unit test
    # test_oracle_parameters_give_an_unattenuated_outcome_slope.
    assert 0.75 <= imputed_ratio <= 1.25
    assert imputed_ratio - naive_ratio >= 0.15


def test_slope_ratio_moves_with_the_residual_noise_in_the_expected_direction(fit: Fit) -> None:
    """Inflating sigma adds noise to imputed ratios, so the slope falls; shrinking it, rises."""
    outcome, a_rows, b_rows, truth = _two_sided(fit)
    slope_true = _slope(outcome, truth)
    ratios = []
    for factor in (0.5, 1.0, 2.0):
        post = replace(fit.samples, sigma=fit.samples.sigma * factor)
        rounds = draw_rounds(
            fit.design, post, n_rounds=N_COVERAGE_ROUNDS, seed=SEED + 2, mnar_delta=0.0
        )
        ratios.append(_mean_slope(outcome, rounds, a_rows, b_rows) / slope_true)
    print(f"slope ratio at sigma x (0.5, 1, 2): {[round(r, 3) for r in ratios]}")
    assert ratios[0] > ratios[1] > ratios[2]
