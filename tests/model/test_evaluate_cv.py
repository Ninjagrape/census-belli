"""Cross-validation of the evaluate stage on simulated tournaments (real MCMC).

Two claims about ``pipeline.evaluators.cv.run_cross_validation``, the entry
point the stage calls:

1. **Signal.** With real skill differences and a real force effect, held-out
   accuracy and Brier beat what a coin gives (accuracy above 0.6, Brier below
   0.24), in the imputed arm the gates use.
2. **Null leakage guard.** Skills are all equal and the force ratio is
   irrelevant: outcomes are drawn independently of everything. The imputed
   rounds are deliberately leaky (``x_diff`` encodes the outcome, the way impute
   conditioning on the outcome does), while the measurements are the true
   troops. Neither arm may score above chance, because a held-out battle's
   prediction never reads its own imputed ``x_diff``. A positive control in the
   same test shows the check can see leakage: the *same* fold fits, asked to
   predict the held-out battles from the leaky ``x_diff``, score clearly above
   chance.

The folds are fitted exactly as the stage fits them. The positive control reuses
those fits (``fit_pooled`` is wrapped to keep each fold's result), so it costs no
extra sampling. A fit that misses the convergence gate is retried on a new seed
(``fit_until_converged``); an accuracy or leakage assertion is never retried.
"""

from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from pipeline.evaluators import cv as cv_module
from pipeline.evaluators.config import EvaluateParams
from pipeline.evaluators.cv import CVResult, run_cross_validation
from pipeline.evaluators.fit import PooledResult
from pipeline.evaluators.predict import binary_metrics, predict_category_probs
from pipeline.modelers.config import ModelParams, load_model_params
from pipeline.modelers.design import build_design, x_difference
from pipeline.modelers.records import ModelDesign, SideInput
from tests.model.conftest import fit_until_converged
from tests.model.model_sim import LEVELS, SimTournament, simulate_tournament

pytestmark = pytest.mark.model

SEED = 20261014
N_FOLDS = 5
N_BINS = 5
# 24 generals, about 200 battles: the size the brief asks for.
GENERALS_PER_ERA = 12
BATTLES_PER_ERA = 100
N_ROUNDS = 1
OBS_SD = 0.1
ROUND_NOISE_SD = 0.15
# The leak added to imputed x_diff: 1.5 * (y_a - 0.5) * 2, y_a in {0, .25, .5, .75, 1}.
LEAK_SIZE = 1.5

# Fewer, shorter chains than the stage's defaults (3000 draws): the claims here
# are about prediction, which a 500-draw posterior serves. The sampler-diagnostic
# bar is the model recovery test's (rhat 1.02, ESS 400, no divergences).
MODEL_PARAMS: dict[str, Any] = {
    "sampler": {"draws": 500, "tune": 500, "chains": 4, "cores": 1, "target_accept": 0.9},
    "write": {"max_rhat": 1.02, "min_ess_bulk": 400, "max_divergences": 0},
    "pooling": {"draws_kept_per_round": 200},
}
EVAL_PARAMS: dict[str, Any] = {
    "held_out_folds": N_FOLDS,
    "cv_rounds": N_ROUNDS,
    "eval_draws": 500,
    "calibration_bins": N_BINS,
}

# Symmetric level probabilities for the null: P(level k) = P(level 4 - k), so
# orientation carries nothing and every predictor is at chance by construction.
NULL_LEVEL_PROBS = (0.12, 0.23, 0.30, 0.23, 0.12)


@dataclass(frozen=True)
class FoldFits:
    """A cross-validation run and the fold fits behind it."""

    cv: CVResult
    fits: list[tuple[np.ndarray, PooledResult]]
    seconds: float
    attempts: int


@dataclass(frozen=True)
class Scenario:
    """A design with its truth, imputed rounds and measurements."""

    design: ModelDesign
    x_true: np.ndarray
    x_rounds: list[np.ndarray]
    measured: tuple[np.ndarray, np.ndarray]


def _model_params() -> ModelParams:
    mp = load_model_params(MODEL_PARAMS)
    sampler = dataclasses.replace(mp.sampler, draws=EVAL_PARAMS["eval_draws"])
    return dataclasses.replace(mp, sampler=sampler)


def _eval_params(seed: int) -> EvaluateParams:
    return EvaluateParams.from_mapping({**EVAL_PARAMS, "seed": seed})


def _measurements(design: ModelDesign, tour: SimTournament) -> tuple[np.ndarray, np.ndarray]:
    """True log troops with a small sd, per side index: the measurement arm."""
    mean = np.array([tour.log_troops[int(s)] for s in design.side_ids])
    return mean, np.full(design.n_sides, OBS_SD)


def _round_noise(rng: np.random.Generator, n: int) -> list[np.ndarray]:
    return [ROUND_NOISE_SD * rng.standard_normal(n) for _ in range(N_ROUNDS)]


def _signal_scenario() -> Scenario:
    tour = simulate_tournament(
        SEED, generals_per_era=GENERALS_PER_ERA, battles_per_era=BATTLES_PER_ERA
    )
    design = build_design(tour.sides)
    x_true = x_difference(design, tour.log_troops)
    rng = np.random.default_rng(SEED + 1)
    rounds = [x_true + noise for noise in _round_noise(rng, design.n_battles)]
    return Scenario(design, x_true, rounds, _measurements(design, tour))


def _null_sides(tour: SimTournament, rng: np.random.Generator) -> list[SideInput]:
    """The tournament's sides with outcomes redrawn independently of everything.

    Both sides of a battle report, and the lower side id (side A in the design)
    gets level ``k`` from a symmetric distribution, the other ``4 - k``.
    """
    by_battle: dict[int, list[SideInput]] = {}
    for side in tour.sides:
        by_battle.setdefault(side.battle_id, []).append(side)
    out: list[SideInput] = []
    for sides in by_battle.values():
        first, second = sorted(sides, key=lambda s: s.side_id)
        k = int(rng.choice(5, p=NULL_LEVEL_PROBS))
        out.append(dataclasses.replace(first, outcome=LEVELS[k]))
        out.append(dataclasses.replace(second, outcome=LEVELS[4 - k]))
    return out


def _null_scenario() -> tuple[Scenario, np.ndarray]:
    tour = simulate_tournament(
        SEED + 2, generals_per_era=GENERALS_PER_ERA, battles_per_era=BATTLES_PER_ERA
    )
    rng = np.random.default_rng(SEED + 3)
    design = build_design(_null_sides(tour, rng))
    x_true = x_difference(design, tour.log_troops)
    y_a = (design.k_lo + design.k_hi) / 8.0  # level / 4, midpoint of a censored interval
    leak = LEAK_SIZE * (y_a - 0.5) * 2.0
    rounds = [x_true + leak + noise for noise in _round_noise(rng, design.n_battles)]
    return Scenario(design, x_true, rounds, _measurements(design, tour)), leak


def _run_cv(scenario: Scenario, seed: int) -> FoldFits:
    """One cross-validation run, keeping each imputed-arm fold's fit."""
    fits: list[tuple[np.ndarray, PooledResult]] = []
    original = cv_module.fit_pooled

    def recording(*args: Any, **kwargs: Any) -> PooledResult:
        result = original(*args, **kwargs)
        if str(kwargs.get("label", "")).startswith("cv_fold_"):
            fits.append((np.asarray(kwargs["loglik_weight"]), result))
        return result

    started = time.perf_counter()
    cv_module.fit_pooled = recording  # type: ignore[assignment]
    try:
        cv = run_cross_validation(
            scenario.design,
            scenario.x_rounds,
            _model_params(),
            _eval_params(seed),
            scenario.measured,
        )
    finally:
        cv_module.fit_pooled = original  # type: ignore[assignment]
    return FoldFits(cv, fits, time.perf_counter() - started, 1)


def _all_folds_converged(run: FoldFits) -> bool:
    return all(f.converged for f in run.cv.folds) and all(f.converged for f in run.cv.joint.folds)


def _describe(run: FoldFits) -> str:
    return str([(f.fold, f.converged, f.worst, f.error) for f in run.cv.folds + run.cv.joint.folds])


def _converged_cv(scenario: Scenario) -> FoldFits:
    """Cross-validate, refitting on a new seed only while a fold misses the gate."""
    run, seeds = fit_until_converged(
        lambda seed: _run_cv(scenario, seed),
        seed=SEED,
        converged=_all_folds_converged,
        describe=_describe,
    )
    return dataclasses.replace(run, attempts=len(seeds))


def _leaky_probs(scenario: Scenario, leaky_x: np.ndarray, fits: list[Any]) -> np.ndarray:
    """What a predictor that (wrongly) read the leaky imputed x_diff would say."""
    design = scenario.design
    probs = np.full((design.n_battles, 5), np.nan)
    for weight, result in fits:
        held_out = np.flatnonzero(weight == 0.0)
        per_round = [
            predict_category_probs(design, rnd.draws, leaky_x, held_out)
            for rnd in result.rounds
            if rnd.draws is not None
        ]
        probs[held_out] = np.mean(per_round, axis=0)
    return probs


def _binary(design: ModelDesign, probs: np.ndarray) -> dict[str, Any]:
    keep = np.isfinite(probs).all(axis=1)
    return binary_metrics(probs[keep], design.k_lo[keep], design.k_hi[keep], N_BINS)


@pytest.fixture(scope="module")
def signal() -> tuple[Scenario, FoldFits]:
    scenario = _signal_scenario()
    return scenario, _converged_cv(scenario)


@pytest.fixture(scope="module")
def null() -> tuple[Scenario, np.ndarray, FoldFits]:
    scenario, leak = _null_scenario()
    return scenario, leak, _converged_cv(scenario)


def test_signal_design_has_the_size_the_claims_assume(
    signal: tuple[Scenario, FoldFits],
) -> None:
    scenario, run = signal

    assert 180 <= scenario.design.n_battles <= 220
    assert len(run.fits) == N_FOLDS
    assert all(f.converged for f in run.cv.folds)
    assert run.cv.predicted.all()


def test_held_out_prediction_beats_chance_when_skill_and_force_matter(
    signal: tuple[Scenario, FoldFits],
) -> None:
    _, run = signal
    imputed = run.cv.metrics["binary"]
    joint = run.cv.joint.metrics["binary"]
    print(
        f"\nsignal imputed arm: n={imputed['n']} accuracy={imputed['accuracy']:.3f} "
        f"brier={imputed['brier']:.3f}; joint arm: accuracy={joint['accuracy']:.3f} "
        f"brier={joint['brier']:.3f}; attempts={run.attempts}; {run.seconds:.0f}s"
    )

    assert imputed["n"] >= 100
    assert imputed["accuracy"] > 0.6
    assert imputed["brier"] < 0.24
    # The stage's own baselines cover the same battles.
    assert run.cv.baselines["chance"]["accuracy"] == 0.5


def test_null_outcomes_are_unrelated_to_the_imputed_force_ratio_on_held_out_battles(
    null: tuple[Scenario, np.ndarray, FoldFits],
) -> None:
    """Both arms stay within chance although the training rounds encode the outcome."""
    scenario, leak, run = null
    del leak
    assert run.cv.predicted.all() and run.cv.joint.predicted.all()

    results = {
        "imputed": run.cv.metrics["binary"],
        "joint": run.cv.joint.metrics["binary"],
    }
    for arm, metrics in results.items():
        n = metrics["n"]
        # Under the null a held-out win is a fair coin whatever the prediction,
        # so accuracy ~ Binomial(n, 1/2) / n with sd 0.5 / sqrt(n): allow 3 sd.
        tolerance = 3 * 0.5 / math.sqrt(n)
        print(
            f"\nnull {arm} arm: n={n} accuracy={metrics['accuracy']:.3f} "
            f"brier={metrics['brier']:.4f} tolerance={tolerance:.3f}; "
            f"attempts={run.attempts}; {run.seconds:.0f}s"
        )
        assert n >= 100, arm
        assert abs(metrics["accuracy"] - 0.5) < tolerance, arm
        # A calibrated coin scores 0.25; an informative predictor scores well below.
        assert metrics["brier"] > 0.235, arm
    assert scenario.design.n_battles >= 180


def test_a_predictor_reading_the_leaky_x_diff_would_score_clearly_above_chance(
    null: tuple[Scenario, np.ndarray, FoldFits],
) -> None:
    """Positive control: the leak is real, so the chance-level result above means something."""
    scenario, leak, run = null
    leaky_x = scenario.x_rounds[0]
    probs = _leaky_probs(scenario, leaky_x, run.fits)
    leaked = _binary(scenario.design, probs)
    honest = run.cv.metrics["binary"]
    print(
        f"\nnull positive control (leaky x_diff): accuracy={leaked['accuracy']:.3f} "
        f"brier={leaked['brier']:.3f}; honest imputed arm accuracy={honest['accuracy']:.3f}; "
        f"leak sd={leak.std():.2f}"
    )

    assert leaked["n"] == honest["n"]
    # Binomial sd at n ~ 140 is 0.042, so 0.75 is about six sd above chance.
    assert leaked["accuracy"] > 0.75
    assert leaked["brier"] < 0.2
    assert leaked["accuracy"] - honest["accuracy"] > 0.2
