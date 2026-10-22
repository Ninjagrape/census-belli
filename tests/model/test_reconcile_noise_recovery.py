"""A lone report is trusted at the scale the noise prior says it should be.

The 2026-10 corpus fit gave a singleton slope of 0.61 and a within-battle
log-ratio slope of 0.60: the noise sd was identified only by sides that disagree
and then applied to every lone report
(``docs/handover-log/31-troop-numbers-diagnosis-2026-10-22.md``). Here the data
are generated at the noise the informative prior is centred on (0.22), and the
estimate of a singleton side must follow its report.

The corpus is deliberately plain: every report is unlabelled and engaged, source
biases are zero, so the only thing separating a report from its side's level is
noise, and the slope of estimate on report measures shrinkage alone. Fits real
MCMC (``pytest.mark.model``), no database.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

import tests.model.conftest as sim
from pipeline.reconcilers.model import InflationPrior, ModelPriors, SamplerSettings
from pipeline.reconcilers.recovery import (
    battle_pairs,
    side_report_logs,
    within_battle_slope,
)
from tests.model.conftest import Fit, fit_until_converged, run_fit

pytestmark = pytest.mark.model

SEED = 20261022
NOISE_SD = 0.22
TRUE_M0 = 9.6
TRUE_S0 = 1.2
SAMPLER = SamplerSettings(
    draws=4000, tune=1500, chains=4, cores=1, random_seed=SEED, nuts_sampler="nutpie"
)


def _simulate(
    noise_sd: float, side_groups: list[tuple[int, int]], **ids: int
) -> sim.SimulatedCorpus:
    mp = pytest.MonkeyPatch()
    mp.setattr(sim, "TRUE_SOURCE_SIGMA", dict.fromkeys(sim.SOURCES, noise_sd))
    mp.setattr(sim, "TRUE_U_SOURCE_MAGNITUDE", dict.fromkeys(sim.SOURCE_TYPES, 0.0))
    try:
        return sim.simulate_corpus(
            np.random.default_rng(SEED + sum(ids.values())),
            side_groups=side_groups,
            m0=TRUE_M0,
            s0=TRUE_S0,
            regime_weights={"unlabelled": 1.0},
            scope_weights={"engaged": 1.0},
            ancient_fraction=0.4,
            b_type=dict.fromkeys(sim.SOURCE_TYPES, 0.0),
            a_era=0.0,
            **ids,
        )
    finally:
        mp.undo()


def build_corpus() -> sim.SimulatedCorpus:
    """Mostly singletons, every report at noise 0.22.

    No nuisance offsets other than one constant, so the slope of estimate on
    report measures shrinkage alone. Singletons dominate, as on the real corpus
    (309 of 341 sides).
    """
    return _simulate(NOISE_SD, [(40, 3), (40, 2), (220, 1)])


@pytest.fixture(scope="module")
def corpus() -> sim.SimulatedCorpus:
    return build_corpus()


def _converged(fit: Fit) -> bool:
    worst = fit.summary.diagnostics["worst"]
    return bool(
        worst["max_rhat"] is not None
        and worst["max_rhat"] < 1.02
        and worst["divergences"] == 0
        and (worst["min_ess_bulk"] or 0) > 400
    )


def _fit(corpus: sim.SimulatedCorpus, priors: ModelPriors, inflation: InflationPrior) -> Fit:
    fit, _ = fit_until_converged(
        lambda seed: run_fit(
            corpus.reports,
            priors=priors,
            inflation=inflation,
            sampler=replace(SAMPLER, random_seed=seed),
        ),
        seed=SEED,
        converged=_converged,
        describe=lambda fit: fit.summary.diagnostics["worst"],
    )
    return fit


@pytest.fixture(scope="module")
def informative_fit(
    corpus: sim.SimulatedCorpus, inflation_prior: InflationPrior, troop_priors: ModelPriors
) -> Fit:
    return _fit(corpus, troop_priors, inflation_prior)


def _singleton_slope(fit: Fit, corpus: sim.SimulatedCorpus) -> float:
    estimates = sim.side_estimate_by_id(fit.summary)
    singles = corpus.side_ids_by_report_count[1]
    by_side = {r.side_id: math.log(r.reported_value) for r in corpus.reports}
    x = np.array([by_side[s] for s in singles])
    y = np.array([math.log(estimates[s].value) for s in singles])
    return float(np.polyfit(x, y, 1)[0])


def test_a_singleton_estimate_tracks_its_report_at_the_generating_noise(
    informative_fit: Fit, corpus: sim.SimulatedCorpus
) -> None:
    slope = _singleton_slope(informative_fit, corpus)

    assert slope >= 0.85, slope
    recorded = informative_fit.summary.diagnostics["recovery"]["singleton_slope"]
    assert recorded == pytest.approx(slope, abs=0.02)


def test_the_within_battle_ratio_is_not_compressed(
    informative_fit: Fit, corpus: sim.SimulatedCorpus
) -> None:
    design = informative_fit.design
    battle_of_side = {r.side_id: r.battle_id for r in corpus.reports}
    estimates = sim.side_estimate_by_id(informative_fit.summary)
    log_estimates = {s: math.log(e.value) for s, e in estimates.items()}

    slope, n = within_battle_slope(
        battle_pairs(design, battle_of_side), side_report_logs(design), log_estimates
    )

    assert n >= 20
    assert slope is not None
    assert slope >= 0.8, slope


def test_the_posterior_noise_stays_near_the_generating_value(informative_fit: Fit) -> None:
    sigma = informative_fit.summary.diagnostics["sigma_type"]
    nu = informative_fit.summary.diagnostics["nu"]["mean"]

    assert all(0.10 < v < 0.40 for v in sigma.values() if v is not None), sigma
    assert nu >= 3.0


def test_the_gates_that_read_the_recovery_block_pass_on_a_recovered_fit(
    informative_fit: Fit,
) -> None:
    recovery = informative_fit.summary.diagnostics["recovery"]

    assert recovery["gate_singleton_slope"] is not None
    assert recovery["gate_singleton_slope"] >= 0.85
    assert recovery["gate_spread_ratio"] is not None
    assert recovery["gate_spread_ratio"] >= 0.85
    assert recovery["gate_within_battle_slope"] >= 0.8
    extremity = recovery["gate_between_battle_extremity"]
    assert extremity is None or extremity >= 0.025
