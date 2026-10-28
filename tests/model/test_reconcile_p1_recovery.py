"""P1 recovery: data generated from the P1 side prior come back, and the gates pass.

The corpus follows the generative process the redesign fits
(``docs/handover-log/32-troop-redesign-review-2026-10-23.md``)::

    log y = m0 + e[era_bin] + b[battle] + s_d * zd[side] + b_type + noise

with ``b`` Student-t(4) of scale ``s_b``

with noise 0.22 on modern rows and 0.40 on rows before the noise cutoff, the two
medians the informative prior is centred on. Sides are mostly singletons in
two-sided battles, as on the real corpus, with a share of two-report sides so the
noise is identified apart from ``s_d``. Real MCMC (``pytest.mark.model``), no database.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from pipeline.reconcilers.model import InflationPrior, ModelPriors, SamplerSettings
from pipeline.reconcilers.records import Report, SourceKey
from tests.model.conftest import Fit, fit_until_converged, run_fit

pytestmark = pytest.mark.model

SEED = 20261023
TRUE_M0 = 10.0
TRUE_E = np.array([-0.8, -0.3, 0.0, 0.4, 0.7])  # sums to zero
TRUE_S_B = 0.8
TRUE_S_D = 0.4
TRUE_NOISE_MODERN = 0.22
TRUE_NOISE_ANCIENT = 0.40
TRUE_B_TYPE = {"wikipedia_infobox": 0.15, "wikidata": -0.15}
YEARS = (-300, 700, 1650, 1815, 1944)  # one per era bin
YEAR_WEIGHTS = (0.15, 0.10, 0.15, 0.30, 0.30)
N_BATTLES = 110
AGENTS = Path(__file__).resolve().parents[2] / "agents"
SAMPLER = SamplerSettings(
    draws=1500,
    tune=1500,
    chains=4,
    cores=1,
    target_accept=0.97,
    random_seed=SEED,
    nuts_sampler="nutpie",
)


def simulate_p1(rng: np.random.Generator) -> tuple[list[Report], dict[int, float]]:
    """Two-sided battles from the P1 process; about a third of sides have two reports.

    Returns:
        The reports and each side's true ``mu_side`` (``m0 + e + b + s_d * zd``).
    """
    reports: list[Report] = []
    true_level: dict[int, float] = {}
    report_id = 1
    for battle in range(N_BATTLES):
        bin_ = int(rng.choice(5, p=YEAR_WEIGHTS))
        year = YEARS[bin_]
        zb = rng.standard_t(4)
        for side_in_battle in (0, 1):
            side_id = 2 * battle + side_in_battle + 1
            level = TRUE_M0 + TRUE_E[bin_] + TRUE_S_B * zb + TRUE_S_D * rng.standard_normal()
            true_level[side_id] = float(level)
            n_reports = 2 if rng.random() < 0.33 else 1
            for _ in range(n_reports):
                source_type = ("wikipedia_infobox", "wikidata")[report_id % 2]
                noise = TRUE_NOISE_ANCIENT if year < 500 else TRUE_NOISE_MODERN
                eta = level + TRUE_B_TYPE[source_type] + noise * rng.standard_normal()
                reports.append(
                    Report(
                        report_id=report_id,
                        side_id=side_id,
                        battle_id=battle + 1,
                        source_id=report_id,
                        source_key=SourceKey(source_type, f"https://example.org/{report_id}"),
                        source_type=source_type,
                        quantity="troops",
                        reported_value=math.exp(eta),
                        scope="engaged",
                        year_astronomical=year,
                        claim_regime="unlabelled",
                        lineage_id=report_id,
                    )
                )
                report_id += 1
    return reports, true_level


def simulate_p1_corpus(rng: np.random.Generator) -> list[Report]:
    """The reports of :func:`simulate_p1`, for callers that need no truth."""
    return simulate_p1(rng)[0]


@pytest.fixture(scope="module")
def corpus() -> tuple[list[Report], dict[int, float]]:
    return simulate_p1(np.random.default_rng(SEED))


@pytest.fixture(scope="module")
def reports(corpus: tuple[list[Report], dict[int, float]]) -> list[Report]:
    return corpus[0]


@pytest.fixture(scope="module")
def true_level(corpus: tuple[list[Report], dict[int, float]]) -> dict[int, float]:
    return corpus[1]


@pytest.fixture(scope="module")
def priors() -> ModelPriors:
    params = yaml.safe_load((AGENTS / "reconcile.yaml").read_text(encoding="utf-8"))["params"]
    return ModelPriors.from_params(params, quantity="troops")


def _converged(fit: Fit) -> bool:
    worst = fit.summary.diagnostics["worst"]
    return bool(
        worst["max_rhat"] is not None
        and worst["max_rhat"] < 1.02
        and worst["divergences"] == 0
        and (worst["min_ess_bulk"] or 0) > 400
    )


@pytest.fixture(scope="module")
def fit(reports: list[Report], priors: ModelPriors, inflation_prior: InflationPrior) -> Fit:
    result, _ = fit_until_converged(
        lambda seed: run_fit(
            reports,
            priors=priors,
            inflation=inflation_prior,
            sampler=replace(SAMPLER, random_seed=seed),
        ),
        seed=SEED,
        converged=_converged,
        describe=lambda f: f.summary.diagnostics["worst"],
    )
    return result


def test_singleton_estimates_follow_their_reports(fit: Fit) -> None:
    recovery = fit.summary.diagnostics["recovery"]

    assert recovery["n_singletons"] >= 100
    assert recovery["singleton_slope"] >= 0.85, recovery["singleton_slope"]
    assert recovery["spread_ratio"] >= 0.85, recovery["spread_ratio"]


def test_the_side_deviation_scale_does_not_collapse(fit: Fit) -> None:
    recovery = fit.summary.diagnostics["recovery"]

    assert recovery["s_d_median"] >= 0.25, recovery["s_d_median"]
    assert recovery["s_d_median"] <= 0.65, recovery["s_d_median"]


def test_the_noise_stays_within_one_and_a_half_times_its_prior_median(fit: Fit) -> None:
    recovery = fit.summary.diagnostics["recovery"]

    assert recovery["noise_ratio_modern"] <= 1.5, recovery["noise_ratio_modern"]
    assert recovery["noise_ratio_ancient"] <= 1.5, recovery["noise_ratio_ancient"]
    assert recovery["noise_ratio_modern"] == pytest.approx(1.0, abs=0.35)


def test_the_mixed_predictive_check_does_not_reject_the_generating_model(fit: Fit) -> None:
    ppc = fit.summary.diagnostics["recovery"]["mixed_ppc"]

    assert ppc["tail_p"] >= 0.05, ppc


def test_the_within_battle_slope_matches_the_implied_reliability(fit: Fit) -> None:
    recovery = fit.summary.diagnostics["recovery"]

    assert recovery["within_battle_slope"] >= 0.5, recovery["within_battle_slope"]
    assert 0.0 < recovery["reliability"] < 1.0
    assert recovery["gate_wb_reliability_gap"] <= 0.1, (
        recovery["within_battle_slope"],
        recovery["reliability"],
    )


def test_a_troop_fit_has_no_dispersion_terms_and_records_the_side_prior(fit: Fit) -> None:
    names = set(fit.idata.posterior.data_vars)

    assert {"e", "s_b", "s_d", "tau_e", "zb", "zd", "mu_side"} <= names
    assert not {"s0", "z_side", "sigma_round", "sigma_estimate"} & names


def test_the_recorded_singleton_slope_matches_an_independent_computation(
    fit: Fit, reports: list[Report]
) -> None:
    counts: dict[int, int] = {}
    for r in reports:
        counts[r.side_id] = counts.get(r.side_id, 0) + 1
    singles = [s for s, n in counts.items() if n == 1]
    estimates = {e.side_id: e for e in fit.summary.side_estimates}
    by_side = {r.side_id: math.log(r.reported_value) for r in reports}
    x = np.array([by_side[s] for s in singles])
    y = np.array([math.log(estimates[s].value) for s in singles])
    slope = float(np.polyfit(x, y, 1)[0])

    recovery = fit.summary.diagnostics["recovery"]
    assert slope >= 0.85, slope
    assert recovery["singleton_slope"] == pytest.approx(slope, abs=0.02)
    assert recovery["gate_singleton_slope"] == pytest.approx(slope, abs=0.02)


def test_the_posterior_noise_stays_near_the_generating_value(fit: Fit) -> None:
    diagnostics = fit.summary.diagnostics
    modern = diagnostics["sigma_type"]
    ancient = diagnostics["sigma_type_ancient"]

    assert all(0.10 < v < 0.40 for v in modern.values() if v is not None), modern
    assert all(0.20 < v < 0.65 for v in ancient.values() if v is not None), ancient
    assert diagnostics["nu"] is None  # troops are Normal since P1


def test_the_gates_that_read_the_recovery_block_pass_on_a_recovered_fit(fit: Fit) -> None:
    recovery = fit.summary.diagnostics["recovery"]

    assert recovery["gate_singleton_slope"] >= 0.85
    assert recovery["gate_spread_ratio"] >= 0.85
    assert recovery["gate_within_battle_slope"] >= 0.5
    assert recovery["gate_ppc_tail_p"] >= 0.05
    assert recovery["gate_s_d_median"] >= 0.25
    assert recovery["gate_noise_ratio"] <= 1.5


def test_the_battle_effect_scale_mixes(fit: Fit) -> None:
    import arviz as az

    rhat = float(az.rhat(fit.idata, var_names=["s_b"])["s_b"].values)
    ess = float(az.ess(fit.idata, var_names=["s_b"], method="bulk")["s_b"].values)
    print(f"s_b rhat={rhat:.4f} ess_bulk={ess:.0f}")

    assert rhat < 1.02, rhat
    assert ess > 400, ess


def _interval(samples: np.ndarray, mass: float) -> tuple[float, float]:
    tail = (1.0 - mass) / 2.0
    return float(np.quantile(samples, tail)), float(np.quantile(samples, 1.0 - tail))


def test_the_battle_effect_scale_is_recovered(fit: Fit) -> None:
    s_b = np.asarray(fit.idata.posterior["s_b"].values).ravel()
    lo, hi = _interval(s_b, 0.95)
    print(f"s_b true={TRUE_S_B} mean={s_b.mean():.3f} 95%=({lo:.3f}, {hi:.3f})")

    assert lo <= TRUE_S_B <= hi, (lo, hi)


def test_era_contrasts_between_unanchored_bins_are_recovered(fit: Fit) -> None:
    # Bins 3 and 4 (1800 on) carry no regime or era offset (the level anchor), and
    # bins 1 and 2 share the same offsets (neither is ancient), so each pair's
    # difference is identified. The level of the pre-1800 bins against the anchored
    # ones is traded against g_unlabelled and a_era by design, so e itself is not.
    e = np.asarray(fit.idata.posterior["e"].values).reshape(-1, 5)
    for high, low in ((4, 3), (2, 1)):
        lo, hi = _interval(e[:, high] - e[:, low], 0.95)
        truth = TRUE_E[high] - TRUE_E[low]
        print(f"e[{high}]-e[{low}] true={truth:.2f} mean={np.mean(e[:, high] - e[:, low]):.3f}")
        assert lo <= truth <= hi, (high, low, lo, hi)


def test_the_side_intervals_cover_the_true_side_sizes_at_ninety_percent(
    fit: Fit, true_level: dict[int, float]
) -> None:
    mu = np.asarray(fit.idata.posterior["mu_side"].values)
    mu = mu.reshape(-1, mu.shape[-1])
    positions = {int(side): i for i, side in enumerate(fit.design.side_ids)}
    covered = [
        _interval(mu[:, positions[side]], 0.90)[0]
        <= truth
        <= _interval(mu[:, positions[side]], 0.90)[1]
        for side, truth in true_level.items()
    ]
    fraction = float(np.mean(covered))
    print(f"mu_side 90% coverage = {fraction:.3f} over {len(covered)} sides")

    assert 0.82 <= fraction <= 0.97, fraction
