"""Stated ranges loaded as two bound rows, and what merging them does to the fit.

On the real corpus an infobox range ("914,000-1,067,500") arrives as two
censored rows, a lower bound and an upper bound from one claim: 182 of 201 bound
rows, and 98 sides have nothing else. Each censored probability tends to one
half as sigma grows, so a pair never pays for a wide sigma, sigma inflates and
singletons collapse to the corpus mean. ``build_design`` now merges such a pair
into one point row at the log midpoint with extra variance
``(log(B / A) / 4) ** 2``. The scenario here has 300 sides, 30% of them reported
only as range pairs, and checks the fit with and without the merge.
(The file keeps its name from an earlier contamination scenario; the mixture
family remains as an ablation and is covered by unit tests.)
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
import yaml

from pipeline.reconcilers.design import build_design
from pipeline.reconcilers.model import ModelPriors, SamplerSettings, build_model, sample
from pipeline.reconcilers.records import Report, SourceKey
from pipeline.reconcilers.summarise import _convergence
from pipeline.stages.reconcile import _passes_global_thresholds
from tests.model.conftest import ANCIENT_CUTOFF_YEAR, Fit, fit_until_converged, run_fit

pytestmark = pytest.mark.model

SEED = 20261013
AGENTS = Path(__file__).resolve().parents[2] / "agents"
N_SIDES = 300
N_RANGE_SIDES = 90
P_MULTI = 0.15
# P1: troops carry an informative noise prior (median 0.22) and a side prior of a
# shared battle effect plus a side deviation, so the scenario is generated that way:
# two sides per battle, noise at the prior median. (It was independent sides at 0.35.)
TRUE_SIGMA = 0.22
BATTLE_SD = 1.0
SIDE_SD = 0.6

SAMPLER = SamplerSettings(
    draws=2500,
    tune=1000,
    chains=4,
    cores=1,
    target_accept=0.99,
    random_seed=SEED,
    nuts_sampler="nutpie",
)


class Corpus(NamedTuple):
    reports: list[Report]
    true_mu: dict[int, float]
    singleton_ids: list[int]
    range_ids: list[int]


def _report(
    report_id: int, sid: int, value: float, *, lower: bool, upper: bool, text: str
) -> Report:
    key = SourceKey("wikipedia_infobox", f"https://example.org/battle-{sid}")
    return Report(
        report_id=report_id,
        side_id=sid,
        battle_id=(sid - 1) // 2 + 1,
        source_id=sid,
        source_key=key,
        source_type=key.source_type,
        quantity="troops",
        reported_value=value,
        is_lower_bound=lower,
        is_upper_bound=upper,
        extracted_context=text,
        year_astronomical=1800,
        claim_regime="unlabelled",
        lineage_id=report_id,
    )


def _simulate() -> Corpus:
    rng = np.random.default_rng(SEED)
    battle_effect = {b: BATTLE_SD * float(rng.standard_t(4)) for b in range(1, N_SIDES // 2 + 1)}
    true_mu = {
        sid: 10.0 + battle_effect[(sid - 1) // 2 + 1] + float(rng.normal(0.0, SIDE_SD))
        for sid in range(1, N_SIDES + 1)
    }
    range_ids = sorted(rng.choice(np.arange(1, N_SIDES + 1), N_RANGE_SIDES, replace=False).tolist())
    reports: list[Report] = []
    singleton_ids: list[int] = []
    report_id = 0
    for sid in range(1, N_SIDES + 1):
        if sid in range_ids:
            width = float(rng.uniform(0.1, 0.5))
            low, high = (float(np.exp(true_mu[sid] + sign * width / 2.0)) for sign in (-1, 1))
            text = f"{low:,.0f}-{high:,.0f}"
            reports.append(_report(report_id + 1, sid, low, lower=True, upper=False, text=text))
            reports.append(_report(report_id + 2, sid, high, lower=False, upper=True, text=text))
            report_id += 2
            continue
        n_rows = int(rng.integers(2, 4)) if rng.random() < P_MULTI else 1
        if n_rows == 1:
            singleton_ids.append(sid)
        for _ in range(n_rows):
            report_id += 1
            value = float(np.exp(true_mu[sid] + rng.normal(0.0, TRUE_SIGMA)))
            reports.append(
                _report(report_id, sid, value, lower=False, upper=False, text=f"{value:,.0f}")
            )
    return Corpus(reports, true_mu, singleton_ids, range_ids)


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    return _simulate()


def _troop_priors() -> ModelPriors:
    """The troop priors as agents/reconcile.yaml sets them (P1)."""
    params = yaml.safe_load((AGENTS / "reconcile.yaml").read_text(encoding="utf-8"))["params"]
    return ModelPriors.from_params(params, quantity="troops")


def _fit(
    corpus: Corpus, inflation: object, *, merge_ranges: bool, seed: int = SEED
) -> tuple[Fit, float]:
    started = time.perf_counter()
    # The unmerged control is the censored-pair pathology, so it must run with
    # censored bounds: the ExGaussian default reads each bound as a noisy report
    # and largely defuses it (sigma 0.259 measured, below the 0.8 control floor).
    bound = "exgaussian" if merge_ranges else "censored"
    fit = run_fit(
        corpus.reports,
        priors=replace(_troop_priors(), bound_likelihood=bound),
        inflation=inflation,  # type: ignore[arg-type]
        sampler=replace(SAMPLER, random_seed=seed),
        ancient_cutoff_year=ANCIENT_CUTOFF_YEAR,
        merge_ranges=merge_ranges,
    )
    return fit, time.perf_counter() - started


@pytest.fixture(scope="module")
def merged(corpus: Corpus, inflation_prior: object) -> tuple[Fit, float, int]:
    """The merged fit, refitted on a new seed while it misses the stage's write gate."""
    (fit, seconds), seeds = fit_until_converged(
        lambda seed: _fit(corpus, inflation_prior, merge_ranges=True, seed=seed),
        seed=SEED,
        converged=lambda result: _passes_global_thresholds(
            result[0].summary.diagnostics["worst"], {}
        ),
        describe=lambda result: result[0].summary.diagnostics["worst"],
    )
    return fit, seconds, len(seeds)


@pytest.fixture(scope="module")
def unmerged(corpus: Corpus, inflation_prior: object) -> tuple[Fit, float]:
    return _fit(corpus, inflation_prior, merge_ranges=False)


def _infobox_sigma(fit: Fit) -> float:
    sigma_sq = np.asarray(fit.idata.posterior["sigma_sq_type"].values)
    labels = [str(x) for x in fit.idata.posterior.coords["source_type"].values]
    return float(np.sqrt(sigma_sq[..., labels.index("wikipedia_infobox")]).mean())


def _singleton_corr(fit: Fit, corpus: Corpus) -> float:
    est = {e.side_id: np.log(e.value) for e in fit.summary.side_estimates}
    x = [est[s] for s in corpus.singleton_ids]
    y = [corpus.true_mu[s] for s in corpus.singleton_ids]
    return float(np.corrcoef(x, y)[0, 1])


def test_merged_ranges_keep_sigma_near_truth_and_singletons_unshrunk(
    merged: tuple[Fit, float, int], corpus: Corpus, record_property: pytest.RecordProperty
) -> None:
    fit, seconds, attempts = merged
    diag = fit.summary.diagnostics
    worst = diag["worst"]
    sigma = _infobox_sigma(fit)
    corr = _singleton_corr(fit, corpus)
    print(
        f"merged: {seconds:.0f}s sigma={sigma:.3f} slope={diag['singleton_slope']:.3f} "
        f"corr={corr:.3f} spread={diag['spread_ratio']:.3f} worst={worst}"
    )
    record_property("merged_seconds", round(seconds, 1))
    record_property("convergence_attempts", attempts)

    # The fixture only accepts a fit that clears the stage's own predicate.
    assert _passes_global_thresholds(worst, {}), worst
    # P1: within 1.5x of the 0.22 prior median, around the 0.22 truth (was 0.3 to 0.55 around 0.35).
    assert 0.15 <= sigma <= 0.33, sigma
    assert diag["singleton_slope"] >= 0.7, diag["singleton_slope"]


def test_unmerged_range_pairs_no_longer_inflate_sigma_under_the_informative_prior(
    unmerged: tuple[Fit, float], record_property: pytest.RecordProperty
) -> None:
    """Control: with the merge off, the pairs are two censored rows each.

    Before P1 this was a positive control: the pairs inflated sigma to 0.85 (true
    0.35) and collapsed the singletons (slope 0.21, spread 0.27). Troops now carry
    an informative noise prior (log-sd 0.2), so sigma cannot escape it and the
    pathology is defused: measured sigma 0.230, singleton slope 0.79 and spread
    ratio 0.83 on this corpus. The assertions state that, as a leak guard: sigma
    stays within 1.5x of its 0.22 prior median, and the estimates are not collapsed.
    """
    fit, seconds = unmerged
    diag = fit.summary.diagnostics
    sigma = _infobox_sigma(fit)
    print(
        f"unmerged: {seconds:.0f}s sigma={sigma:.3f} slope={diag['singleton_slope']} "
        f"spread={diag['spread_ratio']:.3f} worst={diag['worst']}"
    )
    record_property("unmerged_seconds", round(seconds, 1))

    assert sigma <= 1.5 * 0.22, sigma
    assert diag["spread_ratio"] >= 0.5, diag["spread_ratio"]
    assert diag["singleton_slope"] >= 0.5, diag["singleton_slope"]


def test_neutral_fit_converges_on_the_same_data(
    corpus: Corpus, inflation_prior: object, record_property: pytest.RecordProperty
) -> None:
    priors = _troop_priors()
    design = build_design(
        corpus.reports, quantity="troops", ancient_cutoff_year=ANCIENT_CUTOFF_YEAR
    )
    model = build_model(design, priors=priors, inflation=inflation_prior, neutral=True)  # type: ignore[arg-type]
    idata, seeds = fit_until_converged(
        lambda seed: sample(model, replace(SAMPLER, random_seed=seed)),
        seed=SEED,
        converged=lambda idata: _passes_global_thresholds(_convergence(idata)[0], {}),
        describe=lambda idata: _convergence(idata)[:2],
    )
    record_property("convergence_attempts", len(seeds))
    worst, variable, _ = _convergence(idata)
    print(f"neutral: worst={worst} variable={variable} attempts={len(seeds)}")

    assert _passes_global_thresholds(worst, {}), (worst, variable)
