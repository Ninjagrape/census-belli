"""Tests for turning a posterior into database-ready reconcile records.

Every ``InferenceData`` here is built by hand with ``az.from_dict`` from known
draws, never sampled from a real model: a real fit takes seconds to minutes
(see ``docs/handover-log.md`` §19.5), and what these tests pin down is the arithmetic
in ``summarise.py``, not whether PyMC converges. Nothing here carries the
``model`` marker.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, date, datetime
from typing import Any

import numpy as np
import pytest

from pipeline.quality import _roll_up_metrics
from pipeline.reconcilers.design import Design, build_design
from pipeline.reconcilers.model import _LEVEL_PRIOR_SD, InflationPrior, ModelPriors
from pipeline.reconcilers.records import ReconcileCounts, Report, SourceKey
from pipeline.reconcilers.summarise import summarise_fit

# ─── Helpers ─────────────────────────────────────────────────────────────────

_COMPUTED_AT = datetime(2026, 9, 29, tzinfo=UTC)


def _report(
    report_id: int,
    *,
    side_id: int = 1,
    battle_id: int = 1,
    source_id: int | None = None,
    value: float = 10_000.0,
    quantity: str = "troops",
    branch: str = "total",
    scope: str = "engaged",
    source_type: str = "wikipedia_infobox",
    url: str = "https://example/a",
    year: int | None = -330,
    regime: str = "unlabelled",
    lineage_id: int | None = None,
) -> Report:
    """Build a Report with everything but the field under test defaulted.

    Mirrors ``tests/unit/test_reconcile_design.py``'s helper of the same
    name; kept separate rather than shared so this file has no import-time
    coupling to another test module's internals.
    """
    return Report(
        report_id=report_id,
        side_id=side_id,
        battle_id=battle_id,
        source_id=report_id if source_id is None else source_id,
        source_key=SourceKey(source_type, url),
        source_type=source_type,
        quantity=quantity,  # type: ignore[arg-type]
        branch=branch,
        reported_value=value,
        scope=scope,
        extracted_context="",
        year_astronomical=year,
        claim_regime=regime,
        # Production gives every report its own lineage unless it copies another.
        lineage_id=report_id if lineage_id is None else lineage_id,
    )


def _design(reports: list[Report], *, ancient_cutoff_year: int = 500) -> Design:
    return build_design(
        reports,
        quantity="troops",
        ancient_cutoff_year=ancient_cutoff_year,
        counts=ReconcileCounts(),
    )


def _build_idata(
    design: Design,
    inflation: InflationPrior,
    *,
    n_chains: int = 2,
    n_draws: int = 300,
    seed: int = 0,
    mu_side_mean: float = 9.2,
    mu_side_sd: float = 0.3,
    per_side_mu: dict[int, np.ndarray] | None = None,
    beta_source_sd: float = 0.2,
    sigma_source_mean: float = 0.3,
    b_type_sd: float = 0.3,
    a_era_sd: float = 0.4,
    level_sd: float = 0.5,
    extra_posterior: dict[str, tuple[list[str], np.ndarray]] | None = None,
    diverging: np.ndarray | None = None,
) -> Any:
    """Build an InferenceData whose coordinates match ``design`` exactly.

    Reading coordinates straight from ``design.coords()`` (rather than
    guessing an order) is what keeps this fixture exercising the same lookup
    path ``summarise.py`` uses against a real fit, where those coordinates
    are literally what ``build_model`` passed to ``pm.Model``.
    """
    import arviz as az  # noqa: PLC0415 - kept out of the test module's import line

    rng = np.random.default_rng(seed)
    coords = design.coords()
    free_regimes = [r for r in design.regimes if r != inflation.anchor_regime]

    n_sides = len(coords["side"])
    n_keys = len(coords["source_key"])
    n_types = len(coords["source_type"])

    posterior: dict[str, np.ndarray] = {}
    dims: dict[str, list[str]] = {}

    mu_side = rng.normal(mu_side_mean, mu_side_sd, size=(n_chains, n_draws, n_sides))
    if per_side_mu:
        for side_id, samples in per_side_mu.items():
            position = coords["side"].index(str(side_id))
            mu_side[:, :, position] = samples.reshape(n_chains, n_draws)
    posterior["mu_side"] = mu_side
    dims["mu_side"] = ["side"]

    posterior["beta_source"] = rng.normal(0.0, beta_source_sd, size=(n_chains, n_draws, n_keys))
    dims["beta_source"] = ["source_key"]

    posterior["sigma_source"] = np.abs(
        rng.normal(sigma_source_mean, 0.02, size=(n_chains, n_draws, n_keys))
    )
    dims["sigma_source"] = ["source_key"]

    posterior["b_type"] = rng.normal(0.0, b_type_sd, size=(n_chains, n_draws, n_types))
    dims["b_type"] = ["source_type"]

    posterior["a_era"] = rng.normal(inflation.era_mu, a_era_sd, size=(n_chains, n_draws))
    # Read by reports_honoured and the type-level sigma diagnostic.
    posterior["g_regime"] = rng.normal(0.0, 0.05, size=(n_chains, n_draws, len(coords["regime"])))
    dims["g_regime"] = ["regime"]
    posterior["delta_level"] = rng.normal(0.0, 0.05, size=(n_chains, n_draws, len(coords["level"])))
    dims["delta_level"] = ["level"]
    posterior["s0"] = np.abs(rng.normal(1.0, 0.1, size=(n_chains, n_draws)))
    posterior["sigma_sq_type"] = np.abs(rng.normal(0.5, 0.05, size=(n_chains, n_draws, n_types)))
    dims["sigma_sq_type"] = ["source_type"]
    posterior["corpus_level"] = rng.normal(9.2, level_sd, size=(n_chains, n_draws))
    posterior["m0"] = rng.normal(9.0, 0.3, size=(n_chains, n_draws))
    posterior["offset_bar"] = rng.normal(0.1, 0.05, size=(n_chains, n_draws))

    build_coords: dict[str, list[str]] = {
        "side": coords["side"],
        "source_key": coords["source_key"],
        "source_type": coords["source_type"],
        "regime": coords["regime"],
        "level": coords["level"],
    }
    if free_regimes:
        posterior["g_free"] = rng.normal(0.5, 0.3, size=(n_chains, n_draws, len(free_regimes)))
        dims["g_free"] = ["regime_free"]
        build_coords["regime_free"] = free_regimes

    if extra_posterior:
        for name, (dim_names, array) in extra_posterior.items():
            posterior[name] = array
            if dim_names:
                dims[name] = dim_names

    sample_stats = {"diverging": diverging} if diverging is not None else None

    return az.from_dict(
        posterior=posterior, coords=build_coords, dims=dims, sample_stats=sample_stats
    )


# ─── Point estimate ─────────────────────────────────────────────────────────────


def test_the_point_estimate_is_the_exponentiated_posterior_median_not_the_mean_of_the_exponential() -> None:  # noqa: E501
    reports = [_report(1, side_id=1)]
    design = _design(reports)
    inflation = InflationPrior()

    # sd = 0.74 is the figure docs/handover-log.md §19.7 and model.py's module docstring
    # cite: exp(sd**2/2) ~= 1.32, so a naive mean(exp(mu)) overstates the
    # median-based estimate by about 32% here.
    mu_samples = np.random.default_rng(1).normal(9.2, 0.74, size=(2, 2000))
    idata = _build_idata(
        design, inflation, per_side_mu={1: mu_samples}, mu_side_sd=0.0001, seed=2, n_draws=2000
    )

    result = summarise_fit(
        idata,
        design,
        reports,
        run_id=1,
        ci_mass=0.95,
        inflation=inflation,
        priors=ModelPriors(),
        computed_at=_COMPUTED_AT,
    )

    estimate = result.side_estimates[0]
    expected_median_based = math.exp(float(np.median(mu_samples)))
    mean_of_exponential = float(np.mean(np.exp(mu_samples)))

    assert math.isclose(estimate.value, expected_median_based, rel_tol=1e-9)
    # The two summaries of the same posterior differ by double digits of
    # percent; if this ever comes back close to `mean_of_exponential` the
    # implementation has regressed to the forbidden mean(exp(mu)).
    assert estimate.value < mean_of_exponential * 0.85


# ─── Source bias fan-out ────────────────────────────────────────────────────────


def test_a_source_key_shared_by_two_source_ids_yields_two_identical_bias_rows() -> None:
    # design.py builds source_keys from (source_type, url) alone, so two
    # reports naming the same document under different source_id rows --
    # the SourceKey.__doc__ scenario of a SELECT-then-INSERT race -- collapse
    # onto one SourceKey by construction. A SourceKey with zero backing
    # reports is therefore not constructible via build_design; this is the
    # closest real analogue, and it is the one the fan-out logic exists for.
    reports = [
        _report(1, side_id=1, source_id=101, url="https://example/shared"),
        _report(2, side_id=1, source_id=102, url="https://example/shared"),
    ]
    design = _design(reports)
    assert design.n_sources == 1  # one SourceKey, two source_id rows behind it
    inflation = InflationPrior()
    idata = _build_idata(design, inflation, seed=3)

    result = summarise_fit(
        idata,
        design,
        reports,
        run_id=1,
        ci_mass=0.95,
        inflation=inflation,
        priors=ModelPriors(),
        computed_at=_COMPUTED_AT,
    )

    assert len(result.source_biases) == 2
    ids = {b.source_id for b in result.source_biases}
    assert ids == {101, 102}
    first, second = result.source_biases
    assert first.bias_mu == second.bias_mu
    assert first.bias_sd == second.bias_sd
    assert first.sigma_mu == second.sigma_mu
    assert first.sigma_sd == second.sigma_sd
    assert first.n_reports == second.n_reports == 2


# ─── Convergence diagnostics ────────────────────────────────────────────────────


def test_the_maximum_rhat_is_computed_over_every_variable_including_the_non_centred_offsets() -> (
    None
):
    reports = [_report(1, side_id=1)]
    design = _design(reports)
    inflation = InflationPrior()

    n_chains, n_draws = 2, 300
    # A non-centred offset that has not mixed: each chain sits in a
    # different place with almost no within-chain spread. Every other
    # posterior variable is drawn i.i.d. from one distribution per chain, so
    # it is this variable, and only this one, that should blow up rhat.
    z_test = np.empty((n_chains, n_draws))
    z_test[0] = np.random.default_rng(4).normal(-3.0, 0.01, n_draws)
    z_test[1] = np.random.default_rng(5).normal(3.0, 0.01, n_draws)

    idata = _build_idata(
        design,
        inflation,
        seed=6,
        extra_posterior={"z_test": ([], z_test)},
    )

    result = summarise_fit(
        idata,
        design,
        reports,
        run_id=1,
        ci_mass=0.95,
        inflation=inflation,
        priors=ModelPriors(),
        computed_at=_COMPUTED_AT,
    )

    worst = result.diagnostics["worst"]
    assert worst["max_rhat"] is not None
    assert worst["max_rhat"] > 1.5
    assert result.diagnostics["worst_variable"]["rhat"] == "z_test"


def test_divergences_are_summed_across_chains_and_draws() -> None:
    reports = [_report(1, side_id=1)]
    design = _design(reports)
    inflation = InflationPrior()

    diverging = np.zeros((2, 300), dtype=int)
    diverging[0, :5] = 1
    diverging[1, :2] = 1

    idata = _build_idata(design, inflation, seed=7, diverging=diverging)

    result = summarise_fit(
        idata,
        design,
        reports,
        run_id=1,
        ci_mass=0.95,
        inflation=inflation,
        priors=ModelPriors(),
        computed_at=_COMPUTED_AT,
    )

    assert result.diagnostics["worst"]["divergences"] == 7


def test_identified_level_is_the_mean_of_m0_plus_offset_bar() -> None:
    reports = [_report(1, side_id=1)]
    design = _design(reports)
    inflation = InflationPrior()
    idata = _build_idata(design, inflation, seed=8)

    result = summarise_fit(
        idata,
        design,
        reports,
        run_id=1,
        ci_mass=0.95,
        inflation=inflation,
        priors=ModelPriors(),
        computed_at=_COMPUTED_AT,
    )

    m0 = np.asarray(idata.posterior["m0"].values)
    offset_bar = np.asarray(idata.posterior["offset_bar"].values)
    expected_mean = float(np.mean(m0 + offset_bar))

    identified = result.diagnostics["identified_level"]
    assert identified["mean"] is not None
    assert math.isclose(identified["mean"], expected_mean, rel_tol=1e-9)


# ─── Method assignment ──────────────────────────────────────────────────────────


def test_a_singleton_side_gets_single_report_debiased_and_a_multi_lineage_side_gets_source_disagreement() -> None:  # noqa: E501
    reports = [
        _report(1, side_id=1, lineage_id=1),  # side 1: one lineage
        _report(2, side_id=2, lineage_id=2, url="https://example/b"),
        _report(3, side_id=2, lineage_id=3, url="https://example/c"),  # side 2: two lineages
    ]
    design = _design(reports)
    inflation = InflationPrior()
    idata = _build_idata(design, inflation, seed=9)

    result = summarise_fit(
        idata,
        design,
        reports,
        run_id=1,
        ci_mass=0.95,
        inflation=inflation,
        priors=ModelPriors(),
        computed_at=_COMPUTED_AT,
    )

    by_side = {e.side_id: e for e in result.side_estimates}
    assert by_side[1].method == "single_report_debiased"
    assert by_side[1].n_reports == 1
    assert by_side[2].method == "source_disagreement"
    assert by_side[2].n_reports == 2


def test_method_counts_the_claims_that_entered_the_fit_not_the_loaded_reports() -> None:
    # The unlabelled figure restates the chronicle row's, so the cross-regime
    # merge drops it: the model saw one claim, whatever the loader counted.
    reports = [
        _report(1, side_id=1, regime="chronicle", url="https://example/a", lineage_id=1),
        _report(2, side_id=1, regime="unlabelled", url="https://example/b", lineage_id=2),
    ]
    design = _design(reports)
    inflation = InflationPrior()
    idata = _build_idata(design, inflation, seed=10)

    result = summarise_fit(
        idata,
        design,
        reports,
        run_id=1,
        ci_mass=0.95,
        inflation=inflation,
        priors=ModelPriors(),
        computed_at=_COMPUTED_AT,
    )

    estimate = result.side_estimates[0]
    assert estimate.n_reports == 1
    assert estimate.n_sources == 1
    assert estimate.method == "single_report_debiased"


def test_zero_valued_rows_count_neither_towards_the_method_nor_the_lineage_count() -> None:
    # The zero row never enters the likelihood, so it cannot make a side a
    # "source_disagreement" or inflate n_reports / n_sources.
    reports = [
        _report(1, side_id=1, value=10_000.0, url="https://example/a", lineage_id=1),
        _report(2, side_id=1, value=0.0, url="https://example/b", lineage_id=2),
    ]
    design = _design(reports)
    inflation = InflationPrior()
    idata = _build_idata(design, inflation, seed=11)

    result = summarise_fit(
        idata,
        design,
        reports,
        run_id=1,
        ci_mass=0.95,
        inflation=inflation,
        priors=ModelPriors(),
        computed_at=_COMPUTED_AT,
    )

    estimate = result.side_estimates[0]
    assert (estimate.n_reports, estimate.n_sources) == (1, 1)
    assert estimate.method == "single_report_debiased"


# ─── n_identifying_sides ─────────────────────────────────────────────────────────


def test_n_identifying_sides_counts_sides_with_at_least_two_reports_of_a_regime() -> None:
    reports = [
        # side 1: two chronicle reports -> identifies chronicle
        _report(1, side_id=1, regime="chronicle", url="https://example/a", lineage_id=1),
        _report(2, side_id=1, regime="chronicle", url="https://example/b", lineage_id=2),
        # side 2: one chronicle report only -> does not identify it
        _report(3, side_id=2, regime="chronicle", url="https://example/c", lineage_id=3),
        # side 2: two unlabelled reports -> identifies unlabelled
        _report(
            4,
            side_id=2,
            regime="unlabelled",
            url="https://example/d",
            lineage_id=4,
            value=20_000.0,
        ),
        _report(
            5,
            side_id=2,
            regime="unlabelled",
            url="https://example/e",
            lineage_id=5,
            value=30_000.0,
        ),
    ]
    design = _design(reports)
    inflation = InflationPrior()
    idata = _build_idata(design, inflation, seed=10)

    result = summarise_fit(
        idata,
        design,
        reports,
        run_id=1,
        ci_mass=0.95,
        inflation=inflation,
        priors=ModelPriors(),
        computed_at=_COMPUTED_AT,
    )

    n_identifying = result.diagnostics["n_identifying_sides"]
    assert n_identifying["chronicle"] == 1
    assert n_identifying["unlabelled"] == 1


# ─── Contraction ─────────────────────────────────────────────────────────────────


def test_contraction_is_near_zero_when_the_posterior_matches_the_prior() -> None:
    reports = [_report(1, side_id=1)]
    design = _design(reports)
    inflation = InflationPrior(era_sd=0.4)
    priors = ModelPriors(bias_prior_sd=1.0)

    # design has one source_type, so the ZeroSumNormal marginal prior sd on
    # b_type is undefined (n=1); this case is exercised separately below.
    idata = _build_idata(
        design,
        inflation,
        seed=11,
        a_era_sd=inflation.era_sd,
        level_sd=_LEVEL_PRIOR_SD,
        n_draws=4000,
    )

    result = summarise_fit(
        idata,
        design,
        reports,
        run_id=1,
        ci_mass=0.95,
        inflation=inflation,
        priors=priors,
        computed_at=_COMPUTED_AT,
    )

    contraction = result.diagnostics["contraction"]
    assert contraction["a_era"] is not None
    assert abs(contraction["a_era"]) < 0.1
    assert contraction["corpus_level"] is not None
    assert abs(contraction["corpus_level"]) < 0.1


def test_contraction_is_near_one_when_the_posterior_is_tight() -> None:
    reports = [_report(1, side_id=1)]
    design = _design(reports)
    inflation = InflationPrior(era_sd=0.4)
    priors = ModelPriors(bias_prior_sd=1.0)

    idata = _build_idata(
        design,
        inflation,
        seed=12,
        a_era_sd=0.001,
        level_sd=0.001,
    )

    result = summarise_fit(
        idata,
        design,
        reports,
        run_id=1,
        ci_mass=0.95,
        inflation=inflation,
        priors=priors,
        computed_at=_COMPUTED_AT,
    )

    contraction = result.diagnostics["contraction"]
    assert contraction["a_era"] is not None
    assert contraction["a_era"] > 0.99
    assert contraction["corpus_level"] is not None
    assert contraction["corpus_level"] > 0.99


def test_b_type_contraction_uses_the_zero_sum_normal_marginal_prior_sd() -> None:
    reports = [
        _report(1, side_id=1, source_type="wikipedia_infobox", url="https://example/a"),
        _report(2, side_id=1, source_type="peer_reviewed", url="https://example/b"),
    ]
    design = _design(reports)
    assert len(design.source_types) == 2
    inflation = InflationPrior()
    priors = ModelPriors(bias_prior_sd=1.0)
    # marginal sd of each entry of a 2-dim ZeroSumNormal(sigma=1.0):
    # 1.0 * sqrt((2-1)/2) = 0.7071...
    prior_marginal_sd = 1.0 * math.sqrt(0.5)

    idata = _build_idata(design, inflation, seed=13, b_type_sd=prior_marginal_sd, n_draws=4000)

    result = summarise_fit(
        idata,
        design,
        reports,
        run_id=1,
        ci_mass=0.95,
        inflation=inflation,
        priors=priors,
        computed_at=_COMPUTED_AT,
    )

    b_type = result.diagnostics["contraction"]["b_type"]
    assert set(b_type) == set(design.source_types)
    for value in b_type.values():
        assert value is not None
        assert abs(value) < 0.1


# ─── JSON round-trip and the quality gate ───────────────────────────────────────


def test_the_diagnostics_payload_round_trips_through_json_and_the_quality_rollup_reads_it() -> None:
    reports = [_report(1, side_id=1)]
    design = _design(reports)
    # A real date, because that is what yaml.safe_load makes of the fitted
    # file's `fitted_on: 2026-09-24`. An empty provenance hid the crash here
    # and let it surface only in the stage's report writer.
    inflation = InflationPrior(provenance={"fitted_on": date(2026, 9, 24), "n_cases": 45})
    diverging = np.zeros((2, 300), dtype=int)
    idata = _build_idata(design, inflation, seed=14, diverging=diverging)

    result = summarise_fit(
        idata,
        design,
        reports,
        run_id=1,
        ci_mass=0.95,
        inflation=inflation,
        priors=ModelPriors(),
        computed_at=_COMPUTED_AT,
    )

    round_tripped = json.loads(json.dumps(result.diagnostics))  # no default=: must be plain JSON
    metrics = _roll_up_metrics(round_tripped)

    assert not metrics.is_empty
    assert metrics.max_rhat is not None
    assert metrics.min_ess_bulk is not None
    assert metrics.divergences == 0


def test_a_side_with_only_upper_bound_reports_is_labelled_bound_only() -> None:
    from dataclasses import replace

    reports = [
        replace(_report(1, side_id=1), is_upper_bound=True),
        replace(_report(2, side_id=1, url="https://example/b"), is_upper_bound=True),
        # A range from one source bounds from both sides: interval evidence.
        replace(_report(3, side_id=2, url="https://example/c"), is_upper_bound=True),
        replace(_report(4, side_id=2, url="https://example/d"), is_lower_bound=True),
    ]
    design = _design(reports)
    inflation = InflationPrior()
    idata = _build_idata(design, inflation, seed=21)

    result = summarise_fit(
        idata,
        design,
        reports,
        run_id=1,
        ci_mass=0.95,
        inflation=inflation,
        priors=ModelPriors(),
        computed_at=_COMPUTED_AT,
    )

    by_side = {e.side_id: e for e in result.side_estimates}
    assert by_side[1].method == "bound_only"
    assert by_side[2].method == "source_disagreement"


# ─── Write gate ──────────────────────────────────────────────────────────────


def test_a_non_finite_rhat_or_ess_fails_the_write_gate() -> None:
    from pipeline.stages.reconcile import _failed_thresholds

    ok = {"max_rhat": 1.001, "min_ess_bulk": 900.0, "divergences": 0}
    assert _failed_thresholds(ok, {}) == []
    assert _failed_thresholds({**ok, "max_rhat": float("nan")}, {})
    assert _failed_thresholds({**ok, "max_rhat": float("inf")}, {})
    assert _failed_thresholds({**ok, "min_ess_bulk": float("nan")}, {})


def _ar1(rng: np.random.Generator, rho: float, shape: tuple[int, int]) -> np.ndarray:
    """Per-chain AR(1) draws with unit marginal variance."""
    out = np.empty(shape)
    out[:, 0] = rng.normal(size=shape[0])
    noise = rng.normal(size=shape) * math.sqrt(1.0 - rho**2)
    for t in range(1, shape[1]):
        out[:, t] = rho * out[:, t - 1] + noise[:, t]
    return out


def _gate_idata(
    *,
    side_rho: float,
    global_rho: float,
    n_draws: int = 1000,
    n_sides: int = 3,
    seed: int = 0,
    side_sd: float = 0.1,
) -> Any:
    import arviz as az  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    n_chains = 4
    mu_side = side_sd * np.stack(
        [_ar1(rng, side_rho, (n_chains, n_draws)) for _ in range(n_sides)], axis=-1
    )
    return az.from_dict(
        posterior={"mu_side": mu_side, "s0": _ar1(rng, global_rho, (n_chains, n_draws))},
        coords={"side": [str(i) for i in range(n_sides)]},
        dims={"mu_side": ["side"]},
    )


def test_a_side_with_low_ess_but_precise_quantiles_does_not_fail_the_write_gate() -> None:
    from pipeline.reconcilers.summarise import _convergence
    from pipeline.stages.reconcile import _passes_global_thresholds

    # rho 0.9 over 4 chains of 1000 draws: bulk ESS on mu_side near 200, below the 400 floor,
    # while each published quantile is still pinned down to a few percent of the
    # interval.
    idata = _gate_idata(side_rho=0.9, global_rho=0.0, n_draws=1000)

    worst, _, _ = _convergence(idata)

    assert worst["min_ess_bulk_side"] < 400 <= worst["min_ess_bulk"]
    assert worst["n_sides_mcse_fail"] == 0, worst["worst_side"]
    assert _passes_global_thresholds(worst, {})


def test_a_global_variable_with_low_ess_still_fails_the_write_gate() -> None:
    from pipeline.reconcilers.summarise import _convergence
    from pipeline.stages.reconcile import _failed_thresholds

    idata = _gate_idata(side_rho=0.0, global_rho=0.97)

    worst, worst_variable, _ = _convergence(idata)

    assert worst["min_ess_bulk"] < 400
    assert worst_variable["ess_bulk"] == "s0"
    assert any("min_ess_bulk" in reason for reason in _failed_thresholds(worst, {}))


def test_sides_whose_quantiles_are_too_noisy_fail_the_write_gate_and_the_worst_is_named() -> None:
    from pipeline.reconcilers.summarise import _convergence
    from pipeline.stages.reconcile import _failed_thresholds

    idata = _gate_idata(side_rho=0.995, global_rho=0.0, n_draws=300, n_sides=3)

    worst, _, _ = _convergence(idata)

    assert worst["n_sides_mcse_fail"] == 3
    assert worst["worst_side"]["side"] in {"0", "1", "2"}
    assert worst["worst_side"]["mcse_frac"] > 0.05
    reasons = _failed_thresholds(worst, {})
    assert any("quantile MCSE" in reason for reason in reasons)


def test_the_mcse_limit_given_to_the_summary_decides_the_count() -> None:
    from pipeline.reconcilers.summarise import _convergence

    idata = _gate_idata(side_rho=0.9, global_rho=0.0)

    strict, _, _ = _convergence(idata, max_quantile_mcse_frac=1e-6)
    default, _, _ = _convergence(idata)

    assert strict["n_sides_mcse_fail"] == 3
    assert default["n_sides_mcse_fail"] == 0


def test_a_side_with_a_noisy_median_fails_the_absolute_limit_though_the_relative_one_passes() -> (
    None
):
    from pipeline.reconcilers.summarise import _convergence
    from pipeline.stages.reconcile import _failed_thresholds

    # A wide posterior (sd 1 in log units) with rho 0.9: the median MCSE is about
    # 0.09, a few percent of the interval width but far above 0.05 log units.
    idata = _gate_idata(side_rho=0.9, global_rho=0.0, side_sd=1.0)

    worst, _, _ = _convergence(idata)
    relaxed, _, _ = _convergence(idata, max_median_mcse_log=10.0)

    assert worst["n_sides_mcse_fail"] == 3
    assert worst["worst_side"]["limit"] == "absolute"
    assert worst["worst_side"]["quantile"] == 0.5
    assert worst["worst_side"]["mcse_frac"] < 0.05
    assert relaxed["n_sides_mcse_fail"] == 0
    assert any("median MCSE" in reason for reason in _failed_thresholds(worst, {}))
    assert _failed_thresholds(relaxed, {}) == []


def test_a_side_failing_only_the_relative_limit_is_labelled_relative() -> None:
    from pipeline.reconcilers.summarise import _convergence

    idata = _gate_idata(side_rho=0.9, global_rho=0.0, side_sd=0.1)

    worst, _, _ = _convergence(idata, max_quantile_mcse_frac=1e-6)

    assert worst["n_sides_mcse_fail"] == 3
    assert worst["worst_side"]["limit"] == "relative"


def test_the_checked_quantiles_follow_the_configured_ci_mass() -> None:
    from pipeline.reconcilers.summarise import _convergence

    idata = _gate_idata(side_rho=0.9, global_rho=0.0)

    narrow, _, _ = _convergence(idata, ci_mass=0.5, max_quantile_mcse_frac=1e-6)
    wide, _, _ = _convergence(idata, ci_mass=0.99, max_quantile_mcse_frac=1e-6)

    assert any(math.isclose(narrow["worst_side"]["quantile"], q) for q in (0.25, 0.5, 0.75))
    assert any(math.isclose(wide["worst_side"]["quantile"], q) for q in (0.005, 0.5, 0.995))
    assert wide["worst_side"]["width"] > narrow["worst_side"]["width"]


def test_the_failing_sides_are_named_with_the_limit_each_breaks() -> None:
    from pipeline.reconcilers.summarise import _convergence

    idata = _gate_idata(side_rho=0.9, global_rho=0.0, side_sd=1.0)

    worst, _, _ = _convergence(idata)

    assert worst["mcse_failed_side_ids"] == [0, 1, 2]
    assert worst["mcse_failed_limits"] == {"0": "absolute", "1": "absolute", "2": "absolute"}
    assert _convergence(_gate_idata(side_rho=0.0, global_rho=0.0))[0]["mcse_failed_side_ids"] == []


def test_a_fit_failing_only_per_side_mcse_passes_the_global_gate_and_names_the_sides() -> None:
    from pipeline.stages.reconcile import (
        _failed_thresholds,
        _mcse_failed_limits,
        _passes_global_thresholds,
    )

    worst = {
        "max_rhat": 1.001,
        "min_ess_bulk": 900.0,
        "divergences": 0,
        "n_sides_mcse_fail": 2,
        "mcse_failed_side_ids": [4, 9],
        "mcse_failed_limits": {"4": "relative", "9": "absolute"},
        "worst_side": {"side": "4", "limit": "relative", "mcse_frac": 0.2, "mcse": 0.1},
    }

    assert _passes_global_thresholds(worst, {})
    assert _mcse_failed_limits(worst) == {4: "relative", 9: "absolute"}
    assert any("quantile MCSE" in reason for reason in _failed_thresholds(worst, {}))


def test_a_global_failure_withholds_the_whole_fit_whatever_the_per_side_state() -> None:
    from pipeline.stages.reconcile import _passes_global_thresholds

    ok = {"max_rhat": 1.001, "min_ess_bulk": 900.0, "divergences": 0}

    assert _passes_global_thresholds(ok, {})
    assert not _passes_global_thresholds({**ok, "max_rhat": 1.3}, {})
    assert not _passes_global_thresholds({**ok, "min_ess_bulk": 10.0}, {})
    assert not _passes_global_thresholds({**ok, "divergences": 3}, {})
    assert _passes_global_thresholds({**ok, "n_sides_mcse_fail": 5}, {})


def test_every_side_keeps_its_column_in_the_neutral_draws_file(tmp_path: Any) -> None:
    from pipeline.reconcilers.neutral import NeutralSummary, write_neutral_draws

    summary = NeutralSummary(
        estimates=[],
        diagnostics={},
        side_ids=(10, 11, 12),
        battle_ids=(1, 1, 2),
        draws=np.arange(12.0).reshape(4, 3),
        m0_mean=0.0,
        s0_mean=1.0,
    )

    path = write_neutral_draws(
        summary, processed_root=tmp_path, run_id=5, database="db_a", started_at="t"
    )

    with np.load(path) as data:
        assert data["side_id"].tolist() == [10, 11, 12]
        assert data["draws"].shape == (4, 3)


def test_widen_bounds_pushes_each_bound_out_in_log_space() -> None:
    import math

    from pipeline.reconcilers.summarise import widen_bounds

    lo, hi = widen_bounds(100.0, 400.0, (0.2, 0.1))

    assert lo == pytest.approx(100.0 * math.exp(-0.2))
    assert hi == pytest.approx(400.0 * math.exp(0.1))


def test_only_failing_sides_get_a_margin_of_twice_their_bound_mcse() -> None:
    from pipeline.reconcilers.summarise import widening_margins

    idata = _gate_idata(side_rho=0.9, global_rho=0.0, side_sd=1.0)

    assert widening_margins(idata, 0.95, []) == {}
    margins = widening_margins(idata, 0.95, [1])

    assert set(margins) == {1}
    assert all(m > 0.0 and math.isfinite(m) for m in margins[1])


# ─── Contamination diagnostics and slope bookkeeping ─────────────────────────────


def _contamination_idata(pi_mean: float) -> tuple[Design, Any]:
    reports = [
        _report(1, side_id=1, value=10_000.0, url="https://example/a"),
        _report(2, side_id=1, value=10_500.0, url="https://example/a"),
        _report(
            3, side_id=1, value=10_000.0 * math.exp(5.0), lineage_id=777, url="https://example/a"
        ),
    ]
    design = _design(reports)
    rng = np.random.default_rng(3)
    shape = (2, 300)
    idata = _build_idata(
        design,
        InflationPrior(),
        mu_side_mean=math.log(10_000.0),
        mu_side_sd=0.01,
        beta_source_sd=0.01,
        extra_posterior={
            "pi_contam": ([], np.clip(rng.normal(pi_mean, 0.01, shape), 0.01, 0.99)),
            "sigma_estimate": ([], np.abs(rng.normal(0.1, 0.01, shape))),
            "sigma_round": ([], np.abs(rng.normal(0.1, 0.01, shape))),
        },
    )
    return design, idata


def test_every_row_above_one_half_is_listed_by_lineage_id_not_by_its_rounded_value() -> None:
    from pipeline.reconcilers.summarise import _contamination_diagnostics

    design, idata = _contamination_idata(0.1)

    out = _contamination_diagnostics(idata, design, 2.0)

    assert out["n_rows_contaminated"] == len(out["contaminated_rows"]) == 1
    row = out["contaminated_rows"][0]
    assert (row["side_id"], row["lineage_id"]) == (1, 777)
    assert row["prob"] > 0.5
    assert len(out["top_contaminated"]) <= 10
    assert out["top_contaminated"][0]["lineage_id"] == 777


def test_a_high_contamination_share_logs_a_warning_without_failing() -> None:
    from structlog.testing import capture_logs

    from pipeline.reconcilers.summarise import _contamination_diagnostics

    design, idata = _contamination_idata(0.4)
    with capture_logs() as captured:
        out = _contamination_diagnostics(idata, design, 2.0)
    assert out["pi_contam"]["q50"] > 0.3
    assert [e["event"] for e in captured if e["log_level"] == "warning"] == [
        "reconcile_contamination_share_is_high"
    ]

    design, idata = _contamination_idata(0.1)
    with capture_logs() as captured:
        _contamination_diagnostics(idata, design, 2.0)
    assert not [e for e in captured if e["log_level"] == "warning"]


def test_zero_valued_estimates_are_counted_and_logged_rather_than_dropped_silently() -> None:
    from structlog.testing import capture_logs

    from pipeline.reconcilers.records import SideEstimate
    from pipeline.reconcilers.summarise import _log_estimates

    def estimate(side_id: int, value: float | None) -> SideEstimate:
        return SideEstimate(
            side_id=side_id,
            quantity="troops",
            value=value,
            lo=value,
            hi=value,
            method="single_report_debiased",
            n_reports=1,
            n_sources=1,
            run_id=1,
            updated_at=_COMPUTED_AT,
        )

    with capture_logs() as captured:
        logs = _log_estimates([estimate(1, 5.0), estimate(2, 0.0), estimate(3, None)])

    assert set(logs) == {1}
    dropped = [e for e in captured if e["event"] == "reconcile_estimates_without_a_log"]
    assert [e["n_dropped"] for e in dropped] == [1]
