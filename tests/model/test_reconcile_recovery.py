"""Parameter-recovery tests for the reconcile hierarchical model.

This is the check that catches a mis-specified model before its estimates
reach the database: a wrong model still samples, converges, and emits
plausible numbers (``docs/handover-log.md`` records that happening three separate
times in this file's own history), so the test has to simulate data from a
*known* truth and check the fit against it, not merely check that it runs.

Everything here fits real MCMC (``pytest.mark.model``), needs no database,
and follows the project's Windows/no-C-compiler constraints:
``nuts_sampler="nutpie"`` and ``cores=1`` throughout.

## Corpus design

The main fit is 8 sources across 4 source types (deliberately few, so the
observation-weighted rotation in ``model.py`` and the zero-sum-within-type
constraint on ``u_source`` both do real work rather than being slack), 302
sides -- 300 at the spec's own report-count mix (90 sides x 4 reports, 90 x
2, 120 x 1) plus 2 extra sides purpose-built for the lineage test -- and a
genuine mix of claim regimes, including ``modern_scholarly`` anchor rows, so
``m0`` is identified in this corpus (see ``model.py``'s module docstring and
``docs/handover-log.md`` sections 19.8-19.9 for why an all-unlabelled corpus is a
different case, covered separately below).

## Judgement calls, collected here rather than scattered in comments

- Only troops are simulated, not casualties: the two quantities are fit
  independently from disjoint report subsets, so a recovery test of one
  quantity's machinery does not need the other to be present.
- Every main-corpus report has ``is_estimate=False`` and ``roundness=0.0``:
  those add two more variance components (``sigma_estimate``,
  ``sigma_round``) this file's threshold list does not ask about, and
  including them would only add unassigned truth values to the simulation.
- Noise is drawn Student-t (df=6), matching the model's own assumed
  ``nu`` prior mean, rather than Normal, so the simulated corpus does not
  flatter a heavy-tailed likelihood it does not actually need.
- The refits (censoring ablation, all-unlabelled, all-ancient-vs-mixed,
  split-source) use smaller corpora and shorter chains than the main fit;
  each test says what it used and why.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from pipeline.reconcilers.model import ModelPriors, SamplerSettings
from tests.model.conftest import (
    ANCIENT_CUTOFF_YEAR,
    MAIN_REGIME_WEIGHTS,
    MAIN_SCOPE_WEIGHTS,
    SOURCE_ID_OF,
    SOURCES,
    TRUE_B_TYPE,
    TRUE_DELTA_LEVEL,
    Fit,
    add_lineage_test_sides,
    coverage_fraction,
    fit_until_converged,
    log_errors,
    mean_log_width,
    run_fit,
    side_estimate_by_id,
    simulate_corpus,
    source_bias_for_source_id,
)

pytestmark = pytest.mark.model

SEED = 20260923

# The main fit's own sampler settings: proven in docs/handover-log.md §19.9 to reach
# ess(m0) > 400 and rhat < 1.02 at 1000 draws / 1000 tune / 4 chains on a
# similarly-shaped corpus.
# The spec's own budget (agents/reconcile.yaml mcmc_samples). At 1000 draws
# corpus_level reached only ess 312; at 2000 it passes both this suite and the
# production gate (reviewer measurement, docs/handover-log.md 20.10).
# The Student-t side prior (side_prior_nu) lets s0 trade off against the tails of
# z_side, which mixes more slowly: at 2000 draws / 1000 tune this corpus gave
# rhat 1.013 and ess 350 on s0, at 4000 / 1500 rhat 1.006 and ess 1144.
MAIN_SAMPLER = SamplerSettings(
    draws=4000, tune=1500, chains=4, cores=1, random_seed=SEED, nuts_sampler="nutpie"
)

TRUE_M0 = 9.4
TRUE_S0 = 1.1

INDEPENDENT_LINEAGE_SIDE = 90_001
REPEATED_LINEAGE_SIDE = 90_002


# ─── The main fit ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def main_truth() -> object:
    """Simulate the main corpus once, before any sampling."""
    rng = np.random.default_rng(SEED)
    corpus = simulate_corpus(
        rng,
        side_groups=[(90, 4), (90, 2), (120, 1)],
        m0=TRUE_M0,
        s0=TRUE_S0,
        regime_weights=MAIN_REGIME_WEIGHTS,
        scope_weights=MAIN_SCOPE_WEIGHTS,
        ancient_fraction=0.6,
        censor_upper_prob=0.10,
        censor_lower_prob=0.03,
    )
    corpus = add_lineage_test_sides(
        rng,
        corpus,
        independent_side_id=INDEPENDENT_LINEAGE_SIDE,
        repeated_side_id=REPEATED_LINEAGE_SIDE,
        start_report_id=1_000_000,
        m0=TRUE_M0,
        s0=TRUE_S0,
    )
    return corpus


def _m0_ess(fit: Fit) -> float:
    import arviz as az

    return float(np.asarray(az.ess(fit.idata, var_names=["m0"], method="bulk")["m0"].values))


def _main_fit_converged(fit: Fit) -> bool:
    """The convergence the main-fit claims rely on: stage gate, clean rhat, m0 identified."""
    from pipeline.stages.reconcile import _passes_global_thresholds

    worst = fit.summary.diagnostics["worst"]
    return (
        _passes_global_thresholds(worst, {})
        and worst["max_rhat"] < 1.01
        and worst["divergences"] == 0
        and _m0_ess(fit) > 400
    )


@pytest.fixture(scope="module")
def main_fit_run(
    main_truth: object, inflation_prior: object, troop_priors: ModelPriors
) -> tuple[Fit, int]:
    """Fit the main corpus, refitting on a new seed while it misses the convergence gate."""
    fit, seeds = fit_until_converged(
        lambda seed: run_fit(
            main_truth.reports,  # type: ignore[attr-defined]
            priors=troop_priors,
            inflation=inflation_prior,  # type: ignore[arg-type]
            sampler=replace(MAIN_SAMPLER, random_seed=seed),
            ancient_cutoff_year=ANCIENT_CUTOFF_YEAR,
        ),
        seed=SEED,
        converged=_main_fit_converged,
        describe=lambda fit: (fit.summary.diagnostics["worst"], f"ess(m0)={_m0_ess(fit):.0f}"),
    )
    return fit, len(seeds)


@pytest.fixture(scope="module")
def main_fit(main_fit_run: tuple[Fit, int]) -> Fit:
    """The accepted main-corpus fit; every main-fit test reads from this."""
    return main_fit_run[0]


def test_source_type_biases_recovered_within_tolerance_and_rank_order(main_fit: Fit) -> None:
    beta = np.asarray(main_fit.idata.posterior["b_type"].values)
    beta = beta.reshape(-1, beta.shape[-1])
    type_labels = [str(x) for x in main_fit.idata.posterior.coords["source_type"].values]

    fitted = {name: float(np.mean(beta[:, type_labels.index(name)])) for name in type_labels}
    errors = {name: fitted[name] - TRUE_B_TYPE[name] for name in type_labels}

    for name, error in errors.items():
        assert abs(error) < 0.15, f"{name}: fitted={fitted[name]:.3f} true={TRUE_B_TYPE[name]:.3f}"

    truth_rank = sorted(TRUE_B_TYPE, key=lambda n: TRUE_B_TYPE[n])
    fitted_rank = sorted(fitted, key=lambda n: fitted[n])
    assert fitted_rank == truth_rank, (fitted, TRUE_B_TYPE)


def test_inflation_is_learned_from_data_not_echoed_from_its_prior(
    main_fit: Fit, inflation_prior: object
) -> None:
    # "chronicle" is this corpus's well-observed regime: true bias 0.15,
    # chosen deliberately far from the fitted prior's mean for that regime
    # (~1.27 -- see moments_for("chronicle") on the real config/inflation_priors.yaml).
    # If the posterior merely echoed the prior, its 95% CI would still contain
    # that prior mean; recovering the true, different value pushes it out.
    prior_mu, _ = inflation_prior.moments_for("chronicle")  # type: ignore[attr-defined]

    g_free = np.asarray(main_fit.idata.posterior["g_free"].values)
    g_free = g_free.reshape(-1, g_free.shape[-1])
    free_labels = [str(x) for x in main_fit.idata.posterior.coords["regime_free"].values]
    samples = g_free[:, free_labels.index("chronicle")]

    lo, hi = np.quantile(samples, [0.025, 0.975])
    posterior_mean = float(np.mean(samples))

    assert not (lo <= prior_mu <= hi), (
        f"posterior CI [{lo:.3f}, {hi:.3f}] still contains the prior mean {prior_mu:.3f}"
    )
    assert abs(posterior_mean - 0.15) < 0.35, posterior_mean


def test_scope_offsets_recovered_within_tolerance(main_fit: Fit) -> None:
    delta = np.asarray(main_fit.idata.posterior["delta_level"].values)
    delta = delta.reshape(-1, delta.shape[-1])
    level_labels = [str(x) for x in main_fit.idata.posterior.coords["level"].values]

    for level, true_value in TRUE_DELTA_LEVEL.items():
        if level not in level_labels:
            continue
        fitted = float(np.mean(delta[:, level_labels.index(level)]))
        message = f"{level}: fitted={fitted:.3f} true={true_value:.3f}"
        assert abs(fitted - true_value) < 0.15, message


def test_side_estimate_coverage_is_calibrated_not_overconfident_or_useless(
    main_fit: Fit, main_truth: object
) -> None:
    coverage = coverage_fraction(main_fit.summary, main_truth.true_log_side)  # type: ignore[attr-defined]
    assert 0.88 <= coverage <= 0.99, coverage


def test_errors_match_the_stated_uncertainty_and_well_reported_sides_are_accurate(
    main_fit: Fit, main_truth: object
) -> None:
    # A fixed RMSE ceiling of 0.35 could not pass: with 40% singleton sides no
    # calibrated estimator gets there (measured 0.366 with coverage correct).
    # What matters is that the error matches the uncertainty the model states,
    # and that sides with four reports are genuinely accurate.
    truth = main_truth.true_log_side  # type: ignore[attr-defined]
    errors = log_errors(main_fit.summary, truth)
    rmse = float(np.sqrt(np.mean(errors**2)))
    estimates = {e.side_id: e for e in main_fit.summary.side_estimates}
    # A 95% interval on the log scale spans 2 * 1.96 posterior sds.
    stated_sd = np.array(
        [
            (math.log(float(estimates[sid].hi)) - math.log(float(estimates[sid].lo))) / 3.92
            for sid in truth
        ]
    )
    rms_stated = float(np.sqrt(np.mean(stated_sd**2)))
    assert 0.85 <= rmse / rms_stated <= 1.15, (rmse, rms_stated)

    four_ids = main_truth.side_ids_by_report_count[4]  # type: ignore[attr-defined]
    four_errors = log_errors(main_fit.summary, truth, four_ids)
    assert float(np.sqrt(np.mean(four_errors**2))) < 0.25


def test_singleton_side_coverage_is_high(main_fit: Fit, main_truth: object) -> None:
    singleton_ids = main_truth.side_ids_by_report_count[1]  # type: ignore[attr-defined]
    coverage = coverage_fraction(main_fit.summary, main_truth.true_log_side, singleton_ids)  # type: ignore[attr-defined]
    assert coverage >= 0.90, coverage


def test_singleton_intervals_are_much_wider_than_four_report_intervals(
    main_fit: Fit, main_truth: object
) -> None:
    singleton_ids = main_truth.side_ids_by_report_count[1]  # type: ignore[attr-defined]
    four_report_ids = main_truth.side_ids_by_report_count[4]  # type: ignore[attr-defined]

    singleton_width = mean_log_width(main_fit.summary, singleton_ids)
    four_report_width = mean_log_width(main_fit.summary, four_report_ids)

    # Four independent reports narrow an interval by sqrt(4) = 2 at most, and
    # any term both sides share (corpus level, regime offsets) pulls the ratio
    # below 2, so the 2.5 this asserted at first could never pass. Measured
    # 2.0. 1.8 still fails if a singleton's interval is quietly narrowed.
    assert singleton_width >= 1.8 * four_report_width, (singleton_width, four_report_width)


def test_convergence_is_clean(
    main_fit: Fit, main_fit_run: tuple[Fit, int], record_property: pytest.RecordProperty
) -> None:
    record_property("convergence_attempts", main_fit_run[1])
    worst = main_fit.summary.diagnostics["worst"]
    assert worst["max_rhat"] is not None and worst["max_rhat"] < 1.01, worst
    assert worst["divergences"] == 0, worst


def _neutral_main_converged(worst: dict[str, object]) -> bool:
    """The convergence the neutral-fit test asserts on the main corpus."""
    rhat, ess = worst["max_rhat"], worst["min_ess_bulk"]
    return (
        rhat is not None
        and rhat <= 1.05  # type: ignore[operator]
        and ess is not None
        and ess >= 400  # type: ignore[operator]
        and worst["divergences"] == 0
    )


@pytest.fixture(scope="module")
def main_neutral_fit(
    main_truth: object, inflation_prior: object, troop_priors: ModelPriors
) -> tuple[object, float, int]:
    """The regime-neutral fit on the main corpus, how long it took and how many attempts."""
    import time

    from pipeline.reconcilers.design import build_design
    from pipeline.reconcilers.model import build_model, sample
    from pipeline.reconcilers.summarise import _convergence

    design = build_design(
        main_truth.reports,  # type: ignore[attr-defined]
        quantity="troops",
        ancient_cutoff_year=ANCIENT_CUTOFF_YEAR,
    )
    started = time.perf_counter()
    model = build_model(
        design,
        priors=troop_priors,
        inflation=inflation_prior,  # type: ignore[arg-type]
        neutral=True,
    )
    idata, seeds = fit_until_converged(
        lambda seed: sample(model, replace(MAIN_SAMPLER, random_seed=seed)),
        seed=SEED,
        converged=lambda idata: _neutral_main_converged(_convergence(idata)[0]),
        describe=lambda idata: _convergence(idata)[:2],
    )
    return idata, time.perf_counter() - started, len(seeds)


def test_the_neutral_fit_clears_the_write_thresholds_on_the_main_corpus(
    main_neutral_fit: tuple[object, float, int], record_property: pytest.RecordProperty
) -> None:
    from pipeline.reconcilers.summarise import _convergence

    idata, seconds, attempts = main_neutral_fit
    record_property("convergence_attempts", attempts)
    worst, worst_variable, _ = _convergence(idata)  # type: ignore[arg-type]
    record_property("neutral_fit_seconds", round(seconds, 1))
    print(f"neutral fit: {seconds:.1f}s worst={worst} at {worst_variable}")

    assert worst["max_rhat"] is not None and worst["max_rhat"] <= 1.05, worst
    assert worst["min_ess_bulk"] is not None and worst["min_ess_bulk"] >= 400, worst
    assert worst["divergences"] == 0, worst


def test_the_corpus_level_is_identified_when_the_anchor_regime_is_present(main_fit: Fit) -> None:
    import arviz as az

    ess = az.ess(main_fit.idata, var_names=["m0"], method="bulk")
    ess_m0 = float(np.asarray(ess["m0"].values))
    assert ess_m0 > 400, ess_m0


def test_five_sources_repeating_one_claim_do_not_narrow_the_interval_like_five_independent_ones(
    main_fit: Fit,
) -> None:
    ests = side_estimate_by_id(main_fit.summary)
    independent = ests[INDEPENDENT_LINEAGE_SIDE]
    repeated = ests[REPEATED_LINEAGE_SIDE]

    width_independent = math.log(float(independent.hi) / float(independent.lo))
    width_repeated = math.log(float(repeated.hi) / float(repeated.lo))

    # Five genuinely independent lineages should narrow markedly harder than
    # five reports repeating one claim under a shared l_lineage term: the
    # repeated side's width should not have collapsed toward the independent
    # side's.
    assert width_repeated / width_independent > 1.3, (width_repeated, width_independent)


# ─── Ablation 1: treating a stated bound as a point value ───────────────────


CENSORING_SAMPLER = SamplerSettings(
    draws=500, tune=500, chains=4, cores=1, random_seed=SEED, nuts_sampler="nutpie"
)


def _censoring_corpus(*, lower: bool = False) -> object:
    """The bound-heavy corpus both censoring ablations fit (60 sides x 2 reports).

    Args:
        lower: Make the 80% bound reports "at least X" rather than "up to X".
    """
    rng = np.random.default_rng(SEED + (2 if lower else 1))
    return simulate_corpus(
        rng,
        side_groups=[(60, 2)],
        m0=TRUE_M0,
        s0=TRUE_S0,
        regime_weights=MAIN_REGIME_WEIGHTS,
        scope_weights=MAIN_SCOPE_WEIGHTS,
        ancient_fraction=0.6,
        censor_upper_prob=0.0 if lower else 0.80,
        censor_lower_prob=0.80 if lower else 0.0,
    )


@pytest.fixture(scope="module")
def censoring_ablation() -> tuple[Fit, Fit, np.ndarray]:
    """Fit the same censored corpus correctly, and again with bounds cleared.

    A smaller corpus (60 sides x 2 reports = 120 obs) and a shorter chain
    (500 draws / 500 tune / 4 chains) than the main fit, since this is a
    ratio comparison between two fits of the same data rather than a
    from-scratch recovery check.
    """
    corpus = _censoring_corpus()
    priors = replace(ModelPriors.from_params({}, quantity="troops"), bound_likelihood="censored")
    from pipeline.reconcilers.model import load_inflation_prior

    inflation = load_inflation_prior()
    sampler = CENSORING_SAMPLER

    correct_fit = run_fit(
        corpus.reports,
        priors=priors,
        inflation=inflation,
        sampler=sampler,
        ancient_cutoff_year=ANCIENT_CUTOFF_YEAR,
    )

    naive_reports = [replace(r, is_upper_bound=False, is_lower_bound=False) for r in corpus.reports]
    naive_fit = run_fit(
        naive_reports,
        priors=priors,
        inflation=inflation,
        sampler=sampler,
        ancient_cutoff_year=ANCIENT_CUTOFF_YEAR,
    )

    true_log = np.array([corpus.true_log_side[sid] for sid in sorted(corpus.true_log_side)])
    return correct_fit, naive_fit, true_log


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Open model design question, docs/handover-log.md 20.4: pure censoring carries no "
        "information on how far below an 'up to X' figure the truth lies, so with "
        "80% bound reports the censored fit is worse (RMSE 1.95) than ignoring the "
        "bounds (0.36). strict=True makes this fail loudly once the design is fixed."
    ),
)
def test_treating_a_stated_bound_as_a_point_value_degrades_the_recovered_scale(
    censoring_ablation: tuple[Fit, Fit, np.ndarray],
) -> None:
    correct_fit, naive_fit, _ = censoring_ablation
    ids = sorted({e.side_id for e in correct_fit.summary.side_estimates})

    true_log = {sid: v for sid, v in zip(ids, censoring_ablation[2], strict=True)}
    rmse_correct = float(np.sqrt(np.mean(log_errors(correct_fit.summary, true_log) ** 2)))
    rmse_naive = float(np.sqrt(np.mean(log_errors(naive_fit.summary, true_log) ** 2)))

    assert rmse_naive >= 1.4 * rmse_correct, (rmse_naive, rmse_correct)


@pytest.fixture(scope="module")
def exgaussian_bounds_fit() -> Fit:
    """The same bound-heavy corpus fitted with ``bound_likelihood: exgaussian``."""
    from pipeline.reconcilers.model import load_inflation_prior

    corpus = _censoring_corpus()
    priors = replace(ModelPriors.from_params({}, quantity="troops"), bound_likelihood="exgaussian")
    return run_fit(
        corpus.reports,  # type: ignore[attr-defined]
        priors=priors,
        inflation=load_inflation_prior(),
        sampler=CENSORING_SAMPLER,
        ancient_cutoff_year=ANCIENT_CUTOFF_YEAR,
    )


def test_an_exgaussian_bound_likelihood_recovers_what_pure_censoring_loses(
    censoring_ablation: tuple[Fit, Fit, np.ndarray],
    exgaussian_bounds_fit: Fit,
) -> None:
    """The ExGaussian bound beats pure censoring and is no worse than ignoring bounds.

    This is not the 1.4x claim of the xfail above. On this corpus a bound's
    excess is a half-normal with sd 0.3, so treating bounds as points costs
    little, and the ExGaussian fit is about level with it rather than clearly
    ahead. What it fixes is the censored fit's drift, which is the open design
    question in docs/handover-log.md 20.4.
    """
    censored_fit, naive_fit, true_values = censoring_ablation
    ids = sorted({e.side_id for e in exgaussian_bounds_fit.summary.side_estimates})
    true_log = {sid: v for sid, v in zip(ids, true_values, strict=True)}

    def rmse(fit: Fit) -> float:
        return float(np.sqrt(np.mean(log_errors(fit.summary, true_log) ** 2)))

    rmse_exgaussian, rmse_censored, rmse_naive = (
        rmse(exgaussian_bounds_fit),
        rmse(censored_fit),
        rmse(naive_fit),
    )
    print(
        f"RMSE exgaussian {rmse_exgaussian:.3f}, censored {rmse_censored:.3f}, "
        f"bounds-as-points {rmse_naive:.3f}"
    )

    assert rmse_censored >= 2.0 * rmse_exgaussian, (rmse_censored, rmse_exgaussian)
    assert rmse_exgaussian <= 1.25 * rmse_naive, (rmse_exgaussian, rmse_naive)


@pytest.fixture(scope="module")
def lower_bound_ablation() -> tuple[Fit, Fit, np.ndarray]:
    """The mirror corpus, 80% "at least X" bounds, fitted censored and ExGaussian.

    The first ablation has upper bounds only, which would not catch a sign
    error in the lower-bound branch of the ExGaussian (it is fitted as
    ``-log y ~ ExGaussian(-eta, ...)``).
    """
    from pipeline.reconcilers.model import load_inflation_prior

    corpus = _censoring_corpus(lower=True)
    base = ModelPriors.from_params({}, quantity="troops")
    inflation = load_inflation_prior()
    fits = [
        run_fit(
            corpus.reports,  # type: ignore[attr-defined]
            priors=replace(base, bound_likelihood=kind),
            inflation=inflation,
            sampler=CENSORING_SAMPLER,
            ancient_cutoff_year=ANCIENT_CUTOFF_YEAR,
        )
        for kind in ("exgaussian", "censored")
    ]
    true_log = np.array([corpus.true_log_side[sid] for sid in sorted(corpus.true_log_side)])  # type: ignore[attr-defined]
    return fits[0], fits[1], true_log


def test_an_exgaussian_lower_bound_likelihood_also_beats_pure_censoring(
    lower_bound_ablation: tuple[Fit, Fit, np.ndarray],
) -> None:
    exgaussian_fit, censored_fit, true_values = lower_bound_ablation
    ids = sorted({e.side_id for e in exgaussian_fit.summary.side_estimates})
    true_log = {sid: v for sid, v in zip(ids, true_values, strict=True)}

    def rmse(fit: Fit) -> float:
        return float(np.sqrt(np.mean(log_errors(fit.summary, true_log) ** 2)))

    rmse_exgaussian, rmse_censored = rmse(exgaussian_fit), rmse(censored_fit)
    print(f"lower bounds: RMSE exgaussian {rmse_exgaussian:.3f}, censored {rmse_censored:.3f}")
    excess = exgaussian_fit.summary.diagnostics["bound_excess"]
    print(f"lower bounds: bound_excess {excess}")

    # Measured: 0.304 against 2.143.
    assert rmse_censored >= 2.0 * rmse_exgaussian, (rmse_censored, rmse_exgaussian)


# ─── Ablation 2: an all-unlabelled corpus ───────────────────────────────────


def _all_unlabelled_converged(worst: dict[str, object]) -> bool:
    """Only rhat is asserted on this corpus: ess(m0) is not meaningful on its shape."""
    rhat = worst["max_rhat"]
    return rhat is not None and rhat < 1.02  # type: ignore[operator]


@pytest.fixture(scope="module")
def all_unlabelled_fit() -> tuple[Fit, float, int]:
    """Fit a corpus where every report is claim_regime='unlabelled'.

    60 sides (20x4, 20x2, 20x1 = 140 obs), scope fixed to 'engaged' so the
    only free nuisance term is the regime offset itself -- the point this
    test checks. 1500 draws / 1500 tune / 4 chains (800/800 sat at rhat 1.02
    on CI, since m0 and the unlabelled offset form a ridge where only their
    sum is identified): only max_rhat is asserted here (the brief accepts
    < 1.02), not ess(m0), which model.py's own docstring says is not
    meaningful on this corpus shape.
    """
    rng = np.random.default_rng(SEED + 2)
    true_offset = 0.55
    corpus = simulate_corpus(
        rng,
        side_groups=[(20, 4), (20, 2), (20, 1)],
        m0=9.3,
        s0=1.0,
        regime_weights={"unlabelled": 1.0},
        scope_weights={"engaged": 1.0},
        ancient_fraction=0.5,
        g_regime={
            "modern_scholarly": 0.0,
            "ancient_claim": 0.0,
            "chronicle": 0.0,
            "administrative_partisan": 0.0,
            "staff_return": 0.0,
            "unlabelled": true_offset,
        },
    )
    priors = ModelPriors.from_params({}, quantity="troops")
    from pipeline.reconcilers.model import load_inflation_prior

    inflation = load_inflation_prior()
    sampler = SamplerSettings(
        draws=1500, tune=1500, chains=4, cores=1, random_seed=SEED, nuts_sampler="nutpie"
    )
    fit, seeds = fit_until_converged(
        lambda seed: run_fit(
            corpus.reports,
            priors=priors,
            inflation=inflation,
            sampler=replace(sampler, random_seed=seed),
            ancient_cutoff_year=ANCIENT_CUTOFF_YEAR,
        ),
        seed=SEED,
        converged=lambda fit: _all_unlabelled_converged(fit.summary.diagnostics["worst"]),
        describe=lambda fit: fit.summary.diagnostics["worst"],
    )
    true_identified_level = 9.3 + true_offset
    return fit, true_identified_level, len(seeds)


def test_an_all_unlabelled_corpus_recovers_the_identified_level_not_m0(
    all_unlabelled_fit: tuple[Fit, float, int], record_property: pytest.RecordProperty
) -> None:
    fit, true_identified_level, attempts = all_unlabelled_fit
    record_property("convergence_attempts", attempts)
    identified = fit.summary.diagnostics["identified_level"]

    assert identified["mean"] is not None
    assert abs(identified["mean"] - true_identified_level) < 0.15, (
        identified["mean"],
        true_identified_level,
    )

    contraction_unlabelled = fit.summary.diagnostics["contraction"]["g_free"]["unlabelled"]
    assert contraction_unlabelled is not None
    assert contraction_unlabelled < 0.2, contraction_unlabelled

    worst = fit.summary.diagnostics["worst"]
    assert worst["max_rhat"] is not None and worst["max_rhat"] < 1.02, worst


# ─── Ablation 3: an all-ancient corpus vs a mixed one ───────────────────────


@pytest.fixture(scope="module")
def era_centring_fits() -> tuple[Fit, Fit, dict[int, float]]:
    """Fit the same true sides twice: once all-ancient, once mixed era.

    40 sides x 2 reports = 80 obs, 600 draws / 600 tune / 4 chains. Both
    corpora share the same per-side truth (same rng draws for side levels),
    differing only in which battles are dated ancient vs later.
    """
    priors = ModelPriors.from_params({}, quantity="troops")
    from pipeline.reconcilers.model import load_inflation_prior

    inflation = load_inflation_prior()
    sampler = SamplerSettings(
        draws=600, tune=600, chains=4, cores=1, random_seed=SEED, nuts_sampler="nutpie"
    )

    rng_ancient = np.random.default_rng(SEED + 3)
    ancient_corpus = simulate_corpus(
        rng_ancient,
        side_groups=[(40, 2)],
        m0=TRUE_M0,
        s0=TRUE_S0,
        regime_weights=MAIN_REGIME_WEIGHTS,
        scope_weights=MAIN_SCOPE_WEIGHTS,
        ancient_fraction=1.0,
    )
    rng_mixed = np.random.default_rng(SEED + 3)
    mixed_corpus = simulate_corpus(
        rng_mixed,
        side_groups=[(40, 2)],
        m0=TRUE_M0,
        s0=TRUE_S0,
        regime_weights=MAIN_REGIME_WEIGHTS,
        scope_weights=MAIN_SCOPE_WEIGHTS,
        ancient_fraction=0.5,
    )

    ancient_fit = run_fit(
        ancient_corpus.reports,
        priors=priors,
        inflation=inflation,
        sampler=sampler,
        ancient_cutoff_year=ANCIENT_CUTOFF_YEAR,
    )
    mixed_fit = run_fit(
        mixed_corpus.reports,
        priors=priors,
        inflation=inflation,
        sampler=sampler,
        ancient_cutoff_year=ANCIENT_CUTOFF_YEAR,
    )
    return ancient_fit, mixed_fit, ancient_corpus.true_log_side


def test_an_all_ancient_corpus_recovers_the_same_side_estimates_as_a_mixed_one(
    era_centring_fits: tuple[Fit, Fit, dict[int, float]],
) -> None:
    ancient_fit, mixed_fit, true_log_side = era_centring_fits

    ancient_errors = log_errors(ancient_fit.summary, true_log_side)
    mixed_errors = log_errors(mixed_fit.summary, true_log_side)

    # An uncentred era covariate turns into a second intercept on an
    # all-ancient corpus and shifts every estimate by a constant; a centred
    # one degrades gracefully to an identically-zero column. The two mean
    # residuals should therefore be close, not offset by a shared constant.
    assert abs(float(np.mean(ancient_errors)) - float(np.mean(mixed_errors))) < 0.15, (
        float(np.mean(ancient_errors)),
        float(np.mean(mixed_errors)),
    )


# ─── Ablation 4: splitting one source into two source_ids ───────────────────


@pytest.fixture(scope="module")
def source_split_fits() -> tuple[Fit, Fit]:
    """Fit a small corpus twice: one source's rows under one source_id, then two.

    30 sides x 2 reports = 60 obs, 400 draws / 400 tune / 4 chains. Neither
    ``build_design`` nor ``build_model`` reads ``source_id`` -- both group by
    ``(source_type, url)`` -- so this is expected to reproduce essentially
    the same fit; the test is a regression guard against that changing.
    """
    rng = np.random.default_rng(SEED + 4)
    corpus = simulate_corpus(
        rng,
        side_groups=[(30, 2)],
        m0=TRUE_M0,
        s0=TRUE_S0,
        regime_weights=MAIN_REGIME_WEIGHTS,
        scope_weights=MAIN_SCOPE_WEIGHTS,
        ancient_fraction=0.6,
    )
    priors = ModelPriors.from_params({}, quantity="troops")
    from pipeline.reconcilers.model import load_inflation_prior

    inflation = load_inflation_prior()
    sampler = SamplerSettings(
        draws=400, tune=400, chains=4, cores=1, random_seed=SEED, nuts_sampler="nutpie"
    )

    target_key = SOURCES[0]
    unsplit_fit = run_fit(
        corpus.reports,
        priors=priors,
        inflation=inflation,
        sampler=sampler,
        ancient_cutoff_year=ANCIENT_CUTOFF_YEAR,
    )

    split_reports = []
    seen = 0
    for r in corpus.reports:
        if r.source_key == target_key:
            new_id = SOURCE_ID_OF[target_key] if seen % 2 == 0 else SOURCE_ID_OF[target_key] + 500
            split_reports.append(replace(r, source_id=new_id))
            seen += 1
        else:
            split_reports.append(r)
    split_fit = run_fit(
        split_reports,
        priors=priors,
        inflation=inflation,
        sampler=sampler,
        ancient_cutoff_year=ANCIENT_CUTOFF_YEAR,
    )
    return unsplit_fit, split_fit


def test_splitting_one_source_into_two_rows_does_not_halve_its_recovered_bias(
    source_split_fits: tuple[Fit, Fit],
) -> None:
    unsplit_fit, split_fit = source_split_fits
    target_key = SOURCES[0]

    assert split_fit.design.n_sources == unsplit_fit.design.n_sources

    unsplit_bias = source_bias_for_source_id(unsplit_fit.summary, SOURCE_ID_OF[target_key])
    split_bias_a = source_bias_for_source_id(split_fit.summary, SOURCE_ID_OF[target_key])
    split_bias_b = source_bias_for_source_id(split_fit.summary, SOURCE_ID_OF[target_key] + 500)

    # The two source_ids sharing a key must be fanned out identically.
    assert split_bias_a.bias_mu == pytest.approx(split_bias_b.bias_mu)

    assert abs(split_bias_a.bias_mu - unsplit_bias.bias_mu) < 0.1, (
        split_bias_a.bias_mu,
        unsplit_bias.bias_mu,
    )


# ─── Nested sources: every key reports exactly one battle ────────────────────
#
# The real corpus is shaped like this: each source key is one battle's infobox,
# so a per-key bias is the battle level under another name. Every test above
# uses cross-classified sources (8 keys across 300 sides), which is why they
# all passed while the production fit gave every side the same estimate.

# The two sides of a battle are alike in scale (Kursk's are both in the millions,
# a skirmish's both in the hundreds), so most of the side spread is battle-level.
# That is what lets a per-key bias stand in for it.
NESTED_BATTLE_SD = 1.9
NESTED_SIDE_SD = 0.6
NESTED_TRUE_SD = 2.0
NESTED_N_BATTLES = 150


def _simulate_nested(
    seed: int,
    *,
    junk_fraction: float = 0.0,
    noise_sd: float = 0.4,
    key_bias_sd: float = 0.3,
    p_second_row: float = 0.4,
    heavy_tail_fraction: float = 0.0,
    battle_sd: float = NESTED_BATTLE_SD,
    contamination_fraction: float = 0.0,
) -> tuple[list, dict[int, float]]:  # type: ignore[type-arg]
    """One source key per battle, covering both sides, plus optional junk rows.

    ``heavy_tail_fraction`` of battles sit 2-3 log units out in either direction
    (the Kurks and the skirmishes), and ``contamination_fraction`` of sides get
    one extra row for a component of the side (a fifth to a half of it) that no
    rule can identify: one-sided contamination below the side total.
    """
    from pipeline.reconcilers.records import Report, SourceKey

    rng = np.random.default_rng(seed)
    reports: list[Report] = []
    truth: dict[int, float] = {}
    report_id = 0
    for battle in range(1, NESTED_N_BATTLES + 1):
        key = SourceKey("wikipedia_infobox", f"https://example.org/battle-{battle}")
        key_bias = rng.normal(0.0, key_bias_sd)
        battle_level = battle_sd * rng.normal()
        if heavy_tail_fraction > 0.0 and rng.random() < heavy_tail_fraction:
            battle_level += float(rng.choice([-1.0, 1.0]) * rng.uniform(2.0, 3.0))
        for side_in_battle in (0, 1):
            side_id = 2 * battle + side_in_battle
            truth[side_id] = 9.2 + battle_level + NESTED_SIDE_SD * rng.normal()
            n_rows = 2 if rng.random() < p_second_row else 1
            component_rows = 0
            if contamination_fraction > 0.0 and rng.random() < contamination_fraction:
                component_rows = 1
            for row_number in range(n_rows + component_rows):
                is_component = row_number >= n_rows
                report_id += 1
                eta = truth[side_id] + key_bias + float(rng.standard_t(6) * noise_sd)
                if is_component:
                    eta += math.log(float(rng.uniform(0.15, 0.5)))
                reports.append(
                    Report(
                        report_id=report_id,
                        side_id=side_id,
                        battle_id=battle,
                        source_id=battle,
                        source_key=key,
                        source_type="wikipedia_infobox",
                        quantity="troops",
                        reported_value=math.exp(eta),
                        extracted_context=f"{math.exp(eta):,.0f}",
                        year_astronomical=-200 if battle % 2 else 1800,
                        claim_regime="unlabelled",
                        lineage_id=report_id,
                    )
                )
    if junk_fraction > 0.0:
        n_junk = int(junk_fraction * len(reports))
        vessels = ("ships", "helicopters", "transports", "galleys")
        sides = list(truth)
        for i in range(n_junk):
            report_id += 1
            side_id = sides[int(rng.integers(len(sides)))]
            battle = side_id // 2
            # Half are small counts, half are vessel counts above the floor
            # that only the context can identify.
            if i % 2 == 0:
                value, context = float(rng.integers(1, 100)), "a plain figure"
            else:
                value = float(rng.integers(150, 900))
                context = f"{value:.0f} {vessels[(i // 2) % len(vessels)]}"
            reports.append(
                Report(
                    report_id=report_id,
                    side_id=side_id,
                    battle_id=battle,
                    source_id=battle,
                    source_key=SourceKey(
                        "wikipedia_infobox", f"https://example.org/battle-{battle}"
                    ),
                    source_type="wikipedia_infobox",
                    quantity="troops",
                    reported_value=value,
                    extracted_context=context,
                    year_astronomical=1800,
                    claim_regime="unlabelled",
                    lineage_id=report_id,
                )
            )
    return reports, truth


def _fit_nested(
    reports: list,  # type: ignore[type-arg]
    inflation: object,
    troop_priors: ModelPriors,
) -> tuple[np.ndarray, np.ndarray]:
    fit = run_fit(
        reports,
        priors=troop_priors,
        inflation=inflation,  # type: ignore[arg-type]
        sampler=SamplerSettings(
            draws=1000, tune=1000, chains=4, cores=1, random_seed=SEED, nuts_sampler="nutpie"
        ),
        ancient_cutoff_year=ANCIENT_CUTOFF_YEAR,
    )
    mu = np.asarray(fit.idata.posterior["mu_side"].values)
    mu = mu.reshape(-1, mu.shape[-1])
    labels = [int(x) for x in fit.idata.posterior.coords["side"].values]
    return np.median(mu, axis=0), np.array(labels)


def _assert_nested_recovery(
    medians: np.ndarray, labels: np.ndarray, truth: dict[int, float]
) -> None:
    true_values = np.array([truth[int(s)] for s in labels])
    assert np.std(medians) >= 0.5 * NESTED_TRUE_SD, np.std(medians)
    assert np.corrcoef(medians, true_values)[0, 1] >= 0.8


def test_a_nested_corpus_keeps_the_spread_of_the_sides(
    inflation_prior: object, troop_priors: ModelPriors
) -> None:
    reports, truth = _simulate_nested(SEED)

    medians, labels = _fit_nested(reports, inflation_prior, troop_priors)

    _assert_nested_recovery(medians, labels, truth)


def test_a_nested_corpus_with_junk_rows_keeps_the_spread_once_they_are_filtered(
    inflation_prior: object, troop_priors: ModelPriors
) -> None:
    from pipeline.reconcilers.plausibility import filter_troop_reports
    from pipeline.reconcilers.records import ReconcileCounts

    reports, truth = _simulate_nested(SEED + 1, junk_fraction=0.10)
    counts = ReconcileCounts()

    kept = filter_troop_reports(reports, counts, min_plausible_troops=100)

    assert counts.excluded_implausible_small + counts.excluded_non_personnel > 0
    medians, labels = _fit_nested(kept, inflation_prior, troop_priors)
    _assert_nested_recovery(medians, labels, truth)


HEAVY_TAIL_SAMPLER = SamplerSettings(
    draws=3000,
    tune=1500,
    chains=4,
    cores=1,
    target_accept=0.95,
    random_seed=SEED,
    nuts_sampler="nutpie",
)


def _fit_nested_intervals(
    reports: list,  # type: ignore[type-arg]
    inflation: object,
    troop_priors: ModelPriors,
    sampler: SamplerSettings | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Medians, 95% bounds, side labels and the convergence summary of ``mu_side``."""
    from pipeline.reconcilers.summarise import _convergence

    fit = run_fit(
        reports,
        priors=troop_priors,
        inflation=inflation,  # type: ignore[arg-type]
        sampler=sampler
        or SamplerSettings(
            draws=1000, tune=1000, chains=4, cores=1, random_seed=SEED, nuts_sampler="nutpie"
        ),
        ancient_cutoff_year=ANCIENT_CUTOFF_YEAR,
    )
    worst, _, _ = _convergence(fit.idata)
    mu = np.asarray(fit.idata.posterior["mu_side"].values)
    mu = mu.reshape(-1, mu.shape[-1])
    labels = np.array([int(x) for x in fit.idata.posterior.coords["side"].values])
    lo, hi = np.quantile(mu, [0.025, 0.975], axis=0)
    return np.median(mu, axis=0), lo, hi, labels, worst


HEAVY_TAIL_SEEDS = (SEED + 2, SEED + 3, SEED + 4)


def _fit_nested_intervals_converged(
    reports: list,  # type: ignore[type-arg]
    inflation: object,
    troop_priors: ModelPriors,
    sampler: SamplerSettings,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]], int]:
    """:func:`_fit_nested_intervals`, refitted on a new seed while it misses the write gate.

    Returns:
        The accepted fit's intervals and convergence summary, and the attempt count.
    """
    from pipeline.stages.reconcile import _passes_global_thresholds

    result, seeds = fit_until_converged(
        lambda seed: _fit_nested_intervals(
            reports, inflation, troop_priors, replace(sampler, random_seed=seed)
        ),
        seed=sampler.random_seed,
        converged=lambda result: _passes_global_thresholds(result[-1], {}),
        describe=lambda result: result[-1],
    )
    return result, len(seeds)


def _simulate_heavy_tail(seed: int) -> tuple[list, dict[int, float]]:  # type: ignore[type-arg]
    return _simulate_nested(
        seed,
        heavy_tail_fraction=0.06,
        contamination_fraction=0.25,
        battle_sd=0.5,
        noise_sd=1.0,
        key_bias_sd=0.5,
        p_second_row=0.0,
    )


def _top_decile_covered(
    lo: np.ndarray, hi: np.ndarray, labels: np.ndarray, truth: dict[int, float]
) -> dict[int, bool]:
    """For each side in the top decile by true level, whether its 95% interval covers it."""
    true_values = np.array([truth[int(s)] for s in labels])
    k = max(1, len(true_values) // 10)
    inside = (lo <= true_values) & (true_values <= hi)
    return {int(labels[i]): bool(inside[i]) for i in np.argsort(true_values)[-k:]}


def test_a_student_t_side_prior_covers_extreme_sides_no_worse_than_a_normal_one(
    inflation_prior: object, troop_priors: ModelPriors, record_property: pytest.RecordProperty
) -> None:
    """A Student-t side prior is no worse than a Normal one at keeping extremes in their intervals.

    A few sides sit 2-3 log units out, and a quarter carry a one-sided component
    row. Most sides are tightly clustered (battle sd 0.5) with noisy single
    reports, so a Normal side prior reads a large side's report as an outlier
    around a typical one and shrinks it toward the middle.

    Three seeds, each fitted twice under the default (student_t) likelihood:
    once with the Student-t side prior and once with a Normal one
    (``side_prior_nu=1e6``). Every fit must clear the stage's write gate first;
    a fit that misses it is refitted on a new seed, as the model stage does, and
    only a miss on every attempt fails. The coverage claims are never retried.
    The top decile by true level is pooled across seeds (k = 90) and compared on
    the same sides: ``t_only`` counts sides the Student-t fit covers and the
    Normal fit does not, ``normal_only`` the reverse. The claim is
    non-inferiority: ``t_only >= normal_only``, and pooled Student-t coverage is
    at least pooled Normal coverage.

    The measured advantage is small, and no absolute coverage floor is asserted.
    Over 5 seeds (SEED + 2 to SEED + 6, 150 top-decile sides) the Student-t side
    prior covers 78 (0.520) and the Normal side prior 72 (0.480); t_only 13,
    normal_only 7. A fixed seed is bit-reproducible on one dependency set (see
    requirements/ci-lock.txt); before CI was locked, two runs on the
    three test seeds gave t_only 10 / normal_only 1 and t_only 11 / normal_only 1
    (pooled coverage 57/90 vs 47/90 in the latter). An
    earlier version asserted an absolute coverage floor of 0.45, which failed on
    CI at 13/30 and is not true of the model.
    """
    normal_priors = replace(troop_priors, side_prior_nu=1.0e6)
    t_only = normal_only = k_total = t_covered = normal_covered = 0
    attempts: list[int] = []
    for seed in HEAVY_TAIL_SEEDS:
        reports, truth = _simulate_heavy_tail(seed)
        sampler = replace(HEAVY_TAIL_SAMPLER, random_seed=seed)
        fits = {}
        for name, priors in (("t", troop_priors), ("normal", normal_priors)):
            (*interval, _), n_attempts = _fit_nested_intervals_converged(
                reports, inflation_prior, priors, sampler
            )
            attempts.append(n_attempts)
            fits[name] = _top_decile_covered(interval[1], interval[2], interval[3], truth)
        t_cover, normal_cover = fits["t"], fits["normal"]
        t_only += sum(t_cover[s] and not normal_cover[s] for s in t_cover)
        normal_only += sum(normal_cover[s] and not t_cover[s] for s in t_cover)
        t_covered += sum(t_cover.values())
        normal_covered += sum(normal_cover.values())
        k_total += len(t_cover)
        print(
            f"heavy-tail seed {seed}: t covers {sum(t_cover.values())}/{len(t_cover)}, "
            f"normal covers {sum(normal_cover.values())}/{len(normal_cover)}"
        )
    record_property("convergence_attempts", ",".join(map(str, attempts)))
    record_property("t_only", t_only)
    record_property("normal_only", normal_only)
    summary = (
        f"heavy-tail pooled k={k_total}: t_only={t_only} normal_only={normal_only} "
        f"t_covered={t_covered} ({t_covered / k_total:.3f}) "
        f"normal_covered={normal_covered} ({normal_covered / k_total:.3f})"
    )
    print(summary)

    assert k_total == 90, k_total
    assert t_only >= normal_only, summary
    assert t_covered >= normal_covered, summary
