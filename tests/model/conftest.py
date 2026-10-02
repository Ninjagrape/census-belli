"""Shared truth, corpus simulation and fit machinery for the reconcile recovery tests.

Nothing here is a test. :func:`simulate_corpus` is the one piece of care: it
must generate data the way the model in ``pipeline/reconcilers/model.py``
actually reads it, or a recovery test would be checking its own simulation
bug rather than the model. In particular the censoring direction mirrors
``design.py``'s own docstring exactly (an "up to X" report's *true* value is
generated *below* X, via ``exp(latent + |noise|)``), and the era covariate is
applied only to ``unlabelled`` rows, matching ``model.py``'s gating.

All randomness goes through an explicit ``numpy.random.Generator`` seeded by
the caller; nothing here reads global numpy state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pipeline.reconcilers.design import Design, build_design
from pipeline.reconcilers.model import (
    InflationPrior,
    ModelPriors,
    SamplerSettings,
    build_model,
    load_inflation_prior,
    sample,
)
from pipeline.reconcilers.records import Report, SideEstimate, SourceBias, SourceKey
from pipeline.reconcilers.summarise import SummaryResult, summarise_fit

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    import arviz as az

# ─── Corpus truth: source types, sources, biases ────────────────────────────

# Four types, two sources each -- deliberately few, so ZeroSumNormal and the
# within-type deviation actually matter (see test_reconcile_recovery.py's
# module docstring).
SOURCE_TYPES: tuple[str, ...] = (
    "peer_reviewed",
    "academic_book",
    "wikipedia_infobox",
    "primary_source",
)

# Zero-sum by construction; spaced by >= 0.4 so a +-0.15 per-type recovery
# error can never flip the rank order.
TRUE_B_TYPE: dict[str, float] = {
    "peer_reviewed": -0.75,
    "academic_book": -0.25,
    "wikipedia_infobox": 0.30,
    "primary_source": 0.70,
}

# Within-type deviations, zero-sum within each pair by construction.
TRUE_U_SOURCE_MAGNITUDE: dict[str, float] = {
    "peer_reviewed": 0.12,
    "academic_book": 0.08,
    "wikipedia_infobox": 0.15,
    "primary_source": 0.10,
}

SOURCES: tuple[SourceKey, ...] = tuple(
    SourceKey(t, f"https://example.org/{t}-{i}") for t in SOURCE_TYPES for i in (1, 2)
)

# A stable source_id per SourceKey, as if each had been inserted once.
SOURCE_ID_OF: dict[SourceKey, int] = {key: 1000 + i for i, key in enumerate(SOURCES)}


def true_u_source(key: SourceKey) -> float:
    """This source's true within-type deviation."""
    magnitude = TRUE_U_SOURCE_MAGNITUDE[key.source_type]
    sign = 1.0 if key.url.endswith("-1") else -1.0
    return sign * magnitude


TRUE_SOURCE_SIGMA: dict[SourceKey, float] = {
    SOURCES[0]: 0.28,
    SOURCES[1]: 0.32,
    SOURCES[2]: 0.25,
    SOURCES[3]: 0.30,
    SOURCES[4]: 0.35,
    SOURCES[5]: 0.33,
    SOURCES[6]: 0.27,
    SOURCES[7]: 0.29,
}

# ─── Claim regimes ───────────────────────────────────────────────────────────

# modern_scholarly is the anchor and must stay exactly 0. "chronicle" is the
# regime the main-fit test picks as "well observed": true value 0.15 sits
# more than a prior sd away from the fitted prior's mean of ~1.27 (see
# tests/model/test_reconcile_recovery.py), so a genuine posterior update is
# distinguishable from an echoed prior.
TRUE_G_REGIME: dict[str, float] = {
    "modern_scholarly": 0.0,
    "ancient_claim": 1.5,
    "chronicle": 0.15,
    "administrative_partisan": 0.7,
    "staff_return": 0.5,
    "unlabelled": 0.85,
}

MAIN_REGIME_WEIGHTS: dict[str, float] = {
    "modern_scholarly": 0.08,
    "ancient_claim": 0.08,
    "chronicle": 0.15,
    "administrative_partisan": 0.08,
    "staff_return": 0.06,
    "unlabelled": 0.55,
}

# ─── Scope offsets (troops) ──────────────────────────────────────────────────

# Cumulative totals, matching the model's cumulative-increment construction;
# "engaged" (the reference) is pinned at 0 by build_design/build_model, not
# here. Increments are 0.18, 0.24, 0.20 -- each comfortably inside
# scope_offset_prior_sd=0.4's mass, so the prior is not fighting the truth.
TRUE_DELTA_LEVEL: dict[str, float] = {
    "engaged": 0.0,
    "available": 0.18,
    "theatre_strength": 0.42,
    "on_paper": 0.62,
    "unknown": 0.05,
}

MAIN_SCOPE_WEIGHTS: dict[str, float] = {
    "engaged": 0.55,
    "available": 0.20,
    "theatre_strength": 0.12,
    "on_paper": 0.08,
    "unknown": 0.05,
}

TRUE_A_ERA: float = -0.42
ANCIENT_YEAR: int = -330
MODERN_YEAR: int = 1815
ANCIENT_CUTOFF_YEAR: int = 500

CI_MASS: float = 0.95
COMPUTED_AT = datetime(2026, 9, 23, tzinfo=UTC)


@dataclass(frozen=True)
class SimulatedCorpus:
    """A simulated troop-report corpus, plus the truth to check a fit against.

    Attributes:
        reports: The simulated reports.
        true_log_side: Each side's true ``mu_side`` (log scale), i.e. the
            corpus-level-plus-side-deviation term the model calls ``mu_side``,
            *not* including any per-report regime/scope/source nuisance
            terms -- those are things sources get wrong about a side, not the
            side's own strength.
        side_ids_by_report_count: Which side ids got 4, 2, or 1 report(s),
            for the singleton-vs-four-report width comparison.
        m0: The corpus-average log side strength truth used to build this
            corpus (the ``m0``-equivalent quantity; not always identified,
            see ``model.py``'s module docstring).
        era_mean: The report-weighted ancient fraction actually realised,
            matching what ``build_design`` will compute.
    """

    reports: list[Report]
    true_log_side: dict[int, float]
    side_ids_by_report_count: dict[int, list[int]]
    m0: float
    era_mean: float


def _draw_noise(rng: np.random.Generator, sigma: float, *, df: float = 6.0) -> float:
    """Student-t noise scaled to ``sigma``, matching the model's assumed family."""
    return float(rng.standard_t(df) * sigma)


def _observe(
    rng: np.random.Generator, eta: float, bound: str
) -> tuple[float, bool, bool]:
    """Turn a latent log value into a reported value and its bound flags.

    Args:
        rng: The generator.
        eta: The latent (noisy) log value for this report.
        bound: ``"point"``, ``"upper"`` or ``"lower"``.

    Returns:
        ``(reported_value, is_upper_bound, is_lower_bound)``. An "upper"
        report's *true* latent value lies below what is reported -- the
        censoring direction ``design.py`` documents -- via
        ``exp(latent + |N(0, 0.3)|)``; "lower" is the mirror image.
    """
    if bound == "upper":
        return math.exp(eta + abs(rng.normal(0.0, 0.3))), True, False
    if bound == "lower":
        return math.exp(eta - abs(rng.normal(0.0, 0.3))), False, True
    return math.exp(eta), False, False


def _weighted_choice(rng: np.random.Generator, weights: dict[str, float]) -> str:
    names = list(weights)
    probs = np.array([weights[n] for n in names], dtype=float)
    probs = probs / probs.sum()
    return str(rng.choice(names, p=probs))


def simulate_corpus(
    rng: np.random.Generator,
    *,
    side_groups: list[tuple[int, int]],
    m0: float,
    s0: float,
    regime_weights: dict[str, float],
    scope_weights: dict[str, float],
    ancient_fraction: float,
    censor_upper_prob: float = 0.0,
    censor_lower_prob: float = 0.0,
    b_type: dict[str, float] = TRUE_B_TYPE,
    g_regime: dict[str, float] = TRUE_G_REGIME,
    delta_level: dict[str, float] = TRUE_DELTA_LEVEL,
    a_era: float = TRUE_A_ERA,
    start_side_id: int = 1,
    start_battle_id: int = 1,
    start_report_id: int = 1,
) -> SimulatedCorpus:
    """Simulate a troop-report corpus from a known truth.

    Args:
        rng: The generator driving every random draw.
        side_groups: ``(n_sides, n_reports_per_side)`` pairs, e.g.
            ``[(90, 4), (90, 2), (120, 1)]``.
        m0: True corpus-average log side strength.
        s0: True side-to-side log-scale spread.
        regime_weights: Sampling probabilities for each claim regime, over
            :data:`TRUE_G_REGIME`'s keys (a subset is fine).
        scope_weights: Sampling probabilities for each scope, over
            :data:`TRUE_DELTA_LEVEL`'s keys (a subset is fine).
        ancient_fraction: Probability a side's battle is ancient.
        censor_upper_prob: Probability a report is an "up to X" bound.
        censor_lower_prob: Probability a report is an "at least X" bound.
        b_type: True per-source-type bias.
        g_regime: True per-regime bias (``"modern_scholarly"`` must be 0).
        delta_level: True cumulative per-scope offset (``"engaged"`` must be 0).
        a_era: True ancient-vs-later contrast, applied only to unlabelled rows.
        start_side_id: First side id to assign.
        start_battle_id: First battle id to assign (two sides per battle).
        start_report_id: First report id to assign.

    Returns:
        The simulated corpus and the truth to check a fit against.
    """
    side_id = start_side_id
    report_id = start_report_id
    true_log_side: dict[int, float] = {}
    side_ids_by_report_count: dict[int, list[int]] = {}

    # Pass 1: decide every row's side, source, regime, scope, bound and year,
    # without yet knowing era_mean (which depends on every row's year at once,
    # exactly as build_design computes it).
    pending: list[dict[str, object]] = []
    for n_sides, n_reports in side_groups:
        for _ in range(n_sides):
            z = rng.normal()
            true_log_side[side_id] = m0 + s0 * z
            side_ids_by_report_count.setdefault(n_reports, []).append(side_id)
            is_ancient = bool(rng.random() < ancient_fraction)
            year = ANCIENT_YEAR if is_ancient else MODERN_YEAR
            battle_id = start_battle_id + (side_id - start_side_id) // 2

            chosen_sources = rng.choice(
                len(SOURCES), size=n_reports, replace=n_reports > len(SOURCES)
            )
            for source_index in chosen_sources:
                key = SOURCES[int(source_index)]
                regime = _weighted_choice(rng, regime_weights)
                scope = _weighted_choice(rng, scope_weights)
                roll = rng.random()
                if roll < censor_upper_prob:
                    bound = "upper"
                elif roll < censor_upper_prob + censor_lower_prob:
                    bound = "lower"
                else:
                    bound = "point"
                pending.append(
                    {
                        "report_id": report_id,
                        "side_id": side_id,
                        "battle_id": battle_id,
                        "source_key": key,
                        "regime": regime,
                        "scope": scope,
                        "bound": bound,
                        "year": year,
                        "ancient_flag": 1.0 if is_ancient else 0.0,
                    }
                )
                report_id += 1
            side_id += 1

    era_mean = float(np.mean([row["ancient_flag"] for row in pending])) if pending else 0.0

    # Pass 2: now that era_mean is known, compute each row's eta and observe it.
    reports: list[Report] = []
    for row in pending:
        key = row["source_key"]
        assert isinstance(key, SourceKey)
        regime = str(row["regime"])
        scope = str(row["scope"])
        era_signed = float(row["ancient_flag"]) - era_mean
        era_term = a_era * era_signed if regime == "unlabelled" else 0.0
        eta = (
            true_log_side[int(row["side_id"])]
            + b_type[key.source_type]
            + true_u_source(key)
            + g_regime[regime]
            + delta_level[scope]
            + era_term
            + _draw_noise(rng, TRUE_SOURCE_SIGMA[key])
        )
        reported_value, is_upper, is_lower = _observe(rng, eta, str(row["bound"]))
        reports.append(
            Report(
                report_id=int(row["report_id"]),
                side_id=int(row["side_id"]),
                battle_id=int(row["battle_id"]),
                source_id=SOURCE_ID_OF[key],
                source_key=key,
                source_type=key.source_type,
                quantity="troops",
                reported_value=reported_value,
                scope=scope,
                is_upper_bound=is_upper,
                is_lower_bound=is_lower,
                extracted_context="",
                year_astronomical=int(row["year"]),
                claim_regime=regime,
                lineage_id=int(row["report_id"]),
                roundness=0.0,
            )
        )

    return SimulatedCorpus(
        reports=reports,
        true_log_side=true_log_side,
        side_ids_by_report_count=side_ids_by_report_count,
        m0=m0,
        era_mean=era_mean,
    )


def add_lineage_test_sides(
    rng: np.random.Generator,
    corpus: SimulatedCorpus,
    *,
    independent_side_id: int,
    repeated_side_id: int,
    start_report_id: int,
    m0: float,
    s0: float,
) -> SimulatedCorpus:
    """Append two extra sides isolating the lineage effect onto a corpus.

    Both sides get 5 reports from 5 distinct sources, regime ``unlabelled``
    and scope ``engaged`` (so nothing but the lineage assignment differs
    between them). ``independent_side_id``'s 5 reports each get their own
    lineage (the corpus default: one report, one lineage); ``repeated_side_id``'s
    5 reports all share a single lineage id, as five sources repeating one
    claim would.

    Args:
        rng: The generator.
        corpus: The corpus to extend; not mutated.
        independent_side_id: Side id for the five-independent-lineages side.
        repeated_side_id: Side id for the five-shared-lineage side.
        start_report_id: First report id to assign to the new reports.
        m0: True corpus-average log side strength, for these two sides' truth.
        s0: True side-to-side log-scale spread.

    Returns:
        A new :class:`SimulatedCorpus` with the two sides appended.
    """
    battle_id = max(r.battle_id for r in corpus.reports) + 1
    report_id = start_report_id
    true_log_side = dict(corpus.true_log_side)
    new_reports: list[Report] = []

    shared_lineage_id = -1  # Sentinel: never produced by report_id-as-lineage.

    for target_side, share_lineage in (
        (independent_side_id, False),
        (repeated_side_id, True),
    ):
        true_log_side[target_side] = m0 + s0 * rng.normal()
        copied_value: float | None = None
        for key in SOURCES[:5]:
            if share_lineage and copied_value is not None:
                # Five sources repeating one claim report the same figure.
                # Drawing fresh noise for each and merely labelling them one
                # lineage simulated five independent observations, so the
                # model was right to narrow the interval and the test could
                # not fail for the reason it names.
                reported_value = copied_value
            else:
                eta = (
                    true_log_side[target_side]
                    + TRUE_B_TYPE[key.source_type]
                    + true_u_source(key)
                    + TRUE_G_REGIME["unlabelled"]
                    + TRUE_DELTA_LEVEL["engaged"]
                    + _draw_noise(rng, TRUE_SOURCE_SIGMA[key])
                )
                reported_value, _, _ = _observe(rng, eta, "point")
                if share_lineage:
                    copied_value = reported_value
            lineage_id = shared_lineage_id if share_lineage else report_id
            new_reports.append(
                Report(
                    report_id=report_id,
                    side_id=target_side,
                    battle_id=battle_id,
                    source_id=SOURCE_ID_OF[key],
                    source_key=key,
                    source_type=key.source_type,
                    quantity="troops",
                    reported_value=reported_value,
                    scope="engaged",
                    extracted_context="",
                    year_astronomical=MODERN_YEAR,
                    claim_regime="unlabelled",
                    lineage_id=lineage_id,
                    roundness=0.0,
                )
            )
            report_id += 1

    side_ids_by_report_count = {
        k: list(v) for k, v in corpus.side_ids_by_report_count.items()
    }
    side_ids_by_report_count.setdefault(5, []).extend([independent_side_id, repeated_side_id])

    return SimulatedCorpus(
        reports=[*corpus.reports, *new_reports],
        true_log_side=true_log_side,
        side_ids_by_report_count=side_ids_by_report_count,
        m0=corpus.m0,
        era_mean=corpus.era_mean,
    )


@dataclass(frozen=True)
class Fit:
    """One completed fit: everything a test needs to check it."""

    design: Design
    idata: az.InferenceData
    summary: SummaryResult


def run_fit(
    reports: list[Report],
    *,
    quantity: str = "troops",
    priors: ModelPriors,
    inflation: InflationPrior,
    sampler: SamplerSettings,
    ancient_cutoff_year: int = ANCIENT_CUTOFF_YEAR,
) -> Fit:
    """Build, sample and summarise one fit, the same three calls the reconcile stage makes.

    Args:
        reports: The simulated (or edited) reports.
        quantity: ``"troops"`` or ``"casualties"``.
        priors: Hierarchy scales.
        inflation: The fitted claim-regime prior.
        sampler: Sampler settings; ``cores`` must stay 1 (see the project's
            Windows/no-compiler note) and ``nuts_sampler`` should stay
            ``"nutpie"``.
        ancient_cutoff_year: The era fallback's cutoff year.

    Returns:
        The design, posterior and summary.
    """
    design = build_design(reports, quantity=quantity, ancient_cutoff_year=ancient_cutoff_year)
    model = build_model(design, priors=priors, inflation=inflation)
    idata = sample(model, sampler)
    summary = summarise_fit(
        idata,
        design,
        reports,
        run_id=1,
        ci_mass=CI_MASS,
        inflation=inflation,
        priors=priors,
        computed_at=COMPUTED_AT,
    )
    return Fit(design=design, idata=idata, summary=summary)


def side_estimate_by_id(summary: SummaryResult) -> dict[int, SideEstimate]:
    """Index a summary's side estimates by side id for easy lookup."""
    return {est.side_id: est for est in summary.side_estimates}


def log_errors(
    summary: SummaryResult, true_log_side: dict[int, float], side_ids: list[int] | None = None
) -> np.ndarray:
    """``log(estimate) - true_log`` for each side, as an array.

    Args:
        summary: The fit's summary.
        true_log_side: Truth from :func:`simulate_corpus`.
        side_ids: Which sides to include, or every side in ``true_log_side``.

    Returns:
        The signed log-scale errors.
    """
    ests = side_estimate_by_id(summary)
    ids = side_ids if side_ids is not None else list(true_log_side)
    return np.array([math.log(float(ests[sid].value)) - true_log_side[sid] for sid in ids])


def coverage_fraction(
    summary: SummaryResult, true_log_side: dict[int, float], side_ids: list[int] | None = None
) -> float:
    """Fraction of sides whose true value falls inside its credible interval.

    Args:
        summary: The fit's summary.
        true_log_side: Truth from :func:`simulate_corpus`.
        side_ids: Which sides to include, or every side in ``true_log_side``.

    Returns:
        The coverage fraction, in ``[0, 1]``.
    """
    ests = side_estimate_by_id(summary)
    ids = side_ids if side_ids is not None else list(true_log_side)
    hits = 0
    for sid in ids:
        est = ests[sid]
        true_value = math.exp(true_log_side[sid])
        if float(est.lo) <= true_value <= float(est.hi):
            hits += 1
    return hits / len(ids)


def mean_log_width(summary: SummaryResult, side_ids: list[int]) -> float:
    """Mean ``log(hi / lo)`` credible-interval width across the given sides."""
    ests = side_estimate_by_id(summary)
    return float(np.mean([math.log(float(ests[sid].hi) / float(ests[sid].lo)) for sid in side_ids]))


def source_bias_for_source_id(summary: SummaryResult, source_id: int) -> SourceBias:
    """The :class:`SourceBias` row written for one ``source_id``.

    ``_source_biases`` fans one fitted bias out across every ``source_id``
    sharing a key, so every row sharing a key carries an identical
    ``bias_mu``/``bias_sd``.
    """
    for bias in summary.source_biases:
        if bias.source_id == source_id:
            return bias
    raise KeyError(f"No source bias for source_id={source_id!r}")


@pytest.fixture(scope="session")
def inflation_prior() -> InflationPrior:
    """The real fitted prior from config/inflation_priors.yaml."""
    return load_inflation_prior()


@pytest.fixture(scope="session")
def troop_priors() -> ModelPriors:
    """Default troop-model priors, matching agents/reconcile.yaml's params."""
    return ModelPriors.from_params({}, quantity="troops")
