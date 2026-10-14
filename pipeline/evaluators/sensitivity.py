"""Sensitivity analysis: a baseline refit and every declared variant."""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog

from pipeline.evaluators.compare import rank_comparison
from pipeline.evaluators.config import EvaluateParams
from pipeline.evaluators.fit import PooledResult, fit_pooled, repool
from pipeline.evaluators.target import PublishedRun, round_x_diffs
from pipeline.evaluators.variants import (
    VariantSpec,
    apply_battle_filter,
    apply_side_transform,
    collapse_three_level,
    designs_equivalent,
    observed_only_x_diff,
    prior_overrides,
)
from pipeline.modelers.config import ModelParams
from pipeline.modelers.design import build_design
from pipeline.modelers.load import ImputedRound
from pipeline.modelers.pool import rank_descending
from pipeline.modelers.records import ModelDesign, SideInput
from pipeline.modelers.summarise import PooledFit

__all__ = [
    "RankTable",
    "SensitivityInputs",
    "SensitivityResult",
    "VariantResult",
    "run_sensitivity",
]

logger = structlog.get_logger()

# Every sensitivity fit shares this seed, so a difference between a variant
# and the baseline is the variant's, not Monte Carlo's.
SENSITIVITY_SEED_OFFSET = 500_000
# The noise-floor refit repeats the baseline under this extra seed offset.
NOISE_FLOOR_SEED_OFFSET = 50_000
# Companion of observed_only: same battles, imputed x_diff.
IMPUTED_SUBSET_NAME = "observed_only_imputed_subset"
OBSERVED_ONLY_NAME = "observed_only"
STATUS_OK = "ok"
STATUS_NO_OP = "no_op"
STATUS_NOT_CONVERGED = "not_converged"
STATUS_ERROR = "error"


@dataclass(frozen=True)
class RankTable:
    """A scored ranking over a set of generals.

    Attributes:
        ids: Raw general ids.
        score: Ranking score (higher is better).
        skill_mean: Pooled posterior mean skill.
        rank: Rank by score, 1 is best.
    """

    ids: np.ndarray
    score: np.ndarray
    skill_mean: np.ndarray
    rank: np.ndarray

    @classmethod
    def of(cls, ids: np.ndarray, score: np.ndarray, skill_mean: np.ndarray) -> RankTable:
        """Build a table, ranking by ``score``."""
        return cls(ids, score, skill_mean, rank_descending(ids, score))


@dataclass(frozen=True)
class SensitivityInputs:
    """Everything the sensitivity fits start from.

    Attributes:
        design: The baseline design.
        sides: The sides the design was built from (after ``--limit``).
        rounds: Imputed rounds, in order.
        x_diffs: ``x_A - x_B`` of every round for ``design``.
        mp: Model parameters.
        params: Evaluate parameters.
        measured: ``(obs_mean, obs_sd)`` per side index.
        load_quality: Loads ``battle_id -> data_quality_score``; called only
            by a variant that needs it, inside that variant's error handling.
        published: The published run.
    """

    design: ModelDesign
    sides: Sequence[SideInput]
    rounds: Sequence[ImputedRound]
    x_diffs: Sequence[np.ndarray]
    mp: ModelParams
    params: EvaluateParams
    measured: tuple[np.ndarray, np.ndarray]
    load_quality: Callable[[], dict[int, float]]
    published: PublishedRun

    @property
    def n_rounds(self) -> int:
        """Rounds each sensitivity fit uses."""
        return self.params.sensitivity_rounds

    @property
    def seed(self) -> int:
        """The shared seed of every sensitivity fit."""
        return self.params.seed + SENSITIVITY_SEED_OFFSET


@dataclass(frozen=True)
class VariantResult:
    """One variant's outcome.

    Attributes:
        spec: The declared variant.
        status: ``ok``, ``no_op``, ``not_converged`` or ``error``.
        message: Why, for every status but ``ok`` and ``no_op``.
        counts: Design sizes and what the transform changed.
        seconds: Wall time.
        table: The variant ranking, only for ``ok``.
        vs_baseline: Rank comparisons against the baseline refit, by ``score``
            and by ``skill_mean``.
        vs_published: The same against the published run.
        vs_imputed_subset: ``observed_only`` only: the same comparisons against
            its companion refit on the imputed rounds, which isolates the
            effect of imputation from the change of sample.
    """

    spec: VariantSpec
    status: str
    message: str | None
    counts: dict[str, Any]
    seconds: float
    table: RankTable | None = None
    vs_baseline: dict[str, Any] | None = None
    vs_published: dict[str, Any] | None = None
    vs_imputed_subset: dict[str, Any] | None = None

    def summary(self) -> dict[str, Any]:
        """The JSON-ready record for the report."""
        return {
            "name": self.spec.name,
            "kind": self.spec.kind,
            "description": self.spec.description,
            "options": dict(self.spec.options),
            "status": self.status,
            "message": self.message,
            "counts": self.counts,
            "seconds": self.seconds,
            "vs_baseline": self.vs_baseline,
            "vs_published": self.vs_published,
            "vs_imputed_subset": self.vs_imputed_subset,
        }


@dataclass(frozen=True)
class SensitivityResult:
    """The baseline refit and its variants.

    Attributes:
        baseline: The baseline fit, None if it raised.
        baseline_error: Why the baseline is unusable, else None.
        baseline_table: The baseline ranking, None unless it converged.
        baseline_vs_published: Baseline refit against the published run.
        variants: One result per declared variant, in order.
        seconds: Wall time of the baseline refit.
        noise_floor: The baseline repeated under another seed (and, when the
            rounds on disk allow, a disjoint set of rounds), compared with the
            baseline. ``status`` is ``ok`` with ``score`` and ``skill_mean``
            comparisons, else ``not_converged`` or ``error`` with ``message``.
            None when the baseline itself is unusable.
        imputed_subset: The ``observed_only`` companion refit, None unless
            ``observed_only`` ran.
    """

    baseline: PooledResult | None
    baseline_error: str | None
    baseline_table: RankTable | None
    baseline_vs_published: dict[str, Any] | None
    variants: list[VariantResult]
    seconds: float
    noise_floor: dict[str, Any] | None = None
    imputed_subset: VariantResult | None = None

    def summary(self) -> dict[str, Any]:
        """The JSON-ready record for the report."""
        return {
            "baseline": {
                "converged": self.baseline is not None and self.baseline.converged,
                "worst": None if self.baseline is None else self.baseline.worst,
                "error": self.baseline_error,
                "seconds": self.seconds,
                "vs_published": self.baseline_vs_published,
            },
            "variants": [v.summary() for v in self.variants],
            IMPUTED_SUBSET_NAME: (
                None if self.imputed_subset is None else self.imputed_subset.summary()
            ),
            "noise_floor": self.noise_floor,
        }


@dataclass(frozen=True)
class _Outcome:
    """A handler result: the design fitted and its pooled summaries."""

    design: ModelDesign
    fit: PooledFit | None
    score: np.ndarray | None
    counts: dict[str, Any]
    worst: dict[str, Any] | None


def _comparisons(
    ref_ids: np.ndarray,
    ref_score: np.ndarray,
    ref_skill: np.ndarray,
    table: RankTable,
    top_n: int,
) -> dict[str, Any]:
    return {
        "score": rank_comparison(ref_ids, ref_score, table.ids, table.score, top_n=top_n),
        "skill_mean": rank_comparison(ref_ids, ref_skill, table.ids, table.skill_mean, top_n=top_n),
    }


def _counts(design: ModelDesign, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {"n_battles": design.n_battles, "n_generals": design.n_generals, **(extra or {})}


def _fitted(
    label: str,
    design: ModelDesign,
    x_diffs: Sequence[np.ndarray],
    inputs: SensitivityInputs,
    counts: Mapping[str, Any] | None = None,
    **fit_kwargs: Any,
) -> _Outcome:
    result = fit_pooled(
        design, x_diffs, inputs.mp, seed=inputs.seed, label=f"sens_{label}", **fit_kwargs
    )
    fit = result.pooled
    return _Outcome(
        design, fit, None if fit is None else fit.score, _counts(design, counts), result.worst
    )


def _refit_sides(
    spec: VariantSpec,
    sides: Sequence[SideInput],
    counts: Mapping[str, Any],
    inputs: SensitivityInputs,
) -> _Outcome | None:
    design = build_design(sides)
    if designs_equivalent(inputs.design, design):
        return None
    if design.n_battles == 0:
        raise ValueError("no battles remain after the transform")
    x_diffs = round_x_diffs(design, inputs.rounds[: inputs.n_rounds])
    return _fitted(spec.name, design, x_diffs, inputs, counts)


def _side_transform(spec: VariantSpec, inputs: SensitivityInputs) -> _Outcome | None:
    sides, counts = apply_side_transform(spec, inputs.sides)
    return _refit_sides(spec, sides, counts, inputs)


def _battle_filter(spec: VariantSpec, inputs: SensitivityInputs) -> _Outcome | None:
    if "min_quality" in spec.options:
        sides, counts = apply_battle_filter(
            spec, inputs.sides, battle_quality=inputs.load_quality()
        )
    else:
        # Measurement-only difference: imputed x_diff is conditioned on the
        # outcome, so filtering on it would select battles by their result.
        # A battle with an unmeasured side is absent from the mapping, so kept.
        by_battle = observed_only_x_diff(inputs.design, *inputs.measured)
        sides, counts = apply_battle_filter(spec, inputs.sides, measured_x_diff=by_battle)
    return _refit_sides(spec, sides, counts, inputs)


def _design_transform(spec: VariantSpec, inputs: SensitivityInputs) -> _Outcome | None:
    design = collapse_three_level(inputs.design)
    if designs_equivalent(inputs.design, design):
        return None
    return _fitted(spec.name, design, inputs.x_diffs[: inputs.n_rounds], inputs)


def _both_measured_subset(
    inputs: SensitivityInputs,
) -> tuple[ModelDesign, dict[int, float], dict[str, Any]]:
    """The design of battles with both sides measured, and their measured x_diff."""
    by_battle = observed_only_x_diff(inputs.design, *inputs.measured)
    sides = [s for s in inputs.sides if s.battle_id in by_battle]
    if not sides:
        raise ValueError("no battle has both sides measured")
    design = build_design(sides)
    if design.n_battles == 0:
        raise ValueError("no measured battle survives the design exclusion rules")
    return design, by_battle, {"battles_with_both_sides_measured": len(by_battle)}


def _x_source(spec: VariantSpec, inputs: SensitivityInputs) -> _Outcome | None:
    design, by_battle, counts = _both_measured_subset(inputs)
    x = np.array([by_battle[int(b)] for b in design.battle_ids])
    return _fitted(spec.name, design, [x], inputs, counts)


def _x_source_imputed(spec: VariantSpec, inputs: SensitivityInputs) -> _Outcome | None:
    """The observed_only battles, fitted on the imputed rounds instead."""
    design, _, counts = _both_measured_subset(inputs)
    x_diffs = round_x_diffs(design, inputs.rounds[: inputs.n_rounds])
    return _fitted(spec.name, design, x_diffs, inputs, counts)


def _priors(spec: VariantSpec, inputs: SensitivityInputs) -> _Outcome | None:
    priors = prior_overrides(inputs.mp.priors, spec.options)
    if priors == inputs.mp.priors:
        return None
    return _fitted(
        spec.name, inputs.design, inputs.x_diffs[: inputs.n_rounds], inputs, priors=priors
    )


def _post_hoc(
    spec: VariantSpec, inputs: SensitivityInputs, baseline: PooledResult
) -> _Outcome | None:
    percentile = float(spec.options.get("replacement_percentile", inputs.mp.replacement_percentile))
    realised = spec.options.get("rank_by") == "realised"
    if percentile == inputs.mp.replacement_percentile and not realised:
        return None
    fit = repool(
        inputs.design,
        baseline.rounds,
        replacement_percentile=percentile,
        rank_by=inputs.mp.rank_by,
    )
    score = fit.realised_mean if realised else fit.score
    return _Outcome(inputs.design, fit, score, _counts(inputs.design), None)


def _dispatch(
    spec: VariantSpec, inputs: SensitivityInputs, baseline: PooledResult
) -> _Outcome | None:
    if spec.kind == "side_transform":
        return _side_transform(spec, inputs)
    if spec.kind == "battle_filter":
        return _battle_filter(spec, inputs)
    if spec.kind == "design_transform":
        return _design_transform(spec, inputs)
    if spec.kind == "x_source":
        if spec.name == IMPUTED_SUBSET_NAME:
            return _x_source_imputed(spec, inputs)
        return _x_source(spec, inputs)
    if spec.kind == "priors":
        return _priors(spec, inputs)
    return _post_hoc(spec, inputs, baseline)


def _finish(
    spec: VariantSpec,
    outcome: _Outcome,
    inputs: SensitivityInputs,
    base: RankTable,
    seconds: float,
) -> VariantResult:
    if outcome.fit is None or outcome.score is None:
        return VariantResult(
            spec,
            STATUS_NOT_CONVERGED,
            f"refit did not converge: {outcome.worst}",
            outcome.counts,
            seconds,
        )
    table = RankTable.of(outcome.design.general_ids, outcome.score, outcome.fit.skill_mean)
    top_n = inputs.params.top_n
    pub = inputs.published
    return VariantResult(
        spec,
        STATUS_OK,
        None,
        outcome.counts,
        seconds,
        table,
        _comparisons(base.ids, base.score, base.skill_mean, table, top_n),
        _comparisons(pub.general_ids, pub.score, pub.skill_mean, table, top_n),
    )


def _run_variant(
    spec: VariantSpec, inputs: SensitivityInputs, baseline: PooledResult, base: RankTable
) -> VariantResult:
    start = time.perf_counter()
    try:
        outcome = _dispatch(spec, inputs, baseline)
        if outcome is None:
            return VariantResult(spec, STATUS_NO_OP, None, {}, time.perf_counter() - start)
        return _finish(spec, outcome, inputs, base, time.perf_counter() - start)
    except Exception as exc:
        # One variant raising must not take the other variants with it.
        message = f"{type(exc).__name__}: {exc}"
        logger.error("evaluate_variant_failed", variant=spec.name, error=message)
        return VariantResult(spec, STATUS_ERROR, message, {}, time.perf_counter() - start)


def _fit_baseline(inputs: SensitivityInputs) -> tuple[PooledResult | None, str | None]:
    try:
        result = fit_pooled(
            inputs.design,
            inputs.x_diffs[: inputs.n_rounds],
            inputs.mp,
            seed=inputs.seed,
            label="sens_baseline",
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.error("evaluate_baseline_failed", error=error)
        return None, error
    if not result.converged:
        return result, f"baseline refit did not converge: {result.worst}"
    return result, None


def _noise_floor(
    inputs: SensitivityInputs, baseline: PooledResult, base: RankTable
) -> dict[str, Any]:
    """Refit the baseline under another seed, on disjoint rounds if enough exist.

    Variant-versus-baseline gaps are only meaningful against this: it is how
    far two fits of the same model differ by Monte Carlo and round choice alone.
    """
    n = inputs.n_rounds
    disjoint = len(inputs.x_diffs) >= 2 * n
    x_diffs = inputs.x_diffs[n : 2 * n] if disjoint else inputs.x_diffs[:n]
    rounds_used = list(range(n, 2 * n)) if disjoint else list(range(n))
    meta = {"rounds_used": rounds_used, "disjoint_rounds": disjoint}
    try:
        result = fit_pooled(
            inputs.design,
            x_diffs,
            inputs.mp,
            seed=inputs.seed + NOISE_FLOOR_SEED_OFFSET,
            label="sens_noise_floor",
        )
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        logger.error("evaluate_noise_floor_failed", error=message)
        return {"status": STATUS_ERROR, "message": message, **meta}
    fit = result.pooled
    if not result.converged or fit is None:
        return {
            "status": STATUS_NOT_CONVERGED,
            "message": f"noise-floor refit did not converge: {result.worst}",
            **meta,
        }
    table = RankTable.of(inputs.design.general_ids, fit.score, fit.skill_mean)
    comparisons = _comparisons(base.ids, base.score, base.skill_mean, table, inputs.params.top_n)
    return {"status": STATUS_OK, "message": None, **meta, **comparisons}


def _with_imputed_subset(
    results: list[VariantResult],
    inputs: SensitivityInputs,
    baseline: PooledResult,
    base: RankTable,
) -> tuple[list[VariantResult], VariantResult | None]:
    """Run the observed_only companion and compare observed_only against it."""
    index = next(
        (
            i
            for i, r in enumerate(results)
            if r.spec.name == OBSERVED_ONLY_NAME and r.status == STATUS_OK
        ),
        None,
    )
    if index is None:
        return results, None
    observed = results[index]
    spec = dataclasses.replace(
        observed.spec,
        name=IMPUTED_SUBSET_NAME,
        description="The observed_only battles fitted on the imputed rounds' x_diff",
    )
    companion = _run_variant(spec, inputs, baseline, base)
    if companion.table is None or observed.table is None:
        return results, companion
    versus = _comparisons(
        companion.table.ids,
        companion.table.score,
        companion.table.skill_mean,
        observed.table,
        inputs.params.top_n,
    )
    updated = list(results)
    updated[index] = dataclasses.replace(observed, vs_imputed_subset=versus)
    return updated, companion


def run_sensitivity(
    inputs: SensitivityInputs, variants: Sequence[VariantSpec]
) -> SensitivityResult:
    """Refit the baseline, a noise-floor repeat and every variant, then compare rankings.

    Args:
        inputs: The shared inputs.
        variants: Parsed variants.

    Returns:
        The baseline and one result per variant. A variant that raises is
        reported as ``error``; a baseline that fails to converge makes every
        variant ``not_converged`` without refitting it. When ``observed_only``
        is ok, its companion ``observed_only_imputed_subset`` is refitted and
        ``observed_only`` is also compared against it.
    """
    start = time.perf_counter()
    baseline, error = _fit_baseline(inputs)
    seconds = time.perf_counter() - start
    fit = None if baseline is None else baseline.pooled
    if baseline is None or fit is None:
        skipped = [VariantResult(v, STATUS_NOT_CONVERGED, error, {}, 0.0) for v in variants]
        return SensitivityResult(baseline, error, None, None, skipped, seconds)
    base = RankTable.of(inputs.design.general_ids, fit.score, fit.skill_mean)
    pub = inputs.published
    vs_pub = _comparisons(pub.general_ids, pub.score, pub.skill_mean, base, inputs.params.top_n)
    results = [_run_variant(v, inputs, baseline, base) for v in variants]
    results, companion = _with_imputed_subset(results, inputs, baseline, base)
    floor = _noise_floor(inputs, baseline, base)
    return SensitivityResult(baseline, None, base, vs_pub, results, seconds, floor, companion)
