"""Held-out cross-validation: fold fits, prediction and scoring."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog

from pipeline.evaluators.config import EvaluateParams
from pipeline.evaluators.fit import PooledResult, fit_joint, fit_pooled
from pipeline.evaluators.folds import assign_folds, train_weight
from pipeline.evaluators.predict import (
    binary_metrics,
    force_only_baseline,
    heldout_x_diff_draws,
    log_score,
    predict_category_probs,
    ranked_probability_score,
)
from pipeline.modelers.config import ModelParams
from pipeline.modelers.ordinal import N_LEVELS
from pipeline.modelers.records import ModelDesign

__all__ = ["CVArm", "CVResult", "FoldRecord", "run_cross_validation", "score_predictions"]

logger = structlog.get_logger()

# Fold fits use seeds seed + FOLD_SEED_STRIDE * (fold + 1) + round, so folds
# never share a round seed with each other or with the sensitivity fits.
FOLD_SEED_STRIDE = 1000
# Offset of the held-out x_diff generator from the base seed.
PREDICTION_SEED_OFFSET = 7_000
# Joint-arm fold fits use seed + JOINT_SEED_OFFSET + fold: far from the imputed
# arm's seeds and from the 100_000 retry offset, so no two fits share a stream.
JOINT_SEED_OFFSET = 2_000_000


@dataclass(frozen=True)
class FoldRecord:
    """One fold's fit status.

    Attributes:
        fold: Fold index.
        n_held_out: Battles held out.
        converged: Whether every round met the write thresholds.
        worst: Worst-case convergence numbers, None if the fit raised.
        error: Exception text if the fit raised, else None.
        seconds: Wall time of the fit and prediction.
    """

    fold: int
    n_held_out: int
    converged: bool
    worst: dict[str, Any] | None
    error: str | None
    seconds: float


@dataclass(frozen=True)
class CVArm:
    """One cross-validation arm: fold records, predictions and scores.

    Attributes:
        folds: One record per fold.
        probs: ``(n_battles, 5)`` predicted level probabilities; NaN rows for
            battles of unconverged folds.
        predicted: Whether a battle has predictions.
        metrics: Scores over the predicted battles, including
            ``on_force_baseline_mask``: the model's binary scores on exactly the
            battles the ``force_only`` baseline covers.
        baselines: ``chance`` and ``force_only`` scores on the same battles.
    """

    folds: list[FoldRecord]
    probs: np.ndarray
    predicted: np.ndarray
    metrics: dict[str, Any]
    baselines: dict[str, Any]


@dataclass(frozen=True)
class CVResult:
    """Cross-validation output.

    The top-level ``folds``, ``probs``, ``predicted``, ``metrics`` and
    ``baselines`` are the imputed arm (trained on imputed ``x_diff``,
    predicting from the measurement-only prior): the one the gates use and the
    one the published ranking resembles. ``joint`` is the joint latent-troops
    arm.

    Attributes:
        folds: Imputed-arm fold records.
        fold_of: Fold id per battle.
        probs: Imputed-arm probabilities, shape ``(n_battles, 5)``.
        predicted: Imputed-arm prediction mask.
        metrics: Imputed-arm scores.
        baselines: Imputed-arm ``chance`` and ``force_only`` scores.
        joint: The joint-latent arm.
    """

    folds: list[FoldRecord]
    fold_of: np.ndarray
    probs: np.ndarray
    predicted: np.ndarray
    metrics: dict[str, Any]
    baselines: dict[str, Any]
    joint: CVArm


def _predict_fold(
    design: ModelDesign,
    result: PooledResult,
    held_out: np.ndarray,
    obs_mean: np.ndarray,
    obs_sd: np.ndarray,
    sigma_w: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Round-averaged probabilities for the held-out battles.

    The held-out battles' imputed ``x_diff`` is never read: impute conditions
    on the outcome, so it would leak the result.
    """
    per_round: list[np.ndarray] = []
    for rnd in result.rounds:
        if rnd.draws is None:
            raise ValueError("converged fold has a round without draws")
        x = heldout_x_diff_draws(design, obs_mean, obs_sd, sigma_w, rnd.draws.n_draws, rng)
        per_round.append(predict_category_probs(design, rnd.draws, x, held_out))
    return np.asarray(np.mean(per_round, axis=0))


def _run_fold(
    design: ModelDesign,
    x_diffs: Sequence[np.ndarray],
    folds: np.ndarray,
    fold: int,
    mp: ModelParams,
    seed: int,
    measured: tuple[np.ndarray, np.ndarray],
) -> tuple[FoldRecord, np.ndarray | None]:
    held_out = np.flatnonzero(folds == fold)
    start = time.perf_counter()
    probs: np.ndarray | None = None
    worst: dict[str, Any] | None = None
    error: str | None = None
    converged = False
    try:
        result = fit_pooled(
            design,
            x_diffs,
            mp,
            seed=seed + FOLD_SEED_STRIDE * (fold + 1),
            label=f"cv_fold_{fold}",
            loglik_weight=train_weight(folds, fold),
        )
        worst, converged = result.worst, result.converged
        if converged:
            rng = np.random.default_rng(seed + PREDICTION_SEED_OFFSET + fold)
            probs = _predict_fold(design, result, held_out, *measured, mp.joint_sigma_w, rng)
    except Exception as exc:
        # One fold's failure must not discard the other folds' predictions.
        error = f"{type(exc).__name__}: {exc}"
        logger.error("evaluate_cv_fold_failed", fold=fold, error=error)
    record = FoldRecord(
        fold, int(held_out.size), converged, worst, error, time.perf_counter() - start
    )
    return record, probs


def _run_joint_fold(
    design: ModelDesign,
    folds: np.ndarray,
    fold: int,
    mp: ModelParams,
    seed: int,
    measured: tuple[np.ndarray, np.ndarray],
) -> tuple[FoldRecord, np.ndarray | None]:
    """One joint-latent fold fit, predicting from its own ``x_diff_latent`` draws.

    With weight 0 a held-out battle's outcome never touches its latent
    difference, so its draws are the measurement prior propagated through the
    fit.
    """
    held_out = np.flatnonzero(folds == fold)
    start = time.perf_counter()
    probs: np.ndarray | None = None
    worst: dict[str, Any] | None = None
    error: str | None = None
    converged = False
    try:
        result = fit_joint(
            design,
            *measured,
            mp,
            seed=seed + JOINT_SEED_OFFSET + fold,
            label=f"cv_joint_fold_{fold}",
            loglik_weight=train_weight(folds, fold),
        )
        worst, converged = result.worst, result.converged
        if converged:
            if result.draws is None or result.draws.x_diff is None:
                raise ValueError("converged joint fit has no x_diff_latent draws")
            probs = predict_category_probs(design, result.draws, result.draws.x_diff, held_out)
    except Exception as exc:
        # The joint arm is diagnostic: its failure is recorded, never fatal.
        error = f"{type(exc).__name__}: {exc}"
        logger.error("evaluate_cv_joint_fold_failed", fold=fold, error=error)
    record = FoldRecord(
        fold, int(held_out.size), converged, worst, error, time.perf_counter() - start
    )
    return record, probs


def score_predictions(
    design: ModelDesign,
    probs: np.ndarray,
    predicted: np.ndarray,
    x_measured: np.ndarray,
    n_bins: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Score predictions and the chance and force-only baselines.

    Args:
        design: The design.
        probs: ``(n_battles, 5)`` probabilities, rows used where ``predicted``.
        predicted: Boolean mask of predicted battles.
        x_measured: Measured ``x_A - x_B`` per battle, NaN if unmeasured.
        n_bins: Calibration bins.

    Returns:
        ``(metrics, baselines)``; baselines cover the same battles.
        ``metrics["on_force_baseline_mask"]`` scores the model on the battles
        with both sides measured, the set ``force_only`` covers.
    """
    k_lo, k_hi = design.k_lo[predicted], design.k_hi[predicted]
    sub = probs[predicted]
    exact = k_lo == k_hi
    uniform = np.full_like(sub, 1.0 / N_LEVELS)
    metrics = {
        "n_battles": design.n_battles,
        "n_predicted": int(predicted.sum()),
        "n_exact": int(exact.sum()),
        "binary": binary_metrics(sub, k_lo, k_hi, n_bins),
        "ranked_probability_score": ranked_probability_score(sub[exact], k_lo[exact]),
        "log_score": log_score(sub, k_lo, k_hi),
    }
    chance_binary = binary_metrics(uniform, k_lo, k_hi, n_bins)
    chance = {
        "accuracy": 0.5,
        "brier": 0.25,
        "n": chance_binary["n"],
        "ranked_probability_score": ranked_probability_score(uniform[exact], k_lo[exact]),
        "log_score": log_score(uniform, k_lo, k_hi),
    }
    masked = np.where(predicted, x_measured, np.nan)
    # Like for like with force_only: the same battles, scored for the model.
    on_mask = np.isfinite(masked)
    metrics["on_force_baseline_mask"] = binary_metrics(
        probs[on_mask], design.k_lo[on_mask], design.k_hi[on_mask], n_bins
    )
    baselines = {
        "chance": chance,
        "force_only": force_only_baseline(masked, design.k_lo, design.k_hi),
    }
    return metrics, baselines


def _score_arm(
    design: ModelDesign,
    records: list[FoldRecord],
    probs: np.ndarray,
    x_measured: np.ndarray,
    n_bins: int,
) -> CVArm:
    predicted = np.isfinite(probs).all(axis=1)
    metrics, baselines = score_predictions(design, probs, predicted, x_measured, n_bins)
    return CVArm(records, probs, predicted, metrics, baselines)


def run_cross_validation(
    design: ModelDesign,
    x_diffs: Sequence[np.ndarray],
    mp: ModelParams,
    params: EvaluateParams,
    measured: tuple[np.ndarray, np.ndarray],
) -> CVResult:
    """Cross-validate the model by battle, in two arms.

    The imputed arm trains on imputed ``x_diff`` rounds and predicts from the
    measurement-only prior. Imputation conditions on the outcome, so that arm
    mixes model quality with a circularity penalty (train and test ``x_diff``
    come from different generators). The joint arm trains and predicts inside
    one joint latent-troops model, so both come from the same generator; the
    two together separate the penalty from the model.

    Args:
        design: The design.
        x_diffs: Imputed ``x_A - x_B`` rounds; the first ``cv_rounds`` train.
        mp: Model parameters for the fits.
        params: Evaluate parameters.
        measured: ``(obs_mean, obs_sd)`` per side index.

    Returns:
        Both arms. A fold that did not converge has no predictions and its
        battles are left out of that arm's scores. The joint arm not
        converging is recorded and does not fail the stage.
    """
    folds = assign_folds(design.n_battles, params.held_out_folds, params.seed)
    shape = (design.n_battles, N_LEVELS)
    probs = np.full(shape, np.nan)
    joint_probs = np.full(shape, np.nan)
    records: list[FoldRecord] = []
    joint_records: list[FoldRecord] = []
    train_x = list(x_diffs)[: params.cv_rounds]
    for fold in range(params.held_out_folds):
        held = folds == fold
        record, fold_probs = _run_fold(design, train_x, folds, fold, mp, params.seed, measured)
        records.append(record)
        if fold_probs is not None:
            probs[held] = fold_probs
        joint_record, joint_fold_probs = _run_joint_fold(
            design, folds, fold, mp, params.seed, measured
        )
        joint_records.append(joint_record)
        if joint_fold_probs is not None:
            joint_probs[held] = joint_fold_probs
    obs_mean = measured[0]
    x_measured = obs_mean[design.side_a] - obs_mean[design.side_b]
    bins = params.calibration_bins
    imputed = _score_arm(design, records, probs, x_measured, bins)
    joint = _score_arm(design, joint_records, joint_probs, x_measured, bins)
    return CVResult(
        imputed.folds,
        folds,
        imputed.probs,
        imputed.predicted,
        imputed.metrics,
        imputed.baselines,
        joint,
    )
