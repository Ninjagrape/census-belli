"""Sampling one model with the write rule's single retry (design R7).

A fit that misses the write thresholds is re-run once on the same data with a
new seed and a higher ``target_accept``. It is never dropped: a failed round
that vanished from the pool would select rounds by how well the sampler liked
them, which is not random.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Final

import structlog

from pipeline.imputers.summarise import is_converged
from pipeline.modelers.config import ModelParams
from pipeline.modelers.pool import round_convergence
from pipeline.reconcilers.model import sample

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    import arviz as az
    import pymc as pm

__all__ = ["RETRY_SEED_OFFSET", "FitOutcome", "fit_with_retry"]

logger = structlog.get_logger()

# Far larger than any round index, so a retry never reuses another round's seed.
RETRY_SEED_OFFSET: Final[int] = 100_000


@dataclass(frozen=True)
class FitOutcome:
    """The accepted (or last) attempt of one fit.

    Attributes:
        idata: The attempt's posterior.
        worst: ``{max_rhat, min_ess_bulk, divergences}`` of that attempt.
        converged: Whether it met the write thresholds.
        attempts: One summary dict per attempt, in order.
    """

    idata: az.InferenceData
    worst: dict[str, Any]
    converged: bool
    attempts: list[dict[str, Any]]


def fit_with_retry(model: pm.Model, params: ModelParams, *, seed: int, label: str) -> FitOutcome:
    """Sample, and retry once if the write thresholds are missed.

    Args:
        model: A built model.
        params: Run parameters (sampler settings and write thresholds).
        seed: Seed of the first attempt.
        label: A name for the log lines (``round_3``, ``joint``).

    Returns:
        The first attempt that converged, else the retry.
    """
    attempts: list[dict[str, Any]] = []
    outcome: FitOutcome | None = None
    for attempt in (0, 1):
        settings = replace(
            params.sampler,
            random_seed=seed + attempt * RETRY_SEED_OFFSET,
            target_accept=params.sampler.target_accept
            if attempt == 0
            else params.retry_target_accept,
        )
        try:
            idata = sample(model, settings)
        except Exception:
            logger.error("model_sampling_failed", fit=label, attempt=attempt)
            raise
        worst = round_convergence(idata)
        converged = is_converged(
            worst,
            max_rhat=params.max_rhat,
            min_ess_bulk=params.min_ess_bulk,
            max_divergences=params.max_divergences,
        )
        attempts.append(
            {
                "attempt": attempt,
                "seed": settings.random_seed,
                "target_accept": settings.target_accept,
                "worst": worst,
                "converged": converged,
            }
        )
        outcome = FitOutcome(idata, worst, converged, attempts)
        if converged:
            break
        logger.warning("model_fit_missed_write_thresholds", fit=label, attempt=attempt, worst=worst)
    assert outcome is not None
    return outcome
