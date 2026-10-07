"""Summaries of a fitted imputation model and of its rounds.

Convergence is rolled up the same way the reconcile stage does it (worst rhat,
worst bulk ESS, summed divergences, under a ``worst`` key), because that is the
object ``pipeline.quality``'s ``diagnostics_json`` handler reads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from pipeline.imputers.conditional import RoundSet
from pipeline.imputers.design import ImputeDesign

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    import arviz as az

__all__ = [
    "ImputedSide",
    "build_diagnostics",
    "convergence",
    "is_converged",
    "summarise_imputed",
]

_LOWER_Q = 2.5
_UPPER_Q = 97.5


@dataclass(frozen=True)
class ImputedSide:
    """Across-round summary of one imputed side, on the exp scale."""

    side_id: int
    battle_id: int
    value: float
    lo: float
    hi: float


def _finite(value: float) -> float | None:
    """Replace a non-finite float with None so the payload stays valid JSON."""
    return value if math.isfinite(value) else None


def convergence(idata: az.InferenceData) -> dict[str, Any]:
    """Roll up rhat, bulk ESS and divergences across every posterior variable.

    Args:
        idata: The fit's posterior and sample stats.

    Returns:
        ``{"max_rhat", "min_ess_bulk", "divergences"}``, with None for a
        metric with no finite value (a single-chain fit has no rhat).
    """
    import arviz as az  # noqa: PLC0415 - deliberately deferred, as in reconcile

    max_rhat = -math.inf
    for data_array in az.rhat(idata).data_vars.values():
        values = np.asarray(data_array.values, dtype=float).ravel()
        values = values[~np.isnan(values)]
        if values.size:
            max_rhat = max(max_rhat, float(values.max()))

    min_ess = math.inf
    for data_array in az.ess(idata, method="bulk").data_vars.values():
        values = np.asarray(data_array.values, dtype=float).ravel()
        values = values[~np.isnan(values)]
        if values.size:
            min_ess = min(min_ess, float(values.min()))

    divergences = 0
    if "sample_stats" in idata.groups() and "diverging" in idata.sample_stats:
        divergences = int(np.asarray(idata.sample_stats["diverging"].values).sum())

    return {
        "max_rhat": _finite(max_rhat),
        "min_ess_bulk": _finite(min_ess),
        "divergences": divergences,
    }


def is_converged(
    worst: dict[str, Any], *, max_rhat: float, min_ess_bulk: float, max_divergences: int
) -> bool:
    """Whether a fit meets the write thresholds.

    A metric that is None (no finite value) fails: unknown convergence is not
    convergence.

    Args:
        worst: The dict from :func:`convergence`.
        max_rhat: Largest acceptable rhat.
        min_ess_bulk: Smallest acceptable bulk ESS.
        max_divergences: Most divergences tolerated.

    Returns:
        True if the fit may be written.
    """
    rhat, ess, div = worst.get("max_rhat"), worst.get("min_ess_bulk"), worst.get("divergences")
    if rhat is None or ess is None or div is None:
        return False
    return bool(rhat <= max_rhat and ess >= min_ess_bulk and div <= max_divergences)


def summarise_imputed(rounds: RoundSet, design: ImputeDesign) -> list[ImputedSide]:
    """Median and 95% interval across rounds for every imputed side.

    Args:
        rounds: The rounds, MNAR delta already applied.
        design: The design.

    Returns:
        One entry per imputed side, on the exp scale.
    """
    out: list[ImputedSide] = []
    for row in np.flatnonzero(rounds.is_imputed):
        column = rounds.values[:, row]
        lo, median, hi = np.percentile(column, [_LOWER_Q, 50.0, _UPPER_Q])
        out.append(
            ImputedSide(
                side_id=int(design.side_ids[row]),
                battle_id=int(design.battle_ids[row]),
                value=float(np.exp(median)),
                lo=float(np.exp(lo)),
                hi=float(np.exp(hi)),
            )
        )
    return out


def build_diagnostics(
    worst: dict[str, Any],
    idata: az.InferenceData,
    *,
    scale: str,
    design: ImputeDesign,
    n_rounds: int,
    reconcile_run_id: int | None,
    written: bool,
) -> dict[str, Any]:
    """Assemble the ``model_runs.diagnostics`` payload.

    Args:
        worst: The dict from :func:`convergence`.
        idata: The posterior, for hyperparameter summaries.
        scale: ``'neutral'`` or ``'published'``.
        design: The design.
        n_rounds: Rounds drawn.
        reconcile_run_id: The reconcile run the likelihood came from.
        written: Whether the fit passed the write thresholds.

    Returns:
        A JSON-serialisable payload with a top-level ``worst`` object.
    """
    hyper: dict[str, float | None] = {}
    for name in ("tau_b", "sigma", "tau_p", "tau_c"):
        if name in idata.posterior:
            hyper[f"{name}_mean"] = _finite(float(idata.posterior[name].mean()))
    return {
        "worst": worst,
        "scale": scale,
        "written": written,
        "n_rounds": n_rounds,
        "n_sides": design.n_sides,
        "n_observed": design.n_observed,
        "reconcile_run_id": reconcile_run_id,
        "hyper": hyper,
    }
