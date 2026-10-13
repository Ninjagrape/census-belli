"""The regime-neutral troops fit's outputs: estimates, diagnostics and draws.

The neutral fit is :func:`pipeline.reconcilers.model.build_model` with
``neutral=True`` on the same design as the published troops fit. This module
turns its posterior into the three things the rest of the pipeline reads:

* ``battle_sides.est_troops_neutral*``, via :class:`NeutralSideEstimate`;
* the ``troops_neutral`` block of ``model_runs.diagnostics``;
* ``neutral_draws_run{run_id}.npz``, the joint posterior draws of ``mu_side``
  that the impute stage conditions on. Draws rather than marginal summaries,
  because two sides of one battle are correlated through the shared
  hyperparameters and a ratio needs that covariance.

Nothing here samples. Convergence thresholds are applied by the stage, as for
the published fit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import structlog

from pipeline.reconcilers.design import Design
from pipeline.reconcilers.records import NeutralSideEstimate
from pipeline.reconcilers.summarise import (
    DEFAULT_MAX_MEDIAN_MCSE_LOG,
    DEFAULT_MAX_QUANTILE_MCSE_FRAC,
    _convergence,
    _coord_labels,
    _finite,
    _pooled,
    _scalar_stats,
    widen_bounds,
    widening_margins,
)

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    import arviz as az

__all__ = [
    "MAX_NEUTRAL_DRAWS",
    "NEUTRAL_SUBDIR",
    "NeutralSummary",
    "neutral_draws_path",
    "summarise_neutral",
    "write_neutral_draws",
]

logger = structlog.get_logger()

MAX_NEUTRAL_DRAWS: Final[int] = 1000
NEUTRAL_SUBDIR: Final[str] = "reconcile"


@dataclass(frozen=True)
class NeutralSummary:
    """What the neutral fit produced, before any thresholds are applied.

    Attributes:
        estimates: One per side in the design.
        diagnostics: JSON-serialisable block for ``diagnostics["troops_neutral"]``.
        side_ids: Side ids in the column order of ``draws``.
        battle_ids: Each side's battle id, aligned with ``side_ids``.
        draws: Thinned pooled draws of ``mu_side``, shape ``(n_draws, n_sides)``.
        m0_mean: Posterior mean of ``m0``.
        s0_mean: Posterior mean of ``s0``.
    """

    estimates: list[NeutralSideEstimate]
    diagnostics: dict[str, Any]
    side_ids: tuple[int, ...]
    battle_ids: tuple[int, ...]
    draws: np.ndarray
    m0_mean: float
    s0_mean: float


def _thin(draws: np.ndarray, limit: int) -> np.ndarray:
    """Thin pooled draws to at most ``limit`` rows, evenly across the pool.

    The pool is chain-major, so even spacing takes draws from every chain.

    Args:
        draws: Array of shape ``(n_pooled, n_sides)``.
        limit: Maximum number of rows to keep.

    Returns:
        The thinned array, a copy.
    """
    n = draws.shape[0]
    if n <= limit:
        return np.array(draws, dtype=np.float64)
    keep = np.linspace(0, n - 1, limit).round().astype(int)
    return np.array(draws[keep], dtype=np.float64)


def summarise_neutral(
    idata: az.InferenceData,
    design: Design,
    battle_of_side: dict[int, int],
    *,
    run_id: int,
    ci_mass: float,
    max_quantile_mcse_frac: float = DEFAULT_MAX_QUANTILE_MCSE_FRAC,
    max_median_mcse_log: float = DEFAULT_MAX_MEDIAN_MCSE_LOG,
) -> NeutralSummary:
    """Summarise the neutral troops fit.

    Args:
        idata: Posterior from the neutral fit.
        design: The design the model was built from.
        battle_of_side: ``side_id`` to ``battle_id`` for every side in the design.
        run_id: The ``model_runs.run_id`` the fit is recorded under.
        ci_mass: Total mass of the credible interval.
        max_quantile_mcse_frac: The write gate's limit on a side's quantile
            MCSE as a share of its interval width.
        max_median_mcse_log: The write gate's absolute limit on the MCSE of a
            side's median, in log units.

    Returns:
        The estimates, diagnostics block and draws.
    """
    mu = _pooled(idata, "mu_side")
    labels = _coord_labels(idata, "side")
    mu = mu[:, [labels.index(str(side_id)) for side_id in design.side_ids]]

    worst, worst_variable, _ = _convergence(
        idata,
        ci_mass=ci_mass,
        max_quantile_mcse_frac=max_quantile_mcse_frac,
        max_median_mcse_log=max_median_mcse_log,
    )
    margins = widening_margins(idata, ci_mass, worst["mcse_failed_side_ids"])

    lower_q = (1.0 - ci_mass) / 2.0
    estimates: list[NeutralSideEstimate] = []
    for i, side_id in enumerate(design.side_ids):
        lo = math.exp(float(np.quantile(mu[:, i], lower_q)))
        hi = math.exp(float(np.quantile(mu[:, i], 1.0 - lower_q)))
        if side_id in margins:
            lo, hi = widen_bounds(lo, hi, margins[side_id])
        estimates.append(
            NeutralSideEstimate(
                side_id=side_id,
                value=math.exp(float(np.median(mu[:, i]))),
                lo=lo,
                hi=hi,
                log_mean=float(np.mean(mu[:, i])),
                log_sd=float(np.std(mu[:, i])),
                run_id=run_id,
            )
        )
    m0 = _scalar_stats(idata, "m0")
    s0 = _scalar_stats(idata, "s0")
    diagnostics: dict[str, Any] = {
        "quantity": design.quantity,
        "n_obs": design.n_obs,
        "n_sides": design.n_sides,
        "n_source_keys_own_effect": design.n_source_keys_own_effect,
        "n_source_keys_pooled": design.n_source_keys_pooled,
        "era_mean": design.era_mean,
        "era_centre_rows": design.era_centre_rows,
        "worst": worst,
        "worst_variable": worst_variable,
        "n_sides_mcse_fail": worst["n_sides_mcse_fail"],
        "n_sides_low_precision": worst["n_sides_mcse_fail"],
        "mcse_failed_side_ids": worst["mcse_failed_side_ids"],
        "mcse_failed_limits": worst["mcse_failed_limits"],
        "worst_side": worst["worst_side"],
        "m0": m0,
        "s0": s0,
    }
    return NeutralSummary(
        estimates=estimates,
        diagnostics=diagnostics,
        side_ids=design.side_ids,
        battle_ids=tuple(battle_of_side[s] for s in design.side_ids),
        draws=_thin(mu, MAX_NEUTRAL_DRAWS),
        m0_mean=_finite(float(np.mean(_pooled(idata, "m0")))) or 0.0,
        s0_mean=_finite(float(np.mean(_pooled(idata, "s0")))) or 0.0,
    )


def neutral_draws_path(processed_root: Path, run_id: int) -> Path:
    """Where a run's neutral draws file lives.

    Args:
        processed_root: The stage's ``processed_root``.
        run_id: The ``model_runs.run_id``.

    Returns:
        ``<processed_root>/reconcile/neutral_draws_run{run_id}.npz``.
    """
    return processed_root / NEUTRAL_SUBDIR / f"neutral_draws_run{run_id}.npz"


def write_neutral_draws(
    summary: NeutralSummary,
    *,
    processed_root: Path,
    run_id: int,
    database: str,
    started_at: str,
) -> Path:
    """Write the draws file the impute stage reads, one column per side.

    Written to a temporary name and renamed, so a reader never sees a partial
    file.

    Args:
        summary: The neutral fit's summary.
        processed_root: The stage's ``processed_root``.
        run_id: The ``model_runs.run_id``.
        database: ``current_database()`` of the run, stored so impute can refuse
            a file written by another database that reached the same run id.
        started_at: That run's ``started_at`` as an ISO string, stored for the
            same check.

    Returns:
        The path written.
    """
    path = neutral_draws_path(processed_root, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(
        tmp,
        run_id=np.int64(run_id),
        database=np.str_(database),
        started_at=np.str_(started_at),
        side_id=np.asarray(summary.side_ids, dtype=np.int64),
        battle_id=np.asarray(summary.battle_ids, dtype=np.int64),
        draws=summary.draws,
        m0_mean=np.float64(summary.m0_mean),
        s0_mean=np.float64(summary.s0_mean),
    )
    tmp.replace(path)
    logger.info(
        "reconcile_neutral_draws_written",
        path=str(path),
        n_draws=int(summary.draws.shape[0]),
        n_sides=len(summary.side_ids),
    )
    return path
