"""The regime-neutral troops fit's outputs: estimates, diagnostics and draws.

The neutral fit is :func:`pipeline.reconcilers.model.build_model` with
``neutral=True`` on the same design as the published troops fit. This module
turns its posterior into the three things the rest of the pipeline reads:

* ``battle_sides.est_troops_neutral*``, via :class:`NeutralSideEstimate`;
* the ``troops_neutral`` block of ``model_runs.diagnostics``;
* ``neutral_draws_run{run_id}.npz``, the joint posterior draws of ``mu_side``
  that the impute stage conditions on. Draws rather than marginal summaries,
  because two sides of one battle are correlated through the shared
  hyperparameters and a ratio needs that covariance. The file also carries,
  row-aligned with those draws, the shared regime parameters the sides load on
  (``g_unlabelled``, ``a_era``, ``offset_bar``, ``m0_draws``) and, per side, the era value and
  unlabelled-row share the design applied, so a reader can tell which sides
  depend on them;
* the per-side likelihood the impute stage uses (``lik_*`` keys; see
  :mod:`pipeline.reconcilers.likelihood_export`): mean and sd from the side's
  point rows at the fitted noise, and loadings on the shared parameters.

Nothing here samples. Convergence thresholds are applied by the stage, as for
the published fit.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import structlog

from pipeline.reconcilers.design import Design
from pipeline.reconcilers.likelihood_export import (
    LOADING_NAMES,
    LikelihoodExport,
    likelihood_export,
)
from pipeline.reconcilers.records import NeutralSideEstimate
from pipeline.reconcilers.recovery import RecoverySettings
from pipeline.reconcilers.summarise import (
    DEFAULT_MAX_MEDIAN_MCSE_LOG,
    DEFAULT_MAX_QUANTILE_MCSE_FRAC,
    _convergence,
    _coord_labels,
    _finite,
    _pooled,
    _scalar_stats,
    recovery_block,
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
        g_unlabelled: The unlabelled regime's ``g_regime`` draws, shape
            ``(n_draws,)``, row-aligned with ``draws``; zeros when the design
            has no unlabelled regime. None if the posterior lacks ``g_regime``.
        a_era: ``a_era`` draws, shape ``(n_draws,)``, row-aligned.
        offset_bar: ``offset_bar`` draws, shape ``(n_draws,)``, row-aligned.
        side_era: Per side, the mean era covariate over its unlabelled rows (the
            value ``a_era`` multiplies); 0 for a side with none. Shape
            ``(n_sides,)``, in ``side_ids`` order.
        side_unlabelled_share: Per side, the share of its design rows whose
            regime is unlabelled. Shape ``(n_sides,)``.
        m0_draws: ``m0`` draws, shape ``(n_draws,)``, row-aligned. ``m0`` is
            ``corpus_level - offset_bar``, so it falls with ``g_unlabelled``.
        likelihood: Each side's likelihood from its point rows, with loadings on
            the shared parameters; None when the posterior lacks the pieces.
    """

    estimates: list[NeutralSideEstimate]
    diagnostics: dict[str, Any]
    side_ids: tuple[int, ...]
    battle_ids: tuple[int, ...]
    draws: np.ndarray
    m0_mean: float
    s0_mean: float
    g_unlabelled: np.ndarray | None = None
    a_era: np.ndarray | None = None
    offset_bar: np.ndarray | None = None
    side_era: np.ndarray | None = None
    side_unlabelled_share: np.ndarray | None = None
    m0_draws: np.ndarray | None = None
    likelihood: LikelihoodExport | None = None


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


def _side_regime_loading(design: Design) -> tuple[np.ndarray, np.ndarray]:
    """Per side, the era value and unlabelled-row share the design applied.

    The model adds ``a_era * era`` on unlabelled rows only and ``g_unlabelled``
    on the same rows, so these two numbers say how strongly a side's reports
    load on the shared regime parameters.

    Args:
        design: The design the neutral fit was built from.

    Returns:
        ``(side_era, side_unlabelled_share)``, each of shape ``(n_sides,)``, in
        ``design.side_ids`` order.
    """
    era = np.zeros(design.n_sides, dtype=np.float64)
    share = np.zeros(design.n_sides, dtype=np.float64)
    if "unlabelled" not in design.regimes:
        return era, share
    unlabelled = design.regime_index == design.regimes.index("unlabelled")
    for position in range(design.n_sides):
        on_side = design.side_index == position
        n_rows = int(on_side.sum())
        rows = on_side & unlabelled
        n_unlabelled = int(rows.sum())
        if n_rows:
            share[position] = n_unlabelled / n_rows
        if n_unlabelled:
            era[position] = float(np.mean(design.era[rows]))
    return era, share


def _shared_draws(
    idata: az.InferenceData, design: Design
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Pooled draws of the shared regime parameters, or None if absent.

    Args:
        idata: Posterior from the neutral fit.
        design: The design, for the regime vocabulary.

    Returns:
        ``(g_unlabelled, a_era, offset_bar, m0)``, each shape ``(n_pooled,)``, in
        the same pooled order as ``mu_side``.
    """
    names = set(idata.posterior.data_vars)
    if not {"g_regime", "a_era", "offset_bar", "m0"} <= names:
        return None
    g_regime = _pooled(idata, "g_regime")
    if "unlabelled" in design.regimes:
        labels = _coord_labels(idata, "regime")
        g_unlabelled = g_regime[:, labels.index("unlabelled")]
    else:
        g_unlabelled = np.zeros(g_regime.shape[0])
    return (
        g_unlabelled,
        _pooled(idata, "a_era"),
        _pooled(idata, "offset_bar"),
        _pooled(idata, "m0"),
    )


def summarise_neutral(
    idata: az.InferenceData,
    design: Design,
    battle_of_side: dict[int, int],
    *,
    run_id: int,
    ci_mass: float,
    max_quantile_mcse_frac: float = DEFAULT_MAX_QUANTILE_MCSE_FRAC,
    max_median_mcse_log: float = DEFAULT_MAX_MEDIAN_MCSE_LOG,
    year_of_side: Mapping[int, int | None] | None = None,
    recovery: RecoverySettings | None = None,
    ancient_cutoff_year: int = 500,
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
        year_of_side: ``side_id`` to the battle's astronomical year, for the
            recovery diagnostics' era groups; omitted sides count as undated.
        recovery: Sizes and thresholds of the recovery diagnostics.
        ancient_cutoff_year: The stage's ancient cutoff.

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
        "sigma_type": _sigma_summary(idata, "sigma_sq_type", square_root=True),
        "sigma_type_ancient": _sigma_summary(idata, "sigma_type_ancient"),
        "nu": _scalar_stats(idata, "nu") if "nu" in idata.posterior.data_vars else None,
        "recovery": recovery_block(
            idata,
            design,
            {e.side_id: math.log(e.value) for e in estimates if e.value > 0},
            battle_of_side=dict(battle_of_side),
            year_of_side=dict(year_of_side or {}),
            settings=recovery or RecoverySettings(),
            ancient_cutoff_year=ancient_cutoff_year,
        ),
    }
    shared = _shared_draws(idata, design)
    side_era, side_share = _side_regime_loading(design)
    return NeutralSummary(
        estimates=estimates,
        diagnostics=diagnostics,
        side_ids=design.side_ids,
        battle_ids=tuple(battle_of_side[s] for s in design.side_ids),
        draws=_thin(mu, MAX_NEUTRAL_DRAWS),
        m0_mean=_finite(float(np.mean(_pooled(idata, "m0")))) or 0.0,
        s0_mean=_finite(float(np.mean(_pooled(idata, "s0")))) or 0.0,
        g_unlabelled=None if shared is None else _thin(shared[0], MAX_NEUTRAL_DRAWS),
        a_era=None if shared is None else _thin(shared[1], MAX_NEUTRAL_DRAWS),
        offset_bar=None if shared is None else _thin(shared[2], MAX_NEUTRAL_DRAWS),
        m0_draws=None if shared is None else _thin(shared[3], MAX_NEUTRAL_DRAWS),
        side_era=side_era,
        side_unlabelled_share=side_share,
        likelihood=None if shared is None else _side_likelihood(idata, design, shared),
    )


def _sigma_summary(
    idata: az.InferenceData, name: str, *, square_root: bool = False
) -> dict[str, float | None]:
    """Posterior mean of a per-source-type noise scale, by type; empty if absent.

    Args:
        idata: Posterior from the neutral fit.
        name: ``sigma_sq_type`` or ``sigma_type_ancient``.
        square_root: Take the square root first (for a variance).

    Returns:
        Source type to posterior mean.
    """
    if name not in idata.posterior.data_vars:
        return {}
    values = _pooled(idata, name)
    if square_root:
        values = np.sqrt(values)
    labels = _coord_labels(idata, "source_type")
    return {label: _finite(float(values[:, i].mean())) for i, label in enumerate(labels)}


def _side_likelihood(
    idata: az.InferenceData,
    design: Design,
    shared: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> LikelihoodExport:
    """Each side's likelihood from its point rows, with loadings on the shared draws.

    Args:
        idata: Posterior from the neutral fit.
        design: The design the model was built from.
        shared: From :func:`_shared_draws`.

    Returns:
        The export. Its loadings regress on the same thinned draws that are
        written to the file, so a round that picks a row of those draws applies
        the loading to the draw it was fitted against.
    """
    names = set(idata.posterior.data_vars)

    def draw_mean(name: str) -> np.ndarray:
        mean: np.ndarray = _pooled(idata, name).mean(axis=0, keepdims=True)
        return mean

    def draw_thinned(name: str) -> np.ndarray:
        return _thin(_pooled(idata, name), MAX_NEUTRAL_DRAWS)

    theta = np.column_stack(
        [
            _thin(shared[0], MAX_NEUTRAL_DRAWS),
            _thin(shared[1], MAX_NEUTRAL_DRAWS),
            _thin(shared[2], MAX_NEUTRAL_DRAWS),
        ]
    )
    return likelihood_export(
        draw_mean,
        draw_thinned,
        design,
        theta,
        LOADING_NAMES,
        has_nu="nu" in names,
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
    arrays: dict[str, Any] = {
        "run_id": np.int64(run_id),
        "database": np.str_(database),
        "started_at": np.str_(started_at),
        "side_id": np.asarray(summary.side_ids, dtype=np.int64),
        "battle_id": np.asarray(summary.battle_ids, dtype=np.int64),
        "draws": summary.draws,
        "m0_mean": np.float64(summary.m0_mean),
        "s0_mean": np.float64(summary.s0_mean),
    }
    likelihood = summary.likelihood
    if likelihood is not None:
        arrays.update(
            {
                "lik_mean": np.asarray(likelihood.mean, dtype=np.float64),
                "lik_sd": np.asarray(likelihood.sd, dtype=np.float64),
                "lik_has_point": np.asarray(likelihood.has_point, dtype=bool),
                "lik_n_point_rows": np.asarray(likelihood.n_point_rows, dtype=np.int64),
                "lik_loadings": np.asarray(likelihood.loadings, dtype=np.float64),
                "lik_loading_names": np.asarray(likelihood.loading_names, dtype=np.str_),
            }
        )
    shared = {
        "g_unlabelled": summary.g_unlabelled,
        "a_era": summary.a_era,
        "offset_bar": summary.offset_bar,
        "m0_draws": summary.m0_draws,
        "side_era": summary.side_era,
        "side_unlabelled_share": summary.side_unlabelled_share,
    }
    arrays.update({k: np.asarray(v, dtype=np.float64) for k, v in shared.items() if v is not None})
    np.savez(tmp, **arrays)
    tmp.replace(path)
    logger.info(
        "reconcile_neutral_draws_written",
        path=str(path),
        n_draws=int(summary.draws.shape[0]),
        n_sides=len(summary.side_ids),
    )
    return path
