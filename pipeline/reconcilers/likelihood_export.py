"""What each side's own reports say, exported for the impute stage.

The impute stage used to recover this by dividing reconcile's side prior
``N(m0, s0^2)`` out of the neutral posterior. The prior is a Student-t, so for a
side far from ``m0`` (the largest and smallest armies) the posterior is wide and
near-bimodal and the Gaussian division judged it weak: 75 of 341 sides were
discarded, and impute's kept-pair log-ratio slope was 0.43
(``docs/handover-log/31-troop-numbers-diagnosis-2026-10-22.md``, Q2).

Here the likelihood is built directly from the side's **point rows** at the
fitted noise, with nothing to divide out:

* each point row contributes an adjusted log report ``y_r - o_r``, where ``o_r``
  is the row's regime, scope, source and era offset at its posterior mean;
* a row's Fisher information for a location parameter under Student-t(nu) noise
  of scale ``sigma_r`` is ``(nu + 1) / ((nu + 3) sigma_r^2)`` (``1 / sigma_r^2``
  for a Normal fit);
* rows combine by precision, so a side's likelihood is
  ``N(sum_r I_r (y_r - o_r) / sum_r I_r, 1 / sum_r I_r)``.

A side with no point row (bound-only) has no likelihood and is missing.

## Shared error

The offsets are uncertain, and shared across sides. Their effect on a side's
likelihood mean is exported as loadings on the shared parameters
``(g_unlabelled, a_era, offset_bar)``: the least-squares slope of the side's
per-draw likelihood mean on the centred draw of each. Impute draws one of the
exported parameter draws per round and shifts every side by its loading times
it, so the cross-battle correlation of the reconcile error survives. Offsets not
among the three (other regimes, source and scope terms beyond what ``offset_bar``
averages) stay as unmodelled shared error.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np

from pipeline.reconcilers.design import Design
from pipeline.reconcilers.summarise import _row_eta_and_var

_Draw = Callable[[str], np.ndarray]

__all__ = [
    "LOADING_NAMES",
    "LikelihoodExport",
    "combine_rows",
    "fisher_information",
    "likelihood_export",
    "regress_loadings",
]

LOADING_NAMES: Final[tuple[str, ...]] = ("g_unlabelled", "a_era", "offset_bar")
# A shared parameter this flat across draws carries no error to propagate.
_THETA_VARIANCE_FLOOR: Final[float] = 1e-12


@dataclass(frozen=True)
class LikelihoodExport:
    """Per-side likelihood summaries from the neutral fit, in ``design.side_ids`` order.

    Attributes:
        has_point: Whether the side has at least one point row.
        mean: Likelihood mean on the log scale, NaN where ``has_point`` is False.
        sd: Likelihood sd (Fisher), NaN where ``has_point`` is False.
        n_point_rows: Number of point rows behind each side's likelihood.
        loadings: ``(n_sides, p)`` shift of ``mean`` per unit of each shared
            parameter; zero rows for a side without a likelihood.
        loading_names: The ``p`` shared parameters kept, in column order.
    """

    has_point: np.ndarray
    mean: np.ndarray
    sd: np.ndarray
    n_point_rows: np.ndarray
    loadings: np.ndarray
    loading_names: tuple[str, ...]


def fisher_information(sd: np.ndarray, nu: float | None) -> np.ndarray:
    """Fisher information about a location parameter, per observation.

    Args:
        sd: Scale of each observation (the model's ``sd``, not the Student-t's
            standard deviation).
        nu: Degrees of freedom, or None for Normal noise.

    Returns:
        ``(nu + 1) / ((nu + 3) sd^2)``; ``1 / sd^2`` when ``nu`` is None.
    """
    precision: np.ndarray = 1.0 / np.maximum(np.asarray(sd, dtype=float), 1e-12) ** 2
    if nu is None:
        return precision
    information: np.ndarray = precision * (nu + 1.0) / (nu + 3.0)
    return information


def combine_rows(
    adjusted: np.ndarray, information: np.ndarray, side_of_row: np.ndarray, n_sides: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine rows into per-side likelihoods by precision.

    Args:
        adjusted: Adjusted log reports, ``(n_rows,)`` or ``(n_draws, n_rows)``.
        information: Fisher information per row, ``(n_rows,)``.
        side_of_row: Position of each row's side, ``(n_rows,)``.
        n_sides: Number of sides.

    Returns:
        ``(mean, precision, n_rows)``. ``mean`` has the leading shape of
        ``adjusted`` with ``n_sides`` columns, NaN for a side with no row;
        ``precision`` and ``n_rows`` are ``(n_sides,)``.
    """
    precision = np.bincount(side_of_row, weights=information, minlength=n_sides)
    n_rows = np.bincount(side_of_row, minlength=n_sides)
    weighted = np.atleast_2d(adjusted) * information[None, :]
    totals = np.zeros((weighted.shape[0], n_sides))
    np.add.at(totals.T, side_of_row, weighted.T)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(precision > 0, totals / np.where(precision > 0, precision, 1.0), np.nan)
    if np.ndim(adjusted) == 1:
        mean = mean[0]
    return mean, precision, n_rows


def regress_loadings(
    mean_draws: np.ndarray, theta: np.ndarray, names: tuple[str, ...]
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Slope of each side's per-draw likelihood mean on the centred shared parameters.

    Args:
        mean_draws: ``(n_draws, n_sides)``, NaN for a side without a likelihood.
        theta: ``(n_draws, len(names))`` draws of the shared parameters, row-aligned
            with ``mean_draws``.
        names: Column names of ``theta``.

    Returns:
        ``(loadings, kept_names)``: ``loadings`` is ``(n_sides, p)`` with ``p``
        the number of columns whose variance exceeds a floor, zero rows for a
        side without a likelihood.
    """
    n_draws, n_sides = mean_draws.shape
    keep = [i for i in range(theta.shape[1]) if float(np.var(theta[:, i])) > _THETA_VARIANCE_FLOOR]
    kept_names = tuple(names[i] for i in keep)
    if not keep or n_draws <= len(keep) + 1:
        return np.zeros((n_sides, 0)), ()
    centred = theta[:, keep] - theta[:, keep].mean(axis=0)
    has_mean = np.all(np.isfinite(mean_draws), axis=0)
    loadings = np.zeros((n_sides, len(keep)))
    if has_mean.any():
        target = mean_draws[:, has_mean]
        target = target - target.mean(axis=0)
        coef = np.linalg.lstsq(centred, target, rcond=None)[0]
        loadings[has_mean] = coef.T
    return loadings, kept_names


def likelihood_export(
    draw_mean: _Draw,
    draw_thinned: _Draw,
    design: Design,
    theta: np.ndarray,
    theta_names: tuple[str, ...] = LOADING_NAMES,
    *,
    has_nu: bool,
) -> LikelihoodExport:
    """Build every side's likelihood from a fitted neutral model.

    Args:
        draw_mean: Name to the posterior mean of that variable, with a leading
            axis of length 1 (the shape :func:`pipeline.reconcilers.summarise._row_eta_and_var`
            reads).
        draw_thinned: Name to the thinned posterior draws, leading axis the draws.
        design: The design the fit used.
        theta: Thinned draws of the shared parameters, row-aligned with
            ``draw_thinned``.
        theta_names: Column names of ``theta``.
        has_nu: Whether the fit is Student-t (``nu`` is a variable).

    Returns:
        The export.
    """
    n_sides = design.n_sides
    rows = design.point_rows
    mean = np.full(n_sides, np.nan)
    sd = np.full(n_sides, np.nan)
    n_rows = np.zeros(n_sides, dtype=np.int64)
    has_point = np.zeros(n_sides, dtype=bool)
    if rows.size == 0:
        return LikelihoodExport(has_point, mean, sd, n_rows, np.zeros((n_sides, 0)), ())

    side_of_row = design.side_index[rows]
    y = design.log_y[rows]
    nu = float(draw_mean("nu").reshape(-1)[0]) if has_nu else None

    def offsets(draw: _Draw) -> np.ndarray:
        eta, _ = _row_eta_and_var(draw, design, rows)
        shared = draw("mu_side")[:, design.side_index[rows]]
        result: np.ndarray = eta - shared
        return result

    eta_mean, var_mean = _row_eta_and_var(draw_mean, design, rows)
    info = fisher_information(np.sqrt(var_mean[0]), nu)
    base_offset = eta_mean[0] - draw_mean("mu_side")[0, side_of_row]
    mean_side, precision, n_rows = combine_rows(y - base_offset, info, side_of_row, n_sides)
    has_point = precision > 0
    mean = np.where(has_point, mean_side, np.nan)
    sd = np.where(has_point, 1.0 / np.sqrt(np.where(has_point, precision, 1.0)), np.nan)

    adjusted_draws = y[None, :] - offsets(draw_thinned)
    mean_draws, _, _ = combine_rows(adjusted_draws, info, side_of_row, n_sides)
    loadings, kept = regress_loadings(mean_draws, theta, theta_names)
    return LikelihoodExport(has_point, mean, sd, n_rows, loadings, kept)
