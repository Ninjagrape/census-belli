"""Turning reconcile's output into a likelihood for the imputation model.

## Why the prior has to be divided out

Reconcile's neutral fit gives each side a posterior built from its reports
*and* from an independent side prior ``N(m0, s0^2)``. Feeding that posterior to
the imputation model as if it were data would count reconcile's prior as
evidence, and would count it again through the imputation model's own era and
covariate prior. Dividing the prior out leaves what the reports alone say::

    P_lik = S_post^-1 - I / s0^2
    m_lik = P_lik^-1 (S_post^-1 mu_post - (m0 / s0^2) 1)

A side whose likelihood precision is at most ``min_precision`` (0.5, i.e. a
likelihood sd of about 1.4) carries essentially no information of its own, so
it is treated as missing and the rest of its battle is recomputed without it.
If the recomputed ``P_lik`` is still not positive definite (empirical
covariance from at most 1000 draws is noisy), the battle falls back to its
diagonal.

## The side prior's mean moves with theta

``m0`` is not a constant of the fit: reconcile samples ``corpus_level`` and
derives ``m0 = corpus_level - offset_bar``, and ``offset_bar`` is mostly
``w_u * g_unlabelled``. In the posterior ``m0`` therefore falls by about
``w_u`` per unit ``g``. A shift of the posterior means by ``B dtheta`` carries
back to the likelihood mean as ``B + P^-1 (B - 1 c') / s0^2`` with
``c = d m0 / d theta``, not ``B + P^-1 B / s0^2`` (which assumes ``m0`` fixed
and over-states the loading of an unlabelled side by ``1 / (P s0^2)``). ``c``
comes from the exported ``m0`` draws; an old file without them falls back to the
fixed-``m0`` formula.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import structlog

from pipeline.imputers.records import BattleLikelihood, SideRow

__all__ = [
    "DivisionResult",
    "LikelihoodStats",
    "NeutralDraws",
    "SharedError",
    "fit_shared_error",
    "shared_error_summary",
    "divide_prior",
    "likelihoods_from_draws",
    "likelihoods_from_published",
    "load_neutral_draws",
]

logger = structlog.get_logger()

# Guards the diagonal fallback against a degenerate (constant) column.
_VARIANCE_FLOOR = 1e-9
# The 95% interval on the log scale spans 2 * 1.96 sd.
_INTERVAL_TO_SD = 3.92
_MIN_PUBLISHED_SD = 0.05
# A shared parameter this flat across draws carries no error to propagate.
_THETA_VARIANCE_FLOOR = 1e-12


@dataclass(frozen=True)
class DivisionResult:
    """Outcome of dividing reconcile's side prior out of one battle's posterior.

    Attributes:
        keep: Indices (into the input vectors) of the sides that carry a
            likelihood. Everything else is treated as missing.
        m_lik: Likelihood mean for the kept sides, shape ``(len(keep),)``.
        precision: Likelihood precision, ``(len(keep), len(keep))``.
        method: ``'joint'`` or ``'diagonal'``.
    """

    keep: tuple[int, ...]
    m_lik: np.ndarray
    precision: np.ndarray
    method: str


_OPTIONAL_ARRAYS = (
    "g_unlabelled",
    "a_era",
    "offset_bar",
    "side_era",
    "side_unlabelled_share",
    "m0_draws",
)


@dataclass(frozen=True)
class NeutralDraws:
    """The contents of reconcile's ``neutral_draws_run{run_id}.npz``.

    The last six fields are absent from files written before reconcile exported
    the shared regime draws, and are then None. ``g_unlabelled``, ``a_era``,
    ``offset_bar`` and ``m0_draws`` have shape ``(n_draws,)`` and are
    row-aligned with ``draws``; ``side_era`` and ``side_unlabelled_share`` have
    shape ``(n_sides,)``, in ``side_id`` order. ``m0_draws`` is the side prior's
    mean, which moves with the regime parameters (see the module docstring); a
    file with the regime arrays but not ``m0_draws`` is treated as having a
    fixed ``m0``.
    """

    run_id: int
    side_id: np.ndarray
    battle_id: np.ndarray
    draws: np.ndarray
    m0_mean: float
    s0_mean: float
    database: str | None = None
    started_at: str | None = None
    g_unlabelled: np.ndarray | None = None
    a_era: np.ndarray | None = None
    offset_bar: np.ndarray | None = None
    side_era: np.ndarray | None = None
    side_unlabelled_share: np.ndarray | None = None
    m0_draws: np.ndarray | None = None


@dataclass(frozen=True)
class SharedError:
    """Reconcile's shared parameter error, as a regression of side draws on theta.

    Attributes:
        names: The shared parameters kept (a constant or absent one is dropped).
        theta_centered: ``(n_draws, p)`` draws of theta minus their mean, one
            row per reconcile draw. A round picks one row.
        mean_abs_loading: Mean ``|B_s|`` over in-scope sides and components.
        explained_share: Share of the in-scope sides' total draw variance that
            theta explains, in ``[0, 1]``.
        m0_loading: ``(p,)`` regression of reconcile's side-prior mean ``m0`` on
            centred theta, or None when the file has no ``m0`` draws (``m0``
            is then taken as fixed).
        loading_gap: Mean absolute gap, over kept sides and components, between
            the recovered likelihood loading and the design's (``-share`` on
            ``g_unlabelled``, ``-share * side_era`` on ``a_era``); None until
            computed or when the per-side arrays are absent.
    """

    names: tuple[str, ...]
    theta_centered: np.ndarray
    mean_abs_loading: float
    explained_share: float
    m0_loading: np.ndarray | None = None
    loading_gap: float | None = None


@dataclass
class LikelihoodStats:
    """Counts from building likelihoods, for the run's log and counters.

    ``shared`` is None when the shared error is not propagated, and
    ``shared_reason`` then says why.
    """

    stale_sides: int = 0
    weak_sides: int = 0
    diagonal_battles: int = 0
    shared: SharedError | None = None
    shared_reason: str = "disabled"


def _is_positive_definite(matrix: np.ndarray) -> bool:
    """Whether a symmetric matrix has a Cholesky factor.

    Args:
        matrix: A square matrix.

    Returns:
        True if it is positive definite.
    """
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        return False
    return True


def _likelihood_precision(cov: np.ndarray, s0: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Invert a posterior covariance and subtract the side prior's precision.

    Args:
        cov: Posterior covariance of the sides, ``(k, k)``.
        s0: Reconcile's side-prior sd.

    Returns:
        ``(S_post^-1, P_lik)``, or None when the covariance is singular.
    """
    try:
        posterior_precision = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(posterior_precision)):
        return None
    posterior_precision = (posterior_precision + posterior_precision.T) / 2.0
    return posterior_precision, posterior_precision - np.eye(len(cov)) / s0**2


def _diagonal_division(
    mu: np.ndarray, var: np.ndarray, m0: float, s0: float, min_precision: float
) -> DivisionResult:
    """The per-side fallback: marginal variances only, no cross terms.

    Args:
        mu: Posterior means of the candidate sides.
        var: Their posterior variances.
        m0: Reconcile's side-prior mean.
        s0: Reconcile's side-prior sd.
        min_precision: Sides at or below this likelihood precision are dropped.

    Returns:
        A diagonal :class:`DivisionResult`; ``keep`` indexes ``mu``.
    """
    var = np.maximum(var, _VARIANCE_FLOOR)
    precision = 1.0 / var - 1.0 / s0**2
    keep = np.flatnonzero(precision > min_precision)
    p = precision[keep]
    m_lik = (mu[keep] / var[keep] - m0 / s0**2) / p
    return DivisionResult(tuple(int(i) for i in keep), m_lik, np.diag(p), "diagonal")


def _diagonal_on_subset(
    mu: np.ndarray,
    cov: np.ndarray,
    keep: np.ndarray,
    m0: float,
    s0: float,
    min_precision: float,
) -> DivisionResult:
    """Diagonal fallback over the sides in ``keep``, indexed back to the input.

    Args:
        mu: Posterior means of all candidate sides.
        cov: Their posterior covariance.
        keep: Indices of the sides still candidates.
        m0: Reconcile's side-prior mean.
        s0: Reconcile's side-prior sd.
        min_precision: See :func:`_diagonal_division`.

    Returns:
        A diagonal result whose ``keep`` indexes the original ``mu``, not the
        subset (the diagonal path may drop further sides).
    """
    fallback = _diagonal_division(mu[keep], np.diag(cov)[keep], m0, s0, min_precision)
    return DivisionResult(
        tuple(int(keep[i]) for i in fallback.keep),
        fallback.m_lik,
        fallback.precision,
        "diagonal",
    )


def _weak_sides(cov: np.ndarray, s0: float, min_precision: float) -> np.ndarray:
    """Which sides carry at most ``min_precision`` of likelihood precision.

    Args:
        cov: Posterior covariance of all the battle's sides.
        s0: Reconcile's side-prior sd.
        min_precision: The threshold.

    Returns:
        Boolean per side; True means weak. Uses the marginal variances when
        ``cov`` is singular.
    """
    first = _likelihood_precision(cov, s0)
    if first is None:
        var = np.maximum(np.diag(cov), _VARIANCE_FLOOR)
        marginal: np.ndarray = 1.0 / var - 1.0 / s0**2
        return marginal <= min_precision
    weak: np.ndarray = np.diag(first[1]) <= min_precision
    return weak


def divide_prior(
    mu_post: np.ndarray,
    s_post: np.ndarray,
    m0: float,
    s0: float,
    min_precision: float,
    weakness_cov: np.ndarray | None = None,
) -> DivisionResult:
    """Divide reconcile's independent side prior out of one battle's posterior.

    Args:
        mu_post: Posterior mean of the observed sides' log troop counts.
        s_post: Their posterior covariance, ``(k, k)``.
        m0: Reconcile's fitted side-prior mean (posterior mean of ``m0``).
        s0: Reconcile's fitted side-prior sd (posterior mean of ``s0``).
            Both are posterior means, so their own uncertainty is ignored in
            the division; at corpus scale they are pinned by thousands of
            sides and the omission is immaterial.
        min_precision: A side whose likelihood-precision diagonal is at or
            below this is treated as missing.
        weakness_cov: The covariance to classify weak sides on, when ``s_post``
            has been residualised on the shared parameters. Classifying on the
            raw covariance keeps the set of missing sides, and so the
            ``missing_data_log`` rows, the same whether or not the shared error
            is propagated. Defaults to ``s_post``.

    Returns:
        The sides kept, with the likelihood mean and precision for them. An
        empty ``keep`` means no side carries a likelihood.
    """
    mu = np.atleast_1d(np.asarray(mu_post, dtype=float))
    cov = np.atleast_2d(np.asarray(s_post, dtype=float))
    raw = cov if weakness_cov is None else np.atleast_2d(np.asarray(weakness_cov, dtype=float))

    keep = np.flatnonzero(~_weak_sides(raw, s0, min_precision))
    if keep.size == 0:
        return DivisionResult((), np.empty(0), np.empty((0, 0)), "joint")

    division = _likelihood_precision(cov[np.ix_(keep, keep)], s0)
    if division is None:
        return _diagonal_on_subset(mu, cov, keep, m0, s0, min_precision)
    posterior_precision, precision = division
    if not _is_positive_definite(precision):
        return _diagonal_on_subset(mu, cov, keep, m0, s0, min_precision)

    rhs = posterior_precision @ mu[keep] - (m0 / s0**2)
    m_lik = np.linalg.solve(precision, rhs)
    return DivisionResult(tuple(int(i) for i in keep), m_lik, precision, "joint")


def load_neutral_draws(path: Path) -> NeutralDraws:
    """Read reconcile's neutral draws file and validate its shapes.

    Args:
        path: ``neutral_draws_run{run_id}.npz``.

    Returns:
        The arrays and scalars.

    Raises:
        ValueError: If the arrays disagree in length or are not finite.
    """
    with np.load(path) as data:
        run_id = int(data["run_id"])
        side_id = np.asarray(data["side_id"], dtype=np.int64)
        battle_id = np.asarray(data["battle_id"], dtype=np.int64)
        draws = np.asarray(data["draws"], dtype=np.float64)
        m0 = float(data["m0_mean"])
        s0 = float(data["s0_mean"])
        database = str(data["database"]) if "database" in data.files else None
        started_at = str(data["started_at"]) if "started_at" in data.files else None
        optional = {
            key: np.asarray(data[key], dtype=np.float64)
            for key in _OPTIONAL_ARRAYS
            if key in data.files
        }
    if draws.ndim != 2 or draws.shape[1] != len(side_id) or len(battle_id) != len(side_id):
        raise ValueError(
            f"{path}: draws {draws.shape} does not match {len(side_id)} sides / "
            f"{len(battle_id)} battles"
        )
    if not np.all(np.isfinite(draws)):
        raise ValueError(f"{path}: draws contain non-finite values")
    if not s0 > 0:
        raise ValueError(f"{path}: s0_mean must be positive, got {s0}")
    n_draws, n_sides = draws.shape
    expected = {
        "g_unlabelled": (n_draws,),
        "a_era": (n_draws,),
        "offset_bar": (n_draws,),
        "m0_draws": (n_draws,),
        "side_era": (n_sides,),
        "side_unlabelled_share": (n_sides,),
    }
    for key, array in optional.items():
        if array.shape != expected[key]:
            raise ValueError(f"{path}: {key} has shape {array.shape}, expected {expected[key]}")
    return NeutralDraws(run_id, side_id, battle_id, draws, m0, s0, database, started_at, **optional)


def _likelihood_loading(
    posterior_loading: np.ndarray,
    precision: np.ndarray,
    s0: float,
    m0_loading: np.ndarray | None = None,
) -> np.ndarray:
    """Carry a shift of the posterior means back through the prior division.

    The likelihood mean is ``P^-1 (S^-1 m - m0/s0^2 1)`` with
    ``S^-1 = P + I/s0^2``. Differentiating in theta with ``m0`` allowed to move
    (``c = d m0 / d theta``) gives
    ``B_lik = B + P^-1 (B - 1 c') / s0^2``: reconcile's side prior pulls the
    posterior toward ``m0``, so the data alone moved more, unless the prior mean
    moved with them. With ``c = 0`` (``m0`` fixed) the prior pull is all that is
    undone.

    Args:
        posterior_loading: ``(k, p)`` regression loadings of the kept sides.
        precision: The kept sides' likelihood precision, ``(k, k)``.
        s0: Reconcile's side-prior sd.
        m0_loading: ``(p,)`` loading of ``m0`` on theta, or None for fixed ``m0``.

    Returns:
        ``(k, p)`` loadings of the likelihood mean.
    """
    pull = posterior_loading
    if m0_loading is not None:
        pull = posterior_loading - m0_loading[None, :]
    loading: np.ndarray = posterior_loading + np.linalg.solve(precision, pull) / s0**2
    return loading


def _theta_matrix(neutral: NeutralDraws) -> tuple[tuple[str, ...], np.ndarray] | None:
    """Stack the shared parameters that vary across draws.

    Args:
        neutral: The loaded draws file.

    Returns:
        Names and an ``(n_draws, p)`` matrix, or None when none is usable.
    """
    names: list[str] = []
    columns: list[np.ndarray] = []
    for name in ("g_unlabelled", "a_era"):
        values: np.ndarray | None = getattr(neutral, name)
        if values is None or float(np.var(values)) <= _THETA_VARIANCE_FLOOR:
            continue
        names.append(name)
        columns.append(values)
    if not columns:
        return None
    return tuple(names), np.column_stack(columns)


def fit_shared_error(
    neutral: NeutralDraws, columns: Sequence[int]
) -> tuple[SharedError, np.ndarray, np.ndarray] | str:
    """Regress side draws on the shared reconcile parameters, all sides at once.

    ``draws[:, s] = alpha_s + B_s . (theta - mean theta) + e_s``, one
    least-squares solve for every column.

    Args:
        neutral: The loaded draws file.
        columns: Columns of ``neutral.draws`` to regress (the in-scope sides).

    Returns:
        ``(shared, loadings, residual_draws)``: ``loadings`` is
        ``(len(columns), p)`` and ``residual_draws`` is ``(n_draws,
        len(columns))`` with each column's mean added back, so it feeds
        :func:`divide_prior` in place of the raw draws. When the regression is
        not possible, a string naming why.
    """
    stacked = _theta_matrix(neutral)
    if stacked is None:
        absent = neutral.g_unlabelled is None and neutral.a_era is None
        return "shared_arrays_absent" if absent else "theta_has_no_variance"
    names, theta = stacked
    n_draws = theta.shape[0]
    if n_draws <= len(names) + 1 or not columns:
        return "too_few_draws"
    theta_centered = theta - theta.mean(axis=0)
    y = neutral.draws[:, list(columns)]
    y_mean = y.mean(axis=0)
    y_centered = y - y_mean
    # m0 rides in the same solve as an extra column; it is not a side.
    with_m0 = neutral.m0_draws is not None
    target = y_centered
    if neutral.m0_draws is not None:
        target = np.column_stack([y_centered, neutral.m0_draws - neutral.m0_draws.mean()])
    full_coef = np.linalg.lstsq(theta_centered, target, rcond=None)[0]
    coef = full_coef[:, : y.shape[1]]
    m0_loading = full_coef[:, -1].copy() if with_m0 else None
    residual = y_centered - theta_centered @ coef
    total = float(y_centered.var(axis=0).sum())
    explained = 0.0 if total <= 0 else 1.0 - float(residual.var(axis=0).sum()) / total
    shared = SharedError(
        names=names,
        theta_centered=theta_centered,
        mean_abs_loading=float(np.abs(coef).mean()),
        explained_share=min(max(explained, 0.0), 1.0),
        m0_loading=m0_loading,
    )
    return shared, coef.T, residual + y_mean


def shared_error_summary(
    shared: SharedError | None, reason: str, theta_index: Sequence[int] | None = None
) -> dict[str, object]:
    """Describe whether and how the shared error was propagated, for the reports.

    Args:
        shared: The fitted shared error, or None.
        reason: Why it is None (ignored otherwise).
        theta_index: The reconcile draw each round used.

    Returns:
        A JSON-serialisable mapping.
    """
    if shared is None:
        return {"propagated": False, "reason": reason}
    return {
        "propagated": True,
        "theta": list(shared.names),
        "mean_abs_loading": shared.mean_abs_loading,
        "explained_variance_share": shared.explained_share,
        "m0_comoves": shared.m0_loading is not None,
        "m0_loading": None if shared.m0_loading is None else [float(c) for c in shared.m0_loading],
        "loading_gap_vs_design": shared.loading_gap,
        "theta_draw_index": None if theta_index is None else [int(i) for i in theta_index],
    }


def _design_loading_gap(
    neutral: NeutralDraws,
    shared: SharedError | None,
    columns: Sequence[int],
    shift: np.ndarray,
) -> float | None:
    """Mean absolute gap between recovered and design likelihood loadings.

    A report on an unlabelled row is ``y + g + a_era * era``, so the data's
    loading of a side's latent value on ``g`` is ``-share`` and on ``a_era`` is
    ``-share * era``. This is a check on the regression and the prior division,
    not an input: a large gap means the back-propagation is wrong.

    Args:
        neutral: The loaded draws file.
        shared: The fitted shared error, for the component names.
        columns: Columns of ``neutral.draws`` for the kept sides.
        shift: ``(len(columns), p)`` recovered likelihood loadings.

    Returns:
        The mean absolute gap over sides and components, or None when the
        per-side design arrays are absent.
    """
    share = neutral.side_unlabelled_share
    era = neutral.side_era
    if shared is None or share is None or era is None:
        return None
    design = np.column_stack(
        [
            -share[list(columns)] * (era[list(columns)] if name == "a_era" else 1.0)
            for name in shared.names
        ]
    )
    return float(np.abs(shift - design).mean())


def likelihoods_from_draws(
    neutral: NeutralDraws,
    in_scope_side_ids: set[int],
    *,
    min_precision: float,
    propagate_shared: bool = True,
) -> tuple[list[BattleLikelihood], LikelihoodStats]:
    """Build one likelihood per battle from the neutral fit's draws.

    Args:
        neutral: The loaded draws file.
        in_scope_side_ids: Sides that exist in the database now. A side in
            the file but not here (deleted since the fit) is ignored.
        min_precision: See :func:`divide_prior`.
        propagate_shared: Build each likelihood from the draws' residuals after
            regressing on the shared parameters, and attach the loadings so a
            round can shift them (see :func:`fit_shared_error`). Falls back to
            the raw draws, logging ``impute_shared_error_not_propagated``, when
            the file lacks the arrays or theta does not vary.

    Returns:
        The likelihoods, and counts of stale, weak and diagonal-fallback cases.

    A battle with only some sides' columns builds a block from those present;
    the absent side gets only the prior, though shared battle and regime
    parameters still carry some of its information.
    """
    stats = LikelihoodStats()
    columns_by_battle: dict[int, list[int]] = defaultdict(list)
    for column, (sid, bid) in enumerate(zip(neutral.side_id, neutral.battle_id, strict=True)):
        if int(sid) in in_scope_side_ids:
            columns_by_battle[int(bid)].append(column)
        else:
            stats.stale_sides += 1

    draws = neutral.draws
    loadings: np.ndarray | None = None
    if propagate_shared:
        in_scope = sorted(c for cols in columns_by_battle.values() for c in cols)
        fit = fit_shared_error(neutral, in_scope)
        if isinstance(fit, str):
            stats.shared_reason = fit
        else:
            stats.shared, in_scope_loadings, residual = fit
            draws = neutral.draws.copy()
            draws[:, in_scope] = residual
            loadings = np.zeros((neutral.draws.shape[1], in_scope_loadings.shape[1]))
            loadings[in_scope] = in_scope_loadings
    if stats.shared is None:
        logger.warning("impute_shared_error_not_propagated", reason=stats.shared_reason)
    elif stats.shared.m0_loading is None:
        # Old file: the side prior's mean is taken as fixed, which overstates
        # the loading of sides whose prior mean actually moves with theta.
        logger.warning("impute_m0_draws_absent", hint="re-run reconcile for the corrected loading")
    m0_loading = None if stats.shared is None else stats.shared.m0_loading
    design_gaps: list[float] = []

    out: list[BattleLikelihood] = []
    for battle_id in sorted(columns_by_battle):
        columns = columns_by_battle[battle_id]
        block = draws[:, columns]
        mu = block.mean(axis=0)
        if block.shape[0] > 1:
            cov = np.atleast_2d(np.cov(block, rowvar=False))
        else:
            cov = np.diag(np.full(len(columns), _VARIANCE_FLOOR))
        raw_cov = None
        if loadings is not None:
            raw_block = neutral.draws[:, columns]
            raw_cov = np.atleast_2d(np.cov(raw_block, rowvar=False)) if len(raw_block) > 1 else cov
        result = divide_prior(
            mu, cov, neutral.m0_mean, neutral.s0_mean, min_precision, weakness_cov=raw_cov
        )
        stats.weak_sides += len(columns) - len(result.keep)
        if not result.keep:
            continue
        if result.method == "diagonal":
            stats.diagonal_battles += 1
        side_ids = tuple(int(neutral.side_id[columns[i]]) for i in result.keep)
        shift = None
        if loadings is not None:
            kept_columns = [columns[i] for i in result.keep]
            shift = _likelihood_loading(
                loadings[kept_columns], result.precision, neutral.s0_mean, m0_loading
            )
            gap = _design_loading_gap(neutral, stats.shared, kept_columns, shift)
            if gap is not None:
                design_gaps.append(gap)
        out.append(
            BattleLikelihood(
                battle_id, side_ids, result.m_lik, result.precision, result.method, shift
            )
        )
    if stats.shared is not None and design_gaps:
        stats.shared = replace(stats.shared, loading_gap=float(np.mean(design_gaps)))
    return out, stats


def likelihoods_from_published(
    sides: Sequence[SideRow], *, default_sd: float
) -> list[BattleLikelihood]:
    """Build diagonal likelihoods from the published estimates.

    Documented limitation of the sensitivity variant: sides within a battle
    share a regime mix, so their published errors are correlated, and the
    published interval already contains reconcile's side prior. Neither is
    corrected here.

    Args:
        sides: Every in-scope side; those without a published estimate are
            skipped.
        default_sd: Log sd to use when an estimate has no usable interval.

    Returns:
        One likelihood per battle with at least one published side.
    """
    by_battle: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    for side in sides:
        if side.published is None:
            continue
        estimate, lo, hi = side.published
        if estimate <= 0:
            continue
        if lo is not None and hi is not None and lo > 0 and hi > lo:
            sd = max(float(np.log(hi / lo)) / _INTERVAL_TO_SD, _MIN_PUBLISHED_SD)
        else:
            sd = default_sd
        by_battle[side.battle_id].append((side.side_id, float(np.log(estimate)), sd))

    out: list[BattleLikelihood] = []
    for battle_id in sorted(by_battle):
        rows = by_battle[battle_id]
        out.append(
            BattleLikelihood(
                battle_id,
                tuple(r[0] for r in rows),
                np.array([r[1] for r in rows]),
                np.diag([1.0 / r[2] ** 2 for r in rows]),
                "published",
            )
        )
    return out
