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
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import structlog

from pipeline.imputers.records import BattleLikelihood, SideRow

__all__ = [
    "DivisionResult",
    "LikelihoodStats",
    "NeutralDraws",
    "divide_prior",
    "find_latest_draws",
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
_DRAWS_FILE = re.compile(r"neutral_draws_run(\d+)\.npz$")


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


@dataclass(frozen=True)
class NeutralDraws:
    """The contents of reconcile's ``neutral_draws_run{run_id}.npz``."""

    run_id: int
    side_id: np.ndarray
    battle_id: np.ndarray
    draws: np.ndarray
    m0_mean: float
    s0_mean: float


@dataclass
class LikelihoodStats:
    """Counts from building likelihoods, for the run's log and counters."""

    stale_sides: int = 0
    weak_sides: int = 0
    diagonal_battles: int = 0


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
        A diagonal :class:`DivisionResult`.
    """
    var = np.maximum(var, _VARIANCE_FLOOR)
    precision = 1.0 / var - 1.0 / s0**2
    keep = np.flatnonzero(precision > min_precision)
    p = precision[keep]
    m_lik = (mu[keep] / var[keep] - m0 / s0**2) / p
    return DivisionResult(tuple(int(i) for i in keep), m_lik, np.diag(p), "diagonal")


def divide_prior(
    mu_post: np.ndarray,
    s_post: np.ndarray,
    m0: float,
    s0: float,
    min_precision: float,
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

    Returns:
        The sides kept, with the likelihood mean and precision for them. An
        empty ``keep`` means no side carries a likelihood.
    """
    mu = np.atleast_1d(np.asarray(mu_post, dtype=float))
    cov = np.atleast_2d(np.asarray(s_post, dtype=float))
    n = len(mu)
    all_ix = np.arange(n)

    first = _likelihood_precision(cov, s0)
    if first is None:
        return _diagonal_division(mu, np.diag(cov), m0, s0, min_precision)

    weak = np.diag(first[1]) <= min_precision
    keep = all_ix[~weak]
    if keep.size == 0:
        return DivisionResult((), np.empty(0), np.empty((0, 0)), "joint")

    if weak.any():
        sub_cov = cov[np.ix_(keep, keep)]
        second = _likelihood_precision(sub_cov, s0)
        if second is None:
            return _diagonal_division(mu[keep], np.diag(sub_cov), m0, s0, min_precision)
        posterior_precision, precision = second
    else:
        posterior_precision, precision = first

    if not _is_positive_definite(precision):
        sub_cov = cov[np.ix_(keep, keep)]
        fallback = _diagonal_division(mu[keep], np.diag(sub_cov), m0, s0, min_precision)
        # Map the diagonal path's own survivors back to the original indices.
        return DivisionResult(
            tuple(int(keep[i]) for i in fallback.keep),
            fallback.m_lik,
            fallback.precision,
            "diagonal",
        )

    rhs = posterior_precision @ mu[keep] - (m0 / s0**2)
    m_lik = np.linalg.solve(precision, rhs)
    return DivisionResult(tuple(int(i) for i in keep), m_lik, precision, "joint")


def find_latest_draws(root: Path) -> Path | None:
    """Find the draws file with the highest reconcile run id.

    Args:
        root: Directory holding ``neutral_draws_run{run_id}.npz`` files.

    Returns:
        The newest file, or None if there is none.
    """
    best: tuple[int, Path] | None = None
    if not root.is_dir():
        return None
    for path in root.iterdir():
        match = _DRAWS_FILE.search(path.name)
        if match and (best is None or int(match.group(1)) > best[0]):
            best = (int(match.group(1)), path)
    return None if best is None else best[1]


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
    if draws.ndim != 2 or draws.shape[1] != len(side_id) or len(battle_id) != len(side_id):
        raise ValueError(
            f"{path}: draws {draws.shape} does not match {len(side_id)} sides / "
            f"{len(battle_id)} battles"
        )
    if not np.all(np.isfinite(draws)):
        raise ValueError(f"{path}: draws contain non-finite values")
    if not s0 > 0:
        raise ValueError(f"{path}: s0_mean must be positive, got {s0}")
    return NeutralDraws(run_id, side_id, battle_id, draws, m0, s0)


def likelihoods_from_draws(
    neutral: NeutralDraws,
    in_scope_side_ids: set[int],
    *,
    min_precision: float,
) -> tuple[list[BattleLikelihood], LikelihoodStats]:
    """Build one likelihood per battle from the neutral fit's draws.

    Args:
        neutral: The loaded draws file.
        in_scope_side_ids: Sides that exist in the database now. A side in
            the file but not here (deleted since the fit) is ignored.
        min_precision: See :func:`divide_prior`.

    Returns:
        The likelihoods, and counts of stale, weak and diagonal-fallback cases.
    """
    stats = LikelihoodStats()
    columns_by_battle: dict[int, list[int]] = defaultdict(list)
    for column, (sid, bid) in enumerate(zip(neutral.side_id, neutral.battle_id, strict=True)):
        if int(sid) in in_scope_side_ids:
            columns_by_battle[int(bid)].append(column)
        else:
            stats.stale_sides += 1

    out: list[BattleLikelihood] = []
    for battle_id in sorted(columns_by_battle):
        columns = columns_by_battle[battle_id]
        block = neutral.draws[:, columns]
        mu = block.mean(axis=0)
        if block.shape[0] > 1:
            cov = np.atleast_2d(np.cov(block, rowvar=False))
        else:
            cov = np.diag(np.full(len(columns), _VARIANCE_FLOOR))
        result = divide_prior(mu, cov, neutral.m0_mean, neutral.s0_mean, min_precision)
        stats.weak_sides += len(columns) - len(result.keep)
        if not result.keep:
            continue
        if result.method == "diagonal":
            stats.diagonal_battles += 1
        side_ids = tuple(int(neutral.side_id[columns[i]]) for i in result.keep)
        out.append(
            BattleLikelihood(battle_id, side_ids, result.m_lik, result.precision, result.method)
        )
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
