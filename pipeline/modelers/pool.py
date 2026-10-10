"""Pooling across imputation rounds, ranking and convergence roll-up (design 7, 9; R6, R7).

The pooled posterior is the mixture: an equal number of draws from every
round, concatenated. Rubin's rules are computed alongside per quantity::

    W = mean of within-round variances         B = variance of round means
    T = W + (1 + 1/M) B                        FMI = (1 + 1/M) B / T

Reported ``sd`` is ``sqrt(T)``, and intervals are the mixture quantiles
widened about the mixture mean by ``sqrt(T / Var_mixture)``. FMI uses no
degrees-of-freedom correction and is a lower bound while the shared-error
limitation (handover-log 21.7) stands. rhat and ESS are never computed on
concatenated rounds: :func:`round_convergence` is per round, and
:func:`combine_rounds` takes worst cases.

The roll-up key names (``max_rhat``, ``min_ess_bulk``, ``divergences``,
``worst``) are exactly those ``pipeline/quality.py`` reads.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from pipeline.imputers.summarise import convergence, is_converged

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    import arviz as az

__all__ = [
    "RubinResult",
    "combine_rounds",
    "mixture",
    "rank_descending",
    "round_convergence",
    "rubin",
    "widen_interval",
]


@dataclass(frozen=True)
class RubinResult:
    """Rubin's-rules summary per column.

    Attributes:
        mean: Mean of the round means.
        within: ``W``.
        between: ``B`` (zero when there is a single round).
        total: ``T``.
        fmi: Fraction of missing information, 0 where ``T`` is 0.
        sd: ``sqrt(T)``.
    """

    mean: np.ndarray
    within: np.ndarray
    between: np.ndarray
    total: np.ndarray
    fmi: np.ndarray
    sd: np.ndarray


def mixture(draws_by_round: Sequence[np.ndarray]) -> np.ndarray:
    """Concatenate an equal number of draws from every round.

    Each round contributes ``min`` over rounds of its draw count, taken at
    evenly spaced indices so chain structure is not favoured.

    Args:
        draws_by_round: Arrays of shape ``(T_m, ...)``.

    Returns:
        Array of shape ``(M * min T_m, ...)``.

    Raises:
        ValueError: If no rounds are given.
    """
    if not draws_by_round:
        raise ValueError("no rounds to pool")
    keep = min(d.shape[0] for d in draws_by_round)
    picked = [d[np.linspace(0, d.shape[0] - 1, keep).round().astype(int)] for d in draws_by_round]
    return np.concatenate(picked, axis=0)


def rubin(draws_by_round: Sequence[np.ndarray]) -> RubinResult:
    """Rubin's rules for each column of per-round posterior draws.

    Args:
        draws_by_round: ``M`` arrays of shape ``(T_m, ...)``; columns are
            quantities (generals).

    Returns:
        The summary, each field of shape ``draws.shape[1:]``.
    """
    n_rounds = len(draws_by_round)
    means = np.stack([d.mean(axis=0) for d in draws_by_round])
    within = np.stack([d.var(axis=0, ddof=1) for d in draws_by_round]).mean(axis=0)
    between = means.var(axis=0, ddof=1) if n_rounds > 1 else np.zeros_like(within)
    total = within + (1.0 + 1.0 / n_rounds) * between
    inflation = (1.0 + 1.0 / n_rounds) * between
    fmi = np.divide(inflation, total, out=np.zeros_like(total), where=total > 0)
    return RubinResult(
        mean=means.mean(axis=0),
        within=within,
        between=between,
        total=total,
        fmi=fmi,
        sd=np.sqrt(total),
    )


def widen_interval(
    pooled: np.ndarray, total: np.ndarray, level: float = 0.95
) -> tuple[np.ndarray, np.ndarray]:
    """Mixture quantiles widened about the mixture mean by ``sqrt(T / Var_mixture)``.

    Args:
        pooled: Mixture draws ``(N, ...)``.
        total: Rubin ``T`` per column.
        level: Central interval mass.

    Returns:
        ``(lower, upper)``, each of shape ``pooled.shape[1:]``. A column with
        zero mixture variance is left unwidened.
    """
    tail = (1.0 - level) / 2.0 * 100.0
    lo, hi = np.percentile(pooled, [tail, 100.0 - tail], axis=0)
    centre = pooled.mean(axis=0)
    variance = pooled.var(axis=0, ddof=1)
    factor = np.sqrt(np.divide(total, variance, out=np.ones_like(total), where=variance > 0))
    return centre + factor * (lo - centre), centre + factor * (hi - centre)


def rank_descending(general_ids: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Rank 1 for the largest value, ties broken by smaller ``general_id``.

    Args:
        general_ids: Raw ids, shape ``(G,)``.
        values: Score per general (higher is better).

    Returns:
        Integer ranks starting at 1, aligned with the inputs.
    """
    order = np.lexsort((general_ids, -values))
    ranks = np.empty(values.shape[0], dtype=np.int64)
    ranks[order] = np.arange(1, values.shape[0] + 1)
    return ranks


def round_convergence(idata: az.InferenceData) -> dict[str, Any]:
    """One round's ``{max_rhat, min_ess_bulk, divergences}`` (None where undefined).

    Args:
        idata: The round's fit.

    Returns:
        The roll-up dict.
    """
    return convergence(idata)


def combine_rounds(
    per_round: Sequence[dict[str, Any]],
    *,
    max_rhat: float = 1.02,
    min_ess_bulk: float = 800.0,
    max_divergences: int = 0,
) -> dict[str, Any]:
    """Worst case over rounds, plus the all-rounds-passed flag.

    A round whose metric is None fails: unknown convergence is not convergence.

    Args:
        per_round: Each round's :func:`round_convergence` output.
        max_rhat: Largest acceptable rhat per round.
        min_ess_bulk: Smallest acceptable bulk ESS per round.
        max_divergences: Most divergences tolerated per round.

    Returns:
        ``{"worst": {max_rhat, min_ess_bulk, divergences}, "converged": "true"|"false",
        "rounds_failed": [indices]}``. ``converged`` is a string because
        report reads ``diagnostics->>'converged'``.
    """
    failed = [
        i
        for i, r in enumerate(per_round)
        if not is_converged(
            r, max_rhat=max_rhat, min_ess_bulk=min_ess_bulk, max_divergences=max_divergences
        )
    ]
    rhats = [r["max_rhat"] for r in per_round if r.get("max_rhat") is not None]
    esses = [r["min_ess_bulk"] for r in per_round if r.get("min_ess_bulk") is not None]
    divs = [r["divergences"] for r in per_round if r.get("divergences") is not None]
    # A missing metric must not look healthy in the roll-up either.
    worst = {
        "max_rhat": max(rhats) if len(rhats) == len(per_round) and rhats else None,
        "min_ess_bulk": min(esses) if len(esses) == len(per_round) and esses else None,
        "divergences": int(sum(divs)) if len(divs) == len(per_round) and divs else None,
    }
    return {
        "worst": worst,
        "converged": "true" if per_round and not failed else "false",
        "rounds_failed": failed,
    }
