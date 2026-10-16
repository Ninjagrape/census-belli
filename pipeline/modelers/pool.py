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

import math
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


# Raw skill = skill_centred + sigma_skill * z-bar, a function of quantities that
# are already gated, so gating it again adds nothing. The common-shift direction
# is carried by z_skill (through z-bar) and by mu_unknown / mu_unattributed,
# which stay gated; this is redundancy, not non-identification. Raw ``skill`` is
# reported for reference only.
_REFERENCE_VARIABLES = ("skill", "z_skill", "skill_centred")


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _variable_convergence(posterior: Any, name: str) -> dict[str, float | None]:
    """``{max_rhat, min_ess_bulk}`` of one variable (None where undefined)."""
    import arviz as az  # noqa: PLC0415 - deferred like the imputers' import

    subset = posterior[[name]]
    rhat = np.asarray(az.rhat(subset)[name].values, dtype=float)
    ess = np.asarray(az.ess(subset, method="bulk")[name].values, dtype=float)
    return {
        "max_rhat": _finite(float(np.nanmax(rhat))) if np.isfinite(rhat).any() else None,
        "min_ess_bulk": _finite(float(np.nanmin(ess))) if np.isfinite(ess).any() else None,
    }


def round_convergence(idata: az.InferenceData) -> dict[str, Any]:
    """One round's ``{max_rhat, min_ess_bulk, divergences}`` (None where undefined).

    The roll-up covers every posterior variable except raw ``skill``, plus
    ``skill_centred`` (skill minus its per-draw mean across generals). Raw
    ``skill`` is left out because it is a function of gated quantities (see
    ``_REFERENCE_VARIABLES``). ``reference`` repeats rhat and ESS for raw
    ``skill``, ``z_skill`` and ``skill_centred`` separately, so a gate miss can
    be traced.

    Args:
        idata: The round's fit.

    Returns:
        The roll-up dict, with a ``reference`` entry per skill-like variable.
    """
    import arviz as az  # noqa: PLC0415 - deferred like the imputers' import

    posterior = idata.posterior
    if "skill" not in posterior or "general" not in posterior["skill"].dims:
        return convergence(idata)
    centred = posterior["skill"] - posterior["skill"].mean("general")
    with_centred = posterior.assign(skill_centred=centred)
    groups: dict[str, Any] = {"posterior": with_centred.drop_vars("skill")}
    if "sample_stats" in idata.groups():
        groups["sample_stats"] = idata.sample_stats
    gated = convergence(az.InferenceData(**groups))
    gated["reference"] = {
        name: _variable_convergence(with_centred, name) for name in _REFERENCE_VARIABLES
    }
    return gated


def _worst_of(values: Sequence[float | None], pick: Any) -> float | None:
    """The worst value over rounds, None when any round has none."""
    return None if any(v is None for v in values) else float(pick(values))


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
    out: dict[str, Any] = {
        "worst": worst,
        "converged": "true" if per_round and not failed else "false",
        "rounds_failed": failed,
    }
    if per_round and all("reference" in r for r in per_round):
        out["reference"] = {
            name: {
                "max_rhat": _worst_of([r["reference"][name]["max_rhat"] for r in per_round], max),
                "min_ess_bulk": _worst_of(
                    [r["reference"][name]["min_ess_bulk"] for r in per_round], min
                ),
            }
            for name in per_round[0]["reference"]
        }
    return out
