"""Prior predictive check for the reconcile model.

A parameter the data cannot identify returns its prior, and a prior that
implies absurd armies shows up as absurd estimates only after a long fit. This
draws from the prior of the very model that is about to be sampled and records
two things: how wide the prior on a side's strength is, in troops, and how
much of the prior-predictive mass for the point observations lies outside a
plausible range of army sizes.

It samples with its own explicit seed through PyMC's prior sampler, which
draws from a generator it creates itself. The posterior fit's seed stream is
not consumed or advanced, so adding this check cannot change a published
estimate.

Censored rows are left out of the share: a ``pm.Censored`` prior draw is
clipped at the row's own observed value, which would pull the share towards
zero. Point rows are the unbiased sample. Pymc is imported inside the function
for the same reason as in ``model.py``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import structlog

from pipeline.reconcilers.design import Design

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    import pymc as pm

__all__ = [
    "DEFAULT_DRAWS",
    "DEFAULT_SEED",
    "DEFAULT_TROOP_BOUNDS",
    "prior_predictive_check",
]

logger = structlog.get_logger()

DEFAULT_DRAWS: Final[int] = 500
DEFAULT_SEED: Final[int] = 20261014
DEFAULT_TROOP_BOUNDS: Final[tuple[float, float]] = (50.0, 5_000_000.0)

_QUANTILES: Final[tuple[float, ...]] = (0.01, 0.5, 0.99)
_POINT_VARIABLE: Final[str] = "obs_point"


def prior_predictive_check(
    model: pm.Model,
    design: Design,
    *,
    draws: int = DEFAULT_DRAWS,
    seed: int = DEFAULT_SEED,
    troop_bounds: tuple[float, float] = DEFAULT_TROOP_BOUNDS,
) -> dict[str, Any]:
    """Sample the prior of a built model and summarise what it implies.

    Args:
        model: The unsampled model from :func:`pipeline.reconcilers.model.build_model`.
        design: The design the model was built from.
        draws: Number of prior-predictive draws; 0 or less skips the check.
        seed: Seed for the prior sampler only.
        troop_bounds: ``(low, high)`` plausible army sizes. The outside share is
            reported for troops designs only.

    Returns:
        A JSON-serialisable block: ``draws``, ``seed``, ``strength_quantiles``
        (``p01``, ``p50``, ``p99`` of ``exp(mu_side)``), ``share_outside`` (the
        share of prior-predictive point observations outside the bounds, None
        for casualties or when there are no point rows) and ``bounds``. When
        disabled, ``{"skipped": "disabled"}``; when the sampler raises,
        ``{"status": "failed", "error": ...}``, which the warning gate fails on
        (a check that could not run is not evidence the prior is sensible).
    """
    if draws <= 0:
        return {"skipped": "disabled"}
    import pymc as pm  # noqa: PLC0415 - deliberately deferred; see module docstring

    variables = ["mu_side"]
    has_points = bool(design.point_rows.size)
    if has_points:
        variables.append(_POINT_VARIABLE)
    try:
        with model:
            idata = pm.sample_prior_predictive(
                draws=draws, var_names=variables, random_seed=seed, return_inferencedata=True
            )
    except Exception as exc:
        logger.warning(
            "reconcile_prior_predictive_failed", error=str(exc), error_type=type(exc).__name__
        )
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"[:500]}

    strength = np.exp(np.asarray(idata.prior["mu_side"]).reshape(-1))
    p01, p50, p99 = (float(np.quantile(strength, q)) for q in _QUANTILES)
    block: dict[str, Any] = {
        "draws": draws,
        "seed": seed,
        "strength_quantiles": {
            "p01": _json_float(p01),
            "p50": _json_float(p50),
            "p99": _json_float(p99),
        },
        "share_outside": None,
        "bounds": [troop_bounds[0], troop_bounds[1]],
    }
    if design.quantity == "troops" and has_points:
        observed = np.asarray(idata.prior_predictive[_POINT_VARIABLE]).reshape(-1)
        lo, hi = math.log(troop_bounds[0]), math.log(troop_bounds[1])
        outside = (observed < lo) | (observed > hi)
        block["share_outside"] = float(np.mean(outside))
    return block


def _json_float(value: float) -> float | None:
    """Return ``value`` if finite, else None, so the block stays valid JSON."""
    return value if math.isfinite(value) else None
