"""Posterior predictive checks by stratum and by general (report only, no gate)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Final

import numpy as np

from pipeline.modelers.checks import CATEGORY_NAMES, top_indices
from pipeline.modelers.model import PosteriorDraws
from pipeline.modelers.ordinal import N_LEVELS, category_probs, linear_predictor
from pipeline.modelers.records import ModelDesign

__all__ = ["era_bucket", "posterior_predictive_report", "simulate_levels"]

ANCIENT_BEFORE: Final[float] = 500.0
MEDIEVAL_BEFORE: Final[float] = 1500.0
EARLY_MODERN_BEFORE: Final[float] = 1800.0
UNDATED: Final[str] = "undated"
UNKNOWN_TYPE: Final[str] = "unknown"
# A stratum with fewer exact-outcome battles has intervals too wide, and
# sampling noise in its frequencies too large, to flag anything.
MIN_STRATUM_BATTLES: Final[int] = 10
GENERAL_INTERVAL: Final[float] = 0.90
WIN_LEVEL: Final[int] = 3
LOSS_LEVEL: Final[int] = 1


def era_bucket(year: float) -> str:
    """Era of a battle year (astronomical, BC negative).

    Args:
        year: The year, NaN if undated.

    Returns:
        ``ancient`` (before 500), ``medieval`` (before 1500), ``early_modern``
        (before 1800), ``modern``, or ``undated``.
    """
    if math.isnan(year):
        return UNDATED
    if year < ANCIENT_BEFORE:
        return "ancient"
    if year < MEDIEVAL_BEFORE:
        return "medieval"
    if year < EARLY_MODERN_BEFORE:
        return "early_modern"
    return "modern"


def simulate_levels(
    design: ModelDesign,
    draws: PosteriorDraws,
    x_diff: np.ndarray,
    *,
    n_sim: int,
    seed: int,
) -> np.ndarray:
    """Simulate every battle's outcome level from the posterior.

    The same construction as
    :func:`pipeline.modelers.checks.posterior_predictive_category_check`: one
    simulated outcome per battle for each of ``n_sim`` evenly spaced draws.

    Args:
        design: The design.
        draws: One round's flattened draws.
        x_diff: ``(n_battles,)`` fixed difference, or ``(T, n_battles)``.
        n_sim: Number of draws to simulate from.
        seed: Generator seed.

    Returns:
        Integer levels 0..4, shape ``(n_sim, n_battles)``.
    """
    rng = np.random.default_rng(seed)
    picks = np.linspace(0, draws.n_draws - 1, min(n_sim, draws.n_draws)).round().astype(int)
    x = x_diff if x_diff.ndim == 1 else x_diff[picks]
    eta = linear_predictor(
        design.side_a,
        design.side_b,
        draws.side_scores[picks],
        x,
        design.defender_diff,
        draws.beta_force[picks],
        draws.beta_def[picks],
    )
    probs = category_probs(eta, draws.c1[picks, None], draws.c2[picks, None])
    u = rng.random(eta.shape)[..., None]
    return np.asarray(np.minimum((u > np.cumsum(probs, axis=-1)).sum(axis=-1), N_LEVELS - 1))


def _stratum_check(
    design: ModelDesign, sim: np.ndarray, mask: np.ndarray, level: float
) -> dict[str, Any]:
    """Observed level frequencies against the predictive interval, exact battles only."""
    exact = (design.k_lo == design.k_hi) & mask
    n = int(exact.sum())
    if n < MIN_STRATUM_BATTLES:
        return {"n_exact": n, "evaluable": False, "outside": []}
    observed = np.bincount(design.k_lo[exact], minlength=N_LEVELS) / n
    freq = np.stack([(sim[:, exact] == k).mean(axis=1) for k in range(N_LEVELS)], axis=1)
    tail = (1.0 - level) / 2.0 * 100.0
    lo, hi = np.percentile(freq, [tail, 100.0 - tail], axis=0)
    outside = [CATEGORY_NAMES[k] for k in range(N_LEVELS) if not lo[k] <= observed[k] <= hi[k]]
    return {
        "n_exact": n,
        "evaluable": True,
        "observed": dict(zip(CATEGORY_NAMES, observed.tolist(), strict=True)),
        "lower": lo.tolist(),
        "upper": hi.tolist(),
        "outside": outside,
    }


def _stratified(
    design: ModelDesign, sim: np.ndarray, labels: np.ndarray, level: float
) -> dict[str, Any]:
    return {
        str(label): _stratum_check(design, sim, labels == label, level)
        for label in sorted(set(labels.tolist()))
    }


def _decided_shares(
    design: ModelDesign, sim: np.ndarray, general_index: int
) -> dict[str, Any] | None:
    """Observed and simulated win share of one general over decided battles."""
    entries = design.cmd_general == general_index
    sides = np.unique(design.cmd_side[entries])
    battles = sides // 2
    is_a = design.side_a[battles] == sides
    won = np.where(is_a, design.k_lo[battles] >= WIN_LEVEL, design.k_hi[battles] <= LOSS_LEVEL)
    lost = np.where(is_a, design.k_hi[battles] <= LOSS_LEVEL, design.k_lo[battles] >= WIN_LEVEL)
    decided = won | lost
    n = int(decided.sum())
    if n == 0:
        return None
    levels = sim[:, battles[decided]]
    a_side = is_a[decided]
    sim_win = np.where(a_side, levels >= WIN_LEVEL, levels <= LOSS_LEVEL).sum(axis=1)
    sim_loss = np.where(a_side, levels <= LOSS_LEVEL, levels >= WIN_LEVEL).sum(axis=1)
    total = sim_win + sim_loss
    share = np.divide(sim_win, total, out=np.full(total.shape, np.nan), where=total > 0)
    tail = (1.0 - GENERAL_INTERVAL) / 2.0 * 100.0
    lo, hi = np.nanpercentile(share, [tail, 100.0 - tail])
    observed = float(won[decided].sum()) / n
    return {
        "n_decided": n,
        "observed_win_share": observed,
        "predictive_lo": float(lo),
        "predictive_hi": float(hi),
        "flagged": bool(not lo <= observed <= hi),
    }


def _general_checks(
    design: ModelDesign,
    sim: np.ndarray,
    score: np.ndarray,
    names: Mapping[int, str],
    top_n: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for g in top_indices(design.general_ids, score, top_n):
        stats = _decided_shares(design, sim, int(g))
        if stats is None:
            continue
        general_id = int(design.general_ids[g])
        rows.append({"general_id": general_id, "name": names.get(general_id, ""), **stats})
    return rows


def posterior_predictive_report(
    design: ModelDesign,
    draws: PosteriorDraws,
    x_diff: np.ndarray,
    *,
    battle_types: Mapping[int, str | None],
    score: np.ndarray,
    names: Mapping[int, str],
    top_n: int,
    n_sim: int,
    level: float,
    seed: int,
) -> dict[str, Any]:
    """Run the stratified category check and the per-general win-rate check.

    Both read the same simulation, from one round's draws and its fixed
    ``x_diff``, so they describe fit to the data the model was fitted on.

    Args:
        design: The design.
        draws: One round's flattened draws.
        x_diff: That round's ``x_A - x_B``.
        battle_types: ``battle_id`` to ``battle_type``.
        score: Baseline ranking score per general, aligned with
            ``design.general_ids``; picks the top generals.
        names: ``general_id`` to name.
        top_n: Generals checked individually.
        n_sim: Draws to simulate from.
        level: Predictive interval mass of the category check.
        seed: Generator seed.

    Returns:
        ``category`` strata (``overall``, ``era``, ``battle_type``),
        ``per_general`` rows, and ``flagged`` listing the strata with
        levels outside their interval and the generals outside their 90%
        interval.
    """
    sim = simulate_levels(design, draws, x_diff, n_sim=n_sim, seed=seed)
    eras = np.array([era_bucket(float(y)) for y in design.year])
    types = np.array([battle_types.get(int(b)) or UNKNOWN_TYPE for b in design.battle_ids])
    category = {
        "overall": _stratum_check(design, sim, np.ones(design.n_battles, dtype=bool), level),
        "era": _stratified(design, sim, eras, level),
        "battle_type": _stratified(design, sim, types, level),
    }
    per_general = _general_checks(design, sim, score, names, top_n)
    groups = {
        "overall": {"all": category["overall"]},
        "era": category["era"],
        "battle_type": category["battle_type"],
    }
    flagged_strata = [
        {"group": group, "stratum": label, "outside": res["outside"]}
        for group, strata in groups.items()
        for label, res in strata.items()
        if res["outside"]
    ]
    return {
        "level": level,
        "general_interval": GENERAL_INTERVAL,
        "n_sim": int(sim.shape[0]),
        "min_stratum_battles": MIN_STRATUM_BATTLES,
        "category": category,
        "per_general": per_general,
        "flagged": {
            "strata": flagged_strata,
            "generals": [r["name"] or r["general_id"] for r in per_general if r["flagged"]],
        },
    }
