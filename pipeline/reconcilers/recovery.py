"""Does the fit give back the reports? Recovery diagnostics for the troop fits.

Convergence statistics cannot see a fit that shrinks every lone report toward
the corpus mean: the 2026-10 corpus fit converged cleanly with a singleton slope
of 0.61 and a within-battle log-ratio slope of 0.60
(``docs/handover-log/31-troop-numbers-diagnosis-2026-10-22.md``). These
functions compare the fitted estimates with the reports directly, and
:func:`recovery_diagnostics` packs them, with the values the quality gates read,
into the run's ``model_runs.diagnostics``.

Pure numpy: no database, no pymc. Everything takes the design and plain mappings,
so it runs on a published or a regime-neutral fit alike (the neutral fit is
shifted by a regime-neutral offset, which a slope ignores and the absolute
ratios do not: read those on the published fit).

## Definitions

- ``report`` of a side: the mean log value of its point rows (a side with only
  bound rows has none and is left out of every statistic here).
- **Singleton**: a side with exactly one design row, a point row.
- **Within-battle log-ratio slope**: over battles with exactly two sides that
  each have a point row, the least-squares slope through the origin of the
  estimated log ratio on the reported log ratio. Through the origin because the
  log ratio is antisymmetric (which side is "A" is arbitrary), so a fit with an
  intercept would depend on that choice. Sides are ordered by side id.
- **Ratio** of a group: the geometric mean of ``estimate / report`` over its
  sides.
- **Between-battle variance check**: the second moment of the reported log ratio
  across those battles, against its posterior predictive distribution under the
  fitted noise (a Student-t when the fit has ``nu``). A fit whose noise is
  larger than the corpus can bear predicts a spread the reports do not show.

## Gates skip, loudly

A statistic over too few sides or battles is noise. Each gate value is None
below its minimum and the reason is recorded in ``gates_skipped``, so the SQL
gate that reads None passes and the diagnostics say why, instead of a small
fixture silently passing or failing on a number that means nothing.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import structlog

from pipeline.reconcilers.design import Design

__all__ = [
    "BattlePair",
    "RecoverySettings",
    "battle_pairs",
    "between_battle_ppc",
    "group_ratio",
    "recovery_diagnostics",
    "side_report_logs",
    "singleton_points",
    "slope_through_origin",
    "within_battle_slope",
]

logger = structlog.get_logger()

LARGE_ARMY: Final[float] = 50_000.0
SMALL_ARMY: Final[float] = 10_000.0
MODERN_FROM_YEAR: Final[int] = 1800
_MIN_SLOPE_POINTS: Final[int] = 3


@dataclass(frozen=True)
class RecoverySettings:
    """Minimum sample sizes and thresholds for the recovery gates.

    Attributes:
        min_sides: Fewest singleton sides (singleton slope) or sides (spread
            ratio) for those gates to apply.
        min_battles: Fewest two-sided battles for the within-battle slope and
            the between-battle variance check to apply.
        min_post1800_sides: Fewest post-1800 sides reported above 50,000 for the
            ratio band to apply.
        ratio_band: ``(lo, hi)`` the post-1800 geometric ratio should sit in.
        ppc_draws: Posterior draws used by the variance check.
        ppc_seed: Seed of the check's own generator, so the fit's is untouched.
    """

    min_sides: int = 10
    min_battles: int = 10
    min_post1800_sides: int = 5
    ratio_band: tuple[float, float] = (0.85, 1.15)
    ppc_draws: int = 500
    ppc_seed: int = 20261022

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> RecoverySettings:
        """Read the settings from an agent spec's ``params``.

        Args:
            params: The spec's ``params`` mapping.

        Returns:
            The settings, defaults for anything absent.
        """
        d = cls()
        return cls(
            min_sides=int(params.get("recovery_min_sides", d.min_sides)),
            min_battles=int(params.get("recovery_min_battles", d.min_battles)),
            min_post1800_sides=int(params.get("recovery_min_post1800_sides", d.min_post1800_sides)),
            ratio_band=(
                float(params.get("post1800_ratio_band_lo", d.ratio_band[0])),
                float(params.get("post1800_ratio_band_hi", d.ratio_band[1])),
            ),
            ppc_draws=int(params.get("recovery_ppc_draws", d.ppc_draws)),
            ppc_seed=int(params.get("recovery_ppc_seed", d.ppc_seed)),
        )


@dataclass(frozen=True)
class BattlePair:
    """The two point-reported sides of one battle.

    Attributes:
        battle_id: The battle.
        side_a: The lower side id.
        side_b: The higher side id.
        rows_a: Design rows of ``side_a``'s point reports.
        rows_b: Design rows of ``side_b``'s point reports.
    """

    battle_id: int
    side_a: int
    side_b: int
    rows_a: np.ndarray
    rows_b: np.ndarray


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def side_report_logs(design: Design) -> dict[int, float]:
    """Mean log value of each side's point rows.

    Args:
        design: The design the fit used.

    Returns:
        ``side_id`` to mean log report; a side with no point row is absent.
    """
    sums: dict[int, float] = defaultdict(float)
    counts: dict[int, int] = defaultdict(int)
    for row in design.point_rows:
        side_id = design.side_ids[int(design.side_index[row])]
        sums[side_id] += float(design.log_y[row])
        counts[side_id] += 1
    return {side_id: sums[side_id] / counts[side_id] for side_id in sums}


def singleton_points(
    design: Design, log_estimates: Mapping[int, float]
) -> tuple[list[float], list[float]]:
    """Report and estimate of every singleton side that has an estimate.

    Args:
        design: The design the fit used.
        log_estimates: ``side_id`` to the log of the estimate.

    Returns:
        ``(reports, estimates)``, parallel lists of log values.
    """
    counts = np.bincount(design.side_index, minlength=design.n_sides)
    xs: list[float] = []
    ys: list[float] = []
    for row in design.point_rows:
        position = int(design.side_index[row])
        side_id = design.side_ids[position]
        if counts[position] != 1 or side_id not in log_estimates:
            continue
        xs.append(float(design.log_y[row]))
        ys.append(log_estimates[side_id])
    return xs, ys


def slope_through_origin(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Least-squares slope of ``y`` on ``x`` through the origin.

    Args:
        x: Regressor.
        y: Response, same length.

    Returns:
        ``sum(x*y) / sum(x*x)``, or None when fewer than three points exist or
        ``x`` is identically zero.
    """
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    if xs.size < _MIN_SLOPE_POINTS:
        return None
    denominator = float(np.sum(xs * xs))
    if denominator <= 0.0:
        return None
    return _finite(float(np.sum(xs * ys)) / denominator)


def battle_pairs(design: Design, battle_of_side: Mapping[int, int]) -> list[BattlePair]:
    """Battles with exactly two sides that each have a point report.

    Args:
        design: The design the fit used.
        battle_of_side: ``side_id`` to ``battle_id``.

    Returns:
        One :class:`BattlePair` per such battle, ordered by battle id.
    """
    rows_of_side: dict[int, list[int]] = defaultdict(list)
    for row in design.point_rows:
        rows_of_side[design.side_ids[int(design.side_index[row])]].append(int(row))
    sides_of_battle: dict[int, list[int]] = defaultdict(list)
    for side_id in rows_of_side:
        if side_id in battle_of_side:
            sides_of_battle[battle_of_side[side_id]].append(side_id)
    pairs: list[BattlePair] = []
    for battle_id in sorted(sides_of_battle):
        sides = sorted(sides_of_battle[battle_id])
        if len(sides) != 2:
            continue
        a, b = sides
        pairs.append(
            BattlePair(
                battle_id,
                a,
                b,
                np.asarray(rows_of_side[a], dtype=int),
                np.asarray(rows_of_side[b], dtype=int),
            )
        )
    return pairs


def within_battle_slope(
    pairs: Sequence[BattlePair],
    report_logs: Mapping[int, float],
    log_estimates: Mapping[int, float],
) -> tuple[float | None, int]:
    """Slope of the estimated log ratio on the reported one, over two-sided battles.

    Args:
        pairs: From :func:`battle_pairs`.
        report_logs: From :func:`side_report_logs`.
        log_estimates: ``side_id`` to the log of the estimate.

    Returns:
        ``(slope, n_battles)``; the slope is None below three battles.
    """
    xs: list[float] = []
    ys: list[float] = []
    for pair in pairs:
        if pair.side_a not in log_estimates or pair.side_b not in log_estimates:
            continue
        xs.append(report_logs[pair.side_a] - report_logs[pair.side_b])
        ys.append(log_estimates[pair.side_a] - log_estimates[pair.side_b])
    return slope_through_origin(xs, ys), len(xs)


def group_ratio(
    report_logs: Mapping[int, float],
    log_estimates: Mapping[int, float],
    side_ids: Sequence[int],
) -> tuple[float | None, int]:
    """Geometric mean of ``estimate / report`` over the given sides.

    Args:
        report_logs: From :func:`side_report_logs`.
        log_estimates: ``side_id`` to the log of the estimate.
        side_ids: The group.

    Returns:
        ``(ratio, n)``; the ratio is None for an empty group.
    """
    gaps = [
        log_estimates[s] - report_logs[s]
        for s in side_ids
        if s in log_estimates and s in report_logs
    ]
    if not gaps:
        return None, 0
    return _finite(math.exp(float(np.mean(gaps)))), len(gaps)


def between_battle_ppc(
    pairs: Sequence[BattlePair],
    y_obs: np.ndarray,
    eta: np.ndarray,
    var: np.ndarray,
    rows: np.ndarray,
    nu: np.ndarray | None,
    *,
    seed: int,
) -> dict[str, Any]:
    """Posterior predictive check of the between-battle spread of the reported log ratio.

    The statistic is the mean over battles of the squared reported log ratio.
    Each posterior draw regenerates every point report of those battles from the
    fitted location and noise, forms the same ratios, and the observed statistic
    is placed among the replicates.

    Args:
        pairs: From :func:`battle_pairs`.
        y_obs: Observed log report of every design row, ``(n_obs,)``.
        eta: Fitted location of ``rows`` per draw, ``(n_draws, len(rows))``.
        var: Fitted variance of ``rows`` per draw, same shape.
        rows: The design rows ``eta`` and ``var`` describe.
        nu: Student-t degrees of freedom per draw, ``(n_draws,)``, or None for a
            Normal likelihood.
        seed: Seed of the replication noise.

    Returns:
        ``observed``, ``replicated_median``, ``p_value`` (share of replicates at
        or above the observed statistic) and ``extremity`` (the smaller of that
        and its complement; near 0 means the observed spread is far in a tail).
    """
    position = {int(r): i for i, r in enumerate(rows)}
    rng = np.random.default_rng(seed)
    n_draws = eta.shape[0]
    noise = rng.standard_normal(eta.shape)
    if nu is not None:
        scale = np.sqrt(np.maximum(rng.chisquare(nu[:, None], size=eta.shape), 1e-12) / nu[:, None])
        noise = noise / scale
    replicated = eta + np.sqrt(var) * noise

    def side_mean(matrix: np.ndarray, side_rows: np.ndarray) -> np.ndarray:
        columns = [position[int(r)] for r in side_rows]
        mean: np.ndarray = matrix[:, columns].mean(axis=1)
        return mean

    observed_x = np.array(
        [float(np.mean(y_obs[p.rows_a]) - np.mean(y_obs[p.rows_b])) for p in pairs]
    )
    replicated_x = np.column_stack(
        [side_mean(replicated, p.rows_a) - side_mean(replicated, p.rows_b) for p in pairs]
    )
    observed = float(np.mean(observed_x**2))
    statistic = np.mean(replicated_x**2, axis=1)
    p_value = float(np.mean(statistic >= observed))
    return {
        "observed": _finite(observed),
        "replicated_median": _finite(float(np.median(statistic))),
        "p_value": _finite(p_value),
        "extremity": _finite(min(p_value, 1.0 - p_value)),
        "n_battles": len(pairs),
        "n_draws": int(n_draws),
    }


def recovery_diagnostics(
    design: Design,
    log_estimates: Mapping[int, float],
    battle_of_side: Mapping[int, int],
    year_of_side: Mapping[int, int | None],
    *,
    settings: RecoverySettings,
    ancient_cutoff_year: int,
    singleton_slope: float | None,
    spread_ratio: float | None,
    ppc: dict[str, Any] | None,
) -> dict[str, Any]:
    """Pack the recovery statistics and the values the quality gates read.

    Args:
        design: The design the fit used.
        log_estimates: ``side_id`` to the log of the estimate.
        battle_of_side: ``side_id`` to ``battle_id``.
        year_of_side: ``side_id`` to the battle's astronomical year (None if
            undated).
        settings: Minimum sizes and thresholds.
        ancient_cutoff_year: The stage's ancient cutoff, for the ancient group.
        singleton_slope: From :func:`pipeline.reconcilers.summarise.singleton_slope`.
        spread_ratio: From :func:`pipeline.reconcilers.summarise.spread_ratio`.
        ppc: From :func:`between_battle_ppc`, or None when it could not run.

    Returns:
        A JSON-serialisable mapping. ``gate_*`` entries are the numbers the gates
        compare (None when skipped, with the reason in ``gates_skipped``).
    """
    report_logs = side_report_logs(design)
    pairs = battle_pairs(design, battle_of_side)
    wb_slope, n_pairs = within_battle_slope(pairs, report_logs, log_estimates)

    counts = np.bincount(design.side_index, minlength=design.n_sides)
    n_singletons = len(singleton_points(design, log_estimates)[0])
    n_spread = sum(1 for s in report_logs if s in log_estimates)

    def sides_where(predicate: Any) -> list[int]:
        return [s for s, x in report_logs.items() if predicate(s, math.exp(x))]

    def year(s: int) -> int | None:
        return year_of_side.get(s)

    modern_large = sides_where(
        lambda s, v: year(s) is not None and (year(s) or 0) >= MODERN_FROM_YEAR and v > LARGE_ARMY
    )
    ancient_large = sides_where(
        lambda s, v: year(s) is not None and (year(s) or 0) < ancient_cutoff_year and v > LARGE_ARMY
    )
    small = sides_where(lambda _s, v: v < SMALL_ARMY)
    ratio_modern, n_modern = group_ratio(report_logs, log_estimates, modern_large)
    ratio_ancient, n_ancient = group_ratio(report_logs, log_estimates, ancient_large)
    ratio_small, n_small = group_ratio(report_logs, log_estimates, small)

    skipped: dict[str, str] = {}
    gate_singleton = singleton_slope if n_singletons >= settings.min_sides else None
    if gate_singleton is None:
        skipped["singleton_slope"] = (
            f"{n_singletons} singleton sides, need {settings.min_sides}"
            if n_singletons < settings.min_sides
            else "slope undefined"
        )
    gate_spread = spread_ratio if n_spread >= settings.min_sides else None
    if gate_spread is None:
        skipped["spread_ratio"] = (
            f"{n_spread} point-reported sides, need {settings.min_sides}"
            if n_spread < settings.min_sides
            else "ratio undefined"
        )
    gate_wb = wb_slope if n_pairs >= settings.min_battles else None
    if gate_wb is None:
        skipped["within_battle_slope"] = (
            f"{n_pairs} two-sided battles, need {settings.min_battles}"
            if n_pairs < settings.min_battles
            else "slope undefined"
        )
    gate_band: int | None = None
    if ratio_modern is not None and n_modern >= settings.min_post1800_sides:
        lo, hi = settings.ratio_band
        gate_band = 0 if lo <= ratio_modern <= hi else 1
    else:
        skipped["post1800_ratio"] = (
            f"{n_modern} post-{MODERN_FROM_YEAR} sides reported above {LARGE_ARMY:g}, "
            f"need {settings.min_post1800_sides}"
        )
    gate_ppc: float | None = None
    if ppc is not None and n_pairs >= settings.min_battles:
        gate_ppc = ppc.get("extremity")
    if gate_ppc is None:
        skipped["between_battle_ppc"] = (
            f"{n_pairs} two-sided battles, need {settings.min_battles}"
            if n_pairs < settings.min_battles
            else "check did not run"
        )
    for gate, reason in skipped.items():
        logger.info("reconcile_recovery_gate_skipped", gate=gate, reason=reason)

    return {
        "singleton_slope": singleton_slope,
        "n_singletons": n_singletons,
        "spread_ratio": spread_ratio,
        "n_spread_sides": n_spread,
        "within_battle_slope": wb_slope,
        "n_two_sided_battles": n_pairs,
        "post1800_large_ratio": ratio_modern,
        "n_post1800_large": n_modern,
        "ancient_large_ratio": ratio_ancient,
        "n_ancient_large": n_ancient,
        "small_ratio": ratio_small,
        "n_small": n_small,
        "n_rows_per_side_max": int(counts.max()) if counts.size else 0,
        "between_battle_ppc": ppc,
        "gate_singleton_slope": gate_singleton,
        "gate_spread_ratio": gate_spread,
        "gate_within_battle_slope": gate_wb,
        "gate_post1800_band_violation": gate_band,
        "gate_between_battle_extremity": gate_ppc,
        "gates_skipped": skipped,
    }
