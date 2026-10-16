"""Pure table logic: headline versus under-determined, and rank overlap.

Nothing here ranks. Ranks are the model stage's stored values; this module
only filters and orders by them and flags where neighbouring intervals overlap.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from pipeline.reporters.records import GeneralEstimate, PooledDraws, RubinInfo

__all__ = [
    "RankedGeneral",
    "has_stored_war_interval",
    "overlap_share",
    "split_generals",
    "top_headline",
    "war_interval",
    "z_for_level",
]


@dataclass(frozen=True)
class RankedGeneral:
    """A general as the tables show it.

    Attributes:
        estimate: The stored estimate.
        rubin: Model-stage per-general columns (empty when unavailable).
        is_headline: Whether the general has enough credited battles to be ranked.
        rank: The stored rank for headline generals; None for under-determined
            ones, whose rank is never shown.
        war_lo: Lower end of the WAR interval (stored posterior interval, or
            the normal approximation for a run that stored none).
        war_hi: Upper end of the WAR interval.
        war_interval_approx: Whether ``war_lo``/``war_hi`` are the normal
            approximation rather than the stored posterior interval.
        p_above_next: P(score above the next headline general's), recomputed from
            the pooled draws when available, else the stored value.
        overlaps_next: Whether the interval overlaps the next listed headline
            general's by more than the threshold. None when there is no next
            listed general, when either interval is missing, or when the flag
            is switched off; never False for "could not compare".
    """

    estimate: GeneralEstimate
    rubin: RubinInfo
    is_headline: bool
    rank: int | None
    war_lo: float | None
    war_hi: float | None
    overlaps_next: bool | None = None
    war_interval_approx: bool = False
    p_above_next: float | None = None


def z_for_level(ci_level: float) -> float:
    """The two-sided normal quantile for a central interval level.

    Args:
        ci_level: A level in (0, 1).

    Returns:
        The z multiplier, 1.96 for 0.95.
    """
    return NormalDist().inv_cdf(0.5 + ci_level / 2.0)


def has_stored_war_interval(estimate: GeneralEstimate) -> bool:
    """Whether the model stage stored a posterior WAR interval for this general."""
    return estimate.war_ci_lo is not None and estimate.war_ci_hi is not None


def war_interval(estimate: GeneralEstimate, z: float) -> tuple[float, float] | None:
    """The stored posterior WAR interval, else ``total_war +/- z * war_sd``.

    The normal approximation is only a fallback for runs from before the
    model stage stored the interval; it ignores skew and between-general
    correlation.

    Args:
        estimate: The stored estimate.
        z: The multiplier from :func:`z_for_level`.

    Returns:
        The interval, or None when neither is available.
    """
    if estimate.war_ci_lo is not None and estimate.war_ci_hi is not None:
        return estimate.war_ci_lo, estimate.war_ci_hi
    if estimate.total_war is None or estimate.war_sd is None:
        return None
    half = z * estimate.war_sd
    return estimate.total_war - half, estimate.total_war + half


def overlap_share(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Overlap length as a share of the narrower interval.

    A zero-width interval inside the other counts as full overlap; outside it,
    none.

    Args:
        a: One ``(lo, hi)`` interval.
        b: The other.

    Returns:
        A share in [0, 1].
    """
    narrow = min(a[1] - a[0], b[1] - b[0])
    overlap = min(a[1], b[1]) - max(a[0], b[0])
    if overlap < 0:
        return 0.0
    if narrow <= 0:
        return 1.0
    return min(1.0, overlap / narrow)


def _flag_overlaps(
    headline: Sequence[tuple[GeneralEstimate, tuple[float, float] | None]], threshold: float
) -> list[bool | None]:
    """For each headline row, whether it overlaps the next listed row's interval.

    The last row, and any row where either interval is missing, gets None.
    """
    flags: list[bool | None] = [None] * len(headline)
    for i in range(len(headline) - 1):
        this, nxt = headline[i][1], headline[i + 1][1]
        if this is not None and nxt is not None:
            flags[i] = overlap_share(this, nxt) > threshold
    return flags


def _order(estimate: GeneralEstimate) -> tuple[int, int, int]:
    """Stored-rank order, unranked last, ties by id."""
    unranked = estimate.rank is None
    return (1 if unranked else 0, 0 if unranked else int(estimate.rank or 0), estimate.general_id)


def split_generals(
    estimates: Sequence[GeneralEstimate],
    rubin: Mapping[int, RubinInfo],
    *,
    min_battles: int,
    z: float,
    overlap_threshold: float,
    show_overlap: bool = True,
    pooled: PooledDraws | None = None,
) -> tuple[tuple[RankedGeneral, ...], tuple[RankedGeneral, ...]]:
    """Split generals into the headline ranking and the under-determined list.

    Headline generals have at least ``min_battles`` credited battles and keep
    their stored rank and order. Under-determined ones are listed by credited
    battles, then name, with no rank.

    Args:
        estimates: Every stored estimate of the run.
        rubin: Per-general model-stage columns by general id.
        min_battles: Fewest credited battles for a headline rank.
        z: The multiplier for the WAR interval.
        overlap_threshold: Overlap share above which neighbours are flagged.
        show_overlap: Whether to flag overlaps at all.
        pooled: Pooled draws; when given, P(above next) is recomputed against
            the next headline general rather than read from the stored column,
            which was computed over every general including hidden ones.

    Returns:
        ``(headline, under_determined)``.
    """
    eligible = sorted((e for e in estimates if e.n_battles >= min_battles), key=_order)
    intervals = [(e, war_interval(e, z)) for e in eligible]
    flags: list[bool | None] = (
        _flag_overlaps(intervals, overlap_threshold) if show_overlap else [None] * len(eligible)
    )
    p_next = _headline_p_next([e for e, _ in intervals], pooled)
    headline = tuple(
        RankedGeneral(
            estimate=e,
            rubin=rubin.get(e.general_id, RubinInfo()),
            is_headline=True,
            rank=e.rank,
            war_lo=iv[0] if iv else None,
            war_hi=iv[1] if iv else None,
            overlaps_next=flag,
            war_interval_approx=iv is not None and not has_stored_war_interval(e),
            p_above_next=pn,
        )
        for (e, iv), flag, pn in zip(intervals, flags, p_next, strict=True)
    )
    thin = sorted((e for e in estimates if e.n_battles < min_battles), key=_thin_order)
    under = tuple(
        RankedGeneral(
            estimate=e,
            rubin=rubin.get(e.general_id, RubinInfo()),
            is_headline=False,
            rank=None,
            war_lo=None,
            war_hi=None,
        )
        for e in thin
    )
    return headline, under


def _headline_p_next(
    eligible: Sequence[GeneralEstimate], pooled: PooledDraws | None
) -> list[float | None]:
    """P(i above the next eligible general), draw by draw; stored values without draws."""
    if pooled is None:
        return [e.p_above_next_rank for e in eligible]
    out: list[float | None] = [None] * len(eligible)
    for i in range(len(eligible) - 1):
        upper = pooled.draws.get(eligible[i].general_id)
        lower = pooled.draws.get(eligible[i + 1].general_id)
        if upper is not None and lower is not None:
            out[i] = float(np.mean(upper > lower))
    return out


def _thin_order(estimate: GeneralEstimate) -> tuple[int, str, int]:
    return -estimate.n_battles, estimate.name.casefold(), estimate.general_id


def top_headline(headline: Sequence[RankedGeneral], top_n: int) -> tuple[RankedGeneral, ...]:
    """The first ``top_n`` headline generals in stored-rank order.

    Args:
        headline: The headline list from :func:`split_generals`.
        top_n: How many to keep.

    Returns:
        The leading rows.
    """
    return tuple(headline[:top_n])
