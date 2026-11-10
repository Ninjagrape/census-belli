"""Turn :class:`CoverageStats` into the six panels of "Data coverage and caveats".

One plain-data dictionary feeds both the matplotlib PNGs and the explorer's
plotly tab, so the text, the report figures and the explorer share one source.
A panel whose data is missing is left out and logged; nothing here raises.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Final

import structlog

from pipeline.reporters.coverage import (
    ERA_ORDER,
    REPORT_ERAS,
    CoverageStats,
    era_label,
)
from pipeline.reporters.coverage_series import TIMELINE_BIN, credit_by_general
from pipeline.reporters.tables import RankedGeneral

__all__ = [
    "ATTRIBUTION_METHODS",
    "CONTRACTION_LINE",
    "CREDIT_TOP",
    "PANEL_FILES",
    "build_panels",
    "contraction",
    "expected_figures",
]

logger = structlog.get_logger()

CREDIT_TOP: Final[int] = 25
CONTRACTION_LINE: Final[float] = 0.25
OTHER_METHOD: Final[str] = "other"
ATTRIBUTION_METHODS: Final[tuple[str, ...]] = ("llm", "rule_single", "default_split", "manual")
# Panel key to the PNG it is drawn into.
PANEL_FILES: Final[dict[str, str]] = {
    "timeline": "coverage_timeline.png",
    "components": "coverage_components.png",
    "credit_source": "coverage_credit_source.png",
    "info_vs_rank": "coverage_info_vs_rank.png",
    "troop_provenance": "coverage_troop_provenance.png",
    "fame_bias": "coverage_fame_bias.png",
}


def contraction(skill_sd: float, sigma_skill: float) -> float | None:
    """How far the data pulled a general's skill in from the prior: 1 - (sd / sigma)^2.

    Args:
        skill_sd: The stored posterior sd of the general's skill.
        sigma_skill: The model's skill scale.

    Returns:
        The contraction, or None when either input is not a usable number.
    """
    if not (math.isfinite(skill_sd) and math.isfinite(sigma_skill)) or sigma_skill <= 0:
        return None
    return 1.0 - (skill_sd / sigma_skill) ** 2


def _skip(panel: str, reason: str) -> None:
    logger.warning("report_coverage_panel_skipped", panel=panel, reason=reason)


def _timeline(stats: CoverageStats) -> dict[str, Any] | None:
    if not stats.timeline:
        _skip("timeline", "no dated credited battles")
        return None
    return {
        # A bin straddling an era cut takes the era of its midpoint.
        "bins": [
            {"start": b.start, "n": b.n, "era": era_label(b.start + TIMELINE_BIN // 2)}
            for b in stats.timeline
        ],
        "empty_spans": [list(s) for s in stats.empty_spans],
        "eras": [{"start": start, "label": label} for start, label in REPORT_ERAS],
    }


def _components(stats: CoverageStats) -> dict[str, Any] | None:
    if not stats.component_sizes:
        _skip("components", "no component ids stored for the run")
        return None
    return {
        "sizes": list(stats.component_sizes),
        "markers": [
            {"name": m.name, "rank": m.rank, "component": m.component_id} for m in stats.top_markers
        ],
    }


def _credit_source(
    stats: CoverageStats, headline: Sequence[RankedGeneral]
) -> dict[str, Any] | None:
    by_general = credit_by_general(stats.credit_source)
    rows: list[dict[str, Any]] = []
    for g in headline[:CREDIT_TOP]:
        weights = by_general.get(g.estimate.general_id)
        if not weights:
            continue
        folded: dict[str, float] = {}
        for method, weight in weights.items():
            key = method if method in ATTRIBUTION_METHODS else OTHER_METHOD
            folded[key] = folded.get(key, 0.0) + weight
        rows.append({"name": g.estimate.name, "rank": g.headline_rank, "weights": folded})
    if not rows:
        _skip("credit_source", "no credited weight for the headline generals")
        return None
    used = {m for r in rows for m in r["weights"]}
    methods = [m for m in (*ATTRIBUTION_METHODS, OTHER_METHOD) if m in used]
    return {"methods": methods, "rows": rows}


def _info(stats: CoverageStats, headline: Sequence[RankedGeneral]) -> dict[str, Any] | None:
    if stats.sigma_skill is None:
        logger.warning(
            "report_coverage_contraction_unavailable",
            reason="sigma_skill not stored on the run; drawing effective battles only",
        )
    points = []
    for g in headline:
        eff = g.rubin.effective_battles
        if eff is None or g.headline_rank is None:
            continue
        shrink = (
            None
            if stats.sigma_skill is None
            else contraction(g.estimate.skill_sd, stats.sigma_skill)
        )
        points.append(
            {
                "rank": g.headline_rank,
                "name": g.estimate.name,
                "effective": eff,
                "contraction": shrink,
            }
        )
    if not points:
        _skip("info_vs_rank", "no effective battles stored for the headline generals")
        return None
    return {"points": points, "line": CONTRACTION_LINE, "sigma": stats.sigma_skill}


def _troops(stats: CoverageStats) -> dict[str, Any] | None:
    if not stats.troop_eras:
        _skip("troop_provenance", "no battle sides for the credited battles")
        return None
    panel: dict[str, Any] = {
        "eras": [
            {
                "era": e.era,
                "multi": e.multi_source,
                "single": e.single_report,
                "imputed": e.imputed,
                "other": e.other,
                "bound": e.bound_only,
            }
            for e in stats.troop_eras
        ],
        "order": [e for e in ERA_ORDER if any(t.era == e for t in stats.troop_eras)],
    }
    if stats.reported_vs_reconciled:
        panel["scatter"] = [
            {"era": era_label(p.year), "reported": p.reported, "reconciled": p.reconciled}
            for p in stats.reported_vs_reconciled
        ]
    else:
        logger.warning("report_coverage_subpanel_skipped", panel="troop_provenance", part="scatter")
    return panel


def _fame(stats: CoverageStats) -> dict[str, Any] | None:
    bins = [b for b in stats.fame_bins if b.n_credited]
    if not bins:
        _skip("fame_bias", "no led-side outcomes")
        return None
    order = ["1", "2", "3-4", "5-9", "10+"]
    bins.sort(key=lambda b: order.index(b.label) if b.label in order else len(order))
    return {
        "bins": [
            {"label": b.label, "n": b.n_credited, "wins": b.n_wins, "rate": b.n_wins / b.n_credited}
            for b in bins
        ]
    }


def build_panels(stats: CoverageStats, headline: Sequence[RankedGeneral]) -> dict[str, Any]:
    """Assemble every panel whose data exists.

    Args:
        stats: The coverage numbers, with component sizes added where known.
        headline: Headline generals in rank order.

    Returns:
        ``panel key -> plain data``, in :data:`PANEL_FILES` order; missing panels are absent.
    """
    built = {
        "timeline": _timeline(stats),
        "components": _components(stats),
        "credit_source": _credit_source(stats, headline),
        "info_vs_rank": _info(stats, headline),
        "troop_provenance": _troops(stats),
        "fame_bias": _fame(stats),
    }
    return {key: panel for key, panel in built.items() if panel is not None}


def expected_figures(panels: dict[str, Any]) -> tuple[str, ...]:
    """File names of the figures that should exist, one per panel that has data."""
    return tuple(PANEL_FILES[key] for key in PANEL_FILES if key in panels)
