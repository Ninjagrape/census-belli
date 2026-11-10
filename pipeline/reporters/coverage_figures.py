"""Matplotlib PNGs for "Data coverage and caveats", one per panel of :mod:`coverage_panels`."""

from __future__ import annotations

import io
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import matplotlib
import structlog

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from pipeline.reporters.atomic import write_bytes_atomic  # noqa: E402
from pipeline.reporters.coverage import ERA_ORDER  # noqa: E402
from pipeline.reporters.coverage_panels import PANEL_FILES  # noqa: E402

__all__ = ["ERA_COLORS", "write_coverage_figures"]

logger = structlog.get_logger()

DPI = 120
# Okabe-Ito colours, which stay apart for the common colour-vision deficiencies.
ERA_COLORS: Final[dict[str, str]] = dict(
    zip(
        ERA_ORDER,
        ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"),
        strict=False,
    )
)
METHOD_COLORS: Final[dict[str, str]] = {
    "llm": "#0072B2",
    "rule_single": "#D55E00",
    "default_split": "#E69F00",
    "manual": "#009E73",
    "other": "#999999",
    "bound": "#CC79A7",
}
PROVENANCE_COLORS: Final[tuple[str, str, str]] = ("#0072B2", "#E69F00", "#D55E00")
SHADE = "#cccccc"
HIGHLIGHT = "#D55E00"
BASE = "#9db4c0"


def _timeline(fig: Figure, ax: Axes, panel: dict[str, Any]) -> None:
    for b in panel["bins"]:
        ax.bar(b["start"], b["n"], width=25, align="edge", color=ERA_COLORS.get(b["era"], BASE))
    top = ax.get_ylim()[1]
    for first, last in panel["empty_spans"]:
        ax.axvspan(first, last + 1, color=SHADE, alpha=0.5, zorder=0)
        if last - first >= 100:
            ax.text(
                (first + last) / 2,
                top * 0.9,
                "no battles",
                ha="center",
                va="top",
                fontsize=7,
                color="#555555",
            )
    handles = [Rectangle((0, 0), 1, 1, color=ERA_COLORS[e]) for e in ERA_ORDER]
    ax.legend(
        handles,
        ERA_ORDER,
        fontsize=7,
        loc="upper center",
        ncol=3,
        title="Coloured by the era of each bin's midpoint",
        title_fontsize=7,
    )
    ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
    first, last = panel["bins"][0]["start"], panel["bins"][-1]["start"] + 25
    ax.set_xlim(min(-500, first), max(2000, last))
    ax.set(xlabel="Year", ylabel="Credited battles per 25 years", title="Corpus timeline")


def _components(fig: Figure, ax: Axes, panel: dict[str, Any]) -> None:
    sizes = panel["sizes"]
    colors = [HIGHLIGHT if i == 0 else BASE for i in range(len(sizes))]
    ax.bar(range(len(sizes)), sizes, color=colors)
    ax.set_yscale("log")
    for m in panel["markers"]:
        ax.plot(m["component"], sizes[m["component"]], "k^", markersize=6)
    ax.set(
        xlabel="Component (0 is the largest)",
        ylabel="Generals (log scale)",
        title="Comparison components; triangles mark the top-ranked generals",
    )


def _credit_source(fig: Figure, ax: Axes, panel: dict[str, Any]) -> None:
    rows = panel["rows"]
    left = [0.0] * len(rows)
    for method in panel["methods"]:
        widths = [r["weights"].get(method, 0.0) for r in rows]
        ax.barh(range(len(rows)), widths, left=left, color=METHOD_COLORS[method], label=method)
        left = [a + b for a, b in zip(left, widths, strict=True)]
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r['rank']}. {r['name']}" for r in rows], fontsize=7)
    ax.invert_yaxis()
    ax.legend(fontsize=7)
    ax.set(xlabel="Credited weight", title="Where each headline general's credit comes from")


def _info(fig: Figure, ax: Axes, panel: dict[str, Any]) -> None:
    pts = panel["points"]
    ax.scatter([p["rank"] for p in pts], [p["effective"] for p in pts], color=BASE, s=14)
    ax.set(xlabel="Headline rank", ylabel="Effective battles", title="Information against rank")
    shrunk = [p for p in pts if p["contraction"] is not None]
    if not shrunk:
        return
    twin = ax.twinx()
    twin.scatter(
        [p["rank"] for p in shrunk], [p["contraction"] for p in shrunk], color=HIGHLIGHT, s=14
    )
    twin.axhline(panel["line"], color=HIGHLIGHT, linestyle="--", linewidth=0.8)
    twin.set_ylabel("Skill contraction 1 - (sd/sigma)^2", color=HIGHLIGHT)


def _troops(fig: Figure, ax: Axes, panel: dict[str, Any]) -> None:
    by_era = {e["era"]: e for e in panel["eras"]}
    order = panel["order"]
    bottom = [0.0] * len(order)
    totals = [
        sum(by_era[e][k] for k in ("multi", "single", "imputed", "other", "bound")) or 1
        for e in order
    ]
    labels = [("multi", "multi-source"), ("single", "single report"), ("imputed", "imputed")]
    colors = list(PROVENANCE_COLORS)
    if any(by_era[e]["other"] for e in order):
        labels.append(("other", "other method"))
        colors.append(METHOD_COLORS["other"])
    if any(by_era[e]["bound"] for e in order):
        labels.append(("bound", "bound only"))
        colors.append(METHOD_COLORS["bound"])
    for (key, label), color in zip(labels, colors, strict=True):
        share = [by_era[e][key] / t for e, t in zip(order, totals, strict=True)]
        ax.bar(order, share, bottom=bottom, color=color, label=label)
        bottom = [a + b for a, b in zip(bottom, share, strict=True)]
    ax.tick_params(axis="x", labelrotation=30, labelsize=7)
    ax.legend(fontsize=7)
    ax.set(ylabel="Share of sides", title="Troop provenance by era")


def _scatter(ax: Axes, points: list[dict[str, Any]]) -> None:
    for era in ERA_ORDER:
        rows = [p for p in points if p["era"] == era]
        if rows:
            ax.scatter(
                [p["reported"] for p in rows],
                [p["reconciled"] for p in rows],
                s=8,
                color=ERA_COLORS[era],
                label=era,
                alpha=0.7,
            )
    lo = min(min(p["reported"], p["reconciled"]) for p in points)
    hi = max(max(p["reported"], p["reconciled"]) for p in points)
    ax.plot([lo, hi], [lo, hi], color="#555555", linestyle="--", linewidth=0.8)
    ax.set(xscale="log", yscale="log", xlabel="Mean reported total", ylabel="Reconciled total")
    ax.legend(fontsize=6)


def _fame(fig: Figure, ax: Axes, panel: dict[str, Any]) -> None:
    bins = panel["bins"]
    bars = ax.bar([b["label"] for b in bins], [b["rate"] for b in bins], color=BASE)
    for bar, b in zip(bars, bins, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"n={b['n']}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set(
        xlabel="Credited battles of the general",
        ylabel="Win rate of top commanders",
        ylim=(0, 1.1),
        title="Fame bias",
    )


_DRAW: Final[dict[str, Callable[[Figure, Axes, dict[str, Any]], None]]] = {
    "timeline": _timeline,
    "components": _components,
    "credit_source": _credit_source,
    "info_vs_rank": _info,
    "fame_bias": _fame,
}


def _render(key: str, panel: dict[str, Any]) -> bytes:
    scatter = panel.get("scatter") if key == "troop_provenance" else None
    if key == "credit_source":
        size = (7.0, max(3.0, 0.28 * len(panel["rows"]) + 1.2))
    else:
        size = (11.0, 4.2) if scatter else (7.0, 4.2)
    fig, axes = plt.subplots(
        1,
        2 if scatter else 1,
        figsize=size,
        squeeze=False,
        gridspec_kw={"width_ratios": [1, 1]} if scatter else None,
    )
    if key == "troop_provenance":
        _troops(fig, axes[0][0], panel)
        if scatter:
            _scatter(axes[0][1], scatter)
    else:
        _DRAW[key](fig, axes[0][0], panel)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=DPI)
    plt.close(fig)
    return buffer.getvalue()


def write_coverage_figures(panels: dict[str, Any], directory: Path) -> tuple[str, ...]:
    """Write one PNG per panel; a panel that fails to draw is logged and skipped.

    Args:
        panels: Output of :func:`pipeline.reporters.coverage_panels.build_panels`.
        directory: The report's ``figures`` directory.

    Returns:
        The file names written, in panel order.
    """
    written: list[str] = []
    for key, name in PANEL_FILES.items():
        if key not in panels:
            continue
        try:
            write_bytes_atomic(directory / name, _render(key, panels[key]))
        except Exception as exc:
            logger.warning("report_coverage_figure_failed", panel=key, error=str(exc))
            plt.close("all")
            continue
        written.append(name)
    return tuple(written)
