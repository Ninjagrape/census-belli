"""Evaluation figures. A plotting failure is logged and skipped, never raised."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import structlog

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from pipeline.evaluators.sensitivity import RankTable  # noqa: E402
from pipeline.evaluators.target import PublishedRun  # noqa: E402

__all__ = ["write_figures"]

logger = structlog.get_logger()

DPI = 120
POINT_COLOR = "#2a6f97"
POINT_ALPHA = 0.6
POINT_SIZE = 14


def _save(fig: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def plot_calibration(calibration: Sequence[Mapping[str, Any]], path: Path) -> Path:
    """Favourite's predicted against observed win rate by bin, with the diagonal.

    Args:
        calibration: ``binary_metrics`` ``calibration_favourite`` bins.
        path: Output PNG.

    Returns:
        The path.
    """
    bins = [b for b in calibration if b["count"] > 0]
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0.5, 1], [0.5, 1], color="#999999", linestyle="--", label="perfect")
    if bins:
        sizes = [30 + 8 * b["count"] for b in bins]
        ax.scatter(
            [b["mean_p"] for b in bins],
            [b["observed"] for b in bins],
            s=sizes,
            color=POINT_COLOR,
            alpha=POINT_ALPHA,
        )
    ax.set(
        xlim=(0.5, 1),
        ylim=(0.5, 1),
        xlabel="Predicted P(favourite wins)",
        ylabel="Observed share the favourite won",
    )
    ax.set_title("Held-out calibration (marker size = battles)")
    ax.legend(loc="upper left")
    return _save(fig, path)


def plot_rank_stability(
    top_ids: Sequence[int],
    names: Mapping[int, str],
    tables: Mapping[str, RankTable],
    path: Path,
) -> Path:
    """Rank of the baseline top generals under each variant.

    Args:
        top_ids: Baseline top generals, best first.
        names: ``general_id`` to name.
        tables: Column label to ranking; a general absent from a column is blank.
        path: Output PNG.

    Returns:
        The path.
    """
    grid = np.full((len(top_ids), len(tables)), np.nan)
    for j, table in enumerate(tables.values()):
        rank_of = dict(zip(table.ids.tolist(), table.rank.tolist(), strict=True))
        for i, g in enumerate(top_ids):
            grid[i, j] = rank_of.get(int(g), np.nan)
    fig, ax = plt.subplots(figsize=(1.2 + 0.9 * len(tables), 1.5 + 0.28 * len(top_ids)))
    shown = ax.imshow(np.minimum(grid, 3 * max(len(top_ids), 1)), cmap="viridis_r", aspect="auto")
    for (i, j), value in np.ndenumerate(grid):
        if np.isfinite(value):
            ax.text(j, i, f"{int(value)}", ha="center", va="center", fontsize=7, color="white")
    ax.set_xticks(range(len(tables)), list(tables), rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(top_ids)), [names.get(int(g), str(g)) for g in top_ids], fontsize=7)
    ax.set_title("Rank of the baseline top generals")
    fig.colorbar(shown, ax=ax, label="rank (capped)")
    return _save(fig, path)


def plot_ci_width(published: PublishedRun, path: Path) -> Path:
    """Credible interval width against battles fought.

    Args:
        published: The published run.
        path: Output PNG.

    Returns:
        The path.
    """
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.scatter(
        published.n_battles,
        published.ci_hi - published.ci_lo,
        color=POINT_COLOR,
        alpha=POINT_ALPHA,
        s=POINT_SIZE,
    )
    ax.set(xlabel="Battles credited", ylabel="95% interval width (skill)")
    ax.set_title("Uncertainty falls with evidence")
    return _save(fig, path)


def plot_skill_vs_battles(published: PublishedRun, path: Path) -> Path:
    """Posterior mean skill against battles fought.

    Args:
        published: The published run.
        path: Output PNG.

    Returns:
        The path.
    """
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.scatter(
        published.n_battles,
        published.skill_mean,
        color=POINT_COLOR,
        alpha=POINT_ALPHA,
        s=POINT_SIZE,
    )
    ax.axhline(0.0, color="#999999", linewidth=0.8)
    ax.set(xlabel="Battles credited", ylabel="Posterior mean skill")
    ax.set_title("Skill against sample size")
    return _save(fig, path)


def _attempt(name: str, make: Callable[[], Path], written: list[Path]) -> None:
    try:
        written.append(make())
    except Exception as exc:
        # A missing figure must not cost the report.
        plt.close("all")
        logger.warning("evaluate_plot_failed", figure=name, error=f"{type(exc).__name__}: {exc}")


def write_figures(
    directory: Path,
    *,
    calibration: Sequence[Mapping[str, Any]],
    top_ids: Sequence[int],
    names: Mapping[int, str],
    rank_tables: Mapping[str, RankTable],
    published: PublishedRun,
) -> list[Path]:
    """Write every figure that can be drawn.

    Args:
        directory: Output directory.
        calibration: Held-out favourite-frame calibration bins.
        top_ids: Baseline top generals, best first.
        names: ``general_id`` to name.
        rank_tables: Rankings for the stability plot, baseline first.
        published: The published run.

    Returns:
        The paths written; a figure that failed is logged and absent.
    """
    written: list[Path] = []
    _attempt(
        "calibration",
        lambda: plot_calibration(calibration, directory / "calibration.png"),
        written,
    )
    if rank_tables:
        _attempt(
            "rank_stability",
            lambda: plot_rank_stability(
                top_ids, names, rank_tables, directory / "rank_stability.png"
            ),
            written,
        )
    _attempt("ci_width", lambda: plot_ci_width(published, directory / "ci_width.png"), written)
    _attempt(
        "skill_vs_battles",
        lambda: plot_skill_vs_battles(published, directory / "skill_vs_battles.png"),
        written,
    )
    return written
