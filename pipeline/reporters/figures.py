"""Static figures for report.md. A plotting failure is logged and skipped."""

from __future__ import annotations

import io
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import structlog

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from pipeline.reporters.atomic import write_bytes_atomic  # noqa: E402
from pipeline.reporters.records import EvaluationInfo  # noqa: E402
from pipeline.reporters.tables import RankedGeneral  # noqa: E402

__all__ = ["FOREST_NAME", "copy_evaluation_figures", "write_forest_plot"]

logger = structlog.get_logger()

FOREST_NAME = "forest_top.png"
DPI = 120
POINT_COLOR = "#2a6f97"
ROW_HEIGHT = 0.28
MIN_HEIGHT = 3.0


def write_forest_plot(top: Sequence[RankedGeneral], directory: Path, ci_level: float) -> str | None:
    """Skill with its stored credible interval for the headline top generals.

    Rows follow the stored WAR rank, best at the top; the plot shows skill,
    which is why the points need not descend.

    Args:
        top: The leading headline generals.
        directory: The report's ``figures`` directory.
        ci_level: The level of the stored interval, for the title.

    Returns:
        The file name written, or None when there is nothing to plot or the
        plot failed.
    """
    if not top:
        return None
    try:
        fig, ax = plt.subplots(figsize=(7.0, max(MIN_HEIGHT, ROW_HEIGHT * len(top) + 1.2)))
        ys = list(range(len(top)))
        means = [g.estimate.skill_mean for g in top]
        lows = [g.estimate.skill_mean - g.estimate.skill_ci_lo for g in top]
        highs = [g.estimate.skill_ci_hi - g.estimate.skill_mean for g in top]
        ax.errorbar(means, ys, xerr=[lows, highs], fmt="o", color=POINT_COLOR, ecolor="#9db4c0")
        ax.set_yticks(ys)
        labels = [f"{'n/a' if g.rank is None else g.rank}. {g.estimate.name}" for g in top]
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.axvline(0.0, color="#999999", linestyle="--", linewidth=0.8)
        ax.set(xlabel="Latent skill", title=f"Skill with {ci_level:.0%} credible interval")
        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=DPI)
        plt.close(fig)
        write_bytes_atomic(directory / FOREST_NAME, buffer.getvalue())
    except Exception as exc:
        logger.warning("report_forest_plot_failed", error=str(exc))
        plt.close("all")
        return None
    return FOREST_NAME


def copy_evaluation_figures(evaluation: EvaluationInfo | None, directory: Path) -> tuple[str, ...]:
    """Copy evaluate's PNGs next to the report's own figures.

    Args:
        evaluation: The current evaluation info, or None.
        directory: The report's ``figures`` directory.

    Returns:
        The copied file names.
    """
    if evaluation is None:
        return ()
    names: list[str] = []
    for source in evaluation.figure_files:
        write_bytes_atomic(directory / source.name, source.read_bytes())
        names.append(source.name)
    return tuple(names)
