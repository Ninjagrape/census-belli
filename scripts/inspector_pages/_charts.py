"""Plotly figure builders for the inspector. Pure functions: frame in, figure out."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

MAX_STRIP_SIDES = 40
Z95 = 1.96


def _positive(df: pd.DataFrame, *cols: str) -> pd.DataFrame:
    """Rows where every named column is a positive number."""
    mask = pd.Series(True, index=df.index)
    for col in cols:
        values = pd.to_numeric(df[col], errors="coerce")
        mask &= values > 0
    return df[mask]


def strip_plot(
    reports: pd.DataFrame,
    sides: pd.DataFrame,
    side_ids: list[int],
) -> go.Figure | None:
    """Reported troop values per side with the fitted estimates overlaid.

    Args:
        reports: Output of ``troop_reports_detail``.
        sides: Output of ``reconcile_sides``.
        side_ids: Sides to draw (capped at ``MAX_STRIP_SIDES``).

    Returns:
        A figure, or None when there is nothing to draw.
    """
    chosen = side_ids[:MAX_STRIP_SIDES]
    sub_sides = sides[sides["side_id"].isin(chosen)].copy()
    if sub_sides.empty:
        return None
    sub_sides["label"] = sub_sides["battle_name"] + " / " + sub_sides["side_label"]
    labels = dict(zip(sub_sides["side_id"], sub_sides["label"], strict=True))
    fig = go.Figure()

    rep = _positive(reports[reports["side_id"].isin(chosen)], "reported_value").copy()
    rep["label"] = rep["side_id"].map(labels)
    fig.add_trace(
        go.Scatter(
            x=rep["reported_value"],
            y=rep["label"],
            mode="markers",
            name="reported",
            marker={"size": 8, "opacity": 0.7},
            text=rep["source_title"],
            hovertemplate="%{y}<br>%{x:,.0f}<br>%{text}<extra>reported</extra>",
        )
    )
    for centre, name, symbol in (
        ("est_troops_total", "published", "diamond"),
        ("est_troops_neutral", "neutral", "square"),
    ):
        est = _positive(sub_sides, centre)
        lo, hi = f"{centre}_lo", f"{centre}_hi"
        fig.add_trace(
            go.Scatter(
                x=est[centre],
                y=est["label"],
                mode="markers",
                name=name,
                marker={"size": 11, "symbol": symbol},
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": (est[hi] - est[centre]).clip(lower=0),
                    "arrayminus": (est[centre] - est[lo]).clip(lower=0),
                },
            )
        )
    fig.update_xaxes(type="log", title="troops (log scale)")
    fig.update_layout(height=max(320, 28 * len(chosen) + 120), legend={"orientation": "h"})
    return fig


def ci_width_hist(sides: pd.DataFrame) -> go.Figure | None:
    """Histogram of log(hi/lo) for the published and neutral intervals.

    Args:
        sides: Output of ``reconcile_sides``.

    Returns:
        A figure, or None when no side has an interval.
    """
    frames = []
    for label, lo, hi in (
        ("published", "est_troops_total_lo", "est_troops_total_hi"),
        ("neutral", "est_troops_neutral_lo", "est_troops_neutral_hi"),
    ):
        ok = _positive(sides, lo, hi)
        if not ok.empty:
            width = np.log(ok[hi].astype(float) / ok[lo].astype(float))
            frames.append(pd.DataFrame({"log_hi_over_lo": width, "scale": label}))
    if not frames:
        return None
    data = pd.concat(frames, ignore_index=True)
    fig = px.histogram(data, x="log_hi_over_lo", color="scale", barmode="overlay", opacity=0.65)
    fig.update_xaxes(title="log(hi / lo)")
    return fig


def neutral_vs_published(sides: pd.DataFrame) -> go.Figure | None:
    """Log-log scatter of neutral against published estimates.

    Args:
        sides: Output of ``reconcile_sides``.

    Returns:
        A figure, or None when no side has both estimates.
    """
    ok = _positive(sides, "est_troops_total", "est_troops_neutral")
    if ok.empty:
        return None
    fig = px.scatter(
        ok,
        x="est_troops_total",
        y="est_troops_neutral",
        hover_data=["battle_name", "side_label", "est_troops_method"],
        log_x=True,
        log_y=True,
    )
    low = float(min(ok["est_troops_total"].min(), ok["est_troops_neutral"].min()))
    high = float(max(ok["est_troops_total"].max(), ok["est_troops_neutral"].max()))
    fig.add_trace(
        go.Scatter(x=[low, high], y=[low, high], mode="lines", name="y = x", line={"dash": "dash"})
    )
    return fig


def bias_forest(sources: pd.DataFrame) -> go.Figure | None:
    """Forest plot of fitted source bias (log scale) with 95 percent intervals.

    Args:
        sources: Output of ``source_bias``.

    Returns:
        A figure, or None when no source has been fitted.
    """
    fitted = sources[pd.to_numeric(sources["bias_n_troop_reports"], errors="coerce") > 0].copy()
    if fitted.empty:
        return None
    fitted["label"] = fitted["title"].fillna("source " + fitted["source_id"].astype(str))
    fitted = fitted.sort_values("troop_bias_mu")
    fig = go.Figure(
        go.Scatter(
            x=fitted["troop_bias_mu"],
            y=fitted["label"],
            mode="markers",
            error_x={"type": "data", "array": Z95 * fitted["troop_bias_sd"].astype(float)},
        )
    )
    fig.add_vline(x=0, line_dash="dash")
    fig.update_xaxes(title="log-scale bias (0 = unbiased)")
    fig.update_layout(height=max(300, 26 * len(fitted) + 120))
    return fig


def observed_vs_imputed(rounds: pd.DataFrame) -> go.Figure | None:
    """Overlaid distributions of observed and imputed log troops.

    Args:
        rounds: Output of ``read_rounds``.

    Returns:
        A figure, or None when there are no rounds.
    """
    if rounds.empty:
        return None
    data = rounds.assign(kind=np.where(rounds["is_imputed"].astype(bool), "imputed", "observed"))
    fig = px.histogram(
        data, x="log_troops", color="kind", barmode="overlay", opacity=0.65, histnorm="probability"
    )
    fig.update_xaxes(title="log troops")
    return fig


def round_spread(rounds: pd.DataFrame) -> go.Figure | None:
    """Between-round spread of log troops for each imputed side.

    Args:
        rounds: Output of ``read_rounds``.

    Returns:
        A box plot of the most uncertain imputed sides, or None when no side
        was imputed or only one round exists.
    """
    if rounds.empty or rounds["round"].nunique() < 2:
        return None
    imputed = rounds[rounds["is_imputed"].astype(bool)]
    if imputed.empty:
        return None
    spread = imputed.groupby("side_id")["log_troops"].std().sort_values(ascending=False)
    top = spread.head(MAX_STRIP_SIDES).index
    data = imputed[imputed["side_id"].isin(top)].copy()
    data["side"] = "side " + data["side_id"].astype(str)
    order: Any = ["side " + str(i) for i in top]
    fig = px.box(data, x="log_troops", y="side", points="all", category_orders={"side": order})
    fig.update_layout(height=max(320, 24 * len(top) + 120))
    return fig
