"""Reconcile: troop reports against the estimates fitted from them."""

from __future__ import annotations

import streamlit as st

from pipeline.inspector.files import find_neutral_draws, summarise_neutral_draws
from scripts.inspector_pages import _charts, _ctx

ctx = _ctx.get_ctx()
st.title("Reconcile")
_ctx.require_db(ctx)

sides = _ctx.q(ctx, "reconcile_sides", ctx.year_range)
reports = _ctx.q(ctx, "troop_reports_detail", ctx.year_range)

st.subheader("Reports against estimates, per side")
if sides is not None and reports is not None and not sides.empty:
    with_reports = sorted(reports["side_id"].unique().tolist())
    chosen = st.multiselect(
        "Sides to plot",
        options=with_reports,
        default=with_reports[: _charts.MAX_STRIP_SIDES // 2],
        format_func=lambda sid: (
            f"{sid}: {sides.loc[sides['side_id'] == sid, 'battle_name'].iloc[0]}"
        ),
    )
    _ctx.show_fig(_charts.strip_plot(reports, sides, chosen), "Select at least one side.")
else:
    st.info("No sides or reports to plot.")

st.subheader("Interval width and neutral against published")
if sides is not None and not sides.empty:
    left, right = st.columns(2)
    with left:
        _ctx.show_fig(_charts.ci_width_hist(sides), "No intervals fitted yet.")
    with right:
        _ctx.show_fig(_charts.neutral_vs_published(sides), "No side has both estimates.")

st.subheader("Source bias")
sources = _ctx.q(ctx, "source_bias")
if sources is not None and not sources.empty:
    _ctx.show_fig(_charts.bias_forest(sources), "No source has been fitted (all rows are prior).")
_ctx.table(sources, "No sources.")

st.subheader("Per-side estimates")
_ctx.table(sides, "No sides.")

st.subheader("Troop reports")
_ctx.table(reports, "No troop reports.")

st.subheader("Casualties")
_ctx.table(_ctx.q(ctx, "casualty_estimates", ctx.year_range), "No sides.")
_ctx.table(_ctx.q(ctx, "casualty_reports_detail", ctx.year_range), "No casualty reports.")

with st.expander("Neutral draws files"):
    files = find_neutral_draws(ctx.data_root)
    if not files:
        st.info(f"No neutral draws under `{ctx.data_root / 'processed' / 'reconcile'}`.")
    else:
        picked = st.selectbox("File", files, index=len(files) - 1, format_func=lambda p: p.name)
        _ctx.table(summarise_neutral_draws(picked), "File holds no draws.")
