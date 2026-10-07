"""Extract: what was pulled out of the articles, and how complete it is."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from scripts.inspector_pages import _ctx

ctx = _ctx.get_ctx()
st.title("Extract")
_ctx.require_db(ctx)

st.subheader("Battles by century and type")
st.caption("Century axis is year_astronomical, so BC battles are included.")
century = _ctx.q(ctx, "battles_by_century", ctx.year_range)
if _ctx.table(century, "No dated battles in range."):
    st.plotly_chart(px.bar(century, x="century_start", y="n", color="battle_type"), width="stretch")

st.subheader("Field completeness")
left, right = st.columns(2)
for column, name, query in (
    (left, "battles", "null_fraction_battles"),
    (right, "battle_sides", "null_fraction_battle_sides"),
):
    with column:
        st.caption(name)
        frame = _ctx.q(ctx, query, ctx.year_range)
        if _ctx.table(frame, f"No {name} rows."):
            fig = px.bar(frame, x="null_fraction", y="column_name", orientation="h")
            fig.update_layout(height=max(300, 22 * len(frame) + 80))
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, width="stretch")

st.subheader("Troop reports by scope, branch and extraction method")
_ctx.table(_ctx.q(ctx, "troop_report_breakdown", ctx.year_range), "No troop reports.")

st.subheader("LLM calls")
_ctx.table(_ctx.q(ctx, "llm_summary"), "No LLM calls logged.")
