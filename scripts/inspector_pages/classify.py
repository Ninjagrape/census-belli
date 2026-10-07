"""Classify: command roles, attribution weights and missingness."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from scripts.inspector_pages import _ctx

ctx = _ctx.get_ctx()
st.title("Classify")
_ctx.require_db(ctx)

st.subheader("Roles and attribution methods")
roles = _ctx.q(ctx, "role_counts", ctx.year_range)
if _ctx.table(roles, "No commander rows."):
    st.plotly_chart(
        px.bar(roles, x="command_role", y="n", color="attribution_method"), width="stretch"
    )

st.subheader("Attribution weights")
weights = _ctx.q(ctx, "attribution_weights", ctx.year_range)
sums = _ctx.q(ctx, "side_weight_sums", ctx.year_range)
left, right = st.columns(2)
with left:
    if weights is not None and not weights.empty:
        st.plotly_chart(px.histogram(weights, x="attribution_weight"), width="stretch")
with right:
    if sums is not None and not sums.empty:
        fig = px.histogram(sums, x="weight_sum")
        fig.update_xaxes(title="sum of weights per side (1.0 expected)")
        st.plotly_chart(fig, width="stretch")
_ctx.table(weights, "No commander rows.")

st.subheader("Attribution review queue")
_ctx.table(_ctx.q(ctx, "review_queue", 500), "The review queue is empty.")

st.subheader("Missing data by field and class")
missing = _ctx.q(ctx, "missing_data_matrix", ctx.year_range)
if missing is not None and not missing.empty:
    matrix = missing.pivot_table(
        index="field_name", columns="missingness_class", values="n", fill_value=0
    )
    st.dataframe(matrix, width="stretch")
else:
    _ctx.table(missing, "missing_data_log is empty.")

st.subheader("is_defender coverage")
defender = _ctx.q(ctx, "defender_coverage", ctx.year_range)
if _ctx.table(defender, "No sides."):
    st.plotly_chart(px.bar(defender, x="is_defender", y="n_sides"), width="stretch")
