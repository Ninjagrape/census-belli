"""Resolve: entity resolution outcomes."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from pipeline.inspector.files import read_resolution_log
from scripts.inspector_pages import _ctx

ctx = _ctx.get_ctx()
st.title("Resolve")

st.subheader("Resolution log")
log, bad_lines = read_resolution_log(ctx.data_root)
if bad_lines:
    st.warning(f"{bad_lines} malformed line(s) in resolution_log.jsonl were skipped.")
if log.empty:
    st.info(f"No resolution log at `{ctx.data_root / 'processed' / 'resolution_log.jsonl'}`.")
else:
    if "status" in log.columns:
        counts = log.groupby(["status", "method"], dropna=False).size().reset_index(name="n")
        st.plotly_chart(px.bar(counts, x="status", y="n", color="method"), width="stretch")
    st.dataframe(log, hide_index=True, width="stretch")

_ctx.require_db(ctx)

st.subheader("Generals")
_ctx.table(_ctx.q(ctx, "generals_summary", ctx.year_range, 1000), "No generals.")

st.subheader("Generals without a Wikidata id")
_ctx.table(
    _ctx.q(ctx, "generals_missing_wikidata", ctx.year_range, 1000),
    "Every general is linked to a Wikidata id.",
)

st.subheader("Commanders per side")
dist = _ctx.q(ctx, "commanders_per_side", ctx.year_range)
if _ctx.table(dist, "No sides."):
    st.plotly_chart(px.bar(dist, x="n_commanders", y="n_sides"), width="stretch")
