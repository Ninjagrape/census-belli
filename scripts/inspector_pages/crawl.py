"""Crawl: raw files on disk and the crawl log."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from pipeline.inspector.files import raw_file_counts
from scripts.inspector_pages import _ctx

ctx = _ctx.get_ctx()
st.title("Crawl")

st.subheader("Raw files")
_ctx.table(raw_file_counts(ctx.data_root), f"No files under `{ctx.data_root / 'raw'}`.")

_ctx.require_db(ctx)

st.subheader("Fetches by HTTP status and extraction method")
status = _ctx.q(ctx, "crawl_status_counts")
if _ctx.table(status, "crawl_log is empty."):
    st.plotly_chart(
        px.bar(status, x="http_status", y="n", color="extraction_method"), width="stretch"
    )

st.subheader("Fetches over time")
over_time = _ctx.q(ctx, "crawl_over_time")
if _ctx.table(over_time, "No fetches recorded."):
    st.plotly_chart(px.bar(over_time, x="day", y="n_fetches"), width="stretch")

st.subheader("Rows with errors")
_ctx.table(_ctx.q(ctx, "crawl_errors", 500), "No crawl errors recorded.")
