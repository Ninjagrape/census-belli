"""Overview: database target, table sizes, latest runs, and the quality gates."""

from __future__ import annotations

import streamlit as st
import structlog

from pipeline.inspector.connection import redact_url
from pipeline.inspector.frames import pd
from pipeline.inspector.gates import run_all_gates
from scripts.inspector_pages import _ctx

logger = structlog.get_logger()

ctx = _ctx.get_ctx()
st.title("Overview")
st.caption(f"Database: `{redact_url(ctx.url) if ctx.url else 'not configured'}`")
_ctx.require_db(ctx)

st.subheader("Rows per table")
_ctx.table(_ctx.q(ctx, "table_counts"), "No tables found. Has the schema been applied?")

st.subheader("Latest model run per type")
_ctx.table(_ctx.q(ctx, "latest_model_runs"), "No model runs yet.")

st.subheader("Quality gates")
st.caption("Runs every stage's `quality_checks` on a read-only connection.")
if st.button("Run all stage gates"):
    try:
        with _ctx.get_engine(ctx.url).connect() as conn:
            gates = run_all_gates(conn)
    except Exception as exc:
        logger.warning("inspector_gates_failed", error=str(exc))
        st.warning("Could not run the gates; details are in the server log.")
    else:
        gates = gates.assign(result=gates["passed"].map({True: "pass", False: "FAIL"}))
        summary = (
            gates.assign(
                failed_error=(~gates["passed"]) & (gates["severity"] == "error"),
                failed_warning=(~gates["passed"]) & (gates["severity"] != "error"),
            )
            .groupby("stage", sort=False)
            .agg(
                checks=("name", "count"),
                passed=("passed", "sum"),
                failed_error=("failed_error", "sum"),
                failed_warning=("failed_warning", "sum"),
            )
            .reset_index()
        )
        st.dataframe(summary, hide_index=True, width="stretch")
        matrix = gates.pivot_table(
            index="name", columns="stage", values="result", aggfunc="first", fill_value=""
        )
        st.caption("Pass/fail matrix (check by stage)")
        st.dataframe(matrix, width="stretch")
        st.caption("Detail")
        st.dataframe(pd.DataFrame(gates).drop(columns=["passed"]), hide_index=True, width="stretch")
