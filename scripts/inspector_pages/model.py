"""Model: the Bayesian model stage is not built yet; show what exists."""

from __future__ import annotations

import streamlit as st

from scripts.inspector_pages import _ctx

ctx = _ctx.get_ctx()
st.title("Model")
st.info("The model, evaluate and report stages are not built yet.")
_ctx.require_db(ctx)

runs = _ctx.q(ctx, "model_runs")
if runs is not None and not runs.empty:
    st.subheader("Model runs")
    _ctx.table(runs)

skills = _ctx.q(ctx, "skill_estimates", 200)
if skills is not None and not skills.empty:
    st.subheader("General skill estimates")
    _ctx.table(skills)
