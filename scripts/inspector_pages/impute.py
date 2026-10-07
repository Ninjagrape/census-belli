"""Impute: the imputation log, rounds on disk and the gate report."""

from __future__ import annotations

import streamlit as st

from pipeline.inspector.files import (
    list_imputed_scales,
    read_gate_report,
    read_manifests,
    read_rounds,
)
from pipeline.inspector.frames import pd
from scripts.inspector_pages import _charts, _ctx

ctx = _ctx.get_ctx()
st.title("Impute")

st.subheader("Imputation rounds on disk")
scales = list_imputed_scales(ctx.data_root)
if not scales:
    st.info(f"No imputed data under `{ctx.data_root / 'imputed'}`.")
else:
    scale = st.selectbox("Scale", scales)
    rounds = read_rounds(ctx.data_root, scale)
    if rounds.empty:
        st.info(f"No rounds for scale `{scale}`.")
    else:
        n_rounds = int(rounds["round"].nunique())
        n_imputed = int(rounds.loc[rounds["is_imputed"].astype(bool), "side_id"].nunique())
        st.write(f"{n_rounds} round(s), {rounds['side_id'].nunique()} sides, {n_imputed} imputed.")
        left, right = st.columns(2)
        with left:
            _ctx.show_fig(_charts.observed_vs_imputed(rounds), "No rounds.")
        with right:
            _ctx.show_fig(
                _charts.round_spread(rounds), "Needs two or more rounds with imputed sides."
            )
    st.caption("Manifests")
    _ctx.table(read_manifests(ctx.data_root, scale), "No manifests.")

    st.subheader("Gate report")
    report = read_gate_report(ctx.data_root, scale)
    if not report:
        st.info("No gate_report.json for this scale.")
    else:
        written = report.get("rounds_written")
        header = (
            f"run {report.get('run_id')}, scale {report.get('scale')}, rounds written: {written}"
        )
        (st.success if written else st.warning)(header)
        _ctx.table(pd.DataFrame(report.get("gates", [])), "The report lists no gates.")

_ctx.require_db(ctx)

st.subheader("Imputed rows in missing_data_log")
_ctx.table(_ctx.q(ctx, "imputed_rows", ctx.year_range), "No rows have been imputed.")
