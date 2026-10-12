"""Model: the pooled run's diagnostics, battle counts and ranking."""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

from pipeline.inspector.frames import pd
from scripts.inspector_pages import _ctx

EXCLUSION_LABELS = {
    "battles_multi_side": "not exactly two sides",
    "battles_no_outcome": "no outcome on either side",
    "battles_outcome_conflict": "sides disagree on win, draw or loss",
    "battles_no_creditable_side": "no side has a creditable commander",
    "battles_general_both_sides": "same general on both sides",
}
RANKING_LIMIT = 200


def _parse(text: Any) -> dict[str, Any]:
    """Diagnostics JSON text to a dict; empty when absent or malformed."""
    try:
        value = json.loads(str(text))
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _show_counts(counts: dict[str, Any]) -> None:
    """Battles in scope, included, and each exclusion reason."""
    left, mid, right = st.columns(3)
    left.metric("Battles in scope", counts.get("battles_in_scope", "n/a"))
    mid.metric("Included", counts.get("battles_included", "n/a"))
    right.metric("Excluded", counts.get("battles_excluded", "n/a"))
    reasons = pd.DataFrame(
        [
            {"reason": label, "battles": int(counts[key])}
            for key, label in EXCLUSION_LABELS.items()
            if key in counts
        ]
    )
    _ctx.table(reasons, "This run recorded no exclusion counters.")


ctx = _ctx.get_ctx()
st.title("Model")
_ctx.require_db(ctx)

runs = _ctx.q(ctx, "pooled_runs")
if runs is None:
    st.stop()
if runs.empty:
    st.info("No pooled model run yet. Run the model stage.")
    st.stop()

labels = {
    int(r.run_id): f"run {r.run_id} ({r.scale}, converged: {r.converged})"
    for r in runs.itertuples()
}
# The newest neutral run is the primary ranking; a published-scale run is a
# sensitivity check and is listed after it, never preselected over it.
neutral = runs[runs["scale"] == "neutral"]
ordered = [int(i) for i in neutral["run_id"]] + [
    int(i) for i in runs[runs["scale"] != "neutral"]["run_id"]
]
run_id = int(st.selectbox("Pooled run", ordered, format_func=lambda i: labels[int(i)]))
row = runs[runs["run_id"] == run_id].iloc[0]
diag = _parse(row["diagnostics"])

st.subheader("Run")
header = (
    f"{row['run_name']}, scale {row['scale']}, imputation run "
    f"{diag.get('imputation_run_id', 'n/a')}, {diag.get('n_rounds', 'n/a')} round(s)"
)
if row["completed_at"] is None or pd.isna(row["completed_at"]):
    st.warning(f"{header}. Not completed.")
elif diag.get("estimates_written"):
    st.success(f"{header}. Estimates written.")
else:
    st.warning(f"{header}. No estimates written: {diag.get('rounds_failed', 'a')} round(s) failed.")
if diag.get("worst"):
    st.caption("Worst convergence statistics across rounds")
    st.json(diag["worst"], expanded=False)
gates = diag.get("gates") or {}
if gates:
    st.caption("Gates")
    _ctx.table(
        pd.DataFrame(
            [{"gate": name, "result": "pass" if ok else "FAIL"} for name, ok in gates.items()]
        )
    )

st.subheader("Battles")
_show_counts(diag.get("counts") or {})

st.subheader("Ranking")
ranking = _ctx.q(ctx, "run_ranking", run_id, RANKING_LIMIT)
_ctx.table(ranking, "This run wrote no estimates.")
