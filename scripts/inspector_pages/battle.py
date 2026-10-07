"""Battle drill-down: everything the pipeline holds about one battle."""

from __future__ import annotations

import streamlit as st

from scripts.inspector_pages import _ctx

ctx = _ctx.get_ctx()
st.title("Battle drill-down")
_ctx.require_db(ctx)

term = st.text_input("Search battles by name", value="")
matches = _ctx.q(ctx, "search_battles", term, ctx.year_range, 100)
if matches is None:
    st.stop()
if matches.empty:
    st.info("No battles match.")
    st.stop()

labels = {
    int(row.battle_id): f"{row.name} ({row.year_astronomical})" for row in matches.itertuples()
}
battle_id = st.selectbox("Battle", list(labels), format_func=lambda bid: labels[bid])

SECTIONS = (
    ("Battle", "battle_row"),
    ("Sides", "battle_sides"),
    ("Commanders", "battle_commanders"),
    ("Troop reports", "battle_troop_reports"),
    ("Casualty reports", "battle_casualty_reports"),
    ("Missing data log", "battle_missing_data"),
    ("LLM calls", "battle_llm_calls"),
    ("Crawl log", "battle_crawl_log"),
)
for title, query in SECTIONS:
    st.subheader(title)
    _ctx.table(_ctx.q(ctx, query, int(battle_id)), "None recorded.")
