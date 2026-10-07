"""Read-only dashboard for inspecting the data at each pipeline stage.

Run with::

    streamlit run scripts/inspector_app.py

The database URL defaults to ``INSPECT_DATABASE_URL``, else ``DATABASE_URL``
(after loading ``.env``). Every connection is opened read-only.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st  # noqa: E402

from pipeline.inspector.connection import default_inspect_url, redact_url  # noqa: E402
from scripts.inspector_pages import _ctx  # noqa: E402

PAGES = [
    ("inspector_pages/overview.py", "Overview"),
    ("inspector_pages/crawl.py", "Crawl"),
    ("inspector_pages/extract.py", "Extract"),
    ("inspector_pages/resolve.py", "Resolve"),
    ("inspector_pages/reconcile.py", "Reconcile"),
    ("inspector_pages/classify.py", "Classify"),
    ("inspector_pages/impute.py", "Impute"),
    ("inspector_pages/battle.py", "Battle drill-down"),
    ("inspector_pages/model.py", "Model"),
]


def _sidebar() -> None:
    """Draw the sidebar controls and publish them to session state."""
    st.sidebar.header("Data sources")
    url = st.sidebar.text_input(
        "Database URL",
        value=default_inspect_url(),
        type="password",
        key=_ctx.URL_KEY,
        help="Defaults to INSPECT_DATABASE_URL, else DATABASE_URL.",
    )
    st.sidebar.caption(f"Target: `{redact_url(url) if url else 'not configured'}`")
    st.sidebar.text_input(
        "Data root", value=os.environ.get("INSPECT_DATA_ROOT", "data"), key=_ctx.ROOT_KEY
    )
    _year_filter(url)


def _year_filter(url: str) -> None:
    """Draw the year slider when the database has dated battles."""
    st.session_state[_ctx.YEARS_KEY] = None
    if not url or _ctx.ping(url):
        return
    ctx = _ctx.get_ctx()
    bounds = _ctx.q(ctx, "battle_year_bounds")
    if bounds is None or bounds.empty or bounds.iloc[0].isna().any():
        return
    low, high = int(bounds.loc[0, "min_year"]), int(bounds.loc[0, "max_year"])
    if low >= high:
        return
    chosen = st.sidebar.slider(
        "Year (astronomical, negative = BC)", low, high, (low, high), key="inspect_year_slider"
    )
    if tuple(chosen) != (low, high):
        st.session_state[_ctx.YEARS_KEY] = tuple(chosen)


def main() -> None:
    """Configure the page, draw the sidebar and run the selected page."""
    st.set_page_config(page_title="General WAR inspector", layout="wide")
    _sidebar()
    pages = [st.Page(path, title=title, default=(i == 0)) for i, (path, title) in enumerate(PAGES)]
    st.navigation(pages).run()


main()
