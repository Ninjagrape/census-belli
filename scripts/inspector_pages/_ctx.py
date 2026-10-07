"""Shared plumbing for the inspector pages: engine, cached queries, display helpers.

No SQL lives here. Queries come from :mod:`pipeline.inspector.queries` and are
looked up by name so that one ``st.cache_data`` wrapper, keyed on the database
URL, covers all of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import structlog
from sqlalchemy import Engine, text

from pipeline.inspector import queries
from pipeline.inspector.connection import read_only_engine

logger = structlog.get_logger()

URL_KEY = "inspect_db_url"
ROOT_KEY = "inspect_data_root"
YEARS_KEY = "inspect_year_filter"
CACHE_TTL_SECONDS = 60


@dataclass(frozen=True)
class Ctx:
    """The sidebar selections every page needs."""

    url: str
    data_root: Path
    year_range: tuple[int, int] | None


def get_ctx() -> Ctx:
    """Read the sidebar selections from session state.

    Returns:
        The current context; an empty URL means no database is configured.
    """
    years = st.session_state.get(YEARS_KEY)
    return Ctx(
        url=str(st.session_state.get(URL_KEY, "")),
        data_root=Path(str(st.session_state.get(ROOT_KEY, "data"))),
        year_range=(int(years[0]), int(years[1])) if years else None,
    )


@st.cache_resource(show_spinner=False)
def get_engine(url: str) -> Engine:
    """Create (once per URL) the read-only engine.

    Args:
        url: The database URL.

    Returns:
        A read-only engine.
    """
    return read_only_engine(url)


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def ping(url: str) -> str:
    """Check that the database answers.

    Args:
        url: The database URL.

    Returns:
        An empty string when reachable, else a short reason.
    """
    if not url:
        return "No database URL is configured."
    try:
        with get_engine(url).connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        logger.warning("inspector_db_unreachable", error=type(exc).__name__)
        logger.debug("inspector_db_unreachable_detail", error=str(exc))
        return "The database is unreachable. Check the URL in the sidebar."
    return ""


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _run_query(url: str, name: str, args: tuple[Any, ...]) -> pd.DataFrame:
    """Run one named query on a fresh read-only connection (cached)."""
    if name not in queries.__all__:
        raise KeyError(f"unknown inspector query: {name}")
    with get_engine(url).connect() as conn:
        frame: pd.DataFrame = getattr(queries, name)(conn, *args)
    return frame


def require_db(ctx: Ctx) -> None:
    """Stop the page with a notice when the database cannot be used.

    Args:
        ctx: The current context.
    """
    problem = ping(ctx.url)
    if problem:
        st.warning(problem)
        st.stop()


def q(ctx: Ctx, name: str, *args: Any) -> pd.DataFrame | None:
    """Run a named query, degrading to a warning on failure.

    Args:
        ctx: The current context.
        name: A function name from ``pipeline.inspector.queries``.
        *args: Arguments after the connection; must be hashable.

    Returns:
        The result, or None when the query failed (a warning has been shown).
    """
    try:
        return _run_query(ctx.url, name, args)
    except Exception as exc:
        logger.warning("inspector_query_failed", query=name, error=str(exc))
        st.warning(
            f"Query `{name}` failed. The schema may be missing or out of date; "
            "details are in the server log."
        )
        return None


def table(df: pd.DataFrame | None, empty: str = "No rows.", **kwargs: Any) -> bool:
    """Show a frame, or an info notice when it is missing or empty.

    Args:
        df: The frame, or None after a failed query.
        empty: The message for an empty frame.
        **kwargs: Passed to ``st.dataframe``.

    Returns:
        True when something was drawn.
    """
    if df is None:
        return False
    if df.empty:
        st.info(empty)
        return False
    st.dataframe(df, hide_index=True, width="stretch", **kwargs)
    return True


def show_fig(fig: Any | None, empty: str) -> None:
    """Draw a plotly figure, or an info notice when there is nothing to plot.

    Args:
        fig: The figure, or None.
        empty: The message when ``fig`` is None.
    """
    if fig is None:
        st.info(empty)
    else:
        st.plotly_chart(fig, width="stretch")
