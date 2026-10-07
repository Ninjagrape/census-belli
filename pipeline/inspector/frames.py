"""The single place pandas is imported for the inspector.

pandas ships no type information and ``pandas-stubs`` is not a project
dependency, so ``DataFrame`` is aliased to ``Any`` here once rather than
sprinkling ignores through every query module.
"""

from __future__ import annotations

from typing import Any, TypeAlias

import pandas as pd  # type: ignore[import-untyped]
from sqlalchemy import text
from sqlalchemy.engine import Connection

__all__ = ["YEAR_FILTER", "Frame", "empty_frame", "pd", "query_frame", "year_params"]

Frame: TypeAlias = Any

# Applied to the ``b`` alias (battles). CAST keeps psycopg from rejecting an
# untyped NULL parameter. A NULL bound means "no filter", so battles with no
# date are kept unless the caller narrows the range.
YEAR_FILTER = (
    "(CAST(:y_lo AS INT) IS NULL OR b.year_astronomical >= :y_lo) "
    "AND (CAST(:y_hi AS INT) IS NULL OR b.year_astronomical <= :y_hi)"
)


def year_params(year_range: tuple[int, int] | None) -> dict[str, Any]:
    """Bind parameters for :data:`YEAR_FILTER`.

    Args:
        year_range: Inclusive ``(low, high)`` astronomical years, or None.

    Returns:
        Parameters naming ``y_lo`` and ``y_hi``.
    """
    if year_range is None:
        return {"y_lo": None, "y_hi": None}
    return {"y_lo": int(year_range[0]), "y_hi": int(year_range[1])}


def query_frame(conn: Connection, sql: str, params: dict[str, Any] | None = None) -> Frame:
    """Run a parameterised query and return a DataFrame.

    Args:
        conn: An open connection.
        sql: SQL text with ``:name`` placeholders only.
        params: Values for the placeholders.

    Returns:
        The result set; empty (with its columns) when no rows match.
    """
    result = conn.execute(text(sql), params or {})
    return pd.DataFrame(result.fetchall(), columns=list(result.keys()))


def empty_frame(columns: list[str]) -> Frame:
    """Build an empty DataFrame with the given columns.

    Args:
        columns: Column names.

    Returns:
        A zero-row DataFrame.
    """
    return pd.DataFrame({c: [] for c in columns})
