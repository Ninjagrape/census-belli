"""Resolve-stage queries: generals, aliases and commanders per side."""

from __future__ import annotations

from sqlalchemy.engine import Connection

from pipeline.inspector.frames import YEAR_FILTER, Frame, query_frame, year_params

__all__ = ["commanders_per_side", "generals_missing_wikidata", "generals_summary"]

_GENERALS_SELECT = f"""
    SELECT g.general_id, g.canonical_name, g.wikidata_id, g.nationality, g.era,
           COALESCE(bc.n_battles, 0) AS n_battles,
           COALESCE(al.n_aliases, 0) AS n_aliases
    FROM generals g
    LEFT JOIN (
        SELECT c.general_id, COUNT(DISTINCT c.battle_id) AS n_battles
        FROM battle_commanders c
        JOIN battles b ON b.battle_id = c.battle_id
        WHERE {YEAR_FILTER}
        GROUP BY c.general_id
    ) bc ON bc.general_id = g.general_id
    LEFT JOIN (
        SELECT general_id, COUNT(*) AS n_aliases
        FROM general_aliases
        GROUP BY general_id
    ) al ON al.general_id = g.general_id
"""


def generals_summary(
    conn: Connection, year_range: tuple[int, int] | None = None, limit: int = 1000
) -> Frame:
    """Generals with battle and alias counts, busiest first.

    Args:
        conn: An open connection.
        year_range: Optional inclusive astronomical year filter on battles.
        limit: Maximum rows.

    Returns:
        One row per general.
    """
    return query_frame(
        conn,
        f"{_GENERALS_SELECT} ORDER BY n_battles DESC, g.canonical_name LIMIT :limit",
        {**year_params(year_range), "limit": limit},
    )


def generals_missing_wikidata(
    conn: Connection, year_range: tuple[int, int] | None = None, limit: int = 1000
) -> Frame:
    """Generals the resolve stage could not link to a Wikidata id.

    Args:
        conn: An open connection.
        year_range: Optional inclusive astronomical year filter on battles.
        limit: Maximum rows.

    Returns:
        One row per unlinked general.
    """
    return query_frame(
        conn,
        f"{_GENERALS_SELECT} WHERE g.wikidata_id IS NULL "
        "ORDER BY n_battles DESC, g.canonical_name LIMIT :limit",
        {**year_params(year_range), "limit": limit},
    )


def commanders_per_side(conn: Connection, year_range: tuple[int, int] | None = None) -> Frame:
    """Distribution of the number of commanders per side.

    Sides with no commander rows count as zero.

    Args:
        conn: An open connection.
        year_range: Optional inclusive astronomical year filter.

    Returns:
        Columns ``n_commanders`` and ``n_sides``.
    """
    return query_frame(
        conn,
        f"""
        SELECT n_commanders, COUNT(*) AS n_sides
        FROM (
            SELECT s.side_id, COUNT(c.bc_id) AS n_commanders
            FROM battle_sides s
            JOIN battles b ON b.battle_id = s.battle_id
            LEFT JOIN battle_commanders c ON c.side_id = s.side_id
            WHERE {YEAR_FILTER}
            GROUP BY s.side_id
        ) t
        GROUP BY n_commanders
        ORDER BY n_commanders
        """,
        year_params(year_range),
    )
