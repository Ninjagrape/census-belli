"""Crawl and extract queries."""

from __future__ import annotations

from sqlalchemy.engine import Connection

from pipeline.inspector.frames import YEAR_FILTER, Frame, query_frame, year_params

__all__ = [
    "battles_by_century",
    "crawl_errors",
    "crawl_over_time",
    "crawl_status_counts",
    "llm_summary",
    "null_fraction_battle_sides",
    "null_fraction_battles",
    "troop_report_breakdown",
]


def crawl_status_counts(conn: Connection) -> Frame:
    """Crawl rows by HTTP status and extraction method.

    Args:
        conn: An open connection.

    Returns:
        Columns ``http_status``, ``extraction_method``, ``n``.
    """
    return query_frame(
        conn,
        """
        SELECT COALESCE(http_status::text, 'none') AS http_status,
               COALESCE(extraction_method::text, 'none') AS extraction_method,
               COUNT(*) AS n
        FROM crawl_log
        GROUP BY 1, 2
        ORDER BY n DESC
        """,
    )


def crawl_errors(conn: Connection, limit: int = 500) -> Frame:
    """Crawl rows that recorded an error, newest first.

    Args:
        conn: An open connection.
        limit: Maximum rows.

    Returns:
        The error rows.
    """
    return query_frame(
        conn,
        """
        SELECT crawl_id, url, fetched_at, http_status, errors
        FROM crawl_log
        WHERE errors IS NOT NULL AND errors <> ''
        ORDER BY fetched_at DESC
        LIMIT :limit
        """,
        {"limit": limit},
    )


def crawl_over_time(conn: Connection) -> Frame:
    """Fetches per day.

    Args:
        conn: An open connection.

    Returns:
        Columns ``day`` and ``n_fetches``.
    """
    return query_frame(
        conn,
        """
        SELECT date_trunc('day', fetched_at) AS day, COUNT(*) AS n_fetches
        FROM crawl_log
        GROUP BY 1
        ORDER BY 1
        """,
    )


def battles_by_century(conn: Connection, year_range: tuple[int, int] | None = None) -> Frame:
    """Battle counts by century start and battle type.

    The axis comes from ``year_astronomical``, never from a Python date, so BC
    battles are counted.

    Args:
        conn: An open connection.
        year_range: Optional inclusive astronomical year filter.

    Returns:
        Columns ``century_start``, ``battle_type``, ``n``.
    """
    return query_frame(
        conn,
        f"""
        SELECT (FLOOR(b.year_astronomical / 100.0) * 100)::int AS century_start,
               b.battle_type::text AS battle_type,
               COUNT(*) AS n
        FROM battles b
        WHERE b.year_astronomical IS NOT NULL AND {YEAR_FILTER}
        GROUP BY 1, 2
        ORDER BY 1, 2
        """,
        year_params(year_range),
    )


def null_fraction_battles(conn: Connection, year_range: tuple[int, int] | None = None) -> Frame:
    """Fraction of NULLs per ``battles`` column.

    Uses ``jsonb_each`` over each row so column names come from the data, not
    from interpolated SQL.

    Args:
        conn: An open connection.
        year_range: Optional inclusive astronomical year filter.

    Returns:
        Columns ``column_name``, ``n_rows``, ``n_null``, ``null_fraction``.
    """
    return query_frame(
        conn,
        f"""
        SELECT kv.key AS column_name,
               COUNT(*) AS n_rows,
               COUNT(*) FILTER (WHERE kv.value = 'null'::jsonb) AS n_null,
               (COUNT(*) FILTER (WHERE kv.value = 'null'::jsonb))::float8
                   / NULLIF(COUNT(*), 0) AS null_fraction
        FROM battles b, LATERAL jsonb_each(to_jsonb(b)) AS kv
        WHERE {YEAR_FILTER}
        GROUP BY kv.key
        ORDER BY null_fraction DESC, column_name
        """,
        year_params(year_range),
    )


def null_fraction_battle_sides(
    conn: Connection, year_range: tuple[int, int] | None = None
) -> Frame:
    """Fraction of NULLs per ``battle_sides`` column.

    Args:
        conn: An open connection.
        year_range: Optional inclusive astronomical year filter.

    Returns:
        Columns ``column_name``, ``n_rows``, ``n_null``, ``null_fraction``.
    """
    return query_frame(
        conn,
        f"""
        SELECT kv.key AS column_name,
               COUNT(*) AS n_rows,
               COUNT(*) FILTER (WHERE kv.value = 'null'::jsonb) AS n_null,
               (COUNT(*) FILTER (WHERE kv.value = 'null'::jsonb))::float8
                   / NULLIF(COUNT(*), 0) AS null_fraction
        FROM battle_sides s
        JOIN battles b ON b.battle_id = s.battle_id,
        LATERAL jsonb_each(to_jsonb(s)) AS kv
        WHERE {YEAR_FILTER}
        GROUP BY kv.key
        ORDER BY null_fraction DESC, column_name
        """,
        year_params(year_range),
    )


def troop_report_breakdown(conn: Connection, year_range: tuple[int, int] | None = None) -> Frame:
    """Troop reports by scope, branch and extraction method.

    Args:
        conn: An open connection.
        year_range: Optional inclusive astronomical year filter.

    Returns:
        Columns ``scope``, ``branch``, ``extraction_method``, ``n``.
    """
    return query_frame(
        conn,
        f"""
        SELECT COALESCE(r.scope, 'none') AS scope,
               r.branch::text AS branch,
               COALESCE(r.extraction_method::text, 'none') AS extraction_method,
               COUNT(*) AS n
        FROM troop_reports r
        JOIN battle_sides s ON s.side_id = r.side_id
        JOIN battles b ON b.battle_id = s.battle_id
        WHERE {YEAR_FILTER}
        GROUP BY 1, 2, 3
        ORDER BY n DESC
        """,
        year_params(year_range),
    )


def llm_summary(conn: Connection) -> Frame:
    """LLM calls by stage, status and model, with tokens and cost.

    Args:
        conn: An open connection.

    Returns:
        One row per stage, status and model.
    """
    return query_frame(
        conn,
        """
        SELECT stage, status, model,
               COUNT(*) AS n_calls,
               COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
               COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
               COALESCE(SUM(cost_usd), 0)::float8 AS cost_usd,
               AVG(latency_ms)::float8 AS mean_latency_ms
        FROM llm_calls
        GROUP BY stage, status, model
        ORDER BY stage, status, model
        """,
    )
