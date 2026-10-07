"""Battle drill-down queries.

``date_start`` is only ever cast to text in SQL: a BC date cannot be built as
a Python ``date``.
"""

from __future__ import annotations

from sqlalchemy.engine import Connection

from pipeline.inspector.frames import YEAR_FILTER, Frame, query_frame, year_params

__all__ = [
    "battle_casualty_reports",
    "battle_commanders",
    "battle_crawl_log",
    "battle_llm_calls",
    "battle_missing_data",
    "battle_row",
    "battle_sides",
    "battle_troop_reports",
    "battle_year_bounds",
    "search_battles",
]

_LLM_RESPONSE_CHARS = 2000


def battle_year_bounds(conn: Connection) -> Frame:
    """Smallest and largest astronomical year among dated battles.

    Args:
        conn: An open connection.

    Returns:
        One row with ``min_year``, ``max_year`` (NULL when no battle is dated).
    """
    return query_frame(
        conn,
        "SELECT MIN(year_astronomical) AS min_year, "
        "MAX(year_astronomical) AS max_year FROM battles",
    )


def search_battles(
    conn: Connection,
    term: str,
    year_range: tuple[int, int] | None = None,
    limit: int = 100,
) -> Frame:
    """Find battles by case-insensitive name substring.

    Args:
        conn: An open connection.
        term: Text to look for; ``%`` and ``_`` in it are matched literally.
        year_range: Optional inclusive astronomical year filter.
        limit: Maximum rows.

    Returns:
        Matching battles.
    """
    escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return query_frame(
        conn,
        f"""
        SELECT b.battle_id, b.name, b.year_astronomical, b.date_start::text AS date_start,
               b.battle_type::text AS battle_type
        FROM battles b
        WHERE b.name ILIKE '%' || :term || '%' AND {YEAR_FILTER}
        ORDER BY b.name
        LIMIT :limit
        """,
        {**year_params(year_range), "term": escaped, "limit": limit},
    )


def battle_row(conn: Connection, battle_id: int) -> Frame:
    """The battles row for one battle.

    Args:
        conn: An open connection.
        battle_id: The battle.

    Returns:
        Zero or one row.
    """
    return query_frame(
        conn,
        """
        SELECT battle_id, name, wikidata_id, wikipedia_url,
               date_start::text AS date_start, date_end::text AS date_end,
               date_precision, year_astronomical, location_name, latitude, longitude,
               battle_type::text AS battle_type,
               array_to_string(terrain, ', ') AS terrain, fortified, weather,
               data_quality_score, n_sources, needs_review, review_notes
        FROM battles
        WHERE battle_id = :battle_id
        """,
        {"battle_id": battle_id},
    )


def battle_sides(conn: Connection, battle_id: int) -> Frame:
    """Sides of one battle with every estimate column.

    Args:
        conn: An open connection.
        battle_id: The battle.

    Returns:
        One row per side.
    """
    return query_frame(
        conn,
        """
        SELECT side_id, side_label, polity, outcome::text AS outcome, is_defender,
               est_troops_total, est_troops_total_lo, est_troops_total_hi,
               est_troops_neutral, est_troops_neutral_lo, est_troops_neutral_hi,
               est_troops_method, est_troops_n_reports, est_troops_n_sources,
               est_casualties, est_casualties_lo, est_casualties_hi, est_casualties_method
        FROM battle_sides
        WHERE battle_id = :battle_id
        ORDER BY side_id
        """,
        {"battle_id": battle_id},
    )


def battle_commanders(conn: Connection, battle_id: int) -> Frame:
    """Commanders of one battle with roles, weights and evidence.

    Args:
        conn: An open connection.
        battle_id: The battle.

    Returns:
        One row per commander.
    """
    return query_frame(
        conn,
        """
        SELECT c.bc_id, c.side_id, g.canonical_name AS general_name, g.wikidata_id,
               c.command_role::text AS command_role, c.hierarchy_rank,
               c.reports_to_bc_id, c.attribution_weight, c.attribution_method,
               c.confidence, c.role_evidence
        FROM battle_commanders c
        JOIN generals g ON g.general_id = c.general_id
        WHERE c.battle_id = :battle_id
        ORDER BY c.side_id, c.hierarchy_rank, c.bc_id
        """,
        {"battle_id": battle_id},
    )


def battle_troop_reports(conn: Connection, battle_id: int) -> Frame:
    """Troop reports for one battle with source titles.

    Args:
        conn: An open connection.
        battle_id: The battle.

    Returns:
        One row per report.
    """
    return query_frame(
        conn,
        """
        SELECT r.report_id, r.side_id, r.reported_value, r.scope, r.branch::text AS branch,
               r.is_estimate, r.is_upper_bound, r.is_lower_bound,
               src.title AS source_title, r.extracted_context
        FROM troop_reports r
        JOIN battle_sides s ON s.side_id = r.side_id
        JOIN sources src ON src.source_id = r.source_id
        WHERE s.battle_id = :battle_id
        ORDER BY r.side_id, r.report_id
        """,
        {"battle_id": battle_id},
    )


def battle_casualty_reports(conn: Connection, battle_id: int) -> Frame:
    """Casualty reports for one battle with source titles.

    Args:
        conn: An open connection.
        battle_id: The battle.

    Returns:
        One row per report.
    """
    return query_frame(
        conn,
        """
        SELECT r.report_id, r.side_id, r.casualty_type, r.reported_value, r.is_estimate,
               src.title AS source_title, r.extracted_context
        FROM casualty_reports r
        JOIN battle_sides s ON s.side_id = r.side_id
        JOIN sources src ON src.source_id = r.source_id
        WHERE s.battle_id = :battle_id
        ORDER BY r.side_id, r.report_id
        """,
        {"battle_id": battle_id},
    )


def battle_missing_data(conn: Connection, battle_id: int) -> Frame:
    """Missing-data log rows for one battle.

    Args:
        conn: An open connection.
        battle_id: The battle.

    Returns:
        One row per log entry.
    """
    return query_frame(
        conn,
        """
        SELECT log_id, side_id, field_name, missingness_class::text AS missingness_class,
               was_imputed, imputation_method, imputed_value, imputed_lo, imputed_hi,
               imputation_round, notes
        FROM missing_data_log
        WHERE battle_id = :battle_id
        ORDER BY log_id
        """,
        {"battle_id": battle_id},
    )


def battle_llm_calls(conn: Connection, battle_id: int) -> Frame:
    """LLM calls made for one battle; the response is truncated for display.

    Args:
        conn: An open connection.
        battle_id: The battle.

    Returns:
        One row per call.
    """
    return query_frame(
        conn,
        """
        SELECT call_id, stage, provider, model, status, prompt_tokens, completion_tokens,
               cost_usd::float8 AS cost_usd, latency_ms, error,
               left(response_json::text, :chars) AS response_json, created_at
        FROM llm_calls
        WHERE battle_id = :battle_id
        ORDER BY call_id
        """,
        {"battle_id": battle_id, "chars": _LLM_RESPONSE_CHARS},
    )


def battle_crawl_log(conn: Connection, battle_id: int) -> Frame:
    """Crawl log rows for one battle.

    Args:
        conn: An open connection.
        battle_id: The battle.

    Returns:
        One row per fetch.
    """
    return query_frame(
        conn,
        """
        SELECT crawl_id, url, fetched_at, http_status,
               extraction_method::text AS extraction_method, fields_extracted, errors
        FROM crawl_log
        WHERE battle_id = :battle_id
        ORDER BY fetched_at
        """,
        {"battle_id": battle_id},
    )
