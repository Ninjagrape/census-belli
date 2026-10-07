"""Classify and impute queries: roles, attribution, missingness."""

from __future__ import annotations

from sqlalchemy.engine import Connection

from pipeline.inspector.frames import YEAR_FILTER, Frame, query_frame, year_params

__all__ = [
    "attribution_weights",
    "defender_coverage",
    "imputed_rows",
    "missing_data_matrix",
    "review_queue",
    "role_counts",
    "side_weight_sums",
]


def role_counts(conn: Connection, year_range: tuple[int, int] | None = None) -> Frame:
    """Commander rows by role and attribution method.

    Args:
        conn: An open connection.
        year_range: Optional inclusive astronomical year filter.

    Returns:
        Columns ``command_role``, ``attribution_method``, ``n``.
    """
    return query_frame(
        conn,
        f"""
        SELECT c.command_role::text AS command_role,
               COALESCE(c.attribution_method, 'none') AS attribution_method,
               COUNT(*) AS n
        FROM battle_commanders c
        JOIN battles b ON b.battle_id = c.battle_id
        WHERE {YEAR_FILTER}
        GROUP BY 1, 2
        ORDER BY n DESC
        """,
        year_params(year_range),
    )


def attribution_weights(conn: Connection, year_range: tuple[int, int] | None = None) -> Frame:
    """Each commander row's attribution weight and its side's total.

    Args:
        conn: An open connection.
        year_range: Optional inclusive astronomical year filter.

    Returns:
        One row per commander, with ``side_weight_sum``.
    """
    return query_frame(
        conn,
        f"""
        SELECT c.bc_id, c.battle_id, b.name AS battle_name, c.side_id,
               g.canonical_name AS general_name, c.command_role::text AS command_role,
               c.hierarchy_rank, c.attribution_weight, c.attribution_method,
               c.confidence,
               SUM(c.attribution_weight) OVER (PARTITION BY c.side_id) AS side_weight_sum
        FROM battle_commanders c
        JOIN battles b ON b.battle_id = c.battle_id
        JOIN generals g ON g.general_id = c.general_id
        WHERE {YEAR_FILTER}
        ORDER BY c.battle_id, c.side_id, c.hierarchy_rank
        """,
        year_params(year_range),
    )


def side_weight_sums(conn: Connection, year_range: tuple[int, int] | None = None) -> Frame:
    """Sum of attribution weights per side, for sides with commanders.

    Args:
        conn: An open connection.
        year_range: Optional inclusive astronomical year filter.

    Returns:
        Columns ``side_id``, ``battle_name``, ``n_commanders``, ``weight_sum``.
    """
    return query_frame(
        conn,
        f"""
        SELECT c.side_id, b.name AS battle_name, COUNT(*) AS n_commanders,
               SUM(c.attribution_weight) AS weight_sum
        FROM battle_commanders c
        JOIN battles b ON b.battle_id = c.battle_id
        WHERE {YEAR_FILTER}
        GROUP BY c.side_id, b.name
        ORDER BY c.side_id
        """,
        year_params(year_range),
    )


def review_queue(conn: Connection, limit: int = 500) -> Frame:
    """The low-confidence attribution review queue.

    Args:
        conn: An open connection.
        limit: Maximum rows.

    Returns:
        Rows from ``v_attribution_review_queue``.
    """
    return query_frame(
        conn,
        """
        SELECT canonical_name, battle_name, command_role::text AS command_role,
               confidence, attribution_method, role_evidence
        FROM v_attribution_review_queue
        LIMIT :limit
        """,
        {"limit": limit},
    )


def missing_data_matrix(conn: Connection, year_range: tuple[int, int] | None = None) -> Frame:
    """Missing-data log rows by field and missingness class.

    Args:
        conn: An open connection.
        year_range: Optional inclusive astronomical year filter.

    Returns:
        Columns ``field_name``, ``missingness_class``, ``n``.
    """
    return query_frame(
        conn,
        f"""
        SELECT m.field_name, m.missingness_class::text AS missingness_class,
               COUNT(*) AS n
        FROM missing_data_log m
        JOIN battles b ON b.battle_id = m.battle_id
        WHERE {YEAR_FILTER}
        GROUP BY 1, 2
        ORDER BY 1, 2
        """,
        year_params(year_range),
    )


def defender_coverage(conn: Connection, year_range: tuple[int, int] | None = None) -> Frame:
    """How many sides have ``is_defender`` set.

    Args:
        conn: An open connection.
        year_range: Optional inclusive astronomical year filter.

    Returns:
        Columns ``is_defender`` (defender, attacker, unknown) and ``n_sides``.
    """
    return query_frame(
        conn,
        f"""
        SELECT CASE WHEN s.is_defender IS NULL THEN 'unknown'
                    WHEN s.is_defender THEN 'defender' ELSE 'attacker' END AS is_defender,
               COUNT(*) AS n_sides
        FROM battle_sides s
        JOIN battles b ON b.battle_id = s.battle_id
        WHERE {YEAR_FILTER}
        GROUP BY 1
        ORDER BY 1
        """,
        year_params(year_range),
    )


def imputed_rows(conn: Connection, year_range: tuple[int, int] | None = None) -> Frame:
    """Missing-data rows that were imputed.

    Args:
        conn: An open connection.
        year_range: Optional inclusive astronomical year filter.

    Returns:
        One row per imputed log entry.
    """
    return query_frame(
        conn,
        f"""
        SELECT m.log_id, m.battle_id, b.name AS battle_name, m.side_id,
               s.side_label, m.field_name, m.missingness_class::text AS missingness_class,
               m.imputation_method, m.imputed_value, m.imputed_lo, m.imputed_hi,
               m.imputation_round, m.notes
        FROM missing_data_log m
        JOIN battles b ON b.battle_id = m.battle_id
        LEFT JOIN battle_sides s ON s.side_id = m.side_id
        WHERE m.was_imputed AND {YEAR_FILTER}
        ORDER BY m.battle_id, m.side_id, m.log_id
        """,
        year_params(year_range),
    )
