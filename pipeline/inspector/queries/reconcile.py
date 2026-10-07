"""Reconcile-stage queries: reports versus the estimates fitted from them."""

from __future__ import annotations

from sqlalchemy.engine import Connection

from pipeline.inspector.frames import YEAR_FILTER, Frame, query_frame, year_params

__all__ = [
    "casualty_estimates",
    "casualty_reports_detail",
    "reconcile_sides",
    "source_bias",
    "troop_reports_detail",
]


def reconcile_sides(conn: Connection, year_range: tuple[int, int] | None = None) -> Frame:
    """Per-side troop estimates, published and neutral, with provenance.

    Args:
        conn: An open connection.
        year_range: Optional inclusive astronomical year filter.

    Returns:
        One row per side.
    """
    return query_frame(
        conn,
        f"""
        SELECT s.side_id, s.battle_id, b.name AS battle_name, b.year_astronomical,
               s.side_label, s.is_defender,
               s.est_troops_total, s.est_troops_total_lo, s.est_troops_total_hi,
               s.est_troops_neutral, s.est_troops_neutral_lo, s.est_troops_neutral_hi,
               s.est_troops_method, s.est_troops_n_reports, s.est_troops_n_sources,
               s.est_troops_run_id, s.est_troops_neutral_run_id
        FROM battle_sides s
        JOIN battles b ON b.battle_id = s.battle_id
        WHERE {YEAR_FILTER}
        ORDER BY s.battle_id, s.side_id
        """,
        year_params(year_range),
    )


def troop_reports_detail(conn: Connection, year_range: tuple[int, int] | None = None) -> Frame:
    """Every troop report with its side, source and context.

    Args:
        conn: An open connection.
        year_range: Optional inclusive astronomical year filter.

    Returns:
        One row per troop report.
    """
    return query_frame(
        conn,
        f"""
        SELECT r.report_id, r.side_id, s.battle_id, b.name AS battle_name,
               s.side_label, r.reported_value, r.scope, r.branch::text AS branch,
               r.is_estimate, r.is_upper_bound, r.is_lower_bound,
               src.source_id, src.title AS source_title,
               src.source_type::text AS source_type,
               r.extraction_method::text AS extraction_method,
               r.extracted_context
        FROM troop_reports r
        JOIN battle_sides s ON s.side_id = r.side_id
        JOIN battles b ON b.battle_id = s.battle_id
        JOIN sources src ON src.source_id = r.source_id
        WHERE {YEAR_FILTER}
        ORDER BY s.battle_id, r.side_id, r.report_id
        """,
        year_params(year_range),
    )


def source_bias(conn: Connection) -> Frame:
    """Fitted per-source bias and observation noise for troop reports.

    ``bias_n_troop_reports = 0`` means the prior, untouched.

    Args:
        conn: An open connection.

    Returns:
        One row per source.
    """
    return query_frame(
        conn,
        """
        SELECT source_id, title, source_type::text AS source_type,
               troop_bias_mu, troop_bias_sd, troop_sigma_mu, troop_sigma_sd,
               bias_n_troop_reports, bias_run_id
        FROM sources
        ORDER BY bias_n_troop_reports DESC NULLS LAST, source_id
        """,
    )


def casualty_estimates(conn: Connection, year_range: tuple[int, int] | None = None) -> Frame:
    """Per-side casualty estimates with provenance.

    Args:
        conn: An open connection.
        year_range: Optional inclusive astronomical year filter.

    Returns:
        One row per side.
    """
    return query_frame(
        conn,
        f"""
        SELECT s.side_id, s.battle_id, b.name AS battle_name, s.side_label,
               s.est_casualties, s.est_casualties_lo, s.est_casualties_hi,
               s.est_casualties_method, s.est_casualties_n_reports,
               s.est_casualties_n_sources, s.est_casualties_run_id
        FROM battle_sides s
        JOIN battles b ON b.battle_id = s.battle_id
        WHERE {YEAR_FILTER}
        ORDER BY s.battle_id, s.side_id
        """,
        year_params(year_range),
    )


def casualty_reports_detail(conn: Connection, year_range: tuple[int, int] | None = None) -> Frame:
    """Every casualty report with its side and source.

    Args:
        conn: An open connection.
        year_range: Optional inclusive astronomical year filter.

    Returns:
        One row per casualty report.
    """
    return query_frame(
        conn,
        f"""
        SELECT r.report_id, r.side_id, s.battle_id, b.name AS battle_name,
               s.side_label, r.casualty_type, r.reported_value, r.is_estimate,
               src.title AS source_title, r.extracted_context
        FROM casualty_reports r
        JOIN battle_sides s ON s.side_id = r.side_id
        JOIN battles b ON b.battle_id = s.battle_id
        JOIN sources src ON src.source_id = r.source_id
        WHERE {YEAR_FILTER}
        ORDER BY s.battle_id, r.side_id, r.report_id
        """,
        year_params(year_range),
    )
