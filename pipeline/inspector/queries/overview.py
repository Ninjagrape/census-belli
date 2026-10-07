"""Overview queries: table sizes and the latest model runs."""

from __future__ import annotations

from sqlalchemy.engine import Connection

from pipeline.inspector.frames import Frame, query_frame

__all__ = ["latest_model_runs", "model_runs", "skill_estimates", "table_counts"]


def table_counts(conn: Connection) -> Frame:
    """Exact row count of every public table.

    ``query_to_xml`` runs one static count per catalog entry, so no table name
    is ever interpolated into SQL by this code.

    Args:
        conn: An open connection.

    Returns:
        Columns ``table_name`` and ``n_rows``.
    """
    return query_frame(
        conn,
        """
        SELECT t.table_name,
               (xpath('/row/c/text()',
                      query_to_xml(format('SELECT count(*) AS c FROM %I.%I',
                                          t.table_schema, t.table_name),
                                   false, true, '')))[1]::text::bigint AS n_rows
        FROM information_schema.tables t
        WHERE t.table_schema = 'public' AND t.table_type = 'BASE TABLE'
        ORDER BY t.table_name
        """,
    )


def latest_model_runs(conn: Connection) -> Frame:
    """The most recent model run of each ``model_type``.

    Args:
        conn: An open connection.

    Returns:
        One row per model type, with timestamps and diagnostics as JSON text.
    """
    return query_frame(
        conn,
        """
        SELECT DISTINCT ON (model_type)
               model_type, run_id, run_name,
               started_at, completed_at,
               diagnostics::text AS diagnostics
        FROM model_runs
        ORDER BY model_type, run_id DESC
        """,
    )


def model_runs(conn: Connection) -> Frame:
    """Every model run, newest first.

    Args:
        conn: An open connection.

    Returns:
        One row per run.
    """
    return query_frame(
        conn,
        """
        SELECT run_id, run_name, model_type, imputation_set, started_at,
               completed_at, diagnostics::text AS diagnostics, notes
        FROM model_runs
        ORDER BY run_id DESC
        """,
    )


def skill_estimates(conn: Connection, limit: int = 200) -> Frame:
    """Latest general skill estimates, best rank first.

    Args:
        conn: An open connection.
        limit: Maximum rows.

    Returns:
        Skill rows joined to general names.
    """
    return query_frame(
        conn,
        """
        SELECT e.run_id, e.rank, g.canonical_name, e.skill_mean, e.skill_sd,
               e.skill_ci_lo, e.skill_ci_hi, e.total_war, e.n_battles
        FROM general_skill_estimates e
        JOIN generals g ON g.general_id = e.general_id
        ORDER BY e.run_id DESC, e.rank NULLS LAST
        LIMIT :limit
        """,
        {"limit": limit},
    )
