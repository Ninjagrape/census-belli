"""Model-stage queries: pooled runs and the ranking each one wrote."""

from __future__ import annotations

from sqlalchemy.engine import Connection

from pipeline.inspector.frames import Frame, query_frame

__all__ = ["pooled_runs", "run_ranking"]


def pooled_runs(conn: Connection) -> Frame:
    """Every pooled model run, newest first, primary and sensitivity alike.

    The published-scale sensitivity run has its own ``model_type`` suffix, so
    the ``scale`` column is what tells the two apart at a glance.

    Args:
        conn: An open connection.

    Returns:
        One row per pooled run; ``diagnostics`` is JSON text.
    """
    return query_frame(
        conn,
        """
        SELECT run_id, run_name, model_type,
               COALESCE(config ->> 'scale', 'neutral') AS scale,
               started_at, completed_at,
               diagnostics ->> 'converged' AS converged,
               diagnostics::text AS diagnostics
        FROM model_runs
        WHERE model_type IN ('bradley_terry_hierarchical',
                             'bradley_terry_hierarchical_published')
        ORDER BY run_id DESC
        """,
    )


def run_ranking(conn: Connection, run_id: int, limit: int = 200) -> Frame:
    """One run's ranking with credible intervals, best rank first.

    Args:
        conn: An open connection.
        run_id: The pooled run.
        limit: Maximum rows.

    Returns:
        Skill and WAR summaries joined to general names. Empty when the run
        wrote no estimates (a round failed to converge).
    """
    return query_frame(
        conn,
        """
        SELECT e.rank, g.canonical_name, e.n_battles,
               e.total_war, e.war_per_battle, e.war_sd,
               e.skill_mean, e.skill_sd, e.skill_ci_lo, e.skill_ci_hi,
               e.skill_ci90_lo, e.skill_ci90_hi
        FROM general_skill_estimates e
        JOIN generals g ON g.general_id = e.general_id
        WHERE e.run_id = :run_id
        ORDER BY e.rank NULLS LAST, g.canonical_name
        LIMIT :limit
        """,
        {"run_id": int(run_id), "limit": int(limit)},
    )
