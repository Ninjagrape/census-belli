"""Independent recounts for the ``disclosures_match_db`` gate.

The coverage numbers come from the queries in :mod:`pipeline.reporters.coverage`.
Re-running those would only prove they are deterministic, so the gate counts the
same quantities a second way: plain row fetches with the tallying done in Python.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from sqlalchemy import text
from sqlalchemy.engine import Connection

from pipeline.reporters.coverage import CoverageStats
from pipeline.reporters.coverage_series import guarded

__all__ = ["COUNT_KEYS", "differences", "recount", "used_counts"]

COUNT_KEYS: Final[tuple[str, ...]] = (
    "credited_sides",
    "rule_single_sides",
    "n_headline",
    "credited_battles",
)

_COMMANDER_ROWS = text(
    "SELECT bc.battle_id, bc.general_id, bc.side_id, bc.attribution_method, bc.attribution_weight "
    "FROM battle_commanders bc JOIN battle_war_details d "
    "ON d.battle_id = bc.battle_id AND d.general_id = bc.general_id "
    "WHERE d.run_id = :run_id"
)
_BATTLES_PER_GENERAL = text(
    "SELECT general_id, COUNT(DISTINCT battle_id) FROM battle_war_details "
    "WHERE run_id = :run_id GROUP BY general_id"
)
_DATED_BATTLES = text(
    "SELECT COUNT(DISTINCT d.battle_id) FROM battle_war_details d "
    "JOIN battles b ON b.battle_id = d.battle_id "
    "WHERE d.run_id = :run_id AND b.year_astronomical IS NOT NULL"
)


def _sides(conn: Connection, run_id: int) -> tuple[int, int]:
    """Credited sides and rule_single ones; one side per (battle, general), heaviest weight."""
    best: dict[tuple[int, int], tuple[float, int, str | None]] = {}
    for battle_id, general_id, side_id, method, weight in conn.execute(
        _COMMANDER_ROWS, {"run_id": run_id}
    ):
        # Heavier weight first, then lower side id: the order the coverage query documents.
        candidate = (-(-1.0 if weight is None else float(weight)), int(side_id), method)
        key = (int(battle_id), int(general_id))
        if key not in best or candidate[:2] < best[key][:2]:
            best[key] = candidate
    chosen = {(k[0], v[1]): v[2] for k, v in best.items()}
    return len(chosen), sum(1 for m in chosen.values() if m == "rule_single")


def recount(conn: Connection, run_id: int, *, min_battles: int) -> dict[str, int | None]:
    """The gate's own counts; a quantity that cannot be queried is None.

    Args:
        conn: An open connection.
        run_id: The reported model run.
        min_battles: The headline threshold.

    Returns:
        ``COUNT_KEYS -> count``.
    """
    sides = guarded(conn, "gate_sides", lambda: _sides(conn, run_id), None)
    headline = guarded(
        conn,
        "gate_headline",
        lambda: sum(
            1 for _, n in conn.execute(_BATTLES_PER_GENERAL, {"run_id": run_id}) if n >= min_battles
        ),
        None,
    )
    battles = guarded(
        conn,
        "gate_battles",
        lambda: int(conn.execute(_DATED_BATTLES, {"run_id": run_id}).scalar_one()),
        None,
    )
    return {
        "credited_sides": None if sides is None else sides[0],
        "rule_single_sides": None if sides is None else sides[1],
        "n_headline": headline,
        "credited_battles": battles,
    }


def used_counts(stats: CoverageStats | None) -> dict[str, int | None]:
    """The same quantities as the sentences were built from them."""
    if stats is None:
        return dict.fromkeys(COUNT_KEYS)
    return {
        "credited_sides": stats.credited_sides,
        "rule_single_sides": stats.rule_single_sides,
        "n_headline": stats.n_headline,
        "credited_battles": stats.corpus_battles,
    }


def differences(
    used: Mapping[str, int | None], recounted: Mapping[str, int | None]
) -> dict[str, tuple[int | None, int | None]]:
    """``key -> (used, recounted)`` for each count that disagrees."""
    return {
        k: (used.get(k), recounted.get(k)) for k in COUNT_KEYS if used.get(k) != recounted.get(k)
    }
