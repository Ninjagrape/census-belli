"""Reading the impute stage's inputs from the database.

Years come from ``battles.year_astronomical``, never ``date_start``: a BC date
cannot be decoded in Python (docs/handover-log.md 5.1).

Two keyed covariates are derived here, both as plain functions so they can be
tested without a database:

- **Polity** is keyed on the Wikidata id when the side has one, else on the
  polity string, else ``'unknown'``. The id is stabler than the label
  ("Kingdom of France" versus "France").
- **Polity and commander 'unknown'** is one pooled level, on purpose. A side
  with no polity (or no commander) is not a random draw from the polity
  population: missing attribution tracks obscurity, so the pooled level acts as
  a missing-indicator category and absorbs that shift instead of smearing it
  into the named levels.
- **Commander** is the side's primary commander, the one with the largest
  ``attribution_weight`` (ties go to the lowest ``bc_id``, so the choice is
  deterministic). The effect is about army size, "this general fields large
  armies", not about skill.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from sqlalchemy import text

from pipeline.imputers.records import UNKNOWN, SideRow

__all__ = [
    "load_commanders",
    "load_mnar_side_ids",
    "load_side_rows",
    "load_sides_with_troop_reports",
    "polity_key",
    "primary_commander_keys",
    "latest_written_neutral_run_id",
    "published_run_id",
]

_LOAD_SIDES = text(
    """
    SELECT bs.side_id, bs.battle_id, b.year_astronomical,
           CAST(bs.outcome AS TEXT), CAST(b.battle_type AS TEXT),
           bs.is_defender, bs.polity, bs.wikidata_id,
           bs.est_troops_total, bs.est_troops_total_lo, bs.est_troops_total_hi
    FROM battle_sides bs
    JOIN battles b ON b.battle_id = bs.battle_id
    ORDER BY bs.side_id
    """
)

_NEUTRAL_RUN = text(
    """
    SELECT run_id FROM model_runs
    WHERE model_type = 'source_disagreement'
      AND completed_at IS NOT NULL
      AND diagnostics -> 'troops_neutral' ->> 'written' = 'true'
    ORDER BY run_id DESC
    LIMIT 1
    """
)

_LOAD_COMMANDERS = text(
    "SELECT side_id, bc_id, general_id, attribution_weight FROM battle_commanders"
)

_LOAD_MNAR = text(
    """
    SELECT DISTINCT side_id FROM missing_data_log
    WHERE field_name = 'troop_total'
      AND missingness_class = 'mnar'
      AND side_id IS NOT NULL
    """
)

_SIDES_WITH_REPORTS = text("SELECT DISTINCT side_id FROM troop_reports")

_PUBLISHED_RUN = text("SELECT MAX(est_troops_run_id) FROM battle_sides")


def polity_key(wikidata_id: str | None, polity: str | None) -> str:
    """Key a side's polity for the random effect.

    Args:
        wikidata_id: ``battle_sides.wikidata_id``.
        polity: ``battle_sides.polity``.

    Returns:
        The Wikidata id, else the stripped polity string, else ``'unknown'``.
    """
    if wikidata_id and wikidata_id.strip():
        return wikidata_id.strip()
    if polity and polity.strip():
        return polity.strip()
    return UNKNOWN


def primary_commander_keys(
    rows: Iterable[tuple[int, int, int, float | None]],
) -> dict[int, str]:
    """Pick each side's primary commander.

    Args:
        rows: ``(side_id, bc_id, general_id, attribution_weight)``.

    Returns:
        ``side_id -> 'general_{id}'`` for the commander with the largest
        attribution weight (NULL counts as 0), ties broken by lowest
        ``bc_id``. Sides with no row are absent; callers map them to
        ``'unknown'``.
    """
    best: dict[int, tuple[float, int, int]] = {}
    for side_id, bc_id, general_id, weight in rows:
        candidate = (-(0.0 if weight is None else float(weight)), int(bc_id), int(general_id))
        if side_id not in best or candidate < best[side_id]:
            best[side_id] = candidate
    return {side_id: f"general_{value[2]}" for side_id, value in best.items()}


def load_commanders(conn: Any) -> dict[int, str]:
    """Load every side's primary-commander key.

    Args:
        conn: An open database connection.

    Returns:
        See :func:`primary_commander_keys`.
    """
    rows = conn.execute(_LOAD_COMMANDERS).fetchall()
    return primary_commander_keys((int(r[0]), int(r[1]), int(r[2]), r[3]) for r in rows)


def load_mnar_side_ids(conn: Any) -> set[int]:
    """Load sides whose troop row is classified missing-not-at-random.

    Args:
        conn: An open database connection.

    Returns:
        Their side ids.
    """
    return {int(r[0]) for r in conn.execute(_LOAD_MNAR).fetchall()}


def load_sides_with_troop_reports(conn: Any) -> set[int]:
    """Sides that have at least one ``troop_reports`` row, usable or not.

    Used only to tell a side with no reports at all from one whose reports
    reconcile could not use, when counting the sides impute treats as missing.

    Args:
        conn: An open database connection.

    Returns:
        Their side ids.
    """
    return {int(r[0]) for r in conn.execute(_SIDES_WITH_REPORTS).fetchall()}


def latest_written_neutral_run_id(conn: Any) -> int | None:
    """The newest completed reconcile run whose neutral fit was written.

    Args:
        conn: An open database connection.

    Returns:
        The run id, or None if no run has written a neutral fit.
    """
    value = conn.execute(_NEUTRAL_RUN).scalar()
    return None if value is None else int(value)


def published_run_id(conn: Any) -> int | None:
    """The newest reconcile run that wrote a troop estimate.

    Args:
        conn: An open database connection.

    Returns:
        The run id, or None if no estimate has been written.
    """
    value = conn.execute(_PUBLISHED_RUN).scalar()
    return None if value is None else int(value)


def load_side_rows(conn: Any) -> list[SideRow]:
    """Load every side with its covariates.

    Args:
        conn: An open database connection.

    Returns:
        One :class:`SideRow` per ``battle_sides`` row, ordered by side id.
    """
    commanders = load_commanders(conn)
    mnar = load_mnar_side_ids(conn)
    out: list[SideRow] = []
    for row in conn.execute(_LOAD_SIDES).fetchall():
        side_id = int(row[0])
        estimate = row[8]
        published: tuple[float, float | None, float | None] | None = None
        if estimate is not None:
            published = (
                float(estimate),
                None if row[9] is None else float(row[9]),
                None if row[10] is None else float(row[10]),
            )
        out.append(
            SideRow(
                side_id=side_id,
                battle_id=int(row[1]),
                year=None if row[2] is None else int(row[2]),
                outcome=row[3],
                battle_type=row[4],
                is_defender=None if row[5] is None else bool(row[5]),
                polity_key=polity_key(row[7], row[6]),
                commander_key=commanders.get(side_id, UNKNOWN),
                mnar=side_id in mnar,
                published=published,
            )
        )
    return out


def subset_sides(sides: Sequence[SideRow], limit: int | None) -> list[SideRow]:
    """Keep every side of the first ``limit`` battles, for smoke runs.

    Args:
        sides: All sides.
        limit: Maximum number of battles, or None for all.

    Returns:
        The sides of the first ``limit`` battles by id.
    """
    if limit is None:
        return list(sides)
    keep = set(sorted({s.battle_id for s in sides})[:limit])
    return [s for s in sides if s.battle_id in keep]
