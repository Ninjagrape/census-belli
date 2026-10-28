"""Writing imputed values back to ``missing_data_log``.

One row per imputed side: ``was_imputed`` TRUE, the method string
``bayes_joint_v1_{scale}_log``, the across-round median as ``imputed_value``
and the 2.5/97.5 percentiles of the rounds as ``imputed_lo``/``imputed_hi``.
Those are percentiles of a handful of rounds (20 by default), not of every
retained posterior draw: that would mean thousands of conditional draws of the
whole dataset for a summary nobody reads closely, so ``notes`` says how many
rounds they rest on. The scale is also stated in ``notes`` because on the neutral
scale the value is a ratio-scale quantity, not a troop count, and must never be displayed as one.

Only the neutral (primary) scale may call this. The published scale is a
sensitivity run and must never touch ``missing_data_log``, so it cannot
overwrite the primary's imputation record; the stage enforces that.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import structlog
from sqlalchemy import text

from pipeline.imputers.records import ImputeCounts
from pipeline.imputers.summarise import ImputedSide

__all__ = ["imputation_method", "retire_kept_sides", "write_imputed_log"]

logger = structlog.get_logger()

_NOTE_PREFIX = "imputed_scale="

# A previous run's prefix is stripped before the new one is added, so re-runs
# replace the scale note instead of stacking one per run.
_UPDATE = text(
    """
    UPDATE missing_data_log SET
        was_imputed       = TRUE,
        imputation_method = :method,
        imputed_value     = :value,
        imputed_lo        = :lo,
        imputed_hi        = :hi,
        notes             = :note || regexp_replace(
                                COALESCE(notes, ''), '^imputed_scale=[a-z]+[^;]*; ?', ''
                            )
    WHERE side_id = :side_id AND field_name = 'troop_total'
    """
)

_INSERT = text(
    """
    INSERT INTO missing_data_log (
        battle_id, side_id, field_name, was_imputed, imputation_method,
        imputed_value, imputed_lo, imputed_hi, notes
    ) VALUES (
        :battle_id, :side_id, 'troop_total', TRUE, :method,
        :value, :lo, :hi, :note
    )
    """
)


_RETIRE = text(
    """
    DELETE FROM missing_data_log
    WHERE side_id = ANY(CAST(:side_ids AS INT[]))
      AND field_name = 'troop_total'
      AND was_imputed = TRUE
      AND imputation_method = :method
    """
)


def retire_kept_sides(conn: Any, kept_side_ids: Sequence[int], *, scale: str) -> int:
    """Remove this scale's imputation rows for sides that now carry a likelihood.

    A side that earlier runs imputed (it was weak, bound-only or unmodelled) but
    that has a point report in the current reconcile fit is no longer missing
    troop data, so its row must not say so. Only rows this method wrote are
    removed; rows other stages wrote for a side that is still imputed are kept
    and updated by :func:`write_imputed_log`.

    Args:
        conn: An open database connection.
        kept_side_ids: Sides that have a likelihood in this run.
        scale: ``'neutral'`` or ``'published'``.

    Returns:
        The number of rows deleted.
    """
    if not kept_side_ids:
        return 0
    result = conn.execute(
        _RETIRE,
        {"side_ids": [int(s) for s in kept_side_ids], "method": imputation_method(scale)},
    )
    return int(result.rowcount or 0)


def imputation_method(scale: str) -> str:
    """The ``imputation_method`` string for a scale."""
    return f"bayes_joint_v1_{scale}_log"


def _note(scale: str, n_rounds: int) -> str:
    """The notes prefix that states the scale and what the interval rests on.

    No semicolon inside the parenthesis: the update statement strips a previous
    prefix up to the first one.
    """
    interval = f"imputed_lo and imputed_hi are the 2.5 and 97.5 percentiles of {n_rounds} rounds"
    if scale == "neutral":
        return (
            f"{_NOTE_PREFIX}neutral (regime-neutral log ratio scale, for force ratios only, "
            f"not a troop count, {interval}); "
        )
    return f"{_NOTE_PREFIX}published (published troop scale, {interval}); "


def write_imputed_log(
    conn: Any,
    summaries: Sequence[ImputedSide],
    *,
    scale: str,
    n_rounds: int,
    counts: ImputeCounts,
) -> None:
    """Record every imputed side in ``missing_data_log``.

    Updates the side's existing ``troop_total`` row (extract and reconcile
    create them, classify gives them a missingness class) and inserts one when
    none exists, so a side that was imputed because its reconcile estimate was
    too weak to use still leaves a trace.

    Args:
        conn: An open database connection.
        summaries: Across-round summaries from
            :func:`pipeline.imputers.summarise.summarise_imputed`.
        scale: ``'neutral'`` or ``'published'``.
        n_rounds: Rounds the summaries were computed from, stated in the note.
        counts: Counters to accumulate row counts into.
    """
    method = imputation_method(scale)
    note = _note(scale, n_rounds)
    for item in summaries:
        params = {
            "side_id": item.side_id,
            "method": method,
            "value": item.value,
            "lo": item.lo,
            "hi": item.hi,
            "note": note,
        }
        result = conn.execute(_UPDATE, params)
        if result.rowcount:
            counts.log_rows_updated += int(result.rowcount)
        else:
            conn.execute(_INSERT, {**params, "battle_id": item.battle_id})
            counts.log_rows_inserted += 1
