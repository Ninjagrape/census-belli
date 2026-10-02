"""
Writing classify's decisions into the database.

Three things about the writes are load-bearing, all following from
handover.md §12.3's rule for this stage specifically: classify is the last
writer of ``command_role``, and unlike resolve it does not need to defer to
anything already on the row.

**Classify is authoritative on role.** Where resolve's upsert only fills a
``command_role`` still at 'unknown', classify overwrites it unconditionally --
a side's role/rank/weight/method/confidence are recomputed and rewritten
every run, including one resolve itself already guessed at.

**Battle type and missingness never move backwards.** A ``battles.battle_type``
already something other than 'unknown' is never touched (the guard is in the
``WHERE`` clause, not just in Python, per the brief), and ``fortified`` is
only ever added to, never cleared: ``COALESCE(:fortified, fortified)`` keeps
whatever the row already had when this run has no fortification evidence.
A ``missing_data_log`` row moves out of 'unclassified' at most once; the same
``WHERE missingness_class = 'unclassified'`` guard is what makes a second run
a no-op rather than a duplicated note.

**Idempotent by construction.** Every write here is a plain ``UPDATE`` keyed
by a primary key already on the row (``bc_id``, ``battle_id`` or ``log_id``),
so re-running with the same decisions produces the same final values -- there
is no insert-then-conflict path to converge, unlike resolve's identities.
"""

from __future__ import annotations

from typing import Any, Final

import structlog
from sqlalchemy import text

from pipeline.classifiers.records import BattleTypeDecision, MissingnessDecision, RoleDecision

__all__ = [
    "write_battle_type_decisions",
    "write_missingness_decisions",
    "write_role_decisions",
]

logger = structlog.get_logger()

_NOTE_PREFIX: Final[str] = "classify: "

_UPDATE_ROLE = text(
    """
    UPDATE battle_commanders SET
        command_role       = CAST(:command_role AS command_role),
        hierarchy_rank     = :hierarchy_rank,
        reports_to_bc_id   = :reports_to_bc_id,
        attribution_weight = :attribution_weight,
        attribution_method = :attribution_method,
        confidence         = :confidence
    WHERE bc_id = :bc_id
    """
)

_UPDATE_BATTLE_TYPE = text(
    """
    UPDATE battles SET
        battle_type = CAST(:battle_type AS battle_type),
        fortified   = COALESCE(:fortified, fortified)
    WHERE battle_id = :battle_id
      AND battle_type = 'unknown'
    """
)

# Guarded on missingness_class = 'unclassified' in the WHERE clause itself,
# not only by the caller: a second run of classify sees a row this UPDATE
# already moved to 'observed' and it no longer matches, so the note is
# written exactly once. That guard is what makes a re-run idempotent rather
# than merely harmless.
_MARK_BATTLE_TYPE_OBSERVED = text(
    """
    UPDATE missing_data_log SET
        missingness_class = 'observed',
        notes = CASE
            WHEN notes IS NULL OR notes = '' THEN :note
            ELSE notes || E'\\n' || :note
        END
    WHERE battle_id = :battle_id
      AND field_name = 'battle_type'
      AND missingness_class = 'unclassified'
    """
)

_UPDATE_MISSINGNESS = text(
    """
    UPDATE missing_data_log SET
        missingness_class = CAST(:missingness_class AS missingness_class),
        notes = CASE
            WHEN notes IS NULL OR notes = '' THEN :note
            ELSE notes || E'\\n' || :note
        END
    WHERE log_id = :log_id
      AND missingness_class = 'unclassified'
    """
)


def write_role_decisions(conn: Any, decisions: list[RoleDecision]) -> int:
    """Write every command-role decision onto its ``battle_commanders`` row.

    Args:
        conn: An open database connection.
        decisions: Every role decision this run produced, from the
            deterministic pass and, where one was needed and answered, the
            LLM pass.

    Returns:
        How many rows were updated.
    """
    for decision in decisions:
        conn.execute(
            _UPDATE_ROLE,
            {
                "bc_id": decision.bc_id,
                "command_role": decision.command_role,
                "hierarchy_rank": decision.hierarchy_rank,
                "reports_to_bc_id": decision.reports_to_bc_id,
                "attribution_weight": decision.attribution_weight,
                "attribution_method": decision.attribution_method,
                "confidence": decision.confidence,
            },
        )
    return len(decisions)


def write_battle_type_decisions(conn: Any, decisions: list[BattleTypeDecision]) -> int:
    """Write every battle-type decision, and log the ones that resolved one.

    Args:
        conn: An open database connection.
        decisions: One decision per battle whose ``battle_type`` was still
            'unknown' when the stage read it.

    Returns:
        How many battles had a type other than 'unknown' inferred and
        recorded (an 'unknown' decision still runs the ``UPDATE`` -- it is a
        harmless no-op since the row is already 'unknown' -- but marks no
        ``missing_data_log`` row 'observed', since nothing was actually
        observed).
    """
    typed = 0
    for decision in decisions:
        conn.execute(
            _UPDATE_BATTLE_TYPE,
            {
                "battle_id": decision.battle_id,
                "battle_type": decision.battle_type,
                "fortified": decision.fortified,
            },
        )
        if decision.battle_type == "unknown":
            continue
        typed += 1
        note = f"{_NOTE_PREFIX}inferred {decision.battle_type!r} ({decision.evidence})"
        conn.execute(
            _MARK_BATTLE_TYPE_OBSERVED,
            {"battle_id": decision.battle_id, "note": note},
        )
    return typed


def write_missingness_decisions(conn: Any, decisions: list[MissingnessDecision]) -> int:
    """Write every missingness decision onto its ``missing_data_log`` row.

    Args:
        conn: An open database connection.
        decisions: Every row :func:`pipeline.classifiers.missingness
            .classify_missingness` reached a decision for. A row it declined
            to classify (returned None) is not represented here and is left
            'unclassified', which is the correct outcome for it.

    Returns:
        How many rows were updated.
    """
    written = 0
    for decision in decisions:
        note = f"{_NOTE_PREFIX}{decision.reason}"
        conn.execute(
            _UPDATE_MISSINGNESS,
            {
                "log_id": decision.log_id,
                "missingness_class": decision.missingness_class,
                "note": note,
            },
        )
        written += 1
    return written
