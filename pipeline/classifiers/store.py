"""
Writing classify's decisions into the database.

Three things about the writes are load-bearing, all following from
docs/handover-log.md §12.3's rule for this stage specifically: classify is the last
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
    "log_unlogged_troop_totals",
    "write_terrain",
    "write_defender_flags",
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

# The one place a non-'unknown' battle_type is rewritten. The EXISTS guard
# limits it to a type an earlier classify run chose by rule (its log note says
# so), so a type extract or a person set is untouched; fortified is cleared
# because only a rule-derived siege or similar set it. Once campaign, a battle
# no longer matches, so a re-run is a no-op.
_REPLACE_RULE_TYPE_WITH_CAMPAIGN = text(
    """
    UPDATE battles SET
        battle_type = 'campaign',
        fortified   = NULL
    WHERE battle_id = :battle_id
      AND battle_type NOT IN ('unknown', 'campaign')
      AND EXISTS (
          SELECT 1 FROM missing_data_log mdl
          WHERE mdl.battle_id = battles.battle_id
            AND mdl.field_name = 'battle_type'
            AND mdl.notes LIKE 'classify: inferred%'
      )
    """
)

_NOTE_CAMPAIGN_REPLACEMENT = text(
    """
    UPDATE missing_data_log SET
        notes = notes || E'\\n' || :note
    WHERE battle_id = :battle_id
      AND field_name = 'battle_type'
      AND missingness_class = 'observed'
      AND notes LIKE 'classify: inferred%'
    """
)

_UPDATE_TERRAIN = text(
    """
    UPDATE battles SET terrain = CAST(:terrain AS TEXT[])
    WHERE battle_id = :battle_id
      AND (terrain IS NULL OR cardinality(terrain) = 0)
    """
)

_MARK_TERRAIN_OBSERVED = text(
    """
    UPDATE missing_data_log SET
        missingness_class = 'observed',
        notes = CASE
            WHEN notes IS NULL OR notes = '' THEN :note
            ELSE notes || E'\\n' || :note
        END
    WHERE battle_id = :battle_id
      AND field_name = 'terrain'
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
        if decision.replaces_rule_type:
            replaced = conn.execute(
                _REPLACE_RULE_TYPE_WITH_CAMPAIGN, {"battle_id": decision.battle_id}
            )
            if int(replaced.rowcount or 0):
                typed += 1
                conn.execute(
                    _NOTE_CAMPAIGN_REPLACEMENT,
                    {
                        "battle_id": decision.battle_id,
                        "note": (
                            f"{_NOTE_PREFIX}reclassified {decision.battle_type!r} "
                            f"({decision.evidence})"
                        ),
                    },
                )
            continue
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


_LOG_UNLOGGED_TROOP_TOTALS = text(
    """
    INSERT INTO missing_data_log (battle_id, side_id, field_name, notes)
    SELECT bs.battle_id, bs.side_id, 'troop_total',
           :note
    FROM battle_sides bs
    WHERE bs.est_troops_total IS NULL
      AND NOT EXISTS (
          SELECT 1 FROM missing_data_log mdl
          WHERE mdl.side_id = bs.side_id AND mdl.field_name = 'troop_total'
      )
    """
)


def log_unlogged_troop_totals(conn: Any) -> int:
    """Log a ``troop_total`` gap for every side that has none and no row for it.

    Extract logs a side with no reports and reconcile logs one whose reports
    were all unusable, but a side can still end up with neither: the extract
    rows are cleared when a battle is re-extracted, and a side with no reports
    never reaches reconcile at all. Impute selects its targets from this log,
    so an unlogged gap is silently never imputed. Classify runs after both
    stages and owns the missingness classes, so it closes the hole here and
    the new rows go through the same heuristics as the others.

    Args:
        conn: An open database connection.

    Returns:
        How many rows were inserted.
    """
    result = conn.execute(
        _LOG_UNLOGGED_TROOP_TOTALS,
        {"note": f"{_NOTE_PREFIX}no reconciled troop estimate and no earlier log entry"},
    )
    return int(result.rowcount or 0)


_UPDATE_DEFENDER = text(
    """
    UPDATE battle_sides SET is_defender = :is_defender
    WHERE side_id = :side_id AND is_defender IS NULL
    """
)


def write_defender_flags(conn: Any, flags: dict[int, bool]) -> int:
    """Set ``is_defender`` on sides where it is still unset.

    Args:
        conn: An open database connection.
        flags: ``side_id`` to the flag, from
            :func:`pipeline.classifiers.defender.infer_siege_defenders`.

    Returns:
        How many sides were updated. A flag already on the row is never
        overwritten.
    """
    written = 0
    for side_id, is_defender in flags.items():
        result = conn.execute(_UPDATE_DEFENDER, {"side_id": side_id, "is_defender": is_defender})
        written += int(result.rowcount or 0)
    return written


def write_terrain(conn: Any, terrain_by_battle: dict[int, tuple[list[str], str]]) -> int:
    """Write inferred terrain tags where the column is still empty.

    Args:
        conn: An open database connection.
        terrain_by_battle: ``battle_id`` to ``(tags, evidence)``. A battle with
            no tags is skipped, so it keeps its ``missing_data_log`` row.

    Returns:
        How many battles were given terrain. A tag list already on the row is
        never overwritten, and the battle's 'terrain' log row moves from
        'unclassified' to 'observed' once.
    """
    written = 0
    for battle_id, (tags, evidence) in terrain_by_battle.items():
        if not tags:
            continue
        result = conn.execute(_UPDATE_TERRAIN, {"battle_id": battle_id, "terrain": tags})
        if not int(result.rowcount or 0):
            continue
        written += 1
        conn.execute(
            _MARK_TERRAIN_OBSERVED,
            {"battle_id": battle_id, "note": f"{_NOTE_PREFIX}inferred terrain {tags} ({evidence})"},
        )
    return written
