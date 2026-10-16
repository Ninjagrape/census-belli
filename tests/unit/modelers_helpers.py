"""Builders shared by the modelers unit tests. Not a test module."""

from __future__ import annotations

from typing import Any

from pipeline.modelers.records import CommanderInput, SideInput


def cmd(general_id: int, weight: float = 1.0, method: str = "rule_single") -> CommanderInput:
    """One commander."""
    return CommanderInput(general_id, weight, method)


def side(
    side_id: int,
    battle_id: int,
    generals: tuple[int, ...] = (),
    *,
    outcome: str | None = "victory",
    year: int | None = 1800,
    is_defender: bool | None = None,
    method: str = "rule_single",
    weights: tuple[float, ...] | None = None,
    battle_type: str | None = None,
    log_troops: float | None = None,
) -> SideInput:
    """A side with the given generals, equal weights unless given."""
    ws = weights or tuple(1.0 / len(generals) for _ in generals)
    return SideInput(
        side_id=side_id,
        battle_id=battle_id,
        year=year,
        outcome=outcome,
        is_defender=is_defender,
        commanders=tuple(cmd(g, w, method) for g, w in zip(generals, ws, strict=True)),
        battle_type=battle_type,
        log_troops=log_troops,
    )


def battle(
    battle_id: int,
    a: tuple[int, ...],
    b: tuple[int, ...],
    *,
    a_outcome: str | None = "victory",
    b_outcome: str | None = "defeat",
    **kwargs: Any,
) -> list[SideInput]:
    """Two sides with side ids ``2 * battle_id`` and ``2 * battle_id + 1``."""
    return [
        side(2 * battle_id, battle_id, a, outcome=a_outcome, **kwargs),
        side(2 * battle_id + 1, battle_id, b, outcome=b_outcome, **kwargs),
    ]
