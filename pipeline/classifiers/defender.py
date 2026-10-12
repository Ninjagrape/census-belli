"""
Per-side defender flag (``battle_sides.is_defender``) for sieges.

Impute and model both read the flag, and nothing else wrote it. It is set
here only where the data makes it unambiguous, and left NULL otherwise: the
infobox side ordering does not say who held the place (Acre lists the
defenders first, El Arish the attackers), and a polity's outcome says who won,
not who was besieged.

The one deterministic signal is the siege's own title. "Siege of Massilia"
names the besieged place, and a side whose label carries that place name
("City of Massilia") is the garrison. Where neither side or both sides carry
the name ("Siege of Toulon (1793)" against "France" and "French Royalists"),
nothing is written.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

__all__ = ["SideLabel", "besieged_place", "infer_siege_defenders"]

_SIEGE_TITLE_RE: Final = re.compile(
    r"^\s*siege of\s+(?:the\s+)?(?P<place>.+?)\s*(?:\(.*\))?\s*$", re.I
)


@dataclass(frozen=True)
class SideLabel:
    """What the defender rule reads about one side.

    Attributes:
        side_id: The ``battle_sides`` primary key.
        label: ``side_label``, the name the infobox used.
        polity: ``polity``, or None.
    """

    side_id: int
    label: str
    polity: str | None = None


def besieged_place(battle_name: str) -> str | None:
    """Pull the besieged place out of a "Siege of X" title.

    Args:
        battle_name: The battle's name.

    Returns:
        The place name with any trailing parenthetical (a year or date range)
        removed, or None when the name is not a "Siege of ..." title.
    """
    match = _SIEGE_TITLE_RE.match(battle_name or "")
    if match is None:
        return None
    place = match.group("place").strip()
    return place or None


def infer_siege_defenders(battle_name: str, sides: Sequence[SideLabel]) -> dict[int, bool]:
    """Decide which side of a siege is the defender, when the title says.

    Args:
        battle_name: The battle's name.
        sides: Every side of the battle.

    Returns:
        ``side_id`` to ``is_defender`` for both sides of a two-sided siege in
        which exactly one side's label or polity carries the besieged place's
        name; an empty mapping in every other case.
    """
    place = besieged_place(battle_name)
    if place is None or len(sides) != 2:
        return {}

    pattern = re.compile(rf"\b{re.escape(place)}\b", re.IGNORECASE)
    holders = [s for s in sides if pattern.search(s.label) or pattern.search(s.polity or "")]
    if len(holders) != 1:
        return {}
    return {s.side_id: s is holders[0] for s in sides}
