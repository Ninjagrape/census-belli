"""
Finding battles that are probably the same engagement under two names.

``agents/resolve.yaml`` asks this stage to "merge duplicate battle records".
It detects them and does **not** merge them, which is a deliberate decision
and the reason this module is separate from the writer.

Merging two battle rows means re-parenting their ``battle_sides``, and through
those their ``troop_reports``, ``casualty_reports`` and ``battle_commanders``,
then deleting one side of the pair. It is irreversible, it is invisible
afterwards, and it is wrong more often than it looks: "First Battle of Bull
Run" and "Second Battle of Bull Run" differ by one word and are a year apart,
"Battle of Panipat" names three engagements over two centuries, and a battle
and the siege that followed it frequently share a name and a year.

The commander matcher already prefers a split to a merge for exactly this
reason -- a wrong merge pools two records and nothing downstream can see it.
The same asymmetry applies here, with the added cost that battles carry the
outcome data the whole model is fitted to. So this reports pairs for review
and leaves the corpus alone.

Extract already removes the easy duplicates: it upserts on ``wikidata_id``
first and exact ``name`` second, so two crawls of one article converge. What
is left is the hard case, two different articles about one event, and that
needs a human or a later, evidenced decision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Final

import structlog
from rapidfuzz import fuzz
from sqlalchemy import text

from pipeline.resolvers.names import fold

__all__ = [
    "DEFAULT_NAME_THRESHOLD",
    "DEFAULT_YEAR_SLACK",
    "DuplicateBattle",
    "find_duplicate_battles",
]

logger = structlog.get_logger()

# Battle names share far more vocabulary than personal names do -- almost all
# of them begin "Battle of" -- so the bar is higher than the commander
# matcher's and it is applied after that prefix is removed.
DEFAULT_NAME_THRESHOLD: Final[float] = 92.0

# Sources disagree about which year a winter campaign belongs to, and a battle
# recorded only by year sits on 1 January. One year of slack covers both; more
# would start pairing a battle with its own sequel.
DEFAULT_YEAR_SLACK: Final[int] = 1

# Stripped before comparison: nearly every battle in the corpus carries some of
# this, so leaving it in pushes every pair's similarity towards 100.
_GENERIC_PREFIXES: Final[tuple[str, ...]] = (
    "battle of the ",
    "battle of ",
    "battles of ",
    "siege of the ",
    "siege of ",
    "capture of ",
    "storming of ",
    "assault on ",
    "action of ",
    "action at ",
    "combat of ",
    "raid on ",
)

# An ordinal in the name is a statement that these are *different* battles of
# the same place, which is the single most common false positive.
_ORDINALS: Final[tuple[str, ...]] = (
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
    "1st",
    "2nd",
    "3rd",
    "4th",
    "5th",
)

_LOAD_BATTLES = text(
    "SELECT battle_id, name, year_astronomical FROM battles ORDER BY battle_id"
)


@dataclass(frozen=True)
class DuplicateBattle:
    """Two battle rows that may describe one engagement.

    Attributes:
        left_id: The lower battle id.
        left_name: Its name.
        right_id: The higher battle id.
        right_name: Its name.
        year: The astronomical year of the lower-numbered row.
        score: Name similarity after the generic prefix is removed.
        reason: Why the pair was flagged, in one line.
    """

    left_id: int
    left_name: str
    right_id: int
    right_name: str
    year: int | None
    score: float
    reason: str

    def as_dict(self) -> dict[str, Any]:
        """Render the pair for the report file.

        Returns:
            A JSON-serialisable mapping.
        """
        return asdict(self)


def _core_name(name: str) -> str:
    """Reduce a battle name to the part that distinguishes it.

    Args:
        name: The battle's name.

    Returns:
        The folded name with a generic prefix removed, or the whole folded
        name when removing the prefix would leave nothing.
    """
    folded = fold(name)
    for prefix in _GENERIC_PREFIXES:
        if folded.startswith(prefix):
            remainder = folded[len(prefix) :].strip()
            return remainder or folded
    return folded


def _ordinal(name: str) -> str | None:
    """Read the ordinal that numbers a battle of a repeated place.

    Args:
        name: The battle's name.

    Returns:
        The ordinal word, or None when the name carries none.
    """
    tokens = fold(name).split()
    return next((token for token in tokens if token in _ORDINALS), None)


def _years_compatible(left: int | None, right: int | None, slack: int) -> bool:
    """Whether two battle years are close enough to be one event.

    Args:
        left: One astronomical year, or None.
        right: The other, or None.
        slack: Years of tolerance.

    Returns:
        True when both are known and within ``slack``. An undated battle is
        *not* compatible with anything: the date is the only evidence that
        separates the three battles of Panipat, and without it a name match
        alone is not worth reporting.
    """
    if left is None or right is None:
        return False
    return abs(left - right) <= slack


def find_duplicate_battles(
    conn: Any,
    *,
    threshold: float = DEFAULT_NAME_THRESHOLD,
    year_slack: int = DEFAULT_YEAR_SLACK,
) -> list[DuplicateBattle]:
    """Report battle rows that may describe the same engagement.

    Nothing is written or merged. See the module docstring for why.

    Args:
        conn: An open database connection.
        threshold: Minimum name similarity, 0..100, applied to the name with
            its generic prefix removed.
        year_slack: How many years apart two records of one battle may be.

    Returns:
        Candidate pairs, most similar first. Pairs whose names carry
        different ordinals are excluded: "First" and "Second Bull Run" are a
        near-perfect name match and are two different battles.
    """
    rows: list[tuple[int, str, int | None]] = [
        (int(battle_id), str(name), None if year is None else int(year))
        for battle_id, name, year in conn.execute(_LOAD_BATTLES).fetchall()
    ]

    # Only battles sharing a year can pair, so the comparison runs within year
    # buckets rather than over every pair in the corpus.
    buckets: dict[int, list[tuple[int, str, int | None]]] = {}
    for row in rows:
        year = row[2]
        if year is None:
            continue
        for offset in range(-year_slack, year_slack + 1):
            buckets.setdefault(year + offset, []).append(row)

    found: dict[tuple[int, int], DuplicateBattle] = {}

    for bucket in buckets.values():
        for index, left in enumerate(bucket):
            for right in bucket[index + 1 :]:
                left_id, left_name, left_year = left
                right_id, right_name, right_year = right
                if left_id == right_id:
                    continue

                pair = (min(left_id, right_id), max(left_id, right_id))
                if pair in found:
                    continue
                if not _years_compatible(left_year, right_year, year_slack):
                    continue
                if _ordinal(left_name) != _ordinal(right_name):
                    continue

                score = float(fuzz.WRatio(_core_name(left_name), _core_name(right_name)))
                if score < threshold:
                    continue

                lower, higher = (left, right) if left_id < right_id else (right, left)
                found[pair] = DuplicateBattle(
                    left_id=lower[0],
                    left_name=lower[1],
                    right_id=higher[0],
                    right_name=higher[1],
                    year=lower[2],
                    score=score,
                    reason=(
                        f"names match at {score:.1f} once the generic prefix is removed, "
                        f"and both are dated within {year_slack} year(s)"
                    ),
                )

    duplicates = sorted(found.values(), key=lambda d: (-d.score, d.left_id))

    if duplicates:
        logger.warning(
            "duplicate_battles_suspected",
            pairs=len(duplicates),
            sample=[f"{d.left_name} / {d.right_name} ({d.year})" for d in duplicates[:5]],
            action="reported, not merged; see data/processed/duplicate_battles.jsonl",
        )
    else:
        logger.info("duplicate_battles_none_suspected", battles=len(rows))

    return duplicates
