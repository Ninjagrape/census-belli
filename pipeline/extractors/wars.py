"""Pick the war and the campaign a battle belongs to from its ``part_of`` names.

The merged ``part_of`` list is a union over sources: Wikidata Q-ids, DBpedia
labels, the infobox ``partof`` field and campaignbox names, in source-priority
order. Nothing in it says which entry is a war and which a campaign, so a small
keyword rule decides. It is deliberately blunt: an entry matching neither list
is ignored rather than guessed at, and a battle with no match keeps NULL
``war_id`` and ``campaign_id``.

Hierarchy is not recoverable from names. When several entries qualify, the
first one wins, because the infobox lists the parent conflict first and the
sources are merged best-first. The result is a single war and a single campaign
per battle, not a chain.
"""

from __future__ import annotations

import re
from typing import Final, NamedTuple

__all__ = ["WarCampaign", "classify_part_of"]

# Campaign words are tested first: "Western Desert campaign of the Mediterranean
# and Middle East theatre of World War II" mentions a war but is a campaign.
_CAMPAIGN_RE: Final = re.compile(r"\b(?:campaign|offensive|operation)s?\b", re.IGNORECASE)
_WAR_RE: Final = re.compile(
    r"\b(?:wars?|revolutions?|conquests?|rebellions?|crusades?|invasions?|revolts?"
    r"|uprisings?|insurrections?|conflicts?)\b",
    re.IGNORECASE,
)
_QID_RE: Final = re.compile(r"^Q\d+$")


class WarCampaign(NamedTuple):
    """The war and campaign names chosen for one battle, either may be None."""

    war: str | None
    campaign: str | None


def _is_joined_form(name: str, others: list[str]) -> bool:
    """Whether a name is another entry plus a qualifier, such as a combined label.

    Sources often add a second label that glues two entries together
    ("Third Silesian War and Seven Years' War", "Italian campaign of 1796 in the
    War of the First Coalition"). Keeping both would create a war row for a
    phrase no one calls a war.
    """
    lowered = name.lower()
    return any(
        len(other) < len(name) and other.lower() in lowered
        for other in others
        if other.lower() != lowered
    )


def classify_part_of(part_of: list[str]) -> WarCampaign:
    """Choose the war and the campaign out of a battle's ``part_of`` names.

    Args:
        part_of: Names (and Q-ids, which are skipped) the sources gave.

    Returns:
        The first war-like name and the first campaign-like name, each None
        when no entry qualifies. Q-ids are skipped because a bare id cannot be
        paired with a name safely.
    """
    names = [name.strip() for name in part_of if name and not _QID_RE.match(name.strip())]
    names = [name for name in names if name]
    unjoined = [name for name in names if not _is_joined_form(name, names)]

    campaign = next((n for n in unjoined if _CAMPAIGN_RE.search(n)), None)
    war = next((n for n in unjoined if not _CAMPAIGN_RE.search(n) and _WAR_RE.search(n)), None)
    return WarCampaign(war, campaign)
