"""
Terrain tags (``battles.terrain``) read from the article text.

Nothing wrote the column before, so impute and the model saw it empty for
every battle. The vocabulary is deliberately small and the rule conservative:
a tag needs either a mention in the article's lead or at least three mentions
in the whole text, because one passing "river" in a long article describes
where a column marched, not where the battle was fought. At most three tags
are kept, the ones mentioned most, so a battle fought across a river on a
plain beside a wood is not reduced to a bag of every landscape word.

Pure and database-free like the rest of :mod:`pipeline.classifiers`; the
store writes only where the column is still empty.
"""

from __future__ import annotations

import re
from typing import Final

__all__ = [
    "MAX_TERRAIN_TAGS",
    "TERRAIN_VOCABULARY",
    "infer_terrain",
]

# Characters of the cleaned article counted as its lead.
_LEAD_CHARS: Final[int] = 1500
_MIN_TOTAL_MENTIONS: Final[int] = 3
MAX_TERRAIN_TAGS: Final[int] = 3

# Order is the tie-break when two tags are mentioned equally often. "urban" is
# phrases about fighting in a built-up area, never the bare word "city": most
# sieges and many field battles name a city that was not itself the terrain.
_PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "mountainous": re.compile(
        r"\bmountain(?:s|ous)?\b|\balps\b|\balpine\b|\bhighlands?\b|\bdefiles?\b|\bgorges?\b",
        re.IGNORECASE,
    ),
    "river": re.compile(r"\brivers?\b|\bfords?\b|\bstreams?\b", re.IGNORECASE),
    "forested": re.compile(
        r"\bforests?\b|\bwoods?\b|\bwoodlands?\b|\bjungles?\b|\bwooded\b|\bbocage\b",
        re.IGNORECASE,
    ),
    "desert": re.compile(r"\bdeserts?\b|\bsahara\b|\bdunes?\b", re.IGNORECASE),
    "urban": re.compile(
        r"\burban (?:warfare|combat|fighting)\b|\bstreet[- ](?:fighting|to[- ]street)\b"
        r"|\bhouse[- ]to[- ]house\b|\bfighting in the streets\b",
        re.IGNORECASE,
    ),
    "coastal": re.compile(r"\bcoasts?\b|\bcoastal\b|\bbeach(?:es)?\b|\bshores?\b", re.IGNORECASE),
    "plain": re.compile(r"\bplains?\b|\bopen (?:field|country|ground)\b", re.IGNORECASE),
    "marsh": re.compile(r"\bmarsh(?:es|land)?\b|\bswamps?\b|\bbogs?\b|\bfens?\b", re.IGNORECASE),
    "hills": re.compile(
        r"\bhills?\b|\bhilly\b|\bhilltops?\b|\bhillsides?\b|\bridges?\b|\bhigh ground\b",
        re.IGNORECASE,
    ),
    "steppe": re.compile(r"\bsteppes?\b|\bgrasslands?\b", re.IGNORECASE),
    "snow": re.compile(
        r"\bsnow(?:s|y|storms?|fall|bound)?\b|\bice\b|\bicy\b|\bblizzards?\b", re.IGNORECASE
    ),
    "lake": re.compile(r"\blakes?\b|\blakeshore\b", re.IGNORECASE),
    "island": re.compile(r"\bislands?\b|\bislets?\b", re.IGNORECASE),
}

# Phrases that contain a terrain word without describing terrain. Each match is
# blanked (same length, so the lead window does not move) before counting. The
# "Ford" entry is case-sensitive: a capitalised Ford is a surname, a lower-case
# ford a river crossing.
_FALSE_POSITIVES: Final[tuple[re.Pattern[str], ...]] = (
    # "desert" as a verb: "began to desert", "men would desert".
    re.compile(
        r"\b(?:to|would|will|could|might|may|must|should|not|did|didn't|began|started|then"
        r"|men|troops|soldiers)\s+desert\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bhighlands?(?:ers?)?\s+(?:brigades?|regiments?|divisions?)\b", re.IGNORECASE),
    re.compile(r"\bmountain\s+(?:artillery|divisions?|troops)\b", re.IGNORECASE),
    re.compile(r"\balpine\s+(?:troops|infantry|corps|divisions?|regiments?)\b|\balpini\b", re.I),
    re.compile(r"\bFord\b"),
    re.compile(r"\bplains\s+(?:of\s+abraham|indians?)\b", re.IGNORECASE),
    re.compile(r"\bcoast\s+guards?\b|\bgold\s+coast\b", re.IGNORECASE),
    re.compile(r"\bbog(?:s|ged|ging)?\s+down\b", re.IGNORECASE),
)


def _mask_false_positives(text: str) -> str:
    """Blank out phrases that only look like terrain words.

    Args:
        text: Cleaned article prose.

    Returns:
        The text with each :data:`_FALSE_POSITIVES` match replaced by spaces of
        the same length.
    """
    for pattern in _FALSE_POSITIVES:
        text = pattern.sub(lambda m: " " * len(m.group(0)), text)
    return text


TERRAIN_VOCABULARY: Final[tuple[str, ...]] = tuple(_PATTERNS)


def infer_terrain(article_text: str | None) -> tuple[list[str], str]:
    """Pick the terrain tags an article's text supports.

    Args:
        article_text: Cleaned article prose, or None when no article was read.

    Returns:
        ``(tags, evidence)``: up to ``MAX_TERRAIN_TAGS`` tags from
        :data:`TERRAIN_VOCABULARY`, most mentioned first (vocabulary order on
        a tie), and a one-line account of the counts. Empty tags and empty
        evidence when there is no text or no tag clears the bar.
    """
    if not article_text or not article_text.strip():
        return [], ""

    article_text = _mask_false_positives(article_text)
    lead = article_text[:_LEAD_CHARS]
    scored: list[tuple[int, int, str]] = []
    for order, (tag, pattern) in enumerate(_PATTERNS.items()):
        total = len(pattern.findall(article_text))
        in_lead = len(pattern.findall(lead))
        if in_lead >= 1 or total >= _MIN_TOTAL_MENTIONS:
            scored.append((-total, order, tag))

    scored.sort()
    chosen = scored[:MAX_TERRAIN_TAGS]
    tags = [tag for _neg, _order, tag in chosen]
    evidence = ", ".join(f"{tag} x{-neg}" for neg, _order, tag in chosen)
    return tags, evidence
