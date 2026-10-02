"""
Battle-type inference, the covariate handover.md §4.2 left null.

``pipeline/extractors/infobox.py`` used to set ``battle_type`` from which
infobox template a battle used. That broke because Wikipedia uses the
generic "Infobox military conflict" template for nearly every battle,
including naval ones -- Actium, Trafalgar and Midway all use it, and none
carries a ``ships=``/``vessels=`` field to fall back on. The fix left
``battle_type`` unset whenever the template does not determine it, which is
almost always, and pushed inference here, where there is more to go on than
a template name: categories, the troop branches troop_reports actually
recorded (a ``ships1`` field maps to branch 'naval' regardless of which
template held it), and vocabulary in the strength text itself.

:func:`infer_battle_type` never defaults to 'field' for the absence of a
signal. See its docstring for why: doing so would repeat §4.2 in reverse,
just with the wrong label moved from "every battle" to "every battle nobody
bothered to check".
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Final

from pipeline.classifiers.records import BattleTypeDecision

__all__ = [
    "extract_categories",
    "infer_battle_type",
]

_CATEGORY_RE: Final = re.compile(r"\[\[\s*Category\s*:\s*([^\]|]+)", re.IGNORECASE)

_SIEGE_NAME_RE: Final = re.compile(r"^\s*siege of\b", re.IGNORECASE)
_SIEGE_CATEGORY_RE: Final = re.compile(r"\bsieges?\b", re.IGNORECASE)

_NAVAL_CATEGORY_RE: Final = re.compile(
    r"\bnaval (?:battles?|engagements?|operations?|actions?)\b", re.IGNORECASE
)
# Ship vocabulary in a strength string. "ship-of-the-line" and "man-of-war"
# still match on the "ship"/"man" token: \b sits at the letter/hyphen
# boundary either way.
_SHIP_VOCAB_RE: Final = re.compile(
    r"\b(?:ships?|fleets?|galleys?|triremes?|frigates?|men-of-war|man-of-war"
    r"|flotillas?|squadrons?|vessels?|warships?|men-o'-war)\b",
    re.IGNORECASE,
)

_AERIAL_CATEGORY_RE: Final = re.compile(
    r"\b(?:aerial|air) (?:battles?|operations?|engagements?)\b", re.IGNORECASE
)

_AMPHIBIOUS_RE: Final = re.compile(r"\bamphibious\b|\blandings?\b", re.IGNORECASE)

_LAND_BRANCHES: Final[frozenset[str]] = frozenset({"infantry", "cavalry", "artillery", "armour"})


def extract_categories(wikitext: str) -> list[str]:
    """List a wikitext article's ``[[Category:...]]`` names.

    Args:
        wikitext: The full MediaWiki source of an article.

    Returns:
        Category names in document order, trimmed, without the leading
        "Category:" or a trailing sort key (``[[Category:Foo|Bar]]`` yields
        "Foo"). Duplicates are kept, since callers only ever search this
        list rather than count it.
    """
    return [m.group(1).strip() for m in _CATEGORY_RE.finditer(wikitext) if m.group(1).strip()]


def infer_battle_type(
    battle_id: int,
    battle_name: str,
    article_text: str | None,
    categories: Sequence[str],
    troop_branches: Sequence[str],
    strength_strings: Sequence[str],
) -> BattleTypeDecision:
    """Infer a battle's type from whatever evidence extraction produced.

    Checked in order, each a stronger or cheaper signal than the next:

    1. **Siege**, from the battle's name or a "Sieges ..." category. Neither
       needs an article read, so this is checked first regardless of what
       else is available. The schema's ``battle_type`` enum has only one
       siege-from-the-besieger's-view value, ``siege_offensive`` -- there is
       no neutral "a siege happened" member and no ``siege_defensive`` this
       function can respect without knowing which side is being described,
       which is a per-side fact this article-level function does not carry.
       By convention, then, ``siege_offensive`` here means "this was a
       siege", and any per-side defender/attacker distinction is left
       unrecorded rather than guessed.
    2. **Amphibious**, when there is strong evidence of both naval and land
       forces *and* the text or a category actually says "amphibious" or
       "landing" -- two branches of evidence alone is also what a combined
       land-and-naval campaign looks like, and this function does not try
       to tell those apart without the word that names one of them.
    3. **Naval**, from a naval category, a 'naval' troop branch (however
       the number reached troop_reports -- a ``ships1`` infobox field, an
       LLM extraction, doesn't matter here), or ship vocabulary in a
       strength string.
    4. **Aerial**, from an air-battle category.
    5. **Field**, but *only* when ``article_text`` is non-empty, i.e. an
       article was actually read and carried none of the above signals.
       Defaulting to 'field' when no article was read would recreate
       §4.2's bug with the sign flipped: instead of every naval battle
       mislabelled 'field', every *unread* battle would be, silently, and
       an unread battle is exactly the one this function has the least
       business guessing about.
    6. **Unknown**, otherwise -- no article, no category, no troop or
       strength signal. Left for a human or a later pass, not guessed.

    Args:
        battle_id: The battle this decision is about, threaded straight
            into the result.
        battle_name: The battle's name, checked for a "Siege of" prefix.
        article_text: Cleaned article prose (see
            ``pipeline.extractors.article.clean_article_text``), or None
            when no article was read for this battle.
        categories: The article's ``[[Category:...]]`` names, e.g. from
            :func:`extract_categories`.
        troop_branches: Every ``troop_branch`` value recorded for this
            battle's troop reports (from any source, any side).
        strength_strings: Raw strength-field or troop-report text, searched
            for ship vocabulary the branch alone might not have captured.

    Returns:
        A decision carrying the matched evidence and a confidence that is
        highest for an explicit category, lower for an inferred branch or
        vocabulary match, and 0.0 for 'unknown'.
    """
    if _SIEGE_NAME_RE.match(battle_name or ""):
        return BattleTypeDecision(
            battle_id=battle_id,
            battle_type="siege_offensive",
            fortified=True,
            evidence="battle name starts with 'Siege of'",
            confidence=0.95,
        )

    siege_category = next((c for c in categories if _SIEGE_CATEGORY_RE.search(c)), None)
    if siege_category is not None:
        return BattleTypeDecision(
            battle_id=battle_id,
            battle_type="siege_offensive",
            fortified=True,
            evidence=f"category {siege_category!r} names a siege",
            confidence=0.85,
        )

    naval_category = next((c for c in categories if _NAVAL_CATEGORY_RE.search(c)), None)
    branches_lower = {b.strip().lower() for b in troop_branches}
    has_naval_branch = "naval" in branches_lower
    ship_match = _SHIP_VOCAB_RE.search(" ".join(strength_strings))
    naval_signal = naval_category is not None or has_naval_branch or ship_match is not None

    has_land_branch = bool(branches_lower & _LAND_BRANCHES)
    amphibious_mentioned = any(_AMPHIBIOUS_RE.search(c) for c in categories) or bool(
        article_text and _AMPHIBIOUS_RE.search(article_text)
    )

    if naval_signal and has_land_branch and amphibious_mentioned:
        return BattleTypeDecision(
            battle_id=battle_id,
            battle_type="amphibious",
            fortified=None,
            evidence="naval and land troop branches present, with an amphibious/landing mention",
            confidence=0.75,
        )

    if naval_signal:
        if naval_category is not None:
            evidence, confidence = f"category {naval_category!r} names a naval battle", 0.9
        elif has_naval_branch:
            evidence, confidence = "a troop report carries the 'naval' branch", 0.75
        else:
            assert ship_match is not None  # naval_signal guarantees one of the three
            evidence = f"ship vocabulary in strength text: {ship_match.group(0)!r}"
            confidence = 0.6
        return BattleTypeDecision(
            battle_id=battle_id,
            battle_type="naval",
            fortified=None,
            evidence=evidence,
            confidence=confidence,
        )

    aerial_category = next((c for c in categories if _AERIAL_CATEGORY_RE.search(c)), None)
    if aerial_category is not None:
        return BattleTypeDecision(
            battle_id=battle_id,
            battle_type="aerial",
            fortified=None,
            evidence=f"category {aerial_category!r} names an air battle",
            confidence=0.85,
        )

    if article_text and article_text.strip():
        return BattleTypeDecision(
            battle_id=battle_id,
            battle_type="field",
            fortified=None,
            evidence="article text was read and carried no naval, siege, or aerial signal",
            confidence=0.55,
        )

    return BattleTypeDecision(
        battle_id=battle_id,
        battle_type="unknown",
        fortified=None,
        evidence="no article text, and no category, troop-branch, or strength-text signal",
        confidence=0.0,
    )
