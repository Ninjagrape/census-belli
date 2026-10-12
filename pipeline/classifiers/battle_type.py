"""
Battle-type inference, the covariate docs/handover-log.md §4.2 left null.

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
from collections import Counter
from collections.abc import Sequence
from typing import Final
from urllib.parse import unquote

from pipeline.classifiers.records import BattleTypeDecision

__all__ = [
    "extract_categories",
    "infer_battle_type",
]

_CATEGORY_RE: Final = re.compile(r"\[\[\s*Category\s*:\s*([^\]|]+)", re.IGNORECASE)
# Rendered HTML (what crawl stores) carries categories as links, not wikitext.
_HTML_CATEGORY_RE: Final = re.compile(r'href="/wiki/Category:([^"#?]+)', re.IGNORECASE)

_SIEGE_NAME_RE: Final = re.compile(r"^\s*siege of\b", re.IGNORECASE)
_SIEGE_CATEGORY_RE: Final = re.compile(r"\bsieges?\b", re.IGNORECASE)

_NAVAL_CATEGORY_RE: Final = re.compile(
    r"\bnaval (?:battles?|engagements?|operations?|actions?)\b", re.IGNORECASE
)
# Ship vocabulary in a strength string. "ship-of-the-line" and "man-of-war"
# still match on the "ship"/"man" token: \b sits at the letter/hyphen
# boundary either way. Bare "squadron(s)" is deliberately absent: it is a
# cavalry and air-force unit as often as a naval one ("8 squadrons of
# cavalry", RAF squadrons in the Battle of Britain), so only "naval squadron"
# counts.
_SHIP_VOCAB_RE: Final = re.compile(
    r"\b(?:ships?|fleets?|galleys?|triremes?|frigates?|men-of-war|man-of-war"
    r"|flotillas?|naval squadrons?|vessels?|warships?|men-o'-war)\b",
    re.IGNORECASE,
)

_AERIAL_CATEGORY_RE: Final = re.compile(
    r"\b(?:aerial|air) (?:battles?|operations?|engagements?)\b", re.IGNORECASE
)

# Bare "landing" is deliberately absent from the article-text rule: it fires
# on "Harrison's Landing" (Malvern Hill) and on any land battle whose army
# crossed a river by "landing" boats. "Amphibious" repeated through an
# article is the signal; one passing use is not. The battle name and
# categories may use either word.
_AMPHIBIOUS_RE: Final = re.compile(r"\bamphibious\b", re.IGNORECASE)
_AMPHIBIOUS_NAME_RE: Final = re.compile(r"\bamphibious\b|\blandings?\b", re.IGNORECASE)
_AMPHIBIOUS_MIN_MENTIONS: Final[int] = 3

_LAND_BRANCHES: Final[frozenset[str]] = frozenset({"infantry", "cavalry", "artillery", "armour"})


def _ship_vocab_dominates(strength_strings: Sequence[str]) -> re.Match[str] | None:
    """Return a ship-vocabulary match only when ship strings are at least half.

    Marathon, Malvern Hill and the Yom Kippur War each carry one or two ship
    or "naval vessel" strings among many ordinary troop counts. A passing
    fleet is not a naval battle; a strength section that is mostly ships is.

    Args:
        strength_strings: Raw strength-field or troop-report text.

    Returns:
        The first ship-vocabulary match, or None when there is none or the
        strings that do not mention ships outnumber those that do.
    """
    distinct = {s.strip() for s in strength_strings if s and s.strip()}
    ship_strings = [s for s in distinct if _SHIP_VOCAB_RE.search(s)]
    if not ship_strings or len(ship_strings) < len(distinct) - len(ship_strings):
        return None
    return _SHIP_VOCAB_RE.search(ship_strings[0])


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
    names = [m.group(1).strip() for m in _CATEGORY_RE.finditer(wikitext)]
    names += [
        unquote(m.group(1)).replace("_", " ").strip() for m in _HTML_CATEGORY_RE.finditer(wikitext)
    ]
    return list(dict.fromkeys(n for n in names if n))


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
    2. **Amphibious**, when there is evidence of both naval and land
       forces *and* the battle name says "amphibious"/"landing" or the
       article says "amphibious" at least three times. A bare "landing" in
       article text is not enough (Harrison's Landing, Malvern Hill), and a
       category such as "Amphibious operations" is not either (Marathon).
    3. **Naval**, from a naval category; or from ship vocabulary that
       dominates the strength strings (at least as many distinct strings mention
       ships as do not); or from a fleet-only report set (only 'naval'/'total'
       branches) with any ship vocabulary. A passing fleet among many troop
       counts (Marathon, Malvern Hill, the Yom Kippur War) is not enough.
    4. **Aerial**, from an air-battle category, unless ground-force branches
       are reported too. Checked before the ship-vocabulary match (but after
       a naval category), since air battles use "squadrons" and "carriers" too.
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
    ship_match = _ship_vocab_dominates(strength_strings)
    branch_counts = Counter(b.strip().lower() for b in troop_branches)
    n_naval = branch_counts["naval"]
    n_land = sum(branch_counts[b] for b in _LAND_BRANCHES)
    if ship_match is None and n_naval >= max(2, n_land):
        # Ship strings are outnumbered by unsplit "total" counts, but at least
        # as many reports are branch-labelled naval as land (Actium's ships1,
        # ships2 against two infantry strengths).
        ship_match = _SHIP_VOCAB_RE.search(" ".join(strength_strings))
    any_ship_vocab = _SHIP_VOCAB_RE.search(" ".join(strength_strings)) is not None
    aerial_category = next((c for c in categories if _AERIAL_CATEGORY_RE.search(c)), None)

    has_land_branch = bool(branches_lower & _LAND_BRANCHES)
    # Reports that are only ever 'naval' or an unsplit 'total' describe a
    # fleet-only engagement (Okpo, Artemisium); a land branch alongside means
    # the naval report may be one detachment of a land battle (Malvern Hill).
    naval_only_forces = has_naval_branch and not (branches_lower - {"naval", "total"})
    amphibious_mentioned = (
        bool(_AMPHIBIOUS_NAME_RE.search(battle_name or ""))
        or len(_AMPHIBIOUS_RE.findall(article_text or "")) >= _AMPHIBIOUS_MIN_MENTIONS
    )

    # A naval branch alone is weak evidence of a naval battle (a land battle
    # can carry one marine or naval-brigade report), but together with land
    # troops and an explicit "amphibious"/"landing" mention it is enough.
    naval_for_amphibious = naval_category is not None or has_naval_branch or any_ship_vocab
    if naval_for_amphibious and has_land_branch and amphibious_mentioned:
        return BattleTypeDecision(
            battle_id=battle_id,
            battle_type="amphibious",
            fortified=None,
            evidence="naval and land troop branches present, with an amphibious/landing mention",
            confidence=0.75,
        )

    if naval_category is not None:
        return BattleTypeDecision(
            battle_id=battle_id,
            battle_type="naval",
            fortified=None,
            evidence=f"category {naval_category!r} names a naval battle",
            confidence=0.9,
        )

    # Before ship vocabulary: an air battle's strength text routinely says
    # "squadrons" or "aircraft carriers", which is not evidence of a naval one.
    # Campaign articles (Stalingrad, Bagration) carry an "Aerial operations"
    # category too; with ground forces reported alongside, it is not an air battle.
    if aerial_category is not None and not has_land_branch:
        return BattleTypeDecision(
            battle_id=battle_id,
            battle_type="aerial",
            fortified=None,
            evidence=f"category {aerial_category!r} names an air battle",
            confidence=0.85,
        )

    if naval_only_forces and any_ship_vocab:
        return BattleTypeDecision(
            battle_id=battle_id,
            battle_type="naval",
            fortified=None,
            evidence="only naval/unsplit troop reports, with ship vocabulary in the strength text",
            confidence=0.75,
        )

    if ship_match is not None:
        return BattleTypeDecision(
            battle_id=battle_id,
            battle_type="naval",
            fortified=None,
            evidence=f"ship vocabulary in strength text: {ship_match.group(0)!r}",
            confidence=0.75 if has_naval_branch else 0.6,
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
