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

import structlog

from pipeline.classifiers.records import BattleTypeDecision

logger = structlog.get_logger()

__all__ = [
    "CAMPAIGN_MIN_SPAN_DAYS",
    "SIEGE_CAMPAIGN_MIN_SPAN_DAYS",
    "extract_categories",
    "infer_battle_type",
    "infer_campaign",
]

_CATEGORY_RE: Final = re.compile(r"\[\[\s*Category\s*:\s*([^\]|]+)", re.IGNORECASE)
# Rendered HTML (what crawl stores) carries categories as links, not wikitext.
_HTML_CATEGORY_RE: Final = re.compile(r'href="/wiki/Category:([^"#?]+)', re.IGNORECASE)

_SIEGE_NAME_RE: Final = re.compile(r"^\s*siege of\b", re.IGNORECASE)
_SIEGE_CATEGORY_RE: Final = re.compile(r"\bsieges?\b", re.IGNORECASE)
# "Sieges involving Israel" is a Wikipedia country-grouping category that sits
# on articles about operations (Operation Badr) as well as on sieges; it names a
# participant, not the kind of event, so it is not a siege signal.
_INVOLVING_CATEGORY_RE: Final = re.compile(r"\binvolving\b", re.IGNORECASE)

# A war or a multi-month operation is not a battle, and the model excludes it.
# "War" is case-sensitive so "Warsaw" and "wartime" never match. A title that
# also says battle, siege, skirmish or raid names an engagement inside a war.
_WAR_NAME_RE: Final = re.compile(r"\b(?:Wars?|Crisis)\b")
# "Operation Bagration", "Operation Overlord": a named military operation, not
# an engagement, whatever its dates. Case-sensitive on the name so "Operational
# ..." and a lowercase "operation" in running text never match.
_OPERATION_NAME_RE: Final = re.compile(r"^\s*Operation\s+[A-Z]")
_CAMPAIGN_WORD_RE: Final = re.compile(r"\bcampaigns?\b", re.IGNORECASE)
_ENGAGEMENT_NAME_RE: Final = re.compile(r"\b(?:battles?|sieges?|skirmish|raid)\b", re.IGNORECASE)
# A title that is itself an invasion or conquest names a multi-phase operation.
_CAMPAIGN_OPERATION_NAME_RE: Final = re.compile(
    r"\b(?:invasions?|conquests?)\s+of\b", re.IGNORECASE
)
# Titles that name one engagement are not demoted on the 180-day span: Verdun
# (300 days) and Stalingrad really are single "battles" in the literature, and
# a short span rule cannot tell them from an operation.
_ENGAGEMENT_TITLE_PREFIX_RE: Final = re.compile(r"^\s*(?:battles?|sieges?)\s+of\b", re.IGNORECASE)
_SIEGE_TITLE_PREFIX_RE: Final = re.compile(r"^\s*sieges?\s+of\b", re.IGNORECASE)
# Longer than this, for a title that does not name an engagement, and the
# article covers an operation, not a single engagement.
CAMPAIGN_MIN_SPAN_DAYS: Final[int] = 180
# A siege over two years is many engagements, so the "Siege of" exemption
# above stops here. Two years, not one, keeps year-long sieges such as Rome
# (537-538, 375 days) as single sieges. The Siege of Leningrad (872 days)
# decided it: crediting a commander with a "victory" over 900 days is not a
# battle result. The user chose this on 2026-10-16, overriding the earlier
# Leningrad exemption.
SIEGE_CAMPAIGN_MIN_SPAN_DAYS: Final[int] = 730
# A date known only to the decade or coarser cannot belong to one engagement.
_COARSE_DATE_PRECISIONS: Final[frozenset[str]] = frozenset({"decade", "century", "millennium"})

# Guerrilla only from the word itself in a category, or repeated in the text.
# Insurgency vocabulary is deliberately not used: urban set-piece battles of an
# insurgency (the Battles of Fallujah) are full of it and are not guerrilla fights.
_GUERRILLA_CATEGORY_RE: Final = re.compile(r"\bguerrilla\b", re.IGNORECASE)
_GUERRILLA_TEXT_RE: Final = re.compile(r"\bguerrillas?\b", re.IGNORECASE)
_GUERRILLA_MIN_MENTIONS: Final[int] = 4
# Text rules (guerrilla, amphibious) also need density: a long article that
# mentions a word four times in 100,000 characters is not about it. One mention
# per 1000 characters is roughly a mention per screen of text.
_MIN_MENTIONS_PER_1000_CHARS: Final[float] = 1.0

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
# How much of the article counts as its lead for "said it up front" signals.
_LEAD_CHARS: Final[int] = 1500

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


def infer_campaign(
    battle_id: int,
    battle_name: str,
    span_days: int | None = None,
    date_precision: str | None = None,
) -> BattleTypeDecision | None:
    """Recognise an article that covers a war or a multi-month operation.

    Needs no article text, so it also runs over battles an earlier pass typed
    by rule (see ``write_battle_type_decisions``).

    Args:
        battle_id: The battle this decision is about.
        battle_name: The article title.
        span_days: ``date_end - date_start`` in days when both are known to the
            day, else None. Computed in SQL because a BC date cannot be held
            by ``datetime.date``. A negative span (end before start) is bad
            data and meets no span rule.
        date_precision: The battle's ``date_precision``, when known.

    Returns:
        A 'campaign' decision when the title names a war, crisis, campaign or
        "Operation X" without also naming a battle, siege or raid; when the
        dates are known only to the decade or coarser; when the dates span
        more than ``CAMPAIGN_MIN_SPAN_DAYS`` (a "Battle of"/"Siege of" title
        is exempt) or a siege spans more than ``SIEGE_CAMPAIGN_MIN_SPAN_DAYS``;
        None otherwise.
    """
    name = battle_name or ""
    engagement = _ENGAGEMENT_NAME_RE.search(name) is not None
    if date_precision in _COARSE_DATE_PRECISIONS:
        return _campaign(
            battle_id,
            name,
            f"dates are only known to the {date_precision}, not one engagement",
            0.8,
        )
    if not engagement and _OPERATION_NAME_RE.match(name):
        return _campaign(battle_id, name, "title names an operation, not a battle or siege", 0.85)
    if not engagement and (_WAR_NAME_RE.search(name) or _CAMPAIGN_WORD_RE.search(name)):
        return _campaign(
            battle_id, name, "title names a war or campaign, not a battle or siege", 0.9
        )
    if not engagement and _CAMPAIGN_OPERATION_NAME_RE.search(name):
        return _campaign(
            battle_id, name, "title names an invasion or conquest, not a battle or siege", 0.85
        )
    if (
        span_days is not None
        and span_days > SIEGE_CAMPAIGN_MIN_SPAN_DAYS
        and _SIEGE_TITLE_PREFIX_RE.match(name)
    ):
        return _campaign(
            battle_id,
            name,
            f"siege spans {span_days} days (more than {SIEGE_CAMPAIGN_MIN_SPAN_DAYS})",
            0.8,
        )
    if (
        span_days is not None
        and span_days > CAMPAIGN_MIN_SPAN_DAYS
        and not _ENGAGEMENT_TITLE_PREFIX_RE.match(name)
    ):
        return _campaign(
            battle_id,
            name,
            f"dates span {span_days} days (more than {CAMPAIGN_MIN_SPAN_DAYS})",
            0.8,
        )
    return None


def _campaign(battle_id: int, name: str, reason: str, confidence: float) -> BattleTypeDecision:
    """Build a campaign decision and log why the battle was demoted.

    Args:
        battle_id: The battle this decision is about.
        name: The article title.
        reason: The evidence line stored with the decision.
        confidence: Confidence in the demotion.

    Returns:
        A 'campaign' decision.
    """
    logger.info("classify_campaign_demoted", battle_id=battle_id, battle_name=name, reason=reason)
    return BattleTypeDecision(
        battle_id=battle_id,
        battle_type="campaign",
        fortified=None,
        evidence=reason,
        confidence=confidence,
    )


def _mention_density(pattern: re.Pattern[str], text: str) -> tuple[int, float]:
    """Count a pattern's matches and their rate per 1000 characters.

    Args:
        pattern: The word to count.
        text: The article text; may be empty.

    Returns:
        ``(mentions, mentions per 1000 characters)``; the rate is 0.0 for no text.
    """
    mentions = len(pattern.findall(text))
    return mentions, (mentions * 1000.0 / len(text)) if text else 0.0


def infer_battle_type(
    battle_id: int,
    battle_name: str,
    article_text: str | None,
    categories: Sequence[str],
    troop_branches: Sequence[str],
    strength_strings: Sequence[str],
    span_days: int | None = None,
    date_precision: str | None = None,
) -> BattleTypeDecision:
    """Infer a battle's type from whatever evidence extraction produced.

    Checked in order, each a stronger or cheaper signal than the next:

    0. **Campaign**, from a title that names a war or campaign, or dates that
       span more than 180 days (:func:`infer_campaign`; a title starting
       "Battle of"/"Siege of" is exempt from the span rule, though a siege over
       730 days is not), or dates known only to the decade or coarser. An
       "Operation X", "Invasion of" or "Conquest of" title is a campaign. It wins over every
       other type: an article about a war is not a battle whatever its
       categories say, and the model excludes it.
    1. **Siege**, from the battle's name or a "Sieges ..." category. Neither
       needs an article read, so this is checked first regardless of what
       else is available. The schema's ``battle_type`` enum has only one
       siege-from-the-besieger's-view value, ``siege_offensive`` -- there is
       no neutral "a siege happened" member and no ``siege_defensive`` this
       function can respect without knowing which side is being described,
       which is a per-side fact this article-level function does not carry.
       By convention, then, ``siege_offensive`` here means "this was a
       siege", and any per-side defender/attacker distinction is left
       unrecorded rather than guessed. ``siege_defensive`` is never produced:
       which side was besieged is ``battle_sides.is_defender``, set per side
       from the title and from the role classification, so a second article-level
       value would only repeat it. A category of the form "Sieges involving X"
       counts only when the article lead also says "siege" (Operation Badr
       carries one and its lead does not).
    2. **Amphibious**, when there is evidence of both naval and land
       forces *and* the battle name says "amphibious"/"landing" or the
       article says "amphibious" at least three times. A bare "landing" in
       article text is not enough (Harrison's Landing, Malvern Hill), and a
       category such as "Amphibious operations" is not either (Marathon). With
       no troop-branch evidence the article alone suffices only when it says
       "amphibious" at least three times, the first within its lead.
    3. **Naval**, from a naval category; or from ship vocabulary that
       dominates the strength strings (at least as many distinct strings mention
       ships as do not); or from a fleet-only report set (only 'naval'/'total'
       branches) with any ship vocabulary. A passing fleet among many troop
       counts (Marathon, Malvern Hill, the Yom Kippur War) is not enough.
    4. **Aerial**, from an air-battle category, unless ground-force branches
       are reported too. Checked before the ship-vocabulary match (but after
       a naval category), since air battles use "squadrons" and "carriers" too.
    5. **Guerrilla**, from a category naming guerrilla warfare, or the word
       "guerrilla" at least four times in the article, once in its lead and
       at least one mention per 1000 characters. After the naval and
       aerial checks, so a fleet action is never guerrilla.
    6. **Field**, but *only* when ``article_text`` is non-empty, i.e. an
       article was actually read and carried none of the above signals.
       Defaulting to 'field' when no article was read would recreate
       §4.2's bug with the sign flipped: instead of every naval battle
       mislabelled 'field', every *unread* battle would be, silently, and
       an unread battle is exactly the one this function has the least
       business guessing about.
    7. **Unknown**, otherwise -- no article, no category, no troop or
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
        span_days: ``date_end - date_start`` in days when known, for the
            campaign rule.
        date_precision: The battle's ``date_precision``, for the campaign rule.

    Returns:
        A decision carrying the matched evidence and a confidence that is
        highest for an explicit category, lower for an inferred branch or
        vocabulary match, and 0.0 for 'unknown'.
    """
    campaign = infer_campaign(battle_id, battle_name, span_days, date_precision)
    if campaign is not None:
        return campaign

    if _SIEGE_NAME_RE.match(battle_name or ""):
        return BattleTypeDecision(
            battle_id=battle_id,
            battle_type="siege_offensive",
            fortified=True,
            evidence="battle name starts with 'Siege of'",
            confidence=0.95,
        )

    # "Sieges involving X" is a participant grouping, so it counts only when the
    # article's own lead says "siege" (Operation Badr's lead does not).
    lead_says_siege = _SIEGE_CATEGORY_RE.search((article_text or "")[:_LEAD_CHARS]) is not None
    siege_category = next(
        (
            c
            for c in categories
            if _SIEGE_CATEGORY_RE.search(c)
            and (lead_says_siege or not _INVOLVING_CATEGORY_RE.search(c))
        ),
        None,
    )
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
    amphibious_count, amphibious_density = _mention_density(_AMPHIBIOUS_RE, article_text or "")
    amphibious_dense = (
        amphibious_count >= _AMPHIBIOUS_MIN_MENTIONS
        and amphibious_density >= _MIN_MENTIONS_PER_1000_CHARS
    )
    amphibious_mentioned = bool(_AMPHIBIOUS_NAME_RE.search(battle_name or "")) or amphibious_dense

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

    lead = (article_text or "")[:_LEAD_CHARS]
    if amphibious_dense and _AMPHIBIOUS_RE.search(lead) is not None:
        # No troop-branch evidence needed: the article says so repeatedly and
        # up front (Inchon, Gallipoli). Weaker than the branch-backed rule.
        return BattleTypeDecision(
            battle_id=battle_id,
            battle_type="amphibious",
            fortified=None,
            evidence="article says 'amphibious' repeatedly, first in its lead",
            confidence=0.6,
        )

    # A naval category is dropped when land reports outnumber naval ones and no
    # ship vocabulary dominates: Fort Donelson sits in a naval category for its
    # gunboats but is a land battle by its strength reports. Actium keeps its
    # category because its naval reports match its land ones (ship_match is set
    # above).
    land_outweighs_naval = n_land > n_naval and ship_match is None
    if naval_category is not None and not land_outweighs_naval:
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

    guerrilla_category = next((c for c in categories if _GUERRILLA_CATEGORY_RE.search(c)), None)
    guerrilla_mentions, guerrilla_density = _mention_density(_GUERRILLA_TEXT_RE, article_text or "")
    guerrilla_in_text = (
        guerrilla_mentions >= _GUERRILLA_MIN_MENTIONS
        and guerrilla_density >= _MIN_MENTIONS_PER_1000_CHARS
        and _GUERRILLA_TEXT_RE.search((article_text or "")[:_LEAD_CHARS]) is not None
    )
    if guerrilla_category is not None or guerrilla_in_text:
        return BattleTypeDecision(
            battle_id=battle_id,
            battle_type="guerrilla",
            fortified=None,
            evidence=(
                f"category {guerrilla_category!r} names guerrilla warfare"
                if guerrilla_category is not None
                else f"article says 'guerrilla' {guerrilla_mentions} times"
            ),
            confidence=0.8 if guerrilla_category is not None else 0.6,
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
