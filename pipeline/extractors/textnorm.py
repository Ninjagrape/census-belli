"""
Text normalisation and quantity parsing shared by the extractors.

Two jobs live here. The first is turning MediaWiki markup into readable
text without losing the structure that carries meaning -- a ``<br />``
between two commanders is the only thing separating them, so stripping tags
naively welds two names into one.

The second is reading numbers out of prose, and specifically reading what a
number *means*. ``troop_reports.scope`` drives the force-ratio covariate in
the final model, so "30,000 men were engaged" and "Napoleon had 30,000 men
in the theatre" must not produce the same row. :func:`infer_scope` looks for
the wording that distinguishes them and returns ``unknown`` when it finds
none, because guessing ``engaged`` would silently inflate the number of
battles the model believes it can compute a force ratio for.
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass, replace
from typing import Final

import mwparserfromhell
import structlog

from pipeline.extractors.records import coerce_scope

__all__ = [
    "MODERN_ESTIMATE_PATTERN",
    "Quantity",
    "classify_figure",
    "is_equipment_count",
    "infer_branch",
    "infer_casualty_type",
    "infer_scope",
    "is_modern_estimate",
    "normalise_label",
    "parse_number",
    "parse_quantities",
    "resolve_scope",
    "split_list_items",
    "strip_markup",
    "tidy_spacing",
]

logger = structlog.get_logger()


# ─── Markup ──────────────────────────────────────────────────────────────────

_REF_RE: Final = re.compile(r"<ref[^>]*?/>|<ref[^>]*>.*?</ref>", re.IGNORECASE | re.DOTALL)
_COMMENT_RE: Final = re.compile(r"<!--.*?-->", re.DOTALL)
_BREAK_RE: Final = re.compile(r"<\s*br\s*/?\s*>|</\s*li\s*>|<\s*hr\s*/?\s*>", re.IGNORECASE)
_WHITESPACE_RE: Final = re.compile(r"[ \t ]+")
_BLANKLINES_RE: Final = re.compile(r"\n{3,}")

# Templates whose parameters are list items rather than formatting arguments.
# mwparserfromhell's strip_code drops templates whole, so these are expanded
# to newline-separated text first or several commanders collapse into one.
_LIST_TEMPLATES: Final[frozenset[str]] = frozenset(
    {
        "ubl",
        "ubt",
        "unbulleted list",
        "plainlist",
        "plain list",
        "hlist",
        "flatlist",
        "collapsible list",
        "bulleted list",
        "bulletedlist",
        "blist",
        "bdl",
        "udl",
        "unbulletedlist",
        "ublist",
        "indented plainlist",
        "ordered list",
        "olist",
        "ubli",
    }
)

# Keyed parameters of the list templates that format the list instead of
# holding an item. Any other keyed parameter is an item whose text contained an
# "=" (or, for ``bdl|wrap=``, a definition list), so it is kept.
_LIST_OPTION_RE: Final = re.compile(
    r"^(?:class|style|title|bullets|padding|margin|list_?style|item\d*_?style|indent|hlist.*|ul_?style|li_?style"
    r"|start|type|collapsed|expanded|framestyle|titlestyle|liststyle)$"
)
# Definition-list prefixes ("; Livy" / ": 48,200 killed") that survive as text.
_DEFINITION_PREFIX_RE: Final = re.compile(r"^[ 	]*[;:]+[ 	]*", re.MULTILINE)

# Templates that wrap content for presentation only; their first positional
# parameter is the content and should survive.
_PASSTHROUGH_TEMPLATES: Final[frozenset[str]] = frozenset(
    {"nowrap", "nobold", "noitalic", "small", "big", "nobr", "nsmdns", "center", "centre"}
)

# Templates that render a flag followed by the country's name. The flag is an
# image and is dropped, but the name is text the rendered page shows, so the
# first positional parameter survives. The icon-only siblings (flagicon,
# flagdeco, flagicon image, symbol) are deliberately absent: they render no
# text and fall through to strip_code, which drops them.
_FLAG_NAME_TEMPLATES: Final[frozenset[str]] = frozenset(
    {"flag", "flagcountry", "flagu", "flag country"}
)

# Templates whose rendered output is a fixed glyph.
_LITERAL_TEMPLATES: Final[dict[str, str]] = {
    "ndash": "–",
    "mdash": "—",
    "nbsp": " ",
    "snd": " – ",
    "*": "•",
    "•": "•",
    "bull": "•",
    "!": "|",
}

# Interlanguage-link wrappers: the displayed text is a named positional.
_TEXT_PARAM_TEMPLATES: Final[dict[str, int]] = {"ill": 0, "lang": 1}

# Namespaces whose links are images, never text.
_MEDIA_LINK_RE: Final = re.compile(r"^\s*:?\s*(?:file|image|media)\s*:", re.IGNORECASE)

# Fate and status markers on commander names. Kept as a note rather than
# discarded: whether a commander was killed mid-battle is a real signal, and
# it is cheaper to carry it now than to re-read the article later.
_MARKER_TEMPLATES: Final[dict[str, str]] = {
    "kia": "KIA",
    "killed in action": "KIA",
    "wia": "WIA",
    "wounded in action": "WIA",
    "pow": "POW",
    "prisoner of war": "POW",
    "mia": "MIA",
    "executed": "executed",
    "surrendered": "surrendered",
    "surrender": "surrendered",
    "dow": "DOW",
    "died of wounds": "DOW",
}

_CIRCA_TEMPLATES: Final[frozenset[str]] = frozenset(
    {"circa", "c.", "ca", "ca.", "approx", "approximately"}
)


def _expand_templates(raw: str) -> str:
    """Rewrite the templates that carry content into plain text.

    Args:
        raw: A wikitext fragment, typically one infobox field value.

    Returns:
        The fragment with list, passthrough, marker and circa templates
        replaced by text, leaving other templates for ``strip_code``.
    """
    code = mwparserfromhell.parse(raw)

    for template in reversed(code.filter_templates(recursive=True)):
        name = str(template.name).strip().lower()

        if name in _LIST_TEMPLATES:
            items = [
                _DEFINITION_PREFIX_RE.sub("", str(p.value).strip())
                for p in template.params
                if not p.showkey or not _LIST_OPTION_RE.match(str(p.name).strip().lower())
            ]
            code.replace(template, "\n" + "\n".join(i for i in items if i) + "\n")
        elif name in _PASSTHROUGH_TEMPLATES and template.params:
            code.replace(template, str(template.params[0].value))
        elif name in _MARKER_TEMPLATES:
            code.replace(template, f" ({_MARKER_TEMPLATES[name]})")
        elif name in _CIRCA_TEMPLATES:
            rest = str(template.params[0].value) if template.params else ""
            code.replace(template, f"c. {rest}")
        elif name in _FLAG_NAME_TEMPLATES and template.params:
            code.replace(template, str(template.params[0].value))
        elif name in _TEXT_PARAM_TEMPLATES:
            positional = [p for p in template.params if not p.showkey]
            index = _TEXT_PARAM_TEMPLATES[name]
            if index < len(positional):
                code.replace(template, str(positional[index].value))
        elif name in _LITERAL_TEMPLATES:
            code.replace(template, _LITERAL_TEMPLATES[name])

    return _drop_media_links(str(code))


def _drop_media_links(text: str) -> str:
    """Remove ``[[File:...]]`` and ``[[Image:...]]`` links entirely.

    ``strip_code`` keeps a wikilink's display text, and for a file link the
    last pipe-separated option is that text, so ``[[File:Flag.svg|border|25px]]``
    leaked ``border|25px`` into commander names and ``25px`` into troop counts.

    Args:
        text: Wikitext after template expansion.

    Returns:
        The text without any media link.
    """
    code = mwparserfromhell.parse(text)
    for link in reversed(code.filter_wikilinks(recursive=True)):
        if not _MEDIA_LINK_RE.match(str(link.title)):
            continue
        try:
            code.remove(link)
        except ValueError:
            # Already removed along with an enclosing media link's caption.
            continue
    return str(code)


def strip_markup(raw: str) -> str:
    """Reduce a wikitext fragment to plain text, preserving line structure.

    Args:
        raw: A wikitext fragment.

    Returns:
        Plain text. Line breaks that separated list items in the source are
        preserved as newlines; references, comments and remaining templates
        are removed.
    """
    if not raw:
        return ""

    text = _COMMENT_RE.sub("", raw)
    text = _REF_RE.sub("", text)
    text = _BREAK_RE.sub("\n", text)
    text = _expand_templates(text)

    try:
        text = mwparserfromhell.parse(text).strip_code(normalize=True, collapse=True)
    except Exception as exc:
        # A parser failure on one field must not take down the article.
        logger.warning("wikitext_strip_failed", error=str(exc))
        text = re.sub(r"\[\[([^\]|]*\|)?([^\]]*)\]\]", r"\2", text)
        text = re.sub(r"\{\{[^{}]*\}\}", "", text)

    text = html.unescape(text)
    text = _WHITESPACE_RE.sub(" ", text)
    text = _BLANKLINES_RE.sub("\n\n", text)
    return tidy_spacing("\n".join(line.strip() for line in text.splitlines()).strip())


_SPACE_BEFORE_PUNCT_RE: Final = re.compile(r"[ \t]+(?=[,.;:)\]])|(?<=[(\[])[ \t]+")
_SPACE_BEFORE_POSSESSIVE_RE: Final = re.compile(r"[ \t]+(?=['’]s\b)")


def tidy_spacing(text: str) -> str:
    """Remove the stray spaces left by joining inline HTML elements with spaces.

    Reading a rendered cell with ``get_text(" ")`` is needed so adjacent
    elements do not weld together, but it also puts a space between a link
    and the punctuation that follows it ("Octavian 's forces", "Rome , Italy").

    Args:
        text: Text that may hold spaces before punctuation or inside brackets.

    Returns:
        The text with those spaces removed. A space before a colon is only
        removed when it follows a letter, so "Total : 5" becomes "Total: 5".
    """
    text = _SPACE_BEFORE_POSSESSIVE_RE.sub("", text)
    return _SPACE_BEFORE_PUNCT_RE.sub("", text)


_ITEM_SPLIT_RE: Final = re.compile(r"[\n;•]+|^\s*[*#]+\s*", re.MULTILINE)


def split_list_items(value: str) -> list[str]:
    """Split a cleaned field value into the items it lists.

    Infobox commander and combatant fields hold several entries separated by
    line breaks, bullets or semicolons. Commas are deliberately not treated
    as separators, because they appear inside names and ranks far more often
    than between them.

    Args:
        value: Cleaned text, as returned by :func:`strip_markup`.

    Returns:
        Non-empty items in source order.
    """
    parts = _ITEM_SPLIT_RE.split(value)
    items: list[str] = []
    for part in parts:
        cleaned = part.strip(" \t*#-–—,")
        if cleaned:
            items.append(cleaned)
    return items


def normalise_label(value: str) -> str:
    """Fold a side or name label for comparison.

    Args:
        value: A raw label, e.g. ``"the French Empire"``.

    Returns:
        A lowercase, accent-free, punctuation-free key. Used only for
        matching; the original label is what gets stored.
    """
    decomposed = unicodedata.normalize("NFKD", value)
    ascii_only = "".join(c for c in decomposed if not unicodedata.combining(c))
    stripped = re.sub(r"[^a-z0-9 ]+", " ", ascii_only.lower())
    collapsed = re.sub(r"\s+", " ", stripped).strip()
    return re.sub(r"^the ", "", collapsed)


# ─── Numbers ─────────────────────────────────────────────────────────────────

_MULTIPLIERS: Final[dict[str, float]] = {
    "thousand": 1_000.0,
    "million": 1_000_000.0,
    "billion": 1_000_000_000.0,
}

_NUM_PATTERN: Final = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_MULT_PATTERN: Final = r"(?:\s*(?:thousand|million|billion))?"

_NUMBER_RE: Final = re.compile(
    rf"(?P<num>{_NUM_PATTERN})(?P<mult>{_MULT_PATTERN})",
    re.IGNORECASE,
)
_BETWEEN_RE: Final = re.compile(
    rf"between\s+(?P<lo>{_NUM_PATTERN}{_MULT_PATTERN})\s+and\s+(?P<hi>{_NUM_PATTERN}{_MULT_PATTERN})",
    re.IGNORECASE,
)
_DASH_RANGE_RE: Final = re.compile(
    rf"(?P<lo>{_NUM_PATTERN}{_MULT_PATTERN})\s*(?:[-–—]|\bto\b)\s*(?P<hi>{_NUM_PATTERN}{_MULT_PATTERN})",
    re.IGNORECASE,
)
_ORDINAL_RE: Final = re.compile(r"\s*(?:st|nd|rd|th)\b", re.IGNORECASE)
# "12 legions" counts formations, not men. Recording it as a troop report
# would put a force of twelve into the force-ratio covariate.
_FORMATION_RE: Final = re.compile(
    r"\s*(?:[A-Za-z]+\s+)?(?:legions?|corps|divisions?|regiments?|battalions?|brigades?|squadrons?"
    r"|companies|cohorts?|maniples?|armies|columns?|wings?)\b",
    re.IGNORECASE,
)
_YEAR_PREFIX_RE: Final = re.compile(
    r"(?:\bin\b|\bof\b|january|february|march|april|may|june|july|august|september"
    r"|october|november|december)\s*$",
    re.IGNORECASE,
)
_ERA_SUFFIX_RE: Final = re.compile(r"\s*(?:BCE?|AD|CE)\b")
# "16-17 October:" is a date heading inside a strength field, not a force.
_MONTH_AFTER_RE: Final = re.compile(
    r"\s*(?:January|February|March|April|May|June|July|August|September|October|November"
    r"|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\b\.?",
    re.IGNORECASE,
)

_ESTIMATE_RE: Final = re.compile(
    r"\b(?:about|approximately|approx\.?|roughly|around|some|estimated|an estimate"
    r"|circa|perhaps|possibly|nearly|almost|upwards of)\b|c\.\s|ca\.\s|~",
    re.IGNORECASE,
)
_UPPER_RE: Final = re.compile(
    r"\b(?:up to|at most|no more than|fewer than|less than|under)\b", re.IGNORECASE
)
_LOWER_RE: Final = re.compile(
    r"\b(?:at least|more than|over|in excess of|upwards of|at minimum)\b", re.IGNORECASE
)

_SEGMENT_SPLIT_RE: Final = re.compile(r"[\n;]+")


@dataclass(frozen=True)
class Quantity:
    """A number read out of prose, with what the prose said about it.

    ``lead`` is the text between the previous number (or the line start) and
    this one, and ``noun`` the words directly after it. They are what tells
    "Total: 86,400" from "25,000 Romans" and "II Corps: 20,000" when the
    branch keyword alone says only "total". ``trailing`` is the text after the
    number up to the next number, where a parenthetical source label
    ("(modern estimates)") sits. ``phase`` is the date or phase heading the
    figure falls under ("16-17 October"), ``block_first`` marks the figures on
    the first line under such a heading, and the last two flags record whether
    the figure's own wording says it was engaged or is a stated total.
    """

    value: float
    branch: str = "total"
    scope: str = "unknown"
    is_estimate: bool = False
    is_lower_bound: bool = False
    is_upper_bound: bool = False
    context: str = ""
    lead: str = ""
    noun: str = ""
    trailing: str = ""
    phase: str = ""
    block_first: bool = False
    explicit_engaged: bool = False
    total_worded: bool = False


def parse_number(token: str) -> float | None:
    """Read a single numeric token, including grouped and scaled forms.

    Args:
        token: Text such as ``"30,000"``, ``"1.5 million"`` or ``"12000"``.

    Returns:
        The value, or None if the token holds no number.
    """
    match = _NUMBER_RE.search(token)
    if match is None:
        return None
    try:
        value = float(match.group("num").replace(",", ""))
    except ValueError:  # pragma: no cover - the pattern guarantees a number
        return None
    multiplier = (match.group("mult") or "").strip().lower()
    return value * _MULTIPLIERS.get(multiplier, 1.0)


def _looks_like_year(text: str, match: re.Match[str]) -> bool:
    """Decide whether a bare four-digit number is a date rather than a count.

    Args:
        text: The segment the number was found in.
        match: The number match within that segment.

    Returns:
        True if the number is preceded by a month name or "in", or followed
        by an era marker, all of which make a year far likelier than a
        troop count.
    """
    raw = match.group("num")
    if len(raw) != 4 or not raw.isdigit():
        return False
    if not 1000 <= int(raw) <= 2100:
        return False
    before = text[: match.start()]
    after = text[match.end() :]
    return bool(_YEAR_PREFIX_RE.search(before)) or bool(_ERA_SUFFIX_RE.match(after))


# ─── Scope, branch and casualty type ─────────────────────────────────────────

# Ordered most specific first. on_paper and theatre_strength are tested
# before engaged because a sentence can carry both ("30,000 on paper, of
# whom 22,000 were engaged") and the qualifier is the informative half.
_ENGAGED_RE: Final = re.compile(
    r"\b(?:engaged|took part|present at the battle|committed to the (?:battle|engagement)"
    r"|deployed|on the field|fought (?:at|in)|in the battle(?: itself)?|participated"
    r"|brought to the field)\b",
    re.IGNORECASE,
)
_TOTAL_WORD_RE: Final = re.compile(r"\btotal\b", re.IGNORECASE)

_SCOPE_PATTERNS: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    (
        re.compile(
            r"\b(?:on paper|paper strength|nominal(?:ly)?|establishment strength"
            r"|authorised strength|authorized strength|muster rolls?|on the rolls"
            r"|official strength)\b",
            re.IGNORECASE,
        ),
        "on_paper",
    ),
    (
        re.compile(
            r"\b(?:theat(?:re|er)|in the region|across the (?:region|province|peninsula|empire)"
            r"|campaign strength|overall strength|total forces in|throughout the campaign"
            r"|army of the \w+ numbered|strength in \w+ was)\b",
            re.IGNORECASE,
        ),
        "theatre_strength",
    ),
    (
        re.compile(
            r"\b(?:available|at (?:his|her|their) disposal|could call upon|in reserve"
            r"|held in reserve|garrison of|mustered|raised|of whom only|not all of whom"
            r"|did not reach the field|never reached the (?:field|battle))\b",
            re.IGNORECASE,
        ),
        "available",
    ),
    (_ENGAGED_RE, "engaged"),
)

# A figure counting people rather than an army: a migrating nation, its
# non-combatants, a city's inhabitants. Such a figure says nothing about
# the force that fought, so it gets its own scope instead of "engaged".
_POPULATION_RE: Final = re.compile(
    r"\b(?:people(?![\'’]s)|peoples(?![\'’])|population|non-?combatants?"
    r"|civilians?(?!\s+(?:levies|militia|volunteers|fighters|defenders|troops|soldiers))"
    r"|inhabitants|refugees)\b",
    re.IGNORECASE,
)

_BRANCH_PATTERNS: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    (
        re.compile(
            r"\b(?:infantry(?:men)?|foot(?:men)?|legionaries|hoplites|musketeers|riflemen"
            r"|skirmishers|slingers|spearmen|pikemen|swordsmen|grenadiers|fusiliers)\b",
            re.I,
        ),
        "infantry",
    ),
    (
        re.compile(
            r"\b(?:cavalry(?:men)?|horse(?:men)?|horse archers|cuirassiers|hussars|knights"
            r"|dragoons|lancers|cataphracts)\b",
            re.I,
        ),
        "cavalry",
    ),
    (
        re.compile(
            r"\b(?:artillery(?:men)?|gunners|guns?|cannons?|field pieces?|siege engines?"
            r"|howitzers?)\b",
            re.I,
        ),
        "artillery",
    ),
    (
        re.compile(
            r"\b(?:ships?|galleys?|vessels?|warships?|triremes?|men-of-war|fleet of|sailors|seamen"
            r"|marines|frigates?|destroyers?|battleships?|cruisers?|submarines?|corvettes?|carriers?)\b",
            re.I,
        ),
        "naval",
    ),
    (
        re.compile(
            r"\b(?:aircraft|aeroplanes?|airplanes?|planes?|bombers?"
            r"|fighter[- ](?:jets?|planes?))\b",
            re.I,
        ),
        "air",
    ),
    (re.compile(r"\b(?:tanks?|armou?r(?:ed)?|panzers?|AFVs?)\b", re.I), "armour"),
    (re.compile(r"\b(?:irregulars?|militia|partisans?|tribesmen|levies)\b", re.I), "irregular"),
    (
        re.compile(r"\b(?:men|soldiers|troops|total|strength|combatants?|effectives?)\b", re.I),
        "total",
    ),
)

# "17,000 picked fighters" counts men. A bare "fighters" only means aircraft
# when the same field speaks of aircraft, bombers or air forces.
_BARE_FIGHTERS_RE: Final = re.compile(r"\bfighters?\b", re.IGNORECASE)
_AIR_CONTEXT_RE: Final = re.compile(
    r"\b(?:aircraft|aeroplanes?|airplanes?|planes?|bombers?|air\s*force|luftwaffe|jets?|sorties)\b",
    re.IGNORECASE,
)

_CASUALTY_PATTERNS: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    # "killed or wounded" is one undifferentiated figure, so it is a total
    # rather than being claimed for either bucket.
    (
        re.compile(
            r"\b(?:killed|dead|deaths?|slain|fatalities)\b\s*(?:or|and|,)\s*"
            r"\b(?:wounded|injured)\b",
            re.I,
        ),
        "total",
    ),
    (re.compile(r"\b(?:killed|dead|deaths?|slain|fatalities|kia)\b", re.I), "killed"),
    (re.compile(r"\b(?:wounded|injured|wia)\b", re.I), "wounded"),
    (re.compile(r"\b(?:captured|prisoners?|pows?|taken prisoner|surrendered)\b", re.I), "captured"),
    (re.compile(r"\b(?:missing|unaccounted)\b", re.I), "missing"),
)


def infer_scope(context: str) -> str:
    """Read what a troop number refers to from the wording around it.

    This is the most consequential judgement the extract stage makes. A
    number reported as theatre strength and stored as engaged overstates one
    side's force ratio, and the model has no way to detect it downstream.

    Args:
        context: The sentence or field value the number appeared in.

    Returns:
        A member of :data:`~pipeline.extractors.records.SCOPES`. ``unknown``
        when the wording says nothing either way, which is common and is not
        a failure. ``population`` when the figure counts people (a nation,
        non-combatants, inhabitants) rather than an army.
    """
    if _counts_only_people(context):
        return "population"
    return _infer_combat_scope(context)


_POPULATION_NEAR_WORDS: Final = 6


def _counts_only_people(context: str) -> bool:
    """Whether every figure in a text counts people by its own adjacent words.

    A population word elsewhere in the clause ("30,000 men drawn from the
    Boeotian people") says nothing about the figure, so each figure is judged
    from the few words directly around it, cut at its neighbouring numbers.

    Args:
        context: The sentence or field value the number appeared in.

    Returns:
        True when the text holds at least one figure and each one has
        population wording of its own.
    """
    found = list(_NUMBER_RE.finditer(context))
    if not found:
        return False
    previous_end = 0
    for index, match in enumerate(found):
        next_start = found[index + 1].start() if index + 1 < len(found) else len(context)
        before = context[previous_end : match.start()].split()[-_POPULATION_NEAR_WORDS:]
        after = context[match.end() : next_start].split()[:2]
        if not _POPULATION_RE.search(" ".join([*before, *after])):
            return False
        previous_end = match.end()
    return True


def _infer_combat_scope(context: str) -> str:
    """Read the force-size scope from wording, ignoring population wording.

    Args:
        context: The sentence or field value the number appeared in.

    Returns:
        A member of :data:`~pipeline.extractors.records.SCOPES` other than
        ``population``; ``unknown`` when the wording says nothing.
    """
    for pattern, scope in _SCOPE_PATTERNS:
        if pattern.search(context):
            return scope
    return "unknown"


def resolve_scope(declared: str | None, context: str, fallback: str = "unknown") -> str:
    """Settle the scope of one troop report.

    Wording in the source outranks a scope an LLM declared. The model is
    asked to judge scope and often does, but when it says ``engaged`` and
    the sentence it quoted says "in the theatre", the sentence is the
    evidence and the label is the inference.

    Args:
        declared: Scope as an extractor or LLM reported it, if any.
        context: The wording the number came from.
        fallback: Scope to use when neither the wording nor the declaration
            settles it. Callers parsing a field with defined semantics, such
            as an infobox strength field, pass their documented default here.

    Returns:
        A member of :data:`~pipeline.extractors.records.SCOPES`.
    """
    inferred = infer_scope(context)
    stated = coerce_scope(declared)

    if inferred != "unknown":
        if stated not in ("unknown", inferred):
            logger.info(
                "troop_scope_conflict_wording_wins",
                declared=stated,
                inferred=inferred,
                context=context[:160],
            )
        return inferred

    if stated != "unknown":
        return stated

    return coerce_scope(fallback)


def _branch_match(context: str, *, air_context: bool = False) -> str | None:
    """Look for an arm-of-service keyword.

    Args:
        context: Text to search.
        air_context: Whether the whole field speaks of aircraft, which is what
            lets a bare "fighters" count as aircraft instead of men.

    Returns:
        The branch the keyword implies, or None when none is present.
    """
    for pattern, branch in _BRANCH_PATTERNS:
        if pattern.search(context):
            return branch
        if branch == "air" and air_context and _BARE_FIGHTERS_RE.search(context):
            return "air"
    return None


def infer_branch(context: str) -> str:
    """Read which arm of service a number counts.

    Args:
        context: The sentence or field value the number appeared in.

    Returns:
        A member of :data:`~pipeline.extractors.records.BRANCHES`.
    """
    return _branch_match(context) or "total"


def _branch_near(segment: str, start: int, end: int, *, air_context: bool = False) -> str:
    """Read the branch for one number from the words closest to it.

    A clause can hold several numbers with different branches -- "95,000, of
    whom 16,000 cavalry" -- so the words immediately after the number, then
    immediately before it, are consulted before the clause as a whole. Each
    window stops at the next number, so one figure's qualifier cannot be read
    onto its neighbour.

    Args:
        segment: The clause the number was found in.
        start: Start offset of the number within the clause.
        end: End offset of the number within the clause.
        air_context: Whether the whole field speaks of aircraft.

    Returns:
        A member of :data:`~pipeline.extractors.records.BRANCHES`.
    """
    after = re.split(r"\d", segment[end : end + 36], maxsplit=1)[0]
    before = re.split(r"\d", segment[max(0, start - 36) : start])[-1]

    # Comma-delimited clause, which is where a qualifier usually sits when it
    # is further than a few words away ("12 legions, around 60,000 men").
    clause_start = segment.rfind(",", 0, start) + 1
    clause_end = segment.find(",", end)
    clause = segment[clause_start : clause_end if clause_end != -1 else len(segment)]

    windows = [after, before, clause]
    # The whole segment is only safe as a last resort when it holds a single
    # number; otherwise one figure's branch bleeds onto its neighbour.
    if len(_NUMBER_RE.findall(segment)) <= 1:
        windows.append(segment)

    for window in windows:
        branch = _branch_match(window, air_context=air_context)
        if branch is not None:
            return branch
    return "total"


def infer_casualty_type(context: str) -> str:
    """Read what kind of loss a casualty number counts.

    Args:
        context: The sentence or field value the number appeared in.

    Returns:
        A member of :data:`~pipeline.extractors.records.CASUALTY_TYPES`.
    """
    for pattern, casualty_type in _CASUALTY_PATTERNS:
        if pattern.search(context):
            return casualty_type
    return "total"


# ─── Quantity extraction ─────────────────────────────────────────────────────


def _after_window(segment: str, end: int) -> str:
    """The text right after a number, up to the next digit.

    Args:
        segment: The clause the number was found in.
        end: End offset of the number.

    Returns:
        At most 60 characters, cut at the next number so one figure's wording
        is never read onto its neighbour.
    """
    return re.split(r"\d", segment[end : end + 60], maxsplit=1)[0]


def _lead_label(segment: str, start: int) -> str:
    """The label ending in a colon directly before a number, if there is one.

    Args:
        segment: The clause the number was found in.
        start: Start offset of the number.

    Returns:
        The words since the previous number when they end in a colon
        ("Total:", "Civilians:"), else an empty string. Wording that does not
        end in a colon may belong to the previous figure instead.
    """
    previous_end = 0
    for found in _NUMBER_RE.finditer(segment):
        if found.end() > start:
            break
        previous_end = found.end()
    lead = segment[previous_end:start].strip()
    return lead if lead.endswith(":") else ""


def _figure_wording(
    segment: str, start: int, end: int, default_scope: str
) -> tuple[str, bool, bool]:
    """Read the scope and the two scope flags for one figure from its own words.

    Population wording is judged per figure: "30,000 men and 10,000 civilians"
    holds an army and a population, and a clause-wide match would call both
    population. A lone figure in its clause may use the whole clause.

    Args:
        segment: The clause the number was found in.
        start: Start offset of the number.
        end: End offset of the number.
        default_scope: Scope to fall back on.

    Returns:
        The scope, whether the figure's own words say it was engaged, and
        whether they call it a total.
    """
    local = f"{_lead_label(segment, start)} {_after_window(segment, end)}"
    lone = len(_NUMBER_RE.findall(segment)) <= 1
    if _POPULATION_RE.search(local) or (lone and _POPULATION_RE.search(segment)):
        return "population", False, False
    scope = _infer_combat_scope(segment)
    if scope == "unknown":
        scope = coerce_scope(default_scope)
    return (
        scope,
        bool(_ENGAGED_RE.search(local)),
        bool(_TOTAL_WORD_RE.search(local)),
    )


def _range_quantities(
    segment: str, default_scope: str, *, air_context: bool = False
) -> tuple[list[Quantity], str]:
    """Pull ranges out of a segment, returning them and the remaining text.

    Args:
        segment: One clause of a field value or sentence.
        default_scope: Scope to fall back on.
        air_context: Whether the whole field speaks of aircraft.

    Returns:
        The range endpoints as quantities, and the segment with the matched
        spans blanked so the single-number pass does not see them twice.
    """
    quantities: list[Quantity] = []
    remainder = segment

    for pattern in (_BETWEEN_RE, _DASH_RANGE_RE):
        for match in list(pattern.finditer(remainder)):
            low = parse_number(match.group("lo"))
            high = parse_number(match.group("hi"))
            if low is None or high is None or high < low:
                continue
            if _MONTH_AFTER_RE.match(remainder[match.end() :]):
                # Blanked so the single-number pass does not read "16" alone.
                remainder = (
                    remainder[: match.start()]
                    + " " * (match.end() - match.start())
                    + remainder[match.end() :]
                )
                continue
            branch = _branch_near(segment, match.start(), match.end(), air_context=air_context)
            scope, engaged, total_worded = _figure_wording(
                segment, match.start(), match.end(), default_scope
            )
            estimate = bool(_ESTIMATE_RE.search(segment))
            lead, noun = _surround(segment, match.start(), match.end())
            trailing = _after_window(segment, match.end())
            for value, bound in ((low, "lower"), (high, "upper")):
                quantities.append(
                    Quantity(
                        value=value,
                        branch=branch,
                        scope=scope,
                        is_estimate=estimate,
                        is_lower_bound=bound == "lower",
                        is_upper_bound=bound == "upper",
                        context=segment.strip(),
                        lead=lead,
                        noun=noun,
                        trailing=trailing,
                        explicit_engaged=engaged,
                        total_worded=total_worded,
                    )
                )
            remainder = (
                remainder[: match.start()]
                + " " * (match.end() - match.start())
                + remainder[match.end() :]
            )

    return quantities, remainder


_NOUN_STOP_RE: Final = re.compile(r"[,;:()\[\]\d]")
_LEAD_NOISE_RE: Final = re.compile(
    r"\b(?:about|approximately|approx|roughly|around|some|estimated|circa|perhaps|possibly"
    r"|nearly|almost|over|under|up to|at least|at most|more than|less than|fewer than|c|ca|est"
    r"|strength|forces?|army|total|combined|overall|entire|whole)\b",
    re.IGNORECASE,
)
# Words after a number that describe the men in general rather than naming a
# contingent, so "30,000 men" or "86,000 engaged" is a side-level figure.
_GENERIC_NOUNS: Final[frozenset[str]] = frozenset(
    {
        "men",
        "soldiers",
        "troops",
        "combatants",
        "effectives",
        "personnel",
        "strong",
        "engaged",
        "total",
        "present",
        "regulars",
        "regular",
        "fighting",
        "in",
        "at",
        "on",
        "of",
        "to",
        "or",
        "and",
        "plus",
        "from",
        "by",
        "with",
        "under",
        "after",
        "during",
        "est",
        "estimated",
        "approx",
        "available",
        "deployed",
        "men-at-arms",
        "warriors",
        "ration",
        "combat",
        "operational",
        "strength",
        "force",
        "forces",
        "effective",
        "manpower",
        "initial",
        "overall",
    }
)
_EQUIPMENT_NOUN_RE: Final = re.compile(
    r"^(?:guns?|cannons?|pieces|howitzers?|mortars?|ships?|vessels?|galleys?|warships?|triremes?"
    r"|frigates?|destroyers?|battleships?|cruisers?|submarines?|corvettes?|carriers?"
    r"|aircraft|planes?|aeroplanes?|bombers?|tanks?|vehicles?|gunboats?|junks?"
    r"|boats?|barges?|elephants?|oxen|camels?|chariots?|wagons?|supply)\b",
    re.IGNORECASE,
)


def _surround(segment: str, start: int, end: int) -> tuple[str, str]:
    """Read the words before and after one number in its clause.

    Args:
        segment: The clause the number was found in.
        start: Start offset of the number.
        end: End offset of the number.

    Returns:
        The text since the previous number (or the clause start), and the
        words right after the number up to the next number or punctuation.
    """
    previous_end = 0
    for found in _NUMBER_RE.finditer(segment):
        if found.end() > start:
            break
        previous_end = found.end()
    lead = segment[previous_end:start].strip()
    tail = segment[end:]
    cut = _NOUN_STOP_RE.search(tail)
    noun = (tail[: cut.start()] if cut else tail).strip()
    return lead, noun


def is_equipment_count(quantity: Quantity) -> bool:
    """Whether a figure counts guns, ships, aircraft or vehicles rather than men.

    Args:
        quantity: A parsed figure.

    Returns:
        True when the words right after the number name equipment. Such a
        count must never be summed with, or mistaken for, a headcount.
    """
    if any(_EQUIPMENT_NOUN_RE.match(w) for w in quantity.noun.split()[:3]):
        return True
    # "300 fighters" in an air field counts aircraft; "17,000 picked fighters"
    # elsewhere counts men, and the branch parser has already told them apart.
    return quantity.branch == "air" and bool(_BARE_FIGHTERS_RE.search(quantity.noun))


def is_labelled_figure(quantity: Quantity) -> bool:
    """Whether words before the number label it ("Wellington's army:", "II Corps:").

    Args:
        quantity: A parsed figure.

    Returns:
        True when the lead holds a label once estimate words are removed.
        Such a label names a unit, a phase or a date as often as a part of
        the side, so the figure is not safe to sum.
    """
    residue = _LEAD_NOISE_RE.sub(" ", quantity.lead)
    return bool(re.search(r"[A-Za-z]{2,}", residue) or re.search(r"\b[IVX]{1,4}\b", residue))


def is_orphan_figure(quantity: Quantity) -> bool:
    """Whether a figure has neither a label nor a noun saying what it counts.

    Args:
        quantity: A parsed figure.

    Returns:
        True for a bare number such as the "31,000" left under a heading the
        page dropped.
    """
    return not re.search(r"[A-Za-z]", quantity.noun) and not re.search(r"\w", quantity.lead)


# "Modern estimate(s)", "modern estimation", "modern est.", "modern historians":
# the regime parser's modern_scholarly wording, shared with reconcile's
# claims.classify_regime so the two cannot drift apart.
MODERN_ESTIMATE_PATTERN: Final = r"\bmodern\s+(?:estimat|est\b|source|histor|scholar)\w*"
_MODERN_LABEL_RE: Final = re.compile(MODERN_ESTIMATE_PATTERN, re.IGNORECASE)
_MODERN_LEAD_RE: Final = re.compile(rf"{MODERN_ESTIMATE_PATTERN}\s*:\s*$", re.IGNORECASE)
_BRACKETED_RE: Final = re.compile(r"\([^()]*\)|\[[^\[\]]*\]")

# "15,000 under Ruechel": a force under a named commander is a component.
# Case matters: "20,000 under arms" is still a side figure.
_UNDER_NAME_RE: Final = re.compile(
    r"^(?:(?:men|troops|soldiers)\s+)?(?:directly\s+|immediately\s+)?under\s+"
    r"(?:(?:the\s+)?(?:command|orders|leadership)\s+of\s+)?"
    r"[A-Z\u00C0-\u00DE]"
)


def _lead_has_label(lead: str) -> bool:
    """Whether a lead holds a label once estimate words are removed.

    Args:
        lead: The words between the previous number and this one.

    Returns:
        True for a unit, nationality or phase label ("II Corps:", "French:").
    """
    residue = _LEAD_NOISE_RE.sub(" ", lead)
    return bool(re.search(r"[A-Za-z]{2,}", residue) or re.search(r"\b[IVX]{1,4}\b", residue))


def is_modern_estimate(quantity: Quantity) -> bool:
    """Whether a figure is labelled as a modern scholarly estimate.

    A label after the figure ("(modern estimates)") binds to that figure; a
    label before it counts only in the "Modern estimate: 108,000" form, since
    wording ahead of a figure may belong to the previous one.

    Args:
        quantity: A parsed figure.

    Returns:
        True when the figure carries modern-estimate wording of its own and
        names no contingent, unit or equipment.
    """
    if not (_MODERN_LABEL_RE.search(quantity.trailing) or _MODERN_LEAD_RE.search(quantity.lead)):
        return False
    words = re.findall(r"[^\W\d_][\w'-]*", quantity.noun)
    if words and words[0].lower() not in _GENERIC_NOUNS:
        return False
    lead = _BRACKETED_RE.sub(" ", _MODERN_LEAD_RE.sub(" ", quantity.lead))
    return not _lead_has_label(lead)


def classify_figure(quantity: Quantity, *, first_in_field: bool) -> str:
    """Decide whether a headcount is the side's total, one arm, or one component.

    Args:
        quantity: A parsed figure.
        first_in_field: Whether the figure sits on the first line of its
            field (or of its date or phase block). A bare number is a
            side-level total only there; the same bare number on a later line
            follows labels the page dropped (nested contingents) and is a
            component.

    Returns:
        ``"arm"`` when the wording names an arm of service, ``"component"``
        for a contingent, nationality, unit or labelled sub-force, and
        ``"total"`` for an explicit total, a labelled modern estimate, or an
        unqualified side figure.
    """
    if quantity.branch != "total":
        return "arm"

    if _UNDER_NAME_RE.match(quantity.noun):
        return "component"

    if is_modern_estimate(quantity):
        return "total"

    if re.search(r"\btotal\b", f"{quantity.lead} {quantity.noun}", re.IGNORECASE):
        return "total"

    # Whatever precedes the number once estimate and generic words are
    # removed is a label ("Wellington's army:", "II Corps:", "French:").
    if is_labelled_figure(quantity):
        return "component"

    words = re.findall(r"[^\W\d_][\w'-]*", quantity.noun)
    if words and words[0].lower() not in _GENERIC_NOUNS:
        return "component"

    if is_orphan_figure(quantity) and not first_in_field:
        return "component"
    return "total"


# Date and phase headings that sit on a line of their own and label the figures
# below them: "16-17 October:", "Initial strength:", "At the start:", "On day 2:".
_DASH_CLASS: Final = "[-\u2010-\u2015\u2212]"
_MONTHS_PATTERN: Final = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
    r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)
_PHASE_WORDS_PATTERN: Final = (
    r"initial(?:\s+(?:strength|force|forces|deployment))?"
    r"|at\s+the\s+(?:start|outset|beginning)(?:\s+of\s+the\s+(?:battle|engagement))?"
    r"|(?:on\s+)?day\s+\d+"
    r"|(?:on\s+)?the\s+(?:first|second|third|fourth|final)\s+day"
)
_PHASE_PATTERN: Final = (
    rf"(?:(?:on|as of|by|from|at)\s+)?\d{{1,2}}(?:\s*{_DASH_CLASS}\s*\d{{1,2}})?"
    rf"\s+(?:{_MONTHS_PATTERN})\.?(?:\s+\d{{4}})?|{_PHASE_WORDS_PATTERN}"
)
_PHASE_HEADING_RE: Final = re.compile(rf"^\W*(?P<phase>{_PHASE_PATTERN})\s*:\s*$", re.IGNORECASE)
_PHASE_INLINE_RE: Final = re.compile(
    rf"^\W*(?P<phase>{_PHASE_PATTERN})\s*:\s*(?=\S)", re.IGNORECASE
)


def _phase_heading(segment: str) -> str | None:
    """The phase named by a line that is only a heading, else None.

    Args:
        segment: One line of a field value.

    Returns:
        The heading text without its colon, or None for any other line.
    """
    match = _PHASE_HEADING_RE.match(segment.strip())
    return match.group("phase").strip() if match else None


def _decorate(
    quantities: list[Quantity], segment: str, phase: str, *, block_first: bool, inline: str | None
) -> list[Quantity]:
    """Attach the phase and block position to one line's figures.

    Args:
        quantities: The figures found on the line.
        segment: The line.
        phase: The heading currently in force, or an empty string.
        block_first: Whether this is the first figure line under that heading.
        inline: A phase named at the start of the line itself, if any.

    Returns:
        The figures, with the heading prefixed to the context when it came
        from a line above, so the row stays auditable.
    """
    label = inline or phase
    context = segment.strip()
    if phase and not inline:
        context = f"{phase}: {context}"
    decorated: list[Quantity] = []
    for quantity in quantities:
        if quantity.total_worded and not inline:
            # "Total: 250,000" after dated phases sums the field; it belongs
            # to no phase.
            decorated.append(replace(quantity, block_first=block_first))
            continue
        lead = quantity.lead
        if inline:
            lead = re.sub(rf"^\W*(?:{_PHASE_PATTERN})\s*:\s*", "", quantity.lead, flags=re.I)
        decorated.append(
            replace(
                quantity,
                phase=label,
                block_first=block_first,
                context=context,
                lead=lead,
            )
        )
    return decorated


def parse_quantities(text: str, *, default_scope: str = "unknown") -> list[Quantity]:
    """Read every troop-like number out of a piece of text.

    A range becomes two quantities, flagged as the lower and upper bound, so
    that the reconcile stage sees the interval the source actually gave
    rather than a midpoint nobody wrote down.

    A line that is only a date or phase heading ("16-17 October:", "Initial
    strength:") labels the figures below it until the next heading; the label
    is kept on each figure's ``phase`` and prefixed to its context. A field
    that words one figure as deployed or engaged and another as a total reads
    the total as the larger ``available`` pool.

    Args:
        text: Cleaned text, typically one infobox strength field.
        default_scope: Scope to apply when the wording does not settle it.
            Callers with defined field semantics pass their default here;
            callers reading free prose should leave it ``"unknown"``.

    Returns:
        The quantities found, in source order.
    """
    if not text:
        return []

    quantities: list[Quantity] = []
    air_context = bool(_AIR_CONTEXT_RE.search(text))
    phase = ""
    awaiting_first_line = False

    for segment in _SEGMENT_SPLIT_RE.split(text):
        if not segment.strip():
            continue

        heading = _phase_heading(segment)
        if heading is not None:
            phase, awaiting_first_line = heading, True
            continue

        ranged, remainder = _range_quantities(segment, default_scope, air_context=air_context)
        found = list(ranged)

        for match in _NUMBER_RE.finditer(remainder):
            trailing = remainder[match.end() :]
            if (
                _ORDINAL_RE.match(trailing)
                or _FORMATION_RE.match(trailing)
                or _MONTH_AFTER_RE.match(trailing)
            ):
                continue
            if _looks_like_year(remainder, match):
                continue
            value = parse_number(match.group(0))
            if value is None or value <= 0:
                continue

            before = remainder[max(0, match.start() - 60) : match.start()]
            lead, noun = _surround(segment, match.start(), match.end())
            scope, engaged, total_worded = _figure_wording(
                segment, match.start(), match.end(), default_scope
            )
            found.append(
                Quantity(
                    value=value,
                    branch=_branch_near(
                        segment, match.start(), match.end(), air_context=air_context
                    ),
                    scope=scope,
                    is_estimate=bool(_ESTIMATE_RE.search(before)),
                    is_lower_bound=bool(_LOWER_RE.search(before)),
                    is_upper_bound=bool(_UPPER_RE.search(before)),
                    context=segment.strip(),
                    lead=lead,
                    noun=noun,
                    trailing=_after_window(segment, match.end()),
                    explicit_engaged=engaged,
                    total_worded=total_worded,
                )
            )

        inline_match = _PHASE_INLINE_RE.match(segment)
        inline = inline_match.group("phase").strip() if inline_match else None
        if found and (phase or inline):
            found = _decorate(found, segment, phase, block_first=awaiting_first_line, inline=inline)
        if found:
            awaiting_first_line = False
        quantities.extend(found)

    return _totals_beside_engaged_figures_are_available(quantities)


def _totals_beside_engaged_figures_are_available(quantities: list[Quantity]) -> list[Quantity]:
    """Read "96,000 total" next to "40,000 deployed" as the larger available pool.

    A lone "Total: 37,000" says nothing about scope and keeps its default; the
    contrast with a figure worded as deployed or engaged is what shows the
    total is wider than the force that fought.

    Args:
        quantities: Every figure of one field.

    Returns:
        The figures, with such totals rescoped to ``available``.
    """
    if not any(q.explicit_engaged for q in quantities):
        return quantities
    return [
        replace(q, scope="available")
        if q.total_worded and not q.explicit_engaged and q.scope in ("engaged", "unknown")
        else q
        for q in quantities
    ]
