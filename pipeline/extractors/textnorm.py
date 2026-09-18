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
from dataclasses import dataclass
from typing import Final

import mwparserfromhell
import structlog

from pipeline.extractors.records import coerce_scope

__all__ = [
    "Quantity",
    "infer_branch",
    "infer_casualty_type",
    "infer_scope",
    "normalise_label",
    "parse_number",
    "parse_quantities",
    "resolve_scope",
    "split_list_items",
    "strip_markup",
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
    }
)

# Templates that wrap content for presentation only; their first positional
# parameter is the content and should survive.
_PASSTHROUGH_TEMPLATES: Final[frozenset[str]] = frozenset(
    {"nowrap", "nobold", "noitalic", "small", "big", "nobr", "nsmdns"}
)

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
}

_CIRCA_TEMPLATES: Final[frozenset[str]] = frozenset({"circa", "c.", "ca", "ca.", "approx"})


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
            items = [str(p.value).strip() for p in template.params if not p.showkey]
            code.replace(template, "\n" + "\n".join(i for i in items if i) + "\n")
        elif name in _PASSTHROUGH_TEMPLATES and template.params:
            code.replace(template, str(template.params[0].value))
        elif name in _MARKER_TEMPLATES:
            code.replace(template, f" ({_MARKER_TEMPLATES[name]})")
        elif name in _CIRCA_TEMPLATES:
            rest = str(template.params[0].value) if template.params else ""
            code.replace(template, f"c. {rest}")

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
    return "\n".join(line.strip() for line in text.splitlines()).strip()


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
    r"\s*(?:legions?|corps|divisions?|regiments?|battalions?|brigades?|squadrons?"
    r"|companies|cohorts?|maniples?|armies|columns?|wings?)\b",
    re.IGNORECASE,
)
_YEAR_PREFIX_RE: Final = re.compile(
    r"(?:\bin\b|\bof\b|january|february|march|april|may|june|july|august|september"
    r"|october|november|december)\s*$",
    re.IGNORECASE,
)
_ERA_SUFFIX_RE: Final = re.compile(r"\s*(?:BCE?|AD|CE)\b")

_ESTIMATE_RE: Final = re.compile(
    r"\b(?:about|approximately|approx\.?|roughly|around|some|estimated|an estimate"
    r"|circa|perhaps|possibly|nearly|almost|upwards of)\b|c\.\s|ca\.\s|~",
    re.IGNORECASE,
)
_UPPER_RE: Final = re.compile(r"\b(?:up to|at most|no more than|fewer than|less than|under)\b",
                              re.IGNORECASE)
_LOWER_RE: Final = re.compile(r"\b(?:at least|more than|over|in excess of|upwards of|at minimum)\b",
                              re.IGNORECASE)

_SEGMENT_SPLIT_RE: Final = re.compile(r"[\n;]+")


@dataclass(frozen=True)
class Quantity:
    """A number read out of prose, with what the prose said about it."""

    value: float
    branch: str = "total"
    scope: str = "unknown"
    is_estimate: bool = False
    is_lower_bound: bool = False
    is_upper_bound: bool = False
    context: str = ""


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
    (
        re.compile(
            r"\b(?:engaged|took part|present at the battle|committed to the (?:battle|engagement)"
            r"|deployed|on the field|fought (?:at|in)|in the battle(?: itself)?|participated"
            r"|brought to the field)\b",
            re.IGNORECASE,
        ),
        "engaged",
    ),
)

_BRANCH_PATTERNS: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    (re.compile(r"\b(?:infantry|foot(?:men)?|legionaries|hoplites|musketeers)\b", re.I),
     "infantry"),
    (re.compile(r"\b(?:cavalry|horse(?:men)?|horse archers|cuirassiers|hussars|knights)\b", re.I),
     "cavalry"),
    (re.compile(r"\b(?:artillery|guns?|cannons?|field pieces?|siege engines?|howitzers?)\b", re.I),
     "artillery"),
    (re.compile(r"\b(?:ships?|galleys?|vessels?|warships?|triremes?|men-of-war|fleet of)\b", re.I),
     "naval"),
    (re.compile(r"\b(?:aircraft|aeroplanes?|airplanes?|planes?|bombers?|fighters?)\b", re.I),
     "air"),
    (re.compile(r"\b(?:tanks?|armou?r(?:ed)?|panzers?|AFVs?)\b", re.I), "armour"),
    (re.compile(r"\b(?:irregulars?|militia|partisans?|tribesmen|levies)\b", re.I), "irregular"),
    (re.compile(r"\b(?:men|soldiers|troops|total|strength|combatants?|effectives?)\b", re.I),
     "total"),
)

_CASUALTY_PATTERNS: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    # "killed or wounded" is one undifferentiated figure, so it is a total
    # rather than being claimed for either bucket.
    (re.compile(r"\b(?:killed|dead|deaths?|slain|fatalities)\b\s*(?:or|and|,)\s*"
                r"\b(?:wounded|injured)\b", re.I), "total"),
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
        a failure.
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


def _branch_match(context: str) -> str | None:
    """Look for an arm-of-service keyword.

    Args:
        context: Text to search.

    Returns:
        The branch the keyword implies, or None when none is present.
    """
    for pattern, branch in _BRANCH_PATTERNS:
        if pattern.search(context):
            return branch
    return None


def infer_branch(context: str) -> str:
    """Read which arm of service a number counts.

    Args:
        context: The sentence or field value the number appeared in.

    Returns:
        A member of :data:`~pipeline.extractors.records.BRANCHES`.
    """
    return _branch_match(context) or "total"


def _branch_near(segment: str, start: int, end: int) -> str:
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
        branch = _branch_match(window)
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


def _range_quantities(segment: str, default_scope: str) -> tuple[list[Quantity], str]:
    """Pull ranges out of a segment, returning them and the remaining text.

    Args:
        segment: One clause of a field value or sentence.
        default_scope: Scope to fall back on.

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
            branch = _branch_near(segment, match.start(), match.end())
            scope = resolve_scope(None, segment, default_scope)
            estimate = bool(_ESTIMATE_RE.search(segment))
            quantities.append(
                Quantity(
                    value=low,
                    branch=branch,
                    scope=scope,
                    is_estimate=estimate,
                    is_lower_bound=True,
                    context=segment.strip(),
                )
            )
            quantities.append(
                Quantity(
                    value=high,
                    branch=branch,
                    scope=scope,
                    is_estimate=estimate,
                    is_upper_bound=True,
                    context=segment.strip(),
                )
            )
            remainder = remainder[: match.start()] + " " * (match.end() - match.start()) \
                + remainder[match.end() :]

    return quantities, remainder


def parse_quantities(text: str, *, default_scope: str = "unknown") -> list[Quantity]:
    """Read every troop-like number out of a piece of text.

    A range becomes two quantities, flagged as the lower and upper bound, so
    that the reconcile stage sees the interval the source actually gave
    rather than a midpoint nobody wrote down.

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

    for segment in _SEGMENT_SPLIT_RE.split(text):
        if not segment.strip():
            continue

        ranged, remainder = _range_quantities(segment, default_scope)
        quantities.extend(ranged)

        for match in _NUMBER_RE.finditer(remainder):
            trailing = remainder[match.end() :]
            if _ORDINAL_RE.match(trailing) or _FORMATION_RE.match(trailing):
                continue
            if _looks_like_year(remainder, match):
                continue
            value = parse_number(match.group(0))
            if value is None or value <= 0:
                continue

            before = remainder[max(0, match.start() - 60) : match.start()]
            quantities.append(
                Quantity(
                    value=value,
                    branch=_branch_near(segment, match.start(), match.end()),
                    scope=resolve_scope(None, segment, default_scope),
                    is_estimate=bool(_ESTIMATE_RE.search(before)),
                    is_lower_bound=bool(_LOWER_RE.search(before)),
                    is_upper_bound=bool(_UPPER_RE.search(before)),
                    context=segment.strip(),
                )
            )

    return quantities
