"""
Turning what a source wrote into keys a matcher can compare.

Wikipedia infoboxes name commanders in every form a human might: with rank
("Gen. George S. Patton"), with honorific ("Sir Edward Codrington"), with
regnal number ("Ramesses II"), inverted ("Wellesley, Arthur"), annotated with
their fate ("Marcus Licinius Crassus †"), and sometimes not naming anybody at
all ("Unknown", "various local chieftains").

This module produces *keys*, never replacements. The surface form a source
used is evidence -- it becomes a ``general_aliases`` row -- so nothing here
overwrites it. :func:`name_keys` returns several normalised spellings of one
mention and the matcher succeeds if any of them matches any candidate key,
which is deliberately generous: the lifespan gate and the ambiguity margin in
:mod:`pipeline.resolvers.matcher` are what keep the generosity safe.

Two rules that look like omissions and are not:

- **Regnal numerals are never stripped.** "Ramesses II" and "Ramesses III" are
  different men with different records, and folding them together would pool
  two commanders' skill into one estimate.
- **Titles are stripped only from the front, and only into an extra key.**
  "Duke of Wellington" is the common name, not an ornament on one, so the
  unstripped form has to stay in play.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final

from pipeline.extractors.textnorm import normalise_label

__all__ = [
    "LEADING_TITLES",
    "clean_surface",
    "fold",
    "is_placeholder",
    "is_title_free",
    "name_keys",
]

# Ranks, honorifics and offices that can stand in front of a personal name.
# Longest first, because "field marshal" must win over "marshal". Stripping
# these produces an *additional* key; see the module docstring.
LEADING_TITLES: Final[tuple[str, ...]] = (
    "field marshal",
    "lieutenant general",
    "lieutenant colonel",
    "brigadier general",
    "major general",
    "vice admiral",
    "rear admiral",
    "grand admiral",
    "grand duke",
    "grand prince",
    "fleet admiral",
    "generalissimo",
    "generalfeldmarschall",
    "generaloberst",
    "grand vizier",
    "lord protector",
    "prince regent",
    "archduke",
    "archbishop",
    "brigadier",
    "commodore",
    "commander",
    "lieutenant",
    "chancellor",
    "president",
    "proconsul",
    "princess",
    "marshall",
    "marshal",
    "general",
    "admiral",
    "colonel",
    "captain",
    "sergeant",
    "governor",
    "emperor",
    "empress",
    "dictator",
    "praetor",
    "tribune",
    "legate",
    "consul",
    "caliph",
    "sultan",
    "maharaja",
    "shogun",
    "daimyo",
    "hetman",
    "voivode",
    "despot",
    "duchess",
    "viscount",
    "countess",
    "baroness",
    "cardinal",
    "bishop",
    "prince",
    "major",
    "queen",
    "king",
    "tsar",
    "czar",
    "kaiser",
    "shah",
    "emir",
    "pasha",
    "pope",
    "doge",
    "duke",
    "earl",
    "count",
    "baron",
    "raja",
    "lord",
    "lady",
    "dame",
    "sir",
    "gen",
    "lt",
    "col",
    "maj",
    "capt",
    "cpt",
    "adm",
    "cdr",
    "brig",
    "sgt",
    "mgen",
    "ltgen",
    "st",
    "saint",
    "don",
    "dom",
)

# Mentions that name no one. Writing a generals row for any of these would
# create a phantom commander who fought in every battle whose infobox said
# "Unknown" -- a single entity with hundreds of battles and no person behind
# it, which would sit near the top of any skill ranking.
_PLACEHOLDERS: Final[frozenset[str]] = frozenset(
    {
        "",
        "unknown",
        "unknown commanders",
        "unnamed",
        "none",
        "na",
        "n a",
        "nil",
        "various",
        "various commanders",
        "several",
        "others",
        "other",
        "et al",
        "and others",
        "multiple",
        "multiple commanders",
        "many",
        "anonymous",
        "disputed",
        "not known",
        "no commander",
        "local leaders",
        "tribal leaders",
        "unclear",
        "unspecified",
    }
)

# Footnote markers, citation brackets, and the parenthetical fate annotations
# infoboxes hang off a name.
_BRACKETED_RE: Final = re.compile(r"\[[^\]]*\]|\([^)]*\)|\{[^}]*\}")
# Daggers, crosses, skulls and the like mark a commander killed in the action.
_FATE_SYMBOL_RE: Final = re.compile(r"[†‡✝☠#* ]+")
_FATE_WORDS_RE: Final = re.compile(
    r"\b(?:POW|WIA|KIA|MIA|DOW|executed|captured|killed in action|surrendered)\b",
    re.IGNORECASE,
)
_CONNECTIVE_RE: Final = re.compile(r"^\s*(?:and|&|with|under|plus)\b\s*", re.IGNORECASE)
_WHITESPACE_RE: Final = re.compile(r"\s+")

# A trailing segment short enough, and free enough of titles and digits, to be
# a forename in an inverted "Surname, Forename" spelling.
_INVERSION_STOPWORDS: Final[frozenset[str]] = frozenset(
    {"of", "the", "jr", "sr", "duke", "earl", "count", "baron", "lord", "prince", "st"}
)

_ABSENCE_HEAD_RE: Final = re.compile(
    r"^(?:unknown|unnamed|various|several|numerous|multiple|assorted)\b"
)


def clean_surface(raw: str) -> str:
    """Strip annotation from a mention, keeping its spelling and case.

    This is what becomes a ``general_aliases.alias_name`` and, for a new
    entity, a ``generals.canonical_name``. It removes what an editor added
    around the name and nothing that is part of it.

    Args:
        raw: The name exactly as a source wrote it.

    Returns:
        The name with citation brackets, parentheticals, fate markers and
        leading connectives removed, and whitespace collapsed.
    """
    text = unicodedata.normalize("NFKC", raw)
    text = _BRACKETED_RE.sub(" ", text)
    text = _FATE_WORDS_RE.sub(" ", text)
    text = _FATE_SYMBOL_RE.sub(" ", text)
    text = _CONNECTIVE_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text.strip(" ,;:-–—")


def fold(raw: str) -> str:
    """Fold a name to a comparison key.

    Args:
        raw: A name, cleaned or not.

    Returns:
        A lowercase, accent-free, punctuation-free key. Empty when the input
        holds no word characters.
    """
    return normalise_label(clean_surface(raw))


def is_placeholder(raw: str) -> bool:
    """Whether a mention names nobody.

    Args:
        raw: The name as a source wrote it.

    Returns:
        True for the vocabulary of non-names infoboxes use, for anything that
        folds to nothing, and for a mention with no letters in it. Such a
        mention is logged to ``missing_data_log`` rather than made a general.
    """
    folded = fold(raw)
    if folded in _PLACEHOLDERS:
        return True
    if not any(char.isalpha() for char in folded):
        return True
    # "unknown roman commander", "several gallic chieftains": the qualifier
    # varies but the head noun is still an absence of a name.
    return bool(_ABSENCE_HEAD_RE.match(folded))


def is_title_free(raw: str) -> bool:
    """Whether a name carries no leading rank or honorific.

    Args:
        raw: A name as a source wrote it.

    Returns:
        True when stripping leading titles would change nothing. Used to
        choose which surface form becomes a general's canonical name:
        "Gen. Patton" and "Patton" are the same person, and the one without
        the rank is the better name to publish.
    """
    folded = fold(raw)
    return bool(folded) and folded == _strip_leading_titles(folded)


def _strip_leading_titles(folded: str) -> str:
    """Remove every leading rank or honorific from a folded key.

    Args:
        folded: A key from :func:`fold`.

    Returns:
        The key with leading titles removed, or the input unchanged when
        stripping would leave nothing. "general" alone stays "general".
    """
    current = folded
    while True:
        for title in LEADING_TITLES:
            prefix = f"{title} "
            if current.startswith(prefix):
                remainder = current[len(prefix) :].strip()
                if remainder:
                    current = remainder
                    break
        else:
            return current


def _uninvert(cleaned: str) -> str:
    """Flip a "Surname, Forename" spelling into reading order.

    Args:
        cleaned: A surface form from :func:`clean_surface`.

    Returns:
        The flipped name, or an empty string when the comma does not mark an
        inversion. "Arthur Wellesley, 1st Duke of Wellington" is not
        inverted, and flipping it would produce nonsense, so a tail carrying
        a digit or a title is left alone.
    """
    if cleaned.count(",") != 1:
        return ""

    head, tail = (part.strip() for part in cleaned.split(","))
    if not head or not tail:
        return ""

    tokens = fold(tail).split()
    if not tokens or len(tokens) > 2:
        return ""
    if any(token in _INVERSION_STOPWORDS or token.isdigit() for token in tokens):
        return ""

    return f"{tail} {head}"


def name_keys(raw: str) -> tuple[str, ...]:
    """Generate every normalised spelling a mention should match on.

    Args:
        raw: The name as a source wrote it.

    Returns:
        Keys in decreasing fidelity to the source: the folded surface form
        first, then the title-stripped form, the uninverted form, and the
        form with a leading "of" removed. Deduplicated, empties dropped, and
        empty overall for a mention that names nobody.
    """
    cleaned = clean_surface(raw)
    if not cleaned:
        return ()

    keys: list[str] = []

    def add(value: str) -> None:
        if value and value not in keys:
            keys.append(value)

    surface = fold(cleaned)
    add(surface)

    stripped = _strip_leading_titles(surface)
    add(stripped)

    # "Duke of Wellington" strips to "of wellington"; the territorial name on
    # its own is how Wikidata aliases it.
    if stripped.startswith("of "):
        add(stripped[3:].strip())

    add(fold(_uninvert(cleaned)))

    return tuple(keys)
