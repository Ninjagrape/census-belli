"""
Deterministic, non-statistical classification for the reconcile stage.

Three functions here feed the hierarchical bias model without touching it:
:func:`classify_regime` labels the kind of claim a report's text makes,
:func:`era_flag` says whether a battle is ancient without ever guessing on a
missing date, and :func:`roundness` scores how conventional a reported figure
looks. :func:`assign_lineages` groups reports that repeat one underlying
claim rather than offering independent corroboration of it.

None of this fits a model or touches a database. Each function takes plain
values and :class:`~pipeline.reconcilers.records.Report` instances and
returns plain values, which is what lets the classification rules be pinned
against real text -- see ``tests/unit/test_reconcile_claims.py`` -- without a
network or a database.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from typing import Final

from pipeline.reconcilers.records import Report

__all__ = ["assign_lineages", "classify_regime", "era_flag", "roundness"]

# Citation refs -- [12], [a], [nb 1], [citation needed], [better source
# needed] -- appear in 52.7% of real Wikipedia infobox strength strings.
# Stripped before every other check so a bracket landing mid-phrase (or
# immediately after a keyword, with no space) can never split a match.
_CITATION_REF: Final[re.Pattern[str]] = re.compile(r"\[[^\]]*\]")

# Named ancient authors are evidence on their own -- Livy naming a figure is
# an ancient claim whether or not the surrounding text says "ancient
# sources". Wikipedia bios are inconsistent about whether the label ancient
# provenance in prose or just cite a name, so both are checked.
_ANCIENT_AUTHORS: Final[tuple[str, ...]] = (
    "herodotus",
    "arrian",
    "livy",
    "polybius",
    "plutarch",
    "diodorus",
    "curtius",
    "tacitus",
    "caesar",
    "thucydides",
    "xenophon",
    "josephus",
    "appian",
)

# Same reasoning as the ancient authors, for the medieval chronicle regime.
_CHRONICLERS: Final[tuple[str, ...]] = (
    "froissart",
    "fulcher",
    "albert of aix",
    "villehardouin",
    "joinville",
    "matthew paris",
    "raymond of aguilers",
)

# Checked in this order -- the first pattern to match wins. modern_scholarly
# is checked first because it is the model's anchor and because a string
# naming both a modern estimate and an ancient source alongside it
# ("52,930 (modern estimates) 250,000 (ancient sources)") is presenting a
# deliberate contrast, and the modern figure is the one closer to what the
# report's own reported_value usually is in that pattern. Every other
# ordering choice here is a tiebreak, not a claim that the categories are
# mutually exclusive in the source text.
_REGIME_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    (
        "modern_scholarly",
        re.compile(
            r"\bmodern\s+(?:estimate|source|histor|scholar)\w*\b",
            re.IGNORECASE,
        ),
    ),
    (
        "ancient_claim",
        re.compile(
            r"\bancient\s+(?:source|estimate|claim)\w*\b"
            r"|\bprimary\b(?:\s+\w+){0,2}\s+sources?\b"
            r"|'s\s+claim\b"
            r"|\bclaimed\s+by\b"
            r"|\b(?:" + "|".join(_ANCIENT_AUTHORS) + r")\b",
            re.IGNORECASE,
        ),
    ),
    (
        "chronicle",
        re.compile(
            r"\bchronicles?\b|\bchroniclers?\b|\bcontemporary\s+account\w*\b"
            r"|\b(?:" + "|".join(re.escape(name) for name in _CHRONICLERS) + r")\b",
            re.IGNORECASE,
        ),
    ),
    (
        "administrative_partisan",
        re.compile(
            r"\bmuster(?:s|ed|ing|roll)?\b|\bpay\s*roll\b|\bregisters?\b"
            r"|\bcensus(?:es)?\b|\bestablishments?\b|\brolls?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "staff_return",
        re.compile(
            r"\bmorning\s+states?\b|\bpresent\s+for\s+duty\b|\beffectives?\b"
            r"|\border\s+of\s+battle\b|\bstaff\s+returns?\b",
            re.IGNORECASE,
        ),
    ),
)


def classify_regime(text: str | None, *, value: float | None = None) -> str:
    """Classify a report's surrounding text onto :data:`CLAIM_REGIMES`.

    Deterministic keyword matching only. It never falls back to a battle's
    era: a string with no marker is genuinely ambiguous about who is making
    the claim, and :func:`era_flag` is a separate, explicit signal the caller
    combines with this one rather than something this function should guess
    on its behalf.

    Calibrated against 382 real Wikipedia infobox strength strings (see
    ``tests/fixtures/gold/arsht/arsht_infobox.jsonl``): only a small minority
    carry an explicit provenance marker, so most reports classify as
    ``"unlabelled"``. That is the expected, correct outcome of a corpus where
    most infoboxes simply state a number without saying whose figure it is,
    not a sign the patterns need to be loosened until more of the corpus
    matches something.

    **A field naming two traditions at once is ambiguous, not modern.** Real
    infoboxes routinely carry both: Gaugamela's strength field reads
    ``"52,930-100,000 (modern estimates) 250,000-1,000,000 (ancient
    sources)"``, and Yarmouk's names modern, Arab and Roman figures in one
    line. Each such field yields several ``troop_reports`` rows that may all
    share it as their context. Returning the first match would label
    Herodotus' million ``modern_scholarly`` -- the anchor, pinned to exactly
    zero bias -- so no bias parameter would be free to absorb it and the
    side's estimate would be dragged up by an order of magnitude. That is the
    worst available direction for the error, so a multi-regime field with no
    way to tell which number is being asked about returns ``"unlabelled"``
    and falls back to the era signal.

    Passing ``value`` resolves the ambiguity properly: the regime whose marker
    sits nearest that number in the text wins.

    Args:
        text: The report's extracted context, or a whole infobox field. May
            be ``None`` or empty.
        value: The report's own reported figure, when the caller knows it.
            Used only to pick between markers in a multi-regime field.

    Returns:
        A member of
        :data:`pipeline.reconcilers.records.CLAIM_REGIMES`, defaulting to
        ``"unlabelled"``.
    """
    if not text:
        return "unlabelled"
    cleaned = _CITATION_REF.sub(" ", text)

    hits: list[tuple[str, int]] = []
    for regime, pattern in _REGIME_PATTERNS:
        match = pattern.search(cleaned)
        if match is not None:
            hits.append((regime, match.start()))

    if not hits:
        return "unlabelled"
    if len(hits) == 1:
        return hits[0][0]

    anchor = _value_position(cleaned, value)
    if anchor is None:
        # Several traditions named and nothing to say which one this report
        # came from. Deferring to the era signal is the safe error.
        return "unlabelled"
    return min(hits, key=lambda hit: abs(hit[1] - anchor))[0]



def _value_position(text: str, value: float | None) -> int | None:
    """Locate a reported figure inside its context text.

    Args:
        text: The context, already stripped of citation refs.
        value: The figure to find, or None.

    Returns:
        The character offset of the figure's first occurrence, or None when
        there is no value or it does not appear. Digit-group separators vary,
        so the search is done on the digits alone.
    """
    if value is None:
        return None
    digits = f"{int(round(value)):d}"
    for candidate in (digits, f"{int(round(value)):,d}"):
        found = text.find(candidate)
        if found != -1:
            return found
    # Figures are often written abbreviated ("1.04 million"); fall back to the
    # leading digits, which still localise the claim well enough to choose a
    # nearer marker.
    if len(digits) > 3:
        found = text.find(digits[:3])
        if found != -1:
            return found
    return None


def era_flag(year_astronomical: int | None, ancient_cutoff_year: int) -> float | None:
    """Say whether a battle counts as ancient, without guessing on no date.

    Args:
        year_astronomical: The battle's astronomical year (0 = 1 BC, -1 = 2
            BC, ...), or None when the battle is undated.
        ancient_cutoff_year: The astronomical year below which a battle
            counts as ancient, read from the model config.

    Returns:
        ``1.0`` if the year is before the cutoff, ``0.0`` if not, or ``None``
        when the year is unknown. ``None`` must never collapse to ``0.0``:
        ``(year_astronomical or 0) < ancient_cutoff_year`` would read an
        undated battle's year as 1 BC and silently declare it ancient, which
        is exactly backwards for the majority of undated battles, most of
        which are undated because they are obscure modern skirmishes, not
        because they are lost to antiquity.
    """
    if year_astronomical is None:
        return None
    return 1.0 if year_astronomical < ancient_cutoff_year else 0.0


def roundness(value: float) -> float:
    """Score how conventional a reported figure looks, on a 0..1 scale.

    Ancient and medieval troop figures cluster on traditional, rhetorically
    round values (an army of "100,000", a fleet of "1,000 ships") far more
    than a real headcount would by chance. A high roundness score is not
    evidence the number is wrong, only that it is more likely a coarsened,
    remembered, or rhetorical figure than a precise count -- which is why the
    reconcile model uses it to inflate that report's observation variance
    rather than to discard or correct the value.

    The score is the fraction of the number's digits that are trailing
    zeros, out of the maximum possible for its digit count (a leading digit
    can never be zero, so an ``n``-digit number has at most ``n - 1``
    trailing zeros). Equivalently, it is the complement of significant-figure
    density: ``1,000,000`` has one significant figure across seven digits and
    scores close to 1.0; ``52,930`` has four significant figures across five
    digits and scores low; ``86,400`` falls in between. The measure is
    monotone in trailing-zero count for a fixed digit count by construction,
    which is the only property asked of it -- the exact curve is a modelling
    choice, not a claim about a known-correct scale.

    Args:
        value: The reported value. Only its magnitude matters.

    Returns:
        A score in ``[0.0, 1.0]``. Non-finite or non-positive input and
        single-digit magnitudes score ``0.0``: there are no zeros to be
        trailing, so there is nothing for this measure to detect.
    """
    magnitude = abs(value)
    if magnitude != magnitude or magnitude in (float("inf"), float("-inf")):  # NaN check
        return 0.0
    integer_part = int(round(magnitude))
    digits = str(integer_part)
    n_digits = len(digits)
    if n_digits <= 1:
        return 0.0
    trailing_zeros = len(digits) - len(digits.rstrip("0"))
    return trailing_zeros / (n_digits - 1)


# Wikidata and DBpedia are structurally derived from Wikipedia -- both are
# built by scraping or mirroring infobox and article data, not by independent
# research -- so a report from either that repeats a Wikipedia figure is not
# a second opinion. Treating the two as independent evidence narrows the
# reconcile model's credible interval by roughly sqrt(N) exactly where the
# corpus has already been fooled once, by whichever editor wrote the
# original Wikipedia number.
_DERIVED_SOURCE_TYPES: Final[frozenset[str]] = frozenset({"wikidata", "dbpedia"})
_WIKIPEDIA_SOURCE_TYPES: Final[frozenset[str]] = frozenset({"wikipedia_infobox", "wikipedia_body"})


def _is_derived_wikipedia_pair(a: Report, b: Report) -> bool:
    """Whether one of a pair is a Wikipedia source and the other derived from it."""
    a_derived = a.source_type in _DERIVED_SOURCE_TYPES
    b_derived = b.source_type in _DERIVED_SOURCE_TYPES
    a_wiki = a.source_type in _WIKIPEDIA_SOURCE_TYPES
    b_wiki = b.source_type in _WIKIPEDIA_SOURCE_TYPES
    return (a_derived and b_wiki) or (b_derived and a_wiki)


def _values_agree(a: float, b: float, relative_tolerance: float) -> bool:
    """Whether two values are within tolerance of each other, scale-free."""
    denominator = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / denominator <= relative_tolerance


def _report_subgroup_key(report: Report) -> tuple[int, str, str]:
    """The (side, quantity, branch-or-casualty-type) group a report belongs to."""
    sub = report.branch if report.quantity == "troops" else report.casualty_type
    return (report.side_id, report.quantity, sub)


def assign_lineages(
    reports: Sequence[Report], *, relative_tolerance: float = 0.01
) -> dict[int, int]:
    """Group reports that repeat one claim rather than corroborating it independently.

    Two reports on the same ``(side_id, quantity, branch_or_casualty_type)``
    share a lineage when their values agree within `relative_tolerance` and
    at least one of them is a Wikidata or DBpedia report matching a Wikipedia
    infobox or body report -- see the module-level comment on
    :data:`_DERIVED_SOURCE_TYPES` for why that specific pairing, and only
    that one, counts as repetition rather than corroboration.

    Two genuinely independent sources (two ``peer_reviewed`` books, say) that
    happen to report the same value are deliberately *not* merged, even
    though their values agree: over-merging destroys real corroborating
    evidence exactly as badly as under-merging manufactures fake evidence,
    and this function has no way to tell independent agreement from
    repetition except the derived/Wikipedia pairing, so it does not guess.

    Args:
        reports: The reports to group. Order does not affect the grouping,
            only the numbering of lineage ids.
        relative_tolerance: How close two values must be, as a fraction of
            the larger magnitude, to count as the same claim.

    Returns:
        A mapping from ``report_id`` to a lineage id. Every report gets an
        id, including one with no matches, which gets a lineage of its own.
    """
    parent: dict[int, int] = {report.report_id: report.report_id for report in reports}

    def find(report_id: int) -> int:
        while parent[report_id] != report_id:
            parent[report_id] = parent[parent[report_id]]
            report_id = parent[report_id]
        return report_id

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            # Deterministic regardless of input order: the lower report_id
            # always becomes the root, so re-running on the same reports in
            # a different order cannot change which id a lineage settles on.
            higher, lower = max(root_a, root_b), min(root_a, root_b)
            parent[higher] = lower

    subgroups: dict[tuple[int, str, str], list[Report]] = defaultdict(list)
    for report in reports:
        subgroups[_report_subgroup_key(report)].append(report)

    for members in subgroups.values():
        for i, first in enumerate(members):
            for second in members[i + 1 :]:
                if _is_derived_wikipedia_pair(first, second) and _values_agree(
                    first.reported_value, second.reported_value, relative_tolerance
                ):
                    union(first.report_id, second.report_id)

    # Renumber roots to small, deterministic ids in first-appearance order,
    # rather than exposing arbitrary report_id values as lineage ids -- the
    # id itself carries no meaning, only which reports share one.
    lineage_of_root: dict[int, int] = {}
    lineage_of_report: dict[int, int] = {}
    next_lineage_id = 1
    for report in reports:
        root = find(report.report_id)
        if root not in lineage_of_root:
            lineage_of_root[root] = next_lineage_id
            next_lineage_id += 1
        lineage_of_report[report.report_id] = lineage_of_root[root]
    return lineage_of_report
