"""
Missingness classification: task B of ``agents/classify.yaml``.

Every ``missing_data_log`` row starts 'unclassified'. This module decides
whether a specific missing field should move to 'mcar', 'mar' or 'mnar', by
the heuristics ``agents/classify.yaml``'s ``missingness_prompt`` lays out --
this is the deterministic pass; the prompt itself is for whatever a human
reviewer or a later LLM pass wants to double check, not called from here.

The one rule that matters more than any heuristic: **a row that already
carries a class other than 'unclassified' is never touched.** Resolve writes
'mnar' directly onto ``commander_general_id`` rows for an unresolved mention
(see ``pipeline.resolvers.store.UNRESOLVED_FIELD`` and docs/handover-log.md §12.3's
sibling rule for ``command_role``), and this stage has no business
overwriting a decision another stage already made for a different reason.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from pipeline.classifiers.records import MissingnessDecision

__all__ = [
    "IMPUTE_FIELDS",
    "MissingnessInput",
    "classify_missingness",
]

# The only fields the impute stage consumes. A missing_data_log row for
# anything else (e.g. 'commander_general_id', which resolve owns) is out of
# scope here regardless of how confidently a heuristic might fire on it.
IMPUTE_FIELDS: Final[frozenset[str]] = frozenset(
    {"troop_total", "casualties", "battle_type", "terrain", "fortified"}
)

_TEMPLATE_FIELDS: Final[frozenset[str]] = frozenset({"battle_type", "terrain", "fortified"})

_ANCIENT_MEDIEVAL_CUTOFF: Final[int] = 1500
_MAR_ERA_START: Final[int] = 1700
_MAR_ERA_END: Final[int] = 1914

# First year of the post-1914 regime rule: wartime states from the First World
# War on had the means (censorship, ministries of information) to suppress or
# inflate their own figures.
_REGIME_ERA_START: Final[int] = 1914

# Why a regime's own figures go missing. Two different mechanisms, kept apart
# because impute and a reader of the log should not read them as one.
_SUPPRESSION: Final[str] = (
    "an authoritarian state that suppressed or distorted its own strength and loss "
    "figures; whether the number is published depends on the number"
)
_RECORDS_LOST: Final[str] = (
    "German forces in 1944-45, whose records were largely destroyed or never "
    "compiled in the collapse of the authoritarian state; the gap tracks how "
    "catastrophic the situation was, which the missing figure itself reflects"
)

# Authoritarian regimes whose own strength and loss figures were suppressed,
# distorted or lost, as (pattern, first year, last year, reason). The pattern is
# matched against the whole of the side's polity (see _normalise), never as a
# substring: "Coalition forces in Iraq" is not Iraq. Years are inclusive and
# whole years, because the input carries only a year; a regime that began
# mid-year is listed from that year. A curated list, not a classifier of
# regimes: a state is here only for the years it was one, so "Germany" counts in
# 1933-1945 and not in 1870. Extend it with a reason, not by guesswork.
_AUTHORITARIAN_STATES: Final[tuple[tuple[re.Pattern[str], int, int, str], ...]] = tuple(
    (re.compile(pattern, re.IGNORECASE), first, last, reason)
    for pattern, first, last, reason in (
        (
            r"(?:the )?(?:soviet union|ussr|soviet russia|union of soviet socialist republics)",
            1917,
            1991,
            _SUPPRESSION,
        ),
        (r"(?:nazi germany|third reich|german reich|germany)", 1933, 1943, _SUPPRESSION),
        (r"(?:nazi germany|third reich|german reich|germany)", 1944, 1945, _RECORDS_LOST),
        (r"(?:imperial japan|empire of japan|japan)", 1931, 1945, _SUPPRESSION),
        (r"(?:fascist italy|kingdom of italy|italy)", 1922, 1943, _SUPPRESSION),
        (
            r"(?:italian social republic|republic of sal[oò]|sal[oò] republic)",
            1943,
            1945,
            _SUPPRESSION,
        ),
        (r"(?:north korea|democratic people's republic of korea|dprk)", 1948, 2100, _SUPPRESSION),
        (r"(?:people's republic of china|prc|communist china)", 1949, 2100, _SUPPRESSION),
        (
            r"(?:chinese communist party|communist party of china|ccp|people's liberation army"
            r"|pla|chinese communists?|communists?|red army of china)",
            1927,
            1949,
            _SUPPRESSION,
        ),
        (r"(?:viet ?minh|democratic republic of vietnam)", 1945, 1954, _SUPPRESSION),
        (
            r"(?:north vietnam|vietnam|viet ?cong|democratic republic of vietnam"
            r"|socialist republic of vietnam)",
            1954,
            1979,
            _SUPPRESSION,
        ),
        (
            r"(?:francoist spain|nationalist spain|spanish state|francoist faction)",
            1936,
            1975,
            _SUPPRESSION,
        ),
        (r"(?:egypt|republic of egypt|united arab republic)", 1952, 2100, _SUPPRESSION),
        (r"(?:ba'?athist syria|syria|syrian arab republic)", 1963, 2100, _SUPPRESSION),
        (r"(?:ba'?athist iraq|iraq)", 1968, 2003, _SUPPRESSION),
        (r"(?:khmer rouge|democratic kampuchea)", 1975, 1979, _SUPPRESSION),
        (r"(?:iran|islamic republic of iran)", 1979, 2100, _SUPPRESSION),
        (r"(?:argentina|argentine junta)", 1976, 1983, _SUPPRESSION),
        (
            r"(?:serbia|fr yugoslavia|federal republic of yugoslavia|serbia and montenegro"
            r"|yugoslavia)",
            1989,
            2000,
            _SUPPRESSION,
        ),
        (r"(?:libya|libyan arab jamahiriya)", 1969, 2011, _SUPPRESSION),
    )
)


@dataclass(frozen=True)
class MissingnessInput:
    """What :func:`classify_missingness` needs to know about one missing field.

    Deliberately small: everything here is either already on the
    ``missing_data_log`` row or a fact about the row's battle/side that a
    stage runner can look up in one join, not a synopsis of the whole
    battle. None of it requires the LLM prompt's fuller context (war,
    region, "what IS known") -- those are for the prompt path, not this one.

    Attributes:
        log_id: The ``missing_data_log`` row this input describes.
        field_name: The missing field's name.
        current_class: The row's ``missingness_class`` right now. Anything
            other than 'unclassified' short-circuits to no decision.
        year_astronomical: The battle's year, astronomical numbering
            (docs/handover-log.md §4.5), or None when undated.
        this_side_has_troops: Whether *this* side already has a troop
            number (only meaningful for 'troop_total'/'casualties'; ignored
            for the template fields).
        other_side_has_troops: Whether the battle's other side has one.
        any_side_has_troops: Whether any side of the battle has one at all.
            Kept separate from the this/other pair because "nobody has any
            numbers" and "the other side does but this one doesn't" are
            different heuristics below.
        side_outcome: 'winner', 'loser', or None when the side's outcome
            relative to this battle is not known.
        side_polity: The side's ``polity``, or None. The post-1914 regime rule
            reads this alone.
        side_label: The side's ``side_label``, or None. Not consulted by the
            regime rule: labels are free text ("Coalition forces in Iraq").
    """

    log_id: int
    field_name: str
    current_class: str
    year_astronomical: int | None
    this_side_has_troops: bool = False
    other_side_has_troops: bool = False
    any_side_has_troops: bool = False
    side_outcome: str | None = None
    side_polity: str | None = None
    side_label: str | None = None


def _normalise(polity: str) -> str:
    """Fold a polity string for anchored matching.

    Args:
        polity: A ``battle_sides.polity`` value.

    Returns:
        The string with curly apostrophes straightened, surrounding space and a
        trailing parenthetical dropped, and case folded.
    """
    straight = polity.replace("’", "'").replace("‘", "'").replace("ʼ", "'")
    return re.sub(r"\s*\([^)]*\)\s*$", "", straight).strip().casefold()


def _authoritarian_state(record: MissingnessInput) -> tuple[str, str] | None:
    """Find the authoritarian regime a side belonged to, if any.

    Args:
        record: The missing field's inputs.

    Returns:
        ``(polity, reason)`` when the side's whole polity matches a curated regime
        that held power in the battle's year (after 1914), else None.
    """
    year = record.year_astronomical
    if year is None or year < _REGIME_ERA_START or not record.side_polity:
        return None
    polity = _normalise(record.side_polity)
    for pattern, first_year, last_year, reason in _AUTHORITARIAN_STATES:
        if first_year <= year <= last_year and pattern.fullmatch(polity) is not None:
            return record.side_polity.strip(), reason
    return None


def classify_missingness(record: MissingnessInput) -> MissingnessDecision | None:
    """Apply the spec's missingness heuristics to one missing field.

    Args:
        record: What is known about the missing field and its battle.

    Returns:
        A decision, or None when the row should stay 'unclassified' --
        either because it already carries a different class, is outside
        this stage's field vocabulary, is undated (every heuristic below
        needs an era), or simply matches none of the heuristics. None is a
        legitimate, common outcome: the spec's heuristics are a short list
        of patterns strong enough to act on, not an exhaustive decision
        procedure, and a genuinely ambiguous row is supposed to wait for a
        human or an LLM pass rather than be forced into a class.
    """
    if record.current_class != "unclassified":
        return None
    if record.field_name not in IMPUTE_FIELDS:
        return None
    if record.year_astronomical is None:
        return None

    if record.field_name in _TEMPLATE_FIELDS:
        # MAR, not the MCAR the spec's "template inconsistency" heuristic would
        # suggest. A missing terrain or battle type plausibly tracks how
        # obscure and how thinly documented a battle is, and after classify
        # has run a battle_type still 'unknown' means no article was read --
        # an observed fact, which is MAR by definition. MCAR is the special
        # case of MAR, so imputing under MAR stays correct if the gap really
        # is random; imputing under MCAR when it is not lets impute ignore the
        # covariates that explain it. The weaker assumption is the safe one.
        return MissingnessDecision(
            log_id=record.log_id,
            missingness_class="mar",
            reason=(
                f"{record.field_name} missing; plausibly tracks documentation level "
                "and article availability, both observed"
            ),
        )

    # From here on, field_name is 'troop_total' or 'casualties'.

    regime = _authoritarian_state(record)
    if regime is not None:
        polity, why = regime
        return MissingnessDecision(
            log_id=record.log_id,
            missingness_class="mnar",
            reason=f"post-1914 figure missing for a side of {polity!r}, {why}",
        )

    if record.field_name == "casualties" and record.this_side_has_troops:
        # A different mechanism from a missing troop total. The side's strength
        # is documented, so the casualty gap is not thin documentation: it is
        # the loss figure itself that went unrecorded. For a defeated side that
        # is plausibly because the losses were too heavy to count (an army
        # destroyed in the field or on the retreat); for a victor it follows
        # observed variables (era, size, how the battle is written up).
        if record.side_outcome == "loser":
            return MissingnessDecision(
                log_id=record.log_id,
                missingness_class="mnar",
                reason=(
                    "casualties missing although this side's strength is known, and the "
                    "side lost; heavy losses are the figures most often left unrecorded"
                ),
            )
        return MissingnessDecision(
            log_id=record.log_id,
            missingness_class="mar",
            reason=(
                "casualties missing although this side's strength is known; the gap "
                "tracks era and how thoroughly the battle is written up, both observed"
            ),
        )

    if not record.any_side_has_troops and record.year_astronomical < _ANCIENT_MEDIEVAL_CUTOFF:
        return MissingnessDecision(
            log_id=record.log_id,
            missingness_class="mnar",
            reason=(
                "ancient/medieval battle (year < 1500) with no troop data on either "
                "side; records more likely lost because the battle was obscure, and "
                "obscurity correlates with the scale the missing number would show"
            ),
        )

    if (
        record.side_outcome == "loser"
        and not record.this_side_has_troops
        and record.other_side_has_troops
    ):
        return MissingnessDecision(
            log_id=record.log_id,
            missingness_class="mnar",
            reason=(
                "only the winning side's numbers are known; the losing side's "
                "records are more likely suppressed or destroyed than merely absent"
            ),
        )

    if (
        _MAR_ERA_START <= record.year_astronomical <= _MAR_ERA_END
        and not record.this_side_has_troops
        and record.other_side_has_troops
    ):
        return MissingnessDecision(
            log_id=record.log_id,
            missingness_class="mar",
            reason=(
                "1700-1914 battle missing this side's numbers while the other "
                "side's are known; plausibly explained by which side kept better "
                "records (an observed, era-linked variable), not by the missing "
                "value itself"
            ),
        )

    return None
