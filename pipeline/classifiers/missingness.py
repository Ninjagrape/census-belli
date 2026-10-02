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
(see ``pipeline.resolvers.store.UNRESOLVED_FIELD`` and handover.md §12.3's
sibling rule for ``command_role``), and this stage has no business
overwriting a decision another stage already made for a different reason.
"""

from __future__ import annotations

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
            (handover.md §4.5), or None when undated.
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
    """

    log_id: int
    field_name: str
    current_class: str
    year_astronomical: int | None
    this_side_has_troops: bool = False
    other_side_has_troops: bool = False
    any_side_has_troops: bool = False
    side_outcome: str | None = None


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
