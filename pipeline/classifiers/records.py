"""
The records the classify stage passes between its steps.

Classify does two unrelated jobs on the same tables (see ``agents/classify.yaml``):
it turns a side's already-extracted commander mentions into a command
hierarchy (role, rank, who-reports-to-whom, attribution weight), and it
labels why a value the pipeline never got is missing. Neither job needs a
database open to be worked out -- both are a function of what was already
extracted -- so every record here is a plain, database-free value the pure
modules (:mod:`pipeline.classifiers.roles`, :mod:`pipeline.classifiers.battle_type`,
:mod:`pipeline.classifiers.missingness`) can be built, transformed and
tested without one.

The enum tuples mirror ``config/schema.sql`` exactly and are declared once
here so a typo in a hand-written enum literal fails in ``__post_init__``
rather than at INSERT time three modules away.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Final

__all__ = [
    "APPARENT_ROLES",
    "BATTLE_TYPES",
    "COMMAND_ROLES",
    "MISSINGNESS_CLASSES",
    "BattleTypeDecision",
    "ClassifyCounts",
    "CommanderRow",
    "MissingnessDecision",
    "RoleDecision",
    "SideGroup",
    "to_jsonable",
]

# battle_commanders.command_role / RoleDecision.command_role. Matches
# config/schema.sql's command_role enum exactly, including 'unknown' as the
# not-yet-classified value.
COMMAND_ROLES: Final[tuple[str, ...]] = (
    "sovereign",
    "supreme_commander",
    "theatre_commander",
    "field_commander",
    "subordinate",
    "nominal",
    "unknown",
)

# What the extract stage's LLM wrote to CommanderMention.apparent_role
# (pipeline/extractors/records.py's own COMMAND_ROLES, confusingly -- that
# name there means "apparent role vocabulary"). Same six roles as
# COMMAND_ROLES above, but 'unclear' where the schema enum says 'unknown';
# extract's own to_schema_command_role() is what maps one onto the other.
# CommanderRow.apparent_role is validated against this, separate vocabulary,
# because it is extraction's guess, not classify's decision.
APPARENT_ROLES: Final[tuple[str, ...]] = (
    "sovereign",
    "supreme_commander",
    "theatre_commander",
    "field_commander",
    "subordinate",
    "nominal",
    "unclear",
)

# battles.battle_type / BattleTypeDecision.battle_type.
BATTLE_TYPES: Final[tuple[str, ...]] = (
    "field",
    "siege_offensive",
    "siege_defensive",
    "naval",
    "aerial",
    "combined",
    "amphibious",
    "guerrilla",
    "unknown",
)

# missing_data_log.missingness_class / MissingnessDecision.missingness_class.
MISSINGNESS_CLASSES: Final[tuple[str, ...]] = (
    "observed",
    "mcar",
    "mar",
    "mnar",
    "unclassified",
)


def _check_enum(value: str, vocabulary: tuple[str, ...], field_name: str) -> None:
    """Raise if a field's value is not one of its declared vocabulary.

    Args:
        value: The field's value.
        vocabulary: The tuple it must belong to.
        field_name: Name reported in the error, for a fast diagnosis.

    Raises:
        ValueError: If ``value`` is not in ``vocabulary``.
    """
    if value not in vocabulary:
        raise ValueError(f"{field_name}={value!r} is not one of {vocabulary}")


@dataclass(frozen=True)
class CommanderRow:
    """One commander on one side of one battle, as classify reads it in.

    This is what a ``battle_commanders`` row (joined to ``generals`` for the
    name) looks like before classify has touched it. ``command_role`` is
    whatever resolve already wrote -- usually 'unknown', but see
    handover.md §12.3: a re-run must not clobber a role classify already set.

    Attributes:
        bc_id: The ``battle_commanders`` primary key.
        battle_id: The battle this row belongs to.
        side_id: The side this row belongs to.
        general_id: The resolved commander.
        name: The commander's canonical name, for prompts and matching.
        command_role: The role already on the row (often 'unknown').
        role_evidence: Extracted text supporting a role, if any.
        apparent_role: What extraction guessed before resolution, kept
            distinct from ``command_role`` because it uses a different
            vocabulary (``APPARENT_ROLES``, extract's own guess) and because
            collapsing the two would lose the evidence that classify itself
            has not yet decided.
        listing_order: Position in the infobox's commander list, 0 first.
            The only ordering signal available with no article evidence;
            see roles.py for why it stands in for seniority.
        attribution_method: How the row's current weight was decided. A side
            whose rows all say ``llm`` is left alone on a re-run: its roles
            look internally consistent, and without this the fixed weight
            table would quietly overwrite the model's reading of the article.
    """

    bc_id: int
    battle_id: int
    side_id: int
    general_id: int
    name: str
    command_role: str = "unknown"
    role_evidence: str = ""
    apparent_role: str = "unclear"
    listing_order: int = 0
    attribution_method: str = "equal"

    def __post_init__(self) -> None:
        _check_enum(self.command_role, COMMAND_ROLES, "command_role")
        _check_enum(self.apparent_role, APPARENT_ROLES, "apparent_role")


@dataclass(frozen=True)
class SideGroup:
    """One side of one battle, with every commander classify must rank.

    Attributes:
        battle_id: The battle this side belongs to.
        side_id: The side's primary key.
        side_label: The side's label, e.g. "Roman Republic (Octavian)".
        battle_name: The battle's name, for prompts.
        year_astronomical: The battle's year in astronomical numbering
            (negative for BC, matching handover.md §4.5), or None when
            undated.
        commanders: Every commander on this side, in listing order.
    """

    battle_id: int
    side_id: int
    side_label: str
    battle_name: str
    year_astronomical: int | None
    commanders: tuple[CommanderRow, ...] = ()


@dataclass(frozen=True)
class RoleDecision:
    """What classify concluded about one commander's role and weight.

    Attributes:
        bc_id: The ``battle_commanders`` row this decision updates.
        command_role: A member of ``COMMAND_ROLES``.
        hierarchy_rank: 0 for the top tactical commander on the side, 1 for
            a direct report, and so on.
        reports_to_bc_id: Another commander's ``bc_id`` on the *same* side,
            or None. Never set across sides -- see roles.py.
        attribution_weight: 0..1, this commander's share of the side's
            credit. Every side's weights are built to sum to 1.0 (see the
            quality gate in agents/classify.yaml).
        attribution_method: How the decision was reached: 'rule_single',
            'rule_roles', 'default_split', or 'llm'.
        confidence: 0..1.
        needs_review: True when a human should look at this row before it
            is trusted -- every 'default_split' decision, plus anything the
            LLM step could not resolve cleanly.
        reason: One line explaining the decision, for the review queue and
            for ``role_evidence``.
    """

    bc_id: int
    command_role: str
    hierarchy_rank: int
    attribution_weight: float
    attribution_method: str
    confidence: float
    needs_review: bool
    reason: str = ""
    reports_to_bc_id: int | None = None

    def __post_init__(self) -> None:
        _check_enum(self.command_role, COMMAND_ROLES, "command_role")


@dataclass(frozen=True)
class BattleTypeDecision:
    """What classify concluded about one battle's type.

    Attributes:
        battle_id: The battle this decision is about.
        battle_type: A member of ``BATTLE_TYPES``.
        fortified: Whether the engagement involved fortifications, or None
            when there is no basis to say. True by convention for every
            ``siege_offensive`` -- see battle_type.py for why the enum
            gives siege only one, attacker-framed value.
        evidence: What specifically drove the decision (a category name, a
            matched vocabulary term, or "no signal found").
        confidence: 0..1.
    """

    battle_id: int
    battle_type: str
    evidence: str
    confidence: float
    fortified: bool | None = None

    def __post_init__(self) -> None:
        _check_enum(self.battle_type, BATTLE_TYPES, "battle_type")


@dataclass(frozen=True)
class MissingnessDecision:
    """What classify concluded about why one field is missing.

    Attributes:
        log_id: The ``missing_data_log`` row this decision updates.
        missingness_class: A member of ``MISSINGNESS_CLASSES``.
        reason: Which heuristic fired, in one line, for the imputation
            stage's audit trail.
    """

    log_id: int
    missingness_class: str
    reason: str = ""

    def __post_init__(self) -> None:
        _check_enum(self.missingness_class, MISSINGNESS_CLASSES, "missingness_class")


class ClassifyCounts:
    """Row and decision counts for one stage run, for the summary log.

    Mutable by design, like :class:`pipeline.resolvers.records.ResolveCounts`:
    this is a counter a stage runner increments as it goes, not a value
    rebuilt from scratch each time.
    """

    def __init__(self) -> None:
        self.sides_processed = 0
        self.single_commander = 0
        self.rule_roles = 0
        self.default_split = 0
        self.llm_classified = 0
        self.llm_calls = 0
        self.needs_review = 0
        self.battles_typed = 0
        self.battle_type_naval = 0
        self.battle_type_siege = 0
        self.battle_type_aerial = 0
        self.battle_type_amphibious = 0
        self.battle_type_field = 0
        self.battle_type_unknown = 0
        self.missing_fields_seen = 0
        self.missingness_classified = 0
        self.missingness_left_unclassified = 0

    def as_dict(self) -> dict[str, int]:
        """Render the counts for a structlog call.

        Returns:
            A flat mapping of counter name to value.
        """
        return {
            "sides_processed": self.sides_processed,
            "single_commander": self.single_commander,
            "rule_roles": self.rule_roles,
            "default_split": self.default_split,
            "llm_classified": self.llm_classified,
            "llm_calls": self.llm_calls,
            "needs_review": self.needs_review,
            "battles_typed": self.battles_typed,
            "battle_type_naval": self.battle_type_naval,
            "battle_type_siege": self.battle_type_siege,
            "battle_type_aerial": self.battle_type_aerial,
            "battle_type_amphibious": self.battle_type_amphibious,
            "battle_type_field": self.battle_type_field,
            "battle_type_unknown": self.battle_type_unknown,
            "missing_fields_seen": self.missing_fields_seen,
            "missingness_classified": self.missingness_classified,
            "missingness_left_unclassified": self.missingness_left_unclassified,
        }


def to_jsonable(record: Any) -> dict[str, Any]:
    """Render a dataclass record as a JSON-serialisable mapping.

    Args:
        record: Any dataclass instance defined in this module.

    Returns:
        A nested dict of plain types, suitable for ``json.dumps``.
    """
    return asdict(record)
