"""
The records every extractor in the extract stage produces.

Four extractors feed this stage -- the infobox parser, the LLM body pass,
the Wikidata mapper and the citation pass -- and the merger has to compare
what they said. That only works if they all speak in the same shapes, so
each one returns a :class:`SourceExtraction`: what this one source claims
about one battle, tagged with where the claim came from.

Nothing here decides anything. The types are deliberately permissive about
disagreement: two sources reporting different troop numbers produce two
:class:`TroopReport` rows, not one reconciled figure, because the reconcile
stage models source disagreement and cannot model reports it never sees.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Final

__all__ = [
    "BRANCHES",
    "CASUALTY_TYPES",
    "COMMAND_ROLES",
    "OUTCOME_LEVELS",
    "SCOPES",
    "BattleExtraction",
    "BattleFacts",
    "CasualtyReport",
    "CommanderMention",
    "Disagreement",
    "ExtractionFailure",
    "MissingField",
    "Provenance",
    "SideExtraction",
    "SourceExtraction",
    "coerce_branch",
    "coerce_casualty_type",
    "coerce_scope",
    "to_jsonable",
    "to_schema_command_role",
]

# Vocabularies. These mirror the enums in config/schema.sql and the enums in
# agents/extract.yaml's output schema; anything outside them is coerced
# rather than written, so a model inventing a value cannot break an INSERT.
SCOPES: Final[tuple[str, ...]] = (
    "engaged",
    "available",
    "theatre_strength",
    "on_paper",
    "unknown",
)

BRANCHES: Final[tuple[str, ...]] = (
    "total",
    "infantry",
    "cavalry",
    "artillery",
    "naval",
    "air",
    "armour",
    "irregular",
    "other",
)

CASUALTY_TYPES: Final[tuple[str, ...]] = (
    "killed",
    "wounded",
    "captured",
    "missing",
    "total",
)

COMMAND_ROLES: Final[tuple[str, ...]] = (
    "sovereign",
    "supreme_commander",
    "theatre_commander",
    "field_commander",
    "subordinate",
    "nominal",
    "unclear",
)

OUTCOME_LEVELS: Final[tuple[str, ...]] = (
    "decisive_victory",
    "victory",
    "pyrrhic_victory",
    "indecisive",
    "defeat",
    "decisive_defeat",
)

# agents/extract.yaml calls the residual role "unclear"; the command_role enum
# in the schema calls it "unknown". One translation point, here.
_ROLE_TO_SCHEMA: Final[dict[str, str]] = {"unclear": "unknown"}


def coerce_scope(value: str | None) -> str:
    """Map an extracted scope onto the vocabulary, defaulting to unknown.

    Args:
        value: A scope string from an extractor or an LLM response.

    Returns:
        A member of :data:`SCOPES`. Anything unrecognised becomes
        ``"unknown"`` rather than ``"engaged"``: an unreadable scope is not
        evidence that the number refers to troops on the field.
    """
    normalised = (value or "").strip().lower().replace(" ", "_")
    return normalised if normalised in SCOPES else "unknown"


def coerce_branch(value: str | None) -> str:
    """Map an extracted branch onto the vocabulary.

    Args:
        value: A branch string from an extractor or an LLM response.

    Returns:
        A member of :data:`BRANCHES`, defaulting to ``"total"``.
    """
    normalised = (value or "").strip().lower()
    if normalised in ("armor", "armored", "armoured", "tanks"):
        normalised = "armour"
    return normalised if normalised in BRANCHES else "total"


def coerce_casualty_type(value: str | None) -> str:
    """Map an extracted casualty type onto the vocabulary.

    Args:
        value: A casualty type from an extractor or an LLM response.

    Returns:
        A member of :data:`CASUALTY_TYPES`, defaulting to ``"total"``.
    """
    normalised = (value or "").strip().lower()
    return normalised if normalised in CASUALTY_TYPES else "total"


def to_schema_command_role(value: str | None) -> str:
    """Translate an extracted role into the ``command_role`` enum.

    Args:
        value: A role from the LLM's ``apparent_role`` field.

    Returns:
        A value the ``command_role`` enum accepts.
    """
    normalised = (value or "").strip().lower()
    if normalised not in COMMAND_ROLES:
        return "unknown"
    return _ROLE_TO_SCHEMA.get(normalised, normalised)


@dataclass(frozen=True)
class Provenance:
    """Where one claim came from.

    Attributes:
        source_type: A ``source_type`` enum value, e.g. ``wikipedia_infobox``.
        extraction_method: An ``extraction_method`` enum value, e.g.
            ``infobox_parser``.
        source_ref: A URL or raw-file path identifying the exact document.
        source_title: Human-readable title, used in prompts and review queues.
    """

    source_type: str
    extraction_method: str
    source_ref: str = ""
    source_title: str = ""


@dataclass
class TroopReport:
    """One source's statement of one force's size.

    ``scope`` is the field that matters downstream: it decides whether a
    number belongs in the force-ratio covariate at all. It is never inferred
    from convenience, and ``extracted_context`` keeps the wording so the
    judgement can be re-checked without re-crawling.
    """

    side_label: str
    reported_value: float
    provenance: Provenance
    branch: str = "total"
    scope: str = "unknown"
    is_estimate: bool = False
    is_upper_bound: bool = False
    is_lower_bound: bool = False
    extracted_context: str = ""
    page_or_section: str | None = None


@dataclass
class CasualtyReport:
    """One source's statement of one force's losses."""

    side_label: str
    reported_value: float
    provenance: Provenance
    casualty_type: str = "total"
    is_estimate: bool = False
    extracted_context: str = ""


@dataclass
class CommanderMention:
    """A named person described as commanding, pre entity resolution.

    The resolve stage turns these into ``generals`` and ``battle_commanders``
    rows. Until then the name is whatever the source wrote, because
    normalising it here would destroy the alias evidence resolve needs.
    """

    side_label: str
    name: str
    provenance: Provenance
    apparent_role: str = "unclear"
    role_evidence: str = ""
    note: str = ""
    mentions: int = 1


@dataclass
class BattleFacts:
    """Battle-level scalars as one source reports them."""

    name: str | None = None
    wikidata_id: str | None = None
    wikipedia_url: str | None = None
    date_start: str | None = None
    date_end: str | None = None
    date_precision: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    location_name: str | None = None
    battle_type: str | None = None
    terrain: list[str] = field(default_factory=list)
    fortified: bool | None = None
    weather: str | None = None
    victor: str | None = None
    outcome_level: str | None = None
    outcome_evidence: str | None = None
    part_of: list[str] = field(default_factory=list)


@dataclass
class SideExtraction:
    """One belligerent as one source describes it."""

    label: str
    aliases: list[str] = field(default_factory=list)
    polity: str | None = None
    outcome: str | None = None
    commanders: list[CommanderMention] = field(default_factory=list)
    troop_reports: list[TroopReport] = field(default_factory=list)
    casualty_reports: list[CasualtyReport] = field(default_factory=list)


@dataclass
class SourceExtraction:
    """Everything one source yielded about one battle."""

    provenance: Provenance
    facts: BattleFacts = field(default_factory=BattleFacts)
    sides: list[SideExtraction] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Disagreement:
    """Two or more sources giving different values for one scalar field.

    Recorded rather than resolved. The merger still has to write one value
    into ``battles``, but which sources dissented is evidence, and throwing
    it away would make the chosen value look more certain than it is.
    """

    field_name: str
    chosen: str
    chosen_source: str
    rejected: dict[str, str]


@dataclass(frozen=True)
class MissingField:
    """A field no source supplied, destined for ``missing_data_log``."""

    field_name: str
    side_label: str | None = None
    notes: str = ""


@dataclass(frozen=True)
class ExtractionFailure:
    """One extraction attempt that did not produce usable data.

    Per the project's error-handling rule these do not abort the batch. They
    flag the battle for review and travel with it so the review queue can say
    what actually went wrong.
    """

    battle_name: str
    source_ref: str
    status: str
    error: str
    passage_index: int | None = None


@dataclass
class BattleExtraction:
    """The merged view of one battle, ready to be written out."""

    slug: str
    name: str
    facts: BattleFacts = field(default_factory=BattleFacts)
    sides: list[SideExtraction] = field(default_factory=list)
    disagreements: list[Disagreement] = field(default_factory=list)
    missing: list[MissingField] = field(default_factory=list)
    failures: list[ExtractionFailure] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    n_sources: int = 0

    @property
    def needs_review(self) -> bool:
        """Whether a human should look at this battle before it is modelled."""
        return bool(self.failures) or bool(self.disagreements)

    @property
    def review_notes(self) -> str:
        """A one-line summary of why the battle is flagged, or an empty string."""
        reasons: list[str] = []
        if self.failures:
            statuses = sorted({f.status for f in self.failures})
            reasons.append(f"{len(self.failures)} extraction failure(s): {', '.join(statuses)}")
        if self.disagreements:
            fields = sorted({d.field_name for d in self.disagreements})
            reasons.append(f"source disagreement on {', '.join(fields)}")
        return "; ".join(reasons)


def to_jsonable(record: Any) -> dict[str, Any]:
    """Render a dataclass record as a JSON-serialisable mapping.

    Args:
        record: Any dataclass instance defined in this module.

    Returns:
        A nested dict of plain types, suitable for ``json.dumps``.
    """
    return asdict(record)
