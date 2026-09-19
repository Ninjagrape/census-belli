"""
The records the resolve stage passes between its steps.

The stage turns many surface mentions into few canonical identities, and
every step in that funnel needs to say the same three things: what was
written in the source, which Wikidata entity it might be, and how confident
the decision was. Those are :class:`Mention`, :class:`Candidate` and
:class:`Decision`.

Nothing here decides anything. In particular a :class:`Decision` carrying
``status="new"`` is not a claim that the person is absent from Wikidata, only
that this corpus found no candidate for them; the distinction is what the
resolution log exists to preserve.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Final, Literal

__all__ = [
    "METHODS",
    "BattleContext",
    "Candidate",
    "Decision",
    "Identity",
    "Mention",
    "MentionGroup",
    "ResolveCounts",
    "to_jsonable",
]

# How a decision was reached. Recorded on every resolution-log row and, for
# linked mentions, folded into the confidence written to battle_commanders.
METHODS: Final[tuple[str, ...]] = (
    "exact_wikidata",  # a candidate label or alias matched a mention key exactly
    "fuzzy_contextual",  # rapidfuzz above threshold, lifespan-compatible, unambiguous
    "corpus_fuzzy",  # matched an identity already resolved from this corpus
    "llm_disambiguation",  # the deterministic steps left it ambiguous
    "new_entity",  # no viable candidate; a corpus-local identity
    "placeholder",  # the mention names nobody ("Unknown", "various")
    "llm_failed",  # the disambiguation call did not return usable data
)

DecisionStatus = Literal["linked", "new", "ambiguous", "unresolved"]


@dataclass(frozen=True)
class Candidate:
    """One Wikidata entity that a commander mention might refer to.

    Attributes:
        qid: The item id, e.g. ``"Q517"``.
        label: The English label.
        description: The English description, shown to the LLM because it is
            usually the line that separates a general from his grandson.
        aliases: Other names the entity is known by.
        birth_year: Astronomical year of birth, or None when unknown.
        death_year: Astronomical year of death, or None when unknown.
        country: A country or polity label, used as a tiebreak only.
        occupations: Occupation labels, used to prefer military candidates.
        wikipedia_url: English Wikipedia URL, when the entity has one.
    """

    qid: str
    label: str
    description: str = ""
    aliases: tuple[str, ...] = ()
    birth_year: int | None = None
    death_year: int | None = None
    country: str = ""
    occupations: tuple[str, ...] = ()
    wikipedia_url: str = ""

    @property
    def is_military(self) -> bool:
        """Whether any occupation label looks military.

        Returns:
            True if an occupation mentions a military calling. Used as a
            tiebreak between otherwise equal candidates, never as a filter:
            sovereigns who commanded in person rarely carry a military
            occupation on Wikidata.
        """
        haystack = " ".join(self.occupations).lower()
        return any(
            token in haystack
            for token in ("military", "officer", "general", "admiral", "condottiero", "samurai")
        )


@dataclass(frozen=True)
class BattleContext:
    """What resolve knows about the battle a mention was made in.

    Attributes:
        slug: The battle's stable identifier in ``data/raw``.
        name: The battle's name.
        year: Astronomical year of the battle, or None when undated. This is
            the gate that stops a mention of "Hannibal" in 1811 linking to
            Hannibal Barca, so an undated battle is a genuinely weaker match.
        war: The war or campaign the battle was part of, when known.
        date_text: The battle's date as stored, for the LLM prompt.
    """

    slug: str
    name: str
    year: int | None = None
    war: str = ""
    date_text: str = ""


@dataclass(frozen=True)
class Mention:
    """One commander named by one source in one battle.

    This is a row of ``data/processed/commanders_raw.jsonl`` as the extract
    stage wrote it, with its battle context attached.
    """

    battle_slug: str
    battle_name: str
    side_label: str
    name: str
    apparent_role: str = "unclear"
    role_evidence: str = ""
    source_type: str = "wikipedia_infobox"
    extraction_method: str = "infobox_parser"
    source_ref: str = ""
    source_title: str = ""
    mentions: int = 1
    polity: str = ""
    context: BattleContext | None = None


@dataclass
class MentionGroup:
    """Every mention sharing one normalised name, resolved as a unit.

    Grouping first is what makes the stage affordable: a commander named in
    forty battles is one Wikidata lookup and at most one LLM call, not forty.

    Attributes:
        key: The primary normalised key the group was built on.
        keys: Every key any member mention generates, for matching.
        display_name: The most frequent surface form, used as the canonical
            name when no Wikidata label is adopted.
        surface_forms: Every distinct surface form seen, which become aliases.
        mentions: The member mentions.
        confidence: The confidence of the decision that resolved this group.
            Held here rather than on the identity because two groups can
            merge into one person on very different evidence, and a
            ``battle_commanders`` row should carry the confidence of the
            decision that put it there, not of its strongest sibling.
        method: How that decision was reached.
    """

    key: str
    keys: tuple[str, ...]
    display_name: str
    surface_forms: list[str] = field(default_factory=list)
    mentions: list[Mention] = field(default_factory=list)
    confidence: float = 0.0
    method: str = "new_entity"

    @property
    def years(self) -> list[int]:
        """Astronomical years of the battles this group was mentioned in.

        Returns:
            Every known year, unsorted and with duplicates, as the lifespan
            gate and the ``years_active`` range both want the spread.
        """
        return [m.context.year for m in self.mentions if m.context and m.context.year is not None]

    @property
    def polities(self) -> set[str]:
        """The polities this group's mentions fought for, lowercased."""
        return {m.polity.strip().lower() for m in self.mentions if m.polity.strip()}


@dataclass(frozen=True)
class Decision:
    """What the stage concluded about one :class:`MentionGroup`.

    Attributes:
        status: ``linked`` to a Wikidata entity, ``new`` for a corpus-local
            identity, ``ambiguous`` for a group awaiting the LLM step, or
            ``unresolved`` for one that gets no ``battle_commanders`` row.
        canonical_name: The name to store on ``generals``.
        method: A member of :data:`METHODS`.
        qid: The matched entity, when linked.
        confidence: 0..1, written to ``battle_commanders.confidence``.
        score: The fuzzy score behind the decision, when there was one.
        runner_up: The next-best candidate, recorded so a later review can
            see what the match was chosen over.
        reasoning: Why, in one line. From the LLM when it decided.
        candidates_considered: How many candidates survived the context gate.
        candidate: The matched entity in full, when linked.
    """

    status: DecisionStatus
    canonical_name: str
    method: str
    qid: str | None = None
    confidence: float = 0.0
    score: float | None = None
    runner_up: str | None = None
    reasoning: str = ""
    candidates_considered: int = 0
    candidate: Candidate | None = None


@dataclass
class Identity:
    """One canonical person, and every mention that resolved to them.

    Two groups that link to the same Q-id become one identity; that merge is
    the whole point of the stage, and it is also where alias evidence
    accumulates.
    """

    key: str
    canonical_name: str
    qid: str | None = None
    wikipedia_url: str = ""
    nationality: str = ""
    confidence: float = 0.0
    method: str = "new_entity"
    aliases: list[str] = field(default_factory=list)
    groups: list[MentionGroup] = field(default_factory=list)
    general_id: int | None = None

    @property
    def mentions(self) -> list[Mention]:
        """Every mention resolving to this identity."""
        return [m for group in self.groups for m in group.mentions]

    @property
    def years(self) -> list[int]:
        """Astronomical years of every battle this identity appears in."""
        return [year for group in self.groups for year in group.years]


class ResolveCounts:
    """Row and decision counts for one stage run, for the summary log."""

    def __init__(self) -> None:
        self.mentions = 0
        self.groups = 0
        self.identities = 0
        self.linked = 0
        self.new_entities = 0
        self.unresolved = 0
        self.llm_calls = 0
        self.generals_written = 0
        self.aliases_written = 0
        self.commanders_written = 0
        self.missing_rows = 0
        self.battles_not_in_db = 0
        self.sides_not_in_db = 0
        self.duplicate_battles = 0

    def as_dict(self) -> dict[str, int]:
        """Render the counts for a structlog call.

        Returns:
            A flat mapping of counter name to value.
        """
        return {
            "mentions": self.mentions,
            "groups": self.groups,
            "identities": self.identities,
            "linked": self.linked,
            "new_entities": self.new_entities,
            "unresolved": self.unresolved,
            "llm_calls": self.llm_calls,
            "generals_written": self.generals_written,
            "aliases_written": self.aliases_written,
            "commanders_written": self.commanders_written,
            "missing_rows": self.missing_rows,
            "battles_not_in_db": self.battles_not_in_db,
            "sides_not_in_db": self.sides_not_in_db,
            "duplicate_battles": self.duplicate_battles,
        }


def to_jsonable(record: Any) -> dict[str, Any]:
    """Render a dataclass record as a JSON-serialisable mapping.

    Args:
        record: Any dataclass instance defined in this module.

    Returns:
        A nested dict of plain types, suitable for ``json.dumps``.
    """
    return asdict(record)
