"""
The targeted outcome request: who won, for battles the merge could not decide.

The deterministic path (infobox result text, demonym matching, the coalition
rule) decides most battles. A battle left with two or more sides and no
outcome on any of them is asked about once, with a short and bounded prompt:
the infobox result line verbatim, each side as the infobox lists it, and the
start of the article. The model only names the winner. How decisive the win
was is not its call; that is read from the result text, the way the
deterministic path reads it, so the response variable's gradation never comes
from an LLM's impression.

Requests are cached by hash in ``llm_calls`` like every other extract call,
and an uncached one is deferred to its own batch file, so outcomes can be
answered without touching the passage requests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import structlog

from pipeline.extractors.article import clean_article_text, render_template, split_sections
from pipeline.extractors.infobox import parse_result
from pipeline.extractors.merger import apply_winner
from pipeline.extractors.records import (
    DEFERRED_STATUS,
    BattleExtraction,
    ExtractionFailure,
    SideExtraction,
)
from pipeline.llm import CallStatus, LLMService

__all__ = [
    "OUTCOME_LEAD_MAX_CHARS",
    "OUTCOME_METHOD",
    "OUTCOME_STAGE",
    "OutcomePass",
    "apply_outcome_answer",
    "lead_excerpt",
    "needs_outcome_request",
    "render_outcome_prompt",
    "resolve_outcome",
]

logger = structlog.get_logger()

# The service's stage name: it names the batch file (extract_outcome_<ts>.jsonl)
# and the ``llm_calls.stage`` of every answer, which keeps these calls apart
# from the passage requests without a schema change.
OUTCOME_STAGE: Final[str] = "extract_outcome"
# Recorded on BattleExtraction.outcome_method for outcomes this path wrote.
OUTCOME_METHOD: Final[str] = "llm_outcome"

# How much of the article's lead goes into the prompt. A battle article's
# opening paragraphs carry the result; more text costs tokens and invites the
# model to answer from the war's wider story rather than this battle.
OUTCOME_LEAD_MAX_CHARS: Final[int] = 2500
# Infobox editors list the principal commander first, so the rest add length
# without helping to tell the sides apart.
_MAX_COMMANDERS_PER_SIDE: Final[int] = 6
_MAX_BELLIGERENTS_PER_SIDE: Final[int] = 12

_WINNER_INDECISIVE: Final[str] = "indecisive"
_WINNER_UNCLEAR: Final[str] = "unclear"
_GRADED_LEVELS: Final[frozenset[str]] = frozenset(
    {"decisive_victory", "victory", "pyrrhic_victory"}
)


@dataclass(frozen=True)
class OutcomePass:
    """Everything the outcome request needs, read from the agent spec.

    Attributes:
        service: A service for the outcome model, stage ``extract_outcome``.
        system: ``outcome_prompt.system``.
        template: ``outcome_prompt.user_template``.
        schema: ``outcome_prompt.output_schema``.
        max_lead_chars: Cap on the article excerpt.
    """

    service: LLMService
    system: str
    template: str
    schema: dict[str, Any]
    max_lead_chars: int = OUTCOME_LEAD_MAX_CHARS


def needs_outcome_request(battle: BattleExtraction) -> bool:
    """Whether the deterministic path left this battle without a winner.

    Args:
        battle: The merged battle.

    Returns:
        True when the battle has at least two sides and none of them has an
        outcome. A battle the sources call indecisive already carries an
        outcome on every side, so it is never asked about.
    """
    if len(battle.sides) < 2:
        return False
    if battle.facts.outcome_level == _WINNER_INDECISIVE:
        return False
    return all(side.outcome is None for side in battle.sides)


def lead_excerpt(raw: str, *, is_html: bool | None = None, max_chars: int) -> str:
    """Cut the opening of an article down to the prompt's character budget.

    Args:
        raw: Article source, wikitext or HTML.
        is_html: Force the input format; None sniffs it.
        max_chars: Maximum characters returned.

    Returns:
        The text before the first heading, or the first section when the
        article has no lead, truncated at ``max_chars``. Empty when the
        article has no usable prose.
    """
    sections = split_sections(clean_article_text(raw, is_html=is_html))
    if not sections:
        return ""
    return sections[0][1][:max_chars].rstrip()


def _side_block(index: int, side: SideExtraction) -> str:
    """Render one side the way the infobox lists it.

    Args:
        index: 1-based position of the side.
        side: The merged side.

    Returns:
        A short block naming the label, belligerents and commanders.
    """
    belligerents = [side.label, *side.aliases][:_MAX_BELLIGERENTS_PER_SIDE]
    # Only the infobox carries an order; a commander from another source has
    # none and goes after those that do, in name order so the prompt is stable.
    commanders = sorted(
        side.commanders,
        key=lambda c: (c.infobox_order is None, c.infobox_order or 0, c.name),
    )[:_MAX_COMMANDERS_PER_SIDE]
    return "\n".join(
        [
            f"Side {index}: {side.label}",
            f"  Belligerents: {'; '.join(belligerents)}",
            f"  Commanders: {'; '.join(c.name for c in commanders) or 'none listed'}",
        ]
    )


def render_outcome_prompt(template: str, battle: BattleExtraction, lead: str, max_lead: int) -> str:
    """Fill the outcome user template for one battle.

    Args:
        template: ``outcome_prompt.user_template``.
        battle: The merged battle.
        lead: The article excerpt from :func:`lead_excerpt`.
        max_lead: The character cap the excerpt was cut to.

    Returns:
        The prompt. Everything in it comes from ordered lists, so the same
        battle always renders the same bytes and hits the cache.
    """
    sides = "\n".join(_side_block(i, side) for i, side in enumerate(battle.sides, start=1))
    return render_template(
        template,
        {
            "battle_name": battle.name,
            "infobox_result": battle.facts.outcome_evidence or "(none)",
            "sides": sides,
            "lead_chars": str(max_lead),
            "lead_excerpt": lead or "(no article text)",
        },
    )


def _level_for(battle: BattleExtraction) -> str:
    """Pick the winner's outcome level without asking the model.

    Args:
        battle: The merged battle.

    Returns:
        The level the sources already gave, else the one the result text
        grades, else a plain victory.
    """
    if battle.facts.outcome_level in _GRADED_LEVELS:
        return str(battle.facts.outcome_level)
    _victor, level = parse_result(battle.facts.outcome_evidence or "")
    return level if level in _GRADED_LEVELS and level is not None else "victory"


def apply_outcome_answer(battle: BattleExtraction, data: dict[str, Any]) -> bool:
    """Write an outcome answer onto the battle's sides.

    Args:
        battle: The merged battle, modified in place.
        data: The model's answer: ``winner`` and ``evidence``.

    Returns:
        True when outcomes were written. A winner that is "unclear", or not
        one of the side labels exactly, writes nothing.
    """
    winner = str(data.get("winner") or "").strip()
    evidence = str(data.get("evidence") or "").strip()

    if winner.lower() == _WINNER_INDECISIVE:
        for side in battle.sides:
            side.outcome = _WINNER_INDECISIVE
        if battle.facts.outcome_level is None:
            battle.facts.outcome_level = _WINNER_INDECISIVE
    else:
        chosen = next((s for s in battle.sides if s.label == winner), None)
        if chosen is None:
            if winner.lower() != _WINNER_UNCLEAR:
                logger.warning("outcome_winner_not_a_side", battle=battle.name, winner=winner)
            return False
        level = _level_for(battle)
        apply_winner(battle.sides, chosen, level)
        if battle.facts.outcome_level is None:
            battle.facts.outcome_level = level

    battle.outcome_method = OUTCOME_METHOD
    battle.notes.append(
        f"outcome decided by the {OUTCOME_STAGE} request: {winner!r} ({evidence!r})"
    )
    return True


def _drop_outcome_gaps(battle: BattleExtraction) -> None:
    """Withdraw the outcome rows from a battle's missing-data list.

    Args:
        battle: The battle, modified in place.
    """
    battle.missing = [
        m
        for m in battle.missing
        if not (m.field_name == "outcome" and m.side_label is not None)
        and not (m.field_name == "outcome_level" and battle.facts.outcome_level is not None)
    ]


def resolve_outcome(
    battle: BattleExtraction,
    outcome_pass: OutcomePass,
    *,
    lead: str,
    battle_id: int | None = None,
) -> str:
    """Ask the outcome question for a battle that needs it and apply the answer.

    A deferred request is an answer that has not arrived, not a field no
    source reported, so the battle's outcome gaps are withdrawn from the
    missing-data list until it does, as for a deferred passage.

    Args:
        battle: The merged battle, modified in place.
        outcome_pass: The prompt, schema and service for the pass.
        lead: The article excerpt, already cut to the pass's cap.
        battle_id: Database id recorded on the audit row, when known.

    Returns:
        ``skipped`` when the deterministic path had decided, otherwise
        ``applied``, ``unclear``, ``deferred`` or ``failed``.
    """
    if not needs_outcome_request(battle):
        return "skipped"

    metadata: dict[str, Any] = {"battle_slug": battle.slug}
    if battle_id is not None:
        metadata["battle_id"] = battle_id

    response = outcome_pass.service.complete(
        system=outcome_pass.system,
        user=render_outcome_prompt(
            outcome_pass.template, battle, lead, outcome_pass.max_lead_chars
        ),
        json_schema=outcome_pass.schema,
        schema_name="outcome",
        metadata=metadata,
    )

    if response.status is CallStatus.DEFERRED:
        battle.failures.append(
            ExtractionFailure(
                battle_name=battle.name,
                source_ref=OUTCOME_STAGE,
                status=DEFERRED_STATUS,
                error="outcome request pending offline processing",
            )
        )
        _drop_outcome_gaps(battle)
        return "deferred"

    if not response.ok or response.data is None:
        battle.failures.append(
            ExtractionFailure(
                battle_name=battle.name,
                source_ref=OUTCOME_STAGE,
                status=response.status.value,
                error=response.error or "no data returned",
            )
        )
        return "failed"

    if not apply_outcome_answer(battle, response.data):
        return "unclear"
    _drop_outcome_gaps(battle)
    return "applied"
