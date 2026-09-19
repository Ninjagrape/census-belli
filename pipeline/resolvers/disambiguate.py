"""
Step 3: asking a model to break the ties the matcher could not.

Only groups the deterministic steps marked ``ambiguous`` reach here, which on
a healthy corpus is a small minority. The question put to the model is
correspondingly narrow -- choose among these candidates, or say none fits --
and it is asked once per group, not once per mention.

Three answers come back, and each maps onto a different risk:

``<Q-id>``
    Linked, at the model's stated confidence capped below an exact match's.

``NEW_ENTITY``
    A corpus-local identity. This is the model declining to merge, which is
    the safer direction (see :mod:`pipeline.resolvers.matcher`).

``UNCERTAIN``
    Also a corpus-local identity, at lower confidence. A split the ranking
    can absorb beats a merge that pools two commanders' records, and the
    resolution log records that it was the model that hesitated.

A call that fails outright is different from all three: it is not an answer,
so the group is left **unresolved** and logged to ``missing_data_log``. The
alternative -- inventing an entity because an API timed out -- would put the
outage into the data where nothing downstream could see it.
"""

from __future__ import annotations

from typing import Any, Final

import structlog

from pipeline.extractors.article import render_template
from pipeline.llm import LLMService
from pipeline.resolvers.records import Candidate, Decision, MentionGroup

__all__ = [
    "MAX_CANDIDATES_IN_PROMPT",
    "OUTPUT_SCHEMA_NAME",
    "disambiguate",
    "format_candidates",
    "prompt_parts",
]

logger = structlog.get_logger()

OUTPUT_SCHEMA_NAME: Final[str] = "resolution"

# An LLM tie-break is never trusted as far as an exact label match, however
# sure the model says it is.
_MAX_LLM_CONFIDENCE: Final[float] = 0.9
_MAX_NEW_ENTITY_CONFIDENCE: Final[float] = 0.7
_UNCERTAIN_CONFIDENCE: Final[float] = 0.4

# More than this and the prompt is mostly noise; the matcher has already
# ordered them, so the tail is the part worth dropping.
MAX_CANDIDATES_IN_PROMPT: Final[int] = 8

_ROLE_EVIDENCE_CHARS: Final[int] = 600


def prompt_parts(spec: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    """Read the prompt and output schema from the agent spec.

    Args:
        spec: The loaded ``agents/resolve.yaml``.

    Returns:
        The system prompt, the user template and the output schema, exactly
        as the spec defines them.

    Raises:
        ValueError: If any is missing. Improvising a prompt here would make
            the audit trail describe a question that was never asked, and
            ``llm_calls`` hashes the prompt, so the cache would diverge from
            the spec as well.
    """
    prompt = spec.get("prompt") or {}
    system = prompt.get("system")
    template = prompt.get("user_template")
    schema = prompt.get("output_schema")

    if not system or not template or not isinstance(schema, dict):
        raise ValueError(
            "agents/resolve.yaml must define prompt.system, prompt.user_template and "
            "prompt.output_schema; the stage loads them at run time rather than "
            "carrying a copy."
        )

    return str(system), str(template), schema


def format_candidates(candidates: list[Candidate]) -> str:
    """Render candidates for the prompt.

    Args:
        candidates: The viable candidates, best first.

    Returns:
        One block per candidate, carrying the Q-id the model must answer
        with, the label, the description, the lifespan and any aliases.
        Lifespan is rendered in astronomical years, negative for BC, because
        that is the form the rest of the stage uses and a "BC" suffix invites
        the model to do its own arithmetic.
    """
    if not candidates:
        return "(none)"

    lines: list[str] = []
    for candidate in candidates[:MAX_CANDIDATES_IN_PROMPT]:
        born = "?" if candidate.birth_year is None else str(candidate.birth_year)
        died = "?" if candidate.death_year is None else str(candidate.death_year)
        parts = [f"- {candidate.qid}: {candidate.label} ({born} to {died}, astronomical years)"]
        if candidate.description:
            parts.append(f"  description: {candidate.description}")
        if candidate.country:
            parts.append(f"  citizenship: {candidate.country}")
        if candidate.occupations:
            parts.append(f"  occupations: {', '.join(candidate.occupations)}")
        if candidate.aliases:
            parts.append(f"  also known as: {', '.join(candidate.aliases[:6])}")
        lines.append("\n".join(parts))

    return "\n".join(lines)


def _prompt_values(group: MentionGroup, candidates: list[Candidate]) -> dict[str, str]:
    """Build the template values for one group.

    The first mention supplies the battle context. A group spanning several
    battles could supply any of them; the first-loaded one is used so that a
    re-run renders the same prompt and hits the ``llm_calls`` cache rather
    than paying again.

    Args:
        group: The ambiguous group.
        candidates: Its viable candidates.

    Returns:
        Replacements for the spec's ``user_template`` placeholders.
    """
    mention = group.mentions[0]
    context = mention.context
    evidence = next((m.role_evidence for m in group.mentions if m.role_evidence.strip()), "")

    return {
        "name": group.display_name,
        "battle_name": mention.battle_name,
        "date": context.date_text if context and context.date_text else "unknown",
        "war": context.war if context and context.war else "unknown",
        "side_label": mention.side_label or "unknown",
        "apparent_role": mention.apparent_role or "unclear",
        "role_evidence": evidence[:_ROLE_EVIDENCE_CHARS],
        "candidates_formatted": format_candidates(candidates),
    }


def _confidence(payload: dict[str, Any], ceiling: float) -> float:
    """Read the model's confidence, capped and clamped.

    Args:
        payload: The parsed response.
        ceiling: The most this decision type may claim.

    Returns:
        A confidence in 0..``ceiling``. A missing or unreadable value becomes
        half the ceiling rather than zero: the model did answer, it just did
        not quantify itself.
    """
    raw = payload.get("confidence")
    if not isinstance(raw, (int, float, str)):
        return round(ceiling / 2, 3)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return round(ceiling / 2, 3)
    return round(min(ceiling, max(0.0, value)), 3)


def disambiguate(
    group: MentionGroup,
    candidates: list[Candidate],
    service: LLMService,
    *,
    system: str,
    template: str,
    schema: dict[str, Any],
) -> Decision:
    """Ask the model which candidate a group refers to.

    Args:
        group: The ambiguous mention group.
        candidates: Its viable candidates, best first.
        service: The stage's LLM service, which caches on the request hash
            and writes the ``llm_calls`` audit row.
        system: ``prompt.system`` from the spec.
        template: ``prompt.user_template`` from the spec.
        schema: ``prompt.output_schema`` from the spec.

    Returns:
        A final decision: ``linked``, ``new``, or ``unresolved`` when the
        call itself failed.
    """
    user = render_template(template, _prompt_values(group, candidates))

    response = service.complete(
        system=system,
        user=user,
        json_schema=schema,
        schema_name=OUTPUT_SCHEMA_NAME,
        metadata={"group": group.key, "mentions": len(group.mentions)},
    )

    if not response.ok or response.data is None:
        logger.warning(
            "resolve_disambiguation_failed",
            group=group.key,
            status=response.status.value,
            error=response.error,
        )
        return Decision(
            status="unresolved",
            canonical_name=group.display_name,
            method="llm_failed",
            candidates_considered=len(candidates),
            reasoning=f"disambiguation call failed: {response.error or response.status.value}",
        )

    payload = response.data
    match = str(payload.get("match") or "").strip()
    reasoning = str(payload.get("reasoning") or "")[:500]
    by_qid = {c.qid: c for c in candidates}

    if match in by_qid:
        candidate = by_qid[match]
        return Decision(
            status="linked",
            canonical_name=candidate.label,
            method="llm_disambiguation",
            qid=candidate.qid,
            confidence=_confidence(payload, _MAX_LLM_CONFIDENCE),
            candidates_considered=len(candidates),
            candidate=candidate,
            reasoning=reasoning,
        )

    if match.upper() == "NEW_ENTITY":
        return Decision(
            status="new",
            canonical_name=group.display_name,
            method="llm_disambiguation",
            confidence=_confidence(payload, _MAX_NEW_ENTITY_CONFIDENCE),
            candidates_considered=len(candidates),
            reasoning=reasoning or "model judged none of the candidates to match",
        )

    if match.upper() == "UNCERTAIN":
        return Decision(
            status="new",
            canonical_name=group.display_name,
            method="llm_disambiguation",
            confidence=_UNCERTAIN_CONFIDENCE,
            candidates_considered=len(candidates),
            reasoning=f"model uncertain; kept separate rather than merged. {reasoning}".strip(),
        )

    # A Q-id that was never offered. The schema cannot express "one of these
    # eight ids", so this is the one wrong answer that still validates.
    logger.warning(
        "resolve_disambiguation_unknown_match",
        group=group.key,
        match=match[:40],
        offered=len(candidates),
    )
    return Decision(
        status="new",
        canonical_name=group.display_name,
        method="llm_disambiguation",
        confidence=_UNCERTAIN_CONFIDENCE,
        candidates_considered=len(candidates),
        reasoning=f"model returned {match[:40]!r}, which was not among the candidates offered",
    )
