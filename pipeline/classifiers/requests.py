"""
Building the command-role LLM request, task A's only LLM call.

Only a side whose deterministic decision (:func:`pipeline.classifiers.roles.
classify_side_roles`) came back ``attribution_method == "default_split"``
needs this: a single commander is unambiguous, and a side whose roles are
already consistent needs no model at all. That is deliberate, and matches
handover.md's note that classify makes far fewer calls than extract -- most
sides never reach here.

The request is built entirely from :class:`pipeline.classifiers.records.SideGroup`
and an already-selected excerpt, with no database access, so
``scripts/llm_offline.py``'s ``_classify_requests`` (currently a stub) can
build the same requests a live run would build, and both land on the same
``request_hash`` for a given input.

The spec's ``command_role_prompt.user_template`` uses ``{{`` / ``}}`` around
its literal JSON example, which is exactly what :meth:`str.format` treats as
an escaped brace -- so, unlike ``pipeline.extractors.article.render_template``
(a regex substitution chosen so an article's own braces can never raise),
this module renders with ``str.format`` directly.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Final

from pipeline.classifiers.records import CommanderRow, SideGroup
from pipeline.classifiers.roles import COMMAND_ROLE_RESPONSE_SCHEMA
from pipeline.llm.base import LLMRequest

__all__ = [
    "build_role_request",
    "select_excerpt",
]

_SCHEMA_NAME: Final[str] = "command_role"
_DEFAULT_MAX_TOKENS: Final[int] = 4096

_SURNAME_RE: Final = re.compile(r"[^\w'-]+", re.UNICODE)


def _format_date(year_astronomical: int | None) -> str:
    """Render a battle year for the prompt.

    Args:
        year_astronomical: Astronomical-numbering year (handover.md §4.5),
            or None when undated.

    Returns:
        A BC/AD year string, or "unknown" when undated. Astronomical year 0
        is 1 BC and year -30 is 31 BC (``BC_year = 1 - year_astronomical``
        for any ``year_astronomical <= 0``), matching the convention the
        rest of the pipeline stores dates in.
    """
    if year_astronomical is None:
        return "unknown"
    if year_astronomical <= 0:
        return f"{1 - year_astronomical} BC"
    return str(year_astronomical)


def _format_commanders(commanders: Sequence[CommanderRow]) -> str:
    """Render the "commanders listed" block of the prompt.

    Args:
        commanders: The side's commanders, any order (sorted here by
            listing order for a stable, deterministic prompt).

    Returns:
        One line per commander, in listing order.
    """
    ordered = sorted(commanders, key=lambda c: c.listing_order)
    if not ordered:
        return "(none)"
    return "\n".join(f"- {c.name} (apparent role: {c.apparent_role})" for c in ordered)


def _format_role_evidence(commanders: Sequence[CommanderRow]) -> str:
    """Render the "extracted role evidence" block of the prompt.

    Args:
        commanders: The side's commanders.

    Returns:
        One line per commander naming what evidence, if any, extraction
        already found for their role.
    """
    ordered = sorted(commanders, key=lambda c: c.listing_order)
    if not ordered:
        return "(none)"
    lines = []
    for c in ordered:
        evidence = c.role_evidence.strip() or "(no evidence extracted)"
        lines.append(f"- {c.name}: {evidence}")
    return "\n".join(lines)


def _surname(name: str) -> str:
    """The last token of a name, for a loose "surname match".

    Args:
        name: A commander's full name.

    Returns:
        The final whitespace-separated token, stripped of punctuation
        other than an internal apostrophe or hyphen, casefolded. Empty for
        a blank name.
    """
    tokens = [t for t in _SURNAME_RE.split(name.strip()) if t]
    return tokens[-1].casefold() if tokens else ""


def select_excerpt(passages: Sequence[str], names: Sequence[str], max_chars: int) -> str:
    """Choose article text worth sending the command-role prompt.

    Args:
        passages: Article text chunks in document order -- e.g. the
            ``.text`` of each ``pipeline.extractors.article.Passage``, or
            any other ordered split of the cleaned article.
        names: The side's commander names. A passage mentioning any
            commander's full name (casefold match) or surname (last-token
            match) is preferred.
        max_chars: Character budget for the returned excerpt.

    Returns:
        The matching passages joined in document order, each kept whole
        except the one that would cross ``max_chars``, which is truncated
        to fit. When no passage mentions any commander -- an article that
        talks about the battle without naming who led it, or a name form
        the article does not use -- falls back to the passages from the
        top of the article, still capped, so the prompt never goes out
        with an empty excerpt when an article exists. Deterministic: the
        same inputs always produce the same string, which is what keeps
        the resulting request's hash stable across a re-run.
    """
    full_names = {n.strip().casefold() for n in names if n.strip()}
    surnames = {s for s in (_surname(n) for n in names) if s}

    def mentions_a_commander(passage: str) -> bool:
        haystack = passage.casefold()
        if any(name in haystack for name in full_names):
            return True
        return any(re.search(rf"\b{re.escape(sn)}\b", haystack) for sn in surnames)

    matches = [p for p in passages if mentions_a_commander(p)]
    chosen = matches or list(passages)

    out: list[str] = []
    used = 0
    for chunk in chosen:
        if used >= max_chars:
            break
        remaining = max_chars - used
        piece = chunk if len(chunk) <= remaining else chunk[:remaining]
        out.append(piece)
        used += len(piece)

    return "\n\n".join(out)


def build_role_request(side: SideGroup, article_excerpt: str, spec: dict[str, Any]) -> LLMRequest:
    """Build the command-role classification request for one side.

    Args:
        side: The side to classify. Only sensible for a side whose
            deterministic decision was 'default_split' -- see the module
            docstring.
        article_excerpt: The article text to include, e.g. from
            :func:`select_excerpt`. Passed in rather than computed here so
            this function stays free of any opinion about how the excerpt
            was chosen.
        spec: The loaded ``agents/classify.yaml``, read for
            ``command_role_prompt.system``/``user_template`` and
            ``params.llm_max_tokens``/``params.llm_temperature``.

    Returns:
        A request ready for :meth:`pipeline.llm.service.LLMService.complete`,
        or for :func:`pipeline.llm.offline.export_pending`.

    Raises:
        ValueError: If the spec defines no ``command_role_prompt.system`` or
            ``.user_template``. Improvising either would ask a question
            ``agents/classify.yaml`` never specified, and silently diverge
            the request hash from what a spec-driven run would compute.
    """
    prompt = spec.get("command_role_prompt") or {}
    system = prompt.get("system")
    template = prompt.get("user_template")
    if not system or not template:
        raise ValueError(
            "agents/classify.yaml must define command_role_prompt.system and "
            "command_role_prompt.user_template; the stage loads them at run "
            "time rather than carrying a copy."
        )

    user = str(template).format(
        battle_name=side.battle_name,
        date=_format_date(side.year_astronomical),
        side_label=side.side_label,
        commanders_formatted=_format_commanders(side.commanders),
        role_evidence_formatted=_format_role_evidence(side.commanders),
        article_excerpt=article_excerpt,
    )

    params = spec.get("params") or {}
    max_tokens = int(params.get("llm_max_tokens", _DEFAULT_MAX_TOKENS))
    temperature = params.get("llm_temperature")

    return LLMRequest(
        system=str(system),
        user=user,
        json_schema=COMMAND_ROLE_RESPONSE_SCHEMA,
        schema_name=_SCHEMA_NAME,
        max_tokens=max_tokens,
        temperature=temperature,
        metadata={"battle_id": side.battle_id, "side_id": side.side_id},
    )
