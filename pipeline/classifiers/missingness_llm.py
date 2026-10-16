"""
The LLM half of missingness classification, for rows the heuristics leave open.

:func:`pipeline.classifiers.missingness.classify_missingness` returns None for
a row it has no rule for, and those rows used to stay ``unclassified`` for
good. This module builds the request that asks the classify model about such a
row and turns a parsed answer into a :class:`MissingnessDecision`. It is pure,
like the rest of :mod:`pipeline.classifiers`; the stage sends the request
through the same offline cache or live service as the command-role calls.

The request is built from stable facts only (see
:class:`pipeline.classifiers.records.MissingnessContext`), so its hash is the
same on the run that defers it and the run that applies the imported answer.
"""

from __future__ import annotations

from typing import Any, Final

from pipeline.classifiers.records import MissingnessContext, MissingnessDecision
from pipeline.classifiers.requests import format_year
from pipeline.llm.base import LLMRequest

__all__ = [
    "MISSINGNESS_RESPONSE_SCHEMA",
    "apply_llm_missingness",
    "build_missingness_request",
    "era_label",
]

_SCHEMA_NAME: Final[str] = "missingness"
_DEFAULT_MAX_TOKENS: Final[int] = 512

# 'observed' and 'unclassified' are states of the row, not mechanisms the
# model may answer with.
_ANSWERABLE: Final[tuple[str, ...]] = ("mcar", "mar", "mnar")

MISSINGNESS_RESPONSE_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "missingness_type": {"type": "string", "enum": list(_ANSWERABLE)},
        "reason": {"type": "string"},
    },
    "required": ["missingness_type", "reason"],
    "additionalProperties": False,
}

# Era buckets, astronomical years. Coarse on purpose: the prompt's heuristics
# are stated per era, not per decade.
_ERAS: Final[tuple[tuple[int, str], ...]] = (
    (500, "ancient (before 500)"),
    (1500, "medieval (500-1499)"),
    (1700, "early modern (1500-1699)"),
    (1815, "18th century to 1814"),
    (1914, "1815-1913"),
    (1946, "world wars (1914-1945)"),
)
_MODERN: Final[str] = "post-1945"


def era_label(year_astronomical: int | None) -> str:
    """Name the era a year falls in, for the prompt.

    Args:
        year_astronomical: The battle's year, or None when undated.

    Returns:
        A short era label; "unknown" when undated.
    """
    if year_astronomical is None:
        return "unknown"
    for upper, label in _ERAS:
        if year_astronomical < upper:
            return label
    return _MODERN


def _yes_no(flag: bool) -> str:
    """Render a boolean as the prompt's "known" / "missing"."""
    return "known" if flag else "missing"


def _observed_summary(context: MissingnessContext) -> str:
    """List what is known about the battle and the side, one fact per line.

    Args:
        context: The row's context.

    Returns:
        Bulleted lines. Only facts classify does not rewrite are included.
    """
    lines = []
    if context.side_label:
        lines.append(
            f"- side: {context.side_label}" + (f" ({context.polity})" if context.polity else "")
        )
    lines.append(f"- this side's outcome: {context.outcome or 'unknown'}")
    lines.append(f"- this side's troop total: {_yes_no(context.this_side_has_troops)}")
    lines.append(f"- the other side's troop total: {_yes_no(context.other_side_has_troops)}")
    lines.append(f"- this side's casualties: {_yes_no(context.this_side_has_casualties)}")
    lines.append(f"- the other side's casualties: {_yes_no(context.other_side_has_casualties)}")
    return "\n".join(lines)


def build_missingness_request(context: MissingnessContext, spec: dict[str, Any]) -> LLMRequest:
    """Build the request asking why one field is missing.

    Args:
        context: The missing field and what is known around it.
        spec: The loaded ``agents/classify.yaml``, read for
            ``missingness_prompt.system``/``user_template`` and
            ``params.missingness_max_tokens``.

    Returns:
        A request ready for the offline provider or a live service.

    Raises:
        ValueError: If the spec defines no ``missingness_prompt.system`` or
            ``.user_template``.
    """
    prompt = spec.get("missingness_prompt") or {}
    system = prompt.get("system")
    template = prompt.get("user_template")
    if not system or not template:
        raise ValueError(
            "agents/classify.yaml must define missingness_prompt.system and "
            "missingness_prompt.user_template"
        )

    user = str(template).format(
        battle_name=context.battle_name,
        date=format_year(context.year_astronomical),
        war=context.war_name or "no war recorded",
        region=context.region or "region unknown",
        era=era_label(context.year_astronomical),
        field_name=context.field_name,
        observed_fields_summary=_observed_summary(context),
    )
    params = spec.get("params") or {}
    return LLMRequest(
        system=str(system),
        user=user,
        json_schema=MISSINGNESS_RESPONSE_SCHEMA,
        schema_name=_SCHEMA_NAME,
        max_tokens=int(params.get("missingness_max_tokens", _DEFAULT_MAX_TOKENS)),
        temperature=params.get("llm_temperature"),
        metadata={"battle_id": context.battle_id, "log_id": context.log_id},
    )


def apply_llm_missingness(log_id: int, data: dict[str, Any]) -> MissingnessDecision | None:
    """Turn a parsed missingness answer into a decision.

    Args:
        log_id: The ``missing_data_log`` row the request was built for.
        data: The response body, matching ``MISSINGNESS_RESPONSE_SCHEMA``.

    Returns:
        A decision, or None when the answer does not name one of mcar, mar or
        mnar (the row then stays unclassified rather than taking a guess).
    """
    value = str(data.get("missingness_type") or "").strip().lower()
    if value not in _ANSWERABLE:
        return None
    reason = str(data.get("reason") or "").strip() or "no reason given"
    return MissingnessDecision(
        log_id=log_id, missingness_class=value, reason=f"LLM classification: {reason}"
    )
