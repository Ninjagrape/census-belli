"""
Command-role classification: task A of ``agents/classify.yaml``.

Every side needs three things decided: each commander's role, the hierarchy
(who reports to whom), and an attribution weight that sums to 1.0 across the
side. Three situations reach different amounts of certainty about all three:

1. **One commander.** No ambiguity is possible. Rule, no review.
2. **Several commanders whose roles are already consistent** -- resolve, or
   a prior classify run, already wrote a sensible role assignment (exactly
   one field commander, the rest in non-field roles, nothing 'unknown').
   Keep it, and derive rank/weight/reports_to from a fixed role table. Rule,
   no review.
3. **Everything else.** The deterministic signal runs out: roles are absent,
   inconsistent, or contradictory. This is the fallback the spec's prompt
   describes for when *it* also has nothing to go on -- 0.6/0.4 by listing
   order for two commanders, equal split for three or more, role 'unknown' --
   and it is also exactly the set of sides worth spending an LLM call on.
   :func:`build_role_request` in requests.py is only ever built for a side
   whose deterministic decision came out this way.

:func:`apply_llm_classification` is the fourth step: turning the model's
answer for a (3)-side back into :class:`RoleDecision` rows.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Final

from pipeline.classifiers.records import CommanderRow, RoleDecision, SideGroup

__all__ = [
    "COMMAND_ROLE_RESPONSE_SCHEMA",
    "apply_llm_classification",
    "classify_side_roles",
    "normalize_name",
]

# ─── Weight table for the "roles already consistent" path ────────────────────
#
# Not measured -- there is no ground truth for "how much of Actium was
# Agrippa's" -- so these are judgement calls, chosen to land in the ranges
# agents/classify.yaml's own prompt suggests to the LLM (sovereign 0.05-0.15,
# subordinate 0.2-0.4) after normalisation. Raw scores, not weights: each
# side's scores are summed and divided through, so only the *ratios* between
# roles matter, not the absolute numbers.
#
# A sovereign or nominal figure gets the smallest share regardless of how
# many other commanders are present. A field commander anchors the side and
# is scored well above any one subordinate, so adding subordinates dilutes
# the field commander's share without ever making a single subordinate
# outweigh them for a plausible commander count (up to five or six a side).
_ROLE_BASE_SCORE: Final[dict[str, float]] = {
    "field_commander": 0.65,
    "theatre_commander": 0.55,
    "supreme_commander": 0.55,
    "subordinate": 0.30,
    "sovereign": 0.10,
    "nominal": 0.10,
}

# Roles that field_commander.rule_roles path checks as valid siblings.
_NON_FIELD_ROLES: Final[frozenset[str]] = frozenset(
    {"sovereign", "supreme_commander", "theatre_commander", "subordinate", "nominal"}
)

_WHITESPACE_RE: Final = re.compile(r"\s+")

# The LLM never offers 'unknown'; it is only ever the fallback for a role
# the model returned that is not one of these six.
_LLM_OFFERED_ROLES: Final[frozenset[str]] = frozenset(
    {
        "sovereign",
        "supreme_commander",
        "theatre_commander",
        "field_commander",
        "subordinate",
        "nominal",
    }
)

COMMAND_ROLE_RESPONSE_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "classifications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "command_role": {"type": "string"},
                    "hierarchy_rank": {"type": "integer", "minimum": 0},
                    "reports_to": {"type": ["string", "null"]},
                    "attribution_weight": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "reasoning": {"type": "string"},
                },
                "required": ["name", "command_role", "hierarchy_rank", "attribution_weight"],
            },
        },
        "needs_review": {"type": "boolean"},
        "review_reason": {"type": ["string", "null"]},
    },
    "required": ["classifications"],
}


def normalize_name(raw: str) -> str:
    """Fold a name to the key ``apply_llm_classification`` matches names on.

    Args:
        raw: A commander name, from either a :class:`CommanderRow` or the
            LLM's response.

    Returns:
        The name casefolded with internal whitespace collapsed and leading
        or trailing whitespace stripped. Deliberately shallow -- unlike
        ``pipeline.resolvers.names.fold`` this does not strip titles or
        uninvert "Surname, Forename" spellings, because both sides of the
        match are names this stage already received from the same
        resolved-commander list, not raw source text.
    """
    return _WHITESPACE_RE.sub(" ", raw.strip()).casefold()


def _normalize_weights(raw: Sequence[float]) -> list[float]:
    """Scale non-negative scores to weights summing to exactly 1.0.

    Args:
        raw: One score per commander, same order as the output.

    Returns:
        Weights in the same order. When every score is zero (or the list
        is empty), an equal split is returned instead of a division by
        zero. The last weight absorbs whatever floating-point remainder is
        left after scaling the others, so the sum is exactly 1.0 rather
        than merely close to it -- the quality gate in agents/classify.yaml
        checks ``ABS(SUM(attribution_weight) - 1.0)``, and "close" should
        not depend on how many commanders happened to be on the side.
    """
    n = len(raw)
    if n == 0:
        return []

    total = sum(raw)
    weights = [1.0 / n] * n if total <= 0 else [max(0.0, r) / total for r in raw]

    weights[-1] += 1.0 - sum(weights)
    return weights


def _is_consistent(commanders: Sequence[CommanderRow]) -> bool:
    """Whether a side's existing roles need no deterministic override.

    Args:
        commanders: Every commander on the side.

    Returns:
        True when exactly one commander is 'field_commander', every other
        commander holds a non-field, non-'unknown' role, and there is more
        than one commander (a single commander is handled separately, by
        :func:`classify_side_roles`, so this never has to consider it).
    """
    if len(commanders) < 2:
        return False

    roles = [c.command_role for c in commanders]
    field_commanders = sum(1 for r in roles if r == "field_commander")
    if field_commanders != 1:
        return False

    return all(r in _NON_FIELD_ROLES for r in roles if r != "field_commander")


def _rule_roles_decisions(commanders: Sequence[CommanderRow]) -> list[RoleDecision]:
    """Derive rank, reports_to and weight for an already-consistent side.

    Args:
        commanders: Every commander on the side, roles already consistent
            per :func:`_is_consistent`.

    Returns:
        One decision per commander, in the same order as ``commanders``.
    """
    field_commander = next(c for c in commanders if c.command_role == "field_commander")
    scores = [_ROLE_BASE_SCORE[c.command_role] for c in commanders]
    weights = _normalize_weights(scores)

    decisions: list[RoleDecision] = []
    for commander, weight in zip(commanders, weights, strict=True):
        is_field = commander.command_role == "field_commander"
        reports_to = (
            field_commander.bc_id
            if commander.command_role == "subordinate" and not is_field
            else None
        )
        decisions.append(
            RoleDecision(
                bc_id=commander.bc_id,
                command_role=commander.command_role,
                hierarchy_rank=0 if is_field else 1,
                reports_to_bc_id=reports_to,
                attribution_weight=weight,
                attribution_method="rule_roles",
                confidence=0.8,
                needs_review=False,
                reason=(
                    "existing command_role assignments were internally consistent "
                    "(one field_commander, the rest non-field, none unknown); "
                    "hierarchy and weight derived from the role weight table"
                ),
            )
        )
    return decisions


def _default_split_decisions(commanders: Sequence[CommanderRow]) -> list[RoleDecision]:
    """The spec's fallback for a side with no usable role signal.

    Args:
        commanders: Every commander on the side, in listing order.

    Returns:
        One decision per commander: role 'unknown', method 'default_split',
        flagged for review. Two commanders split 0.6/0.4 by listing order
        (documented as a seniority proxy, not a fact); three or more split
        equally. Listing order is the infobox's own order -- the first name
        an editor wrote is usually, not always, the more senior; the article
        excerpt an LLM call would see is what actually settles it, which is
        why this path is exactly what routes a side to the LLM.
    """
    ordered = sorted(commanders, key=lambda c: c.listing_order)
    n = len(ordered)

    raw = [0.6, 0.4] if n == 2 else [1.0] * n
    weights = _normalize_weights(raw)

    reason = (
        "no single commander and existing roles were absent or inconsistent; "
        "used the spec's fallback split rather than guess a role"
    )
    return [
        RoleDecision(
            bc_id=commander.bc_id,
            command_role="unknown",
            hierarchy_rank=0,
            reports_to_bc_id=None,
            attribution_weight=weight,
            attribution_method="default_split",
            confidence=0.2,
            needs_review=True,
            reason=reason,
        )
        for commander, weight in zip(ordered, weights, strict=True)
    ]


def classify_side_roles(side: SideGroup) -> list[RoleDecision]:
    """Deterministic role classification for one side.

    Args:
        side: The side to classify.

    Returns:
        One :class:`RoleDecision` per commander on the side, in
        ``side.commanders`` order. An empty side yields an empty list.
        Callers building LLM requests should do so only for a side whose
        decisions all came back with ``attribution_method == "default_split"``
        -- that is the fallback this function returns when, and only when,
        it found nothing deterministic to say.
    """
    commanders = side.commanders
    if not commanders:
        return []

    if len(commanders) == 1:
        commander = commanders[0]
        return [
            RoleDecision(
                bc_id=commander.bc_id,
                command_role="field_commander",
                hierarchy_rank=0,
                reports_to_bc_id=None,
                attribution_weight=1.0,
                attribution_method="rule_single",
                confidence=0.95,
                needs_review=False,
                reason="only one commander listed on this side",
            )
        ]

    if _is_consistent(commanders):
        return _rule_roles_decisions(commanders)

    return _default_split_decisions(commanders)


# ─── LLM response mapping ─────────────────────────────────────────────────────


def _fallback_decision(commander: CommanderRow, *, reason: str) -> RoleDecision:
    """Build the per-commander fallback used when the LLM response omits them.

    Args:
        commander: The commander the response did not name.
        reason: Why this row fell back, for the review queue.

    Returns:
        An 'unknown', needs-review decision. Its ``attribution_weight`` is a
        placeholder score, not yet normalised -- :func:`apply_llm_classification`
        renormalises every decision on the side together once all of them,
        matched and fallback alike, are built.
    """
    return RoleDecision(
        bc_id=commander.bc_id,
        command_role="unknown",
        hierarchy_rank=0,
        reports_to_bc_id=None,
        attribution_weight=1.0,
        attribution_method="default_split",
        confidence=0.0,
        needs_review=True,
        reason=reason,
    )


def apply_llm_classification(side: SideGroup, data: dict[str, Any]) -> list[RoleDecision]:
    """Turn a parsed ``command_role_prompt`` response into role decisions.

    Args:
        side: The side the request was built for.
        data: The response's parsed JSON body (``response.data`` from
            :class:`pipeline.llm.base.LLMResponse`), matching
            ``COMMAND_ROLE_RESPONSE_SCHEMA``.

    Returns:
        One :class:`RoleDecision` per commander in ``side.commanders``,
        weights renormalised to sum to 1.0 across the whole side.

        Each response ``classifications`` entry is matched to a commander
        by name (:func:`normalize_name` on both sides). A commander the
        response does not mention falls back to an 'unknown', needs-review
        decision rather than being dropped; an entry naming nobody on this
        side is logged and otherwise ignored, since there is no row to
        attach it to. ``reports_to`` is resolved to a ``bc_id`` only among
        this side's own commanders -- a name naming someone on the other
        side, or nobody at all, resolves to None rather than guessed or
        looked up elsewhere. A self-reference (a commander named as their
        own superior) also resolves to None. A role outside the six the
        prompt offers becomes 'unknown' and flags the row for review, since
        the schema cannot itself reject a value the model invented.
    """
    commanders = side.commanders
    by_key = {normalize_name(c.name): c for c in commanders}

    overall_review = bool(data.get("needs_review"))
    overall_reason = str(data.get("review_reason") or "").strip()

    raw_items = data.get("classifications")
    items = raw_items if isinstance(raw_items, list) else []

    matched: dict[int, RoleDecision] = {}
    weight_source: dict[int, float] = {}

    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        key = normalize_name(name)
        commander = by_key.get(key)
        if commander is None:
            # Not on this side: a hallucinated name, a typo, or (should the
            # prompt ever be misbuilt) a commander from a different side.
            # There is no row to write this to, so it is dropped rather
            # than guessed onto the nearest name.
            continue

        role = str(item.get("command_role") or "").strip().lower()
        role_valid = role in _LLM_OFFERED_ROLES
        final_role = role if role_valid else "unknown"

        reports_to_name = item.get("reports_to")
        reports_to_bc_id: int | None = None
        if isinstance(reports_to_name, str) and reports_to_name.strip():
            target_key = normalize_name(reports_to_name)
            if target_key != key:  # a self-reference is not a hierarchy
                target = by_key.get(target_key)
                if target is not None:  # same side only; never resolved elsewhere
                    reports_to_bc_id = target.bc_id

        try:
            rank = int(item.get("hierarchy_rank", 0))
        except (TypeError, ValueError):
            rank = 0

        try:
            confidence = min(1.0, max(0.0, float(item.get("confidence", 0.5))))
        except (TypeError, ValueError):
            confidence = 0.5

        try:
            weight = min(1.0, max(0.0, float(item.get("attribution_weight", 0.0))))
        except (TypeError, ValueError):
            weight = 0.0

        needs_review = overall_review or not role_valid
        reason_bits = [str(item.get("reasoning") or "").strip()]
        if not role_valid:
            reason_bits.append(f"model returned unrecognised role {role!r}")
        if overall_reason:
            reason_bits.append(overall_reason)
        reason = "; ".join(b for b in reason_bits if b) or "llm command role classification"

        matched[commander.bc_id] = RoleDecision(
            bc_id=commander.bc_id,
            command_role=final_role,
            hierarchy_rank=max(0, rank),
            reports_to_bc_id=reports_to_bc_id,
            attribution_weight=weight,
            attribution_method="llm",
            confidence=confidence,
            needs_review=needs_review,
            reason=reason,
        )
        weight_source[commander.bc_id] = weight

    decisions: list[RoleDecision] = []
    for commander in commanders:
        if commander.bc_id in matched:
            decisions.append(matched[commander.bc_id])
        else:
            decisions.append(
                _fallback_decision(
                    commander,
                    reason="LLM response did not name this commander; used deterministic fallback",
                )
            )
            weight_source[commander.bc_id] = 1.0

    raw_weights = [weight_source[c.bc_id] for c in commanders]
    normalized = _normalize_weights(raw_weights)

    return [
        RoleDecision(
            bc_id=d.bc_id,
            command_role=d.command_role,
            hierarchy_rank=d.hierarchy_rank,
            reports_to_bc_id=d.reports_to_bc_id,
            attribution_weight=w,
            attribution_method=d.attribution_method,
            confidence=d.confidence,
            needs_review=d.needs_review,
            reason=d.reason,
        )
        for d, w in zip(decisions, normalized, strict=True)
    ]
