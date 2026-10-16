"""
Command-role classification: task A of ``agents/classify.yaml``.

Every side needs three things decided: each commander's role, the hierarchy
(who reports to whom), and an attribution weight that sums to 1.0 across the
side. Three situations reach different amounts of certainty about all three:

1. **One commander.** No ambiguity is possible. Rule, no review.
2. **Several commanders whose roles are already consistent** -- resolve, or
   a prior classify run, already wrote a sensible role assignment (at least
   one field commander, the rest in non-field roles, nothing 'unknown').
   Two or more field commanders (alternating consuls, co-command) are a
   consistent pattern and share the field pool equally. Keep the roles and
   derive rank/weight/reports_to from fixed absolute weights. Rule, no review.
3. **Everything else.** The deterministic signal runs out: roles are absent,
   inconsistent, or contradictory. Fall back to an equal split with role
   'unknown' and flag for review. This is exactly the set of sides worth
   spending an LLM call on. :func:`build_role_request` in requests.py is only
   ever built for a side whose deterministic decision came out this way.

:func:`apply_llm_classification` is the fourth step: turning the model's
answer for a (3)-side back into :class:`RoleDecision` rows.

Whatever path produced them, a side's decisions must pass
:func:`validate_side`. ``hierarchy_rank`` follows the schema comment in
config/schema.sql (0 = top commander on this side, 1 = direct report) and the
prompt's wording ("the top tactical commander has hierarchy_rank 0"): rank 0
is the tactical top, i.e. the field commander(s), not the nominally most
senior person. A supreme commander or sovereign who is not directing the
battle is therefore rank 1 with no ``reports_to``. Supreme/theatre
commanders are rank 0 only when no field commander exists on the side.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import replace
from typing import Any, Final

import structlog

from pipeline.classifiers.records import CommanderRow, RoleDecision, SideGroup

__all__ = [
    "COMMAND_ROLE_RESPONSE_SCHEMA",
    "apply_llm_classification",
    "classify_side_roles",
    "llm_defender_flag",
    "normalize_name",
    "validate_side",
]

logger = structlog.get_logger()

# ─── Weight tables for the "roles already consistent" path ───────────────────
#
# Not measured -- there is no ground truth for "how much of Actium was
# Agrippa's" -- so these are judgement calls.
#
# With a field commander present, each non-field commander gets a fixed
# absolute share, independent of how many other people are listed, and the
# field commander(s) split what is left. A normalised table instead lets a
# long infobox list dilute the person who actually commanded: Agrippa,
# Octavian and two subordinates came out at 0.5-ish under normalisation, which
# is not a defensible reading of Actium. The cap bounds the damage from a
# side with many listed non-field commanders.
_NON_FIELD_WEIGHT: Final[dict[str, float]] = {
    "sovereign": 0.05,
    "nominal": 0.05,
    "supreme_commander": 0.10,
    "theatre_commander": 0.10,
    "subordinate": 0.08,
}
_NON_FIELD_WEIGHT_CAP: Final = 0.30

# Only reached for a side with no field commander, which classify_side_roles
# does not route here today; kept as the normalised fallback so the function
# stays total. Raw scores, not weights: only the ratios matter.
_ROLE_BASE_SCORE: Final[dict[str, float]] = {
    "field_commander": 0.65,
    "theatre_commander": 0.55,
    "supreme_commander": 0.55,
    "subordinate": 0.30,
    "sovereign": 0.10,
    "nominal": 0.10,
}

# Tolerance for "weights sum to 1" in validate_side.
_WEIGHT_SUM_TOLERANCE: Final = 1e-6

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
        # Whether this side held the position (fortress, city, ridge, line) the
        # other side attacked. Null for a meeting engagement or when the text
        # does not say. Optional so an answer cached before the field existed
        # still validates; the prompt always asks for it.
        "is_defender": {"type": ["boolean", "null"]},
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
        True when at least one commander is 'field_commander', every other
        commander holds a non-field, non-'unknown' role, and there is more
        than one commander (a single commander is handled separately, by
        :func:`classify_side_roles`, so this never has to consider it).
        Several field commanders are consistent: alternating consuls and
        joint commands are a real pattern, not a contradiction.
    """
    if len(commanders) < 2:
        return False

    roles = [c.command_role for c in commanders]
    if not any(r == "field_commander" for r in roles):
        return False

    return all(r in _NON_FIELD_ROLES for r in roles if r != "field_commander")


def _rank_for(role: str, *, has_field: bool) -> int:
    """Hierarchy rank for a role on a deterministic path.

    Args:
        role: The commander's role.
        has_field: Whether the side has any field commander.

    Returns:
        0 for a field commander; 0 for a supreme or theatre commander only
        when the side has no field commander; 1 for everyone else.
    """
    if role == "field_commander":
        return 0
    if not has_field and role in {"supreme_commander", "theatre_commander"}:
        return 0
    return 1


def _rule_weights(commanders: Sequence[CommanderRow]) -> list[float]:
    """Weights for a consistent side, in ``commanders`` order.

    Args:
        commanders: Every commander on the side, roles consistent.

    Returns:
        With a field commander: fixed absolute weights for the others
        (scaled down proportionally if they exceed the cap), the rest split
        equally among the field commanders. Without one: the normalised
        role-score table.
    """
    field_count = sum(1 for c in commanders if c.command_role == "field_commander")
    if field_count == 0:
        return _normalize_weights([_ROLE_BASE_SCORE[c.command_role] for c in commanders])

    fixed = [
        0.0 if c.command_role == "field_commander" else _NON_FIELD_WEIGHT[c.command_role]
        for c in commanders
    ]
    total_fixed = sum(fixed)
    if total_fixed > _NON_FIELD_WEIGHT_CAP:
        fixed = [w * _NON_FIELD_WEIGHT_CAP / total_fixed for w in fixed]
        total_fixed = _NON_FIELD_WEIGHT_CAP

    field_share = (1.0 - total_fixed) / field_count
    raw = [
        field_share if c.command_role == "field_commander" else w
        for c, w in zip(commanders, fixed, strict=True)
    ]
    return _normalize_weights(raw)


def _rule_roles_decisions(commanders: Sequence[CommanderRow]) -> list[RoleDecision]:
    """Derive rank, reports_to and weight for an already-consistent side.

    Args:
        commanders: Every commander on the side, roles already consistent
            per :func:`_is_consistent`.

    Returns:
        One decision per commander, in the same order as ``commanders``.
        Subordinates report to the field commander only when there is
        exactly one: under co-command the roles alone do not say whose
        subordinate someone was, so the link is left unset.
    """
    field_commanders = [c for c in commanders if c.command_role == "field_commander"]
    has_field = bool(field_commanders)
    sole_field_id = field_commanders[0].bc_id if len(field_commanders) == 1 else None
    weights = _rule_weights(commanders)

    return [
        RoleDecision(
            bc_id=commander.bc_id,
            command_role=commander.command_role,
            hierarchy_rank=_rank_for(commander.command_role, has_field=has_field),
            reports_to_bc_id=sole_field_id if commander.command_role == "subordinate" else None,
            attribution_weight=weight,
            attribution_method="rule_roles",
            confidence=0.8,
            needs_review=False,
            reason=(
                "existing command_role assignments were internally consistent "
                "(at least one field_commander, the rest non-field, none unknown); "
                "hierarchy and weight derived from the fixed role weights"
            ),
        )
        for commander, weight in zip(commanders, weights, strict=True)
    ]


def _default_split_decisions(commanders: Sequence[CommanderRow]) -> list[RoleDecision]:
    """The fallback for a side with no usable role signal.

    Args:
        commanders: Every commander on the side.

    Returns:
        One decision per commander: role 'unknown', method 'default_split',
        flagged for review, weight 1/n. Listing order is deliberately not
        used to rank anyone: it follows the infobox's (and so bc_id's)
        order, which favours heads of state over the officer who commanded,
        so any unequal split built on it would be systematically biased.
        Every commander gets hierarchy_rank 0 because an unknown role means
        nobody can be placed beneath anyone else; this is the one case where
        several rank-0 rows are valid without being field commanders, and
        :func:`validate_side` allows it for 'default_split' rows only.
    """
    weights = _normalize_weights([1.0] * len(commanders))
    reason = (
        "no single commander and existing roles were absent or inconsistent; "
        "used an equal split rather than guess a role"
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
        for commander, weight in zip(commanders, weights, strict=True)
    ]


def _has_reports_to_cycle(decisions: Sequence[RoleDecision]) -> bool:
    """Whether following ``reports_to_bc_id`` links ever loops back."""
    parent = {d.bc_id: d.reports_to_bc_id for d in decisions}
    for start in parent:
        seen: set[int] = set()
        node: int | None = start
        while node is not None:
            if node in seen:
                return True
            seen.add(node)
            node = parent.get(node)
    return False


def _is_top_rank_exempt(decision: RoleDecision) -> bool:
    """A zero-weight 'unknown' row (a listed non-person) holds no place in the chain."""
    return decision.command_role == "unknown" and decision.attribution_weight == 0.0


def validate_side(decisions: Sequence[RoleDecision]) -> str | None:
    """Check that one side's decisions form a coherent hierarchy.

    Args:
        decisions: Every decision for one side.

    Returns:
        None when valid, otherwise a one-line description of the first
        problem. Rules: weights sum to 1 within 1e-6; at least one rank-0
        row, and exactly one unless every rank-0 row is a field commander
        (co-command) or a 'default_split' unknown (co-equal unknowns);
        ``reports_to_bc_id`` names another commander on this side and the
        links contain no cycle. Zero-weight 'unknown' rows are ignored when
        counting rank-0 rows.
    """
    total = sum(d.attribution_weight for d in decisions)
    if abs(total - 1.0) > _WEIGHT_SUM_TOLERANCE:
        return f"weights sum to {total:.6f}, not 1"

    top = [d for d in decisions if d.hierarchy_rank == 0 and not _is_top_rank_exempt(d)]
    if not top:
        return "no commander has hierarchy rank 0"
    co_equal = all(
        d.command_role == "field_commander"
        or (d.command_role == "unknown" and d.attribution_method == "default_split")
        for d in top
    )
    if len(top) > 1 and not co_equal:
        return f"{len(top)} commanders share hierarchy rank 0 without being co-field commanders"

    ids = {d.bc_id for d in decisions}
    for d in decisions:
        if d.reports_to_bc_id is None:
            continue
        if d.reports_to_bc_id == d.bc_id:
            return f"reports_to of {d.bc_id} is itself"
        if d.reports_to_bc_id not in ids:
            return f"reports_to of {d.bc_id} names a commander not on this side"
    if _has_reports_to_cycle(decisions):
        return "reports_to links contain a cycle"
    return None


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
        it found nothing deterministic to say. A side whose decisions fail
        :func:`validate_side` (a bug here, not bad input) is logged and
        flagged for review rather than written silently.
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

    decisions = (
        _rule_roles_decisions(commanders)
        if _is_consistent(commanders)
        else _default_split_decisions(commanders)
    )
    problem = validate_side(decisions)
    if problem is None:
        return decisions

    logger.error(
        "classify_deterministic_side_invalid",
        battle_id=side.battle_id,
        side_id=side.side_id,
        problem=problem,
    )
    return [replace(d, needs_review=True, reason=f"{d.reason}; {problem}") for d in decisions]


# ─── LLM response mapping ─────────────────────────────────────────────────────


def _match_commander(
    name: str,
    by_key: dict[str, CommanderRow],
    by_surname: dict[str, list[CommanderRow]],
) -> CommanderRow | None:
    """Find the commander on this side an LLM-supplied name refers to.

    Args:
        name: A name from the response.
        by_key: Commanders keyed by :func:`normalize_name` of the full name.
        by_surname: Commanders grouped by the casefolded last token of the name.

    Returns:
        The commander whose full name matches, else the commander whose
        surname matches when exactly one commander on the side has it. None
        when nothing matches or the surname is shared.
    """
    key = normalize_name(name)
    if key in by_key:
        return by_key[key]
    if not key:
        return None
    candidates = by_surname.get(key.rsplit(" ", 1)[-1], [])
    return candidates[0] if len(candidates) == 1 else None


def _reject_llm_answer(side: SideGroup, reason: str, **log_fields: Any) -> list[RoleDecision]:
    """Fall back to the deterministic result for a side the LLM answered badly.

    Args:
        side: The side whose answer is rejected.
        reason: Why, for the review queue and the log.
        **log_fields: Extra structured fields for the warning.

    Returns:
        The deterministic decisions, every one flagged for review.
    """
    logger.warning(
        "classify_llm_answer_rejected",
        battle_id=side.battle_id,
        side_id=side.side_id,
        reason=reason,
        **log_fields,
    )
    return [
        replace(d, needs_review=True, reason=f"LLM answer rejected ({reason}); {d.reason}")
        for d in classify_side_roles(side)
    ]


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
        by casefolded full name, then by casefolded surname when exactly one
        commander on the side has it. An entry naming nobody on this side is
        logged and ignored. If any commander on the side is left unmatched
        the whole answer is rejected for the side: the deterministic result
        is returned flagged for review and a warning names the unmatched
        commanders. No default weight is ever assigned to an unmatched
        commander. The answer is rejected the same way when it fails
        :func:`validate_side`. ``reports_to`` is resolved to a ``bc_id`` only
        among this side's own commanders -- a name naming someone on the other
        side, or nobody at all, resolves to None. A self-reference also
        resolves to None. A role outside the six the prompt offers (including
        the 'unknown' the prompt asks for on a listed non-person) becomes
        'unknown' and flags the row for review.
    """
    commanders = side.commanders
    by_key = {normalize_name(c.name): c for c in commanders}
    by_surname: dict[str, list[CommanderRow]] = {}
    for c in commanders:
        surname = normalize_name(c.name).rsplit(" ", 1)[-1]
        by_surname.setdefault(surname, []).append(c)

    overall_review = bool(data.get("needs_review"))
    overall_reason = str(data.get("review_reason") or "").strip()

    raw_items = data.get("classifications")
    items = raw_items if isinstance(raw_items, list) else []

    matched: dict[int, RoleDecision] = {}

    for item in items:
        if not isinstance(item, dict):
            continue
        commander = _match_commander(str(item.get("name") or ""), by_key, by_surname)
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
            target = _match_commander(reports_to_name, by_key, by_surname)
            # Same side only, never resolved elsewhere; a self-reference is
            # not a hierarchy.
            if target is not None and target.bc_id != commander.bc_id:
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

    unmatched = [c.name for c in commanders if c.bc_id not in matched]
    if unmatched:
        return _reject_llm_answer(side, "unmatched commanders", unmatched=unmatched)

    decisions = [matched[c.bc_id] for c in commanders]
    normalized = _normalize_weights([d.attribution_weight for d in decisions])
    decisions = [
        replace(d, attribution_weight=w) for d, w in zip(decisions, normalized, strict=True)
    ]

    problem = validate_side(decisions)
    if problem is not None:
        return _reject_llm_answer(side, problem)
    return decisions


def llm_defender_flag(decisions: Sequence[RoleDecision], data: dict[str, Any]) -> bool | None:
    """Read the side's ``is_defender`` from an accepted LLM answer.

    Args:
        decisions: What :func:`apply_llm_classification` returned for the side.
        data: The same response body.

    Returns:
        True or False when the answer was accepted and states a boolean; None
        when the answer was rejected (its decisions fall back to the
        deterministic method, so nothing in it is trusted), omitted the field,
        or answered null ("unclear" or a meeting engagement). A non-boolean
        value is treated as null rather than coerced: ``"false"`` and ``0``
        are not an answer to a yes/no question.
    """
    if not decisions or any(d.attribution_method != "llm" for d in decisions):
        return None
    value = data.get("is_defender")
    return value if isinstance(value, bool) else None
