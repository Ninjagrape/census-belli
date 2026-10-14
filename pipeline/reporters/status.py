"""Why a report is provisional."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

from pipeline.reporters.records import ModelGate

__all__ = ["BLOCKING_MODEL_GATES", "provisional_reasons"]

# The model stage records these at warning severity; agents/report.yaml raises
# them to errors, so a failure blocks publication here as it does there.
BLOCKING_MODEL_GATES: Final[frozenset[str]] = frozenset({"exclusion_concentration"})


def _effective_severity(gate: Mapping[str, Any], severities: Mapping[str, str]) -> str:
    own = gate.get("severity")
    if isinstance(own, str) and own:
        return own
    return severities.get(str(gate.get("name", "")), "error")


def _failed_evaluation_gates(
    gates: Sequence[Mapping[str, Any]], severities: Mapping[str, str]
) -> list[str]:
    out: list[str] = []
    for gate in gates:
        if gate.get("passed") or _effective_severity(gate, severities) != "error":
            continue
        message = str(gate.get("message") or "")
        detail = f": {message}" if message else ""
        out.append(f"evaluation gate {gate.get('name', '')} failed{detail}")
    return out


def provisional_reasons(
    model_gates: Sequence[ModelGate] | None,
    evaluation_reason: str,
    *,
    require_passing_diagnostics: bool = True,
    evaluation_gates: Sequence[Mapping[str, Any]] = (),
    evaluation_severities: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Every reason the outputs must be labelled provisional.

    Args:
        model_gates: The model run's gates, or None when its report is missing.
        evaluation_reason: Empty when a current evaluation report was read,
            otherwise why it was not (missing or stale).
        require_passing_diagnostics: Whether failed blocking model gates count.
        evaluation_gates: The evaluate stage's own gates; a failed one at error
            severity is a reason, because the evaluation it vouches for did not hold.
        evaluation_severities: Severity per gate name from the evaluate spec, used
            for gates whose report entry has none. A gate found in neither is
            treated as ``error``, so an unreadable spec blocks rather than passes.

    Returns:
        The reasons, empty when the report is publishable.
    """
    reasons: list[str] = []
    if evaluation_reason:
        reasons.append(evaluation_reason)
    reasons.extend(_failed_evaluation_gates(evaluation_gates, evaluation_severities or {}))
    if not require_passing_diagnostics:
        return tuple(reasons)
    if model_gates is None:
        reasons.append("model gate report is missing or unreadable; the model gates are unverified")
        return tuple(reasons)
    for gate in model_gates:
        blocking = gate.severity == "error" or gate.name in BLOCKING_MODEL_GATES
        if blocking and not gate.passed:
            detail = f": {gate.message}" if gate.message else ""
            reasons.append(f"model gate {gate.name} failed{detail}")
    return tuple(reasons)
