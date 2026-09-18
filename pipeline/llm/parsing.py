"""
Parse and validate structured output from an LLM.

Both providers can constrain generation to a JSON Schema, but neither
guarantee survives every failure mode: a response truncated at the output
ceiling is cut mid-object, and a refusal returns prose. Validating locally
gives every provider one uniform failure mode instead of two vendor-shaped
ones, and keeps the schema in ``agents/<stage>.yaml`` as the single source of
truth for what a valid extraction looks like.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

__all__ = ["parse_structured", "strip_code_fence"]

logger = structlog.get_logger()


def strip_code_fence(text: str) -> str:
    """
    Remove a Markdown code fence wrapping a JSON payload.

    Schema-constrained responses arrive as bare JSON, but a model that falls
    back to prose often wraps the object in ```json ... ```. Unwrapping costs
    nothing and salvages calls that would otherwise be logged as failures.

    Args:
        text: Raw response text.

    Returns:
        The text with a surrounding fence removed, stripped of whitespace.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    # Drop the opening fence, which may carry a language tag, and any closing one.
    body = lines[1:]
    if body and body[-1].strip().startswith("```"):
        body = body[:-1]
    return "\n".join(body).strip()


def parse_structured(
    text: str, json_schema: dict[str, Any]
) -> tuple[dict[str, Any] | None, str | None]:
    """
    Parse response text into a schema-valid object.

    Args:
        text: Raw response text from the provider.
        json_schema: The schema the object must satisfy.

    Returns:
        A ``(data, error)`` pair. On success ``error`` is None; on failure
        ``data`` is None and ``error`` carries a message suitable for the
        ``llm_calls.error`` column. Never raises, so one malformed response
        cannot abort a batch.
    """
    if not text.strip():
        return None, "empty response body"

    try:
        parsed = json.loads(strip_code_fence(text))
    except json.JSONDecodeError as e:
        return None, f"invalid JSON: {e}"

    if not isinstance(parsed, dict):
        return None, f"expected a JSON object, got {type(parsed).__name__}"

    error = _validate(parsed, json_schema)
    if error is not None:
        return None, error

    return parsed, None


def _validate(data: dict[str, Any], json_schema: dict[str, Any]) -> str | None:
    """
    Check an object against a JSON Schema.

    Falls back to a required-keys check when ``jsonschema`` is not installed,
    so that the layer degrades rather than failing closed in a bare
    environment. The fallback is weaker, and says so in the log.

    Args:
        data: The parsed object.
        json_schema: Schema to validate against.

    Returns:
        None if valid, otherwise a description of the first violation.
    """
    try:
        import jsonschema
    except ImportError:
        logger.warning("jsonschema_unavailable_using_required_keys_check")
        missing = [k for k in json_schema.get("required", []) if k not in data]
        if missing:
            return f"missing required field(s): {', '.join(missing)}"
        return None

    try:
        jsonschema.validate(instance=data, schema=json_schema)
    except jsonschema.ValidationError as e:
        location = "/".join(str(p) for p in e.absolute_path) or "(root)"
        return f"schema violation at {location}: {e.message}"
    except jsonschema.SchemaError as e:
        # A broken schema is a configuration bug, but raising here would abort
        # a batch mid-flight. Surface it loudly as a call failure instead.
        logger.error("invalid_json_schema_in_agent_spec", error=str(e))
        return f"invalid schema in agent spec: {e.message}"

    return None
