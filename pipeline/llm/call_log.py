"""
Persistence for the ``llm_calls`` audit table.

Every LLM call in this project is logged with its input hash, output, model
version and token count. That serves three purposes beyond bookkeeping:

1. **Reproducibility.** A ranking derived from LLM-extracted data is only
   defensible if every extraction can be traced to a specific prompt, model
   version and response.
2. **Idempotency.** Stages must be re-runnable without corrupting downstream
   data. Looking up a request hash before calling means a re-run resumes
   rather than re-paying for work already done.
3. **Cost control.** Per-stage token and dollar totals come from aggregating
   this table.

Writes here must never take down a stage: a logging failure is reported and
swallowed, because losing an audit row is bad but losing a completed
extraction that was already paid for is worse.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

import structlog

from pipeline.llm.base import CallStatus, LLMRequest, LLMResponse

__all__ = ["find_completed_call", "log_call"]

logger = structlog.get_logger()

_INSERT_SQL = """
INSERT INTO llm_calls (
    stage, provider, model, input_hash, status,
    prompt_tokens, completion_tokens, cached_prompt_tokens,
    cost_usd, latency_ms, attempts,
    response_json, raw_text, error, battle_id
) VALUES (
    :stage, :provider, :model, :input_hash, :status,
    :prompt_tokens, :completion_tokens, :cached_prompt_tokens,
    :cost_usd, :latency_ms, :attempts,
    CAST(:response_json AS JSONB), :raw_text, :error, :battle_id
)
RETURNING call_id
"""

_LOOKUP_SQL = """
SELECT response_json
FROM llm_calls
WHERE input_hash = :input_hash
  AND status = 'ok'
  AND response_json IS NOT NULL
ORDER BY call_id DESC
LIMIT 1
"""


class _Connection(Protocol):
    """Minimal surface this module needs from a database connection."""

    def execute(self, statement: Any, parameters: Any = None) -> Any: ...


def _to_statement(sql: str) -> Any:
    """
    Wrap textual SQL for SQLAlchemy 2.x, passing it through if unavailable.

    Mirrors the helper in :mod:`pipeline.quality` so that test stubs taking
    raw strings work the same way in both modules.
    """
    try:
        from sqlalchemy import text
    except ImportError:
        logger.debug("sqlalchemy_unavailable_using_raw_sql")
        return sql

    return text(sql)


def find_completed_call(conn: Any, input_hash: str) -> dict[str, Any] | None:
    """
    Look for a successful earlier call with this exact input.

    Args:
        conn: An open database connection, or None to skip the lookup.
        input_hash: The request hash from :func:`pipeline.llm.base.request_hash`.

    Returns:
        The stored response object, or None if there is no successful prior
        call, no connection, or the lookup failed. A None return always means
        "call the provider", so a broken cache costs money but not
        correctness.
    """
    if conn is None:
        return None

    try:
        row = conn.execute(_to_statement(_LOOKUP_SQL), {"input_hash": input_hash}).fetchone()
    except Exception as e:
        logger.warning("llm_call_cache_lookup_failed", error=str(e), input_hash=input_hash[:12])
        return None

    if row is None:
        return None

    stored = row[0]
    # psycopg returns JSONB as a dict; a stub or text column may return a string.
    if isinstance(stored, str):
        try:
            return json.loads(stored)
        except json.JSONDecodeError:
            logger.warning("llm_call_cache_row_unparseable", input_hash=input_hash[:12])
            return None
    return stored if isinstance(stored, dict) else None


def log_call(
    conn: Any,
    stage: str,
    request: LLMRequest,
    response: LLMResponse,
) -> int | None:
    """
    Record one call in ``llm_calls``.

    Args:
        conn: An open database connection, or None to skip persistence.
        stage: Pipeline stage that issued the call, e.g. ``"extract"``.
        request: The request that was sent; supplies ``metadata.battle_id``.
        response: The outcome, successful or not.

    Returns:
        The new ``call_id``, or None if nothing was written.
    """
    if conn is None:
        logger.debug("llm_call_not_logged_no_connection", stage=stage)
        return None

    battle_id = request.metadata.get("battle_id")

    params = {
        "stage": stage,
        "provider": response.provider,
        "model": response.model,
        "input_hash": response.request_hash,
        "status": response.status.value,
        "prompt_tokens": response.usage.input_tokens,
        "completion_tokens": response.usage.output_tokens,
        "cached_prompt_tokens": response.usage.cached_input_tokens,
        "cost_usd": response.cost_usd,
        "latency_ms": response.latency_ms,
        "attempts": response.attempts,
        "response_json": json.dumps(response.data) if response.data is not None else None,
        # Keep the raw body only when the parse failed; on success it is just
        # a second copy of response_json in a table with a row per article.
        "raw_text": response.raw_text if response.status is not CallStatus.OK else None,
        "error": response.error,
        "battle_id": int(battle_id) if battle_id is not None else None,
    }

    try:
        row = conn.execute(_to_statement(_INSERT_SQL), params).fetchone()
    except Exception as e:
        logger.error(
            "llm_call_logging_failed",
            stage=stage,
            model=response.model,
            status=response.status.value,
            error=str(e),
        )
        return None

    return int(row[0]) if row else None
