"""
Provider-agnostic types for LLM calls in the General WAR pipeline.

The extract, resolve and classify stages all need the same thing: send a
system prompt plus one passage, get back JSON matching a declared schema,
and record what the call cost. Those stages should not care which vendor
served the request, so they depend on :class:`LLMProvider` and never on a
vendor SDK directly.

Failure policy follows the project's error-handling rules: a *configuration*
problem (missing API key, unknown provider) raises immediately, because it
is a bug in the run rather than a property of the data. A *call* problem
(rate limit, malformed JSON, safety refusal) does not raise. It comes back
as an :class:`LLMResponse` with a non-OK :class:`CallStatus`, so the caller
can log it, flag the record for manual review, and carry on with the batch.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

__all__ = [
    "CallStatus",
    "LLMConfigError",
    "LLMError",
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "TokenUsage",
    "request_hash",
]


# ─── Errors ──────────────────────────────────────────────────────────────────


class LLMError(Exception):
    """Base class for LLM layer errors."""


class LLMConfigError(LLMError):
    """
    Raised when the layer is misconfigured.

    Missing credentials, an unknown provider name, or a stage spec that asks
    for a model the provider does not serve. These halt the run rather than
    degrading, because every subsequent call would fail the same way.
    """


# ─── Call outcome ────────────────────────────────────────────────────────────


class CallStatus(Enum):
    """Outcome of a single LLM call."""

    OK = "ok"
    # Response arrived but was not valid JSON, or did not match the schema.
    PARSE_ERROR = "parse_error"
    # Provider declined the request on safety grounds.
    REFUSAL = "refusal"
    # Transport or server-side failure that survived every retry.
    API_ERROR = "api_error"
    # Response was cut off by the output token ceiling, so the JSON is partial.
    TRUNCATED = "truncated"

    @property
    def needs_review(self) -> bool:
        """Whether a record produced by this call should be flagged for a human."""
        return self is not CallStatus.OK


# ─── Request ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LLMRequest:
    """
    One structured-extraction request.

    Attributes:
        system: The stage's system prompt, taken from ``agents/<stage>.yaml``.
            Held stable across a batch so providers can cache the prefix.
        user: The passage being extracted from.
        json_schema: JSON Schema the response must satisfy. Providers enforce
            this natively where they can, and it is validated locally either way.
        schema_name: Short name for the schema, used by providers that require
            a named output format.
        max_tokens: Ceiling on output tokens.
        temperature: Sampling temperature, or None to leave it unset. Current
            frontier models reject sampling parameters outright, so this stays
            optional rather than defaulting to 0.0. See
            :func:`pipeline.llm.capabilities.accepts_temperature`.
        metadata: Free-form context recorded alongside the call, e.g.
            ``{"battle_id": 412, "source_id": 9}``. Never sent to the provider.
    """

    system: str
    user: str
    json_schema: dict[str, Any]
    schema_name: str = "extraction"
    max_tokens: int = 4096
    temperature: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ─── Usage and response ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class TokenUsage:
    """
    Token counts for one call.

    ``cached_input_tokens`` is the subset of ``input_tokens`` served from a
    provider-side prompt cache. It is billed at a steep discount, so cost
    calculation needs it separately rather than folded into the input total.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Input plus output, for rough volume reporting."""
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class LLMResponse:
    """
    Result of one call, successful or not.

    ``data`` is populated only when ``status`` is OK. On any other status the
    caller should log the failure and flag the affected record, per the
    project's extraction-failure rule.
    """

    status: CallStatus
    provider: str
    model: str
    request_hash: str
    data: dict[str, Any] | None = None
    raw_text: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)
    cost_usd: float | None = None
    latency_ms: int = 0
    attempts: int = 1
    error: str | None = None

    @property
    def ok(self) -> bool:
        """Whether usable structured data came back."""
        return self.status is CallStatus.OK and self.data is not None


# ─── Provider interface ──────────────────────────────────────────────────────


class LLMProvider(Protocol):
    """
    What the pipeline stages depend on.

    Implementations must not raise for call-level failures; they return a
    non-OK :class:`LLMResponse` instead. Only :class:`LLMConfigError` should
    ever escape :meth:`complete`.
    """

    name: str
    model: str

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Send one request and return its outcome."""
        ...


# ─── Request identity ────────────────────────────────────────────────────────


def request_hash(request: LLMRequest, provider: str, model: str) -> str:
    """
    Compute a stable hash identifying this request.

    Two runs that would send an identical prompt to an identical model produce
    the same hash, which is what makes the LLM stages idempotent: a stage can
    look up the hash in ``llm_calls`` and skip work it has already paid for.
    ``metadata`` is excluded because it is local bookkeeping, not input.

    Args:
        request: The request being sent.
        provider: Provider name, e.g. ``"anthropic"``.
        model: Exact model identifier.

    Returns:
        A hex SHA-256 digest.
    """
    payload = {
        "provider": provider,
        "model": model,
        "system": request.system,
        "user": request.user,
        # sort_keys so that dict ordering never changes the digest
        "schema": json.dumps(request.json_schema, sort_keys=True),
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
