"""
Gemini provider for structured extraction.

This is the intended workhorse for the high-volume, low-judgement stages.
Flash-tier Gemini runs roughly four to six times cheaper per token than the
Anthropic models this pipeline would otherwise use for extraction, which
matters when the unit of work is every battle article on Wikipedia.

Two notes on the SDK surface, both verified against Google's published docs
on 2026-09-18 rather than recalled:

* The current API is ``client.interactions.create``, which takes ``input``
  and ``response_format``. The older ``client.models.generate_content`` with
  a ``GenerateContentConfig`` is the legacy path and is not used here.
* Usage is reported on ``interaction.usage`` with ``total_*`` field names,
  not on a ``usage_metadata`` object with the legacy ``*_token_count`` names.

Failure detection is deliberately conservative. The exact fields Gemini uses
to signal a truncated or safety-blocked response were not confirmable from
the docs, so this client reads those defensively and otherwise leans on the
parse step: a blocked or cut-off response fails schema validation and is
recorded as a non-OK call rather than being mistaken for a success. Exercise
the refusal and truncation branches against a live key before trusting them.
"""

from __future__ import annotations

import os
import time
from typing import Any

import structlog

from pipeline.llm.base import (
    CallStatus,
    LLMConfigError,
    LLMRequest,
    LLMResponse,
    TokenUsage,
    request_hash,
)
from pipeline.llm.capabilities import accepts_temperature, estimate_cost
from pipeline.llm.parsing import parse_structured
from pipeline.llm.retry import (
    DEFAULT_BASE_DELAY,
    DEFAULT_MAX_RETRIES,
    TransientLLMError,
    with_backoff,
)

__all__ = ["GeminiClient"]

logger = structlog.get_logger()

_API_KEY_ENV = "GEMINI_API_KEY"

# Statuses that mean "try again": request timeout, conflict, rate limit.
_TRANSIENT_STATUSES = frozenset({408, 409, 429})
# Statuses that mean the credentials or the model name are wrong, which every
# subsequent call in the batch would hit identically.
_CONFIG_STATUSES = frozenset({401, 403, 404})


class _PermanentCallError(Exception):
    """A request-specific failure that retrying cannot fix."""


class GeminiClient:
    """
    Structured-extraction client for the Gemini Interactions API.

    Satisfies :class:`pipeline.llm.base.LLMProvider`.

    Attributes:
        name: Always ``"gemini"``.
        model: The exact model identifier requests are sent to.
    """

    name = "gemini"

    def __init__(
        self,
        model: str,
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_delay: float = DEFAULT_BASE_DELAY,
        thinking_level: str | None = None,
    ) -> None:
        """
        Initialise the client.

        Args:
            model: Exact model identifier, e.g. ``"gemini-3.8-flash"``.
            max_retries: Retries after the initial attempt for transient failures.
            base_delay: Seconds for the first backoff, doubling thereafter.
            thinking_level: Optional reasoning depth passed through in
                ``generation_config``. Leave unset for mechanical extraction.

        Raises:
            LLMConfigError: If the SDK is not installed or no API key is set.
        """
        self.model = model
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._thinking_level = thinking_level

        try:
            from google import genai
        except ImportError as e:
            raise LLMConfigError(
                "The google-genai package is required for the Gemini provider. "
                "Install it with: pip install google-genai>=2.3"
            ) from e

        api_key = os.environ.get(_API_KEY_ENV)
        if not api_key:
            # Unlike the Anthropic SDK there is no profile-on-disk fallback to
            # defer to, so an unset key here is definitive. Fail now rather
            # than once per article.
            raise LLMConfigError(
                f"{_API_KEY_ENV} is not set. Add it to .env (which is gitignored) "
                "as GEMINI_API_KEY=... using a key from https://aistudio.google.com/apikey"
            )

        self._client = genai.Client(api_key=api_key)

    def complete(self, request: LLMRequest) -> LLMResponse:
        """
        Send one structured-extraction request.

        Args:
            request: The prompt, schema and limits for this call.

        Returns:
            The outcome. Call-level failures come back as a non-OK status
            rather than an exception, so a batch survives a bad article.

        Raises:
            LLMConfigError: On authentication, permission, or unknown-model
                failure, which would affect every subsequent call equally.
        """
        digest = request_hash(request, self.name, self.model)
        started = time.monotonic()

        try:
            raw, attempts = with_backoff(
                lambda: self._send(request),
                max_retries=self._max_retries,
                base_delay=self._base_delay,
                context={"provider": self.name, "model": self.model, **request.metadata},
            )
        except TransientLLMError as e:
            return self._failed(
                request, digest, CallStatus.API_ERROR, str(e), started, self._max_retries + 1
            )
        except _PermanentCallError as e:
            return self._failed(request, digest, CallStatus.API_ERROR, str(e), started, 1)

        return self._interpret(request, raw, digest, started, attempts)

    # ─── Request construction ────────────────────────────────────────────────

    def _send(self, request: LLMRequest) -> Any:
        """
        Perform one API round trip, translating SDK errors into our taxonomy.

        The SDK's exception classes are not imported by name. Classification
        works off the HTTP status carried on the exception, so a change to the
        SDK's error hierarchy degrades to "retry the transient ones, surface
        the rest" instead of breaking the import.

        Args:
            request: The request to send.

        Returns:
            The raw SDK interaction object.

        Raises:
            TransientLLMError: For rate limits, timeouts and 5xx responses.
            LLMConfigError: For authentication, permission and unknown-model
                failures.
            _PermanentCallError: For request-specific 4xx failures.
        """
        try:
            return self._client.interactions.create(**self._build_params(request))
        except Exception as e:
            status = _status_code(e)

            if status in _CONFIG_STATUSES:
                raise LLMConfigError(
                    f"Gemini rejected the request ({status}); check GEMINI_API_KEY "
                    f"and that {self.model} is a current model identifier: {e}"
                ) from e
            if status is None:
                # No HTTP status at all is typically a transport failure.
                raise TransientLLMError(f"transport error: {e}") from e
            if status in _TRANSIENT_STATUSES or status >= 500:
                raise TransientLLMError(f"retryable error {status}: {e}") from e

            raise _PermanentCallError(f"request rejected ({status}): {e}") from e

    def _build_params(self, request: LLMRequest) -> dict[str, Any]:
        """
        Assemble the keyword arguments for one ``interactions.create`` call.

        Args:
            request: The request being sent.

        Returns:
            Keyword arguments ready to splat into the SDK call.
        """
        generation_config: dict[str, Any] = {"max_output_tokens": request.max_tokens}

        if self._thinking_level is not None:
            generation_config["thinking_level"] = self._thinking_level

        if request.temperature is not None and accepts_temperature(self.name, self.model):
            generation_config["temperature"] = request.temperature

        return {
            "model": self.model,
            "input": request.user,
            "system_instruction": request.system,
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": request.json_schema,
            },
            "generation_config": generation_config,
        }

    # ─── Response interpretation ─────────────────────────────────────────────

    def _interpret(
        self, request: LLMRequest, raw: Any, digest: str, started: float, attempts: int
    ) -> LLMResponse:
        """
        Turn an SDK interaction into an :class:`LLMResponse`.

        Args:
            request: The request that produced this interaction.
            raw: The SDK interaction object.
            digest: Precomputed request hash.
            started: Monotonic clock reading from before the first attempt.
            attempts: Number of round trips made.

        Returns:
            The interpreted outcome.
        """
        usage = _read_usage(raw)
        text = getattr(raw, "output_text", "") or ""

        blocked_reason = _blocked_reason(raw)
        if blocked_reason is not None:
            return self._build(
                digest,
                CallStatus.REFUSAL,
                started,
                attempts,
                usage,
                raw_text=text,
                error=f"provider blocked the request ({blocked_reason})",
            )

        if _is_truncated(raw):
            return self._build(
                digest,
                CallStatus.TRUNCATED,
                started,
                attempts,
                usage,
                raw_text=text,
                error=f"response hit the {request.max_tokens} output token ceiling",
            )

        data, parse_error = parse_structured(text, request.json_schema)
        if parse_error is not None:
            return self._build(
                digest,
                CallStatus.PARSE_ERROR,
                started,
                attempts,
                usage,
                raw_text=text,
                error=parse_error,
            )

        return self._build(
            digest, CallStatus.OK, started, attempts, usage, data=data, raw_text=text
        )

    def _build(
        self,
        digest: str,
        status: CallStatus,
        started: float,
        attempts: int,
        usage: TokenUsage,
        *,
        data: dict[str, Any] | None = None,
        raw_text: str = "",
        error: str | None = None,
    ) -> LLMResponse:
        """
        Construct a response, costing the tokens actually consumed.

        Args:
            digest: The request hash.
            status: Outcome of the call.
            started: Monotonic clock reading from before the first attempt.
            attempts: Number of round trips made.
            usage: Token counts reported by the provider.
            data: Parsed object, when the call succeeded.
            raw_text: Raw response body.
            error: Failure description, when the call did not succeed.

        Returns:
            The assembled response.
        """
        return LLMResponse(
            status=status,
            provider=self.name,
            model=self.model,
            request_hash=digest,
            data=data,
            raw_text=raw_text,
            usage=usage,
            cost_usd=estimate_cost(
                self.name,
                self.model,
                usage.input_tokens,
                usage.output_tokens,
                usage.cached_input_tokens,
                usage.cache_write_tokens,
            ),
            latency_ms=int((time.monotonic() - started) * 1000),
            attempts=attempts,
            error=error,
        )

    def _failed(
        self,
        request: LLMRequest,
        digest: str,
        status: CallStatus,
        error: str,
        started: float,
        attempts: int,
    ) -> LLMResponse:
        """
        Build a response for a call that never returned a body.

        Args:
            request: The request that failed; supplies log context.
            digest: The request hash.
            status: Failure status to record.
            error: Failure description.
            started: Monotonic clock reading from before the first attempt.
            attempts: Number of round trips made.

        Returns:
            A response carrying the failure.
        """
        logger.error(
            "llm_call_failed",
            provider=self.name,
            model=self.model,
            status=status.value,
            error=error,
            **request.metadata,
        )
        return LLMResponse(
            status=status,
            provider=self.name,
            model=self.model,
            request_hash=digest,
            latency_ms=int((time.monotonic() - started) * 1000),
            attempts=attempts,
            error=error,
        )


# ─── Defensive response readers ──────────────────────────────────────────────


def _status_code(error: Exception) -> int | None:
    """
    Pull an HTTP status code off an SDK exception.

    Args:
        error: The exception raised by the SDK.

    Returns:
        The status code, or None if the exception carries none.
    """
    for attribute in ("code", "status_code", "status"):
        value = getattr(error, attribute, None)
        if isinstance(value, int):
            return value

    response = getattr(error, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def _blocked_reason(raw: Any) -> str | None:
    """
    Detect a safety-blocked response.

    Reads several plausible field names because the exact signalling field is
    unconfirmed. An unrecognised block still fails schema validation
    downstream and is recorded as a failure, so a miss here costs a less
    precise error message rather than a false success.

    Args:
        raw: The SDK interaction object.

    Returns:
        A short description of the block, or None if not blocked.
    """
    feedback = getattr(raw, "prompt_feedback", None)
    reason = getattr(feedback, "block_reason", None) if feedback else None
    if reason:
        return str(reason)

    details = getattr(raw, "incomplete_details", None)
    reason = getattr(details, "reason", None) if details else None
    if reason and "safet" in str(reason).lower():
        return str(reason)

    return None


def _is_truncated(raw: Any) -> bool:
    """
    Detect a response cut off by the output token ceiling.

    Args:
        raw: The SDK interaction object.

    Returns:
        True if the response was truncated rather than completed.
    """
    details = getattr(raw, "incomplete_details", None)
    reason = getattr(details, "reason", None) if details else None
    if reason and "max" in str(reason).lower():
        return True

    finish_reason = getattr(raw, "finish_reason", None)
    return bool(finish_reason) and "max" in str(finish_reason).lower()


def _read_usage(raw: Any) -> TokenUsage:
    """
    Extract token counts from an SDK interaction.

    Thinking tokens are folded into the output total because they are billed
    at the output rate and this project does not report them separately.

    Args:
        raw: The SDK interaction object.

    Returns:
        Token counts, zeroed where the provider reported nothing.
    """
    usage = getattr(raw, "usage", None)
    if usage is None:
        return TokenUsage()

    thought_tokens = getattr(usage, "total_thought_tokens", 0) or 0

    return TokenUsage(
        input_tokens=getattr(usage, "total_input_tokens", 0) or 0,
        output_tokens=(getattr(usage, "total_output_tokens", 0) or 0) + thought_tokens,
        cached_input_tokens=getattr(usage, "total_cached_tokens", 0) or 0,
    )
