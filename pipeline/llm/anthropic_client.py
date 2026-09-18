"""
Anthropic provider for structured extraction.

Uses the Messages API's structured output constraint
(``output_config.format``) rather than forced tool use, so the response body
is JSON directly and no tool-call unwrapping is needed.

Two details of the current API shape matter here and are easy to get wrong:

* **Sampling parameters were removed on the newer frontier models.** Sending
  ``temperature`` to Opus 4.7+, Sonnet 5, or the Fable family returns a 400
  rather than being ignored, so the parameter is gated on
  :func:`pipeline.llm.capabilities.accepts_temperature`.
* **Schemas must be closed.** ``output_config.format`` expects a schema with
  ``required`` and ``additionalProperties: false``. The schemas in
  ``agents/<stage>.yaml`` are passed through unmodified, so a spec missing
  those keys fails loudly at the first call rather than silently accepting
  extra fields.
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

__all__ = ["AnthropicClient"]

logger = structlog.get_logger()

_API_KEY_ENV = "ANTHROPIC_API_KEY"


class _PermanentCallError(Exception):
    """A request-specific failure that retrying cannot fix."""


class AnthropicClient:
    """
    Structured-extraction client for the Anthropic Messages API.

    Satisfies :class:`pipeline.llm.base.LLMProvider`.

    Attributes:
        name: Always ``"anthropic"``.
        model: The exact model identifier requests are sent to.
    """

    name = "anthropic"

    def __init__(
        self,
        model: str,
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_delay: float = DEFAULT_BASE_DELAY,
        cache_system_prompt: bool = True,
        effort: str | None = None,
        enable_thinking: bool = False,
    ) -> None:
        """
        Initialise the client.

        Args:
            model: Exact model identifier, e.g. ``"claude-sonnet-5"``.
            max_retries: Retries after the initial attempt for transient failures.
            base_delay: Seconds for the first backoff, doubling thereafter.
            cache_system_prompt: Mark the system prompt for prompt caching.
                The stage prompt and schema are identical across every article
                in a batch, so caching them turns the largest stable part of
                each request into a cache read. Has no effect if the prefix is
                below the model's minimum cacheable length.
            effort: Optional ``output_config.effort`` level. Leave unset for
                mechanical extraction; raise it for judgement-heavy stages
                such as command attribution.
            enable_thinking: Request adaptive thinking. Off by default because
                extraction does not benefit from it and it is billed.

        Raises:
            LLMConfigError: If the SDK is not installed.
        """
        self.model = model
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._cache_system_prompt = cache_system_prompt
        self._effort = effort
        self._enable_thinking = enable_thinking

        try:
            import anthropic
        except ImportError as e:
            raise LLMConfigError(
                "The anthropic package is required for the Anthropic provider. "
                "Install it with: pip install anthropic>=1.0"
            ) from e

        self._sdk = anthropic
        # max_retries=0 so that pipeline.llm.retry is the only retry policy in
        # play. Layering the SDK's own retries underneath would multiply the
        # round trips and make the attempt count recorded in llm_calls a lie.
        self._client = anthropic.Anthropic(max_retries=0)

        if not os.environ.get(_API_KEY_ENV):
            # The SDK also accepts ANTHROPIC_AUTH_TOKEN and an `ant auth login`
            # profile, so an unset key is not proof of missing credentials.
            logger.info("anthropic_api_key_env_unset_relying_on_sdk_resolution")

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

        Args:
            request: The request to send.

        Returns:
            The raw SDK ``Message`` object.

        Raises:
            TransientLLMError: For rate limits, timeouts, connection errors
                and 5xx responses.
            LLMConfigError: For authentication, permission and unknown-model
                failures.
            _PermanentCallError: For request-specific 4xx failures.
        """
        sdk = self._sdk
        try:
            return self._client.messages.create(**self._build_params(request))
        except (sdk.AuthenticationError, sdk.PermissionDeniedError) as e:
            raise LLMConfigError(f"Anthropic credentials rejected: {e}") from e
        except sdk.NotFoundError as e:
            raise LLMConfigError(f"Unknown Anthropic model {self.model}: {e}") from e
        except sdk.RateLimitError as e:
            raise TransientLLMError(f"rate limited: {e}") from e
        except sdk.APITimeoutError as e:
            raise TransientLLMError(f"request timed out: {e}") from e
        except sdk.APIConnectionError as e:
            raise TransientLLMError(f"connection error: {e}") from e
        except sdk.APIStatusError as e:
            if e.status_code >= 500:
                raise TransientLLMError(f"server error {e.status_code}: {e}") from e
            # A 400 here is usually passage-specific (over-long input, or a
            # schema the spec declared badly). Permanent for this record, but
            # not a reason to abandon the batch.
            raise _PermanentCallError(f"request rejected ({e.status_code}): {e}") from e

    def _build_params(self, request: LLMRequest) -> dict[str, Any]:
        """
        Assemble the keyword arguments for one ``messages.create`` call.

        Args:
            request: The request being sent.

        Returns:
            Keyword arguments ready to splat into the SDK call.
        """
        system: Any = request.system
        if self._cache_system_prompt:
            system = [
                {
                    "type": "text",
                    "text": request.system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        output_config: dict[str, Any] = {
            "format": {"type": "json_schema", "schema": request.json_schema}
        }
        if self._effort is not None:
            output_config["effort"] = self._effort

        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": request.user}],
            "output_config": output_config,
        }

        if self._enable_thinking:
            params["thinking"] = {"type": "adaptive"}

        if request.temperature is not None and accepts_temperature(self.name, self.model):
            params["temperature"] = request.temperature
        elif request.temperature is not None:
            logger.debug(
                "temperature_omitted_model_rejects_sampling_params",
                model=self.model,
                requested=request.temperature,
            )

        return params

    # ─── Response interpretation ─────────────────────────────────────────────

    def _interpret(
        self, request: LLMRequest, raw: Any, digest: str, started: float, attempts: int
    ) -> LLMResponse:
        """
        Turn an SDK message into an :class:`LLMResponse`.

        Args:
            request: The request that produced this message.
            raw: The SDK ``Message`` object.
            digest: Precomputed request hash.
            started: Monotonic clock reading from before the first attempt.
            attempts: Number of round trips made.

        Returns:
            The interpreted outcome.
        """
        usage = _read_usage(raw)
        text = "".join(
            block.text for block in raw.content if getattr(block, "type", None) == "text"
        )
        stop_reason = getattr(raw, "stop_reason", None)

        if stop_reason == "refusal":
            details = getattr(raw, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            return self._build(
                digest,
                CallStatus.REFUSAL,
                started,
                attempts,
                usage,
                raw_text=text,
                error=f"provider refused the request (category: {category})",
            )

        if stop_reason == "max_tokens":
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


def _read_usage(raw: Any) -> TokenUsage:
    """
    Extract token counts from an SDK message.

    Reads defensively: usage fields have been added over time, and a missing
    one should cost accuracy in the cost column, not raise mid-batch.

    Args:
        raw: The SDK ``Message`` object.

    Returns:
        Token counts, zeroed where the provider reported nothing.
    """
    usage = getattr(raw, "usage", None)
    if usage is None:
        return TokenUsage()

    return TokenUsage(
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
        cached_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
    )
