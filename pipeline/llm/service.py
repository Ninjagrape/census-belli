"""
The LLM entry point for pipeline stages.

A stage should not have to remember to hash its request, check whether the
work was already paid for, and write an audit row. :class:`LLMService` binds
a provider to a stage name and a database connection and does all three
around every call, so a stage runner's inner loop is one method call per
article.

The cache lookup is what makes the LLM stages idempotent. Re-running
``extract`` after a crash re-sends nothing it already completed
successfully: an identical prompt plus an identical model gives an identical
hash, and a hit returns the stored object without a round trip.
"""

from __future__ import annotations

from typing import Any

import structlog

from pipeline.llm.base import (
    CallStatus,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    TokenUsage,
    request_hash,
)
from pipeline.llm.call_log import find_completed_call, log_call
from pipeline.llm.factory import build_provider, llm_params

__all__ = ["LLMService", "StageUsage"]

logger = structlog.get_logger()


class StageUsage:
    """
    Running totals for one stage run.

    Mutable by design: this is a counter, and rebuilding it per call would
    make the totals awkward to read back at the end of a stage.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.cache_hits = 0
        self.failures = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        # True once any call returned a cost of None, meaning a model whose
        # pricing is not in the table. The dollar total is then a floor, not
        # a total, and the summary says so rather than quietly understating.
        self.cost_incomplete = False

    def record(self, response: LLMResponse, *, cached: bool) -> None:
        """
        Fold one call's outcome into the totals.

        Args:
            response: The call's result.
            cached: Whether it was served from the audit table rather than
                from the provider.
        """
        self.calls += 1
        if cached:
            self.cache_hits += 1
            return

        self.input_tokens += response.usage.input_tokens
        self.output_tokens += response.usage.output_tokens
        if response.cost_usd is None:
            self.cost_incomplete = True
        else:
            self.cost_usd += response.cost_usd
        if not response.ok:
            self.failures += 1

    def as_dict(self) -> dict[str, Any]:
        """
        Render the totals for logging.

        Returns:
            A flat mapping suitable for a structlog call.
        """
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "failures": self.failures,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 4),
            "cost_incomplete": self.cost_incomplete,
        }


class LLMService:
    """
    Cached, logged, provider-agnostic LLM access for one stage.

    Attributes:
        stage: The stage name recorded on every audit row.
        provider: The underlying client.
        usage: Running totals for this stage run.
    """

    def __init__(
        self,
        stage: str,
        provider: LLMProvider,
        db_conn: Any = None,
        *,
        default_temperature: float | None = None,
        default_max_tokens: int = 4096,
        use_cache: bool = True,
    ) -> None:
        """
        Initialise the service.

        Args:
            stage: Stage name, e.g. ``"extract"``.
            provider: A constructed provider client.
            db_conn: An open database connection. Without one, calls still
                run but nothing is cached or audited, which is correct for
                unit tests and a warning-worthy state in a real run.
            default_temperature: Temperature applied to requests that do not
                override it.
            default_max_tokens: Output ceiling for requests that do not
                override it.
            use_cache: Whether to check the audit table before calling. Set
                False to force genuine re-extraction, e.g. after editing a
                prompt in a way the hash cannot see.
        """
        self.stage = stage
        self.provider = provider
        self.usage = StageUsage()
        self._conn = db_conn
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens
        self._use_cache = use_cache

        if db_conn is None:
            logger.warning("llm_service_without_db_connection_no_audit_trail", stage=stage)

    @classmethod
    def from_spec(cls, spec: dict[str, Any], db_conn: Any = None, **kwargs: Any) -> LLMService:
        """
        Build a service straight from an agent spec.

        Args:
            spec: A loaded agent spec.
            db_conn: An open database connection.
            **kwargs: Overrides forwarded to :meth:`__init__`.

        Returns:
            A configured service for that stage.

        Raises:
            LLMConfigError: If the spec's provider or model is unusable.
        """
        config = llm_params(spec)
        return cls(
            stage=spec.get("stage", "unknown"),
            provider=build_provider(spec),
            db_conn=db_conn,
            default_temperature=config["temperature"],
            default_max_tokens=config["max_tokens"],
            **kwargs,
        )

    def complete(
        self,
        system: str,
        user: str,
        json_schema: dict[str, Any],
        *,
        schema_name: str = "extraction",
        metadata: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """
        Run one extraction, using the audit table as a cache.

        Args:
            system: The stage's system prompt.
            user: The passage to extract from.
            json_schema: Schema the response must satisfy.
            schema_name: Short name for the schema.
            metadata: Local context recorded with the call, e.g.
                ``{"battle_id": 412}``. Not sent to the provider.
            max_tokens: Override the stage default.
            temperature: Override the stage default.

        Returns:
            The outcome. A non-OK status means the caller should record the
            record in ``missing_data_log`` and continue rather than abort.
        """
        request = LLMRequest(
            system=system,
            user=user,
            json_schema=json_schema,
            schema_name=schema_name,
            max_tokens=max_tokens if max_tokens is not None else self._default_max_tokens,
            temperature=temperature if temperature is not None else self._default_temperature,
            metadata=metadata or {},
        )

        cached = self._lookup(request)
        if cached is not None:
            self.usage.record(cached, cached=True)
            return cached

        response = self.provider.complete(request)
        self.usage.record(response, cached=False)

        log_call(self._conn, self.stage, request, response)

        if not response.ok:
            logger.warning(
                "llm_extraction_needs_review",
                stage=self.stage,
                status=response.status.value,
                error=response.error,
                **request.metadata,
            )

        return response

    def log_summary(self) -> None:
        """Emit this stage's token and cost totals."""
        logger.info(
            "llm_stage_usage",
            stage=self.stage,
            provider=self.provider.name,
            model=self.provider.model,
            **self.usage.as_dict(),
        )

    # ─── Internals ───────────────────────────────────────────────────────────

    def _lookup(self, request: LLMRequest) -> LLMResponse | None:
        """
        Check the audit table for an identical completed call.

        Args:
            request: The request about to be sent.

        Returns:
            A response reconstructed from the stored row, or None to proceed
            with a live call. Token counts and cost come back zeroed, because
            those tokens were billed to the original call and counting them
            again would double the stage's reported spend.
        """
        if not self._use_cache:
            return None

        digest = request_hash(request, self.provider.name, self.provider.model)
        stored = find_completed_call(self._conn, digest)
        if stored is None:
            return None

        logger.debug("llm_cache_hit", stage=self.stage, input_hash=digest[:12])

        return LLMResponse(
            status=CallStatus.OK,
            provider=self.provider.name,
            model=self.provider.model,
            request_hash=digest,
            data=stored,
            usage=TokenUsage(),
            cost_usd=0.0,
            attempts=0,
        )
