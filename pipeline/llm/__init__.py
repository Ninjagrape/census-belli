"""
Provider-agnostic LLM access for the General WAR pipeline.

Stages import from this package rather than from a vendor SDK, so that which
model serves a stage is a decision recorded in ``agents/<stage>.yaml`` and
audited in ``llm_calls``, not a fact buried in stage code.

Typical use in a stage runner::

    from pipeline.llm import LLMService

    service = LLMService.from_spec(spec, db_conn=conn)
    for article in articles:
        result = service.complete(
            system=spec["prompt"]["system"],
            user=article.text,
            json_schema=schema,
            metadata={"battle_id": article.battle_id},
        )
        if not result.ok:
            continue  # already logged and flagged for review
        upsert(result.data)
    service.log_summary()
"""

from __future__ import annotations

from pipeline.llm.base import (
    CallStatus,
    LLMConfigError,
    LLMError,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    TokenUsage,
    request_hash,
)
from pipeline.llm.capabilities import accepts_temperature, estimate_cost, traits_for
from pipeline.llm.factory import PROVIDER_NAMES, build_provider, llm_params
from pipeline.llm.offline import (
    ExportResult,
    ImportResult,
    export_pending,
    import_responses,
    pending_count,
)
from pipeline.llm.service import LLMService, StageUsage

__all__ = [
    "ExportResult",
    "ImportResult",
    "PROVIDER_NAMES",
    "CallStatus",
    "LLMConfigError",
    "LLMError",
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "LLMService",
    "StageUsage",
    "TokenUsage",
    "accepts_temperature",
    "build_provider",
    "estimate_cost",
    "export_pending",
    "import_responses",
    "llm_params",
    "pending_count",
    "request_hash",
    "traits_for",
]
