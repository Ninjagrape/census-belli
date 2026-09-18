"""
Build an LLM provider from a stage's agent spec.

Stages declare which provider and model they want in ``agents/<stage>.yaml``
under ``params``, so switching a stage between vendors is a config edit
rather than a code change:

.. code-block:: yaml

    params:
      llm_provider: gemini
      llm_model: gemini-3.8-flash
      llm_temperature: 0.0
      llm_max_tokens: 4096

``llm_temperature`` is honoured only where the model accepts it. Anthropic
removed sampling parameters on its newer frontier models, so a spec asking
for temperature 0 on one of those gets its determinism from the structured
output constraint instead, and the parameter is dropped with a debug log
rather than causing a 400 on every call in the batch.
"""

from __future__ import annotations

from typing import Any

import structlog

from pipeline.llm.anthropic_client import AnthropicClient
from pipeline.llm.base import LLMConfigError, LLMProvider
from pipeline.llm.gemini_client import GeminiClient
from pipeline.llm.retry import DEFAULT_BASE_DELAY, DEFAULT_MAX_RETRIES

__all__ = ["PROVIDER_NAMES", "build_provider", "llm_params"]

logger = structlog.get_logger()

PROVIDER_NAMES = ("anthropic", "gemini")

# Used when a spec names no provider. Anthropic is the conservative default:
# every stage's prompts were written and reviewed against it, so an omitted
# provider cannot silently reroute a stage to a different model family.
_DEFAULT_PROVIDER = "anthropic"


def llm_params(spec: dict[str, Any]) -> dict[str, Any]:
    """
    Extract the LLM-related parameters from an agent spec.

    Args:
        spec: A loaded agent spec, as returned by
            :func:`pipeline.orchestrator.load_agent_spec`.

    Returns:
        The subset of ``params`` this module understands, with defaults
        filled in. ``temperature`` and ``max_tokens`` come back for the
        caller to put on each :class:`~pipeline.llm.base.LLMRequest`, since
        they belong to a call rather than to a client.
    """
    params = spec.get("params", {}) or {}

    return {
        "provider": params.get("llm_provider", _DEFAULT_PROVIDER),
        "model": params.get("llm_model"),
        "temperature": params.get("llm_temperature"),
        "max_tokens": params.get("llm_max_tokens", 4096),
        "effort": params.get("llm_effort"),
        "thinking": bool(params.get("llm_thinking", False)),
        "max_retries": params.get("llm_max_retries", DEFAULT_MAX_RETRIES),
        "base_delay": params.get("llm_backoff_base", DEFAULT_BASE_DELAY),
    }


def build_provider(spec: dict[str, Any]) -> LLMProvider:
    """
    Construct the provider a stage spec asks for.

    Args:
        spec: A loaded agent spec.

    Returns:
        A ready client satisfying :class:`~pipeline.llm.base.LLMProvider`.

    Raises:
        LLMConfigError: If the provider name is unknown, no model is named,
            the provider's SDK is missing, or its credentials are absent.
            Each of these would fail identically on every call, so they halt
            the stage rather than degrading.
    """
    config = llm_params(spec)
    provider = config["provider"]
    model = config["model"]
    stage = spec.get("stage", "unknown")

    if provider not in PROVIDER_NAMES:
        raise LLMConfigError(
            f"Stage {stage} names unknown llm_provider {provider!r}. "
            f"Valid providers: {', '.join(PROVIDER_NAMES)}"
        )

    if not model:
        raise LLMConfigError(
            f"Stage {stage} sets llm_provider {provider!r} but no llm_model. "
            "Name an exact model identifier; there is no default, because a "
            "wrong guess here would be billed on every article in the batch."
        )

    logger.info("llm_provider_selected", stage=stage, provider=provider, model=model)

    if provider == "gemini":
        return GeminiClient(
            model=model,
            max_retries=config["max_retries"],
            base_delay=config["base_delay"],
            thinking_level=config["effort"],
        )

    return AnthropicClient(
        model=model,
        max_retries=config["max_retries"],
        base_delay=config["base_delay"],
        effort=config["effort"],
        enable_thinking=config["thinking"],
    )
