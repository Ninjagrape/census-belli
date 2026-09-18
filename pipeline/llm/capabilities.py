"""
Per-model traits that change the shape of a request.

Kept in one place because these are facts about vendor models, not decisions
this project makes, and because getting them wrong produces an HTTP 400 on
every call rather than a degraded result.

The pricing and trait tables below were current as of 2026-09-18. They are
vendor-controlled and drift, so :func:`estimate_cost` reports ``None`` for a
model it does not recognise instead of guessing a number that would land in
the cost column of ``llm_calls`` looking authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

__all__ = [
    "ModelTraits",
    "accepts_temperature",
    "estimate_cost",
    "traits_for",
]

logger = structlog.get_logger()


@dataclass(frozen=True)
class ModelTraits:
    """
    Request-shaping facts about one model.

    Attributes:
        input_usd_per_mtok: Price per million input tokens.
        output_usd_per_mtok: Price per million output tokens.
        accepts_sampling_params: Whether temperature / top_p are accepted.
            Anthropic removed sampling parameters on its 4.7-and-later
            frontier models: sending ``temperature`` returns a 400 rather
            than being ignored.
        cache_read_multiplier: Fraction of the input price charged for tokens
            served from the provider's prompt cache.
        cache_write_multiplier: Multiple of the input price charged to place
            tokens in that cache.
    """

    input_usd_per_mtok: float
    output_usd_per_mtok: float
    accepts_sampling_params: bool
    cache_read_multiplier: float = 0.1
    cache_write_multiplier: float = 1.25


# Anthropic models, prices in USD per million tokens.
#
# Sampling parameters were removed on Opus 4.7 and later, Sonnet 5, and the
# Fable family. They remain accepted on Opus 4.6, Sonnet 4.6 and Haiku 4.5.
_ANTHROPIC_TRAITS: dict[str, ModelTraits] = {
    "claude-opus-5": ModelTraits(5.00, 25.00, accepts_sampling_params=False),
    "claude-opus-4-8": ModelTraits(5.00, 25.00, accepts_sampling_params=False),
    "claude-opus-4-7": ModelTraits(5.00, 25.00, accepts_sampling_params=False),
    "claude-opus-4-6": ModelTraits(5.00, 25.00, accepts_sampling_params=True),
    "claude-sonnet-5": ModelTraits(2.00, 10.00, accepts_sampling_params=False),
    "claude-sonnet-4-6": ModelTraits(3.00, 15.00, accepts_sampling_params=True),
    "claude-haiku-4-5": ModelTraits(1.00, 5.00, accepts_sampling_params=True),
    "claude-fable-5-1": ModelTraits(10.00, 50.00, accepts_sampling_params=False),
    "claude-fable-5": ModelTraits(10.00, 50.00, accepts_sampling_params=False),
}

# Gemini models, paid-tier prices in USD per million tokens, verified against
# https://ai.google.dev/gemini-api/docs/pricing on 2026-09-18.
#
# Two caveats on these numbers:
#
# * The 3.7 and 3.8 Flash rates below are promotional and double on
#   2027-01-01 (input $1.50, output $7.50, cache $0.15). Revisit before then;
#   from that date the cost column would silently understate spend.
# * The Pro models are tiered on prompt length and only the <=200k rate is
#   modelled here. Battle articles sit far below that, so the tier never
#   binds in this pipeline, but a long-context experiment would be underpriced.
#
# Gemini accepts sampling parameters on every current model, so temperature 0
# remains available here even though Anthropic has removed it on its newer
# models. Explicit context caching is billed by storage duration rather than
# per written token, so cache_write_multiplier stays at zero.
_GEMINI_TRAITS: dict[str, ModelTraits] = {
    "gemini-3.8-flash": ModelTraits(
        0.75, 3.75, accepts_sampling_params=True, cache_write_multiplier=0.0
    ),
    "gemini-3.7-flash": ModelTraits(
        0.75, 3.75, accepts_sampling_params=True, cache_write_multiplier=0.0
    ),
    "gemini-3.5-flash": ModelTraits(
        1.50, 9.00, accepts_sampling_params=True, cache_write_multiplier=0.0
    ),
    "gemini-3.1-flash-lite": ModelTraits(
        0.25, 1.50, accepts_sampling_params=True, cache_write_multiplier=0.0
    ),
    "gemini-3.1-pro-preview": ModelTraits(
        2.00, 12.00, accepts_sampling_params=True, cache_write_multiplier=0.0
    ),
    "gemini-2.5-pro": ModelTraits(
        1.25, 10.00, accepts_sampling_params=True, cache_write_multiplier=0.0
    ),
    "gemini-2.5-flash": ModelTraits(
        0.30, 2.50, accepts_sampling_params=True, cache_write_multiplier=0.0
    ),
}

_TRAITS_BY_PROVIDER: dict[str, dict[str, ModelTraits]] = {
    "anthropic": _ANTHROPIC_TRAITS,
    "gemini": _GEMINI_TRAITS,
}


def traits_for(provider: str, model: str) -> ModelTraits | None:
    """
    Look up the traits of a model.

    Args:
        provider: Provider name, e.g. ``"anthropic"``.
        model: Exact model identifier.

    Returns:
        The model's traits, or None if this model is not in the table.
    """
    return _TRAITS_BY_PROVIDER.get(provider, {}).get(model)


def accepts_temperature(provider: str, model: str) -> bool:
    """
    Whether a temperature parameter can be sent to this model.

    Unknown models are assumed *not* to accept one. A dropped temperature
    costs determinism that structured output constraints largely recover;
    an unexpected temperature costs a 400 on every call in the batch.

    Args:
        provider: Provider name.
        model: Exact model identifier.

    Returns:
        True if the parameter is safe to send.
    """
    known = traits_for(provider, model)
    if known is None:
        logger.debug("unknown_model_omitting_temperature", provider=provider, model=model)
        return False
    return known.accepts_sampling_params


def estimate_cost(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float | None:
    """
    Estimate the USD cost of one call.

    Cached input is billed at a fraction of the input rate and cache writes at
    a premium, so both are priced separately from uncached input.

    Args:
        provider: Provider name.
        model: Exact model identifier.
        input_tokens: Total input tokens reported by the provider, inclusive
            of any cached portion.
        output_tokens: Output tokens generated.
        cached_input_tokens: Portion of ``input_tokens`` served from cache.
        cache_write_tokens: Tokens written to the cache by this call.

    Returns:
        Cost in USD, or None if this model's pricing is not known, in which
        case the caller should record NULL rather than zero.
    """
    known = traits_for(provider, model)
    if known is None:
        logger.warning(
            "model_pricing_unknown_cost_not_recorded", provider=provider, model=model
        )
        return None

    uncached_input = max(input_tokens - cached_input_tokens, 0)
    per_token_in = known.input_usd_per_mtok / 1_000_000
    per_token_out = known.output_usd_per_mtok / 1_000_000

    return (
        uncached_input * per_token_in
        + cached_input_tokens * per_token_in * known.cache_read_multiplier
        + cache_write_tokens * per_token_in * known.cache_write_multiplier
        + output_tokens * per_token_out
    )
