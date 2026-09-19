"""Unit tests for the provider-agnostic parts of the LLM layer."""

from __future__ import annotations

import pytest

from pipeline.llm.base import CallStatus, LLMRequest, request_hash
from pipeline.llm.capabilities import accepts_temperature, estimate_cost, traits_for
from pipeline.llm.gemini_client import _retry_after_seconds
from pipeline.llm.parsing import parse_structured, strip_code_fence
from pipeline.llm.retry import TransientLLMError, with_backoff

SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "troops": {"type": "integer"}},
    "required": ["name"],
    "additionalProperties": False,
}


def make_request(**overrides: object) -> LLMRequest:
    """Build a request with sensible defaults for hashing tests."""
    defaults = {
        "system": "Extract battle data.",
        "user": "At Actium, Agrippa commanded 400 ships.",
        "json_schema": SCHEMA,
    }
    defaults.update(overrides)
    return LLMRequest(**defaults)  # type: ignore[arg-type]


# ─── Request hashing ─────────────────────────────────────────────────────────


def test_identical_requests_hash_identically():
    # Arrange
    first = make_request()
    second = make_request()

    # Act
    a = request_hash(first, "gemini", "gemini-3.8-flash")
    b = request_hash(second, "gemini", "gemini-3.8-flash")

    # Assert
    assert a == b


def test_hash_changes_when_the_model_changes():
    # Arrange
    request = make_request()

    # Act
    a = request_hash(request, "gemini", "gemini-3.8-flash")
    b = request_hash(request, "gemini", "gemini-3.1-flash-lite")

    # Assert: a re-run on a different model must not reuse cached extractions
    assert a != b


def test_hash_ignores_metadata():
    # Arrange: metadata is local bookkeeping, not model input
    without = make_request()
    with_meta = make_request(metadata={"battle_id": 412})

    # Act / Assert
    assert request_hash(without, "gemini", "m") == request_hash(with_meta, "gemini", "m")


def test_hash_is_stable_across_schema_key_order():
    # Arrange: dict ordering must not invalidate the cache
    reordered = dict(reversed(list(SCHEMA.items())))

    # Act / Assert
    assert request_hash(make_request(), "gemini", "m") == request_hash(
        make_request(json_schema=reordered), "gemini", "m"
    )


# ─── Parsing ─────────────────────────────────────────────────────────────────


def test_parses_a_valid_object():
    # Act
    data, error = parse_structured('{"name": "Actium", "troops": 400}', SCHEMA)

    # Assert
    assert error is None
    assert data == {"name": "Actium", "troops": 400}


def test_strips_a_markdown_code_fence():
    # Arrange
    fenced = '```json\n{"name": "Actium"}\n```'

    # Act
    data, error = parse_structured(fenced, SCHEMA)

    # Assert
    assert error is None
    assert data == {"name": "Actium"}


def test_returns_error_for_truncated_json():
    # Arrange: what a response cut off at the token ceiling looks like
    data, error = parse_structured('{"name": "Act', SCHEMA)

    # Assert
    assert data is None
    assert "invalid JSON" in error


def test_returns_error_when_a_required_field_is_missing():
    # Act
    data, error = parse_structured('{"troops": 400}', SCHEMA)

    # Assert
    assert data is None
    assert error is not None


def test_returns_error_for_empty_body():
    # Act
    data, error = parse_structured("   ", SCHEMA)

    # Assert
    assert data is None
    assert "empty" in error


def test_rejects_a_json_array():
    # Arrange: valid JSON, wrong shape for an extraction record
    data, error = parse_structured('[{"name": "Actium"}]', SCHEMA)

    # Assert
    assert data is None
    assert "object" in error


def test_strip_code_fence_leaves_bare_json_untouched():
    assert strip_code_fence('{"a": 1}') == '{"a": 1}'


# ─── Capabilities ────────────────────────────────────────────────────────────


def test_newer_anthropic_models_reject_sampling_params():
    assert accepts_temperature("anthropic", "claude-sonnet-5") is False
    assert accepts_temperature("anthropic", "claude-opus-5") is False


def test_older_anthropic_models_still_accept_temperature():
    assert accepts_temperature("anthropic", "claude-sonnet-4-6") is True


def test_gemini_accepts_temperature():
    assert accepts_temperature("gemini", "gemini-3.8-flash") is True


def test_unknown_models_are_assumed_to_reject_temperature():
    # A dropped temperature costs determinism; an unexpected one costs a 400
    # on every call in the batch, so the safe default is to omit it.
    assert accepts_temperature("anthropic", "claude-not-a-real-model") is False


def test_cost_is_none_for_an_unpriced_model():
    # Better a NULL cost column than an authoritative-looking wrong number
    assert estimate_cost("gemini", "gemini-unreleased", 1000, 500) is None


def test_cost_uses_published_rates():
    # Arrange: 1M input + 1M output on Sonnet 4.6 at $3 / $15
    cost = estimate_cost("anthropic", "claude-sonnet-4-6", 1_000_000, 1_000_000)

    # Assert
    assert cost == pytest.approx(18.00)


def test_cached_input_is_billed_at_a_discount():
    # Arrange: the whole prompt served from cache
    full = estimate_cost("anthropic", "claude-sonnet-4-6", 1_000_000, 0)
    cached = estimate_cost(
        "anthropic", "claude-sonnet-4-6", 1_000_000, 0, cached_input_tokens=1_000_000
    )

    # Assert
    assert cached < full
    discount = traits_for("anthropic", "claude-sonnet-4-6").cache_read_multiplier
    assert cached == pytest.approx(full * discount)


# ─── Retry ───────────────────────────────────────────────────────────────────


def test_returns_immediately_when_the_first_attempt_succeeds():
    # Act
    result, attempts = with_backoff(lambda: "ok", max_retries=3, base_delay=0.0)

    # Assert
    assert result == "ok"
    assert attempts == 1


def test_retries_transient_failures_then_succeeds():
    # Arrange
    calls = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise TransientLLMError("rate limited")
        return "ok"

    # Act
    result, attempts = with_backoff(flaky, max_retries=3, base_delay=0.0)

    # Assert
    assert result == "ok"
    assert attempts == 3


def test_raises_after_exhausting_retries():
    # Arrange
    def always_fails() -> str:
        raise TransientLLMError("still rate limited")

    # Act / Assert
    with pytest.raises(TransientLLMError):
        with_backoff(always_fails, max_retries=2, base_delay=0.0)


def test_does_not_retry_permanent_failures():
    # Arrange
    calls = {"n": 0}

    def bad_request() -> str:
        calls["n"] += 1
        raise ValueError("schema rejected")

    # Act / Assert
    with pytest.raises(ValueError):
        with_backoff(bad_request, max_retries=3, base_delay=0.0)
    assert calls["n"] == 1


# ─── Call status ─────────────────────────────────────────────────────────────


def test_only_ok_avoids_manual_review():
    assert CallStatus.OK.needs_review is False
    assert CallStatus.PARSE_ERROR.needs_review is True
    assert CallStatus.REFUSAL.needs_review is True
    assert CallStatus.TRUNCATED.needs_review is True
    assert CallStatus.API_ERROR.needs_review is True


# ─── Server-suggested retry delays ───────────────────────────────────────────


def test_a_named_retry_delay_beats_the_computed_backoff(monkeypatch):
    # Blind backoff answers "retry in 29s" with about 2.5s, fails again, and
    # burns every remaining attempt on a limit that was never going to lift.
    # A free-tier Gemini key did exactly that on 2026-09-19, and it looked
    # like a hang rather than a quota error.
    slept: list[float] = []
    monkeypatch.setattr("pipeline.llm.retry.time.sleep", slept.append)

    attempts = {"n": 0}

    def operation() -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise TransientLLMError("rate limited", retry_after=29.0)
        return "ok"

    result, count = with_backoff(operation, max_retries=2, base_delay=2.0)

    assert result == "ok"
    assert count == 2
    assert slept == [29.0]


def test_a_named_delay_is_capped(monkeypatch):
    # Waiting is only worth it if the wait is plausible. Beyond the ceiling
    # the limit is better reported than slept through.
    slept: list[float] = []
    monkeypatch.setattr("pipeline.llm.retry.time.sleep", slept.append)

    def operation() -> str:
        raise TransientLLMError("rate limited", retry_after=99999.0)

    with pytest.raises(TransientLLMError):
        with_backoff(operation, max_retries=1, base_delay=2.0)

    assert slept and slept[0] <= 120.0


def test_a_shorter_named_delay_does_not_shorten_the_backoff(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr("pipeline.llm.retry.time.sleep", slept.append)

    def operation() -> str:
        raise TransientLLMError("blip", retry_after=0.1)

    with pytest.raises(TransientLLMError):
        with_backoff(operation, max_retries=1, base_delay=2.0)

    assert slept and slept[0] >= 2.0


def test_no_named_delay_leaves_the_backoff_alone(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr("pipeline.llm.retry.time.sleep", slept.append)

    def operation() -> str:
        raise TransientLLMError("no hint given")

    with pytest.raises(TransientLLMError):
        with_backoff(operation, max_retries=1, base_delay=2.0)

    assert slept and 2.0 <= slept[0] <= 3.0


class _ErrorWithHeaders(Exception):
    """An SDK error carrying a Retry-After header."""

    def __init__(self, message: str, headers: dict[str, str]) -> None:
        super().__init__(message)
        self.response_headers = headers


def test_retry_after_is_read_from_the_header_when_present():
    error = _ErrorWithHeaders("429 too many requests", {"Retry-After": "42"})
    assert _retry_after_seconds(error) == 42.0


def test_retry_after_is_read_from_geminis_prose_when_there_is_no_header():
    # Gemini states the delay in the message body rather than only a header.
    error = Exception(
        "Error code: 429 - {'error': {'message': 'Rate limit exceeded for model "
        "gemini-3.8-flash (limit: 20 requests per day on Free Tier). "
        "Please retry in 29s or upgrade your tier.'}}"
    )
    assert _retry_after_seconds(error) == 29.0


def test_retry_after_is_none_when_nothing_says_so():
    assert _retry_after_seconds(Exception("500 internal error")) is None
