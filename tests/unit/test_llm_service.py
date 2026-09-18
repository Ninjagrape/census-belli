"""Unit tests for provider selection and the cached, logged call path."""

from __future__ import annotations

from typing import Any

import pytest

from pipeline.llm.base import CallStatus, LLMConfigError, LLMRequest, LLMResponse, TokenUsage
from pipeline.llm.factory import build_provider, llm_params
from pipeline.llm.service import LLMService

SCHEMA = {
    "type": "object",
    "properties": {"commander": {"type": "string"}},
    "required": ["commander"],
    "additionalProperties": False,
}


class FakeProvider:
    """A provider that records its calls and returns canned responses."""

    name = "fake"
    model = "fake-model-1"

    def __init__(self, responses: list[LLMResponse] | None = None) -> None:
        self.requests: list[LLMRequest] = []
        self._responses = responses or []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self._responses:
            return self._responses.pop(0)
        return LLMResponse(
            status=CallStatus.OK,
            provider=self.name,
            model=self.model,
            request_hash="deadbeef",
            data={"commander": "Agrippa"},
            usage=TokenUsage(input_tokens=1200, output_tokens=80),
            cost_usd=0.002,
        )


class FakeResult:
    """Stands in for a SQLAlchemy result."""

    def __init__(self, row: tuple[Any, ...] | None) -> None:
        self._row = row

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._row


class FakeConnection:
    """Records executed statements and replays a scripted lookup result."""

    def __init__(self, cached_row: tuple[Any, ...] | None = None) -> None:
        self.statements: list[tuple[Any, Any]] = []
        self._cached_row = cached_row

    def execute(self, statement: Any, parameters: Any = None) -> FakeResult:
        self.statements.append((statement, parameters))
        if "SELECT" in str(statement):
            return FakeResult(self._cached_row)
        return FakeResult((1,))


# ─── Spec parsing and provider selection ─────────────────────────────────────


def test_defaults_to_anthropic_when_no_provider_is_named():
    # An omitted provider must not silently reroute a stage
    assert llm_params({"params": {"llm_model": "claude-sonnet-4-6"}})["provider"] == "anthropic"


def test_reads_provider_settings_from_the_spec():
    # Arrange
    spec = {
        "stage": "extract",
        "params": {
            "llm_provider": "gemini",
            "llm_model": "gemini-3.8-flash",
            "llm_temperature": 0.0,
            "llm_max_tokens": 2048,
        },
    }

    # Act
    config = llm_params(spec)

    # Assert
    assert config["provider"] == "gemini"
    assert config["model"] == "gemini-3.8-flash"
    assert config["temperature"] == 0.0
    assert config["max_tokens"] == 2048


def test_rejects_an_unknown_provider():
    # Arrange
    spec = {"stage": "extract", "params": {"llm_provider": "openai", "llm_model": "some-model"}}

    # Act / Assert
    with pytest.raises(LLMConfigError, match="unknown llm_provider"):
        build_provider(spec)


def test_rejects_a_spec_with_no_model():
    # Arrange: there is no safe default, since a wrong guess is billed per article
    spec = {"stage": "extract", "params": {"llm_provider": "gemini"}}

    # Act / Assert
    with pytest.raises(LLMConfigError, match="no llm_model"):
        build_provider(spec)


def test_gemini_client_refuses_to_start_without_a_key(monkeypatch):
    # Arrange
    pytest.importorskip("google.genai")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    spec = {
        "stage": "extract",
        "params": {"llm_provider": "gemini", "llm_model": "gemini-3.8-flash"},
    }

    # Act / Assert: fail once at startup, not once per article
    with pytest.raises(LLMConfigError, match="GEMINI_API_KEY"):
        build_provider(spec)


# ─── Service behaviour ───────────────────────────────────────────────────────


def test_calls_the_provider_and_returns_parsed_data():
    # Arrange
    provider = FakeProvider()
    service = LLMService("extract", provider, db_conn=FakeConnection())

    # Act
    result = service.complete("sys", "user", SCHEMA, metadata={"battle_id": 412})

    # Assert
    assert result.ok
    assert result.data == {"commander": "Agrippa"}
    assert len(provider.requests) == 1


def test_writes_an_audit_row_for_every_call():
    # Arrange
    conn = FakeConnection()
    service = LLMService("extract", FakeProvider(), db_conn=conn)

    # Act
    service.complete("sys", "user", SCHEMA)

    # Assert
    inserts = [s for s, _ in conn.statements if "INSERT INTO llm_calls" in str(s)]
    assert len(inserts) == 1


def test_a_cache_hit_skips_the_provider():
    # Arrange: the audit table already holds a successful identical call
    conn = FakeConnection(cached_row=({"commander": "Agrippa"},))
    provider = FakeProvider()
    service = LLMService("extract", provider, db_conn=conn)

    # Act
    result = service.complete("sys", "user", SCHEMA)

    # Assert
    assert result.ok
    assert result.data == {"commander": "Agrippa"}
    assert provider.requests == []
    assert service.usage.cache_hits == 1


def test_a_cache_hit_does_not_double_count_tokens():
    # Arrange
    conn = FakeConnection(cached_row=({"commander": "Agrippa"},))
    service = LLMService("extract", FakeProvider(), db_conn=conn)

    # Act
    service.complete("sys", "user", SCHEMA)

    # Assert: those tokens were billed to the original call
    assert service.usage.input_tokens == 0
    assert service.usage.cost_usd == 0.0


def test_use_cache_false_forces_a_live_call():
    # Arrange
    conn = FakeConnection(cached_row=({"commander": "Agrippa"},))
    provider = FakeProvider()
    service = LLMService("extract", provider, db_conn=conn, use_cache=False)

    # Act
    service.complete("sys", "user", SCHEMA)

    # Assert
    assert len(provider.requests) == 1


def test_a_failed_call_is_returned_not_raised():
    # Arrange: one bad article must not abort the batch
    failure = LLMResponse(
        status=CallStatus.PARSE_ERROR,
        provider="fake",
        model="fake-model-1",
        request_hash="deadbeef",
        error="invalid JSON: unexpected end of input",
    )
    service = LLMService("extract", FakeProvider([failure]), db_conn=FakeConnection())

    # Act
    result = service.complete("sys", "user", SCHEMA)

    # Assert
    assert not result.ok
    assert result.status.needs_review
    assert service.usage.failures == 1


def test_failed_calls_are_still_audited():
    # Arrange
    conn = FakeConnection()
    failure = LLMResponse(
        status=CallStatus.REFUSAL,
        provider="fake",
        model="fake-model-1",
        request_hash="deadbeef",
        error="provider refused the request",
    )
    service = LLMService("extract", FakeProvider([failure]), db_conn=conn)

    # Act
    service.complete("sys", "user", SCHEMA)

    # Assert: a refusal is evidence about the corpus, not something to discard
    inserts = [p for s, p in conn.statements if "INSERT INTO llm_calls" in str(s)]
    assert len(inserts) == 1
    assert inserts[0]["status"] == "refusal"


def test_unpriced_models_mark_the_cost_total_as_incomplete():
    # Arrange
    unpriced = LLMResponse(
        status=CallStatus.OK,
        provider="fake",
        model="fake-model-1",
        request_hash="deadbeef",
        data={"commander": "Agrippa"},
        usage=TokenUsage(input_tokens=100, output_tokens=10),
        cost_usd=None,
    )
    service = LLMService("extract", FakeProvider([unpriced]), db_conn=FakeConnection())

    # Act
    service.complete("sys", "user", SCHEMA)

    # Assert: the reported dollar total is a floor, and says so
    assert service.usage.cost_incomplete is True


def test_stage_defaults_apply_to_requests():
    # Arrange
    provider = FakeProvider()
    service = LLMService(
        "extract",
        provider,
        db_conn=FakeConnection(),
        default_temperature=0.0,
        default_max_tokens=2048,
    )

    # Act
    service.complete("sys", "user", SCHEMA)

    # Assert
    assert provider.requests[0].max_tokens == 2048
    assert provider.requests[0].temperature == 0.0


def test_per_call_overrides_beat_stage_defaults():
    # Arrange
    provider = FakeProvider()
    service = LLMService("extract", provider, db_conn=FakeConnection(), default_max_tokens=2048)

    # Act
    service.complete("sys", "user", SCHEMA, max_tokens=512)

    # Assert
    assert provider.requests[0].max_tokens == 512


def test_runs_without_a_database_connection():
    # Arrange: unit tests and dry runs have no database
    service = LLMService("extract", FakeProvider(), db_conn=None)

    # Act
    result = service.complete("sys", "user", SCHEMA)

    # Assert
    assert result.ok
