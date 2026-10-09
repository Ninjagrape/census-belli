"""Unit tests for the request shape built by AnthropicClient._build_params."""

from __future__ import annotations

from typing import Any

import pytest

from pipeline.llm.anthropic_client import AnthropicClient
from pipeline.llm.base import LLMRequest

SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}},
    "required": ["name"],
    "additionalProperties": False,
}


def make_request(**overrides: Any) -> LLMRequest:
    """Build a request with sensible defaults for parameter-shape tests."""
    defaults: dict[str, Any] = {
        "system": "Extract battle data.",
        "user": "At Actium, Agrippa commanded 400 ships.",
        "json_schema": SCHEMA,
        "temperature": 0.0,
    }
    defaults.update(overrides)
    return LLMRequest(**defaults)


def make_client(model: str, **kwargs: Any) -> AnthropicClient:
    """Construct a client without touching the network; the SDK call is never made."""
    return AnthropicClient(model, **kwargs)


def test_haiku_5_5_sends_thinking_disabled_when_thinking_not_requested():
    # Arrange
    client = make_client("claude-haiku-5-5", enable_thinking=False)

    # Act
    params = client._build_params(make_request())

    # Assert
    assert params["thinking"] == {"type": "disabled"}


def test_haiku_5_5_sends_thinking_disabled_at_low_effort():
    client = make_client("claude-haiku-5-5", effort="low")

    params = client._build_params(make_request())

    assert params["thinking"] == {"type": "disabled"}


def test_haiku_5_5_omits_thinking_key_at_max_effort():
    # Arrange
    client = make_client("claude-haiku-5-5", effort="max", enable_thinking=False)

    # Act
    params = client._build_params(make_request())

    # Assert
    assert "thinking" not in params


def test_haiku_5_5_sends_adaptive_thinking_when_requested():
    client = make_client("claude-haiku-5-5", enable_thinking=True)

    params = client._build_params(make_request())

    assert params["thinking"] == {"type": "adaptive"}


def test_sonnet_4_6_sends_no_thinking_key_by_default():
    client = make_client("claude-sonnet-4-6", enable_thinking=False)

    params = client._build_params(make_request())

    assert "thinking" not in params


def test_haiku_5_5_sends_no_temperature():
    # Arrange
    client = make_client("claude-haiku-5-5", enable_thinking=False)

    # Act
    params = client._build_params(make_request(temperature=0.0))

    # Assert
    assert "temperature" not in params


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_haiku_5_5_disables_thinking_at_each_compatible_effort(effort: str):
    client = make_client("claude-haiku-5-5", effort=effort)

    params = client._build_params(make_request())

    assert params["thinking"] == {"type": "disabled"}
    assert params["output_config"]["effort"] == effort
