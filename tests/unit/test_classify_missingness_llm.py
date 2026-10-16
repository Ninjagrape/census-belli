"""Unit tests for the LLM half of missingness classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from pipeline.classifiers.missingness_llm import (
    MISSINGNESS_RESPONSE_SCHEMA,
    apply_llm_missingness,
    build_missingness_request,
    era_label,
)
from pipeline.classifiers.records import MISSINGNESS_CLASSES, MissingnessContext
from pipeline.llm.base import request_hash

SPEC_PATH = Path(__file__).resolve().parent.parent.parent / "agents" / "classify.yaml"


@pytest.fixture(scope="module")
def spec() -> dict[str, Any]:
    return yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))


def _context(**overrides: Any) -> MissingnessContext:
    defaults: dict[str, Any] = {
        "log_id": 7,
        "battle_id": 3,
        "field_name": "casualties",
        "battle_name": "Battle of Test",
        "year_astronomical": 1850,
        "side_label": "Side A",
        "polity": "Austria",
        "outcome": "loser",
        "this_side_has_troops": True,
    }
    defaults.update(overrides)
    return MissingnessContext(**defaults)


def test_schema_enum_is_a_subset_of_the_database_enum() -> None:
    values = set(MISSINGNESS_RESPONSE_SCHEMA["properties"]["missingness_type"]["enum"])

    assert values <= set(MISSINGNESS_CLASSES)
    assert MISSINGNESS_RESPONSE_SCHEMA["required"] == ["missingness_type", "reason"]
    assert MISSINGNESS_RESPONSE_SCHEMA["additionalProperties"] is False


def test_request_carries_the_evidence_and_the_strict_schema(spec: dict[str, Any]) -> None:
    request = build_missingness_request(_context(), spec)

    assert "Battle of Test" in request.user
    assert "casualties" in request.user
    assert "1815-1913" in request.user
    assert "this side's troop total: known" in request.user
    assert request.json_schema == MISSINGNESS_RESPONSE_SCHEMA
    assert request.metadata == {"battle_id": 3, "log_id": 7}


def test_the_same_context_hashes_the_same_and_a_changed_fact_does_not(
    spec: dict[str, Any],
) -> None:
    first = request_hash(build_missingness_request(_context(), spec), "anthropic", "m")
    again = request_hash(build_missingness_request(_context(), spec), "anthropic", "m")
    other = request_hash(
        build_missingness_request(_context(this_side_has_troops=False), spec), "anthropic", "m"
    )

    assert first == again
    assert first != other


def test_a_spec_without_the_prompt_is_rejected() -> None:
    with pytest.raises(ValueError, match="missingness_prompt"):
        build_missingness_request(_context(), {})


@pytest.mark.parametrize("value", ["mcar", "mar", "mnar", " MNAR "])
def test_a_valid_answer_becomes_a_decision(value: str) -> None:
    decision = apply_llm_missingness(7, {"missingness_type": value, "reason": "why"})

    assert decision is not None
    assert decision.log_id == 7
    assert decision.missingness_class == value.strip().lower()
    assert "why" in decision.reason


@pytest.mark.parametrize("value", ["observed", "unclassified", "", None, "maybe"])
def test_an_answer_outside_the_mechanisms_is_ignored(value: Any) -> None:
    assert apply_llm_missingness(7, {"missingness_type": value, "reason": "x"}) is None


def test_era_labels() -> None:
    assert era_label(None) == "unknown"
    assert era_label(-200) == "ancient (before 500)"
    assert era_label(1943) == "world wars (1914-1945)"
    assert era_label(1973) == "post-1945"
