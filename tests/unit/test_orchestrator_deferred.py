"""The orchestrator stops after a stage that deferred LLM requests."""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import pytest

from pipeline import orchestrator
from pipeline.quality import StageResult
from pipeline.stages.base import StageOutcome


def _stage_module(outcome: StageOutcome | None) -> types.ModuleType:
    module = types.ModuleType("fake_stage")

    def run(spec: dict[str, Any], context: Any = None) -> StageOutcome | None:
        return outcome

    module.run = run  # type: ignore[attr-defined]
    return module


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace spec loading, stage import and the quality gate."""
    state: dict[str, Any] = {"gates_run": 0, "ran": [], "outcomes": {}}

    monkeypatch.setattr(
        orchestrator, "load_agent_spec", lambda stage, overrides=None: {"stage": stage}
    )

    def fake_import(name: str) -> types.ModuleType:
        stage = name.rsplit(".", 1)[-1]
        state["ran"].append(stage)
        return _stage_module(state["outcomes"].get(stage))

    monkeypatch.setattr(orchestrator.importlib, "import_module", fake_import)

    def fake_gates(spec: dict[str, Any], db_conn: Any) -> list[Any]:
        state["gates_run"] += 1
        return []

    monkeypatch.setattr(orchestrator, "run_quality_checks", fake_gates)
    return state


def test_a_deferring_stage_skips_its_gates_and_marks_the_result_incomplete(
    patched: dict[str, Any],
) -> None:
    patched["outcomes"]["extract"] = StageOutcome(3, Path("data/llm_requests/extract_x.jsonl"))

    result = orchestrator.run_stage("extract")

    assert result.success is True
    assert result.awaiting_llm is True
    assert result.deferred_requests == 3
    assert result.batch_file is not None and result.batch_file.endswith("extract_x.jsonl")
    assert result.halts_pipeline is True
    assert patched["gates_run"] == 0


def test_the_pipeline_stops_after_the_deferring_stage(patched: dict[str, Any]) -> None:
    patched["outcomes"]["extract"] = StageOutcome(1, Path("b.jsonl"))

    results = orchestrator.run_pipeline(["crawl", "extract", "resolve"], stop_on_error=False)

    assert [r.stage for r in results] == ["crawl", "extract"]
    assert patched["ran"] == ["crawl", "extract"]
    assert patched["gates_run"] == 1  # crawl's gate only


def test_a_stage_with_nothing_deferred_runs_its_gates(patched: dict[str, Any]) -> None:
    result = orchestrator.run_stage("extract")

    assert result.awaiting_llm is False
    assert patched["gates_run"] == 1


def test_a_failed_result_without_deferral_does_not_look_awaiting() -> None:
    result = StageResult(stage="x", success=False, duration_seconds=0.0)

    assert result.awaiting_llm is False
    assert result.halts_pipeline is True
