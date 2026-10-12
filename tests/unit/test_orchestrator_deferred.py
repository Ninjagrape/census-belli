"""The orchestrator stops after a stage that deferred LLM requests."""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import pytest

from pipeline import orchestrator
from pipeline.quality import QualityCheckResult, Severity, StageResult
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


def test_continue_on_deferred_runs_the_next_stage(patched: dict[str, Any]) -> None:
    patched["outcomes"]["extract"] = StageOutcome(2, Path("b.jsonl"))

    results = orchestrator.run_pipeline(
        ["crawl", "extract", "resolve"], stop_on_error=True, continue_on_deferred=True
    )

    assert [r.stage for r in results] == ["crawl", "extract", "resolve"]
    assert [r.awaiting_llm for r in results] == [False, True, False]
    assert patched["gates_run"] == 2  # extract's gates stay skipped


def test_all_skips_unbuilt_stages(patched: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(orchestrator, "_stage_is_built", lambda s: s not in {"evaluate", "report"})

    results = orchestrator.run_pipeline(["all"])

    assert [r.stage for r in results] == orchestrator.STAGE_ORDER[:7]


def test_naming_an_unbuilt_stage_fails_before_running_anything(
    patched: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(orchestrator, "_stage_is_built", lambda s: s != "evaluate")

    with pytest.raises(SystemExit) as exc:
        orchestrator.run_pipeline(["crawl", "evaluate"])

    assert exc.value.code == 1
    assert patched["ran"] == []


def test_gate_failure_retries_only_when_the_policy_says_so(
    patched: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = [QualityCheckResult(name="g", passed=False, severity=Severity.ERROR)]
    monkeypatch.setattr(orchestrator, "run_quality_checks", lambda spec, db_conn: bad)

    def spec_with(policy: dict[str, Any]) -> Any:
        return lambda stage, overrides=None: {"stage": stage, "retry_policy": policy}

    monkeypatch.setattr(orchestrator, "load_agent_spec", spec_with({"max_stage_retries": 2}))
    assert orchestrator.run_stage("model").retries_used == 0

    monkeypatch.setattr(
        orchestrator,
        "load_agent_spec",
        spec_with({"max_stage_retries": 2, "retry_on_gate_failure": True}),
    )
    assert orchestrator.run_stage("model").retries_used == 2
