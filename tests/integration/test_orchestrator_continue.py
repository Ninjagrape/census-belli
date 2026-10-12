"""run_pipeline carries on past a deferring stage only when asked to."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from pipeline import orchestrator
from pipeline.stages.base import StageOutcome


@pytest.fixture
def stub_stages(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Install crawl (defers) and extract (runs) as importable stub stages."""
    ran: list[str] = []

    def crawl(spec: dict[str, Any], context: Any = None) -> StageOutcome:
        ran.append("crawl")
        return StageOutcome(4, Path("data/llm_requests/crawl_stub.jsonl"))

    def extract(spec: dict[str, Any], context: Any = None) -> None:
        ran.append("extract")

    for name, fn in (("crawl", crawl), ("extract", extract)):
        module = types.ModuleType(f"pipeline.stages.{name}")
        module.run = fn  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, f"pipeline.stages.{name}", module)
    monkeypatch.setattr(orchestrator, "run_quality_checks", lambda spec, db_conn: [])
    return ran


def test_default_halts_after_the_deferring_stage(stub_stages: list[str]) -> None:
    results = orchestrator.run_pipeline(["crawl", "extract"])

    assert stub_stages == ["crawl"]
    assert [r.stage for r in results] == ["crawl"]


def test_continue_on_deferred_runs_the_next_stage(stub_stages: list[str]) -> None:
    results = orchestrator.run_pipeline(["crawl", "extract"], continue_on_deferred=True)

    assert stub_stages == ["crawl", "extract"]
    assert results[0].awaiting_llm is True
    assert results[0].batch_file is not None
    assert results[1].success is True
