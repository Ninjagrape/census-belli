"""Integration tests for the orchestrator's quality gate.

These are the tests that would have caught the placeholder. Before
``pipeline/quality.py`` existed, ``_run_single_check`` returned
``passed=True`` unconditionally, so a stage whose data violated an
ERROR-severity check still reported success and the pipeline ran on.

No live database is needed: the gate is exercised through a stub connection
that returns scripted scalars for each check the spec declares.
"""

from __future__ import annotations

import sys
import types
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import pytest

from pipeline import orchestrator
from pipeline.quality import Severity

# The classify spec's ERROR-severity gate, used as the representative case.
# agents/classify.yaml:
#   missing_data_all_logged -> "= 0", severity: error
CLASSIFY_ERROR_CHECK = "missing_data_all_logged"


# ─── Stubs ───────────────────────────────────────────────────────────────────


class _StubResult:
    def __init__(self, value: Any) -> None:
        self._value = value

    def scalar(self) -> Any:
        return self._value


class ScriptedConnection:
    """Returns each scripted value in turn, one per executed query."""

    def __init__(self, values: list[Any]) -> None:
        self._values = list(values)
        self.calls = 0

    def begin_nested(self) -> AbstractContextManager[None]:
        """Stand in for SQLAlchemy's SAVEPOINT context manager.

        The runner opens one per check so a failing statement cannot abort the
        transaction the later checks need. There is no real transaction to
        protect here; the isolation itself is proven against a live database in
        tests/integration/test_quality_gates.py.
        """
        return nullcontext()

    def execute(self, statement: Any) -> _StubResult:
        value = self._values[self.calls] if self.calls < len(self._values) else 0
        self.calls += 1
        return _StubResult(value)


@pytest.fixture
def stub_stage(monkeypatch: pytest.MonkeyPatch):
    """Install an importable no-op stage module so run_stage reaches the gate."""

    def _install(stage: str) -> types.ModuleType:
        module = types.ModuleType(f"pipeline.stages.{stage}")
        module.run = lambda spec: None  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, f"pipeline.stages.{stage}", module)
        return module

    return _install


# ─── The gate actually blocks ────────────────────────────────────────────────


def test_error_severity_failure_fails_the_stage(stub_stage) -> None:
    """A violated ERROR check must produce success=False.

    classify.yaml's checks in order:
      all_commanders_classified  "< 200"  warning
      attribution_weights_sum    "< 50"   warning
      missing_data_all_logged    "= 0"    error   <- violated below
    """
    stub_stage("classify")
    # max_stage_retries is 2, so the gate runs three times; the data is still
    # bad on every attempt.
    conn = ScriptedConnection([10, 2, 47] * 3)  # 47 sides missing a log entry

    result = orchestrator.run_stage("classify", db_conn=conn)

    assert result.success is False, "A failing ERROR-severity gate must fail the stage"

    failed = [c for c in result.quality_checks if not c.passed]
    assert [c.name for c in failed] == [CLASSIFY_ERROR_CHECK]
    assert failed[0].severity is Severity.ERROR
    assert failed[0].actual_value == 47


def test_passing_checks_succeed(stub_stage) -> None:
    stub_stage("classify")
    conn = ScriptedConnection([10, 2, 0])  # zero unlogged missing values

    result = orchestrator.run_stage("classify", db_conn=conn)

    assert result.success is True
    assert all(c.passed for c in result.quality_checks)


def test_warning_failure_does_not_fail_the_stage(stub_stage) -> None:
    """Only ERROR severity blocks; a warning is recorded but tolerated."""
    stub_stage("classify")
    conn = ScriptedConnection([9999, 2, 0])  # blows the warning-level threshold

    result = orchestrator.run_stage("classify", db_conn=conn)

    assert result.success is True
    failed = [c for c in result.quality_checks if not c.passed]
    assert [c.name for c in failed] == ["all_commanders_classified"]
    assert failed[0].severity is Severity.WARNING


def test_no_connection_fails_rather_than_passes(stub_stage) -> None:
    """The regression guard: an absent database is not a clean gate."""
    stub_stage("classify")

    result = orchestrator.run_stage("classify", db_conn=None)

    assert result.success is False
    assert all(not c.passed for c in result.quality_checks)


# ─── Retry behaviour ─────────────────────────────────────────────────────────


def test_failing_gate_consumes_retries(stub_stage) -> None:
    """classify.yaml sets max_stage_retries: 2, so the gate runs three times."""
    stub_stage("classify")
    conn = ScriptedConnection([10, 2, 47] * 3)

    result = orchestrator.run_stage("classify", db_conn=conn)

    assert result.success is False
    assert result.retries_used == 2
    assert conn.calls == 9  # 3 checks x 3 attempts


# ─── on_failure is honoured ──────────────────────────────────────────────────


def test_continue_with_logging_does_not_halt_pipeline(stub_stage) -> None:
    """classify.yaml sets on_failure: continue_with_logging.

    A failure there must not stop later stages, which is what the spec asks for
    and what the discarded `on_failure` variable previously prevented.
    """
    stub_stage("classify")
    stub_stage("impute")
    conn = ScriptedConnection([10, 2, 47] * 3)  # classify fails all attempts

    results = orchestrator.run_pipeline(
        ["classify", "impute"],
        stop_on_error=True,
        db_conn=conn,
    )

    assert [r.stage for r in results] == ["classify", "impute"], (
        "impute must still run: classify's spec says continue_with_logging"
    )
    assert results[0].success is False
    assert results[0].on_failure == "continue_with_logging"


def test_pause_and_alert_halts_pipeline(stub_stage) -> None:
    """crawl.yaml sets on_failure: pause_and_alert, so a failure stops the run.

    crawl.yaml's checks in order:
      min_battles_crawled        ">= 3000"  error   <- violated below
      error_rate_below_threshold "< 0.10"   warning
      wikidata_coverage          ">= 1000"  warning
    """
    stub_stage("crawl")
    stub_stage("extract")
    conn = ScriptedConnection([5, 0.01, 2000] * 3)  # only 5 battles crawled

    results = orchestrator.run_pipeline(
        ["crawl", "extract"],
        stop_on_error=True,
        db_conn=conn,
    )

    assert [r.stage for r in results] == ["crawl"], "extract must not run after crawl halts"
    assert results[0].success is False
    assert results[0].halts_pipeline is True


def test_no_stop_on_error_runs_everything(stub_stage) -> None:
    stub_stage("crawl")
    stub_stage("extract")
    conn = ScriptedConnection([5, 0.01, 2000] * 3 + [0.9, 2.5, 10])

    results = orchestrator.run_pipeline(
        ["crawl", "extract"],
        stop_on_error=False,
        db_conn=conn,
    )

    assert [r.stage for r in results] == ["crawl", "extract"]


# ─── Spec coverage ───────────────────────────────────────────────────────────


def test_every_stage_spec_loads() -> None:
    """All nine stages in STAGE_ORDER must have a spec, including report."""
    for stage in orchestrator.STAGE_ORDER:
        spec = orchestrator.load_agent_spec(stage)
        assert spec["stage"] == stage
        assert "quality_checks" in spec, f"{stage} declares no quality gate"
