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
from pathlib import Path
from typing import Any

import pytest

from pipeline import orchestrator
from pipeline.overrides import KINDS_BY_STAGE, OverrideReport, load_overrides
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
        self.commits = 0
        self.rollbacks = 0

    def begin_nested(self) -> AbstractContextManager[None]:
        """Stand in for SQLAlchemy's SAVEPOINT context manager.

        The runner opens one per check so a failing statement cannot abort the
        transaction the later checks need. There is no real transaction to
        protect here; the isolation itself is proven against a live database in
        tests/integration/test_quality_gates.py.
        """
        return nullcontext()

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def execute(self, statement: Any) -> _StubResult:
        value = self._values[self.calls] if self.calls < len(self._values) else 0
        self.calls += 1
        return _StubResult(value)


@pytest.fixture
def stub_stage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Install an importable no-op stage module so run_stage reaches the gate."""
    # Reports go to tmp_path, not the real data/processed.
    real_load = orchestrator.load_agent_spec

    def _load(stage: str) -> dict[str, Any]:
        spec = real_load(stage)
        spec.setdefault("params", {})["processed_root"] = str(tmp_path)
        return spec

    monkeypatch.setattr(orchestrator, "load_agent_spec", _load)

    # The stub reports from an empty file of its own, so these tests do not
    # depend on what config/overrides.yaml holds (write() requires every entry
    # of the stage to be reported). The gate re-hashes the file the report
    # names, which is this one.
    empty_overrides = tmp_path / "overrides_empty.yaml"
    empty_overrides.write_text("version: 1\nentries: []\n", encoding="utf-8")

    def _install(stage: str) -> types.ModuleType:
        module = types.ModuleType(f"pipeline.stages.{stage}")

        def _run(spec: dict[str, Any]) -> None:
            # A stage that owns override kinds leaves a report for its
            # overrides_gate; the real stage writes the same file.
            if stage in KINDS_BY_STAGE:
                root = Path((spec.get("params") or {}).get("processed_root", "data/processed"))
                OverrideReport(stage, load_overrides(empty_overrides)).write(root)

        module.run = _run  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, f"pipeline.stages.{stage}", module)
        return module

    return _install


# ─── The gate actually blocks ────────────────────────────────────────────────


def test_error_severity_failure_fails_the_stage(stub_stage) -> None:
    """A violated ERROR check must produce success=False.

    classify.yaml's checks in order:
      all_commanders_classified  "< 200"  warning
      attribution_weights_sum    "< 50"   warning
      battle_type_unknown_share  "< 0.10" warning
      missing_data_all_logged    "= 0"    error   <- violated below
    """
    stub_stage("classify")
    conn = ScriptedConnection([10, 2, 0.0, 47])  # 47 sides missing a log entry

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
    # overrides_applied reads a report file, not the database, so it can pass.
    sql_checks = [c for c in result.quality_checks if c.name != "overrides_applied"]
    assert sql_checks and all(not c.passed for c in sql_checks)


# ─── Retry behaviour ─────────────────────────────────────────────────────────


def test_failing_gate_does_not_retry_by_default(stub_stage) -> None:
    """classify.yaml leaves retry_on_gate_failure unset, so a gate failure is final."""
    stub_stage("classify")
    conn = ScriptedConnection([10, 2, 0.0, 47] * 3)

    result = orchestrator.run_stage("classify", db_conn=conn)

    assert result.success is False
    assert result.retries_used == 0
    assert conn.calls == 4  # 4 checks x 1 attempt


def test_failing_gate_retries_when_policy_opts_in(stub_stage) -> None:
    """crawl.yaml sets retry_on_gate_failure: true and max_stage_retries: 2."""
    stub_stage("crawl")
    # min_battles_crawled, battle_articles_fetched_ratio (fails), error rate, wikidata coverage.
    conn = ScriptedConnection([5, 0.01, 0.0, 1.0] * 3)

    result = orchestrator.run_stage("crawl", db_conn=conn)

    assert result.success is False
    assert result.retries_used == 2
    assert conn.calls == 12  # 4 checks x 3 attempts


# ─── on_failure is honoured ──────────────────────────────────────────────────


def test_continue_with_logging_does_not_halt_pipeline(stub_stage) -> None:
    """classify.yaml sets on_failure: continue_with_logging.

    A failure there must not stop later stages, which is what the spec asks for
    and what the discarded `on_failure` variable previously prevented.
    """
    stub_stage("classify")
    stub_stage("impute")
    conn = ScriptedConnection([10, 2, 0.0, 47])  # classify fails its one attempt

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
      min_battles_crawled           ">= 0"    error   <- violated below
      battle_articles_fetched_ratio ">= 0.95" error
      error_rate_below_threshold    "< 0.10"  warning
      wikidata_coverage             ">= 0.9"  warning
    """
    stub_stage("crawl")
    stub_stage("extract")
    conn = ScriptedConnection([-2995, 0.01, 0.5, 0.0] * 3)  # 5 battles against a 3000 floor

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
    conn = ScriptedConnection([-2995, 0.01, 0.5, 0.0] * 3 + [0.9, 2.5, 10])

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


# ─── The stub does not depend on the real overrides file ─────────────────────


def test_stub_gate_ignores_a_populated_default_overrides_file(
    stub_stage, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Real entries for the stage must not break the stubbed run."""
    populated = tmp_path / "populated.yaml"
    populated.write_text(
        "version: 1\n"
        "entries:\n"
        "  - id: ov-0001\n"
        "    kind: battle_type\n"
        "    target: {battle: battle_of_actium}\n"
        "    value: field\n"
        "    source: Plutarch\n"
        "    note: ''\n"
        "    reviewed: {by: lee, on: 2026-10-17}\n"
        "  - id: ov-0002\n"
        "    kind: outcome\n"
        "    target: {battle: battle_of_actium}\n"
        "    value: {Octavian: victory, Antony: defeat}\n"
        "    source: Plutarch\n"
        "    note: ''\n"
        "    reviewed: {by: lee, on: 2026-10-17}\n",
        encoding="utf-8",
    )
    assert len(load_overrides(populated).entries) == 2
    monkeypatch.setattr("pipeline.overrides.DEFAULT_PATH", populated)
    stub_stage("classify")

    result = orchestrator.run_stage("classify", db_conn=ScriptedConnection([10, 2, 0]))

    assert result.success is True
    gate = [c for c in result.quality_checks if c.name == "overrides_applied"]
    assert gate and all(c.passed for c in gate)
