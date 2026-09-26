"""Unit tests for the quality gate runner.

The central guarantee under test is that a failing check actually fails. The
previous placeholder returned ``passed=True`` unconditionally, so every one of
these assertions would have passed vacuously against it except by accident.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

import pytest
import yaml

from pipeline.quality import (
    DEFAULT_MAX_RHAT,
    DEFAULT_MIN_ESS_BULK,
    QualityRunner,
    Severity,
    StageResult,
    ThresholdParseError,
    has_blocking_failure,
    parse_threshold,
)

AGENTS_DIR = Path(__file__).resolve().parents[2] / "agents"


# ─── Stub connection ─────────────────────────────────────────────────────────


class _StubResult:
    def __init__(self, value: Any) -> None:
        self._value = value

    def scalar(self) -> Any:
        return self._value


class StubConnection:
    """Returns a canned scalar, or raises, without needing a real database."""

    def __init__(self, value: Any = 0, raises: Exception | None = None) -> None:
        self._value = value
        self._raises = raises
        self.executed: list[Any] = []
        self.savepoints = 0

    def begin_nested(self) -> AbstractContextManager[None]:
        """Stand in for SQLAlchemy's SAVEPOINT context manager.

        The runner wraps every check in one so a failing statement cannot abort
        the transaction the later checks depend on. Rollback has no meaning
        against a stub, so this only records that a savepoint was taken; the
        isolation itself is proven against a real database in
        tests/integration/test_quality_gates.py.
        """
        self.savepoints += 1
        return nullcontext()

    def execute(self, statement: Any) -> _StubResult:
        self.executed.append(statement)
        if self._raises is not None:
            raise self._raises
        return _StubResult(self._value)


# ─── Threshold parsing ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "operator", "value"),
    [
        (">= 3000", ">=", 3000.0),
        ("< 0.10", "<", 0.10),
        ("= 0", "=", 0.0),
        ("< 2.0", "<", 2.0),
        ("> 0.55", ">", 0.55),
        ("< 100", "<", 100.0),
        ("< 200", "<", 200.0),
        ("< 50", "<", 50.0),
        (">= 0.85", ">=", 0.85),
        (">= 0.90", ">=", 0.90),
        (">= 0.99", ">=", 0.99),
        (">= 2.0", ">=", 2.0),
        ("> 0", ">", 0.0),
        ("p > 0.01", ">", 0.01),  # statistical tests carry a leading label
        ("  >=   1e3  ", ">=", 1000.0),  # whitespace and exponent notation
        ("<= -1.5", "<=", -1.5),
    ],
)
def test_parse_threshold_numeric_forms(raw: str, operator: str, value: float) -> None:
    parsed = parse_threshold(raw)
    assert parsed.operator == operator
    assert parsed.value == pytest.approx(value)


def test_parse_threshold_boolean() -> None:
    parsed = parse_threshold("= true")
    assert parsed.operator == "="
    assert parsed.value is True
    assert parsed.compare(True) is True
    assert parsed.compare(False) is False


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "definitely not a threshold",
        "all rhat < 1.02 and ess_bulk > 800 for skill parameters",
        ">=",
        "3000",
    ],
)
def test_parse_threshold_rejects_unparseable(raw: str) -> None:
    with pytest.raises(ThresholdParseError):
        parse_threshold(raw)


@pytest.mark.parametrize(
    ("raw", "actual", "expected"),
    [
        (">= 3000", 3000, True),
        (">= 3000", 2999, False),
        ("< 0.10", 0.09, True),
        ("< 0.10", 0.10, False),
        ("= 0", 0, True),
        ("= 0", 1, False),
        ("> 0.55", 0.56, True),
        ("> 0.55", 0.55, False),
        ("<= 5", 5, True),
        ("<= 5", 6, False),
    ],
)
def test_threshold_comparison_boundaries(raw: str, actual: float, expected: bool) -> None:
    assert parse_threshold(raw).compare(actual) is expected


def test_threshold_rejects_ordering_on_boolean() -> None:
    with pytest.raises(ThresholdParseError):
        parse_threshold("> true").compare(True)


# ─── SQL checks ──────────────────────────────────────────────────────────────


def test_sql_check_passes_when_threshold_met() -> None:
    check = {
        "name": "min_battles_crawled",
        "check": "SELECT COUNT(*) FROM crawl_log",
        "threshold": ">= 3000",
        "severity": "error",
    }
    result = QualityRunner(StubConnection(value=3500)).run_one(check)

    assert result.passed is True
    assert result.actual_value == 3500
    assert result.severity is Severity.ERROR


def test_sql_check_fails_when_threshold_missed() -> None:
    """The regression that matters: a failing check must report passed=False."""
    check = {
        "name": "min_battles_crawled",
        "check": "SELECT COUNT(*) FROM crawl_log",
        "threshold": ">= 3000",
        "severity": "error",
    }
    result = QualityRunner(StubConnection(value=12)).run_one(check)

    assert result.passed is False
    assert result.actual_value == 12
    assert "Expected >= 3000" in result.message


def test_sql_check_fails_without_connection() -> None:
    """No database must not be mistaken for a clean gate."""
    check = {
        "name": "some_check",
        "check": "SELECT 1",
        "threshold": "= 1",
        "severity": "error",
    }
    result = QualityRunner(None).run_one(check)

    assert result.passed is False
    assert "No database connection" in result.message


def test_sql_check_fails_on_null_result() -> None:
    check = {
        "name": "error_rate",
        "check": "SELECT COUNT(*) / NULLIF(COUNT(*), 0) FROM crawl_log",
        "threshold": "< 0.10",
        "severity": "warning",
    }
    result = QualityRunner(StubConnection(value=None)).run_one(check)

    assert result.passed is False
    assert result.actual_value is None
    assert "NULL" in result.message


def test_sql_check_fails_on_execution_error() -> None:
    check = {
        "name": "broken",
        "check": "SELECT * FROM table_that_does_not_exist",
        "threshold": "= 0",
        "severity": "error",
    }
    runner = QualityRunner(StubConnection(raises=RuntimeError("relation does not exist")))
    result = runner.run_one(check)

    assert result.passed is False
    assert "SQL execution failed" in result.message


def test_prose_check_without_method_fails_rather_than_passes() -> None:
    """agents/model.yaml carries prose in `check:` with no parseable threshold."""
    check = {
        "name": "skill_distribution_sensible",
        "check": "The posterior mean skill for the median general should be close to 0",
        "threshold": "",
        "severity": "warning",
    }
    result = QualityRunner(StubConnection(value=0)).run_one(check)

    assert result.passed is False
    assert "method" in result.message


# ─── Non-SQL method checks ───────────────────────────────────────────────────


# diagnostics_json is deliberately absent: it has a handler in
# METHOD_HANDLERS now, so it no longer reports "not implemented". The tests
# for its real behaviour are at the foot of this file.
@pytest.mark.parametrize(
    "method",
    [
        "posterior_check",
        "per_round_check",
        "range_check",
        "statistical_test",
        "brier_score",
        "jaccard_overlap",
    ],
)
def test_unimplemented_methods_fail_visibly(method: str) -> None:
    """An unimplemented gate must not silently pass."""
    check = {"name": f"check_{method}", "method": method, "severity": "error"}
    result = QualityRunner(None).run_one(check)

    assert result.passed is False
    assert "not implemented" in result.message


def test_unknown_method_fails() -> None:
    check = {"name": "mystery", "method": "telepathy", "severity": "warning"}
    result = QualityRunner(None).run_one(check)

    assert result.passed is False
    assert "Unknown check method" in result.message


# ─── Severity and blocking ───────────────────────────────────────────────────


def test_unknown_severity_defaults_to_warning() -> None:
    check = {"name": "x", "check": "SELECT 1", "threshold": "= 1", "severity": "catastrophic"}
    result = QualityRunner(StubConnection(value=1)).run_one(check)

    assert result.severity is Severity.WARNING


def test_has_blocking_failure_only_for_errors() -> None:
    checks = [
        {"name": "warn", "check": "SELECT 1", "threshold": "= 0", "severity": "warning"},
        {"name": "ok", "check": "SELECT 1", "threshold": "= 1", "severity": "error"},
    ]
    results = QualityRunner(StubConnection(value=1)).run_all(checks)

    assert [r.passed for r in results] == [False, True]
    assert has_blocking_failure(results) is False

    blocking = [
        {"name": "bad", "check": "SELECT 1", "threshold": "= 0", "severity": "error"},
    ]
    assert has_blocking_failure(QualityRunner(StubConnection(value=1)).run_all(blocking)) is True


def test_run_all_preserves_spec_order() -> None:
    checks = [
        {"name": "a", "check": "SELECT 1", "threshold": "= 1", "severity": "warning"},
        {"name": "b", "method": "brier_score", "severity": "warning"},
        {"name": "c", "check": "SELECT 1", "threshold": "= 1", "severity": "warning"},
    ]
    results = QualityRunner(StubConnection(value=1)).run_all(checks)

    assert [r.name for r in results] == ["a", "b", "c"]


# ─── StageResult halt semantics ──────────────────────────────────────────────


def test_continue_with_logging_does_not_halt() -> None:
    result = StageResult(
        stage="extract",
        success=False,
        duration_seconds=1.0,
        on_failure="continue_with_logging",
    )
    assert result.halts_pipeline is False


def test_pause_and_alert_halts() -> None:
    result = StageResult(
        stage="crawl",
        success=False,
        duration_seconds=1.0,
        on_failure="pause_and_alert",
    )
    assert result.halts_pipeline is True


def test_success_never_halts() -> None:
    result = StageResult(
        stage="crawl",
        success=True,
        duration_seconds=1.0,
        on_failure="pause_and_alert",
    )
    assert result.halts_pipeline is False


# ─── Every real spec's thresholds must be handled ────────────────────────────


def _spec_checks() -> list[tuple[str, dict[str, Any]]]:
    pairs: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(AGENTS_DIR.glob("*.yaml")):
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        for check in spec.get("quality_checks", []) or []:
            pairs.append((path.stem, check))
    return pairs


def test_agents_dir_has_checks_to_test() -> None:
    assert len(_spec_checks()) > 0, "No quality checks found in agents/*.yaml"


def test_every_spec_check_is_either_sql_or_method() -> None:
    """A check with neither a parseable threshold nor a method can never pass."""
    orphans = []
    for stage, check in _spec_checks():
        if check.get("method") is not None:
            continue
        try:
            parse_threshold(str(check.get("threshold", "")))
        except ThresholdParseError:
            orphans.append(f"{stage}.{check.get('name')}")

    assert orphans == [], (
        "These checks declare no `method:` and no parseable `threshold:`, so they "
        f"will always fail: {orphans}"
    )


def test_no_spec_check_sends_prose_to_the_database() -> None:
    """A check without a `method:` must hold real SQL, not an English description.

    Prose in `check:` reaches Postgres as a query and fails with a syntax error
    that reads like a database problem rather than a spec problem.
    """
    prose = []
    for stage, check in _spec_checks():
        if check.get("method") is not None:
            continue
        sql = str(check.get("check", "")).strip().upper()
        if not (sql.startswith("SELECT") or sql.startswith("WITH")):
            prose.append(f"{stage}.{check.get('name')}")

    assert prose == [], (
        "These checks declare no `method:` but their `check:` is not SQL, so the "
        f"text would be executed as a query: {prose}"
    )


def test_every_spec_method_is_registered() -> None:
    """A `method:` the runner does not know produces 'Unknown check method'."""
    from pipeline.quality import UNIMPLEMENTED_METHODS

    unregistered = {
        f"{stage}.{check.get('name')}": check["method"]
        for stage, check in _spec_checks()
        if check.get("method") is not None and check["method"] not in UNIMPLEMENTED_METHODS
    }

    assert unregistered == {}, (
        f"These specs name check methods the runner does not recognise: {unregistered}"
    )


# ─── diagnostics_json method handler ─────────────────────────────────────────


class _DiagnosticsRow:
    """One model_runs row as the handler's query returns it."""

    def __init__(self, run_id: int, diagnostics: Any) -> None:
        self.run_id = run_id
        self.diagnostics = diagnostics

    def __getitem__(self, index: int) -> Any:
        return (self.run_id, self.diagnostics)[index]


class _DiagnosticsResult:
    def __init__(self, row: _DiagnosticsRow | None) -> None:
        self._row = row

    def first(self) -> _DiagnosticsRow | None:
        return self._row

    def fetchone(self) -> _DiagnosticsRow | None:
        return self._row

    def one_or_none(self) -> _DiagnosticsRow | None:
        return self._row

    def mappings(self) -> _DiagnosticsResult:
        return self

    def scalar(self) -> Any:
        return None if self._row is None else self._row.run_id


class DiagnosticsConnection:
    """Serves scripted model_runs rows, newest first.

    The handler asks for one row with ORDER BY run_id DESC LIMIT 1, so this
    stub holds the rows in the order the database would return them and hands
    back the head of the list.
    """

    def __init__(self, rows: list[_DiagnosticsRow] | None = None) -> None:
        self.rows = rows or []
        self.executed: list[Any] = []
        self.savepoints = 0

    def begin_nested(self) -> AbstractContextManager[None]:
        self.savepoints += 1
        return nullcontext()

    def execute(self, statement: Any, parameters: Any = None) -> _DiagnosticsResult:
        self.executed.append((statement, parameters))
        return _DiagnosticsResult(self.rows[0] if self.rows else None)


def _diagnostics_check(**params: Any) -> dict[str, Any]:
    """Build a diagnostics_json check with the given params."""
    base: dict[str, Any] = {"model_type": "source_disagreement"}
    base.update(params)
    return {
        "name": "model_convergence",
        "check": "all rhat < 1.05 and ess_bulk > 400 for all parameters",
        "method": "diagnostics_json",
        "params": base,
        "severity": "error",
    }


def _worst(max_rhat: float, min_ess_bulk: float, divergences: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "worst": {
            "max_rhat": max_rhat,
            "min_ess_bulk": min_ess_bulk,
            "divergences": divergences,
        },
    }


def test_diagnostics_json_passes_when_the_latest_run_of_that_type_is_converged() -> None:
    conn = DiagnosticsConnection([_DiagnosticsRow(7, _worst(1.004, 1180.0, 0))])
    result = QualityRunner(conn).run_one(_diagnostics_check())
    assert result.passed, result.message


def test_diagnostics_json_fails_when_no_completed_run_of_that_model_type_exists() -> None:
    # An absent run must not read as a clean gate: before the stage has ever
    # run there is nothing to have converged.
    result = QualityRunner(DiagnosticsConnection([])).run_one(_diagnostics_check())
    assert not result.passed
    assert "source_disagreement" in result.message


def test_diagnostics_json_fails_rather_than_passes_when_diagnostics_is_null() -> None:
    # A run that finished without recording diagnostics is a gap in the audit
    # trail, not evidence of convergence.
    conn = DiagnosticsConnection([_DiagnosticsRow(3, None)])
    assert not QualityRunner(conn).run_one(_diagnostics_check()).passed


def test_diagnostics_json_fails_when_the_check_declares_no_model_type() -> None:
    # Defaulting would read some other stage's run and pass this stage on a
    # different model's convergence.
    check = _diagnostics_check()
    del check["params"]
    conn = DiagnosticsConnection([_DiagnosticsRow(7, _worst(1.0, 5000.0, 0))])
    result = QualityRunner(conn).run_one(check)
    assert not result.passed
    assert "model_type" in result.message


def test_diagnostics_json_reads_its_thresholds_from_the_check_params_when_given() -> None:
    # rhat 1.03 clears the module default of 1.05 but not a spec that asks for
    # 1.01, so the spec's number has to be the one that decides.
    conn = DiagnosticsConnection([_DiagnosticsRow(7, _worst(1.03, 5000.0, 0))])
    strict = QualityRunner(conn).run_one(_diagnostics_check(max_rhat=1.01))
    lenient = QualityRunner(conn).run_one(_diagnostics_check(max_rhat=1.10))
    assert not strict.passed
    assert lenient.passed


def test_diagnostics_json_falls_back_to_the_module_default_thresholds() -> None:
    conn = DiagnosticsConnection([_DiagnosticsRow(7, _worst(1.04, DEFAULT_MIN_ESS_BULK + 1, 0))])
    assert QualityRunner(conn).run_one(_diagnostics_check()).passed

    conn = DiagnosticsConnection([_DiagnosticsRow(7, _worst(DEFAULT_MAX_RHAT + 0.01, 5000.0, 0))])
    assert not QualityRunner(conn).run_one(_diagnostics_check()).passed


def test_diagnostics_json_fails_when_divergences_exceed_the_limit() -> None:
    # Divergent transitions mean the sampler did not explore the posterior, so
    # a clean rhat alongside them is not reassurance.
    conn = DiagnosticsConnection([_DiagnosticsRow(7, _worst(1.001, 5000.0, 4))])
    result = QualityRunner(conn).run_one(_diagnostics_check())
    assert not result.passed
    assert "divergence" in result.message.lower()


def test_diagnostics_json_rolls_up_when_the_payload_has_no_worst_object() -> None:
    # An older schema_version wrote per-model objects and no rollup. The gate
    # must still take the worst across them rather than reporting no metrics.
    payload = {
        "schema_version": 1,
        "troops": {"max_rhat": 1.002, "min_ess_bulk": 800.0, "divergences": 0},
        "casualties": {"max_rhat": 1.30, "min_ess_bulk": 600.0, "divergences": 0},
    }
    conn = DiagnosticsConnection([_DiagnosticsRow(7, payload)])
    result = QualityRunner(conn).run_one(_diagnostics_check())
    assert not result.passed, "the worse of the two rhats should decide"


def test_diagnostics_json_reports_no_connection_rather_than_passing() -> None:
    result = QualityRunner(None).run_one(_diagnostics_check())
    assert not result.passed
    assert "connection" in result.message.lower()


def test_diagnostics_json_takes_a_savepoint_like_every_other_check() -> None:
    # Postgres aborts the whole transaction on a failed statement, so a check
    # that does not isolate itself makes every later check report someone
    # else's error instead of its own verdict.
    conn = DiagnosticsConnection([_DiagnosticsRow(7, _worst(1.0, 5000.0, 0))])
    QualityRunner(conn).run_one(_diagnostics_check())
    assert conn.savepoints == 1


def test_every_registered_handler_is_also_a_declared_method() -> None:
    """METHOD_HANDLERS and UNIMPLEMENTED_METHODS must not drift apart."""
    from pipeline.quality import METHOD_HANDLERS, UNIMPLEMENTED_METHODS

    assert set(METHOD_HANDLERS) <= set(UNIMPLEMENTED_METHODS)


def test_every_diagnostics_json_check_in_every_spec_declares_a_model_type() -> None:
    """A diagnostics_json gate with no model_type can never pass."""
    missing = [
        f"{stage}.{check.get('name')}"
        for stage, check in _spec_checks()
        if check.get("method") == "diagnostics_json"
        and not (check.get("params") or {}).get("model_type")
    ]

    assert missing == [], (
        "These diagnostics_json checks name no params.model_type, so the handler "
        f"cannot tell which model run to read: {missing}"
    )
