"""
Quality gate execution for the General WAR pipeline.

Each agent spec in ``agents/*.yaml`` declares a ``quality_checks`` list. Most
checks are SQL queries returning a single scalar, compared against a threshold
string such as ``">= 3000"``. A minority declare a ``method`` instead, naming a
non-SQL check (posterior diagnostics, distributional tests) that needs the
stage's own artefacts rather than the database.

This module executes both kinds. Unimplemented ``method`` checks return a
*failing* result rather than a passing one, so that a gap in coverage stays
visible instead of silently green-lighting a stage.

Usage:
    runner = QualityRunner(conn)
    results = runner.run_all(spec.get("quality_checks", []))
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

import structlog

logger = structlog.get_logger()


# ─── Result types ────────────────────────────────────────────────────────────
# Defined here rather than in orchestrator.py so that orchestrator can import
# from quality without a circular import.


class Severity(Enum):
    """How a failing check should be treated by the orchestrator."""

    WARNING = "warning"
    ERROR = "error"


@dataclass
class QualityCheckResult:
    """Outcome of a single quality check."""

    name: str
    passed: bool
    severity: Severity
    actual_value: Any = None
    threshold: str = ""
    message: str = ""


@dataclass
class StageResult:
    """Outcome of one stage run, including its quality gate results."""

    stage: str
    success: bool
    duration_seconds: float
    quality_checks: list[QualityCheckResult] = field(default_factory=list)
    error: str | None = None
    retries_used: int = 0
    # The spec's retry_policy.on_failure directive. "continue_with_logging"
    # means a failure of this stage should not halt the pipeline.
    on_failure: str = "pause_and_alert"

    @property
    def halts_pipeline(self) -> bool:
        """Whether this result should stop a pipeline running with stop_on_error."""
        return not self.success and self.on_failure != "continue_with_logging"


# ─── Threshold parsing ───────────────────────────────────────────────────────

# Thresholds in the specs take forms like ">= 3000", "< 0.10", "= 0", "< 2.0",
# "> 0.55", "p > 0.01" (statistical tests carry a leading label) and "= true".
_THRESHOLD_RE = re.compile(
    r"""^\s*
        (?:[A-Za-z_][A-Za-z0-9_]*\s*)?      # optional leading label, e.g. "p"
        (?P<op>>=|<=|!=|>|<|=)              # comparison operator
        \s*
        (?P<value>true|false|[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)
        \s*$
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Absolute tolerance for float equality. Most "=" thresholds compare integer
# counts, where this is irrelevant; it matters only for ratio-valued checks.
_FLOAT_ABS_TOL = 1e-9


class ThresholdParseError(ValueError):
    """Raised when a threshold string cannot be interpreted."""


@dataclass(frozen=True)
class Threshold:
    """A parsed comparison, e.g. ``>=`` against ``3000.0``."""

    operator: str
    value: float | bool

    def compare(self, actual: float | bool) -> bool:
        """Evaluate ``actual <operator> value``.

        Args:
            actual: The scalar returned by the check.

        Returns:
            True if the comparison holds.

        Raises:
            ThresholdParseError: If the operator is not valid for the operand type.
        """
        if isinstance(self.value, bool) or isinstance(actual, bool):
            # Only equality is meaningful for booleans.
            actual_bool = bool(actual)
            value_bool = bool(self.value)
            if self.operator == "=":
                return actual_bool == value_bool
            if self.operator == "!=":
                return actual_bool != value_bool
            raise ThresholdParseError(
                f"Operator {self.operator!r} is not valid for a boolean threshold"
            )

        actual_num = float(actual)
        value_num = float(self.value)

        if self.operator == ">=":
            return actual_num >= value_num
        if self.operator == "<=":
            return actual_num <= value_num
        if self.operator == ">":
            return actual_num > value_num
        if self.operator == "<":
            return actual_num < value_num
        if self.operator == "=":
            return math.isclose(actual_num, value_num, abs_tol=_FLOAT_ABS_TOL)
        if self.operator == "!=":
            return not math.isclose(actual_num, value_num, abs_tol=_FLOAT_ABS_TOL)

        raise ThresholdParseError(f"Unknown operator: {self.operator!r}")


def parse_threshold(raw: str) -> Threshold:
    """Parse a threshold string from an agent spec.

    Handles every form present in ``agents/*.yaml``: ``">= 3000"``, ``"< 0.10"``,
    ``"= 0"``, ``"> 0.55"``, ``"p > 0.01"`` and ``"= true"``.

    Args:
        raw: The threshold string as written in the spec.

    Returns:
        A Threshold ready to compare against.

    Raises:
        ThresholdParseError: If the string does not match a supported form.
    """
    if not raw or not raw.strip():
        raise ThresholdParseError("Threshold is empty")

    match = _THRESHOLD_RE.match(raw)
    if match is None:
        raise ThresholdParseError(f"Cannot parse threshold: {raw!r}")

    op = match.group("op")
    value_text = match.group("value").lower()

    value: float | bool
    if value_text == "true":
        value = True
    elif value_text == "false":
        value = False
    else:
        value = float(value_text)

    return Threshold(operator=op, value=value)


# ─── Database access ─────────────────────────────────────────────────────────


class ScalarResult(Protocol):
    """Minimal protocol for a result object exposing a single scalar."""

    def scalar(self) -> Any: ...


class Connection(Protocol):
    """Minimal protocol for a database connection.

    Satisfied by a SQLAlchemy Core ``Connection``. Declared structurally so the
    runner can be exercised in tests with a stub, and so this module does not
    hard-depend on ``pipeline.db`` existing yet.
    """

    def execute(self, statement: Any) -> ScalarResult: ...


def _to_statement(sql: str) -> Any:
    """Wrap raw SQL for execution.

    SQLAlchemy 2.x requires textual SQL to be wrapped in ``text()``. If
    SQLAlchemy is not importable, the raw string is returned so that stub
    connections in tests still work.

    Args:
        sql: The SQL text from the agent spec.

    Returns:
        A SQLAlchemy ``TextClause``, or the original string as a fallback.
    """
    try:
        from sqlalchemy import text
    except ImportError:  # pragma: no cover - exercised only without SQLAlchemy
        logger.debug("sqlalchemy_unavailable_using_raw_sql")
        return sql
    return text(sql)


# ─── Non-SQL check handlers ──────────────────────────────────────────────────

# Several specs declare `method:` instead of SQL. These need artefacts the
# stage itself produces (ArviZ InferenceData, imputed datasets, held-out
# predictions) rather than a database query. They are registered here as
# explicitly unimplemented so that the gap is reported, not silently passed.
#
# To implement one, replace the entry with a callable taking (check, context)
# and returning a QualityCheckResult.
UNIMPLEMENTED_METHODS: dict[str, str] = {
    "diagnostics_json": (
        "Requires model_runs.diagnostics from the reconcile/model stage "
        "(rhat, ess_bulk, divergences via ArviZ)"
    ),
    "posterior_check": "Requires the fitted posterior from the model stage",
    "per_round_check": "Requires the imputed datasets in data/imputed/round_{n}/",
    "range_check": "Requires the imputed values from the impute stage",
    "statistical_test": "Requires observed and imputed distributions for a KS test",
    "brier_score": "Requires held-out predictions from the evaluate stage",
    "jaccard_overlap": "Requires sensitivity variant rankings from the evaluate stage",
    "held_out_accuracy": "Requires the k-fold held-out predictions from the evaluate stage",
}


# ─── Runner ──────────────────────────────────────────────────────────────────


class QualityRunner:
    """Executes the quality checks declared in an agent spec."""

    def __init__(self, conn: Connection | None) -> None:
        """
        Args:
            conn: An open database connection, or None if unavailable. With no
                connection every SQL check fails with an explanatory message
                rather than passing vacuously.
        """
        self._conn = conn

    def run_all(self, checks: list[dict[str, Any]]) -> list[QualityCheckResult]:
        """Run every check in a spec's ``quality_checks`` list.

        Args:
            checks: The raw check dicts loaded from the agent spec.

        Returns:
            One result per check, in spec order.
        """
        results = [self.run_one(check) for check in checks]

        for result in results:
            if result.passed:
                log_fn = logger.info
            elif result.severity is Severity.ERROR:
                log_fn = logger.error
            else:
                log_fn = logger.warning
            log_fn(
                "quality_check",
                name=result.name,
                passed=result.passed,
                actual=result.actual_value,
                threshold=result.threshold,
                message=result.message,
            )

        return results

    def run_one(self, check: dict[str, Any]) -> QualityCheckResult:
        """Run a single quality check.

        Args:
            check: A check dict with at least ``name``; plus either ``check``
                (SQL) or ``method`` (non-SQL), ``threshold``, and ``severity``.

        Returns:
            The check's result. Never raises: an unexpected failure is reported
            as a failing result so one bad check cannot abort the gate.
        """
        name = str(check.get("name", "<unnamed>"))
        threshold_raw = str(check.get("threshold", ""))

        try:
            severity = Severity(check.get("severity", "warning"))
        except ValueError:
            severity = Severity.WARNING
            logger.warning(
                "unknown_severity_defaulting_to_warning",
                name=name,
                severity=check.get("severity"),
            )

        method = check.get("method")
        if method is not None:
            return self._run_method_check(name, str(method), severity, threshold_raw)

        return self._run_sql_check(name, check, severity, threshold_raw)

    def _run_method_check(
        self,
        name: str,
        method: str,
        severity: Severity,
        threshold_raw: str,
    ) -> QualityCheckResult:
        """Handle a check declaring a non-SQL ``method``."""
        reason = UNIMPLEMENTED_METHODS.get(method)
        detail = reason or f"Unknown check method: {method!r}"
        return QualityCheckResult(
            name=name,
            passed=False,
            severity=severity,
            threshold=threshold_raw,
            message=f"Check method {method!r} is not implemented. {detail}",
        )

    def _run_sql_check(
        self,
        name: str,
        check: dict[str, Any],
        severity: Severity,
        threshold_raw: str,
    ) -> QualityCheckResult:
        """Execute a SQL check and compare its scalar against the threshold."""
        sql = str(check.get("check", "")).strip()
        if not sql:
            return QualityCheckResult(
                name=name,
                passed=False,
                severity=severity,
                threshold=threshold_raw,
                message="Check defines neither SQL nor a method",
            )

        try:
            threshold = parse_threshold(threshold_raw)
        except ThresholdParseError as exc:
            return QualityCheckResult(
                name=name,
                passed=False,
                severity=severity,
                threshold=threshold_raw,
                message=(
                    f"{exc}. The check text may be a prose description rather "
                    "than SQL; give it a `method:` key instead."
                ),
            )

        if self._conn is None:
            return QualityCheckResult(
                name=name,
                passed=False,
                severity=severity,
                threshold=threshold_raw,
                message="No database connection available to run this check",
            )

        try:
            actual = self._conn.execute(_to_statement(sql)).scalar()
        except Exception as exc:
            logger.error("quality_check_sql_failed", name=name, error=str(exc))
            return QualityCheckResult(
                name=name,
                passed=False,
                severity=severity,
                threshold=threshold_raw,
                message=f"SQL execution failed: {exc}",
            )

        if actual is None:
            return QualityCheckResult(
                name=name,
                passed=False,
                severity=severity,
                actual_value=None,
                threshold=threshold_raw,
                message=(
                    "Check returned NULL. This usually means the table is empty "
                    "or a NULLIF guard divided by zero."
                ),
            )

        try:
            passed = threshold.compare(actual)
        except (ThresholdParseError, TypeError, ValueError) as exc:
            return QualityCheckResult(
                name=name,
                passed=False,
                severity=severity,
                actual_value=actual,
                threshold=threshold_raw,
                message=f"Could not compare result to threshold: {exc}",
            )

        return QualityCheckResult(
            name=name,
            passed=passed,
            severity=severity,
            actual_value=actual,
            threshold=threshold_raw,
            message="" if passed else f"Expected {threshold_raw}, got {actual}",
        )


def has_blocking_failure(results: list[QualityCheckResult]) -> bool:
    """Report whether any ERROR-severity check failed.

    Args:
        results: Results from a quality gate run.

    Returns:
        True if the pipeline should treat the stage as failed.
    """
    return any(not r.passed and r.severity is Severity.ERROR for r in results)
