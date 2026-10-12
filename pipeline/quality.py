"""
Quality gate execution for the General WAR pipeline.

Each agent spec in ``agents/*.yaml`` declares a ``quality_checks`` list. Most
checks are SQL queries returning a single scalar, compared against a threshold
string such as ``">= 3000"``. A minority declare a ``method`` instead, naming a
non-SQL check (posterior diagnostics, distributional tests) that needs the
stage's own artefacts rather than the database.

This module executes both kinds. A ``method`` with a handler registered in
``METHOD_HANDLERS`` is run by that handler; one without returns a *failing*
result rather than a passing one, so that a gap in coverage stays visible
instead of silently green-lighting a stage.

Usage:
    runner = QualityRunner(conn)
    results = runner.run_all(spec.get("quality_checks", []))
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final, Protocol, cast

import structlog
from sqlalchemy import text

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
    # LLM requests the stage wrote to a batch file instead of answering. A
    # stage that deferred work is incomplete rather than failed: its own
    # deterministic outputs exist, its quality gates were not run, and the
    # pipeline stops until the answers are imported and the stage is re-run.
    deferred_requests: int = 0
    batch_file: str | None = None

    @property
    def awaiting_llm(self) -> bool:
        """Whether the stage is waiting on offline LLM answers."""
        return self.deferred_requests > 0

    @property
    def halts_pipeline(self) -> bool:
        """Whether this result should stop a pipeline running with stop_on_error."""
        if self.awaiting_llm:
            return True
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


class ThresholdParamError(ValueError):
    """Raised when a threshold names a ``{params.x}`` the spec does not define."""


_PARAM_PLACEHOLDER_RE = re.compile(r"\{params\.([A-Za-z_][A-Za-z0-9_]*)\}")


def resolve_threshold_params(raw: str, params: dict[str, Any]) -> str:
    """Substitute ``{params.<name>}`` placeholders in a threshold string.

    Args:
        raw: The threshold as written in the spec.
        params: The stage spec's ``params`` after overrides.

    Returns:
        The threshold with every placeholder replaced.

    Raises:
        ThresholdParamError: If a placeholder names a missing or null param, so
            the check fails instead of comparing against nothing.
    """

    def substitute(match: re.Match[str]) -> str:
        key = match.group(1)
        value = params.get(key)
        if value is None:
            raise ThresholdParamError(
                f"Threshold {raw!r} references params.{key}, which the stage spec does not define"
            )
        return str(value)

    return _PARAM_PLACEHOLDER_RE.sub(substitute, raw)


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


class RowResult(Protocol):
    """Minimal protocol for a result object exposing its first row.

    The SQL check path needs only a scalar, so ``Connection.execute`` is
    declared as returning one. A handler reading several columns of a row casts
    to this instead; a SQLAlchemy ``Result`` satisfies both.
    """

    def first(self) -> Any: ...


class Connection(Protocol):
    """Minimal protocol for a database connection.

    Satisfied by a SQLAlchemy Core ``Connection``. Declared structurally so the
    runner can be exercised in tests with a stub, and so this module does not
    hard-depend on ``pipeline.db`` existing yet.
    """

    def execute(self, statement: Any) -> ScalarResult: ...

    def begin_nested(self) -> AbstractContextManager[Any]:
        """Open a SAVEPOINT, rolled back if the block raises.

        Part of the protocol because gate isolation depends on it: a check whose
        SQL fails must not abort the transaction the remaining checks run in.
        """
        ...


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

# Fallback sampler thresholds, used when a check declares no `params` of its
# own. Deliberately looser than any spec's prose: a gate that has not stated
# its own numbers should catch an unusable fit, not adjudicate a marginal one.
DEFAULT_MAX_RHAT: Final[float] = 1.05
DEFAULT_MIN_ESS_BULK: Final[float] = 400.0
DEFAULT_MAX_DIVERGENCES: Final[int] = 0

# The diagnostics of the newest completed run of one model type. Reading them
# from model_runs rather than an ArviZ file on disk is what lets this handler
# work with nothing but the connection the runner already holds.
_LATEST_DIAGNOSTICS = text(
    "SELECT run_id, diagnostics FROM model_runs "
    "WHERE model_type = :model_type AND completed_at IS NOT NULL "
    "ORDER BY run_id DESC LIMIT 1"
)

# The keys a diagnostics payload carries, in the rolled-up `worst` object the
# sampling stages write and in whatever per-parameter objects preceded it.
_RHAT_KEY: Final[str] = "max_rhat"
_ESS_KEY: Final[str] = "min_ess_bulk"
_DIVERGENCES_KEY: Final[str] = "divergences"


@dataclass(frozen=True)
class _DiagnosticsMetrics:
    """The three sampler metrics a convergence gate compares against."""

    max_rhat: float | None = None
    min_ess_bulk: float | None = None
    divergences: int | None = None

    @property
    def is_empty(self) -> bool:
        """Whether the payload yielded no recognisable metric at all."""
        return self.max_rhat is None and self.min_ess_bulk is None and self.divergences is None

    def summarise(self) -> str:
        """Render the metrics as a short one-line summary.

        Returns:
            Text of the form ``max_rhat=1.01, min_ess_bulk=1200, divergences=0``,
            with ``n/a`` for any metric the payload did not carry.
        """
        parts = [
            f"{_RHAT_KEY}={'n/a' if self.max_rhat is None else self.max_rhat}",
            f"{_ESS_KEY}={'n/a' if self.min_ess_bulk is None else self.min_ess_bulk}",
            f"{_DIVERGENCES_KEY}={'n/a' if self.divergences is None else self.divergences}",
        ]
        return ", ".join(parts)


def _as_float(value: Any) -> float | None:
    """Coerce a diagnostics value to a float, or None if it is not numeric.

    Args:
        value: A value read from the diagnostics JSON.

    Returns:
        The float, or None for anything non-numeric (including booleans, which
        are numeric in Python but never a sampler metric).
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _extract_metrics(payload: dict[str, Any]) -> _DiagnosticsMetrics:
    """Read the sampler metrics out of one diagnostics object.

    Args:
        payload: A diagnostics object, either the rolled-up ``worst`` or the
            whole JSON payload for an older schema.

    Returns:
        The metrics it carries; any key it lacks stays None.
    """
    divergences = _as_float(payload.get(_DIVERGENCES_KEY))
    return _DiagnosticsMetrics(
        max_rhat=_as_float(payload.get(_RHAT_KEY)),
        min_ess_bulk=_as_float(payload.get(_ESS_KEY)),
        divergences=None if divergences is None else int(divergences),
    )


def _roll_up_metrics(diagnostics: dict[str, Any]) -> _DiagnosticsMetrics:
    """Reduce a whole diagnostics payload to its worst case.

    Prefers the ``worst`` object the sampling stages write. Falling back to a
    roll-up across the payload's own keys and its nested objects keeps a run
    written under an older ``schema_version`` gateable rather than unreadable,
    which would otherwise read as a passing gate's silence.

    Args:
        diagnostics: The parsed ``model_runs.diagnostics`` JSON.

    Returns:
        The worst rhat, worst ESS and total divergences found.
    """
    worst = diagnostics.get("worst")
    if isinstance(worst, dict):
        return _extract_metrics(worst)

    candidates = [diagnostics]
    candidates.extend(value for value in diagnostics.values() if isinstance(value, dict))

    max_rhat: float | None = None
    min_ess_bulk: float | None = None
    divergences: int | None = None

    for candidate in candidates:
        found = _extract_metrics(candidate)
        if found.max_rhat is not None:
            max_rhat = found.max_rhat if max_rhat is None else max(max_rhat, found.max_rhat)
        if found.min_ess_bulk is not None:
            min_ess_bulk = (
                found.min_ess_bulk
                if min_ess_bulk is None
                else min(min_ess_bulk, found.min_ess_bulk)
            )
        if found.divergences is not None:
            divergences = (
                found.divergences if divergences is None else divergences + found.divergences
            )
    return _DiagnosticsMetrics(
        max_rhat=max_rhat, min_ess_bulk=min_ess_bulk, divergences=divergences
    )


def _threshold_float(params: dict[str, Any], key: str, default: float) -> float:
    """Read a numeric threshold from a check's params, falling back to a default.

    Args:
        params: The check's ``params`` mapping.
        key: The param name.
        default: The module default to use when the param is absent or unusable.

    Returns:
        The threshold to compare against.
    """
    value = _as_float(params.get(key))
    return default if value is None else value


def _check_diagnostics_json(
    name: str,
    check: dict[str, Any],
    severity: Severity,
    conn: Connection | None,
) -> QualityCheckResult:
    """Gate a stage on the sampler diagnostics of its newest completed run.

    Reads ``model_runs.diagnostics`` for the newest completed run of the
    ``model_type`` the check names, and compares its worst rhat, worst bulk ESS
    and divergence count against the check's thresholds.

    Args:
        name: The check's name, as declared in the agent spec.
        check: The raw check dict; ``params.model_type`` is required, and
            ``params.max_rhat`` / ``min_ess_bulk`` / ``max_divergences`` override
            the module defaults.
        severity: The severity to report the result at.
        conn: An open database connection, or None.

    Returns:
        The check's result. Never raises: every failure path, including a
        database error, is reported as a failing result.
    """
    threshold_raw = str(check.get("threshold", ""))
    raw_params = check.get("params")
    params: dict[str, Any] = raw_params if isinstance(raw_params, dict) else {}

    def failure(
        message: str,
        actual: Any = None,
        threshold: str = threshold_raw,
    ) -> QualityCheckResult:
        """Build a failing result for this check."""
        return QualityCheckResult(
            name=name,
            passed=False,
            severity=severity,
            actual_value=actual,
            threshold=threshold,
            message=message,
        )

    model_type = params.get("model_type")
    if not isinstance(model_type, str) or not model_type.strip():
        # Defaulting would read some other stage's run and pass this stage on
        # another model's convergence.
        return failure(
            "Check declares no `params.model_type`, so there is no way to tell "
            "which model run's diagnostics it should read."
        )
    model_type = model_type.strip()

    max_rhat_limit = _threshold_float(params, "max_rhat", DEFAULT_MAX_RHAT)
    min_ess_limit = _threshold_float(params, "min_ess_bulk", DEFAULT_MIN_ESS_BULK)
    max_divergences_limit = int(
        _threshold_float(params, "max_divergences", float(DEFAULT_MAX_DIVERGENCES))
    )
    effective = (
        f"{_RHAT_KEY} <= {max_rhat_limit}, "
        f"{_ESS_KEY} >= {min_ess_limit}, "
        f"{_DIVERGENCES_KEY} <= {max_divergences_limit}"
    )
    threshold_text = threshold_raw or effective

    if conn is None:
        return failure(
            "No database connection available to run this check",
            threshold=threshold_text,
        )

    try:
        # Same savepoint isolation as the SQL path: a statement that fails
        # against Postgres aborts the surrounding transaction, and every later
        # check would then report that abort instead of its own verdict.
        statement = _LATEST_DIAGNOSTICS.bindparams(model_type=model_type)
        with conn.begin_nested():
            row = cast(RowResult, conn.execute(statement)).first()
    except Exception as exc:
        logger.error("diagnostics_check_query_failed", name=name, error=str(exc))
        return failure(
            f"Reading model_runs.diagnostics failed: {type(exc).__name__}: {exc}",
            threshold=threshold_text,
        )

    if row is None:
        return failure(
            f"No completed run of model_type {model_type!r} exists, so there are "
            "no diagnostics to check. Run the stage that fits it first.",
            threshold=threshold_text,
        )

    run_id, diagnostics = row[0], row[1]

    if not isinstance(diagnostics, dict) or not diagnostics:
        # A run that recorded no diagnostics is unverified, not converged.
        return failure(
            f"Run {run_id} of model_type {model_type!r} recorded no diagnostics, "
            "so its convergence is unknown.",
            threshold=threshold_text,
        )

    metrics = _roll_up_metrics(diagnostics)
    if metrics.is_empty:
        return failure(
            f"Run {run_id} of model_type {model_type!r} has a diagnostics payload "
            f"with no recognisable metrics: expected a `worst` object carrying "
            f"{_RHAT_KEY}, {_ESS_KEY} and {_DIVERGENCES_KEY}.",
            threshold=threshold_text,
        )

    actual = f"run {run_id}: {metrics.summarise()}"
    breaches: list[str] = []
    # A stage that pools several fits writes its own all-fits verdict. When the
    # key is present it must be 'true'; runs without it are judged on metrics only.
    converged_flag = diagnostics.get("converged")
    if converged_flag is not None and str(converged_flag).lower() != "true":
        breaches.append(f"diagnostics.converged is {converged_flag!r}, not 'true'")
    if metrics.max_rhat is not None and metrics.max_rhat > max_rhat_limit:
        breaches.append(f"{_RHAT_KEY} {metrics.max_rhat} > {max_rhat_limit}")
    if metrics.min_ess_bulk is not None and metrics.min_ess_bulk < min_ess_limit:
        breaches.append(f"{_ESS_KEY} {metrics.min_ess_bulk} < {min_ess_limit}")
    if metrics.divergences is not None and metrics.divergences > max_divergences_limit:
        breaches.append(f"{_DIVERGENCES_KEY} {metrics.divergences} > {max_divergences_limit}")

    if breaches:
        return failure(
            f"Run {run_id} did not converge: {'; '.join(breaches)}",
            actual=actual,
            threshold=threshold_text,
        )

    return QualityCheckResult(
        name=name,
        passed=True,
        severity=severity,
        actual_value=actual,
        threshold=threshold_text,
    )


def _gate_result(
    name: str,
    check: dict[str, Any],
    severity: Severity,
    path: Path,
    gate_name: str,
    *,
    missing_hint: str,
    absent_hint: str,
) -> QualityCheckResult:
    """Report one gate from a ``gate_report.json`` written by a stage.

    Shared by the impute and model handlers: both stages compute their
    dataset-level gates before writing and record them in a JSON report, and
    the verdict must be about exactly what is on disk.

    Args:
        name: The check's name, as declared in the agent spec.
        check: The raw check dict (for the threshold text).
        severity: The severity to report the result at.
        path: The report file.
        gate_name: The gate to look up in the report.
        missing_hint: Appended when the file cannot be read.
        absent_hint: Appended when the report has no such gate.

    Returns:
        The check's result. Never raises.
    """
    threshold_raw = str(check.get("threshold", ""))

    def failure(message: str) -> QualityCheckResult:
        return QualityCheckResult(
            name=name, passed=False, severity=severity, threshold=threshold_raw, message=message
        )

    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return failure(f"Cannot read {path}: {type(exc).__name__}. {missing_hint}")

    gates = report.get("gates") if isinstance(report, dict) else None
    entry = next(
        (g for g in gates or [] if isinstance(g, dict) and g.get("name") == gate_name), None
    )
    if entry is None:
        return failure(f"Gate {gate_name!r} is absent from {path}; {absent_hint}")
    passed = bool(entry.get("passed"))
    return QualityCheckResult(
        name=name,
        passed=passed,
        severity=severity,
        actual_value=entry.get("actual"),
        threshold=str(entry.get("threshold", threshold_raw)),
        message="" if passed else str(entry.get("message", "")),
    )


def _check_imputation_gate(
    name: str,
    check: dict[str, Any],
    severity: Severity,
    conn: Connection | None,
) -> QualityCheckResult:
    """Report one dataset-level gate recorded by the impute stage.

    The impute stage computes its gates from the imputed rounds before writing
    and records them in ``{root}/{scale}/gate_report.json``. This handler reads that file rather
    than recomputing, so the verdict is about exactly the rounds on disk.

    Args:
        name: The check's name, as declared in the agent spec.
        check: The raw check dict; ``params.gate`` names the gate (default: the
            check's own name) and ``params.root`` the imputed-data root, ``params.scale`` the scale
            whose report to read (default ``neutral``, the primary).
        severity: The severity to report the result at.
        conn: Unused; the gate reads a file, not the database.

    Returns:
        The check's result. Never raises.
    """
    del conn
    raw_params = check.get("params")
    params: dict[str, Any] = raw_params if isinstance(raw_params, dict) else {}
    path = (
        Path(str(params.get("root", "data/imputed")))
        / str(params.get("scale", "neutral"))
        / "gate_report.json"
    )
    written: Any = None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        written = report.get("rounds_written") if isinstance(report, dict) else None
    except (OSError, ValueError):
        pass
    return _gate_result(
        name,
        check,
        severity,
        path,
        str(params.get("gate", name)),
        missing_hint="Run the impute stage first.",
        absent_hint=(
            f"(rounds_written={written}) the fit did not converge, so no rounds exist to check."
        ),
    )


_LATEST_POOLED_RUN = text(
    "SELECT MAX(run_id) FROM model_runs WHERE model_type = 'bradley_terry_hierarchical'"
)
_MODEL_RUN_DIR = re.compile(r"run(\d+)$")


def _latest_model_run_dir(root: Path, conn: Connection | None) -> Path | None:
    """The diagnostics directory of the newest pooled model run.

    Prefers the newest ``bradley_terry_hierarchical`` row in the database, so
    the gate describes the run the convergence gates read; falls back to the
    highest ``run{N}`` directory when there is no connection or no such row.

    Args:
        root: The diagnostics root.
        conn: A database connection, or None.

    Returns:
        The directory, or None when none exists.
    """
    if conn is not None:
        try:
            with conn.begin_nested():
                run_id = conn.execute(_LATEST_POOLED_RUN).scalar()
        except Exception as exc:
            logger.warning("model_gate_run_lookup_failed", error=str(exc))
            run_id = None
        if run_id is not None:
            return root / f"run{int(run_id)}"
    if not root.is_dir():
        return None
    numbered = [
        (int(m.group(1)), entry)
        for entry in root.iterdir()
        if entry.is_dir() and (m := _MODEL_RUN_DIR.fullmatch(entry.name))
    ]
    return max(numbered)[1] if numbered else None


def _check_model_gate(
    name: str,
    check: dict[str, Any],
    severity: Severity,
    conn: Connection | None,
) -> QualityCheckResult:
    """Report one dataset-level gate recorded by the model stage.

    The model stage records its gates in
    ``{root}/run{pooled_run_id}/gate_report.json``. When a round failed to
    converge the stage writes no gates, so every one of them reports absent and
    the convergence gate carries the real reason.

    Args:
        name: The check's name, as declared in the agent spec.
        check: The raw check dict; ``params.gate`` names the gate (default: the
            check's own name) and ``params.root`` the diagnostics root.
        severity: The severity to report the result at.
        conn: A database connection used to find the newest pooled run.

    Returns:
        The check's result. Never raises.
    """
    raw_params = check.get("params")
    params: dict[str, Any] = raw_params if isinstance(raw_params, dict) else {}
    root = Path(str(params.get("root", "data/processed/model_diagnostics")))
    directory = _latest_model_run_dir(root, conn)
    if directory is None:
        return QualityCheckResult(
            name=name,
            passed=False,
            severity=severity,
            threshold=str(check.get("threshold", "")),
            message=f"No model run directory under {root}. Run the model stage first.",
        )
    return _gate_result(
        name,
        check,
        severity,
        directory / "gate_report.json",
        str(params.get("gate", name)),
        missing_hint="Run the model stage first.",
        absent_hint="a round did not converge, so the stage evaluated no gates.",
    )


MethodHandler = Callable[[str, dict[str, Any], Severity, "Connection | None"], QualityCheckResult]

# A `method:` listed here is executed by its handler. Anything else falls
# through to UNIMPLEMENTED_METHODS for its reason.
METHOD_HANDLERS: dict[str, MethodHandler] = {
    "diagnostics_json": _check_diagnostics_json,
    "imputation_gate": _check_imputation_gate,
    "model_gate": _check_model_gate,
}

# Handler param -> stage spec param it must follow. A method listed here reads
# its input from where the run wrote it, which the run's params decide.
_SPEC_PARAM_BINDINGS: dict[str, dict[str, str]] = {
    "imputation_gate": {"scale": "scale", "root": "imputed_root"},
}

# Why each `method:` without a handler cannot run yet. Every method named in
# any agents/*.yaml must appear here (tests/unit/test_quality.py asserts it),
# including the ones METHOD_HANDLERS implements: this is the reason table the
# runner falls back to, not a list of what is missing. A method with neither a
# handler nor an entry reports "Unknown check method", which is a spec typo
# rather than a gap in coverage.
#
# To implement one, add a handler to METHOD_HANDLERS; the entry here then
# becomes its fallback reason and is never used.
UNIMPLEMENTED_METHODS: dict[str, str] = {
    "diagnostics_json": (
        "Requires model_runs.diagnostics from the reconcile/model stage "
        "(rhat, ess_bulk, divergences via ArviZ)"
    ),
    "posterior_check": "Requires the fitted posterior from the model stage",
    "imputation_gate": "Requires data/imputed/{scale}/gate_report.json from the impute stage",
    "model_gate": (
        "Requires data/processed/model_diagnostics/run{N}/gate_report.json from the model stage"
    ),
    "brier_score": "Requires held-out predictions from the evaluate stage",
    "jaccard_overlap": "Requires sensitivity variant rankings from the evaluate stage",
    "held_out_accuracy": "Requires the k-fold held-out predictions from the evaluate stage",
}


# ─── Runner ──────────────────────────────────────────────────────────────────


class QualityRunner:
    """Executes the quality checks declared in an agent spec."""

    def __init__(self, conn: Connection | None, spec_params: dict[str, Any] | None = None) -> None:
        """
        Args:
            conn: An open database connection, or None if unavailable. With no
                connection every SQL check fails with an explanatory message
                rather than passing vacuously.
            spec_params: The stage spec's ``params`` after overrides. Used to
                resolve ``{params.<name>}`` in thresholds and to point method
                handlers at the output the run actually wrote.
        """
        self._conn = conn
        self._spec_params = spec_params or {}

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

        try:
            threshold_raw = resolve_threshold_params(threshold_raw, self._spec_params)
        except ThresholdParamError as exc:
            return QualityCheckResult(
                name=name,
                passed=False,
                severity=severity,
                threshold=threshold_raw,
                message=str(exc),
            )
        check = {**check, "threshold": threshold_raw}

        method = check.get("method")
        if method is not None:
            return self._run_method_check(name, check, severity, threshold_raw)

        return self._run_sql_check(name, check, severity, threshold_raw)

    def _run_method_check(
        self,
        name: str,
        check: dict[str, Any],
        severity: Severity,
        threshold_raw: str,
    ) -> QualityCheckResult:
        """Handle a check declaring a non-SQL ``method``.

        A method with an entry in :data:`METHOD_HANDLERS` is executed by that
        handler. Anything else reports as explicitly unimplemented, so a gap in
        coverage stays visible instead of silently green-lighting a stage.

        Args:
            name: The check's name, as declared in the agent spec.
            check: The raw check dict. The whole dict is passed, not just the
                method name, because a handler reads its thresholds from
                ``check["params"]``.
            severity: The severity to report the result at.
            threshold_raw: The spec's raw threshold string, for the result.

        Returns:
            The check's result.
        """
        method = str(check.get("method", ""))

        handler = METHOD_HANDLERS.get(method)
        if handler is not None:
            return handler(name, self._bind_spec_params(method, check), severity, self._conn)

        reason = UNIMPLEMENTED_METHODS.get(method)
        detail = reason or f"Unknown check method: {method!r}"
        return QualityCheckResult(
            name=name,
            passed=False,
            severity=severity,
            threshold=threshold_raw,
            message=f"Check method {method!r} is not implemented. {detail}",
        )

    def _bind_spec_params(self, method: str, check: dict[str, Any]) -> dict[str, Any]:
        """Overlay the run's spec params onto a method check's own ``params``.

        The check's literal ``params`` in the YAML are static defaults; the run
        may have been started with ``--set scale=published``. Without this the
        handler reads the default scale's report instead of the one written.

        Args:
            method: The check's ``method`` name.
            check: The raw check dict.

        Returns:
            A copy of the check with the bound params, or the check unchanged.
        """
        bindings = _SPEC_PARAM_BINDINGS.get(method)
        if not bindings:
            return check
        raw = check.get("params")
        params: dict[str, Any] = dict(raw) if isinstance(raw, dict) else {}
        for check_key, spec_key in bindings.items():
            if self._spec_params.get(spec_key) is not None:
                params[check_key] = self._spec_params[spec_key]
        return {**check, "params": params}

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
            # Each check runs inside its own SAVEPOINT. Postgres aborts the
            # whole transaction on a failed statement, so without this one bad
            # check makes every later check in the same run report "current
            # transaction is aborted" instead of its own result -- a single
            # broken gate silently invalidates every gate after it.
            with self._conn.begin_nested():
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
