"""
Pipeline orchestrator for the General WAR project.

Runs stages in sequence, checks quality gates between stages,
handles retries, and logs everything. Designed to be invoked by
Claude Code as an agentic loop or run standalone from the CLI.

Usage:
    python -m pipeline.orchestrator --stages all
    python -m pipeline.orchestrator --stages crawl,extract
    python -m pipeline.orchestrator --stages model --config config/model_alt.yaml
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import sys
import time
from typing import Any

import structlog
import yaml

from pipeline.config import SpecError, load_env, merge_overrides, parse_set_overrides
from pipeline.config import load_agent_spec as _load_spec
from pipeline.db import DatabaseConfigError, get_connection, get_engine
from pipeline.logging_config import configure as configure_logging
from pipeline.quality import (
    QualityCheckResult,
    QualityRunner,
    Severity,
    StageResult,
    has_blocking_failure,
)
from pipeline.stages.base import StageConformanceError, StageContext, check_conforms

logger = structlog.get_logger()

__all__ = [
    "STAGE_ORDER",
    "SpecError",
    "QualityCheckResult",
    "Severity",
    "StageResult",
    "load_agent_spec",
    "run_pipeline",
    "run_quality_checks",
    "run_stage",
]

# ─── Stage definitions ───────────────────────────────────────────────────────

STAGE_ORDER = [
    "crawl",
    "extract",
    "resolve",
    "reconcile",
    "classify",
    "impute",
    "model",
    "evaluate",
    "report",
]


# Severity, QualityCheckResult and StageResult are defined in pipeline.quality
# and re-exported above, so that quality.py can own them without a circular
# import back into this module.


# ─── Agent spec loading ──────────────────────────────────────────────────────


def load_agent_spec(stage: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load the YAML agent spec for a pipeline stage.

    Delegates to pipeline.config so that spec validation and override
    merging behave identically here and in the tooling commands.

    Args:
        stage: The stage name.
        overrides: Parameter overrides to merge into the spec.

    Returns:
        The loaded spec.

    Raises:
        SpecError: If the spec is missing or malformed.
    """
    return _load_spec(stage, overrides)


# ─── Quality gate runner ─────────────────────────────────────────────────────


def run_quality_checks(spec: dict[str, Any], db_conn: Any) -> list[QualityCheckResult]:
    """
    Run all quality checks defined in an agent spec.

    Args:
        spec: The loaded agent spec.
        db_conn: An open database connection, or None. With no connection the
            SQL checks fail rather than pass, so an unavailable database is
            never mistaken for a clean gate.

    Returns:
        A list of results. Any ERROR-severity failure blocks the pipeline.
    """
    params = spec.get("params")
    runner = QualityRunner(db_conn, params if isinstance(params, dict) else None)
    return runner.run_all(spec.get("quality_checks", []) or [])


# ─── Stage runner ────────────────────────────────────────────────────────────

# A stage module sets this True when it manages its own connection on purpose.
# reconcile, impute and model commit a run row before sampling so that a failed
# fit leaves an inspectable incomplete run; on the orchestrator's shared
# connection that row would be rolled back with the failure. Such a stage is
# handed a context with no connection and opens its own, as it did before.
OWNS_CONNECTION_ATTR = "OWNS_DB_CONNECTION"


def _end_transaction(db_conn: Any, *, commit: bool, stage: str) -> None:
    """Commit or roll back the shared connection after a stage attempt.

    Rule: a stage's writes are committed as soon as its ``run`` returns, whether
    or not it deferred LLM requests, and before the quality gates. That makes
    them visible to the next stage, to a stage's own connection and to the CLI
    user. If ``run`` raised, the attempt is rolled back so a retry or the next
    stage never inherits a half-written, possibly aborted transaction.

    Args:
        db_conn: The shared connection, or None.
        commit: True to commit, False to roll back.
        stage: The stage name, for the log.
    """
    if db_conn is None:
        return
    try:
        if commit:
            db_conn.commit()
        else:
            db_conn.rollback()
    except Exception as exc:
        logger.error(
            "stage_transaction_end_failed",
            stage=stage,
            action="commit" if commit else "rollback",
            error=str(exc),
        )
        if commit:
            raise


def _call_stage(runner_module: Any, spec: dict[str, Any], db_conn: Any, stage: str) -> Any:
    """Call a stage's ``run``, handing it the shared connection when it can use one.

    Args:
        runner_module: The imported stage module.
        spec: The loaded agent spec.
        db_conn: The shared connection, or None.
        stage: The stage name, for the log.

    Returns:
        Whatever ``run`` returns: None or a StageOutcome.
    """
    run = runner_module.run
    owns = bool(getattr(runner_module, OWNS_CONNECTION_ATTR, False))
    context = StageContext(db_conn=None if owns else db_conn)
    # check_conforms lets a stage declare run(spec) alone, so only pass a
    # context to a stage that has somewhere to put it.
    params = list(inspect.signature(run).parameters.values())
    accepts_context = any(p.kind is p.VAR_POSITIONAL for p in params) or (
        len([p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]) >= 2
    )
    outcome = run(spec, context) if accepts_context else run(spec)
    _end_transaction(db_conn, commit=True, stage=stage)
    return outcome


def run_stage(
    stage: str,
    config_overrides: dict[str, Any] | None = None,
    db_conn: Any = None,
) -> StageResult:
    """
    Run a single pipeline stage.

    1. Load the agent spec
    2. Import and execute the stage's runner module
    3. Run quality checks
    4. Handle retries on failure

    Args:
        stage: The stage name; must appear in STAGE_ORDER.
        config_overrides: Parameter overrides merged into the spec's params.
        db_conn: An open database connection used to run the quality gate.

    Returns:
        The stage's result, including every quality check that ran.
    """
    spec = load_agent_spec(stage)

    if config_overrides:
        spec = merge_overrides(spec, config_overrides)

    retry_policy = spec.get("retry_policy", {}) or {}
    max_retries = retry_policy.get("max_stage_retries", 1)
    on_failure = retry_policy.get("on_failure", "pause_and_alert")
    # Off by default: reconcile, impute and model sample with a fixed seed, so
    # re-running after a gate failure reproduces it and doubles the runtime.
    # Only a stage that resumes from saved state (crawl) makes progress on retry.
    retry_on_gate_failure = bool(retry_policy.get("retry_on_gate_failure", False))

    for attempt in range(max_retries + 1):
        logger.info("stage_start", stage=stage, attempt=attempt + 1)
        start = time.time()

        try:
            # Import the stage runner module dynamically, then check it
            # satisfies the StageRunner contract before calling it. A stage
            # with a mistyped or mis-signed run() should fail here, not after
            # the upstream stages have already done hours of work.
            runner_module = importlib.import_module(f"pipeline.stages.{stage}")
            try:
                check_conforms(runner_module)
            except StageConformanceError as exc:
                # A broken contract is a code defect, not a transient fault.
                # Retrying cannot fix it, so fail immediately with the reason.
                logger.error("stage_contract_violation", stage=stage, error=str(exc))
                return StageResult(
                    stage=stage,
                    success=False,
                    duration_seconds=time.time() - start,
                    error=str(exc),
                    retries_used=attempt,
                    on_failure=on_failure,
                )
            try:
                outcome = _call_stage(runner_module, spec, db_conn, stage)
            except Exception:
                _end_transaction(db_conn, commit=False, stage=stage)
                raise

            duration = time.time() - start

            if outcome is not None and getattr(outcome, "awaiting_llm", False):
                # The stage's deterministic outputs are written, but the
                # gates measure completeness and would fail on data that is
                # only waiting for its LLM pass. Reporting that as a stage
                # failure would blame the code for a pending step, and
                # retrying cannot help, so the gates are skipped and the
                # result is marked incomplete instead.
                batch = str(outcome.batch_file) if outcome.batch_file else None
                logger.warning(
                    "stage_quality_gates_skipped_awaiting_llm",
                    stage=stage,
                    deferred_requests=outcome.deferred_requests,
                    batch_file=batch,
                )
                return StageResult(
                    stage=stage,
                    success=True,
                    duration_seconds=duration,
                    retries_used=attempt,
                    on_failure=on_failure,
                    deferred_requests=outcome.deferred_requests,
                    batch_file=batch,
                )

            # Run quality gates
            checks = run_quality_checks(spec, db_conn=db_conn)

            has_error = has_blocking_failure(checks)

            if has_error and retry_on_gate_failure and attempt < max_retries:
                logger.warning("stage_quality_failed_retrying", stage=stage, attempt=attempt + 1)
                continue

            return StageResult(
                stage=stage,
                success=not has_error,
                duration_seconds=duration,
                quality_checks=checks,
                retries_used=attempt,
                on_failure=on_failure,
            )

        except Exception as e:
            duration = time.time() - start
            logger.error("stage_exception", stage=stage, error=str(e), attempt=attempt + 1)

            if attempt < max_retries:
                backoff = retry_policy.get("backoff_base", 2.0) ** attempt
                logger.info("retrying_after_backoff", seconds=backoff)
                time.sleep(backoff)
                continue

            logger.error("stage_failed", stage=stage, on_failure=on_failure)

            return StageResult(
                stage=stage,
                success=False,
                duration_seconds=duration,
                error=str(e),
                retries_used=attempt,
                on_failure=on_failure,
            )

    # should not reach here, but just in case
    return StageResult(
        stage=stage,
        success=False,
        duration_seconds=0,
        error="Exhausted retries",
        on_failure=on_failure,
    )


# ─── Pipeline orchestrator ───────────────────────────────────────────────────

# Distinct from 1 (a stage failed) so a wrapper script can tell "needs the
# offline batch processed" from "something broke".
EXIT_AWAITING_LLM = 3


def _announce_deferred(result: StageResult, *, continuing: bool = False) -> None:
    """Tell the user which batch file to process and what to run next.

    Args:
        result: The result of a stage that deferred LLM requests.
        continuing: True if the pipeline carries on past this stage.
    """
    logger.warning(
        "stage_deferred_continuing" if continuing else "pipeline_stopped_awaiting_offline_llm",
        stage=result.stage,
        deferred_requests=result.deferred_requests,
        batch_file=result.batch_file,
        reason=(
            "later stages run on deterministic data only; the result is provisional"
            if continuing
            else "later stages would run on incomplete data"
        ),
        next_step=(
            f"run /process-llm-batch {result.batch_file} in Claude Code, then re-run: "
            f"python -m pipeline.orchestrator --stages {result.stage}"
        ),
    )


def _stage_is_built(stage: str) -> bool:
    """Report whether ``pipeline.stages.<stage>`` exists.

    Args:
        stage: The stage name.

    Returns:
        True if an importable module exists for the stage.
    """
    module_name = f"pipeline.stages.{stage}"
    if module_name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def _log_deferred_summary(results: list[StageResult]) -> None:
    """Log every stage that deferred LLM work, with its batch file.

    Args:
        results: The results of the whole run.
    """
    deferred = [r for r in results if r.awaiting_llm]
    if not deferred:
        return
    logger.warning(
        "pipeline_deferred_summary",
        provisional=True,
        stages=[
            {"stage": r.stage, "deferred_requests": r.deferred_requests, "batch_file": r.batch_file}
            for r in deferred
        ],
        next_step=(
            "run /process-llm-batch on each batch file, then re-run the listed stages "
            "and everything after them"
        ),
    )


def run_pipeline(
    stages: list[str],
    config_overrides: dict[str, Any] | None = None,
    stop_on_error: bool = True,
    db_conn: Any = None,
    continue_on_deferred: bool = False,
) -> list[StageResult]:
    """
    Run a sequence of pipeline stages in order.

    Args:
        stages: list of stage names, or ["all"] for the full pipeline.
        config_overrides: dict of param overrides passed to every stage.
        stop_on_error: if True, halt the pipeline on the first ERROR-severity failure.
        db_conn: an open database connection used to run each stage's quality gate.
        continue_on_deferred: if True, a stage that deferred offline LLM work
            does not halt the pipeline. Later stages then run on deterministic
            data only, so the result is provisional.

    Returns:
        List of StageResult for each stage that ran.
    """
    if stages == ["all"]:
        stages = [s for s in STAGE_ORDER if _stage_is_built(s)]
        skipped = [s for s in STAGE_ORDER if s not in stages]
        if skipped:
            logger.warning("unbuilt_stages_skipped", stages=skipped)

    # Validate before running anything, so a typo or an unbuilt stage does not
    # surface only after the upstream stages have done hours of work.
    for s in stages:
        if s not in STAGE_ORDER:
            logger.error("unknown_stage", stage=s, valid=STAGE_ORDER)
            sys.exit(1)
        if not _stage_is_built(s):
            logger.error("stage_not_built", stage=s, reason=f"no module pipeline/stages/{s}.py")
            sys.exit(1)

    results = []
    for stage in stages:
        result = run_stage(stage, config_overrides, db_conn=db_conn)
        results.append(result)

        logger.info(
            "stage_complete",
            stage=stage,
            success=result.success,
            duration=f"{result.duration_seconds:.1f}s",
            retries=result.retries_used,
        )

        if result.awaiting_llm:
            _announce_deferred(result, continuing=continue_on_deferred)
            if continue_on_deferred:
                continue
            break

        if not result.success:
            if stop_on_error and result.halts_pipeline:
                logger.error("pipeline_halted", failed_stage=stage)
                break
            # The spec set on_failure: continue_with_logging, so a failure here
            # is recorded but the pipeline carries on.
            logger.warning(
                "stage_failed_continuing",
                failed_stage=stage,
                on_failure=result.on_failure,
            )

    _log_deferred_summary(results)

    # Summary
    total_duration = sum(r.duration_seconds for r in results)
    n_passed = sum(1 for r in results if r.success)
    logger.info(
        "pipeline_summary",
        stages_run=len(results),
        stages_passed=n_passed,
        total_duration=f"{total_duration:.1f}s",
    )

    return results


# ─── CLI ─────────────────────────────────────────────────────────────────────


def main() -> None:
    """Run the pipeline from the command line.

    Loads ``.env``, configures logging, opens a database connection and runs
    the requested stages. The connection is the point: the quality gates are
    SQL, and without one every gate reported "No database connection
    available to run this check" rather than its verdict, which made a CLI
    run unable to tell a passing stage from a failing one.
    """
    parser = argparse.ArgumentParser(description="General WAR pipeline orchestrator")
    parser.add_argument(
        "--stages",
        type=str,
        default="all",
        help="Comma-separated stage names, or 'all' for the full pipeline",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file with parameter overrides",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        dest="set_overrides",
        help="Override a spec param, e.g. --set mcmc_samples=5000. Repeatable.",
    )
    parser.add_argument(
        "--no-stop-on-error",
        action="store_true",
        help="Continue running stages even if one fails quality checks",
    )
    parser.add_argument(
        "--continue-past-deferred",
        action="store_true",
        help=(
            "Carry on to the next stage when a stage defers offline LLM work "
            "instead of halting. Later stages then run on deterministic data "
            "only (unresolved LLM groups stay unresolved, multi-commander "
            "sides keep default_split), so the result is provisional. Exit "
            "code 3 if anything deferred and nothing failed."
        ),
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help=(
            "Run without a database connection. Every SQL quality gate then "
            "fails as unrunnable rather than passing, so this is for "
            "inspecting stage mechanics, not for a real run."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="DEBUG, INFO, WARNING, ERROR or CRITICAL. Defaults to $LOG_LEVEL, then INFO.",
    )
    args = parser.parse_args()

    # Order matters: .env may carry LOG_LEVEL and DATABASE_URL, so it has to
    # be read before logging is configured and before the engine is built.
    load_env()
    configure_logging(level=args.log_level)

    stages = [s.strip() for s in args.stages.split(",")]

    config_overrides: dict[str, Any] = {}
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            config_overrides.update(yaml.safe_load(f) or {})
    if args.set_overrides:
        # --set wins over --config, being the more specific instruction.
        config_overrides.update(parse_set_overrides(args.set_overrides))

    def run(db_conn: Any) -> list[StageResult]:
        return run_pipeline(
            stages=stages,
            config_overrides=config_overrides or None,
            stop_on_error=not args.no_stop_on_error,
            db_conn=db_conn,
            continue_on_deferred=args.continue_past_deferred,
        )

    if args.no_db:
        logger.warning("running_without_database", consequence="every SQL gate will fail")
        results = run(None)
    else:
        try:
            engine = get_engine()
        except DatabaseConfigError as exc:
            # A missing DATABASE_URL is a configuration bug, not a data
            # condition. Continuing would run every stage and then report
            # nine identical gate failures, burying the actual cause.
            logger.error("database_not_configured", error=str(exc))
            sys.exit(2)

        try:
            with get_connection(engine=engine) as conn:
                results = run(conn)
        finally:
            engine.dispose()

    # Exit with non-zero if any stage failed
    if any(not r.success for r in results):
        sys.exit(1)
    if any(r.awaiting_llm for r in results):
        sys.exit(EXIT_AWAITING_LLM)


if __name__ == "__main__":
    main()
