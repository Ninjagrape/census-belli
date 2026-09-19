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
from pipeline.stages.base import StageConformanceError, check_conforms

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

def load_agent_spec(
    stage: str, overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
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
    runner = QualityRunner(db_conn)
    return runner.run_all(spec.get("quality_checks", []) or [])


# ─── Stage runner ────────────────────────────────────────────────────────────

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
    retry_policy = spec.get("retry_policy", {}) or {}
    max_retries = retry_policy.get("max_stage_retries", 1)
    on_failure = retry_policy.get("on_failure", "pause_and_alert")

    if config_overrides:
        spec = merge_overrides(spec, config_overrides)

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
            runner_module.run(spec)

            duration = time.time() - start

            # Run quality gates
            checks = run_quality_checks(spec, db_conn=db_conn)

            has_error = has_blocking_failure(checks)

            if has_error and attempt < max_retries:
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

def run_pipeline(
    stages: list[str],
    config_overrides: dict[str, Any] | None = None,
    stop_on_error: bool = True,
    db_conn: Any = None,
) -> list[StageResult]:
    """
    Run a sequence of pipeline stages in order.

    Args:
        stages: list of stage names, or ["all"] for the full pipeline.
        config_overrides: dict of param overrides passed to every stage.
        stop_on_error: if True, halt the pipeline on the first ERROR-severity failure.
        db_conn: an open database connection used to run each stage's quality gate.

    Returns:
        List of StageResult for each stage that ran.
    """
    if stages == ["all"]:
        stages = STAGE_ORDER

    # Validate stage names
    for s in stages:
        if s not in STAGE_ORDER:
            logger.error("unknown_stage", stage=s, valid=STAGE_ORDER)
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


if __name__ == "__main__":
    main()
