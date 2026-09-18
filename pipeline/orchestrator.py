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
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import structlog
import yaml

logger = structlog.get_logger()

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


class Severity(Enum):
    WARNING = "warning"
    ERROR = "error"


@dataclass
class QualityCheckResult:
    name: str
    passed: bool
    severity: Severity
    actual_value: Any = None
    threshold: str = ""
    message: str = ""


@dataclass
class StageResult:
    stage: str
    success: bool
    duration_seconds: float
    quality_checks: list[QualityCheckResult] = field(default_factory=list)
    error: str | None = None
    retries_used: int = 0


# ─── Agent spec loading ──────────────────────────────────────────────────────

def load_agent_spec(stage: str) -> dict:
    """Load the YAML agent spec for a pipeline stage."""
    spec_path = Path(f"agents/{stage}.yaml")
    if not spec_path.exists():
        raise FileNotFoundError(f"Agent spec not found: {spec_path}")
    with open(spec_path) as f:
        return yaml.safe_load(f)


# ─── Quality gate runner ─────────────────────────────────────────────────────

def run_quality_checks(spec: dict, db_conn: Any) -> list[QualityCheckResult]:
    """
    Run all quality checks defined in an agent spec.
    Returns a list of results. Any ERROR-severity failure blocks the pipeline.
    """
    results = []
    checks = spec.get("quality_checks", [])

    for check in checks:
        result = _run_single_check(check, db_conn)
        results.append(result)
        level = "error" if not result.passed and result.severity == Severity.ERROR else "warning"
        log_fn = logger.error if level == "error" else logger.warning if not result.passed else logger.info
        log_fn(
            "quality_check",
            name=result.name,
            passed=result.passed,
            actual=result.actual_value,
            threshold=result.threshold,
        )

    return results


def _run_single_check(check: dict, db_conn: Any) -> QualityCheckResult:
    """Execute a single quality check against the database."""
    name = check["name"]
    severity = Severity(check.get("severity", "warning"))
    threshold = check.get("threshold", "")
    sql = check.get("check", "")

    # TODO: implement actual DB query execution and threshold comparison.
    # For now, return a placeholder that passes.
    # The real implementation will:
    #   1. Execute the SQL query against db_conn
    #   2. Parse the threshold string (e.g. ">= 0.85", "< 100", "= 0")
    #   3. Compare and return pass/fail

    return QualityCheckResult(
        name=name,
        passed=True,  # placeholder
        severity=severity,
        threshold=threshold,
        message="Not yet implemented",
    )


# ─── Stage runner ────────────────────────────────────────────────────────────

def run_stage(stage: str, config_overrides: dict | None = None) -> StageResult:
    """
    Run a single pipeline stage.

    1. Load the agent spec
    2. Import and execute the stage's runner module
    3. Run quality checks
    4. Handle retries on failure
    """
    spec = load_agent_spec(stage)
    retry_policy = spec.get("retry_policy", {})
    max_retries = retry_policy.get("max_stage_retries", 1)

    if config_overrides:
        spec.setdefault("params", {}).update(config_overrides)

    for attempt in range(max_retries + 1):
        logger.info("stage_start", stage=stage, attempt=attempt + 1)
        start = time.time()

        try:
            # Import the stage runner module dynamically
            runner_module = importlib.import_module(f"pipeline.stages.{stage}")
            runner_module.run(spec)

            duration = time.time() - start

            # Run quality gates
            # TODO: pass actual DB connection
            checks = run_quality_checks(spec, db_conn=None)

            has_error = any(
                not c.passed and c.severity == Severity.ERROR for c in checks
            )

            if has_error and attempt < max_retries:
                logger.warning("stage_quality_failed_retrying", stage=stage, attempt=attempt + 1)
                continue

            return StageResult(
                stage=stage,
                success=not has_error,
                duration_seconds=duration,
                quality_checks=checks,
                retries_used=attempt,
            )

        except Exception as e:
            duration = time.time() - start
            logger.error("stage_exception", stage=stage, error=str(e), attempt=attempt + 1)

            if attempt < max_retries:
                backoff = retry_policy.get("backoff_base", 2.0) ** attempt
                logger.info("retrying_after_backoff", seconds=backoff)
                time.sleep(backoff)
                continue

            on_failure = retry_policy.get("on_failure", "pause_and_alert")

            return StageResult(
                stage=stage,
                success=False,
                duration_seconds=duration,
                error=str(e),
                retries_used=attempt,
            )

    # should not reach here, but just in case
    return StageResult(stage=stage, success=False, duration_seconds=0, error="Exhausted retries")


# ─── Pipeline orchestrator ───────────────────────────────────────────────────

def run_pipeline(
    stages: list[str],
    config_overrides: dict | None = None,
    stop_on_error: bool = True,
) -> list[StageResult]:
    """
    Run a sequence of pipeline stages in order.

    Args:
        stages: list of stage names, or ["all"] for the full pipeline.
        config_overrides: dict of param overrides passed to every stage.
        stop_on_error: if True, halt the pipeline on the first ERROR-severity failure.

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
        result = run_stage(stage, config_overrides)
        results.append(result)

        logger.info(
            "stage_complete",
            stage=stage,
            success=result.success,
            duration=f"{result.duration_seconds:.1f}s",
            retries=result.retries_used,
        )

        if not result.success and stop_on_error:
            logger.error("pipeline_halted", failed_stage=stage)
            break

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
        "--no-stop-on-error",
        action="store_true",
        help="Continue running stages even if one fails quality checks",
    )
    args = parser.parse_args()

    stages = [s.strip() for s in args.stages.split(",")]

    config_overrides = None
    if args.config:
        with open(args.config) as f:
            config_overrides = yaml.safe_load(f)

    results = run_pipeline(
        stages=stages,
        config_overrides=config_overrides,
        stop_on_error=not args.no_stop_on_error,
    )

    # Exit with non-zero if any stage failed
    if any(not r.success for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
