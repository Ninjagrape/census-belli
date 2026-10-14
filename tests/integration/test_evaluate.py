"""
Integration tests for the evaluate stage, against a real PostgreSQL.

Seeds the same small simulated tournament as ``test_model.py``, runs the real
model stage with tiny sampler settings, then runs the evaluate stage through
the orchestrator's own path (``run_stage`` with no shared connection, so the
stage opens its own, which is what the CLI does). Every output goes under
``tmp_path`` via spec params: ``output_root``, ``report_path`` and, through a
temporary copy of the model spec, ``diagnostics_root`` and ``imputed_root``.

Sampling is tiny and the write thresholds relaxed: the point is plumbing and
the read-only contract, not convergence or accuracy, which
``tests/model/test_evaluate_cv.py`` owns.

Seeded data is committed (module-scoped fixture on its own connection). Tests
that need a different database state (no target run, a newer model run) change
it inside a transaction the ``conn`` fixture rolls back.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import structlog
import yaml
from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from pipeline.config import load_agent_spec
from pipeline.db import apply_schema, get_engine
from pipeline.orchestrator import run_stage
from pipeline.quality import QualityRunner
from pipeline.stages import evaluate as evaluate_stage
from pipeline.stages.base import StageContext, check_conforms
from tests.integration.test_model import (
    _AVAILABLE,
    _BUMP_SEQUENCE,
    _INSERT_IMPUTE_RUN,
    _REASON,
    FAKE_IMPUTE_RUN_ID,
    POOLED,
    _count_runs,
    _run_committed,
    _seed,
    _spec,
    _write_rounds,
)
from tests.model.model_sim import SimTournament, simulate_tournament

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _AVAILABLE,
        reason=f"No database available ({_REASON}). Run `make db-up` first.",
    ),
]

REPORT_NAME = "evaluation_report.json"
GATE_NAMES = {
    "held_out_accuracy",
    "calibration",
    "cv_converged",
    "ranking_stability",
    "named_checks",
}
TOP_LEVEL_KEYS = {
    "schema_version",
    "model_run_id",
    "imputation_run_id",
    "created_at",
    "counts",
    "settings",
    "disclosures",
    "cv",
    "baselines",
    "sensitivity",
    "named_checks",
    "references",
    "ppc",
    "figures",
    "gates",
    "timings_seconds",
}
# Small enough to run in minutes, large enough that every default variant has
# something to do.
EVAL_PARAMS: dict[str, Any] = {
    "held_out_folds": 2,
    "cv_rounds": 1,
    "sensitivity_rounds": 1,
    "eval_draws": 150,
    "top_n": 5,
    "min_top_overlap": 2,
}


@pytest.fixture(scope="module")
def engine() -> Iterator[Engine]:
    """Provide an engine against a freshly applied schema."""
    eng = get_engine()
    db_name = (eng.url.database or "").lower()
    if not any(token in db_name for token in ("test", "general_war")):
        eng.dispose()
        pytest.skip(f"Refusing to reset database {db_name!r}: not a test database.")
    apply_schema(eng, drop_existing=True)
    yield eng
    eng.dispose()


@pytest.fixture
def conn(engine: Engine) -> Iterator[Connection]:
    """Give each test a transaction, rolled back afterwards."""
    connection = engine.connect()
    transaction = connection.begin()
    try:
        yield connection
    finally:
        transaction.rollback()
        connection.close()


@pytest.fixture(scope="module")
def tournament() -> SimTournament:
    """The same small two-era tournament test_model.py uses."""
    return simulate_tournament(3, generals_per_era=8, battles_per_era=45)


@pytest.fixture(scope="module")
def fitted(
    engine: Engine, tournament: SimTournament, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Any]:
    """Seed (committed), then run the model stage once; return paths and ids."""
    setup = engine.connect()
    try:
        db_side = _seed(setup, tournament)
        setup.execute(_INSERT_IMPUTE_RUN, {"run_id": FAKE_IMPUTE_RUN_ID})
        setup.execute(_BUMP_SEQUENCE)
        setup.commit()
    except BaseException:
        setup.rollback()
        raise
    finally:
        setup.close()
    root = tmp_path_factory.mktemp("imputed")
    _write_rounds(root, tournament, db_side, FAKE_IMPUTE_RUN_ID)
    diagnostics = tmp_path_factory.mktemp("diagnostics")
    seeded = {"root": root}
    model_spec = _spec(seeded, diagnostics)
    _run_committed(engine, model_spec)
    model_spec_path = tmp_path_factory.mktemp("specs") / "model.yaml"
    model_spec_path.write_text(yaml.safe_dump(model_spec), encoding="utf-8")
    with engine.connect() as probe:
        pooled_id = int(
            probe.execute(
                text("SELECT MAX(run_id) FROM model_runs WHERE model_type = :t"), {"t": POOLED}
            ).scalar_one()
        )
    return {"model_spec_path": model_spec_path, "pooled_id": pooled_id, "root": root}


def _overrides(fitted: dict[str, Any], out: Path) -> dict[str, Any]:
    """Spec params that send every evaluate output under ``out``."""
    return {
        **EVAL_PARAMS,
        "model_spec": str(fitted["model_spec_path"]),
        "output_root": str(out / "evaluation"),
        "report_path": str(out / REPORT_NAME),
    }


def _counts(engine: Engine) -> dict[str, int]:
    with engine.connect() as probe:
        return {
            "model_runs": _count_runs(engine),
            "max_run_id": int(
                probe.execute(text("SELECT MAX(run_id) FROM model_runs")).scalar_one()
            ),
            "estimates": int(
                probe.execute(text("SELECT COUNT(*) FROM general_skill_estimates")).scalar_one()
            ),
            "details": int(
                probe.execute(text("SELECT COUNT(*) FROM battle_war_details")).scalar_one()
            ),
        }


def _reject_constant(token: str) -> Any:
    raise ValueError(f"non-strict JSON constant {token}")


def _strict_load(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    assert isinstance(loaded, dict)
    return loaded


@pytest.fixture(scope="module")
def evaluated(
    engine: Engine, fitted: dict[str, Any], tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Any]:
    """Run evaluate once through ``run_stage`` with no shared connection.

    This is the call the CLI makes: the stage opens its own connection, then the
    orchestrator runs the spec's quality checks.
    """
    out = tmp_path_factory.mktemp("evaluate_out")
    before = _counts(engine)
    result = run_stage("evaluate", _overrides(fitted, out), db_conn=None)
    after = _counts(engine)
    return {"out": out, "result": result, "before": before, "after": after}


# ─── Tests ───────────────────────────────────────────────────────────────────


def test_stage_satisfies_the_contract() -> None:
    check_conforms(evaluate_stage)


def test_the_orchestrator_path_completes(evaluated: dict[str, Any]) -> None:
    result = evaluated["result"]

    assert result.success, result.error
    assert result.error is None
    assert {c.name for c in result.quality_checks} == GATE_NAMES


def test_evaluate_writes_no_database_rows(evaluated: dict[str, Any]) -> None:
    assert evaluated["after"] == evaluated["before"]


def test_report_is_strict_json_at_both_locations_for_the_pooled_run(
    fitted: dict[str, Any], evaluated: dict[str, Any]
) -> None:
    out: Path = evaluated["out"]
    run_dir = out / "evaluation" / f"run{fitted['pooled_id']}"

    latest = _strict_load(out / REPORT_NAME)
    copy = _strict_load(run_dir / REPORT_NAME)

    assert latest == copy
    assert (out / REPORT_NAME).read_bytes() == (run_dir / REPORT_NAME).read_bytes()
    assert latest["model_run_id"] == fitted["pooled_id"]
    assert latest["imputation_run_id"] == FAKE_IMPUTE_RUN_ID
    assert set(latest) >= TOP_LEVEL_KEYS
    assert {"imputed", "joint"} <= set(latest["cv"])
    assert {"baseline", "variants", "noise_floor"} <= set(latest["sensitivity"])
    assert latest["disclosures"]
    assert latest["counts"]["n_battles"] > 0


def test_report_carries_every_spec_gate(
    evaluated: dict[str, Any],
) -> None:
    spec = load_agent_spec("evaluate")
    spec_gates = {c["params"]["gate"] for c in spec["quality_checks"]}
    report = _strict_load(evaluated["out"] / REPORT_NAME)

    assert spec_gates == GATE_NAMES
    assert {g["name"] for g in report["gates"]} == spec_gates
    for gate in report["gates"]:
        assert {"passed", "actual", "threshold", "message"} <= set(gate)
        assert isinstance(gate["passed"], bool)


def test_outputs_directory_has_predictions_sensitivity_and_figures(
    fitted: dict[str, Any], evaluated: dict[str, Any]
) -> None:
    out: Path = evaluated["out"]
    run_dir = out / "evaluation" / f"run{fitted['pooled_id']}"
    report = _strict_load(out / REPORT_NAME)

    assert (run_dir / "cv_predictions.csv").is_file()
    assert (run_dir / "sensitivity" / "baseline.csv").is_file()
    assert report["figures"]
    for figure in report["figures"]:
        assert Path(figure).is_file(), figure
        assert Path(figure).is_relative_to(out)


def test_every_variant_in_the_spec_is_reported(evaluated: dict[str, Any]) -> None:
    spec = load_agent_spec("evaluate")
    declared = [v["name"] for v in spec["params"]["sensitivity_variants"]]
    report = _strict_load(evaluated["out"] / REPORT_NAME)

    reported = [v["name"] for v in report["sensitivity"]["variants"]]

    assert reported == declared
    assert report["sensitivity"]["baseline"]["converged"] is True


def test_the_quality_runner_executes_every_evaluate_check(
    conn: Connection, fitted: dict[str, Any], evaluated: dict[str, Any]
) -> None:
    spec = load_agent_spec("evaluate", _overrides(fitted, evaluated["out"]))
    checks = spec["quality_checks"]

    results = {r.name: r for r in QualityRunner(conn, spec["params"]).run_all(checks)}

    assert set(results) == {c["name"] for c in checks}
    for result in results.values():
        assert "not implemented" not in result.message, result.message
        assert "Cannot read" not in result.message, result.message
        assert "stale" not in result.message, result.message
        assert "absent from" not in result.message, result.message
        assert "Run the evaluate stage first" not in result.message, result.message
    report = _strict_load(evaluated["out"] / REPORT_NAME)
    for gate in report["gates"]:
        assert results[gate["name"]].passed is gate["passed"], gate["name"]


def test_a_newer_converged_model_run_makes_the_report_stale(
    conn: Connection, fitted: dict[str, Any], evaluated: dict[str, Any]
) -> None:
    conn.execute(
        text(
            "INSERT INTO model_runs (run_name, model_type, config, completed_at, diagnostics) "
            "VALUES ('newer pooled run', :t, '{}'::jsonb, now(), "
            '\'{"converged": "true", "estimates_written": true}\'::jsonb)'
        ),
        {"t": POOLED},
    )
    spec = load_agent_spec("evaluate", _overrides(fitted, evaluated["out"]))

    results = QualityRunner(conn, spec["params"]).run_all(spec["quality_checks"])

    assert len(results) == len(GATE_NAMES)
    for result in results:
        assert not result.passed, result.name
        assert "stale evaluation report" in result.message, result.message


def test_a_run_that_did_not_converge_is_not_a_target_and_nothing_is_written(
    conn: Connection, fitted: dict[str, Any], tmp_path: Path
) -> None:
    conn.execute(
        text(
            "UPDATE model_runs SET diagnostics = jsonb_set(diagnostics, '{converged}', "
            "'\"false\"'::jsonb) WHERE model_type = :t"
        ),
        {"t": POOLED},
    )

    with structlog.testing.capture_logs() as logs:
        evaluate_stage.run(
            load_agent_spec("evaluate", _overrides(fitted, tmp_path)),
            StageContext(db_conn=conn),
        )

    assert any(entry["event"] == "evaluate_no_target_run" for entry in logs), logs
    assert list(tmp_path.iterdir()) == []


def test_dry_run_prepares_inputs_and_writes_nothing(
    engine: Engine, conn: Connection, fitted: dict[str, Any], tmp_path: Path
) -> None:
    before = _counts(engine)

    with structlog.testing.capture_logs() as logs:
        evaluate_stage.run(
            load_agent_spec("evaluate", _overrides(fitted, tmp_path)),
            StageContext(db_conn=conn, dry_run=True),
        )

    events = [entry["event"] for entry in logs]
    assert "evaluate_inputs_ready" in events
    assert "evaluate_dry_run" in events
    assert list(tmp_path.iterdir()) == []
    assert _counts(engine) == before


def test_rounds_from_another_impute_run_are_refused_before_any_output(
    conn: Connection, fitted: dict[str, Any], tmp_path: Path
) -> None:
    conn.execute(
        text(
            "UPDATE model_runs SET config = jsonb_set(config, '{imputation_run_id}', "
            "to_jsonb(CAST(:other AS int))) WHERE model_type = :t"
        ),
        {"other": FAKE_IMPUTE_RUN_ID - 1, "t": POOLED},
    )

    with pytest.raises(ValueError, match="impute"):
        evaluate_stage.run(
            load_agent_spec("evaluate", _overrides(fitted, tmp_path)),
            StageContext(db_conn=conn),
        )

    assert list(tmp_path.iterdir()) == []
