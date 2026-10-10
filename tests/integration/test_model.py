"""
Integration tests for the model stage, against a real PostgreSQL.

Seeds a small simulated tournament (two eras that never meet, co-commanded and
unknown-commander sides) plus two imputed rounds written to ``tmp_path`` whose
manifests point at a fake impute run, then drives the real stage
(:mod:`pipeline.stages.model`): real (short) nutpie fits, real files, real
``model_runs`` / ``general_skill_estimates`` / ``battle_war_details`` writes.

Sampling is tiny (200 draws, 2 chains) and the write thresholds relaxed: the
point is plumbing, not convergence, which ``tests/model/test_model_recovery.py``
owns. The strict-threshold test restores a hard gate to check the write rule.

Seeded data is committed (module-scoped fixture on its own connection), with
``try/finally`` rollback around every helper that runs the stage, as in
``test_impute.py`` (handover.md 20.2 item 3).
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from pipeline.db import DatabaseConfigError, apply_schema, database_url, get_engine
from pipeline.quality import QualityRunner
from pipeline.stages import model as model_stage
from pipeline.stages.base import StageContext, check_conforms
from tests.model.model_sim import SimTournament, simulate_tournament

AGENTS = Path(__file__).resolve().parents[2] / "agents"
FAKE_IMPUTE_RUN_ID = 777
N_ROUNDS = 2
LOG_LEVEL = 9.5  # realistic absolute log-troop level (about 13,000 men)
POOLED = "bradley_terry_hierarchical"


def _database_available() -> tuple[bool, str]:
    """Check whether DATABASE_URL points at a reachable database."""
    try:
        url = database_url()
    except DatabaseConfigError as exc:
        return False, str(exc)
    try:
        engine = get_engine(url, pool_pre_ping=False)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
    except Exception as exc:
        return False, f"cannot connect: {type(exc).__name__}: {exc}"
    return True, ""


_AVAILABLE, _REASON = _database_available()

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _AVAILABLE,
        reason=f"No database available ({_REASON}). Run `make db-up` first.",
    ),
]


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


# ─── Seeding ─────────────────────────────────────────────────────────────────

_INSERT_BATTLE = text(
    "INSERT INTO battles (name, date_start, date_precision, battle_type) "
    "VALUES (:name, CAST(:date AS DATE), 'year', 'field') RETURNING battle_id"
)
_INSERT_SIDE = text(
    """
    INSERT INTO battle_sides (battle_id, side_label, outcome,
                              est_troops_neutral_log_mean, est_troops_neutral_log_sd)
    VALUES (:battle_id, :label, CAST(:outcome AS outcome_level), :log_mean, :log_sd)
    RETURNING side_id
    """
)
_INSERT_GENERAL = text("INSERT INTO generals (canonical_name) VALUES (:name) RETURNING general_id")
_INSERT_COMMANDER = text(
    """
    INSERT INTO battle_commanders (battle_id, side_id, general_id,
                                   attribution_weight, attribution_method)
    VALUES (:battle_id, :side_id, :general_id, :weight, :method)
    """
)
_INSERT_IMPUTE_RUN = text(
    """
    INSERT INTO model_runs (run_id, run_name, model_type, config, completed_at, diagnostics)
    VALUES (:run_id, 'fake impute', 'imputation', '{}'::jsonb, now(),
            '{"scale": "neutral", "written": true}'::jsonb)
    """
)
# Pushes the serial past the fake id so later inserts cannot collide with it.
_BUMP_SEQUENCE = text(
    "SELECT setval(pg_get_serial_sequence('model_runs', 'run_id'), "
    "GREATEST((SELECT MAX(run_id) FROM model_runs), 1))"
)


def _date_for_year(year: int) -> str:
    """Spell an astronomical year as a date string Postgres accepts (BC aware)."""
    return f"{1 - year:04d}-06-01 BC" if year <= 0 else f"{year:04d}-06-01"


def _seed(setup: Connection, tour: SimTournament) -> dict[int, int]:
    """Insert the tournament.

    Args:
        setup: A connection the caller commits.
        tour: The simulated tournament.

    Returns:
        ``simulated side id -> database side_id``.
    """
    db_general = {
        gid: int(setup.execute(_INSERT_GENERAL, {"name": f"General {gid}"}).scalar_one())
        for gid in sorted(tour.true_skill)
    }
    db_battle: dict[int, int] = {}
    db_side: dict[int, int] = {}
    for side in sorted(tour.sides, key=lambda s: s.side_id):
        if side.battle_id not in db_battle:
            assert side.year is not None
            db_battle[side.battle_id] = int(
                setup.execute(
                    _INSERT_BATTLE,
                    {"name": f"Model Fixture {side.battle_id}", "date": _date_for_year(side.year)},
                ).scalar_one()
            )
        has_measure = side.side_id % 3 != 0
        side_id = int(
            setup.execute(
                _INSERT_SIDE,
                {
                    "battle_id": db_battle[side.battle_id],
                    "label": f"Side {side.side_id}",
                    "outcome": side.outcome,
                    "log_mean": LOG_LEVEL + tour.log_troops[side.side_id] if has_measure else None,
                    "log_sd": 0.4 if has_measure else None,
                },
            ).scalar_one()
        )
        db_side[side.side_id] = side_id
        for cmd in side.commanders:
            setup.execute(
                _INSERT_COMMANDER,
                {
                    "battle_id": db_battle[side.battle_id],
                    "side_id": side_id,
                    "general_id": db_general[cmd.general_id],
                    "weight": cmd.attribution_weight,
                    "method": cmd.attribution_method,
                },
            )
    return db_side


def _write_rounds(root: Path, tour: SimTournament, db_side: dict[int, int], run_id: int) -> None:
    """Write ``N_ROUNDS`` complete imputed rounds under ``root/neutral``."""
    rng = np.random.default_rng(5)
    for n in range(N_ROUNDS):
        directory = root / "neutral" / f"round_{n}"
        directory.mkdir(parents=True)
        lines = ["side_id,battle_id,log_troops,is_imputed,scale"]
        for sim_id, db_id in db_side.items():
            value = LOG_LEVEL + tour.log_troops[sim_id] + 0.15 * rng.standard_normal()
            lines.append(f"{db_id},0,{value!r},False,neutral")
        (directory / "sides.csv").write_text("\n".join(lines), encoding="utf-8")
        (directory / "manifest.json").write_text(
            json.dumps({"run_id": run_id, "scale": "neutral", "round": n}), encoding="utf-8"
        )


@pytest.fixture(scope="module")
def tournament() -> SimTournament:
    """A small two-era tournament."""
    return simulate_tournament(3, generals_per_era=8, battles_per_era=45)


@pytest.fixture(scope="module")
def seeded(
    engine: Engine, tournament: SimTournament, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Any]:
    """Seed the tournament (committed), the fake impute run and the rounds."""
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
    return {"db_side": db_side, "root": root}


def _spec(seeded: dict[str, Any], diagnostics: Path, **overrides: Any) -> dict[str, Any]:
    """Load agents/model.yaml with fast-sampling overrides."""
    raw = yaml.safe_load((AGENTS / "model.yaml").read_text(encoding="utf-8"))
    params: dict[str, Any] = dict(raw.get("params") or {})
    params.update(
        {
            "sampler": {"draws": 200, "tune": 200, "chains": 2, "cores": 1},
            "write": {"max_rhat": 10.0, "min_ess_bulk": 0, "max_divergences": 1_000_000},
            "pooling": {"draws_kept_per_round": 100},
            "prior_predictive": {"draws": 200, "pairs": 100},
            "gates": {"ppc_sims": 100},
            "imputed_root": str(seeded["root"]),
            "diagnostics_root": str(diagnostics),
        }
    )
    params.update(overrides)
    raw["params"] = params
    return dict(raw)


def _run_committed(engine: Engine, spec: dict[str, Any]) -> None:
    """Run the stage on a connection this helper commits, rolling back on failure."""
    connection = engine.connect()
    try:
        model_stage.run(spec, context=StageContext(db_conn=connection))
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def _count_runs(engine: Engine) -> int:
    with engine.connect() as probe:
        return int(probe.execute(text("SELECT COUNT(*) FROM model_runs")).scalar_one())


@pytest.fixture(scope="module")
def fitted(
    engine: Engine, seeded: dict[str, Any], tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Any]:
    """Run the stage once, committed, with relaxed write thresholds."""
    diagnostics = tmp_path_factory.mktemp("diagnostics")
    _run_committed(engine, _spec(seeded, diagnostics))
    with engine.connect() as probe:
        pooled_id = int(
            probe.execute(
                text("SELECT MAX(run_id) FROM model_runs WHERE model_type = :t"), {"t": POOLED}
            ).scalar_one()
        )
    return {**seeded, "diagnostics": diagnostics, "pooled_id": pooled_id}


# ─── Tests ───────────────────────────────────────────────────────────────────


def test_stage_satisfies_the_contract() -> None:
    check_conforms(model_stage)


def test_run_rows_are_two_rounds_one_joint_and_the_pooled_row_last(
    conn: Connection, fitted: dict[str, Any]
) -> None:
    rows = conn.execute(
        text(
            "SELECT run_id, model_type, imputation_set, completed_at IS NOT NULL "
            "FROM model_runs WHERE run_id > :fake ORDER BY run_id"
        ),
        {"fake": FAKE_IMPUTE_RUN_ID},
    ).fetchall()

    assert [r[1] for r in rows] == [
        "bradley_terry_round",
        "bradley_terry_round",
        "bradley_terry_joint_latent",
        POOLED,
    ]
    assert [r[2] for r in rows[:2]] == [0, 1]
    assert all(r[3] for r in rows)
    assert rows[-1][0] == fitted["pooled_id"] == max(r[0] for r in rows)


def test_pooled_row_records_rounds_counts_and_worst_case(
    conn: Connection, fitted: dict[str, Any]
) -> None:
    config, diagnostics = conn.execute(
        text("SELECT config, diagnostics FROM model_runs WHERE run_id = :r"),
        {"r": fitted["pooled_id"]},
    ).one()
    round_ids = [
        int(r[0])
        for r in conn.execute(
            text(
                "SELECT run_id FROM model_runs WHERE model_type = 'bradley_terry_round' "
                "ORDER BY imputation_set"
            )
        ).fetchall()
    ]

    assert config["pooled_round_run_ids"] == round_ids
    assert diagnostics["converged"] == "true"
    assert {"max_rhat", "min_ess_bulk", "divergences"} <= set(diagnostics["worst"])
    assert diagnostics["imputation_run_id"] == FAKE_IMPUTE_RUN_ID
    assert diagnostics["defender_term"] == "omitted_no_data"
    assert diagnostics["counts"]["battles_included"] > 0
    assert diagnostics["components"]["n_components"] >= 2  # two eras never meet
    assert diagnostics["prior_predictive"]["category_freq"]
    assert diagnostics["estimates_written"] is True


def test_estimates_have_intervals_and_unique_ranks(
    conn: Connection, fitted: dict[str, Any]
) -> None:
    rows = conn.execute(
        text(
            "SELECT skill_mean, skill_sd, skill_ci_lo, skill_ci_hi, skill_ci90_lo, "
            "skill_ci90_hi, total_war, war_sd, n_battles, rank "
            "FROM general_skill_estimates WHERE run_id = :r ORDER BY rank"
        ),
        {"r": fitted["pooled_id"]},
    ).fetchall()

    assert len(rows) >= 10
    assert [r[9] for r in rows] == list(range(1, len(rows) + 1))
    for mean, sd, lo, hi, lo90, hi90, war, war_sd, n_battles, _ in rows:
        assert None not in (mean, sd, lo, hi, lo90, hi90, war, war_sd)
        assert lo < mean < hi
        assert lo <= lo90 < hi90 <= hi
        assert sd > 0 and n_battles >= 1
    totals = [r[6] for r in rows]
    assert totals == sorted(totals, reverse=True)  # rank_by total_war


def test_joint_run_has_estimates_and_only_the_pooled_run_has_war_details(
    conn: Connection, fitted: dict[str, Any]
) -> None:
    joint_id = int(
        conn.execute(
            text("SELECT run_id FROM model_runs WHERE model_type = 'bradley_terry_joint_latent'")
        ).scalar_one()
    )

    joint_rows = conn.execute(
        text("SELECT COUNT(*) FROM general_skill_estimates WHERE run_id = :r"), {"r": joint_id}
    ).scalar_one()
    detail_runs = {
        int(r[0]) for r in conn.execute(text("SELECT DISTINCT run_id FROM battle_war_details"))
    }

    assert joint_rows > 0
    assert detail_runs == {fitted["pooled_id"]}


def test_battle_war_details_hold_expected_score_and_reconcile_with_total_war(
    conn: Connection, fitted: dict[str, Any]
) -> None:
    bad = conn.execute(
        text(
            """
            SELECT COUNT(*) FROM (
              SELECT gse.general_id
              FROM general_skill_estimates gse
              LEFT JOIN battle_war_details bwd
                ON bwd.run_id = gse.run_id AND bwd.general_id = gse.general_id
              WHERE gse.run_id = :r
              GROUP BY gse.general_id, gse.total_war
              HAVING ABS(gse.total_war - COALESCE(SUM(bwd.war_contribution), 0)) > 1e-6
            ) t
            """
        ),
        {"r": fitted["pooled_id"]},
    ).scalar_one()
    sample = conn.execute(
        text(
            "SELECT win_prob_model, win_prob_baseline, actual_outcome, attribution_weight "
            "FROM battle_war_details WHERE run_id = :r"
        ),
        {"r": fitted["pooled_id"]},
    ).fetchall()

    assert bad == 0
    assert len(sample) > 20
    for model_p, baseline_p, outcome, weight in sample:
        assert 0.0 <= model_p <= 1.0 and 0.0 <= baseline_p <= 1.0
        assert outcome in (0.0, 0.5, 1.0)
        assert 0.0 < weight <= 1.0


def test_run_directory_has_every_artefact(fitted: dict[str, Any]) -> None:
    directory = fitted["diagnostics"] / f"run{fitted['pooled_id']}"

    for name in (
        "rounds_summary.json",
        "rubin_per_general.csv",
        "components.json",
        "gate_report.json",
        "pooled_draws.npz",
    ):
        assert (directory / name).is_file(), name
    with np.load(directory / "pooled_draws.npz") as data:
        assert data["skill_draws"].shape[0] == N_ROUNDS * 100
        assert {"general_ids", "entry_xwar_mean", "entry_realised_war_mean"} <= set(data.files)
    report = json.loads((directory / "gate_report.json").read_text(encoding="utf-8"))
    assert report["estimates_written"] is True
    assert {g["name"] for g in report["gates"]} == {
        "fmi_top_generals",
        "circularity_check",
        "exclusion_concentration",
        "extreme_ratio_dependence",
        "implausible_career_span",
        "prior_predictive_sensible",
        "top_generals_cross_component",
        "exclusions_reported",
        "posterior_predictive_category_check",
    }


def test_the_quality_runner_executes_every_model_gate(
    conn: Connection, fitted: dict[str, Any]
) -> None:
    spec = yaml.safe_load((AGENTS / "model.yaml").read_text(encoding="utf-8"))
    checks = [
        {**c, "params": {**c["params"], "root": str(fitted["diagnostics"])}}
        if c.get("method") == "model_gate"
        else c
        for c in spec["quality_checks"]
    ]

    results = {r.name: r for r in QualityRunner(conn).run_all(checks)}

    assert set(results) == {c["name"] for c in checks}
    for check in checks:
        result = results[check["name"]]
        assert "Cannot read" not in result.message, result.message
        assert "absent from" not in result.message, result.message
        assert "not implemented" not in result.message, result.message
    assert results["top_generals_have_intervals"].passed, results[
        "top_generals_have_intervals"
    ].message
    assert results["top_generals_cross_component"].actual_value


def test_a_failing_round_writes_no_estimates_and_the_pooled_row_says_so(
    engine: Engine, seeded: dict[str, Any], fitted: dict[str, Any], tmp_path: Path
) -> None:
    del fitted
    before = _count_runs(engine)

    _run_committed(
        engine,
        _spec(
            seeded,
            tmp_path,
            write={"max_rhat": 0.5, "min_ess_bulk": 10**9, "max_divergences": 0},
            pooling={"draws_kept_per_round": 100, "max_rounds": 1},
        ),
    )

    with engine.connect() as probe:
        row = probe.execute(
            text(
                "SELECT run_id, completed_at, diagnostics FROM model_runs "
                "WHERE model_type = :t ORDER BY run_id DESC LIMIT 1"
            ),
            {"t": POOLED},
        ).one()
        estimates = probe.execute(
            text("SELECT COUNT(*) FROM general_skill_estimates WHERE run_id = :r"), {"r": row[0]}
        ).scalar_one()
        details = probe.execute(
            text("SELECT COUNT(*) FROM battle_war_details WHERE run_id = :r"), {"r": row[0]}
        ).scalar_one()
        joint = probe.execute(
            text("SELECT COUNT(*) FROM model_runs WHERE model_type = 'bradley_terry_joint_latent'")
        ).scalar_one()
        round_attempts = probe.execute(
            text(
                "SELECT diagnostics -> 'attempts' FROM model_runs "
                "WHERE model_type = 'bradley_terry_round' ORDER BY run_id DESC LIMIT 1"
            )
        ).scalar_one()
    assert row[1] is not None
    assert row[2]["converged"] == "false"
    assert row[2]["estimates_written"] is False
    assert (estimates, details) == (0, 0)
    assert joint == 1  # the successful run's row only: a failed run skips the joint fit
    assert len(round_attempts) == 2  # retried once, never dropped
    assert round_attempts[1]["target_accept"] == pytest.approx(0.95)
    assert _count_runs(engine) == before + 2  # one round row and the pooled row
    report = json.loads(
        (tmp_path / f"run{row[0]}" / "gate_report.json").read_text(encoding="utf-8")
    )
    assert report["estimates_written"] is False and report["gates"] == []
    gate = QualityRunner(None).run_one(
        {
            "name": "fmi_top_generals",
            "method": "model_gate",
            "params": {"gate": "fmi_top_generals", "root": str(tmp_path)},
            "severity": "warning",
        }
    )
    assert not gate.passed


def test_a_stale_manifest_is_refused_before_any_run_row_is_written(
    engine: Engine, conn: Connection, seeded: dict[str, Any], tmp_path: Path
) -> None:
    stale_root = tmp_path / "imputed"
    shutil.copytree(seeded["root"], stale_root)
    manifest = stale_root / "neutral" / "round_0" / "manifest.json"
    manifest.write_text(
        json.dumps({"run_id": FAKE_IMPUTE_RUN_ID - 1, "scale": "neutral"}), encoding="utf-8"
    )
    before = _count_runs(engine)

    with pytest.raises(ValueError, match="several impute runs"):
        model_stage.run(
            _spec(seeded, tmp_path / "out", imputed_root=str(stale_root)),
            context=StageContext(db_conn=conn),
        )
    for n in range(N_ROUNDS):
        path = stale_root / "neutral" / f"round_{n}" / "manifest.json"
        path.write_text(
            json.dumps({"run_id": FAKE_IMPUTE_RUN_ID - 1, "scale": "neutral"}), encoding="utf-8"
        )
    with pytest.raises(ValueError, match="latest neutral impute run"):
        model_stage.run(
            _spec(seeded, tmp_path / "out", imputed_root=str(stale_root)),
            context=StageContext(db_conn=conn),
        )

    assert _count_runs(engine) == before
    assert not (tmp_path / "out").exists()


def test_a_round_missing_a_side_is_refused(
    conn: Connection, seeded: dict[str, Any], tmp_path: Path
) -> None:
    root = tmp_path / "imputed"
    shutil.copytree(seeded["root"], root)
    sides = root / "neutral" / "round_1" / "sides.csv"
    lines = sides.read_text(encoding="utf-8").splitlines()
    sides.write_text("\n".join(lines[:-3]), encoding="utf-8")

    with pytest.raises(ValueError, match="round 1 is incomplete"):
        model_stage.run(
            _spec(seeded, tmp_path / "out", imputed_root=str(root)),
            context=StageContext(db_conn=conn),
        )


def test_dry_run_writes_nothing(
    engine: Engine, conn: Connection, seeded: dict[str, Any], tmp_path: Path
) -> None:
    before = _count_runs(engine)

    model_stage.run(
        _spec(seeded, tmp_path / "out"), context=StageContext(db_conn=conn, dry_run=True)
    )

    assert _count_runs(engine) == before
    assert not (tmp_path / "out").exists()
