"""
Integration tests for the impute stage, against a real PostgreSQL.

Seeds battles, sides, commanders and ``missing_data_log`` rows from the
synthetic recovery corpus (``tests/model/impute_sim.py``), writes a synthetic
reconcile ``neutral_draws_run*.npz``, and drives the real stage runner
(:mod:`pipeline.stages.impute`): a real (small, fast-sampled) PyMC fit through
nutpie, real files, real ``model_runs`` and ``missing_data_log`` writes.

These need the columns reconcile's neutral-fit migration adds
(``battle_sides.is_defender`` and the ``est_troops_neutral*`` family), so they
cannot run until that migration has landed in ``config/schema.sql``.

Sampling is small everywhere (300 draws, 2 chains) and the write thresholds
are relaxed: the point is plumbing, not convergence, which
``tests/model/test_impute_recovery.py`` owns. The one test that restores a
strict gate does so to check the gate itself.

Seeded data is committed (the module-scoped fixture opens its own connection),
with ``try/finally`` rollback around every fixture that runs the stage: a
stage that raises part-way otherwise leaves a transaction holding row locks,
and every later test waits on them forever (handover.md 20.2 item 3).
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from pipeline.db import DatabaseConfigError, apply_schema, database_url, get_engine
from pipeline.imputers.load import load_side_rows
from pipeline.quality import QualityRunner
from pipeline.stages import impute as impute_stage
from pipeline.stages.base import StageContext, check_conforms
from tests.model.impute_sim import SimCorpus, simulate_corpus

AGENTS = Path(__file__).resolve().parents[2] / "agents"
N_BATTLES = 40
N_ROUNDS = 6
FAKE_RECONCILE_RUN_ID = 999
MNAR_EVERY = 4


def _database_available() -> tuple[bool, str]:
    """Check whether DATABASE_URL points at a reachable database.

    Returns:
        (available, reason). Reason is empty when available.
    """
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
        pytest.skip(
            f"Refusing to reset database {db_name!r}: the name suggests it is not a "
            "test database. Point DATABASE_URL at general_war or a *test* database."
        )

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
    "VALUES (:name, CAST(:date AS DATE), 'year', CAST(:battle_type AS battle_type)) "
    "RETURNING battle_id"
)
_INSERT_SIDE = text(
    """
    INSERT INTO battle_sides (battle_id, side_label, polity, outcome)
    VALUES (:battle_id, :label, :polity, CAST(:outcome AS outcome_level))
    RETURNING side_id
    """
)
_INSERT_GENERAL = text("INSERT INTO generals (canonical_name) VALUES (:name) RETURNING general_id")
_INSERT_COMMANDER = text(
    """
    INSERT INTO battle_commanders (battle_id, side_id, general_id, attribution_weight)
    VALUES (:battle_id, :side_id, :general_id, 1.0)
    """
)
_INSERT_MISSING = text(
    """
    INSERT INTO missing_data_log (battle_id, side_id, field_name, missingness_class, notes)
    VALUES (:battle_id, :side_id, 'troop_total', CAST(:cls AS missingness_class), :notes)
    """
)


# The stage refuses a draws file that is not from the latest reconcile run that
# wrote a neutral fit, so the fixture needs such a run to exist.
_INSERT_RECONCILE_RUN = text(
    """
    INSERT INTO model_runs (run_id, run_name, model_type, config, completed_at, diagnostics)
    VALUES (:run_id, 'fake reconcile', 'source_disagreement', '{}'::jsonb, now(),
            '{"troops_neutral": {"written": true}}'::jsonb)
    """
)


def _date_for_year(year: int | None) -> str | None:
    """Spell an astronomical year as a date string Postgres accepts (BC aware)."""
    if year is None:
        return None
    if year <= 0:
        return f"{1 - year:04d}-06-01 BC"
    return f"{year:04d}-06-01"


def _seed(setup: Connection, corpus: SimCorpus) -> dict[int, int]:
    """Insert the simulated corpus.

    Args:
        setup: A connection the caller commits.
        corpus: The simulated corpus.

    Returns:
        ``simulated side id -> database side_id``.
    """
    db_battle: dict[int, int] = {}
    db_side: dict[int, int] = {}
    db_general: dict[str, int] = {}
    missing_seen = 0
    for side in corpus.sides:
        if side.battle_id not in db_battle:
            db_battle[side.battle_id] = int(
                setup.execute(
                    _INSERT_BATTLE,
                    {
                        "name": f"Impute Fixture {side.battle_id}",
                        "date": _date_for_year(side.year),
                        "battle_type": side.battle_type,
                    },
                ).scalar_one()
            )
        side_id = int(
            setup.execute(
                _INSERT_SIDE,
                {
                    "battle_id": db_battle[side.battle_id],
                    "label": f"Side {side.side_id}",
                    "polity": side.polity_key,
                    "outcome": side.outcome,
                },
            ).scalar_one()
        )
        db_side[side.side_id] = side_id
        if side.commander_key not in db_general:
            db_general[side.commander_key] = int(
                setup.execute(_INSERT_GENERAL, {"name": side.commander_key}).scalar_one()
            )
        setup.execute(
            _INSERT_COMMANDER,
            {
                "battle_id": db_battle[side.battle_id],
                "side_id": side_id,
                "general_id": db_general[side.commander_key],
            },
        )
        if side.side_id in corpus.is_missing:
            missing_seen += 1
            setup.execute(
                _INSERT_MISSING,
                {
                    "battle_id": db_battle[side.battle_id],
                    "side_id": side_id,
                    "cls": "mnar" if missing_seen % MNAR_EVERY == 0 else "mar",
                    "notes": "no troop reports extracted",
                },
            )
    return db_side


def _write_draws(path: Path, corpus: SimCorpus, db_side: dict[int, int], conn: Any) -> None:
    """Write a reconcile-shaped draws file for the observed sides.

    Each side's posterior is its likelihood mean with sd 0.5, independent, so
    dividing out s0 = 2 leaves a precision of 3.75 per side, comfortably above
    the weak-side threshold.
    """
    rng = np.random.default_rng(11)
    observed = [s for s in corpus.sides if s.side_id not in corpus.is_missing]
    side_ids = np.array([db_side[s.side_id] for s in observed], dtype=np.int64)
    battle_ids = np.array(
        [
            int(
                conn.execute(
                    text("SELECT battle_id FROM battle_sides WHERE side_id = :s"),
                    {"s": db_side[s.side_id]},
                ).scalar_one()
            )
            for s in observed
        ],
        dtype=np.int64,
    )
    centres = np.array([corpus.measured_y[s.side_id] for s in observed])
    np.savez(
        path,
        run_id=FAKE_RECONCILE_RUN_ID,
        side_id=side_ids,
        battle_id=battle_ids,
        draws=centres + 0.5 * rng.standard_normal((400, len(observed))),
        m0_mean=9.0,
        s0_mean=2.0,
    )


@pytest.fixture(scope="module")
def corpus() -> SimCorpus:
    """The simulated corpus: a few dozen battles, about 30% of sides missing."""
    return simulate_corpus(7, n_battles=N_BATTLES, p_missing=0.3)


@pytest.fixture(scope="module")
def seeded(
    engine: Engine, corpus: SimCorpus, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Any]:
    """Seed the corpus (committed) and write the draws file.

    Returns:
        The side-id map, the draws path and the imputed-data root.
    """
    setup = engine.connect()
    try:
        db_side = _seed(setup, corpus)
        setup.execute(_INSERT_RECONCILE_RUN, {"run_id": FAKE_RECONCILE_RUN_ID})
        setup.commit()
        draws_dir = tmp_path_factory.mktemp("reconcile")
        draws_path = draws_dir / f"neutral_draws_run{FAKE_RECONCILE_RUN_ID}.npz"
        _write_draws(draws_path, corpus, db_side, setup)
    except BaseException:
        setup.rollback()
        raise
    finally:
        setup.close()
    return {
        "db_side": db_side,
        "draws_path": draws_path,
        "root": tmp_path_factory.mktemp("imputed"),
    }


def _spec(seeded: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """Load agents/impute.yaml with fast-sampling overrides for a test."""
    raw = yaml.safe_load((AGENTS / "impute.yaml").read_text(encoding="utf-8"))
    params = dict(raw.get("params") or {})
    params.update(
        {
            "scale": "neutral",
            "n_imputations": N_ROUNDS,
            "mcmc_samples": 300,
            "mcmc_tune": 300,
            "mcmc_chains": 2,
            "mcmc_cores": 1,
            "write_max_rhat": 10.0,
            "write_min_ess_bulk": 0,
            "write_max_divergences": 1_000_000,
            "draws_path": str(seeded["draws_path"]),
            "imputed_root": str(seeded["root"]),
        }
    )
    params.update(overrides)
    raw["params"] = params
    return dict(raw)


def _run_committed(engine: Engine, spec: dict[str, Any]) -> None:
    """Run the stage on a connection this helper commits, rolling back on failure."""
    conn = engine.connect()
    try:
        impute_stage.run(spec, context=StageContext(db_conn=conn))
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


@pytest.fixture(scope="module")
def imputed(engine: Engine, seeded: dict[str, Any]) -> dict[str, Any]:
    """Run the neutral-scale stage once, committed, over the seeded corpus."""
    _run_committed(engine, _spec(seeded))
    return seeded


def _read_round(root: Path, scale: str, n: int) -> list[dict[str, str]]:
    with (root / scale / f"round_{n}" / "sides.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


# ─── Tests ───────────────────────────────────────────────────────────────────


def test_stage_satisfies_the_contract() -> None:
    from pipeline.stages import impute

    check_conforms(impute)


def test_loader_reads_year_polity_commander_and_mnar(
    conn: Connection, seeded: dict[str, Any], corpus: SimCorpus
) -> None:
    rows = {r.side_id: r for r in load_side_rows(conn)}

    sim_ancient = next(s for s in corpus.sides if s.year is not None and s.year < 0)
    row = rows[seeded["db_side"][sim_ancient.side_id]]
    assert row.year is not None and row.year < 0  # BC survives the round trip via year_astronomical
    assert row.polity_key == sim_ancient.polity_key
    assert row.commander_key.startswith("general_")
    assert any(r.mnar for r in rows.values())
    assert any(r.year is None for r in rows.values())


def test_every_round_has_every_side_and_marks_the_missing_ones(
    conn: Connection, imputed: dict[str, Any], corpus: SimCorpus
) -> None:
    n_sides = int(conn.execute(text("SELECT COUNT(*) FROM battle_sides")).scalar_one())
    missing_db = {imputed["db_side"][s] for s in corpus.is_missing}

    for n in range(N_ROUNDS):
        rows = _read_round(imputed["root"], "neutral", n)
        assert len(rows) == n_sides
        assert {int(r["side_id"]) for r in rows if r["is_imputed"] == "True"} == missing_db
        assert all(np.isfinite(float(r["log_troops"])) for r in rows)
        assert {r["scale"] for r in rows} == {"neutral"}

    manifest = json.loads(
        (imputed["root"] / "neutral" / "round_0" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["scale"] == "neutral"
    assert manifest["reconcile_run_id"] == FAKE_RECONCILE_RUN_ID
    assert manifest["n_imputed"] == len(missing_db)


def test_rounds_differ_because_each_uses_its_own_posterior_draw(imputed: dict[str, Any]) -> None:
    first = [float(r["log_troops"]) for r in _read_round(imputed["root"], "neutral", 0)]
    second = [float(r["log_troops"]) for r in _read_round(imputed["root"], "neutral", 1)]

    assert first != second


def test_one_model_run_row_records_diagnostics_and_scale(
    conn: Connection, imputed: dict[str, Any]
) -> None:
    del imputed
    row = conn.execute(
        text(
            "SELECT completed_at, diagnostics FROM model_runs "
            "WHERE model_type = 'imputation' ORDER BY run_id DESC LIMIT 1"
        )
    ).one()
    assert row[0] is not None
    diagnostics = row[1]
    assert diagnostics["scale"] == "neutral"
    assert diagnostics["written"] is True
    assert {"max_rhat", "min_ess_bulk", "divergences"} <= set(diagnostics["worst"])
    assert diagnostics["reconcile_run_id"] == FAKE_RECONCILE_RUN_ID


def test_missing_data_log_rows_for_imputed_sides_carry_the_summary(
    conn: Connection, imputed: dict[str, Any], corpus: SimCorpus
) -> None:
    ids = [imputed["db_side"][s] for s in corpus.is_missing]
    rows = conn.execute(
        text(
            "SELECT side_id, was_imputed, imputation_method, imputed_value, "
            "imputed_lo, imputed_hi, notes FROM missing_data_log "
            "WHERE field_name = 'troop_total' AND side_id = ANY(:ids)"
        ),
        {"ids": ids},
    ).fetchall()

    assert {r[0] for r in rows} >= set(ids)
    for _, was_imputed, method, value, lo, hi, notes in rows:
        assert was_imputed is True
        assert method == "bayes_joint_v1_neutral_log"
        assert lo < value < hi
        assert "imputed_scale=neutral" in notes
        assert f"percentiles of {N_ROUNDS} rounds" in notes
        assert "no troop reports extracted" in notes  # extract's note survives
    observed = conn.execute(
        text("SELECT COUNT(*) FROM missing_data_log WHERE was_imputed AND NOT side_id = ANY(:ids)"),
        {"ids": ids},
    ).scalar_one()
    assert observed == 0


def test_the_stage_gates_pass_on_the_written_rounds(
    conn: Connection, imputed: dict[str, Any]
) -> None:
    spec = yaml.safe_load((AGENTS / "impute.yaml").read_text(encoding="utf-8"))
    checks = [
        {**c, "params": {**c["params"], "root": str(imputed["root"])}}
        for c in spec["quality_checks"]
        if c.get("method") == "imputation_gate"
    ]

    results = {r.name: r for r in QualityRunner(conn).run_all(checks)}

    for name in ("imputation_completeness", "imputed_ratio_plausible", "imputed_values_plausible"):
        assert results[name].passed, results[name].message
    assert results["distributional_check"].severity.value == "warning"


def test_rerunning_with_the_same_seed_reproduces_rounds_and_does_not_duplicate_log_rows(
    engine: Engine, imputed: dict[str, Any], tmp_path: Path
) -> None:
    before = _read_round(imputed["root"], "neutral", 2)
    with engine.connect() as probe:
        count = int(probe.execute(text("SELECT COUNT(*) FROM missing_data_log")).scalar_one())

    _run_committed(engine, _spec(imputed, imputed_root=str(tmp_path)))

    assert _read_round(tmp_path, "neutral", 2) == before
    with engine.connect() as probe:
        after = int(probe.execute(text("SELECT COUNT(*) FROM missing_data_log")).scalar_one())
        assert after == count
        notes = probe.execute(
            text("SELECT notes FROM missing_data_log WHERE imputation_method IS NOT NULL LIMIT 1")
        ).scalar_one()
    assert notes.count("imputed_scale=") == 1


def test_an_unconverged_fit_writes_no_rounds_and_no_log_rows(
    engine: Engine, imputed: dict[str, Any], tmp_path: Path
) -> None:
    with engine.connect() as probe:
        before = probe.execute(
            text("SELECT imputed_value FROM missing_data_log WHERE was_imputed ORDER BY log_id")
        ).fetchall()

    _run_committed(
        engine,
        _spec(imputed, imputed_root=str(tmp_path), write_max_rhat=0.5, write_min_ess_bulk=10**9),
    )

    assert not (tmp_path / "neutral" / "round_0").exists()
    report = json.loads((tmp_path / "neutral" / "gate_report.json").read_text(encoding="utf-8"))
    assert report["rounds_written"] is False
    with engine.connect() as probe:
        after = probe.execute(
            text("SELECT imputed_value FROM missing_data_log WHERE was_imputed ORDER BY log_id")
        ).fetchall()
        row = probe.execute(
            text(
                "SELECT completed_at, diagnostics FROM model_runs "
                "WHERE model_type = 'imputation' ORDER BY run_id DESC LIMIT 1"
            )
        ).one()
    assert after == before
    assert row[0] is not None and row[1]["written"] is False
    gate = QualityRunner(None).run_one(
        {
            "name": "imputation_completeness",
            "method": "imputation_gate",
            "params": {"gate": "imputation_completeness", "root": str(tmp_path)},
            "severity": "error",
        }
    )
    assert not gate.passed


def test_neutral_scale_without_a_draws_file_fails_loudly(
    conn: Connection, seeded: dict[str, Any], tmp_path: Path
) -> None:
    spec = _spec(seeded, draws_path=None, reconcile_root=str(tmp_path))

    with pytest.raises(FileNotFoundError, match="run reconcile first"):
        impute_stage.run(spec, context=StageContext(db_conn=conn))


def test_a_stale_draws_file_is_refused(
    conn: Connection, seeded: dict[str, Any], tmp_path: Path
) -> None:
    with np.load(seeded["draws_path"]) as data:
        arrays = {name: data[name] for name in data.files}
    arrays["run_id"] = np.array(FAKE_RECONCILE_RUN_ID - 1)
    stale = tmp_path / f"neutral_draws_run{FAKE_RECONCILE_RUN_ID - 1}.npz"
    np.savez(stale, **arrays)

    with pytest.raises(ValueError, match="stale or from a different database"):
        impute_stage.run(
            _spec(seeded, draws_path=str(stale), imputed_root=str(tmp_path)),
            context=StageContext(db_conn=conn),
        )
    assert not (tmp_path / "neutral").exists()


def test_published_scale_writes_files_but_never_missing_data_log(
    conn: Connection, seeded: dict[str, Any], corpus: SimCorpus, tmp_path: Path
) -> None:
    for sim in corpus.sides:
        if sim.side_id in corpus.is_missing:
            continue
        value = float(np.exp(corpus.measured_y[sim.side_id]))
        conn.execute(
            text(
                "UPDATE battle_sides SET est_troops_total = :v, est_troops_total_lo = :lo, "
                "est_troops_total_hi = :hi WHERE side_id = :s"
            ),
            {"v": value, "lo": value / 3.0, "hi": value * 3.0, "s": seeded["db_side"][sim.side_id]},
        )

    log_before = conn.execute(
        text(
            "SELECT log_id, was_imputed, imputation_method, imputed_value, notes "
            "FROM missing_data_log ORDER BY log_id"
        )
    ).fetchall()
    runs_before = int(conn.execute(text("SELECT COUNT(*) FROM model_runs")).scalar_one())

    impute_stage.run(
        _spec(seeded, scale="published", imputed_root=str(tmp_path)),
        context=StageContext(db_conn=conn),
    )

    rows = _read_round(tmp_path, "published", 0)
    assert {r["scale"] for r in rows} == {"published"}
    # A sensitivity run never touches missing_data_log, but records its model run.
    log_after = conn.execute(
        text(
            "SELECT log_id, was_imputed, imputation_method, imputed_value, notes "
            "FROM missing_data_log ORDER BY log_id"
        )
    ).fetchall()
    assert log_after == log_before
    runs_after = int(conn.execute(text("SELECT COUNT(*) FROM model_runs")).scalar_one())
    assert runs_after == runs_before + 1
    assert not (tmp_path / "gate_report.json").exists()
    report = json.loads((tmp_path / "published" / "gate_report.json").read_text(encoding="utf-8"))
    gate = next(g for g in report["gates"] if g["name"] == "imputed_values_plausible")
    assert "not applicable" not in gate["actual"]
