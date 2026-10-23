"""
Integration tests for the reconcile stage's write path, against a real PostgreSQL.

No reconcile estimate has ever been written to a database before this file
(docs/handover-log.md §19.6). Everything upstream -- the loader, the design matrix,
the model, the posterior summary -- has only ever been exercised on synthetic
data or a fixture database with the model itself stubbed out. These tests
drive the real stage runner (:mod:`pipeline.stages.reconcile`) end to end: a
real (small, fast-sampled) PyMC fit through nutpie, real writes to
``battle_sides`` and ``sources``, a real ``model_runs`` row.

Marked ``integration``, not ``model``: this needs pymc, but pymc is a plain
dependency (``pyproject.toml``), not a ``model``-marker-only extra, so it is
already installed in CI's database-backed "test" job -- the same job every
other ``tests/integration/`` file runs in. The ``model`` marker is reserved
for ``tests/model/test_reconcile_recovery.py``'s parameter-recovery fits,
which need no database and run in their own (undated) CI job. Marking this
file ``model`` as well would deselect it under ``-m "not live and not
model"`` and it would never run in the job that actually has a database.

Sampling is small everywhere in this file (200 tune, 200 draws, 2 chains) --
enough to exercise the write path and produce a real (if barely converged)
posterior, not enough to be a convergence test. Nothing here asserts on
rhat or ESS; ``tests/model/test_reconcile_recovery.py`` is where convergence
and parameter recovery are someone else's job to pin down.

Most tests share one seeded corpus and one stage run (the ``reconciled``
fixture, module-scoped) because a real fit is the expensive part of this
file and re-seeding a fresh corpus per assertion would multiply it for no
gain. Three tests cannot share it and seed their own data: the idempotency
test (which must run the stage twice against identical data), the
self-connecting test (which must exercise ``context=None`` and therefore
commits to the real database for real, so it cleans up after itself), and
the singleton-gate test (which fabricates a deliberately-wrong interval by
hand and must not taint the honestly-fitted corpus everything else reads).
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml
from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from pipeline.db import DatabaseConfigError, apply_schema, database_url, get_engine
from pipeline.quality import QualityRunner
from pipeline.stages import reconcile as reconcile_stage
from pipeline.stages.base import StageContext, check_conforms

AGENTS = Path(__file__).resolve().parents[2] / "agents"


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
    """Give each test an empty corpus, rolled back afterwards.

    Note that "empty" only means "nothing this fixture adds": rows the
    module-scoped ``reconciled`` fixture already *committed* are visible
    here too, because they are the same physical database. Tests that must
    not see that shared corpus (the singleton-gate test) simply do not
    depend on ``reconciled``.
    """
    connection = engine.connect()
    transaction = connection.begin()
    try:
        yield connection
    finally:
        transaction.rollback()
        connection.close()


# ─── Spec loading ────────────────────────────────────────────────────────────


def _load_spec(processed_root: Path, **param_overrides: Any) -> dict[str, Any]:
    """Load agents/reconcile.yaml with fast-sampling overrides for a test.

    Args:
        processed_root: Where the stage should write
            ``reconciliation_report.json``. Always a pytest temp directory in
            this file, never the project's real ``data/processed``.
        **param_overrides: Further params overrides, applied last.

    Returns:
        The spec dict, ready to pass to ``reconcile_stage.run``.
    """
    raw = yaml.safe_load((AGENTS / "reconcile.yaml").read_text(encoding="utf-8"))
    params = dict(raw.get("params") or {})
    params.update(
        {
            "processed_root": str(processed_root),
            "mcmc_samples": 200,
            "mcmc_tune": 200,
            "mcmc_chains": 2,
            "mcmc_cores": 1,
            # These tests check plumbing at 200 draws, far below what the
            # write gate demands; test_an_unconverged_fit_writes_no_estimates
            # restores a strict gate to check the gate itself.
            "write_max_rhat": 10.0,
            "write_min_ess_bulk": 0,
            "write_max_divergences": 1_000_000,
            "write_max_quantile_mcse_frac": 1.0e6,
        }
    )
    params.update(param_overrides)
    raw["params"] = params
    return dict(raw)


# ─── Seeding helpers, matching tests/integration/test_reconcile_load.py ─────

_INSERT_BATTLE = text(
    "INSERT INTO battles (name, date_start, date_precision) "
    "VALUES (:name, CAST(:date AS DATE), 'day') RETURNING battle_id"
)
_INSERT_SIDE = text(
    "INSERT INTO battle_sides (battle_id, side_label) VALUES (:battle_id, :label) RETURNING side_id"
)
_INSERT_SOURCE = text(
    "INSERT INTO sources (source_type, url) "
    "VALUES (CAST(:source_type AS source_type), :url) RETURNING source_id"
)
_INSERT_TROOP = text(
    """
    INSERT INTO troop_reports (
        side_id, source_id, branch, reported_value, scope,
        is_estimate, is_upper_bound, is_lower_bound,
        extraction_method, extracted_context
    ) VALUES (
        :side_id, :source_id, CAST(:branch AS troop_branch), :value, :scope,
        :is_estimate, :is_upper_bound, :is_lower_bound,
        CAST('infobox_parser' AS extraction_method), :context
    )
    """
)
_INSERT_CASUALTY = text(
    """
    INSERT INTO casualty_reports (
        side_id, source_id, casualty_type, reported_value, extraction_method
    ) VALUES (
        :side_id, :source_id, :casualty_type, :value,
        CAST('infobox_parser' AS extraction_method)
    )
    """
)
_INSERT_MISSING = text(
    """
    INSERT INTO missing_data_log (battle_id, side_id, field_name, missingness_class, notes)
    VALUES (:battle_id, :side_id, :field_name, CAST('unclassified' AS missingness_class), :notes)
    """
)


def _battle(conn: Connection, name: str, date: str = "1900-01-01") -> int:
    """Insert a battle and return its id."""
    return int(conn.execute(_INSERT_BATTLE, {"name": name, "date": date}).scalar_one())


def _side(conn: Connection, battle_id: int, label: str = "Side A") -> int:
    """Insert a side and return its id."""
    return int(conn.execute(_INSERT_SIDE, {"battle_id": battle_id, "label": label}).scalar_one())


def _source(conn: Connection, url: str, source_type: str = "wikipedia_infobox") -> int:
    """Insert a source and return its id."""
    return int(conn.execute(_INSERT_SOURCE, {"source_type": source_type, "url": url}).scalar_one())


def _troop(
    conn: Connection,
    side_id: int,
    source_id: int,
    value: float,
    *,
    branch: str = "total",
    scope: str = "engaged",
    context: str = "",
) -> None:
    """Insert one troop report."""
    conn.execute(
        _INSERT_TROOP,
        {
            "side_id": side_id,
            "source_id": source_id,
            "branch": branch,
            "value": value,
            "scope": scope,
            "is_estimate": False,
            "is_upper_bound": False,
            "is_lower_bound": False,
            "context": context,
        },
    )


def _casualty(
    conn: Connection, side_id: int, source_id: int, value: float, *, casualty_type: str = "total"
) -> None:
    """Insert one casualty report."""
    conn.execute(
        _INSERT_CASUALTY,
        {
            "side_id": side_id,
            "source_id": source_id,
            "casualty_type": casualty_type,
            "value": value,
        },
    )


# ─── The shared, once-fitted corpus ──────────────────────────────────────────


@pytest.fixture(scope="module")
def seed_ids(engine: Engine) -> dict[str, int]:
    """Seed one corpus covering every scenario the shared fit needs to prove.

    Committed for real (not the rolled-back ``conn`` fixture): the
    module-scoped ``reconciled`` fixture opens its own connection to fit
    against it, and that has to see this data.

    Returns:
        Every id a test might need, by name.
    """
    setup = engine.connect()

    source1 = _source(setup, "doc-a")
    source2 = _source(setup, "doc-b")
    source_dup1 = _source(setup, "shared-doc")
    source_dup2 = _source(setup, "shared-doc")

    battle_alpha = _battle(setup, "Battle Alpha Fixture", "1900-06-15")
    side_multi = _side(setup, battle_alpha, "Multi Side")
    side_singleton = _side(setup, battle_alpha, "Singleton Side")
    _troop(setup, side_multi, source1, 40_000.0)
    _troop(setup, side_multi, source2, 55_000.0)
    _troop(setup, side_singleton, source1, 20_000.0)
    # Gives the casualties quantity one usable report too, so the shared run
    # actually fits both quantities under the one model_runs row -- see
    # test_one_run_covers_both_quantities_...
    _casualty(setup, side_multi, source1, 5_000.0)

    # 216 BC, astronomical -215 -- the same fact test_reconcile_load.py pins,
    # exercised here through the whole stage rather than just the loader.
    battle_bc = _battle(setup, "Battle Bravo Ancient Fixture", "0216-08-02 BC")
    side_bc = _side(setup, battle_bc, "BC Side")
    _troop(setup, side_bc, source1, 50_000.0)

    battle_cav = _battle(setup, "Battle Charlie Cavalry Fixture", "1850-01-01")
    side_cavalry = _side(setup, battle_cav, "Cavalry Only Side")
    _troop(setup, side_cavalry, source1, 5_000.0, branch="cavalry")

    battle_none = _battle(setup, "Battle Delta No Reports Fixture", "1860-01-01")
    side_no_reports = _side(setup, battle_none, "No Reports Side")
    setup.execute(
        _INSERT_MISSING,
        {
            "battle_id": battle_none,
            "side_id": side_no_reports,
            "field_name": "troop_total",
            "notes": "no troop reports extracted",
        },
    )

    battle_dup = _battle(setup, "Battle Echo Duplicate Source Fixture", "1870-01-01")
    side_dup = _side(setup, battle_dup, "Dup Source Side")
    _troop(setup, side_dup, source_dup1, 30_000.0)
    _troop(setup, side_dup, source_dup2, 32_000.0)

    setup.commit()
    setup.close()

    return {
        "source1": source1,
        "source2": source2,
        "source_dup1": source_dup1,
        "source_dup2": source_dup2,
        "side_multi": side_multi,
        "side_singleton": side_singleton,
        "side_bc": side_bc,
        "side_cavalry": side_cavalry,
        "side_no_reports": side_no_reports,
        "side_dup": side_dup,
        "battle_bc": battle_bc,
    }


@pytest.fixture(scope="module")
def reconciled(
    engine: Engine, seed_ids: dict[str, int], tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """Run the real stage once, committed, over the shared corpus.

    Uses ``context=StageContext(db_conn=...)`` and commits explicitly here,
    mirroring how a caller that supplies its own connection is expected to
    manage it (the stage itself never commits a connection it did not open;
    see ``pipeline/stages/reconcile.py``'s module docstring).

    Returns:
        The directory ``reconciliation_report.json`` was written into.
    """
    del seed_ids  # ordering dependency only; ids come back through the dict fixture
    processed_root = tmp_path_factory.mktemp("reconcile_report")
    spec = _load_spec(processed_root)

    # try/finally, because a stage that raises part-way leaves this
    # connection's transaction open, holding row locks on every side it
    # touched, and every later test that runs the stage then waits on them
    # forever instead of failing. That hung this file for 40 minutes twice.
    conn = engine.connect()
    try:
        reconcile_stage.run(spec, context=StageContext(db_conn=conn))
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()

    return processed_root


# ─── Shared-corpus tests ─────────────────────────────────────────────────────


def test_a_side_with_one_report_gets_an_interval_spanning_more_than_a_factor_of_two(
    conn: Connection, reconciled: Path, seed_ids: dict[str, int]
) -> None:
    del reconciled
    lo, hi = conn.execute(
        text(
            "SELECT est_troops_total_lo, est_troops_total_hi FROM battle_sides WHERE side_id = :s"
        ),
        {"s": seed_ids["side_singleton"]},
    ).one()
    assert lo is not None and hi is not None
    assert lo > 0
    # A factor of 2 is LN(2) = 0.693 on the log scale; a singleton side's
    # honest uncertainty (source bias sd alone is ~0.85, docs/handover-log.md §19.7)
    # should clear it comfortably.
    assert math.log(float(hi) / float(lo)) > math.log(2.0)


def test_a_side_whose_only_report_is_a_cavalry_count_gets_a_missing_data_log_row(
    conn: Connection, reconciled: Path, seed_ids: dict[str, int]
) -> None:
    del reconciled
    rows = conn.execute(
        text("SELECT field_name, notes FROM missing_data_log WHERE side_id = :s"),
        {"s": seed_ids["side_cavalry"]},
    ).fetchall()
    assert any(row[0] == "troop_total" and str(row[1]).startswith("reconcile: ") for row in rows), (
        rows
    )


def test_reconcile_does_not_delete_the_missing_data_rows_extract_wrote(
    conn: Connection, reconciled: Path, seed_ids: dict[str, int]
) -> None:
    del reconciled
    rows = conn.execute(
        text(
            "SELECT notes FROM missing_data_log WHERE side_id = :s AND field_name = 'troop_total'"
        ),
        {"s": seed_ids["side_no_reports"]},
    ).fetchall()
    # extract's row (no 'reconcile: ' prefix) must still be there: write_unfillable
    # only clears rows carrying that prefix, and this side has no troop_reports
    # rows at all, so it is never even a candidate for reconcile's own log write.
    assert any(str(row[0]) == "no troop reports extracted" for row in rows), rows


def test_two_source_rows_for_the_same_document_receive_the_same_bias_summary(
    conn: Connection, reconciled: Path, seed_ids: dict[str, int]
) -> None:
    del reconciled
    rows = conn.execute(
        text(
            "SELECT source_id, troop_bias_mu, troop_bias_sd, bias_n_troop_reports "
            "FROM sources WHERE source_id IN (:a, :b)"
        ),
        {"a": seed_ids["source_dup1"], "b": seed_ids["source_dup2"]},
    ).fetchall()
    by_id = {int(row[0]): row[1:] for row in rows}
    a = by_id[seed_ids["source_dup1"]]
    b = by_id[seed_ids["source_dup2"]]
    # One SourceKey, one fitted bias, fanned out identically to both rows --
    # see pipeline/reconcilers/summarise.py's _source_biases.
    assert a == b
    # Both reports counted once against the shared key, not once each.
    assert int(a[2]) == 2


def test_a_bc_battle_is_classified_ancient_without_reading_date_start(
    conn: Connection, reconciled: Path, seed_ids: dict[str, int]
) -> None:
    del reconciled
    year = conn.execute(
        text("SELECT year_astronomical FROM battles WHERE battle_id = :b"),
        {"b": seed_ids["battle_bc"]},
    ).scalar_one()
    assert year == -215

    value = conn.execute(
        text("SELECT est_troops_total FROM battle_sides WHERE side_id = :s"),
        {"s": seed_ids["side_bc"]},
    ).scalar_one()
    # A crash reading date_start (datetime.date has MINYEAR == 1) would have
    # taken the whole troops fit down with it; a written estimate is proof
    # the whole run, this BC side included, went through year_astronomical.
    assert value is not None


def test_every_quality_check_in_the_reconcile_spec_executes(
    conn: Connection, reconciled: Path
) -> None:
    del reconciled
    raw = yaml.safe_load((AGENTS / "reconcile.yaml").read_text(encoding="utf-8"))
    checks = raw["quality_checks"]
    assert len(checks) == 23

    runner = QualityRunner(conn)
    results = runner.run_all(checks)
    assert len(results) == len(checks)
    for result in results:
        assert "SQL execution failed" not in result.message, result
        assert "is not implemented" not in result.message, result


_MISSING_DATA_ALL_LOGGED = text(
    """
    SELECT COUNT(*) FROM battle_sides bs
    WHERE bs.est_troops_total IS NULL
      AND NOT EXISTS (
        SELECT 1 FROM missing_data_log mdl
        WHERE mdl.side_id = bs.side_id AND mdl.field_name = 'troop_total'
      )
    """
)


def test_the_classify_missing_troop_total_gate_passes_after_reconcile_has_run(
    conn: Connection, reconciled: Path
) -> None:
    del reconciled
    # Copied verbatim from agents/classify.yaml's missing_data_all_logged
    # check: every side with no troop total, whether extract logged it (no
    # reports at all) or reconcile did (reports that were all unusable),
    # must have a row.
    count = conn.execute(_MISSING_DATA_ALL_LOGGED).scalar_one()
    assert count == 0


_ESTIMATES_ARE_NOT_STALE = text(
    """
    SELECT COUNT(*)
    FROM battle_sides bs
    WHERE bs.est_troops_total IS NOT NULL
      AND EXISTS (SELECT 1 FROM troop_reports tr WHERE tr.side_id = bs.side_id)
      AND bs.est_troops_run_id IS DISTINCT FROM (
        SELECT MAX(run_id) FROM model_runs WHERE model_type = 'source_disagreement'
      )
    """
)


def test_one_run_covers_both_quantities_and_the_estimates_are_not_stale_gate_reads_zero(
    conn: Connection, reconciled: Path, seed_ids: dict[str, int]
) -> None:
    del reconciled
    # side_multi carries both a troop report and a casualty report (see
    # seed_ids), so this is the one side that can actually show both
    # provenance columns pointing at the same run.
    troop_run_id, casualties_run_id = conn.execute(
        text(
            "SELECT est_troops_run_id, est_casualties_run_id FROM battle_sides WHERE side_id = :s"
        ),
        {"s": seed_ids["side_multi"]},
    ).one()
    assert troop_run_id is not None
    assert casualties_run_id is not None
    assert troop_run_id == casualties_run_id

    # This test must run before any later test commits a newer completed
    # source_disagreement run (the self-connecting test does, deliberately,
    # near the end of this file): MAX(run_id) is global, not scoped to this
    # corpus, so a newer run elsewhere would make every side here look stale
    # for a reason that has nothing to do with reconcile's own idempotency.
    stale = conn.execute(_ESTIMATES_ARE_NOT_STALE).scalar_one()
    assert stale == 0


def _quality_check(name: str) -> dict[str, Any]:
    raw = yaml.safe_load((AGENTS / "reconcile.yaml").read_text(encoding="utf-8"))
    check = dict(next(c for c in raw["quality_checks"] if c["name"] == name))
    # Resolve "{params.<name>}" the way the orchestrator's gate runner does, so
    # this helper does not depend on which side of that change it runs on.
    check["threshold"] = re.sub(
        r"\{params\.(\w+)\}",
        lambda m: str(raw["params"][m.group(1)]),
        str(check.get("threshold", "")),
    )
    return check


def test_the_neutral_fit_writes_estimates_diagnostics_and_a_draws_file(
    conn: Connection, reconciled: Path, seed_ids: dict[str, int]
) -> None:
    import numpy as np

    side = seed_ids["side_multi"]
    row = conn.execute(
        text(
            "SELECT est_troops_neutral, est_troops_neutral_lo, est_troops_neutral_hi, "
            "est_troops_neutral_log_mean, est_troops_neutral_log_sd, "
            "est_troops_neutral_run_id, "
            "is_defender FROM battle_sides WHERE side_id = :s"
        ),
        {"s": side},
    ).one()
    value, lo, hi, log_mean, log_sd, run_id, is_defender = row
    assert run_id is not None
    assert value is not None and lo < value < hi
    assert math.isclose(math.log(value), log_mean, abs_tol=0.5)
    assert log_sd > 0
    assert is_defender is None  # classify fills it later

    diagnostics = conn.execute(
        text("SELECT diagnostics FROM model_runs WHERE run_id = :r"), {"r": run_id}
    ).scalar_one()
    block = diagnostics["troops_neutral"]
    assert block["written"] is True
    assert set(block["worst"]) == {
        "max_rhat",
        "min_ess_bulk",
        "min_ess_bulk_side",
        "divergences",
        "n_sides_mcse_fail",
        "worst_side",
        "mcse_failed_side_ids",
        "mcse_failed_limits",
    }
    assert block["m0"]["mean"] is not None and block["s0"]["mean"] is not None
    # Neutral diagnostics stay out of the rollup model_convergence reads.
    assert "troops_neutral" not in diagnostics["worst"]

    path = reconciled / "reconcile" / f"neutral_draws_run{run_id}.npz"
    assert path.exists()
    with np.load(path) as data:
        assert int(data["run_id"]) == run_id
        assert data["draws"].shape[1] == len(data["side_id"]) == len(data["battle_id"])
        assert data["draws"].shape[0] <= 1000
        assert data["draws"].dtype == np.float64
        assert side in set(data["side_id"].tolist())
        n_sides = len(data["side_id"])
        assert data["lik_mean"].shape == data["lik_sd"].shape == (n_sides,)
        assert data["lik_has_point"].shape == (n_sides,)
        assert data["lik_loadings"].shape[0] == n_sides
        assert data["lik_loadings"].shape[1] == len(data["lik_loading_names"])
        assert np.all(np.isfinite(data["lik_mean"][data["lik_has_point"]]))
        position = int(np.flatnonzero(data["side_id"] == side)[0])
        assert bool(data["lik_has_point"][position])


def test_the_neutral_gates_execute_and_the_estimates_are_not_stale(
    conn: Connection, reconciled: Path
) -> None:
    del reconciled
    # The gate's SQL hard-codes ess >= 400; this file samples 200 draws on
    # purpose, so only execution is asserted for the convergence gate. Its
    # failing branch is pinned by test_an_unconverged_fit_writes_no_estimates.
    converged = QualityRunner(conn).run_one(_quality_check("neutral_fit_converged"))
    assert "SQL execution failed" not in converged.message, converged
    stale = QualityRunner(conn).run_one(_quality_check("neutral_estimates_are_not_stale"))
    assert "SQL execution failed" not in stale.message, stale
    assert stale.passed, stale


# ─── Standalone tests ────────────────────────────────────────────────────────


def test_the_stage_conforms_to_the_orchestrator_contract() -> None:
    check_conforms(reconcile_stage)


def test_running_reconcile_twice_with_the_same_seed_writes_the_same_estimates(
    conn: Connection, tmp_path: Path
) -> None:
    # Adds one more usable side to whatever the shared fixture already
    # committed and re-fits the whole (now slightly larger) troops corpus
    # twice, back to back, inside this test's own rolled-back transaction --
    # both runs see identical data because nothing else touches this
    # connection between them, and agents/reconcile.yaml pins random_seed, so
    # "same seed, same data" is exactly what two calls in a row give.
    source = _source(conn, "idempotency-doc")
    battle = _battle(conn, "Battle Foxtrot Idempotency", "1905-01-01")
    side = _side(conn, battle, "Idempotency Side")
    _troop(conn, side, source, 25_000.0)

    select_estimate = text(
        "SELECT est_troops_total, est_troops_total_lo, est_troops_total_hi "
        "FROM battle_sides WHERE side_id = :s"
    )

    spec_one = _load_spec(tmp_path / "run1")
    reconcile_stage.run(spec_one, context=StageContext(db_conn=conn))
    first = conn.execute(select_estimate, {"s": side}).one()

    spec_two = _load_spec(tmp_path / "run2")
    reconcile_stage.run(spec_two, context=StageContext(db_conn=conn))
    second = conn.execute(select_estimate, {"s": side}).one()

    assert tuple(first) == tuple(second)


_SINGLETON_GATE_SQL = text(
    """
    SELECT COUNT(*)
    FROM battle_sides bs
    JOIN (
        SELECT side_id, COUNT(DISTINCT source_id) AS n_sources
        FROM troop_reports
        WHERE branch = 'total' AND reported_value > 0
        GROUP BY side_id
    ) c ON c.side_id = bs.side_id
    WHERE c.n_sources = 1
      AND bs.est_troops_run_id IS NOT NULL
      AND bs.est_troops_total_lo > 0
      AND LN(bs.est_troops_total_hi / bs.est_troops_total_lo) < 0.8
    """
)


def test_the_singleton_interval_gate_fires_when_an_interval_is_written_too_narrow(
    conn: Connection,
) -> None:
    # Fabricates the interval directly rather than fitting one, so this test
    # is fast and proves the gate can fail rather than that this run's real
    # fit happens to pass it. Deliberately does not depend on `reconciled`:
    # this is the one test that must never see the honestly-fitted corpus,
    # only its own artificially narrow row.
    source = _source(conn, "narrow-doc")
    battle = _battle(conn, "Battle Golf Narrow", "1910-01-01")
    side = _side(conn, battle, "Narrow Side")
    _troop(conn, side, source, 10_000.0)

    run_id = int(
        conn.execute(
            text(
                "INSERT INTO model_runs "
                "(run_name, model_type, config, started_at, completed_at, diagnostics) "
                "VALUES ('fabricated_for_gate_test', 'source_disagreement', "
                "CAST('{}' AS JSONB), now(), now(), CAST('{}' AS JSONB)) "
                "RETURNING run_id"
            )
        ).scalar_one()
    )
    conn.execute(
        text(
            "UPDATE battle_sides SET "
            "est_troops_total = 10000, est_troops_total_lo = 9950, "
            "est_troops_total_hi = 10050, est_troops_run_id = :run_id, "
            "est_troops_method = 'single_report_debiased', est_troops_updated_at = now() "
            "WHERE side_id = :side_id"
        ),
        {"run_id": run_id, "side_id": side},
    )

    fired = conn.execute(_SINGLETON_GATE_SQL).scalar_one()
    assert fired >= 1


def test_reconcile_opens_its_own_connection_when_the_context_carries_none(
    engine: Engine, tmp_path: Path
) -> None:
    # The orchestrator calls runner_module.run(spec) with no context at all
    # (pipeline/orchestrator.py's run_stage), so context=None is the real
    # path, not a hypothetical one. Exercising it means letting the stage
    # open pipeline.db.get_connection() itself, which reads DATABASE_URL from
    # the environment -- the same database `engine` points at, but a
    # separate physical connection uninvolved in any test's rolled-back
    # transaction. It therefore commits for real, so this test seeds its own
    # small, uniquely-named corpus and deletes exactly what it wrote
    # afterwards, in a `finally`, via a fresh connection of its own (deleting
    # inside a transaction this test then rolled back would undo the
    # deletion along with everything else).
    setup = engine.connect()
    source = _source(setup, "self-connect-doc")
    battle = _battle(setup, "Battle Hotel Self Connect", "1920-01-01")
    side = _side(setup, battle, "Self Connect Side")
    _troop(setup, side, source, 15_000.0)
    setup.commit()
    setup.close()

    try:
        spec = _load_spec(tmp_path)
        reconcile_stage.run(spec)  # no context: opens and commits its own connection

        verify = engine.connect()
        try:
            value = verify.execute(
                text("SELECT est_troops_total FROM battle_sides WHERE side_id = :s"),
                {"s": side},
            ).scalar_one()
        finally:
            verify.close()
        assert value is not None
    finally:
        cleanup = engine.connect()
        cleanup.execute(text("DELETE FROM missing_data_log WHERE battle_id = :b"), {"b": battle})
        cleanup.execute(text("DELETE FROM troop_reports WHERE side_id = :s"), {"s": side})
        cleanup.execute(text("DELETE FROM battle_sides WHERE battle_id = :b"), {"b": battle})
        cleanup.execute(text("DELETE FROM battles WHERE battle_id = :b"), {"b": battle})
        cleanup.execute(text("DELETE FROM sources WHERE source_id = :s"), {"s": source})
        # model_runs rows this run wrote are left standing: model_runs is
        # documented as history (config/schema.sql), and nothing references
        # the deleted battle/side/source from it any more.
        cleanup.commit()
        cleanup.close()


def test_an_unconverged_fit_writes_no_estimates(engine: Engine, tmp_path: Path) -> None:
    # An ess floor no fit can reach stands in for a fit that did not converge.
    # The run must still be recorded with its diagnostics, so model_convergence
    # fails loudly, but the side keeps whatever estimate it had before.
    setup = engine.connect()
    source = _source(setup, "unconverged-doc")
    battle = _battle(setup, "Battle India Unconverged", "1915-01-01")
    side = _side(setup, battle, "Unconverged Side")
    _troop(setup, side, source, 12_000.0)
    setup.commit()
    setup.close()

    conn = engine.connect()
    try:
        spec = _load_spec(tmp_path, write_min_ess_bulk=1e12)
        reconcile_stage.run(spec, context=StageContext(db_conn=conn))
        estimate = conn.execute(
            text("SELECT est_troops_total FROM battle_sides WHERE side_id = :s"), {"s": side}
        ).scalar_one()
        completed = conn.execute(
            text(
                "SELECT COUNT(*) FROM model_runs WHERE completed_at IS NOT NULL "
                "AND diagnostics IS NOT NULL AND run_id = (SELECT MAX(run_id) FROM model_runs)"
            )
        ).scalar_one()
        neutral_estimate = conn.execute(
            text("SELECT est_troops_neutral FROM battle_sides WHERE side_id = :s"), {"s": side}
        ).scalar_one()
        neutral_gate = QualityRunner(conn).run_one(_quality_check("neutral_fit_converged"))
        draws_files = list((tmp_path / "reconcile").glob("neutral_draws_run*.npz"))
        withheld_notes = (
            conn.execute(
                text(
                    "SELECT notes FROM missing_data_log "
                    "WHERE side_id = :s AND field_name = 'troop_total'"
                ),
                {"s": side},
            )
            .scalars()
            .all()
        )
        unfillable_gate = QualityRunner(conn).run_one(_quality_check("unfillable_sides_are_logged"))
        populated_gate = QualityRunner(conn).run_one(_quality_check("estimates_populated"))
    finally:
        conn.rollback()
        conn.close()
        cleanup = engine.connect()
        cleanup.execute(text("DELETE FROM missing_data_log WHERE battle_id = :b"), {"b": battle})
        cleanup.execute(text("DELETE FROM troop_reports WHERE side_id = :s"), {"s": side})
        cleanup.execute(text("DELETE FROM battle_sides WHERE battle_id = :b"), {"b": battle})
        cleanup.execute(text("DELETE FROM battles WHERE battle_id = :b"), {"b": battle})
        cleanup.execute(text("DELETE FROM sources WHERE source_id = :s"), {"s": source})
        cleanup.commit()
        cleanup.close()

    assert estimate is None
    assert completed == 1
    # The neutral fit is judged by the same thresholds: nothing written, the
    # gate fails, and there is no fallback to the published estimates.
    assert neutral_estimate is None
    assert draws_files == []
    assert not neutral_gate.passed
    # A withheld side reaches missing_data_log, naming the run and the breach, so
    # unfillable_sides_are_logged agrees with the stage, while estimates_populated
    # still counts it as a failure rather than as unfillable.
    assert len(withheld_notes) == 1
    assert withheld_notes[0].startswith("reconcile: estimate withheld")
    assert "min_ess_bulk" in withheld_notes[0]
    assert unfillable_gate.passed, unfillable_gate
    assert not populated_gate.passed, populated_gate


def test_a_side_failing_only_its_mcse_is_written_with_a_widened_interval(
    engine: Engine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pipeline.reconcilers import summarise as summarise_module

    setup = engine.connect()
    source = _source(setup, "mcse-doc")
    battle = _battle(setup, "Battle India Mcse", "1916-01-01")
    good = _side(setup, battle, "Passing Side")
    bad = _side(setup, battle, "Noisy Side")
    _troop(setup, good, source, 12_000.0)
    _troop(setup, bad, source, 300.0)
    setup.commit()
    setup.close()

    def one_noisy_side(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return 1, None, {str(bad): "relative"}

    before: dict[int, Any] = {}
    real_widen = summarise_module._widen_low_precision

    def record_then_widen(estimates: Any, margins: Any) -> Any:
        before.update({e.side_id: e for e in estimates})
        return real_widen(estimates, margins)

    monkeypatch.setattr(summarise_module, "_side_quantile_mcse", one_noisy_side)
    monkeypatch.setattr(summarise_module, "_widen_low_precision", record_then_widen)

    conn = engine.connect()
    try:
        spec = _load_spec(tmp_path)
        reconcile_stage.run(spec, context=StageContext(db_conn=conn))
        rows = {
            side_id: (value, lo, hi, run_id)
            for side_id, value, lo, hi, run_id in conn.execute(
                text(
                    "SELECT side_id, est_troops_total, est_troops_total_lo, "
                    "est_troops_total_hi, est_troops_run_id "
                    "FROM battle_sides WHERE battle_id = :b"
                ),
                {"b": battle},
            ).all()
        }
        diagnostics = conn.execute(
            text("SELECT diagnostics FROM model_runs WHERE run_id = :r"), {"r": rows[bad][3]}
        ).scalar_one()
        logged = conn.execute(
            text("SELECT COUNT(*) FROM missing_data_log WHERE side_id IN (:g, :b)"),
            {"g": good, "b": bad},
        ).scalar_one()
        unfillable_gate = QualityRunner(conn).run_one(_quality_check("unfillable_sides_are_logged"))
        stale_gate = QualityRunner(conn).run_one(_quality_check("estimates_are_not_stale"))
    finally:
        conn.rollback()
        conn.close()
        cleanup = engine.connect()
        cleanup.execute(text("DELETE FROM missing_data_log WHERE battle_id = :b"), {"b": battle})
        cleanup.execute(
            text("DELETE FROM troop_reports WHERE side_id IN (:g, :b)"), {"g": good, "b": bad}
        )
        cleanup.execute(text("DELETE FROM battle_sides WHERE battle_id = :b"), {"b": battle})
        cleanup.execute(text("DELETE FROM battles WHERE battle_id = :b"), {"b": battle})
        cleanup.execute(text("DELETE FROM sources WHERE source_id = :s"), {"s": source})
        cleanup.commit()
        cleanup.close()

    value, lo, hi, _ = rows[bad]
    assert value is not None and lo is not None and hi is not None
    assert value == pytest.approx(before[bad].value, rel=1e-6)
    assert lo < before[bad].lo and hi > before[bad].hi
    assert rows[good][1] == pytest.approx(before[good].lo, rel=1e-6)
    assert rows[good][2] == pytest.approx(before[good].hi, rel=1e-6)
    assert diagnostics["troops"]["n_sides_low_precision"] == 1
    assert diagnostics["troops"]["mcse_failed_side_ids"] == [bad]
    assert logged == 0
    assert unfillable_gate.passed, unfillable_gate
    assert stale_gate.passed, stale_gate


def test_a_later_estimate_deletes_a_stale_withheld_row_and_withheld_rows_do_not_stack(
    engine: Engine,
) -> None:
    from datetime import UTC, datetime

    from pipeline.reconcilers import store
    from pipeline.reconcilers.records import SideEstimate

    conn = engine.connect()
    source = _source(conn, "stale-doc")
    battle = _battle(conn, "Battle India Stale", "1917-01-01")
    side = _side(conn, battle, "Stale Side")
    _troop(conn, side, source, 5_000.0)
    conn.commit()
    try:
        store.log_withheld_sides(conn, "troops", side_ids={side}, run_id=1, reason="x")
        store.log_withheld_sides(conn, "troops", side_ids={side}, run_id=1, reason="x")
        # A later failed run names a different run id and reason, and must not add a row.
        store.log_withheld_sides(conn, "troops", side_ids={side}, run_id=2, reason="y")
        count = text(
            "SELECT COUNT(*) FROM missing_data_log WHERE side_id = :s "
            "AND notes LIKE 'reconcile: estimate withheld%'"
        )
        assert conn.execute(count, {"s": side}).scalar_one() == 1

        run_id = store.start_model_run(conn, model_type="test", run_name="stale", config={})
        store.write_side_estimates(
            conn,
            [
                SideEstimate(
                    side_id=side,
                    quantity="troops",
                    value=5_000.0,
                    lo=4_000.0,
                    hi=6_000.0,
                    run_id=run_id,
                    n_reports=1,
                    n_sources=1,
                    method="single_report_debiased",
                    updated_at=datetime.now(UTC),
                )
            ],
        )
        assert conn.execute(count, {"s": side}).scalar_one() == 0
    finally:
        conn.rollback()
        conn.close()
        cleanup = engine.connect()
        cleanup.execute(text("DELETE FROM missing_data_log WHERE battle_id = :b"), {"b": battle})
        cleanup.execute(text("DELETE FROM troop_reports WHERE side_id = :s"), {"s": side})
        cleanup.execute(text("DELETE FROM battle_sides WHERE battle_id = :b"), {"b": battle})
        cleanup.execute(text("DELETE FROM battles WHERE battle_id = :b"), {"b": battle})
        cleanup.execute(text("DELETE FROM sources WHERE source_id = :s"), {"s": source})
        cleanup.commit()
        cleanup.close()


def _write_troop_estimate(conn: Connection, side_ids: list[int]) -> None:
    """Write a troop estimate for each side through the production store."""
    from datetime import UTC, datetime

    from pipeline.reconcilers import store
    from pipeline.reconcilers.records import SideEstimate

    run_id = store.start_model_run(conn, model_type="test", run_name="gap", config={})
    store.write_side_estimates(
        conn,
        [
            SideEstimate(
                side_id=side_id,
                quantity="troops",
                value=5_000.0,
                lo=4_000.0,
                hi=6_000.0,
                run_id=run_id,
                n_reports=1,
                n_sources=1,
                method="single_report_debiased",
                updated_at=datetime.now(UTC),
            )
            for side_id in side_ids
        ],
    )


def _log_troop_gap(
    conn: Connection,
    battle: int,
    side: int,
    notes: str,
    *,
    missingness_class: str = "unclassified",
    was_imputed: bool = False,
) -> None:
    conn.execute(
        text(
            "INSERT INTO missing_data_log (battle_id, side_id, field_name, missingness_class, "
            "notes, was_imputed) VALUES (:b, :s, 'troop_total', "
            "CAST(:c AS missingness_class), :n, :i)"
        ),
        {"b": battle, "s": side, "c": missingness_class, "n": notes, "i": was_imputed},
    )


def _gap_notes(conn: Connection, side: int) -> list[str]:
    rows = conn.execute(
        text(
            "SELECT notes FROM missing_data_log WHERE side_id = :s "
            "AND field_name = 'troop_total' ORDER BY log_id"
        ),
        {"s": side},
    ).fetchall()
    return [str(r[0]) for r in rows]


def test_an_estimate_deletes_the_classify_no_estimate_row_for_its_side(conn: Connection) -> None:
    from pipeline.classifiers.records import UNLOGGED_TROOP_TOTAL_NOTE

    source = _source(conn, "gap-doc")
    battle = _battle(conn, "Battle of Gap", "1700-01-01")
    plain = _side(conn, battle, "Plain")
    mnar = _side(conn, battle, "Mnar")
    for side in (plain, mnar):
        _troop(conn, side, source, 5_000.0)
    _log_troop_gap(conn, battle, plain, UNLOGGED_TROOP_TOTAL_NOTE)
    _log_troop_gap(
        conn,
        battle,
        mnar,
        f"{UNLOGGED_TROOP_TOTAL_NOTE} | classify: loser, winner known",
        missingness_class="mnar",
    )

    _write_troop_estimate(conn, [plain, mnar])

    assert _gap_notes(conn, plain) == []
    assert _gap_notes(conn, mnar) == []


def test_other_notes_and_imputed_rows_survive_an_estimate(conn: Connection) -> None:
    from pipeline.classifiers.records import UNLOGGED_TROOP_TOTAL_NOTE

    source = _source(conn, "gap-doc-2")
    battle = _battle(conn, "Battle of Keep", "1700-01-01")
    side = _side(conn, battle, "Keep")
    _troop(conn, side, source, 5_000.0)
    _log_troop_gap(conn, battle, side, "no troop reports extracted")
    _log_troop_gap(conn, battle, side, UNLOGGED_TROOP_TOTAL_NOTE, was_imputed=True)

    _write_troop_estimate(conn, [side])

    assert _gap_notes(conn, side) == ["no troop reports extracted", UNLOGGED_TROOP_TOTAL_NOTE]


def test_a_side_without_an_estimate_keeps_its_classify_row(conn: Connection) -> None:
    from pipeline.classifiers.records import UNLOGGED_TROOP_TOTAL_NOTE

    source = _source(conn, "gap-doc-3")
    battle = _battle(conn, "Battle of Silence Two", "1700-01-01")
    estimated = _side(conn, battle, "Estimated")
    unusable = _side(conn, battle, "Unusable")
    _troop(conn, estimated, source, 5_000.0)
    _log_troop_gap(conn, battle, estimated, UNLOGGED_TROOP_TOTAL_NOTE)
    _log_troop_gap(conn, battle, unusable, UNLOGGED_TROOP_TOTAL_NOTE)

    _write_troop_estimate(conn, [estimated])

    assert _gap_notes(conn, estimated) == []
    assert _gap_notes(conn, unusable) == [UNLOGGED_TROOP_TOTAL_NOTE]


def _fabricate_run(conn: Connection, diagnostics_json: str) -> None:
    conn.execute(
        text(
            "INSERT INTO model_runs "
            "(run_name, model_type, config, started_at, completed_at, diagnostics) "
            "VALUES ('fabricated_for_spread_gate', 'source_disagreement', "
            "CAST('{}' AS JSONB), now(), now() + interval '1 hour', CAST(:d AS JSONB))"
        ),
        {"d": diagnostics_json},
    )


def test_the_collapse_gate_fails_a_run_whose_estimates_barely_spread(conn: Connection) -> None:
    # The pilot corpus gave every side ~35k on a clean convergence, at a spread
    # ratio near 0.005. Nothing else in the spec could see it.
    _fabricate_run(conn, '{"troops": {"spread_ratio": 0.005}}')

    result = QualityRunner(conn).run_one(_quality_check("estimates_not_collapsed"))

    assert not result.passed, result
    assert "SQL execution failed" not in result.message


def test_the_collapse_gate_passes_a_run_whose_estimates_track_the_reports(
    conn: Connection,
) -> None:
    _fabricate_run(conn, '{"troops": {"spread_ratio": 0.9}}')

    result = QualityRunner(conn).run_one(_quality_check("estimates_not_collapsed"))

    assert result.passed, result


def test_the_collapse_gate_now_fails_a_run_that_keeps_only_84_percent_of_the_spread(
    conn: Connection,
) -> None:
    # The 2026-10 corpus fit passed the old 0.25 limit at 0.65 while shrinking every
    # lone report.
    _fabricate_run(conn, '{"troops": {"spread_ratio": 0.84}}')

    assert not QualityRunner(conn).run_one(_quality_check("estimates_not_collapsed")).passed


def test_the_collapse_gate_reads_the_recovery_block_and_skips_when_it_is_null(
    conn: Connection,
) -> None:
    # A recovery block replaces the raw ratio: null means "too few sides", which skips.
    _fabricate_run(
        conn,
        '{"troops": {"spread_ratio": 0.1, "recovery": {"gate_spread_ratio": null}}}',
    )
    assert QualityRunner(conn).run_one(_quality_check("estimates_not_collapsed")).passed

    _fabricate_run(
        conn,
        '{"troops": {"spread_ratio": 0.99, "recovery": {"gate_spread_ratio": 0.5}}}',
    )
    assert not QualityRunner(conn).run_one(_quality_check("estimates_not_collapsed")).passed


def test_the_collapse_gate_does_not_fail_a_run_with_no_measurable_spread(
    conn: Connection,
) -> None:
    # One side, or no point observations: the ratio is undefined, not collapsed.
    _fabricate_run(conn, '{"troops": {"spread_ratio": null}}')

    assert QualityRunner(conn).run_one(_quality_check("estimates_not_collapsed")).passed


def _recovery(**gates: object) -> str:
    """A fabricated diagnostics payload carrying published and neutral recovery blocks."""
    import json

    published = {k: v for k, v in gates.items() if not k.startswith("neutral_")}
    neutral = {k.removeprefix("neutral_"): v for k, v in gates.items() if k.startswith("neutral_")}
    return json.dumps(
        {
            "troops": {"recovery": published},
            "troops_neutral": {"recovery": neutral},
        }
    )


@pytest.mark.parametrize(
    ("gate", "key", "bad", "good"),
    [
        ("singleton_slope_recovers", "gate_singleton_slope", 0.61, 0.9),
        ("within_battle_ratio_recovers", "gate_within_battle_slope", 0.6, 0.85),
    ],
)
def test_the_slope_gates_fail_a_compressed_fit_and_pass_a_recovered_one(
    conn: Connection, gate: str, key: str, bad: float, good: float
) -> None:
    check = _quality_check(gate)
    assert check["severity"] == "error"

    _fabricate_run(conn, _recovery(**{key: good, f"neutral_{key}": good}))
    assert QualityRunner(conn).run_one(check).passed

    _fabricate_run(conn, _recovery(**{key: good, f"neutral_{key}": bad}))
    assert not QualityRunner(conn).run_one(check).passed, "the neutral fit counts too"

    _fabricate_run(conn, _recovery(**{key: bad}))
    assert not QualityRunner(conn).run_one(check).passed


@pytest.mark.parametrize(
    "gate",
    [
        "singleton_slope_recovers",
        "within_battle_ratio_recovers",
        "post1800_ratio_in_band",
        "between_battle_variance_is_plausible",
    ],
)
def test_the_recovery_gates_skip_when_their_inputs_are_too_few(conn: Connection, gate: str) -> None:
    # A small fixture: the diagnostics hold nulls and a reason, and the gate passes.
    _fabricate_run(conn, _recovery(gates_skipped={gate: "too few"}))

    assert QualityRunner(conn).run_one(_quality_check(gate)).passed


def test_the_post_1800_band_and_the_variance_check_are_warnings(conn: Connection) -> None:
    band = _quality_check("post1800_ratio_in_band")
    ppc = _quality_check("between_battle_variance_is_plausible")
    assert band["severity"] == ppc["severity"] == "warning"

    _fabricate_run(conn, _recovery(gate_post1800_band_violation=1))
    assert not QualityRunner(conn).run_one(band).passed
    _fabricate_run(conn, _recovery(gate_post1800_band_violation=0))
    assert QualityRunner(conn).run_one(band).passed

    _fabricate_run(
        conn,
        '{"troops": {"recovery": {"between_battle_ppc": {"extremity": 0.004}}}, '
        '"troops_neutral": {"recovery": {"between_battle_ppc": {"extremity": 0.4}}}}',
    )
    assert not QualityRunner(conn).run_one(ppc).passed


def test_a_run_records_its_spread_ratio_and_source_effect_counts(
    conn: Connection, reconciled: Path
) -> None:
    del reconciled
    diagnostics = conn.execute(
        text(
            "SELECT diagnostics FROM model_runs WHERE model_type = 'source_disagreement' "
            "AND completed_at IS NOT NULL AND diagnostics -> 'troops' ? 'n_sources' "
            "ORDER BY run_id DESC LIMIT 1"
        )
    ).scalar_one()
    troops = diagnostics["troops"]

    assert "spread_ratio" in troops
    recovery = troops["recovery"]
    assert {
        "singleton_slope",
        "within_battle_slope",
        "post1800_large_ratio",
        "between_battle_ppc",
        "gate_singleton_slope",
        "gate_within_battle_slope",
        "gate_post1800_band_violation",
        "gates_skipped",
    } <= set(recovery)
    # The seeded corpus is far below every minimum size, so every gate skipped, loudly.
    assert recovery["gate_singleton_slope"] is None
    assert "singleton_slope" in recovery["gates_skipped"]
    assert troops["noise_prior"]["applies"] is True
    assert troops["nu"]["mean"] is not None and troops["nu"]["mean"] >= 3.0
    assert "recovery" in diagnostics["troops_neutral"]
    honoured = troops["reports_honoured"]
    assert set(honoured) == {"n_checked", "n_violations", "violating_side_ids", "within_side_mad"}
    assert "s0" in troops and "sigma_type" in troops
    assert (
        troops["n_source_keys_own_effect"] + troops["n_source_keys_pooled"] == troops["n_sources"]
    )


def test_the_reports_honoured_gate_fails_on_a_violation_and_passes_without(
    conn: Connection,
) -> None:
    _fabricate_run(
        conn,
        '{"troops": {"reports_honoured": {"n_violations": 2, "violating_side_ids": [1, 2]}}}',
    )
    failing = QualityRunner(conn).run_one(_quality_check("reports_honoured"))
    assert not failing.passed, failing
    assert "SQL execution failed" not in failing.message

    _fabricate_run(conn, '{"troops": {"reports_honoured": {"n_violations": 0}}}')
    assert QualityRunner(conn).run_one(_quality_check("reports_honoured")).passed
