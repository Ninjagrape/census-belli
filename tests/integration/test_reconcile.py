"""
Integration tests for the reconcile stage's write path, against a real PostgreSQL.

No reconcile estimate has ever been written to a database before this file
(handover.md §19.6). Everything upstream -- the loader, the design matrix,
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
    "INSERT INTO battle_sides (battle_id, side_label) VALUES (:battle_id, :label) "
    "RETURNING side_id"
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
    return int(
        conn.execute(_INSERT_SOURCE, {"source_type": source_type, "url": url}).scalar_one()
    )


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
            "SELECT est_troops_total_lo, est_troops_total_hi FROM battle_sides "
            "WHERE side_id = :s"
        ),
        {"s": seed_ids["side_singleton"]},
    ).one()
    assert lo is not None and hi is not None
    assert lo > 0
    # A factor of 2 is LN(2) = 0.693 on the log scale; a singleton side's
    # honest uncertainty (source bias sd alone is ~0.85, handover.md §19.7)
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
    assert any(
        row[0] == "troop_total" and str(row[1]).startswith("reconcile: ") for row in rows
    ), rows


def test_reconcile_does_not_delete_the_missing_data_rows_extract_wrote(
    conn: Connection, reconciled: Path, seed_ids: dict[str, int]
) -> None:
    del reconciled
    rows = conn.execute(
        text(
            "SELECT notes FROM missing_data_log "
            "WHERE side_id = :s AND field_name = 'troop_total'"
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
    assert len(checks) == 8

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
            "SELECT est_troops_run_id, est_casualties_run_id FROM battle_sides "
            "WHERE side_id = :s"
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
