"""Integration tests: a side the filters newly drop must not keep its old estimate.

Against a real PostgreSQL, like the other reconcile integration modules. The
failure these pin: a re-run excludes a side's troop rows (a day number, a vessel
count), the side still has casualty rows, and reconcile neither logged it nor
cleared the troop estimate an earlier fit gave it, so a stale 35,000 sat beside
current numbers with an old run id.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from pipeline.db import DatabaseConfigError, apply_schema, database_url, get_engine
from pipeline.reconcilers import store
from pipeline.reconcilers.load import classify_sides, load_reports, write_unfillable
from pipeline.reconcilers.records import ReconcileCounts


def _available() -> bool:
    try:
        engine = get_engine(database_url(), pool_pre_ping=False)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
    except (DatabaseConfigError, Exception):
        return False
    return True


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _available(), reason="No database available. Run `make db-up` first."),
]


@pytest.fixture(scope="module")
def engine() -> Iterator[Engine]:
    eng = get_engine()
    db_name = (eng.url.database or "").lower()
    if not any(token in db_name for token in ("test", "general_war")):
        eng.dispose()
        pytest.skip(f"Refusing to reset database {db_name!r}.")
    apply_schema(eng, drop_existing=True)
    yield eng
    eng.dispose()


@pytest.fixture
def conn(engine: Engine) -> Iterator[Connection]:
    connection = engine.connect()
    transaction = connection.begin()
    try:
        yield connection
    finally:
        transaction.rollback()
        connection.close()


def _one(conn: Connection, sql: str, **params: object) -> int:
    return int(conn.execute(text(sql), params).scalar_one())


def _battle(conn: Connection, name: str) -> int:
    return _one(
        conn,
        "INSERT INTO battles (name, date_start, date_precision) "
        "VALUES (:n, DATE '1815-06-18', 'day') RETURNING battle_id",
        n=name,
    )


def _side(conn: Connection, battle: int, label: str) -> int:
    return _one(
        conn,
        "INSERT INTO battle_sides (battle_id, side_label) VALUES (:b, :l) RETURNING side_id",
        b=battle,
        l=label,
    )


def _source(conn: Connection, url: str) -> int:
    return _one(
        conn,
        "INSERT INTO sources (source_type, url) "
        "VALUES (CAST('wikipedia_infobox' AS source_type), :u) RETURNING source_id",
        u=url,
    )


def _troop(conn: Connection, side: int, source: int, value: float, context: str) -> None:
    conn.execute(
        text(
            "INSERT INTO troop_reports (side_id, source_id, branch, reported_value, scope, "
            "extraction_method, extracted_context) VALUES (:s, :src, CAST('total' AS "
            "troop_branch), :v, 'engaged', CAST('infobox_parser' AS extraction_method), :c)"
        ),
        {"s": side, "src": source, "v": value, "c": context},
    )


def _casualty(conn: Connection, side: int, source: int, value: float) -> None:
    conn.execute(
        text(
            "INSERT INTO casualty_reports (side_id, source_id, casualty_type, reported_value, "
            "extraction_method) VALUES (:s, :src, 'total', :v, CAST('infobox_parser' AS "
            "extraction_method))"
        ),
        {"s": side, "src": source, "v": value},
    )


def _run(conn: Connection) -> int:
    return store.start_model_run(conn, model_type="source_disagreement", run_name="t", config={})


def test_a_side_with_junk_troops_and_casualties_is_logged_unfillable_for_troops(
    conn: Connection,
) -> None:
    battle = _battle(conn, "Junk With Casualties")
    side = _side(conn, battle, "A")
    source = _source(conn, "d1")
    _troop(conn, side, source, 500.0, "500 transports")
    _casualty(conn, side, source, 1_200.0)

    counts = ReconcileCounts()
    reports = load_reports(conn, ancient_cutoff_year=500, counts=counts)
    modellable, unfillable = classify_sides(reports, conn)
    write_unfillable(conn, unfillable, counts)

    assert side not in modellable
    assert [(u.side_id, u.quantity) for u in unfillable] == [(side, "troops")]
    logged = conn.execute(
        text("SELECT field_name, notes FROM missing_data_log WHERE side_id = :s"), {"s": side}
    ).all()
    assert [tuple(r) for r in logged] == [
        ("troop_total", "reconcile: only implausible or non-personnel figures")
    ]


def test_clearing_nulls_fitted_sides_outside_the_design_and_spares_manual_ones(
    conn: Connection,
) -> None:
    battle = _battle(conn, "Clearing")
    kept = _side(conn, battle, "kept")
    dropped = _side(conn, battle, "dropped")
    manual = _side(conn, battle, "manual")
    run = _run(conn)
    for side in (kept, dropped):
        conn.execute(
            text(
                "UPDATE battle_sides SET est_troops_total = 35000, est_troops_total_lo = 1, "
                "est_troops_total_hi = 9e9, est_troops_run_id = :r, est_troops_n_reports = 2, "
                "est_troops_method = 'x', est_troops_neutral = 1, est_troops_neutral_run_id = :r "
                "WHERE side_id = :s"
            ),
            {"r": run, "s": side},
        )
    conn.execute(
        text("UPDATE battle_sides SET est_troops_total = 777 WHERE side_id = :s"), {"s": manual}
    )

    store.clear_unmodelled(conn, "troops", side_ids={kept}, source_ids=set())
    store.clear_unmodelled_neutral(conn, side_ids={kept})

    row = conn.execute(
        text(
            "SELECT side_id, est_troops_total, est_troops_run_id, est_troops_neutral, "
            "est_troops_neutral_run_id FROM battle_sides WHERE side_id IN (:a, :b, :c) "
            "ORDER BY side_id"
        ),
        {"a": kept, "b": dropped, "c": manual},
    ).all()
    by_side = {r[0]: tuple(r[1:]) for r in row}
    assert by_side[kept] == (35000.0, run, 1.0, run)
    assert by_side[dropped] == (None, None, None, None)
    assert by_side[manual][0] == 777.0


def test_clearing_nulls_the_bias_of_sources_with_no_rows_in_the_design(conn: Connection) -> None:
    run = _run(conn)
    live = _source(conn, "live")
    stale = _source(conn, "stale")
    for source in (live, stale):
        conn.execute(
            text(
                "UPDATE sources SET troop_bias_mu = 0, troop_bias_sd = 0, troop_sigma_mu = 5.4, "
                "troop_sigma_sd = 1, bias_n_troop_reports = 3, bias_run_id = :r "
                "WHERE source_id = :s"
            ),
            {"r": run, "s": source},
        )

    store.clear_unmodelled(conn, "troops", side_ids=set(), source_ids={live})

    rows = conn.execute(
        text(
            "SELECT source_id, troop_bias_mu, troop_sigma_mu, bias_n_troop_reports "
            "FROM sources WHERE source_id IN (:a, :b)"
        ),
        {"a": live, "b": stale},
    ).all()
    by_source = {r[0]: tuple(r[1:]) for r in rows}
    assert by_source[live] == (0.0, 5.4, 3)
    assert by_source[stale] == (None, None, 0)


def test_a_source_that_was_never_fitted_is_left_alone(conn: Connection) -> None:
    source = _source(conn, "never-fitted")

    store.clear_unmodelled(conn, "troops", side_ids=set(), source_ids=set())

    mu, sd = conn.execute(
        text("SELECT troop_bias_mu, troop_bias_sd FROM sources WHERE source_id = :s"),
        {"s": source},
    ).one()
    assert (mu, sd) == (0.0, 1.0)
