"""
Integration tests for the database layer.

These exercise a real PostgreSQL instance: the schema uses enums, JSONB,
array columns, partial indexes and check constraints, none of which SQLite
reproduces, so a substituted database would test nothing that matters here.

Start one with ``make db-up`` and point DATABASE_URL at it. Without a
reachable database every test in this module skips rather than fails, so the
unit suite stays runnable on a machine with no Docker.

The fixture is Actium (31 BC), the project's canonical attribution case:
Agrippa commanded the fleet while Octavian, the sovereign, was present but
not directing the battle. If the schema cannot express that distinction then
the Octavian/Agrippa problem the methodology is built around has nowhere to
live.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import Engine, text

from pipeline.db import (
    DatabaseConfigError,
    apply_schema,
    database_url,
    get_connection,
    get_engine,
    schema_is_applied,
)

# ─── Skip cleanly when no database is reachable ──────────────────────────────


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


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def engine() -> Iterator[Engine]:
    """Provide an engine against a freshly applied schema.

    The schema is rebuilt from scratch so the tests never depend on state left
    by a previous run or by a partially crawled development database.
    """
    eng = get_engine()

    # Refuse to wipe anything that is not an obvious test or dev database.
    # DROP SCHEMA CASCADE is unrecoverable, and an operator with a populated
    # database in DATABASE_URL should not lose it to a test run.
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
def conn(engine: Engine) -> Iterator[Any]:
    """Provide a connection whose writes are rolled back after each test."""
    connection = engine.connect()
    transaction = connection.begin()
    try:
        yield connection
    finally:
        transaction.rollback()
        connection.close()


# ─── Schema application ──────────────────────────────────────────────────────


def test_schema_is_applied(engine: Engine) -> None:
    with engine.connect() as c:
        assert schema_is_applied(c) is True


def test_apply_schema_is_idempotent(engine: Engine) -> None:
    """A second application must no-op rather than half-migrate."""
    assert apply_schema(engine) is False


def test_expected_tables_exist(engine: Engine) -> None:
    expected = {
        "sources",
        "generals",
        "general_aliases",
        "wars",
        "campaigns",
        "battles",
        "battle_sides",
        "battle_commanders",
        "troop_reports",
        "casualty_reports",
        "missing_data_log",
        "crawl_log",
        "llm_calls",
        "model_runs",
        "general_skill_estimates",
        "battle_war_details",
    }
    with engine.connect() as c:
        found = {
            row[0]
            for row in c.execute(
                text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
                )
            )
        }
    assert expected <= found, f"missing tables: {sorted(expected - found)}"


def test_expected_views_exist(engine: Engine) -> None:
    expected = {
        "v_battle_overview",
        "v_troop_source_agreement",
        "v_attribution_review_queue",
    }
    with engine.connect() as c:
        found = {
            row[0]
            for row in c.execute(
                text(
                    "SELECT table_name FROM information_schema.views "
                    "WHERE table_schema = 'public'"
                )
            )
        }
    assert expected <= found, f"missing views: {sorted(expected - found)}"


# ─── The Actium fixture ──────────────────────────────────────────────────────


def _insert_actium(c: Any) -> dict[str, int]:
    """Insert Actium with two sides and three commanders.

    Agrippa commanded tactically for the Octavian side; Octavian was present
    as sovereign; Antony commanded the opposing side. Attribution weights on
    each side sum to 1.0, as agents/classify.yaml requires.

    Args:
        c: An open connection inside a transaction.

    Returns:
        The generated ids, keyed by name.
    """
    source_id = c.execute(
        text(
            "INSERT INTO sources (source_type, title, author, year, credibility) "
            "VALUES (:t, :title, :author, :year, :cred) RETURNING source_id"
        ),
        {
            "t": "academic_book",
            "title": "The Battle of Actium",
            "author": "Synthetic Author",
            "year": 1998,
            "cred": 0.8,
        },
    ).scalar_one()

    war_id = c.execute(
        text(
            "INSERT INTO wars (name, start_year, end_year, region) "
            "VALUES (:n, :s, :e, :r) RETURNING war_id"
        ),
        {"n": "Final War of the Roman Republic", "s": -32, "e": -30, "r": "Mediterranean"},
    ).scalar_one()

    battle_id = c.execute(
        text(
            "INSERT INTO battles (name, wikidata_id, war_id, date_start, date_precision, "
            "battle_type, latitude, longitude, location_name, n_sources) "
            "VALUES (:n, :wd, :war, DATE '0031-09-02 BC', :prec, :bt, :lat, :lon, :loc, :ns) "
            "RETURNING battle_id"
        ),
        {
            "n": "Battle of Actium",
            "wd": "Q184408",
            "war": war_id,
            # 2 September 31 BC, written as a SQL literal rather than bound as
            # a parameter. Python's datetime.date has MINYEAR == 1, so BC dates
            # cannot be represented in Python at all and cannot be passed as a
            # bind parameter. See test_bc_dates_cannot_round_trip_through_python.
            "prec": "day",
            "bt": "naval",
            "lat": 38.9333,
            "lon": 20.7667,
            "loc": "Actium, Ionian Sea",
            "ns": 1,
        },
    ).scalar_one()

    side_octavian = c.execute(
        text(
            "INSERT INTO battle_sides (battle_id, side_label, polity, outcome, "
            "est_troops_total, est_troops_total_lo, est_troops_total_hi) "
            "VALUES (:b, :l, :p, :o, :t, :lo, :hi) RETURNING side_id"
        ),
        {
            "b": battle_id,
            "l": "Octavian's fleet",
            "p": "Roman Republic (Octavian)",
            "o": "decisive_victory",
            "t": 80000.0,
            "lo": 60000.0,
            "hi": 100000.0,
        },
    ).scalar_one()

    side_antony = c.execute(
        text(
            "INSERT INTO battle_sides (battle_id, side_label, polity, outcome, "
            "est_troops_total, est_troops_total_lo, est_troops_total_hi) "
            "VALUES (:b, :l, :p, :o, :t, :lo, :hi) RETURNING side_id"
        ),
        {
            "b": battle_id,
            "l": "Antony and Cleopatra's fleet",
            "p": "Roman Republic (Antony) / Ptolemaic Egypt",
            "o": "decisive_defeat",
            "t": 70000.0,
            "lo": 50000.0,
            "hi": 90000.0,
        },
    ).scalar_one()

    def add_general(name: str, wikidata_id: str) -> int:
        result: int = c.execute(
            text(
                "INSERT INTO generals (canonical_name, wikidata_id, era, nationality) "
                "VALUES (:n, :wd, :era, :nat) RETURNING general_id"
            ),
            {"n": name, "wd": wikidata_id, "era": "Ancient Rome", "nat": "Roman"},
        ).scalar_one()
        return result

    agrippa_id = add_general("Marcus Vipsanius Agrippa", "Q162634")
    octavian_id = add_general("Augustus", "Q1405")
    antony_id = add_general("Mark Antony", "Q51673")

    def add_commander(
        side_id: int,
        general_id: int,
        role: str,
        rank: int,
        weight: float,
        evidence: str,
        reports_to: int | None = None,
    ) -> int:
        result: int = c.execute(
            text(
                "INSERT INTO battle_commanders (battle_id, side_id, general_id, "
                "command_role, hierarchy_rank, attribution_weight, attribution_method, "
                "reports_to_bc_id, source_id, extraction_method, confidence, role_evidence) "
                "VALUES (:b, :s, :g, :role, :rank, :w, :method, :rt, :src, :em, :conf, :ev) "
                "RETURNING bc_id"
            ),
            {
                "b": battle_id,
                "s": side_id,
                "g": general_id,
                "role": role,
                "rank": rank,
                "w": weight,
                "method": "role_heuristic",
                "rt": reports_to,
                "src": source_id,
                "em": "manual",
                "conf": 0.9,
                "ev": evidence,
            },
        ).scalar_one()
        return result

    # Agrippa outranks Octavian tactically despite Octavian being the
    # sovereign: rank 0 is the top *tactical* commander, not the senior person.
    agrippa_bc = add_commander(
        side_octavian,
        agrippa_id,
        "field_commander",
        0,
        0.85,
        "Agrippa commanded the fleet and directed the battle.",
    )
    add_commander(
        side_octavian,
        octavian_id,
        "sovereign",
        1,
        0.15,
        "Octavian was present with the fleet but left the fighting to Agrippa.",
        reports_to=agrippa_bc,
    )
    add_commander(
        side_antony,
        antony_id,
        "field_commander",
        0,
        1.0,
        "Antony commanded his fleet in person.",
    )

    return {
        "battle_id": battle_id,
        "side_octavian": side_octavian,
        "side_antony": side_antony,
        "agrippa": agrippa_id,
        "octavian": octavian_id,
        "antony": antony_id,
        "agrippa_bc": agrippa_bc,
    }


def test_insert_actium_and_query_overview(conn: Any) -> None:
    """End-to-end shape check: insert the fixture, read it back via the view."""
    ids = _insert_actium(conn)

    rows = (
        conn.execute(
            text(
                "SELECT general_name, command_role, hierarchy_rank, attribution_weight, "
                "side_label, outcome, battle_name, battle_type "
                "FROM v_battle_overview WHERE battle_id = :b "
                "ORDER BY side_label, hierarchy_rank"
            ),
            {"b": ids["battle_id"]},
        )
        .mappings()
        .all()
    )

    assert len(rows) == 3, "three commanders were inserted"

    by_name = {r["general_name"]: r for r in rows}
    assert set(by_name) == {"Marcus Vipsanius Agrippa", "Augustus", "Mark Antony"}

    agrippa = by_name["Marcus Vipsanius Agrippa"]
    octavian = by_name["Augustus"]
    antony = by_name["Mark Antony"]

    # The attribution case the project exists to get right.
    assert agrippa["command_role"] == "field_commander"
    assert octavian["command_role"] == "sovereign"
    assert agrippa["hierarchy_rank"] == 0
    assert octavian["hierarchy_rank"] == 1
    assert agrippa["attribution_weight"] > octavian["attribution_weight"]
    assert agrippa["attribution_weight"] > 0.7, (
        "agents/evaluate.yaml expects Agrippa to carry most of the credit"
    )

    assert antony["command_role"] == "field_commander"
    assert antony["outcome"] == "decisive_defeat"
    assert agrippa["outcome"] == "decisive_victory"
    assert agrippa["battle_type"] == "naval"
    assert agrippa["battle_name"] == "Battle of Actium"


def test_attribution_weights_sum_to_one_per_side(conn: Any) -> None:
    """The invariant agents/classify.yaml's attribution_weights_sum gate checks."""
    ids = _insert_actium(conn)

    rows = (
        conn.execute(
            text(
                "SELECT side_id, SUM(attribution_weight) AS total "
                "FROM battle_commanders WHERE battle_id = :b GROUP BY side_id"
            ),
            {"b": ids["battle_id"]},
        )
        .mappings()
        .all()
    )

    assert len(rows) == 2
    for row in rows:
        assert abs(float(row["total"]) - 1.0) < 0.15, (
            f"side {row['side_id']} weights sum to {row['total']}"
        )


def test_bc_date_is_stored_as_bc(conn: Any) -> None:
    """31 BC must be stored as BC, not silently coerced to AD.

    This is where a naive ISO parser produces 31 AD, putting Actium in the
    wrong era and pulling the wrong imputation prior from
    config/imputation_priors.yaml.

    The date is read back through to_char rather than as a Python object; see
    test_bc_dates_cannot_round_trip_through_python for why.
    """
    ids = _insert_actium(conn)

    rendered = conn.execute(
        text("SELECT to_char(date_start, 'YYYY-MM-DD BC') FROM battles WHERE battle_id = :b"),
        {"b": ids["battle_id"]},
    ).scalar_one()
    assert rendered.endswith("BC"), f"expected a BC date, got {rendered}"
    assert rendered.startswith("0031-09-02"), rendered

    # EXTRACT returns a numeric, so it survives where the date adapter does not.
    year = conn.execute(
        text("SELECT EXTRACT(YEAR FROM date_start) FROM battles WHERE battle_id = :b"),
        {"b": ids["battle_id"]},
    ).scalar_one()
    assert int(year) == -30, "31 BC is astronomical year -30; there is no year zero"


def test_bc_dates_cannot_round_trip_through_python(conn: Any) -> None:
    """Record the constraint that BC dates break Python's date type.

    datetime.date has MINYEAR == 1, so no BC date can be constructed, bound as
    a parameter, or decoded from a result. Postgres stores them happily, which
    makes this easy to miss until a stage touches an ancient battle.

    Any stage reading battles.date_start must therefore either filter to AD
    rows or project through to_char/EXTRACT. This test fails if a future
    psycopg gains BC support, at which point the workarounds can be removed.
    """
    import datetime

    with pytest.raises(ValueError):
        datetime.date(-30, 9, 2)

    ids = _insert_actium(conn)

    # Reading the raw column back into Python is what breaks.
    with pytest.raises(Exception) as exc:
        conn.execute(
            text("SELECT date_start FROM battles WHERE battle_id = :b"),
            {"b": ids["battle_id"]},
        ).scalar_one()

    assert "date" in str(exc.value).lower() or "year" in str(exc.value).lower(), (
        f"expected a date decoding failure, got: {exc.value}"
    )


def test_hierarchy_chain_is_navigable(conn: Any) -> None:
    """reports_to_bc_id must resolve to the commander it names."""
    ids = _insert_actium(conn)

    row = (
        conn.execute(
            text(
                "SELECT sub.command_role AS sub_role, sup.command_role AS sup_role "
                "FROM battle_commanders sub "
                "JOIN battle_commanders sup ON sup.bc_id = sub.reports_to_bc_id "
                "WHERE sub.battle_id = :b AND sub.reports_to_bc_id IS NOT NULL"
            ),
            {"b": ids["battle_id"]},
        )
        .mappings()
        .one()
    )

    assert row["sub_role"] == "sovereign"
    assert row["sup_role"] == "field_commander"


def test_one_top_commander_per_side(conn: Any) -> None:
    """Each side must have exactly one rank-0 commander."""
    ids = _insert_actium(conn)

    rows = (
        conn.execute(
            text(
                "SELECT side_id, COUNT(*) AS n FROM battle_commanders "
                "WHERE battle_id = :b AND hierarchy_rank = 0 GROUP BY side_id"
            ),
            {"b": ids["battle_id"]},
        )
        .mappings()
        .all()
    )

    assert len(rows) == 2
    assert all(row["n"] == 1 for row in rows)


# ─── Constraint enforcement ──────────────────────────────────────────────────


def test_attribution_weight_bounds_enforced(conn: Any) -> None:
    """The CHECK constraint must reject a weight outside 0..1."""
    ids = _insert_actium(conn)

    with pytest.raises(Exception) as exc:
        conn.execute(
            text(
                "INSERT INTO battle_commanders (battle_id, side_id, general_id, "
                "command_role, attribution_weight) VALUES (:b, :s, :g, :r, :w)"
            ),
            {
                "b": ids["battle_id"],
                "s": ids["side_antony"],
                "g": ids["agrippa"],
                "r": "subordinate",
                "w": 1.5,
            },
        )
    assert "attribution_weight" in str(exc.value).lower()


def test_duplicate_commander_rejected(conn: Any) -> None:
    """UNIQUE (battle_id, side_id, general_id) must hold."""
    ids = _insert_actium(conn)

    with pytest.raises(Exception) as exc:
        conn.execute(
            text(
                "INSERT INTO battle_commanders (battle_id, side_id, general_id, command_role) "
                "VALUES (:b, :s, :g, :r)"
            ),
            {
                "b": ids["battle_id"],
                "s": ids["side_octavian"],
                "g": ids["agrippa"],
                "r": "subordinate",
            },
        )
    message = str(exc.value).lower()
    assert "unique" in message or "duplicate" in message


def test_wikidata_id_is_unique(conn: Any) -> None:
    """Two generals must not share a Wikidata id.

    This is what resolve.yaml's no_duplicate_generals gate depends on.
    """
    _insert_actium(conn)

    with pytest.raises(Exception) as exc:
        conn.execute(
            text("INSERT INTO generals (canonical_name, wikidata_id) VALUES (:n, :wd)"),
            {"n": "Agrippa (duplicate)", "wd": "Q162634"},
        )
    message = str(exc.value).lower()
    assert "unique" in message or "duplicate" in message


# ─── Connection handling ─────────────────────────────────────────────────────


def test_get_connection_commits_on_success(engine: Engine) -> None:
    marker = "Connection Commit Probe"
    with get_connection(engine=engine) as c:
        c.execute(
            text("INSERT INTO generals (canonical_name, era) VALUES (:n, :e)"),
            {"n": marker, "e": "test"},
        )

    try:
        with engine.connect() as c:
            found = c.execute(
                text("SELECT COUNT(*) FROM generals WHERE canonical_name = :n"),
                {"n": marker},
            ).scalar()
        assert found == 1
    finally:
        with engine.connect() as c:
            c.execute(text("DELETE FROM generals WHERE canonical_name = :n"), {"n": marker})
            c.commit()


def test_get_connection_rolls_back_on_error(engine: Engine) -> None:
    marker = "Connection Rollback Probe"

    with pytest.raises(RuntimeError), get_connection(engine=engine) as c:
        c.execute(
            text("INSERT INTO generals (canonical_name, era) VALUES (:n, :e)"),
            {"n": marker, "e": "test"},
        )
        raise RuntimeError("forced failure inside the context")

    with engine.connect() as c:
        found = c.execute(
            text("SELECT COUNT(*) FROM generals WHERE canonical_name = :n"),
            {"n": marker},
        ).scalar()
    assert found == 0, "the failed transaction must leave nothing behind"


def test_database_url_rejects_missing_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(DatabaseConfigError) as exc:
        database_url()
    assert "DATABASE_URL" in str(exc.value)


def test_password_is_not_logged() -> None:
    """A URL rendered for logs must not carry the password."""
    from pipeline.db import _redact

    redacted = _redact("postgresql+psycopg://someuser:hunter2@localhost:5432/general_war")
    assert "hunter2" not in redacted
    assert "someuser" in redacted


# ─── Quality gate against a real database ────────────────────────────────────


def test_quality_runner_executes_real_sql(conn: Any) -> None:
    """The gate must run spec SQL against Postgres and compare correctly."""
    from pipeline.quality import QualityRunner

    ids = _insert_actium(conn)
    assert ids["battle_id"] is not None

    runner = QualityRunner(conn)

    passing = runner.run_one(
        {
            "name": "generals_exist",
            "check": "SELECT COUNT(*) FROM generals",
            "threshold": ">= 3",
            "severity": "error",
        }
    )
    assert passing.passed is True
    assert passing.actual_value >= 3

    failing = runner.run_one(
        {
            "name": "impossible",
            "check": "SELECT COUNT(*) FROM generals",
            "threshold": ">= 100000",
            "severity": "error",
        }
    )
    assert failing.passed is False


def test_classify_missing_data_gate_sql_is_valid(conn: Any) -> None:
    """agents/classify.yaml's ERROR gate must be executable SQL, not just parseable.

    A gate that fails with a syntax error at run time is no gate at all, and
    the failure would look like a database problem rather than a spec bug.
    """
    from pipeline.config import load_agent_spec
    from pipeline.quality import QualityRunner

    _insert_actium(conn)

    spec = load_agent_spec("classify")
    check = next(c for c in spec["quality_checks"] if c["name"] == "missing_data_all_logged")

    result = QualityRunner(conn).run_one(check)

    assert "SQL execution failed" not in result.message, result.message
    assert result.actual_value is not None
