"""
Integration tests for the extract stage's database writes.

These need a real PostgreSQL instance: the stage writes enum columns, a text
array, and BC dates, none of which a substituted database reproduces. Start
one with ``make db-up`` and point DATABASE_URL at it; without a reachable
database every test here skips rather than fails, so the unit suite stays
runnable on a machine with no Docker.

The fixture is Actium, the project's canonical case. It is also the awkward
one for storage: its date is 31 BC, which ``datetime.date`` cannot hold, and
its troop figures carry three different scopes, which is exactly the
distinction the force-ratio covariate depends on surviving the round trip.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine, text

from pipeline.db import (
    DatabaseConfigError,
    apply_schema,
    database_url,
    get_engine,
)
from pipeline.extractors import (
    ExtractWriteCounts,
    map_entity,
    merge_battle,
    parse_infobox,
    write_battle,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "extract"


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
def conn(engine: Engine) -> Iterator[Any]:
    """Provide a connection whose writes are rolled back after each test."""
    connection = engine.connect()
    transaction = connection.begin()
    try:
        yield connection
    finally:
        transaction.rollback()
        connection.close()


def _actium() -> Any:
    """Build the merged Actium record from the unit fixtures.

    Returns:
        The merged battle, from the naval infobox plus the Wikidata entity.
    """
    import json

    infobox = parse_infobox(
        (FIXTURES / "actium_naval.wikitext").read_text(encoding="utf-8"),
        source_ref="https://en.wikipedia.org/wiki/Battle_of_Actium",
        source_title="Battle of Actium",
    )
    wikidata = map_entity(
        json.loads((FIXTURES / "actium_wikidata.json").read_text(encoding="utf-8")),
        source_ref="https://www.wikidata.org/wiki/Q193320",
    )
    sources = [s for s in (infobox, wikidata) if s is not None]
    return merge_battle("actium", sources, fallback_name="Battle of Actium")


def test_a_merged_battle_writes_every_table_the_stage_owns(conn: Any) -> None:
    counts = ExtractWriteCounts()
    battle_id = write_battle(conn, _actium(), counts)

    assert battle_id > 0
    assert counts.sides == 2
    assert counts.troop_reports > 0

    stored = conn.execute(
        text(
            "SELECT name, wikidata_id, battle_type::text, date_start::text "
            "FROM battles WHERE battle_id = :id"
        ),
        {"id": battle_id},
    ).fetchone()

    assert stored is not None
    assert stored[1] == "Q193320"
    assert stored[2] == "naval"


def test_a_bc_date_survives_the_round_trip(conn: Any) -> None:
    """31 BC has to be storable; a quarter of the corpus predates year one."""
    battle_id = write_battle(conn, _actium())

    stored = conn.execute(
        text("SELECT date_start::text, date_precision FROM battles WHERE battle_id = :id"),
        {"id": battle_id},
    ).fetchone()

    assert stored is not None
    assert stored[0].endswith("BC")
    assert "0031" in stored[0]
    assert stored[1] == "day"


def test_troop_report_scope_and_bounds_reach_the_database(conn: Any) -> None:
    write_battle(conn, _actium())

    rows = conn.execute(
        text(
            "SELECT tr.scope, tr.branch::text, tr.reported_value "
            "FROM troop_reports tr JOIN battle_sides bs ON bs.side_id = tr.side_id "
            "JOIN battles b ON b.battle_id = bs.battle_id WHERE b.name = :name"
        ),
        {"name": "Battle of Actium"},
    ).fetchall()

    scopes = {row[0] for row in rows}
    # The naval infobox reports on-paper strength and engaged strength in one
    # field. Flattening them to 'engaged' would inflate the force ratio.
    assert "on_paper" in scopes
    assert "engaged" in scopes
    assert "naval" in {row[1] for row in rows}


def test_rerunning_the_write_does_not_duplicate_reports(conn: Any) -> None:
    """The stage must be idempotent; a resumed run re-writes some battles."""
    battle = _actium()
    write_battle(conn, battle)
    first = conn.execute(text("SELECT COUNT(*) FROM troop_reports")).scalar()

    write_battle(conn, battle)
    second = conn.execute(text("SELECT COUNT(*) FROM troop_reports")).scalar()

    assert first == second
    assert conn.execute(text("SELECT COUNT(*) FROM battles")).scalar() == 1


def test_missing_fields_are_logged_to_the_database(conn: Any) -> None:
    battle_id = write_battle(conn, _actium())

    rows = conn.execute(
        text("SELECT field_name FROM missing_data_log WHERE battle_id = :id"),
        {"id": battle_id},
    ).fetchall()

    # Actium is a complete record apart from terrain: both sides carry troop
    # reports, casualties, commanders and an outcome. Terrain is a covariate in
    # the model's linear predictor, so its absence is logged rather than left
    # to be silently read as "no terrain effect" downstream.
    #
    # Weather is absent here too and is deliberately *not* logged: no stage
    # reads it, and a row per ancient battle for a field nothing consumes would
    # bury the missingness that matters.
    logged = {row[0] for row in rows}
    assert "terrain" in logged, f"terrain absence must be logged, got {logged}"
    assert "weather" not in logged, "weather is descriptive, not a model covariate"


def test_extraction_coverage_gate_sql_is_valid(conn: Any) -> None:
    """agents/extract.yaml's gates must be executable SQL, not just parseable."""
    from pipeline.config import load_agent_spec
    from pipeline.quality import QualityRunner

    write_battle(conn, _actium())

    spec = load_agent_spec("extract")
    check = next(c for c in spec["quality_checks"] if c["name"] == "extraction_coverage")

    result = QualityRunner(conn).run_one(check)

    assert "SQL execution failed" not in result.message, result.message
    assert result.actual_value is not None


# ─── One bad battle must not take the others down with it ───────────────────


def _record(slug: str, *, battle_type: str | None = None, date_start: str | None = None) -> Any:
    """Build a minimal merged battle with two sides."""
    from pipeline.extractors import BattleExtraction, BattleFacts, SideExtraction

    return BattleExtraction(
        slug=slug,
        name=f"Battle of {slug}",
        facts=BattleFacts(
            name=f"Battle of {slug}",
            date_start=date_start,
            battle_type=battle_type,
        ),
        sides=[SideExtraction(label="A"), SideExtraction(label="B")],
        n_sources=1,
    )


def test_a_failing_battle_is_rolled_back_alone_and_the_rest_commit(engine: Engine) -> None:
    from pipeline.extractors import write_battle_isolated

    records = [
        _record("Alpha", date_start="0216-08-02 BC"),
        # An invalid enum value fails inside Postgres, which aborts the
        # transaction unless the battle has a savepoint of its own.
        _record("Broken", battle_type="not_a_battle_type"),
        _record("Gamma", date_start="Spring 208 BC"),
        _record("Delta", date_start="1809-05-21"),
    ]
    counts = ExtractWriteCounts()
    cache: dict[tuple[str, str], int] = {}
    writer = engine.connect()
    try:
        for record in records:
            write_battle_isolated(writer, record, counts, cache)
        writer.commit()

        reader = engine.connect()
        try:
            rows = reader.execute(
                text(
                    "SELECT name, date_start::text, date_precision, needs_review, review_notes, "
                    "year_astronomical FROM battles ORDER BY name"
                )
            ).fetchall()
        finally:
            reader.close()
    finally:
        writer.rollback()
        writer.execute(text("TRUNCATE battles CASCADE"))
        writer.commit()
        writer.close()

    by_name = {row[0]: row for row in rows}
    assert set(by_name) == {
        "Battle of Alpha",
        "Battle of Broken",
        "Battle of Gamma",
        "Battle of Delta",
    }
    assert counts.failed_battles == 1
    assert counts.battles == 3

    assert by_name["Battle of Alpha"][1] == "0216-08-02 BC"
    assert by_name["Battle of Alpha"][5] == -215
    # Raw infobox text that reaches the writer is parsed, not cast blindly.
    assert tuple(by_name["Battle of Gamma"][1:3]) == ("0208-01-01 BC", "year")
    assert by_name["Battle of Delta"][1] == "1809-05-21"

    broken = by_name["Battle of Broken"]
    assert broken[3] is True
    assert "write_failed" in broken[4]


def test_raw_unparseable_date_text_is_written_as_null_not_a_database_error(conn: Any) -> None:
    from pipeline.extractors import write_battle_isolated

    counts = ExtractWriteCounts()
    battle_id = write_battle_isolated(
        conn, _record("Vague", date_start="in the reign of Nero"), counts, {}
    )

    row = conn.execute(
        text("SELECT date_start, date_precision FROM battles WHERE battle_id = :id"),
        {"id": battle_id},
    ).fetchone()
    assert counts.failed_battles == 0
    assert row is not None
    assert row[0] is None


def test_an_unparsed_infobox_date_flags_the_battle_and_logs_it_missing(conn: Any) -> None:
    source = parse_infobox(
        "{{Infobox military conflict\n| conflict = Battle of Fog\n| date = sometime\n"
        "| place = Somewhere\n| result = Roman victory\n"
        "| combatant1 = Rome\n| combatant2 = Carthage\n}}\n",
        source_ref="x",
        source_title="Battle of Fog",
    )
    assert source is not None
    battle = merge_battle("fog", [source], fallback_name="Battle of Fog")

    battle_id = write_battle(conn, battle)

    row = conn.execute(
        text("SELECT date_start, needs_review, review_notes FROM battles WHERE battle_id = :id"),
        {"id": battle_id},
    ).fetchone()
    note = conn.execute(
        text(
            "SELECT notes FROM missing_data_log WHERE battle_id = :id AND field_name = 'date_start'"
        ),
        {"id": battle_id},
    ).scalar_one()

    assert row is not None
    assert row[0] is None
    assert row[1] is True
    assert "unparsed_date" in row[2]
    assert "sometime" in note
