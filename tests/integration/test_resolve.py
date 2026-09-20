"""
Integration tests for the resolve stage's database writes.

These need a real PostgreSQL instance. Nothing here is reproducible against a
substituted database: the writes go through an enum cast, an ``int4range``, an
``ON CONFLICT`` clause that reads the row it is replacing, and a BC date that
``datetime.date`` cannot hold. Without a reachable database every test skips
rather than fails, so the unit suite stays runnable anywhere.

Four things are worth the cost of a live database, because each fails silently
in a way a fixture would not show:

- **A re-run converges.** Running resolve twice must not double the commander
  rows, nor duplicate the aliases.
- **A re-run does not undo classify.** classify refines ``command_role`` on
  these rows; the upsert must leave a role that has already moved off
  ``unknown``.
- **Unresolved mentions leave a trace.** They are what gives the
  ``resolution_rate`` gate a denominator, and without them the gate divides
  written rows by written rows.
- **The gates run.** Each of the stage's five quality checks has to execute
  against a schema that exists, which is the failure mode §4.1 and §11.4 of
  handover.md each found the hard way.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml
from sqlalchemy import Connection, Engine, text
from sqlalchemy.exc import IntegrityError

from pipeline.db import DatabaseConfigError, apply_schema, database_url, get_engine
from pipeline.quality import QualityRunner
from pipeline.resolvers import (
    Identity,
    Mention,
    MentionGroup,
    ResolveCounts,
    find_duplicate_battles,
    load_battle_index,
    load_side_index,
    log_unresolved,
    write_identity,
)
from pipeline.resolvers.records import BattleContext
from pipeline.resolvers.store import UNRESOLVED_FIELD

AGENTS = Path(__file__).resolve().parents[2] / "agents"

BATTLE_NAME = "Battle of Actium"
SIDE_LABEL = "Octavian"
OTHER_SIDE = "Antony and Cleopatra"
# 31 BC. Astronomical -30: there is a year zero, so the historical year is one
# further from the epoch than the astronomical one (handover §4.5).
BATTLE_DATE = "0031-09-02 BC"
BATTLE_YEAR = -30


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
    """Give each test an empty corpus, rolled back afterwards."""
    connection = engine.connect()
    transaction = connection.begin()
    try:
        yield connection
    finally:
        transaction.rollback()
        connection.close()


# ─── Fixture corpus ──────────────────────────────────────────────────────────


def _seed_battle(conn: Connection, name: str = BATTLE_NAME, date: str = BATTLE_DATE) -> int:
    """Insert one battle with two sides, as the extract stage would.

    Args:
        conn: An open connection.
        name: The battle's name.
        date: A Postgres date literal, BC suffix included.

    Returns:
        The battle id.
    """
    battle_id = int(
        conn.execute(
            text(
                "INSERT INTO battles (name, date_start, date_precision) "
                "VALUES (:name, CAST(:date AS DATE), 'day') RETURNING battle_id"
            ),
            {"name": name, "date": date},
        ).scalar_one()
    )
    for label in (SIDE_LABEL, OTHER_SIDE):
        conn.execute(
            text(
                "INSERT INTO battle_sides (battle_id, side_label, polity) "
                "VALUES (:battle_id, :label, 'Roman Republic')"
            ),
            {"battle_id": battle_id, "label": label},
        )
    return battle_id


def _mention(name: str, *, side: str = SIDE_LABEL, evidence: str = "") -> Mention:
    """Build a mention of one commander at the fixture battle."""
    return Mention(
        battle_slug="Battle_of_Actium",
        battle_name=BATTLE_NAME,
        side_label=side,
        name=name,
        apparent_role="field_commander",
        role_evidence=evidence,
        context=BattleContext(
            slug="Battle_of_Actium",
            name=BATTLE_NAME,
            year=BATTLE_YEAR,
            date_text=BATTLE_DATE,
        ),
    )


def _identity(
    canonical: str,
    *,
    qid: str | None,
    mentions: list[Mention],
    aliases: list[str] | None = None,
    confidence: float = 0.95,
) -> Identity:
    """Build a resolved identity around one group of mentions."""
    group = MentionGroup(
        key=canonical.lower(),
        keys=(canonical.lower(),),
        display_name=canonical,
        surface_forms=sorted({m.name for m in mentions}),
        mentions=mentions,
        confidence=confidence,
        method="exact_wikidata",
    )
    return Identity(
        key=qid or f"new:{canonical.lower()}",
        canonical_name=canonical,
        qid=qid,
        confidence=confidence,
        method="exact_wikidata",
        aliases=aliases or [],
        groups=[group],
    )


def _write(conn: Connection, *identities: Identity) -> ResolveCounts:
    """Write identities through the stage's writer."""
    counts = ResolveCounts()
    battles = load_battle_index(conn)
    sides = load_side_index(conn)
    cache: dict[tuple[str, str], int] = {}
    for identity in identities:
        write_identity(conn, identity, battles, sides, counts, cache)
    return counts


def _scalar(conn: Connection, sql: str, **params: Any) -> Any:
    """Run a one-value query."""
    return conn.execute(text(sql), params).scalar()


# ─── Writing ─────────────────────────────────────────────────────────────────


def test_a_qid_is_never_published_as_a_generals_name(conn: Connection) -> None:
    # The label service answers with the bare Q-id when it has no label in the
    # languages asked for, and that string used to flow through
    # Decision.canonical_name into generals and into the primary alias row.
    # The candidate lookup fixes it upstream; this is the last boundary before
    # it becomes published data, so it is checked again rather than trusted.
    _seed_battle(conn)
    _write(
        conn,
        _identity("Q83235", qid="Q83235", mentions=[_mention("Horatio Nelson")]),
    )

    name = _scalar(
        conn,
        "SELECT canonical_name FROM generals WHERE wikidata_id = :qid",
        qid="Q83235",
    )
    assert name == "Horatio Nelson"

    qid_shaped_primaries = _scalar(
        conn,
        "SELECT COUNT(*) FROM general_aliases a JOIN generals g USING (general_id) "
        "WHERE g.wikidata_id = :qid AND a.is_primary AND a.alias_name ~ '^Q[0-9]+$'",
        qid="Q83235",
    )
    assert qid_shaped_primaries == 0


def test_a_renamed_general_keeps_exactly_one_primary_alias(conn: Connection) -> None:
    # generals.canonical_name is rewritten unconditionally on a re-run, and the
    # mul fix renames anyone previously stored under a bare Q-id. _write_aliases
    # only inserts, and general_aliases has no constraint on is_primary, so the
    # old primary row would otherwise survive beside the new one.
    _seed_battle(conn)
    mentions = [_mention("Horatio Nelson")]
    _write(conn, _identity("Nelson", qid="Q83235", mentions=mentions))
    _write(conn, _identity("Horatio Nelson", qid="Q83235", mentions=mentions))

    primaries = _scalar(
        conn,
        "SELECT COUNT(*) FROM general_aliases a JOIN generals g USING (general_id) "
        "WHERE g.wikidata_id = :qid AND a.is_primary",
        qid="Q83235",
    )
    assert primaries == 1

    primary_name = _scalar(
        conn,
        "SELECT a.alias_name FROM general_aliases a JOIN generals g USING (general_id) "
        "WHERE g.wikidata_id = :qid AND a.is_primary",
        qid="Q83235",
    )
    assert primary_name == "Horatio Nelson"


def test_a_resolved_commander_reaches_battle_commanders(conn: Connection) -> None:
    _seed_battle(conn)
    counts = _write(
        conn,
        _identity(
            "Marcus Vipsanius Agrippa",
            qid="Q48174",
            mentions=[_mention("Agrippa", evidence="Agrippa commanded the fleet")],
            aliases=["Agrippa"],
        ),
    )

    assert counts.generals_written == 1
    assert counts.commanders_written == 1
    assert _scalar(conn, "SELECT COUNT(*) FROM battle_commanders") == 1
    assert (
        _scalar(
            conn,
            "SELECT g.wikidata_id FROM generals g "
            "JOIN battle_commanders bc USING (general_id) LIMIT 1",
        )
        == "Q48174"
    )
    assert (
        _scalar(conn, "SELECT role_evidence FROM battle_commanders LIMIT 1")
        == "Agrippa commanded the fleet"
    )


def test_years_active_comes_back_as_the_astronomical_year(conn: Connection) -> None:
    # The battle is 31 BC, which datetime.date cannot hold. The range is
    # written from the astronomical year, so it reads back as -30 and agrees
    # with battles.year_astronomical.
    _seed_battle(conn)
    _write(conn, _identity("Agrippa", qid="Q48174", mentions=[_mention("Agrippa")]))

    assert _scalar(conn, "SELECT lower(years_active) FROM generals") == BATTLE_YEAR
    assert _scalar(conn, "SELECT year_astronomical FROM battles") == BATTLE_YEAR


def test_running_twice_converges_rather_than_duplicating(conn: Connection) -> None:
    _seed_battle(conn)
    identity = _identity(
        "Marcus Vipsanius Agrippa",
        qid="Q48174",
        mentions=[_mention("Agrippa")],
        aliases=["Agrippa"],
    )
    _write(conn, identity)
    before_aliases = _scalar(conn, "SELECT COUNT(*) FROM general_aliases")

    _write(conn, identity)

    assert _scalar(conn, "SELECT COUNT(*) FROM generals") == 1
    assert _scalar(conn, "SELECT COUNT(*) FROM battle_commanders") == 1
    assert _scalar(conn, "SELECT COUNT(*) FROM general_aliases") == before_aliases


def test_a_rerun_does_not_undo_the_classify_stage(conn: Connection) -> None:
    # classify refines command_role on these rows. Deleting and reinserting --
    # which is what the extract writer does for troop reports -- would discard
    # that silently, so the upsert must leave a decided role alone.
    _seed_battle(conn)
    identity = _identity("Agrippa", qid="Q48174", mentions=[_mention("Agrippa")])
    _write(conn, identity)

    conn.execute(
        text("UPDATE battle_commanders SET command_role = 'theatre_commander', hierarchy_rank = 1")
    )
    _write(conn, identity)

    assert _scalar(conn, "SELECT command_role FROM battle_commanders") == "theatre_commander"
    assert _scalar(conn, "SELECT hierarchy_rank FROM battle_commanders") == 1


def test_two_sources_naming_one_commander_make_one_row(conn: Connection) -> None:
    # The UNIQUE (battle_id, side_id, general_id) says one command per side,
    # and the mention with the fullest evidence is the one classify will read.
    _seed_battle(conn)
    counts = _write(
        conn,
        _identity(
            "Agrippa",
            qid="Q48174",
            mentions=[
                _mention("Agrippa", evidence="short"),
                _mention("Agrippa", evidence="a much longer account of the command"),
            ],
        ),
    )

    assert counts.commanders_written == 1
    assert (
        _scalar(conn, "SELECT role_evidence FROM battle_commanders")
        == "a much longer account of the command"
    )


def test_a_name_collision_never_hijacks_a_wikidata_backed_row(conn: Connection) -> None:
    # A corpus-local identity that happens to share a name with a resolved
    # entity must get its own row, not overwrite the entity's.
    _seed_battle(conn)
    _write(conn, _identity("Scipio", qid="Q1", mentions=[_mention("Scipio")]))
    _write(
        conn,
        _identity("Scipio", qid=None, mentions=[_mention("Scipio", side=OTHER_SIDE)]),
    )

    assert _scalar(conn, "SELECT COUNT(*) FROM generals") == 2
    assert _scalar(conn, "SELECT COUNT(*) FROM generals WHERE wikidata_id = 'Q1'") == 1


def test_a_mention_whose_battle_is_absent_is_counted_not_invented(conn: Connection) -> None:
    # No battle was seeded. Writing one here would produce a battle with
    # commanders and no troops, which nothing downstream could interpret.
    counts = _write(conn, _identity("Agrippa", qid="Q48174", mentions=[_mention("Agrippa")]))

    assert counts.commanders_written == 0
    assert counts.battles_not_in_db == 1
    assert _scalar(conn, "SELECT COUNT(*) FROM generals") == 0


# ─── Unresolved mentions ─────────────────────────────────────────────────────


def test_unresolved_mentions_are_logged_rather_than_dropped(conn: Connection) -> None:
    _seed_battle(conn)
    counts = ResolveCounts()
    log_unresolved(
        conn,
        [_mention("Unknown"), _mention("various local chieftains")],
        load_battle_index(conn),
        load_side_index(conn),
        counts,
    )

    assert counts.missing_rows == 2
    assert (
        _scalar(
            conn,
            "SELECT COUNT(*) FROM missing_data_log WHERE field_name = :f",
            f=UNRESOLVED_FIELD,
        )
        == 2
    )
    # Not random: a mention goes unresolved because of how obscure its subject
    # is, which relates to the commander's own record.
    assert (
        _scalar(
            conn,
            "SELECT DISTINCT missingness_class::text FROM missing_data_log WHERE field_name = :f",
            f=UNRESOLVED_FIELD,
        )
        == "mnar"
    )


def test_logging_unresolved_mentions_twice_does_not_accumulate(conn: Connection) -> None:
    _seed_battle(conn)
    battles, sides = load_battle_index(conn), load_side_index(conn)
    for _ in range(2):
        log_unresolved(conn, [_mention("Unknown")], battles, sides, ResolveCounts())

    assert (
        _scalar(
            conn,
            "SELECT COUNT(*) FROM missing_data_log WHERE field_name = :f",
            f=UNRESOLVED_FIELD,
        )
        == 1
    )


# ─── Duplicate battles ───────────────────────────────────────────────────────


def _pairs(conn: Connection) -> set[tuple[str, str]]:
    """Run the duplicate scan and return the name pairs it reported."""
    return {(d.left_name, d.right_name) for d in find_duplicate_battles(conn)}


def test_one_engagement_under_two_names_is_reported(conn: Connection) -> None:
    _seed_battle(conn, name="Battle of Actium", date=BATTLE_DATE)
    _seed_battle(conn, name="Actium", date=BATTLE_DATE)

    assert ("Battle of Actium", "Actium") in _pairs(conn)


def test_numbered_battles_of_one_place_are_not_reported(conn: Connection) -> None:
    # The single most common false positive. These names differ by one word
    # and the ordinal is the whole point of them.
    _seed_battle(conn, name="First Battle of Bull Run", date="1861-07-21")
    _seed_battle(conn, name="Second Battle of Bull Run", date="1862-08-28")

    assert _pairs(conn) == set()


def test_battles_centuries_apart_are_not_reported(conn: Connection) -> None:
    # Identical names, different engagements. The date is the only thing that
    # separates them, which is why an undated battle pairs with nothing.
    _seed_battle(conn, name="Battle of Panipat", date="1526-04-21")
    _seed_battle(conn, name="Battle of Panipat", date="1761-01-14")

    assert _pairs(conn) == set()


def test_an_undated_battle_pairs_with_nothing(conn: Connection) -> None:
    conn.execute(text("INSERT INTO battles (name) VALUES ('Battle of Actium')"))
    conn.execute(text("INSERT INTO battles (name) VALUES ('Actium')"))

    assert _pairs(conn) == set()


def test_unrelated_battles_in_one_year_are_not_reported(conn: Connection) -> None:
    _seed_battle(conn, name="Battle of Austerlitz", date="1805-12-02")
    _seed_battle(conn, name="Battle of Trafalgar", date="1805-10-21")

    assert _pairs(conn) == set()


def test_duplicates_are_reported_and_never_merged(conn: Connection) -> None:
    # The scan is deliberately read-only: merging battle rows means
    # re-parenting their sides, troop reports and commanders, and a wrong
    # merge destroys the outcome data the model is fitted to.
    _seed_battle(conn, name="Battle of Actium", date=BATTLE_DATE)
    _seed_battle(conn, name="Actium", date=BATTLE_DATE)
    before = _scalar(conn, "SELECT COUNT(*) FROM battles")

    assert len(find_duplicate_battles(conn)) == 1
    assert _scalar(conn, "SELECT COUNT(*) FROM battles") == before


# ─── Quality gates ───────────────────────────────────────────────────────────


def _gates(conn: Connection) -> dict[str, Any]:
    """Run every quality check in agents/resolve.yaml.

    Args:
        conn: An open connection.

    Returns:
        Each check's result, keyed by name.
    """
    spec = yaml.safe_load((AGENTS / "resolve.yaml").read_text(encoding="utf-8"))
    results = QualityRunner(conn).run_all(spec["quality_checks"])
    return {result.name: result for result in results}


def test_every_resolve_gate_executes_against_the_real_schema(conn: Connection) -> None:
    # §4.1 and §11.4 of handover.md are both this failure: a gate whose SQL
    # names something the schema does not have, which can only ever error.
    _seed_battle(conn)
    _write(conn, _identity("Agrippa", qid="Q48174", mentions=[_mention("Agrippa")]))

    results = _gates(conn)
    assert set(results) == {
        "resolution_rate",
        "no_duplicate_generals",
        "wikidata_link_rate",
        "commanders_per_battle",
        "alias_coverage",
    }
    for name, result in results.items():
        assert "does not exist" not in (result.message or ""), f"{name}: {result.message}"
        assert "aborted" not in (result.message or ""), f"{name}: {result.message}"


def test_the_resolution_rate_gate_can_actually_fail(conn: Connection) -> None:
    # The gate this replaced counted battle_commanders.general_id IS NOT NULL.
    # That column is NOT NULL, so the old gate divided written rows by written
    # rows and read 1.0 however many mentions the stage lost. This one has a
    # denominator that grows when resolution fails.
    _seed_battle(conn)
    _write(conn, _identity("Agrippa", qid="Q48174", mentions=[_mention("Agrippa")]))
    assert _gates(conn)["resolution_rate"].actual_value == pytest.approx(1.0)

    log_unresolved(
        conn,
        [_mention(f"Unknown {n}") for n in range(9)],
        load_battle_index(conn),
        load_side_index(conn),
        ResolveCounts(),
    )

    result = _gates(conn)["resolution_rate"]
    assert result.actual_value == pytest.approx(0.1)
    assert not result.passed


def test_the_duplicate_generals_gate_catches_a_split_entity(conn: Connection) -> None:
    _seed_battle(conn)
    _write(conn, _identity("Agrippa", qid="Q48174", mentions=[_mention("Agrippa")]))
    assert _gates(conn)["no_duplicate_generals"].passed

    # generals.wikidata_id is UNIQUE, so the schema already refuses the
    # duplicate this gate looks for. Both defences are wanted: the constraint
    # stops it happening, the gate says so if the constraint is ever relaxed.
    with pytest.raises(IntegrityError):
        conn.execute(
            text("INSERT INTO generals (canonical_name, wikidata_id) VALUES ('Other', 'Q48174')")
        )
