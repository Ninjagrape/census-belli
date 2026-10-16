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
  docs/handover-log.md each found the hard way.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml
from sqlalchemy import Connection, Engine, text

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
    prune_stale_commanders,
    remove_orphan_generals,
    side_of,
    write_identity,
)
from pipeline.resolvers.records import BattleContext
from pipeline.resolvers.store import NON_PERSON_FIELD, UNRESOLVED_FIELD

AGENTS = Path(__file__).resolve().parents[2] / "agents"

BATTLE_NAME = "Battle of Actium"
SIDE_LABEL = "Octavian"
OTHER_SIDE = "Antony and Cleopatra"
# 31 BC. Astronomical -30: there is a year zero, so the historical year is one
# further from the epoch than the astronomical one (handover-log §4.5).
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


def _mention(
    name: str, *, side: str = SIDE_LABEL, evidence: str = "", infobox_order: int | None = None
) -> Mention:
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
        infobox_order=infobox_order,
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


def test_infobox_order_is_written_on_insert_and_replaced_on_update(conn: Connection) -> None:
    _seed_battle(conn)
    agrippa = [_mention("Agrippa", infobox_order=1), _mention("Agrippa", infobox_order=0)]
    _write(conn, _identity("Agrippa", qid="Q48174", mentions=agrippa))
    _write(conn, _identity("Octavian", qid="Q1405", mentions=[_mention("Octavian")]))

    def orders() -> dict[str, int | None]:
        rows = conn.execute(
            text(
                "SELECT g.canonical_name, bc.infobox_order FROM battle_commanders bc "
                "JOIN generals g USING (general_id)"
            )
        ).fetchall()
        return {name: order for name, order in rows}

    # The lowest order among a commander's mentions; a body-only mention is NULL.
    assert orders() == {"Agrippa": 0, "Octavian": None}

    _write(
        conn, _identity("Agrippa", qid="Q48174", mentions=[_mention("Agrippa", infobox_order=2)])
    )
    assert orders()["Agrippa"] == 2


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
    # §4.1 and §11.4 of docs/handover-log.md are both this failure: a gate whose SQL
    # names something the schema does not have, which can only ever error.
    _seed_battle(conn)
    _write(conn, _identity("Agrippa", qid="Q48174", mentions=[_mention("Agrippa")]))

    results = _gates(conn)
    assert set(results) == {
        "resolution_rate",
        "no_duplicate_generals",
        "probable_duplicate_generals",
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


def _add_local_general(conn: Connection, name: str, lo: int | None, hi: int | None) -> None:
    """Insert a corpus-local general (no Q-id) with an optional year range."""
    conn.execute(
        text(
            "INSERT INTO generals (canonical_name, years_active) VALUES "
            "(:n, CASE WHEN CAST(:lo AS INT) IS NULL THEN NULL "
            "ELSE int4range(CAST(:lo AS INT), CAST(:hi AS INT)) END)"
        ),
        {"n": name, "lo": lo, "hi": hi},
    )


def test_the_duplicate_generals_gate_catches_a_split_corpus_local_person(
    conn: Connection,
) -> None:
    # generals.wikidata_id is UNIQUE, so the gate that counted duplicate Q-ids
    # could never fail. The duplicate that can actually happen is a general
    # with no Q-id that is another row under a different spelling of one name.
    _add_local_general(conn, "Gebhard von Blücher", 1806, 1816)
    assert _gates(conn)["no_duplicate_generals"].passed

    _add_local_general(conn, "Gebhard von Blucher", 1813, 1815)
    result = _gates(conn)["no_duplicate_generals"]
    assert result.actual_value == 1
    assert not result.passed


def test_the_duplicate_gate_ignores_namesakes_a_century_apart(conn: Connection) -> None:
    _add_local_general(conn, "Scipio", -210, -200)
    _add_local_general(conn, "Scipio", 1805, 1815)
    assert _gates(conn)["no_duplicate_generals"].passed


def test_the_duplicate_gate_ignores_two_people_wikidata_told_apart(conn: Connection) -> None:
    for qid, lo in (("Q1", 1805), ("Q2", 1806)):
        conn.execute(
            text(
                "INSERT INTO generals (canonical_name, wikidata_id, years_active) "
                "VALUES ('Friedrich Kleist', :q, int4range(:lo, :hi))"
            ),
            {"q": qid, "lo": lo, "hi": lo + 5},
        )
    assert _gates(conn)["no_duplicate_generals"].passed


def test_the_probable_duplicate_gate_catches_one_man_in_two_spellings(
    conn: Connection,
) -> None:
    # "Karl von Wrede" and "Karl Philipp von Wrede": same first and last token,
    # overlapping years, neither linked. Found on the 2026-10 pilot.
    _add_local_general(conn, "Karl von Wrede", 1809, 1815)
    _add_local_general(conn, "Karl Philipp von Wrede", 1809, 1816)
    assert _gates(conn)["no_duplicate_generals"].passed
    assert _gates(conn)["probable_duplicate_generals"].actual_value == 1


def _add_linked_general(conn: Connection, name: str, qid: str, lo: int, hi: int) -> None:
    """Insert a general that carries a Wikidata id."""
    conn.execute(
        text(
            "INSERT INTO generals (canonical_name, wikidata_id, years_active) "
            "VALUES (:n, :q, int4range(:lo, :hi))"
        ),
        {"n": name, "q": qid, "lo": lo, "hi": hi},
    )


def _write_uncertain_split(conn: Connection, name: str, *, uncertain: bool) -> None:
    """Write a corpus-local identity through the stage's writer."""
    identity = _identity(name, qid=None, mentions=[_mention(name)], confidence=0.4)
    identity.method = "llm_uncertain" if uncertain else "new_entity"
    identity.uncertain_split = uncertain
    _write(conn, identity)


def test_an_uncertain_split_beside_a_linked_namesake_is_a_warning_not_an_error(
    conn: Connection,
) -> None:
    # Handover 9.3: UNCERTAIN splits on purpose. Friedrich von Bulow, Georgy
    # Zakharov, William Holmes and Xanthippus each have a Q-id row and a
    # corpus-local row from a battle where the LLM could not tell.
    _seed_battle(conn)
    _add_linked_general(conn, "Xanthippus", "Q1", -60, 0)
    _add_linked_general(conn, "Georgy Zakharov", "Q2", -60, 0)
    _write_uncertain_split(conn, "Xanthippus", uncertain=True)
    _write_uncertain_split(conn, "Georgy Zakharov", uncertain=True)

    notes = _scalar(conn, "SELECT notes FROM generals WHERE wikidata_id IS NULL LIMIT 1")
    assert "kept separate after an uncertain match" in notes
    assert _gates(conn)["no_duplicate_generals"].passed
    # Both pairs show up in the warning, the one-token name included.
    assert _gates(conn)["probable_duplicate_generals"].actual_value == 2


def test_a_local_row_with_no_uncertain_answer_behind_it_still_fails_the_gate(
    conn: Connection,
) -> None:
    _seed_battle(conn)
    _add_linked_general(conn, "William Holmes", "Q3", -60, 0)
    _write_uncertain_split(conn, "William Holmes", uncertain=False)

    result = _gates(conn)["no_duplicate_generals"]
    assert result.actual_value == 1
    assert not result.passed


def test_two_corpus_local_rows_still_fail_even_if_one_was_an_uncertain_split(
    conn: Connection,
) -> None:
    _seed_battle(conn)
    _write_uncertain_split(conn, "Friedrich von Bülow", uncertain=True)
    _add_local_general(conn, "Friedrich von Bulow", -40, -20)

    result = _gates(conn)["no_duplicate_generals"]
    assert result.actual_value == 1
    assert not result.passed


def test_the_gate_text_matches_the_note_the_writer_emits() -> None:
    from pipeline.resolvers.records import UNCERTAIN_SPLIT_NOTE

    spec = yaml.safe_load((AGENTS / "resolve.yaml").read_text(encoding="utf-8"))
    for check in spec["quality_checks"]:
        if check["name"] in ("no_duplicate_generals", "probable_duplicate_generals"):
            assert UNCERTAIN_SPLIT_NOTE in check["check"], check["name"]


def test_one_identity_that_fails_to_write_does_not_lose_the_others(conn: Connection) -> None:
    from pipeline.stages.base import StageContext
    from pipeline.stages.resolve import _write_database

    _seed_battle(conn)
    # A confidence outside [0, 1] violates a CHECK on battle_commanders after
    # the generals row is already inserted. Without a savepoint the aborted
    # transaction fails every identity after it too.
    broken = _identity("Broken", qid="Q1", mentions=[_mention("Broken")], confidence=1.5)
    fine = _identity("Agrippa", qid="Q48174", mentions=[_mention("Agrippa")])
    counts = ResolveCounts()

    _, homeless = _write_database(StageContext(db_conn=conn), [], {}, [broken, fine], counts, {})

    names = {row[0] for row in conn.execute(text("SELECT canonical_name FROM generals"))}
    assert names == {"Agrippa"}
    assert counts.failed_identities == 1
    assert counts.generals_written == 1
    assert homeless == []
    # The failed identity's mention reached no table, so the gate must see it
    # as a miss rather than as a mention that never existed.
    assert (
        _scalar(
            conn,
            "SELECT COUNT(*) FROM missing_data_log WHERE field_name = :f",
            f=UNRESOLVED_FIELD,
        )
        == 1
    )


# ─── Denominators ────────────────────────────────────────────────────────────


def test_a_resolved_mention_with_no_side_row_counts_as_a_miss(conn: Connection) -> None:
    # write_identity used to count sides_not_in_db and forget the mention, so
    # the resolution_rate gate never saw it and read higher than the truth.
    _seed_battle(conn)
    battles, sides = load_battle_index(conn), load_side_index(conn)
    identity = _identity(
        "Agrippa",
        qid="Q48174",
        mentions=[_mention("Agrippa"), _mention("Agrippa", side="A side extract never wrote")],
    )
    counts = ResolveCounts()
    unplaced: list[Mention] = []
    write_identity(conn, identity, battles, sides, counts, {}, unplaced)
    log_unresolved(conn, [], battles, sides, counts, unplaced=unplaced)

    assert counts.sides_not_in_db == 1
    assert (
        _scalar(
            conn,
            "SELECT COUNT(*) FROM missing_data_log WHERE field_name = :f",
            f=UNRESOLVED_FIELD,
        )
        == 1
    )
    # One commander row written, one miss logged: 1 of 2 mentions reached the table.
    assert _gates(conn)["resolution_rate"].actual_value == pytest.approx(0.5)


def test_a_mention_whose_battle_is_missing_is_returned_for_the_file(conn: Connection) -> None:
    # missing_data_log.battle_id is NOT NULL, so there is nowhere in that table
    # to put it. It must come back to the caller, not vanish.
    _seed_battle(conn)
    ghost = _mention("Agrippa")
    ghost = Mention(**{**ghost.__dict__, "battle_name": "Battle that was never written"})
    counts = ResolveCounts()
    homeless = log_unresolved(conn, [ghost], load_battle_index(conn), load_side_index(conn), counts)
    assert homeless == [ghost]
    assert counts.unplaced_rows == 1
    assert counts.missing_rows == 0


def test_non_person_mentions_are_logged_outside_the_resolution_rate(conn: Connection) -> None:
    _seed_battle(conn)
    battles, sides = load_battle_index(conn), load_side_index(conn)
    _write(conn, _identity("Agrippa", qid="Q48174", mentions=[_mention("Agrippa")]))
    counts = ResolveCounts()
    log_unresolved(conn, [], battles, sides, counts, non_person=[_mention("Luftwaffe")])

    assert (
        _scalar(
            conn,
            "SELECT COUNT(*) FROM missing_data_log WHERE field_name = :f",
            f=NON_PERSON_FIELD,
        )
        == 1
    )
    assert _gates(conn)["resolution_rate"].actual_value == pytest.approx(1.0)

    # And a re-run replaces them rather than adding to them.
    log_unresolved(conn, [], battles, sides, ResolveCounts(), non_person=[_mention("Luftwaffe")])
    assert (
        _scalar(
            conn,
            "SELECT COUNT(*) FROM missing_data_log WHERE field_name = :f",
            f=NON_PERSON_FIELD,
        )
        == 1
    )


# ─── Biography ───────────────────────────────────────────────────────────────


def _bio_identity(
    name: str, qid: str, birth: int | None, death: int | None, era: str | None
) -> Identity:
    identity = _identity(name, qid=qid, mentions=[_mention(name)])
    identity.birth_year, identity.death_year, identity.era = birth, death, era
    return identity


def test_born_died_and_era_are_written_including_for_a_bc_birth(conn: Connection) -> None:
    _seed_battle(conn)
    _write(conn, _bio_identity("Agrippa", "Q48174", -61, -11, "ancient"))

    row = conn.execute(
        text(
            "SELECT born_year_astronomical, died_year_astronomical, era FROM generals "
            "WHERE wikidata_id = 'Q48174'"
        )
    ).one()
    assert tuple(row) == (-61, -11, "ancient")


def test_a_rerun_without_a_biography_does_not_blank_an_earlier_one(conn: Connection) -> None:
    _seed_battle(conn)
    _write(conn, _bio_identity("Agrippa", "Q48174", -61, -11, "ancient"))
    _write(conn, _bio_identity("Agrippa", "Q48174", None, None, None))

    row = conn.execute(
        text("SELECT born_year_astronomical, era FROM generals WHERE wikidata_id = 'Q48174'")
    ).one()
    assert tuple(row) == (-61, "ancient")


# ─── Orphan generals ─────────────────────────────────────────────────────────


def _general_with_alias(conn: Connection, name: str, qid: str | None = None) -> int:
    """Insert a general with one alias and return its id."""
    general_id = int(
        conn.execute(
            text(
                "INSERT INTO generals (canonical_name, wikidata_id) "
                "VALUES (:n, :q) RETURNING general_id"
            ),
            {"n": name, "q": qid},
        ).scalar_one()
    )
    conn.execute(
        text("INSERT INTO general_aliases (general_id, alias_name) VALUES (:g, :a)"),
        {"g": general_id, "a": name},
    )
    return general_id


def _exists(conn: Connection, general_id: int) -> bool:
    return bool(_scalar(conn, "SELECT COUNT(*) FROM generals WHERE general_id = :g", g=general_id))


def _seed_orphan_scenario(conn: Connection) -> dict[str, int]:
    """One orphan, one commander-linked, one model-referenced general."""
    battle_id = _seed_battle(conn)
    side_id = int(_scalar(conn, "SELECT MIN(side_id) FROM battle_sides"))
    orphan = _general_with_alias(conn, "Georgy Zakharov")
    linked = _general_with_alias(conn, "Linked Commander", "Q111")
    modelled = _general_with_alias(conn, "Modelled Commander")
    conn.execute(
        text("INSERT INTO battle_commanders (battle_id, side_id, general_id) VALUES (:b, :s, :g)"),
        {"b": battle_id, "s": side_id, "g": linked},
    )
    run_id = int(
        conn.execute(
            text(
                "INSERT INTO model_runs (run_name, model_type, config) "
                "VALUES ('r', 'test', '{}') RETURNING run_id"
            )
        ).scalar_one()
    )
    conn.execute(
        text(
            "INSERT INTO general_skill_estimates (run_id, general_id, skill_mean, skill_sd, "
            "skill_ci_lo, skill_ci_hi, n_battles) VALUES (:r, :g, 0, 1, -1, 1, 1)"
        ),
        {"r": run_id, "g": modelled},
    )
    return {"orphan": orphan, "linked": linked, "modelled": modelled}


def test_an_orphan_general_is_removed_with_its_aliases(conn: Connection) -> None:
    ids = _seed_orphan_scenario(conn)

    removed = remove_orphan_generals(conn)

    assert [general_id for general_id, _ in removed] == [ids["orphan"]]
    assert not _exists(conn, ids["orphan"])
    assert (
        _scalar(conn, "SELECT COUNT(*) FROM general_aliases WHERE general_id = :g", g=ids["orphan"])
        == 0
    )
    assert _exists(conn, ids["linked"])
    assert _exists(conn, ids["modelled"])
    assert (
        _scalar(
            conn, "SELECT COUNT(*) FROM general_aliases WHERE general_id = :g", g=ids["modelled"]
        )
        == 1
    )


def test_a_general_in_battle_war_details_survives_the_orphan_sweep(conn: Connection) -> None:
    battle_id = _seed_battle(conn)
    general_id = _general_with_alias(conn, "Detailed Commander")
    run_id = int(
        conn.execute(
            text(
                "INSERT INTO model_runs (run_name, model_type, config) "
                "VALUES ('r', 'test', '{}') RETURNING run_id"
            )
        ).scalar_one()
    )
    conn.execute(
        text(
            "INSERT INTO battle_war_details (run_id, battle_id, general_id, war_contribution) "
            "VALUES (:r, :b, :g, 0.1)"
        ),
        {"r": run_id, "b": battle_id, "g": general_id},
    )

    assert remove_orphan_generals(conn) == []
    assert _exists(conn, general_id)


def test_an_orphan_no_longer_trips_the_duplicate_generals_gate(conn: Connection) -> None:
    _add_linked_general(conn, "Georgy Zakharov", "Q222", 1900, 1950)
    _general_with_alias(conn, "Georgy Zakharov")
    assert not _gates(conn)["no_duplicate_generals"].passed

    remove_orphan_generals(conn)

    assert _gates(conn)["no_duplicate_generals"].passed


def test_removing_orphans_twice_removes_nothing_the_second_time(conn: Connection) -> None:
    _seed_orphan_scenario(conn)

    assert len(remove_orphan_generals(conn)) == 1
    assert remove_orphan_generals(conn) == []


# ─── Stale links ─────────────────────────────────────────────────────────────


def _run_and_prune(
    conn: Connection,
    *identities: Identity,
    pending: list[Mention] | None = None,
    unresolved: list[Mention] | None = None,
) -> tuple[int, int]:
    """Write identities as one run, then prune what the run did not link.

    Args:
        conn: An open connection.
        identities: The run's resolved people.
        pending: Mentions of groups whose answer is deferred this run.
        unresolved: Mentions the run refused or could not link to anyone.

    Returns:
        Rows removed and sides reset.
    """
    battles = load_battle_index(conn)
    sides = load_side_index(conn)
    counts = ResolveCounts()
    cache: dict[tuple[str, str], int] = {}
    linked: dict[tuple[int, int, int], str] = {}
    for identity in identities:
        write_identity(conn, identity, battles, sides, counts, cache, None, linked)
    mentions = [m for i in identities for m in i.mentions] + (pending or []) + (unresolved or [])
    processed = {p for p in (side_of(m, battles, sides) for m in mentions) if p}
    protected = {p for p in (side_of(m, battles, sides) for m in pending or []) if p}
    return prune_stale_commanders(conn, processed, protected, linked)


def _commanders(conn: Connection, side: str) -> dict[str, tuple[float, str, str]]:
    """Read a side's rows as qid -> (weight, attribution_method, role)."""
    rows = conn.execute(
        text(
            "SELECT g.wikidata_id, bc.attribution_weight, bc.attribution_method, "
            "bc.command_role::text FROM battle_commanders bc "
            "JOIN generals g USING (general_id) "
            "JOIN battle_sides bs ON bs.side_id = bc.side_id WHERE bs.side_label = :side"
        ),
        {"side": side},
    )
    return {str(q): (float(w), str(m), str(r)) for q, w, m, r in rows}


def _classified_state(conn: Connection) -> None:
    """Stand in for classify: give every row an LLM-decided weight."""
    conn.execute(
        text(
            "UPDATE battle_commanders SET attribution_weight = 0.5, "
            "attribution_method = 'llm', hierarchy_rank = 1"
        )
    )


def _first_run(conn: Connection) -> list[Mention]:
    """Run once with two commanders on Octavian's side and one opposite."""
    _seed_battle(conn)
    wrong = _mention("Agrippa")
    steady = _mention("Maecenas")
    other = _mention("Antony", side=OTHER_SIDE)
    _run_and_prune(
        conn,
        _identity("Wrong Agrippa", qid="Q1", mentions=[wrong]),
        _identity("Maecenas", qid="Q2", mentions=[steady]),
        _identity("Mark Antony", qid="Q3", mentions=[other]),
    )
    _classified_state(conn)
    return [wrong, steady, other]


def test_a_relinked_mention_removes_the_old_row_and_resets_the_side(conn: Connection) -> None:
    wrong, steady, other = _first_run(conn)

    removed, reset = _run_and_prune(
        conn,
        _identity("Marcus Vipsanius Agrippa", qid="Q48174", mentions=[wrong]),
        _identity("Maecenas", qid="Q2", mentions=[steady]),
        _identity("Mark Antony", qid="Q3", mentions=[other]),
    )

    assert (removed, reset) == (1, 1)
    side = _commanders(conn, SIDE_LABEL)
    assert set(side) == {"Q2", "Q48174"}
    # Back at the defaults so classify recomputes the side, with the role
    # resolve itself read off the infobox put back.
    assert all(state == (1.0, "equal", "field_commander") for state in side.values())
    # The opposite side lost nothing, so its classified state is kept.
    assert _commanders(conn, OTHER_SIDE) == {"Q3": (0.5, "llm", "field_commander")}


def test_a_detached_mention_removes_its_row(conn: Connection) -> None:
    wrong, steady, other = _first_run(conn)

    removed, reset = _run_and_prune(
        conn,
        _identity("Maecenas", qid="Q2", mentions=[steady]),
        _identity("Mark Antony", qid="Q3", mentions=[other]),
        unresolved=[wrong],
    )

    assert (removed, reset) == (1, 1)
    assert set(_commanders(conn, SIDE_LABEL)) == {"Q2"}


def test_a_side_with_a_deferred_mention_keeps_its_rows(conn: Connection) -> None:
    wrong, steady, other = _first_run(conn)

    removed, reset = _run_and_prune(
        conn,
        _identity("Marcus Vipsanius Agrippa", qid="Q48174", mentions=[wrong]),
        _identity("Mark Antony", qid="Q3", mentions=[other]),
        pending=[steady],
    )

    assert (removed, reset) == (0, 0)
    side = _commanders(conn, SIDE_LABEL)
    assert set(side) == {"Q1", "Q2", "Q48174"}
    assert side["Q2"] == (0.5, "llm", "field_commander")


def test_an_unchanged_rerun_removes_nothing(conn: Connection) -> None:
    wrong, steady, other = _first_run(conn)

    removed, reset = _run_and_prune(
        conn,
        _identity("Wrong Agrippa", qid="Q1", mentions=[wrong]),
        _identity("Maecenas", qid="Q2", mentions=[steady]),
        _identity("Mark Antony", qid="Q3", mentions=[other]),
    )

    assert (removed, reset) == (0, 0)
    assert _commanders(conn, SIDE_LABEL)["Q1"] == (0.5, "llm", "field_commander")


def test_a_removed_row_does_not_leave_a_dangling_reports_to(conn: Connection) -> None:
    wrong, steady, other = _first_run(conn)
    conn.execute(
        text(
            "UPDATE battle_commanders SET reports_to_bc_id = ("
            "SELECT bc.bc_id FROM battle_commanders bc JOIN generals g USING (general_id) "
            "WHERE g.wikidata_id = 'Q1') "
            "WHERE general_id = (SELECT general_id FROM generals WHERE wikidata_id = 'Q2')"
        )
    )

    _run_and_prune(
        conn,
        _identity("Marcus Vipsanius Agrippa", qid="Q48174", mentions=[wrong]),
        _identity("Maecenas", qid="Q2", mentions=[steady]),
        _identity("Mark Antony", qid="Q3", mentions=[other]),
    )

    assert (
        _scalar(conn, "SELECT COUNT(*) FROM battle_commanders WHERE reports_to_bc_id IS NOT NULL")
        == 0
    )
