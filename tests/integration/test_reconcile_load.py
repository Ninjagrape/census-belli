"""Integration tests for the reconcile loader, against a real PostgreSQL.

These run against a live database on purpose. The loader's job is mostly to be
right about what the schema actually contains -- a BC date that cannot be
decoded into Python, an enum psycopg will not infer, a notes prefix another
stage's quality gate depends on -- and every one of those has been a real bug
in this project against fixtures that agreed with the code.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from pipeline.db import DatabaseConfigError, apply_schema, database_url, get_engine
from pipeline.reconcilers.load import classify_sides, load_reports, write_unfillable
from pipeline.reconcilers.records import ReconcileCounts


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


# ─── Seeding helpers ─────────────────────────────────────────────────────────

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


def _battle(conn: Connection, name: str, date: str = "1815-06-18") -> int:
    """Insert a battle and return its id."""
    return int(conn.execute(_INSERT_BATTLE, {"name": name, "date": date}).scalar_one())


def _side(conn: Connection, battle_id: int, label: str = "Side A") -> int:
    """Insert a side and return its id."""
    return int(conn.execute(_INSERT_SIDE, {"battle_id": battle_id, "label": label}).scalar_one())


def _source(conn: Connection, source_type: str = "wikipedia_infobox", url: str = "u1") -> int:
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
    is_estimate: bool = False,
    is_upper_bound: bool = False,
    is_lower_bound: bool = False,
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
            "is_estimate": is_estimate,
            "is_upper_bound": is_upper_bound,
            "is_lower_bound": is_lower_bound,
            "context": context,
        },
    )


def _load(conn: Connection) -> tuple[list[Any], ReconcileCounts]:
    """Run the loader and return its reports alongside the counters."""
    counts = ReconcileCounts()
    reports = load_reports(conn, ancient_cutoff_year=500, counts=counts)
    return reports, counts


def _missing_rows(conn: Connection) -> list[tuple[Any, ...]]:
    """Read every missing_data_log row as (side_id, field_name, notes)."""
    return [
        tuple(row)
        for row in conn.execute(
            text("SELECT side_id, field_name, notes FROM missing_data_log ORDER BY log_id")
        )
    ]


# ─── BC dates ────────────────────────────────────────────────────────────────


def test_a_bc_battle_loads_its_astronomical_year(conn: Connection) -> None:
    # date_start cannot be decoded into Python for any BC battle: datetime.date
    # has MINYEAR == 1. Reading year_astronomical instead is why that generated
    # column exists, and this corpus is heavily ancient.
    battle = _battle(conn, "Cannae", "0216-08-02 BC")
    side = _side(conn, battle, "Rome")
    _troop(conn, side, _source(conn), 86_000.0)

    reports, _ = _load(conn)

    assert len(reports) == 1
    # 216 BC is -215 in astronomical numbering, which has a year zero.
    assert reports[0].year_astronomical == -215


# ─── Exclusions ──────────────────────────────────────────────────────────────


def test_only_total_branch_troop_reports_are_loaded(conn: Connection) -> None:
    # Arrian's "40,000 cavalry and 1,000,000 infantry" is two rows. Treating
    # either as a noisy observation of the total drags the estimate, and there
    # is no per-branch estimate column to put them in instead.
    battle = _battle(conn, "Gaugamela", "0331-10-01 BC")
    side = _side(conn, battle, "Macedon")
    source = _source(conn)
    _troop(conn, side, source, 47_000.0, branch="total")
    _troop(conn, side, source, 7_000.0, branch="cavalry")
    _troop(conn, side, source, 40_000.0, branch="infantry")

    reports, counts = _load(conn)
    troops = [r for r in reports if r.quantity == "troops"]

    assert [r.reported_value for r in troops] == [47_000.0]
    assert counts.excluded_non_total_branch == 2


def test_a_naval_ship_count_never_enters_the_personnel_reports(conn: Connection) -> None:
    # reported_value carries no unit and a real infobox reads "250-400
    # galleys". A ship count in the same covariate as a head count is the
    # handover-log 4.2 class of error.
    battle = _battle(conn, "Actium", "0031-09-02 BC")
    side = _side(conn, battle, "Octavian")
    source = _source(conn)
    _troop(conn, side, source, 400.0, branch="naval")
    _troop(conn, side, source, 16_000.0, branch="total")

    reports, counts = _load(conn)
    troops = [r for r in reports if r.quantity == "troops"]

    assert [r.reported_value for r in troops] == [16_000.0]
    assert counts.excluded_naval == 1


def test_population_scope_rows_are_excluded_and_counted_like_naval_rows(conn: Connection) -> None:
    # Bibracte: "368,000 people" is the migrating nation, not an army. Even a
    # row stored as a total must not reach the fit, and the scope offsets must
    # never see "population" as a level.
    battle = _battle(conn, "Bibracte", "0058-06-01 BC")
    side = _side(conn, battle, "Helvetii")
    source = _source(conn)
    _troop(conn, side, source, 368_000.0, branch="other", scope="population")
    _troop(conn, side, source, 278_000.0, branch="total", scope="population")
    _troop(conn, side, source, 90_000.0, context="90,000 warriors")

    reports, counts = _load(conn)
    troops = [r for r in reports if r.quantity == "troops"]

    assert [r.reported_value for r in troops] == [90_000.0]
    assert counts.excluded_population_scope == 2
    assert counts.excluded_non_total_branch == 0


def test_a_modern_total_supersedes_every_other_total_on_its_side(conn: Connection) -> None:
    # Hazir: the Muslim-source 70,000 and the modern ~7,000 are both totals
    # after extract; the modern one is the anchor and the rest are dropped.
    battle = _battle(conn, "Hazir", "0634-08-01")
    side = _side(conn, battle, "Byzantine Empire")
    source = _source(conn)
    ctx = "70,000 (Muslim source) ~7,000 (Modern estimate)"
    _troop(conn, side, source, 70_000.0, context=ctx)
    _troop(conn, side, source, 7_000.0, context=ctx)
    _troop(conn, side, source, 51_000.0, context="derived: sum of 2 component figures")
    other = _side(conn, battle, "Rashidun")
    _troop(conn, other, source, 17_000.0, context="17,000 (Muslim source)")

    reports, counts = _load(conn)
    troops = sorted((r.side_id, r.reported_value) for r in reports if r.quantity == "troops")

    assert troops == [(side, 7_000.0), (other, 17_000.0)]
    assert counts.excluded_superseded_by_modern == 2
    assert counts.regime_unlabelled == 0


def test_a_non_positive_troop_value_is_excluded_and_counted(conn: Connection) -> None:
    battle = _battle(conn, "Zero Field")
    side = _side(conn, battle)
    source = _source(conn)
    _troop(conn, side, source, 0.0)
    _troop(conn, side, source, 12_000.0)

    reports, counts = _load(conn)
    troops = [r for r in reports if r.quantity == "troops"]

    assert [r.reported_value for r in troops] == [12_000.0]
    assert counts.excluded_non_positive == 1


def test_a_zero_casualty_value_is_kept_because_zero_casualties_is_a_fact(
    conn: Connection,
) -> None:
    battle = _battle(conn, "Bloodless")
    side = _side(conn, battle)
    source = _source(conn)
    conn.execute(
        _INSERT_CASUALTY,
        {"side_id": side, "source_id": source, "casualty_type": "total", "value": 0.0},
    )

    reports, _ = _load(conn)
    casualties = [r for r in reports if r.quantity == "casualties"]

    assert [r.reported_value for r in casualties] == [0.0]


# ─── Source identity ─────────────────────────────────────────────────────────


def test_two_source_rows_sharing_a_type_and_url_get_one_source_key(conn: Connection) -> None:
    # sources has no unique constraint and the extract writer is a
    # SELECT-then-INSERT, so one document can hold several source_ids.
    battle = _battle(conn, "Duplicated Source")
    side = _side(conn, battle)
    first = _source(conn, url="https://en.wikipedia.org/wiki/X")
    second = _source(conn, url="https://en.wikipedia.org/wiki/X")
    assert first != second

    _troop(conn, side, first, 10_000.0)
    _troop(conn, side, second, 11_000.0)

    reports, _ = _load(conn)
    troops = [r for r in reports if r.quantity == "troops"]

    assert len({r.source_key for r in troops}) == 1
    assert len({r.source_id for r in troops}) == 2


# ─── Unfillable sides ────────────────────────────────────────────────────────


def test_a_side_whose_only_report_is_cavalry_is_unfillable_with_a_reason(
    conn: Connection,
) -> None:
    battle = _battle(conn, "Cavalry Only")
    side = _side(conn, battle)
    _troop(conn, side, _source(conn), 5_000.0, branch="cavalry")

    reports, _ = _load(conn)
    modellable, unfillable = classify_sides(reports, conn)

    assert side not in modellable
    assert [u.side_id for u in unfillable] == [side]
    assert unfillable[0].reason


def test_an_unfillable_side_gets_a_missing_data_log_row_prefixed_reconcile(
    conn: Connection,
) -> None:
    battle = _battle(conn, "Needs Logging")
    side = _side(conn, battle)
    _troop(conn, side, _source(conn), 5_000.0, branch="artillery")

    reports, counts = _load(conn)
    _, unfillable = classify_sides(reports, conn)
    write_unfillable(conn, unfillable, counts)

    rows = _missing_rows(conn)
    assert [(r[0], r[1]) for r in rows] == [(side, "troop_total")]
    assert rows[0][2].startswith("reconcile: ")
    assert counts.sides_unfillable == 1


def test_a_rerun_does_not_duplicate_missing_data_log_rows(conn: Connection) -> None:
    # A re-run must converge, not accumulate.
    battle = _battle(conn, "Rerun")
    side = _side(conn, battle)
    _troop(conn, side, _source(conn), 5_000.0, branch="cavalry")

    for _ in range(3):
        reports, counts = _load(conn)
        _, unfillable = classify_sides(reports, conn)
        write_unfillable(conn, unfillable, counts)

    assert len(_missing_rows(conn)) == 1


def test_a_rerun_does_not_delete_the_missing_data_rows_extract_wrote(
    conn: Connection,
) -> None:
    # extract writes troop_total for a side with NO reports at all; reconcile
    # writes it for a side whose reports are all unusable. Both must coexist
    # under one field_name, because classify's missing_data_all_logged gate
    # requires that exact literal. Scoping the clear on the notes prefix is
    # what stops reconcile deleting the other stage's evidence.
    battle = _battle(conn, "Shared Field Name")
    extract_side = _side(conn, battle, "No Reports At All")
    conn.execute(
        _INSERT_MISSING,
        {
            "battle_id": battle,
            "side_id": extract_side,
            "field_name": "troop_total",
            "notes": "extract: no strength field in the infobox",
        },
    )

    reconcile_side = _side(conn, battle, "Cavalry Only")
    _troop(conn, reconcile_side, _source(conn), 5_000.0, branch="cavalry")

    for _ in range(2):
        reports, counts = _load(conn)
        _, unfillable = classify_sides(reports, conn)
        write_unfillable(conn, unfillable, counts)

    rows = _missing_rows(conn)
    notes = sorted(str(r[2]) for r in rows)
    assert len(rows) == 2, rows
    assert notes[0].startswith("extract: ")
    assert notes[1].startswith("reconcile: ")


def test_a_side_with_a_usable_report_is_modellable_and_not_logged(conn: Connection) -> None:
    battle = _battle(conn, "Fine")
    side = _side(conn, battle)
    _troop(conn, side, _source(conn), 20_000.0)

    reports, counts = _load(conn)
    modellable, unfillable = classify_sides(reports, conn)
    write_unfillable(conn, unfillable, counts)

    assert side in modellable
    assert _missing_rows(conn) == []


# ─── Derived fields ──────────────────────────────────────────────────────────


def test_a_regime_marker_in_the_context_reaches_the_loaded_report(conn: Connection) -> None:
    battle = _battle(conn, "Labelled", "0331-10-01 BC")
    side = _side(conn, battle)
    _troop(conn, side, _source(conn), 250_000.0, context="250,000 according to Herodotus")

    reports, _ = _load(conn)

    assert reports[0].claim_regime == "ancient_claim"


def test_a_derived_source_repeating_a_wikipedia_figure_shares_its_lineage(
    conn: Connection,
) -> None:
    # Wikidata and DBpedia are built from Wikipedia, so two rows carrying the
    # same figure are one claim. Counting them as two independent observations
    # narrows the interval by sqrt(2) on evidence that has not grown.
    battle = _battle(conn, "Lineages")
    side = _side(conn, battle)
    wiki = _source(conn, "wikipedia_infobox", "https://en.wikipedia.org/wiki/Y")
    derived = _source(conn, "wikidata", "https://www.wikidata.org/wiki/Q1")
    _troop(conn, side, wiki, 30_000.0)
    _troop(conn, side, derived, 30_000.0)

    reports, _ = _load(conn)
    troops = [r for r in reports if r.quantity == "troops"]

    assert len(troops) == 2
    assert len({r.lineage_id for r in troops}) == 1


def test_the_counts_account_for_every_troop_row_read(conn: Connection) -> None:
    battle = _battle(conn, "Accounting")
    side = _side(conn, battle)
    source = _source(conn)
    _troop(conn, side, source, 10_000.0)
    _troop(conn, side, source, 500.0, branch="cavalry")
    _troop(conn, side, source, 12.0, branch="naval")
    _troop(conn, side, source, 0.0)

    _, counts = _load(conn)

    accounted = (
        counts.troop_reports_used
        + counts.excluded_non_total_branch
        + counts.excluded_naval
        + counts.excluded_non_positive
        + counts.excluded_implausible_small
        + counts.excluded_non_personnel
        + counts.excluded_date_context
        + counts.excluded_component_of_stated_total
        + counts.excluded_population_scope
        + counts.excluded_superseded_by_modern
    )
    assert accounted == 4


# ─── Plausibility filter ─────────────────────────────────────────────────────


def test_junk_rows_are_excluded_and_counted_by_reason(conn: Connection) -> None:
    battle = _battle(conn, "Junk Field")
    side = _side(conn, battle)
    source = _source(conn)
    _troop(conn, side, source, 30_000.0, context="30,000")
    _troop(conn, side, source, 42.0, context="a plain figure")
    _troop(conn, side, source, 140.0, context="140 helicopters")
    _troop(conn, side, source, 13.0, context="13 March:")

    reports, counts = _load(conn)
    troops = [r for r in reports if r.quantity == "troops"]

    assert [r.reported_value for r in troops] == [30_000.0]
    assert counts.excluded_implausible_small == 1
    assert counts.excluded_non_personnel == 1
    assert counts.excluded_date_context == 1


def test_components_of_a_stated_total_are_excluded_and_counted(conn: Connection) -> None:
    battle = _battle(conn, "Cannae Fixture")
    rome = _side(conn, battle, "Rome")
    other = _side(conn, battle, "Carthage")
    source = _source(conn)
    _troop(conn, rome, source, 86_400.0, context="Total: 86,400")
    _troop(conn, rome, source, 25_000.0, context="25,000 Romans")
    _troop(conn, rome, source, 30_000.0, context="30,000 allies")
    _troop(conn, other, source, 21_000.0, context="21,000 Gauls")

    reports, counts = _load(conn)
    kept = sorted(r.reported_value for r in reports if r.quantity == "troops")

    assert kept == [21_000.0, 86_400.0]
    assert counts.excluded_component_of_stated_total == 2


def test_a_side_whose_only_rows_are_junk_is_logged_unfillable(conn: Connection) -> None:
    battle = _battle(conn, "Only Ships")
    side = _side(conn, battle)
    source = _source(conn)
    _troop(conn, side, source, 500.0, context="500 transports")

    reports, _ = _load(conn)
    modellable, unfillable = classify_sides(reports, conn)

    assert side not in modellable
    assert [(u.side_id, u.reason) for u in unfillable] == [
        (side, "only implausible or non-personnel figures")
    ]
