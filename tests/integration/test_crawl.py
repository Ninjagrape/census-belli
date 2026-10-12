"""
Integration test for crawl_log writing.

The unit suite proves the crawler produces the right rows; it cannot prove
those rows fit the table. Column names, the nullable http_status a network
failure needs, and the battle_id foreign key are all facts about PostgreSQL,
so they are checked here against a real database.

Without a reachable database every test in this module skips rather than
fails, keeping the suite runnable on a machine with no Docker.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import Engine, text

from pipeline.config import load_agent_spec
from pipeline.crawlers.fetcher import FetchResult, content_hash, utc_now
from pipeline.crawlers.log import CrawlLogEntry, SqlCrawlLog, entry_from_result
from pipeline.db import (
    DatabaseConfigError,
    apply_schema,
    database_url,
    get_engine,
    schema_is_applied,
)
from pipeline.quality import QualityRunner, Severity

ARTICLE_URL = "https://en.wikipedia.org/wiki/Battle_of_Actium"


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
    """Provide an engine against an applied schema."""
    eng = get_engine()
    with eng.connect() as conn:
        applied = schema_is_applied(conn)
    if not applied:
        apply_schema(eng)
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


def test_crawl_log_accepts_successes_and_failures(conn: Any, clean_log: None) -> None:
    """Both halves of a crawl's record fit the table as written."""
    body = "<html>Actium</html>"
    success = FetchResult(
        url=ARTICLE_URL,
        status=200,
        text=body,
        content_hash=content_hash(body),
        attempts=1,
        fetched_at=utc_now(),
    )
    refused = FetchResult(
        url=f"{ARTICLE_URL}_refused", error="robots_disallowed", attempts=0, fetched_at=utc_now()
    )
    broken = FetchResult(
        url=f"{ARTICLE_URL}_broken",
        status=503,
        error="HTTP 503",
        attempts=4,
        fetched_at=utc_now(),
    )

    # batch_size 1 forces a write per row, so a failure surfaces here rather
    # than at flush time.
    writer = SqlCrawlLog(conn, batch_size=1)
    for result in (success, refused, broken):
        writer.record(entry_from_result(result))
    writer.flush()

    rows = conn.execute(
        text(
            "SELECT url, http_status, content_hash, errors, battle_id "
            "FROM crawl_log WHERE url LIKE :prefix ORDER BY crawl_id"
        ),
        {"prefix": f"{ARTICLE_URL}%"},
    ).all()

    assert writer.written == 3
    assert len(rows) == 3
    assert rows[0].http_status == 200
    assert rows[0].content_hash == content_hash(body)
    assert rows[0].errors is None
    assert rows[1].http_status is None
    assert rows[1].errors == "robots_disallowed"
    assert rows[2].http_status == 503
    assert rows[2].content_hash is None
    assert all(row.battle_id is None for row in rows)


def test_batched_writes_reach_the_table(conn: Any, clean_log: None) -> None:
    """Rows buffered below the batch size still land once flushed."""
    writer = SqlCrawlLog(conn, batch_size=50)
    for index in range(10):
        writer.record(
            entry_from_result(
                FetchResult(
                    url=f"https://en.wikipedia.org/wiki/Batch_{index}",
                    status=200,
                    text="x",
                    content_hash=content_hash("x"),
                    attempts=1,
                    fetched_at=utc_now(),
                )
            )
        )

    buffered = conn.execute(
        text("SELECT COUNT(*) FROM crawl_log WHERE url LIKE '%/wiki/Batch_%'")
    ).scalar()
    writer.flush()
    flushed = conn.execute(
        text("SELECT COUNT(*) FROM crawl_log WHERE url LIKE '%/wiki/Batch_%'")
    ).scalar()

    assert buffered == 0
    assert flushed == 10


# ─── Flush isolation ─────────────────────────────────────────────────────────


@pytest.fixture
def clean_log(engine: Engine) -> Iterator[None]:
    """Empty crawl_log before and after a test.

    SqlCrawlLog commits, which ends the rollback-wrapped transaction of the
    ``conn`` fixture, so rows written here outlive the test.
    """
    with engine.begin() as setup:
        setup.execute(text("DELETE FROM crawl_log"))
    yield
    with engine.begin() as teardown:
        teardown.execute(text("DELETE FROM crawl_log"))


def _entry(url: str, battle_id: int | None = None) -> CrawlLogEntry:
    """Build a log entry for a successful fetch.

    Args:
        url: The URL logged.
        battle_id: A battle foreign key; a missing id makes the insert fail.

    Returns:
        The entry.
    """
    return CrawlLogEntry(
        url=url, fetched_at=utc_now(), http_status=200, content_hash="h", battle_id=battle_id
    )


def test_a_failed_insert_does_not_abort_later_flushes(engine: Engine, clean_log: None) -> None:
    """On Postgres one failed statement poisons the transaction until rollback.

    Without a savepoint the bad row below aborts the transaction, the batch is
    dropped, and every later flush fails with "current transaction is aborted"
    while the log silently stops recording the crawl.
    """
    with engine.connect() as connection:
        writer = SqlCrawlLog(connection, batch_size=100)
        writer.record(_entry("https://example.org/before"))
        writer.record(_entry("https://example.org/bad", battle_id=987654321))
        writer.record(_entry("https://example.org/after-in-same-batch"))
        writer.flush()

        writer.record(_entry("https://example.org/next-flush"))
        writer.flush()

        urls = {row[0] for row in connection.execute(text("SELECT url FROM crawl_log"))}

    assert urls == {
        "https://example.org/before",
        "https://example.org/after-in-same-batch",
        "https://example.org/next-flush",
    }
    assert writer.written == 3


# ─── The crawl stage's quality gates ─────────────────────────────────────────

_PLAN = "crawl-plan://battle-articles?targets={t}&fetched={f}&entities={e}&floor={n}"


def _gate_results(
    connection: Any, plans: list[str], extra_urls: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Log plan rows and extra 200s, then run the crawl spec's gates.

    Args:
        connection: An open connection.
        plans: Plan-row URLs, oldest first.
        extra_urls: Further successfully fetched URLs (citations and the like).

    Returns:
        Gate results keyed by check name.
    """
    writer = SqlCrawlLog(connection)
    for url in extra_urls:
        writer.record(_entry(url))
    for url in plans:
        writer.record(CrawlLogEntry(url=url, fetched_at=utc_now(), http_status=0))
    writer.flush()
    checks = load_agent_spec("crawl")["quality_checks"]
    return {result.name: result for result in QualityRunner(connection).run_all(checks)}


def test_named_corpus_passes_the_gates_whatever_citations_were_fetched(
    engine: Engine, clean_log: None
) -> None:
    """211 of 211 articles passes; 1986 citation 200s neither help nor hurt."""
    citations = tuple(f"https://archive.org/details/c{i}" for i in range(1986))
    with engine.connect() as connection:
        results = _gate_results(
            connection, [_PLAN.format(t=211, f=211, e=211, n=0)], extra_urls=citations
        )

    assert {name: r.passed for name, r in results.items()} == {
        "min_battles_crawled": True,
        "battle_articles_fetched_ratio": True,
        "error_rate_below_threshold": True,
        "wikidata_coverage": True,
    }
    assert results["battle_articles_fetched_ratio"].actual_value == 1.0


def test_a_named_corpus_missing_a_tenth_of_its_articles_fails(
    engine: Engine, clean_log: None
) -> None:
    """The ratio gate is an error, and plentiful citations do not mask it."""
    citations = tuple(f"https://archive.org/details/c{i}" for i in range(500))
    with engine.connect() as connection:
        results = _gate_results(
            connection, [_PLAN.format(t=211, f=190, e=190, n=0)], extra_urls=citations
        )

    ratio = results["battle_articles_fetched_ratio"]
    assert ratio.passed is False
    assert ratio.severity is Severity.ERROR


def test_a_list_page_crawl_still_needs_thousands_of_battles(
    engine: Engine, clean_log: None
) -> None:
    """A run whose lists failed to parse is fully fetched but far below the floor."""
    with engine.connect() as connection:
        results = _gate_results(connection, [_PLAN.format(t=40, f=40, e=40, n=3000)])

    assert results["battle_articles_fetched_ratio"].passed is True
    assert results["min_battles_crawled"].passed is False


def test_the_latest_plan_row_wins(engine: Engine, clean_log: None) -> None:
    """Completing a corpus on a re-run supersedes the earlier partial run."""
    with engine.connect() as connection:
        results = _gate_results(
            connection,
            [
                _PLAN.format(t=211, f=100, e=0, n=0),
                _PLAN.format(t=211, f=211, e=211, n=0),
            ],
        )

    assert results["battle_articles_fetched_ratio"].passed is True


def test_wikidata_coverage_scales_with_the_corpus(engine: Engine, clean_log: None) -> None:
    """Zero entities of 211 articles fails; the old gate also failed on 1000."""
    with engine.connect() as connection:
        results = _gate_results(connection, [_PLAN.format(t=211, f=211, e=0, n=0)])

    assert results["wikidata_coverage"].passed is False
    assert results["wikidata_coverage"].actual_value == 0.0


def test_a_run_that_logged_no_plan_row_does_not_pass(engine: Engine, clean_log: None) -> None:
    """No accounting row means no evidence, not a pass."""
    with engine.connect() as connection:
        results = _gate_results(connection, [])

    assert results["battle_articles_fetched_ratio"].passed is False
    assert results["min_battles_crawled"].passed is False
