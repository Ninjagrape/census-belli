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

from pipeline.crawlers.fetcher import FetchResult, content_hash, utc_now
from pipeline.crawlers.log import SqlCrawlLog, entry_from_result
from pipeline.db import (
    DatabaseConfigError,
    apply_schema,
    database_url,
    get_engine,
    schema_is_applied,
)

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


def test_crawl_log_accepts_successes_and_failures(conn: Any) -> None:
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


def test_batched_writes_reach_the_table(conn: Any) -> None:
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
