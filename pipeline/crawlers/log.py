"""
``crawl_log`` writing.

Every fetch attempt, successful or not, becomes a row. That is what makes the
stage's own quality gates meaningful: the error-rate check is only honest if
failures are recorded as diligently as successes, and ``content_hash`` is
what lets a later re-crawl tell a changed article from an unchanged one.

The writer is an interface with two implementations so that a dry run and the
unit tests can exercise the same code path as a real run without a database.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

import structlog
from sqlalchemy import text

from pipeline.crawlers.fetcher import FetchResult

__all__ = [
    "CrawlLogEntry",
    "CrawlLogWriter",
    "InMemoryCrawlLog",
    "SqlCrawlLog",
    "entry_from_result",
]

logger = structlog.get_logger()

_INSERT_SQL = text(
    """
    INSERT INTO crawl_log (url, fetched_at, http_status, content_hash, errors, battle_id)
    VALUES (:url, :fetched_at, :http_status, :content_hash, :errors, :battle_id)
    """
)


@dataclass(frozen=True)
class CrawlLogEntry:
    """One row of ``crawl_log``.

    Attributes:
        url: The URL fetched.
        fetched_at: When the attempt finished, timezone-aware.
        http_status: HTTP status, or None when no response arrived.
        content_hash: SHA-256 of the body on success, else None.
        errors: The failure reason, else None.
        battle_id: The battle this fetch targeted. Always None during crawl:
            battles have no rows until the resolve stage assigns ids, and the
            column is a foreign key.
    """

    url: str
    fetched_at: datetime
    http_status: int | None = None
    content_hash: str | None = None
    errors: str | None = None
    battle_id: int | None = None

    def as_params(self) -> dict[str, Any]:
        """Return the row as bound parameters for the insert.

        Returns:
            A mapping matching the named parameters in the insert statement.
        """
        return {
            "url": self.url,
            "fetched_at": self.fetched_at,
            "http_status": self.http_status,
            "content_hash": self.content_hash,
            "errors": self.errors,
            "battle_id": self.battle_id,
        }


def entry_from_result(result: FetchResult) -> CrawlLogEntry:
    """Build a log entry from a fetch result.

    Args:
        result: A :class:`~pipeline.crawlers.fetcher.FetchResult`.

    Returns:
        The corresponding log entry.
    """
    return CrawlLogEntry(
        url=result.url,
        fetched_at=result.fetched_at,
        http_status=result.status,
        content_hash=result.content_hash,
        errors=result.error,
    )


class CrawlLogWriter(Protocol):
    """Sink for crawl log rows."""

    def record(self, entry: CrawlLogEntry) -> None:
        """Record one fetch attempt.

        Args:
            entry: The row to write.
        """
        ...

    def flush(self) -> None:
        """Persist anything buffered."""
        ...


class InMemoryCrawlLog:
    """Collects entries without a database, for dry runs and tests."""

    def __init__(self) -> None:
        """Create an empty log."""
        self.entries: list[CrawlLogEntry] = []

    def record(self, entry: CrawlLogEntry) -> None:
        """Record one fetch attempt.

        Args:
            entry: The row that would have been written.
        """
        self.entries.append(entry)

    def flush(self) -> None:
        """No-op; nothing is buffered elsewhere."""

    def urls(self) -> list[str]:
        """Return the URLs recorded, in order.

        Returns:
            One URL per recorded attempt.
        """
        return [entry.url for entry in self.entries]


class SqlCrawlLog:
    """Writes entries to ``crawl_log`` in batches.

    Batching keeps a crawl of thousands of pages from paying a round trip per
    fetch, while the batch stays small enough that a crash loses only the
    tail of the log rather than the run's whole record.
    """

    def __init__(self, conn: Any, batch_size: int = 50) -> None:
        """Create a writer.

        Args:
            conn: An open SQLAlchemy connection.
            batch_size: Rows to buffer before writing.

        Raises:
            ValueError: If the batch size is not positive.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self._conn = conn
        self._batch_size = batch_size
        self._pending: list[CrawlLogEntry] = []
        self.written = 0

    def record(self, entry: CrawlLogEntry) -> None:
        """Buffer one fetch attempt, writing when the batch is full.

        Args:
            entry: The row to write.
        """
        self._pending.append(entry)
        if len(self._pending) >= self._batch_size:
            self.flush()

    def flush(self) -> None:
        """Write and commit any buffered rows.

        A failure to write the log must not lose the crawl itself, so the
        error is logged and the batch dropped rather than raised: the pages
        are already on disk and the crawl can continue.
        """
        if not self._pending:
            return

        batch: Sequence[CrawlLogEntry] = tuple(self._pending)
        self._pending.clear()
        try:
            self._conn.execute(_INSERT_SQL, [entry.as_params() for entry in batch])
            self._conn.commit()
        except Exception as exc:
            logger.error("crawl_log_write_failed", rows=len(batch), error=str(exc))
            return
        self.written += len(batch)
