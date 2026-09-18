"""
Resume state for the crawl stage.

A full crawl is thousands of rate-limited requests and takes hours. Losing it
to a dropped connection, a reboot or a Ctrl-C would be expensive and would
also mean re-requesting pages we have already taken from Wikimedia, which is
the opposite of polite. This module records what has been done so a re-run
picks up where the last one stopped.

The state file is the stage's own bookkeeping, not data: a corrupt or
unreadable file degrades to starting fresh with a warning rather than
aborting the run, since a wasted crawl is recoverable and a crash is not.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

__all__ = [
    "DEFAULT_STATE_PATH",
    "CrawlState",
    "load_state",
    "save_state",
]

logger = structlog.get_logger()

DEFAULT_STATE_PATH = Path("data/raw/crawl_state.json")

# Bumped when the on-disk shape changes incompatibly. An older file is
# discarded rather than misread.
STATE_VERSION = 1


def _timestamp() -> str:
    """Return an ISO-8601 UTC timestamp.

    Returns:
        The current time, e.g. ``2026-09-18T10:31:05.123456+00:00``.
    """
    return datetime.now(UTC).isoformat()


@dataclass
class CrawlState:
    """What a crawl has already accomplished.

    Attributes:
        list_pages_done: Battle-list URLs already parsed for links.
        battle_urls: Deduplicated battle article URLs discovered so far, in
            discovery order.
        fetched: Successfully fetched URL to content hash, covering articles,
            citations and Wikidata payloads alike.
        failed: URL to the last error seen, for URLs that never succeeded.
        wikidata_done: Names of SPARQL queries already run.
        started_at: When the first run began.
        updated_at: When the state was last written.
    """

    list_pages_done: list[str] = field(default_factory=list)
    battle_urls: list[str] = field(default_factory=list)
    fetched: dict[str, str] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)
    wikidata_done: list[str] = field(default_factory=list)
    started_at: str = field(default_factory=_timestamp)
    updated_at: str = field(default_factory=_timestamp)

    _seen: set[str] = field(default_factory=set, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Rebuild the membership index after construction or load."""
        self._seen = set(self.battle_urls)

    def add_battle_urls(self, urls: Iterable[str]) -> list[str]:
        """Record newly discovered battle URLs, ignoring ones already known.

        The same battle appears on several of the seed lists, so this is what
        keeps a battle from being fetched once per list that mentions it.

        Args:
            urls: Candidate URLs, already normalised by the caller.

        Returns:
            The URLs that were new, in the order given.
        """
        added: list[str] = []
        for url in urls:
            if url in self._seen:
                continue
            self._seen.add(url)
            self.battle_urls.append(url)
            added.append(url)
        return added

    def is_fetched(self, url: str) -> bool:
        """Report whether a URL has already been fetched successfully.

        Args:
            url: The URL to check.

        Returns:
            True if a previous run stored a content hash for it.
        """
        return url in self.fetched

    def mark_fetched(self, url: str, content_hash: str | None) -> None:
        """Record a successful fetch.

        Args:
            url: The URL fetched.
            content_hash: Hash of the body, used to detect change on re-crawl.
        """
        self.fetched[url] = content_hash or ""
        self.failed.pop(url, None)

    def mark_failed(self, url: str, error: str) -> None:
        """Record a failed fetch so a re-run can report or retry it.

        Args:
            url: The URL that failed.
            error: The reason, as recorded in ``crawl_log.errors``.
        """
        self.failed[url] = error

    def mark_list_page_done(self, url: str) -> None:
        """Record that a battle-list page has been parsed.

        Args:
            url: The list page URL.
        """
        if url not in self.list_pages_done:
            self.list_pages_done.append(url)

    def mark_query_done(self, name: str) -> None:
        """Record that a named SPARQL query has been run.

        Args:
            name: The query name from the seed config.
        """
        if name not in self.wikidata_done:
            self.wikidata_done.append(name)

    def pending_battles(self, limit: int | None = None) -> list[str]:
        """Return battle URLs still to fetch.

        Args:
            limit: Stop after this many URLs. Used by smoke runs.

        Returns:
            The URLs not yet fetched, in discovery order.
        """
        pending = [url for url in self.battle_urls if url not in self.fetched]
        return pending if limit is None else pending[:limit]

    def to_dict(self) -> dict[str, Any]:
        """Serialise the state for JSON.

        Returns:
            A plain dict with a version marker.
        """
        return {
            "version": STATE_VERSION,
            "started_at": self.started_at,
            "updated_at": _timestamp(),
            "list_pages_done": self.list_pages_done,
            "battle_urls": self.battle_urls,
            "fetched": self.fetched,
            "failed": self.failed,
            "wikidata_done": self.wikidata_done,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CrawlState:
        """Rebuild state from its serialised form.

        Args:
            payload: A dict as produced by :meth:`to_dict`.

        Returns:
            The reconstructed state.

        Raises:
            ValueError: If the payload is not a mapping of the expected shape.
        """
        if not isinstance(payload, dict):
            raise ValueError(f"crawl state must be a mapping, got {type(payload).__name__}")

        version = payload.get("version")
        if version != STATE_VERSION:
            raise ValueError(f"unsupported crawl state version {version!r}")

        def _str_list(key: str) -> list[str]:
            value = payload.get(key, [])
            if not isinstance(value, list):
                raise ValueError(f"{key} must be a list, got {type(value).__name__}")
            return [str(item) for item in value]

        def _str_map(key: str) -> dict[str, str]:
            value = payload.get(key, {})
            if not isinstance(value, dict):
                raise ValueError(f"{key} must be a mapping, got {type(value).__name__}")
            return {str(k): str(v) for k, v in value.items()}

        return cls(
            list_pages_done=_str_list("list_pages_done"),
            battle_urls=_str_list("battle_urls"),
            fetched=_str_map("fetched"),
            failed=_str_map("failed"),
            wikidata_done=_str_list("wikidata_done"),
            started_at=str(payload.get("started_at") or _timestamp()),
            updated_at=str(payload.get("updated_at") or _timestamp()),
        )


def load_state(path: Path | str = DEFAULT_STATE_PATH) -> CrawlState:
    """Load resume state, falling back to a fresh state.

    A missing file is the normal first-run case. An unreadable or malformed
    one is logged and discarded: re-crawling costs time, whereas refusing to
    start costs the whole run.

    Args:
        path: Where the state file lives.

    Returns:
        The loaded state, or an empty one.
    """
    state_path = Path(path)
    if not state_path.exists():
        logger.info("crawl_state_absent", path=str(state_path))
        return CrawlState()

    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        state = CrawlState.from_dict(payload)
    except (OSError, ValueError) as exc:
        logger.warning(
            "crawl_state_unreadable_starting_fresh", path=str(state_path), error=str(exc)
        )
        return CrawlState()

    logger.info(
        "crawl_state_loaded",
        path=str(state_path),
        battles_known=len(state.battle_urls),
        fetched=len(state.fetched),
        lists_done=len(state.list_pages_done),
    )
    return state


def save_state(state: CrawlState, path: Path | str = DEFAULT_STATE_PATH) -> None:
    """Write resume state atomically.

    The write goes to a temporary file in the same directory and is then
    renamed over the target, so an interrupt during the write cannot leave a
    half-written state file that the next run would have to discard.

    Args:
        state: The state to persist.
        path: Where to write it.

    Raises:
        OSError: If the directory cannot be created or written.
    """
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = state_path.with_name(f"{state_path.name}.tmp")

    payload = state.to_dict()
    state.updated_at = str(payload["updated_at"])
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_path, state_path)

    logger.debug("crawl_state_saved", path=str(state_path), fetched=len(state.fetched))
