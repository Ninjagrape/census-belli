"""
Unit tests for the crawl stage.

Nothing here touches the network or a database. HTTP is served by
``httpx.MockTransport``, which exercises the real client, real headers and
real status handling while keeping every request inside the process, and the
crawl_log sink is the in-memory implementation.

The coroutine tests drive their own event loop through the ``sync`` decorator
below rather than relying on pytest-asyncio, so the crawl stage stays testable
in an environment where only the base test dependencies are installed.

The things worth guarding are the ones whose failure is silent: a rate
limiter that does not actually wait, a robots check that passes when
robots.txt is unreachable, a retry loop that gives up on the first attempt,
and deduplication that lets the same battle be fetched once per seed list
that mentions it. Each of those would produce a crawl that looks successful
and is not.
"""

from __future__ import annotations

import asyncio
import functools
import json
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from pipeline.crawlers import citations, dbpedia, fetcher, seeds, state, wikidata, wikipedia
from pipeline.crawlers.fetcher import Fetcher, FetchResult, RateLimiter, content_hash
from pipeline.crawlers.log import CrawlLogEntry, InMemoryCrawlLog, entry_from_result
from pipeline.stages import crawl as crawl_stage
from pipeline.stages.base import StageContext, check_conforms

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "crawl"

LIST_URL = "https://en.wikipedia.org/wiki/List_of_Roman_battles"
OVERLAP_URL = "https://en.wikipedia.org/wiki/List_of_battles_before_301"
ARTICLE_URL = "https://en.wikipedia.org/wiki/Battle_of_Cannae"
USER_AGENT = "GeneralWAR/0.1 (test; nobody@example.com)"


def sync(test: Callable[..., Coroutine[Any, Any, None]]) -> Callable[..., None]:
    """Run a coroutine test on its own event loop.

    Args:
        test: The async test function.

    Returns:
        A synchronous test pytest can collect without an async plugin.
    """

    @functools.wraps(test)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        asyncio.run(test(*args, **kwargs))

    return wrapper


def fixture(name: str) -> str:
    """Read a crawl fixture.

    Args:
        name: Filename under tests/fixtures/crawl.

    Returns:
        The file's text.
    """
    return (FIXTURES / name).read_text(encoding="utf-8")


# ─── Test doubles ────────────────────────────────────────────────────────────


class FakeClock:
    """A monotonic clock that only moves when a test moves it."""

    def __init__(self) -> None:
        """Start at zero."""
        self.now = 0.0

    def __call__(self) -> float:
        """Return the current fake time."""
        return self.now


class RecordingSleep:
    """An awaitable sleep that records its durations instead of waiting."""

    def __init__(self, clock: FakeClock | None = None) -> None:
        """Create the recorder.

        Args:
            clock: When given, the clock is advanced by each sleep.
        """
        self.calls: list[float] = []
        self._clock = clock

    async def __call__(self, seconds: float) -> None:
        """Record a sleep.

        Args:
            seconds: How long the caller wanted to wait.
        """
        self.calls.append(seconds)
        if self._clock is not None:
            self._clock.now += seconds


def make_client(handler: Any) -> httpx.AsyncClient:
    """Build an async client backed by a mock transport.

    Args:
        handler: Callable taking an httpx.Request and returning a Response.

    Returns:
        A client that never touches the network.
    """
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)


def make_fetcher(
    client: httpx.AsyncClient,
    *,
    respect_robots: bool = False,
    max_retries: int = 3,
    sleep: RecordingSleep | None = None,
    interval: float = 0.0,
) -> Fetcher:
    """Build a fetcher wired to fake time.

    Args:
        client: The mock-backed client.
        respect_robots: Whether to consult robots.txt.
        max_retries: Retries after the first attempt.
        sleep: Sleep recorder; a fresh one is made when omitted.
        interval: Rate-limit interval for every host.

    Returns:
        The fetcher.
    """
    recorder = sleep or RecordingSleep()
    return Fetcher(
        client,
        limiter=RateLimiter(interval, clock=FakeClock(), sleep=recorder),
        user_agent=USER_AGENT,
        max_retries=max_retries,
        backoff_base=2.0,
        respect_robots=respect_robots,
        sleep=recorder,
    )


# ─── List parsing and deduplication ──────────────────────────────────────────


def test_parse_battle_links_keeps_only_battle_articles() -> None:
    """Battle links are found; namespaces, indexes and externals are not."""
    links = wikipedia.parse_battle_links(fixture("battle_list.html"), LIST_URL)

    assert links == [
        "https://en.wikipedia.org/wiki/Battle_of_Cannae",
        "https://en.wikipedia.org/wiki/Siege_of_Alesia",
        "https://en.wikipedia.org/wiki/Battle_of_Nan%E2%80%99an",
        "https://en.wikipedia.org/wiki/Battle_of_Actium",
        "https://en.wikipedia.org/wiki/Sack_of_Rome_(410)",
    ]


@pytest.mark.parametrize(
    "unwanted",
    [
        "File:Battle_of_Cannae_map.svg",
        "Category:Battles_of_the_Second_Punic_War",
        "Help:Battles",
        "Special:RandomInCategory/Battles",
        "Talk:Battle_of_Cannae",
        "List_of_battles_before_301",
        "Timeline_of_Roman_battles",
        "Battle_of_the_Navbox",
        "Battle_of_the_Footnote",
        "Battle_of_Nowhere",
        "britannica.com",
    ],
)
def test_parse_battle_links_excludes(unwanted: str) -> None:
    """Namespaces, index pages, page furniture and red links stay out."""
    links = wikipedia.parse_battle_links(fixture("battle_list.html"), LIST_URL)
    assert all(unwanted not in link for link in links)


def test_fragments_and_absolute_links_normalise() -> None:
    """A fragment is dropped and an absolute link survives intact."""
    assert (
        wikipedia.normalise_url("/wiki/Battle_of_Actium#Aftermath", LIST_URL)
        == "https://en.wikipedia.org/wiki/Battle_of_Actium"
    )
    assert wikipedia.normalise_url("/w/index.php?title=X&redlink=1", LIST_URL) is None
    assert wikipedia.normalise_url("https://example.com/wiki/Battle", LIST_URL) is None
    assert wikipedia.normalise_url("#Notes", LIST_URL) is None


def test_same_battle_on_two_lists_is_deduplicated() -> None:
    """The overlap between seed lists collapses to one URL per battle."""
    first = wikipedia.parse_battle_links(fixture("battle_list.html"), LIST_URL)
    second = wikipedia.parse_battle_links(fixture("battle_list_overlap.html"), OVERLAP_URL)

    combined = wikipedia.dedupe_urls([*first, *second])

    assert len(combined) == len(first) + 1
    assert combined[-1] == "https://en.wikipedia.org/wiki/Battle_of_Zama"


def test_percent_encoded_and_decoded_titles_share_one_key() -> None:
    """Two spellings of the same title deduplicate to one article."""
    encoded = "https://en.wikipedia.org/wiki/Battle_of_Nan%E2%80%99an"
    decoded = "https://en.wikipedia.org/wiki/Battle_of_Nan’an"

    assert wikipedia.dedupe_key(encoded) == wikipedia.dedupe_key(decoded)
    assert wikipedia.dedupe_urls([encoded, decoded]) == [encoded]


def test_article_filename_is_readable_and_unique() -> None:
    """Filenames stay legible but cannot collide across distinct URLs."""
    name = wikipedia.article_filename(ARTICLE_URL)
    other = wikipedia.article_filename("https://en.wikipedia.org/wiki/Battle_of_Cannae_(disambig)")

    assert name.startswith("Battle_of_Cannae-")
    assert name.endswith(".html")
    assert name != other
    assert not set(name) & set('<>:"/\\|?*')


# ─── Rate limiting ───────────────────────────────────────────────────────────


@sync
async def test_rate_limiter_waits_between_requests_to_one_host() -> None:
    """The second request to a host waits a full interval."""
    clock = FakeClock()
    sleeper = RecordingSleep(clock)
    limiter = RateLimiter(2.0, {"wikipedia.org": 1.0}, clock=clock, sleep=sleeper)

    first = await limiter.acquire("https://en.wikipedia.org/wiki/A")
    second = await limiter.acquire("https://en.wikipedia.org/wiki/B")

    assert first == 0.0
    assert second == pytest.approx(1.0)
    assert sleeper.calls == [pytest.approx(1.0)]


@sync
async def test_rate_limits_are_per_domain() -> None:
    """A different host is not made to wait behind Wikipedia."""
    clock = FakeClock()
    sleeper = RecordingSleep(clock)
    limiter = RateLimiter(2.0, {"wikipedia.org": 1.0}, clock=clock, sleep=sleeper)

    await limiter.acquire("https://en.wikipedia.org/wiki/A")
    waited = await limiter.acquire("https://doi.org/10.1017")

    assert waited == 0.0
    assert sleeper.calls == []


def test_interval_matches_on_domain_boundaries() -> None:
    """A suffix rule covers subdomains without covering lookalike hosts."""
    limiter = RateLimiter(2.0, {"wikipedia.org": 1.0})

    assert limiter.interval_for("https://en.wikipedia.org/wiki/A") == 1.0
    assert limiter.interval_for("https://wikipedia.org/wiki/A") == 1.0
    assert limiter.interval_for("https://notwikipedia.org/wiki/A") == 2.0
    assert limiter.interval_for("https://query.wikidata.org/sparql") == 2.0


def test_negative_interval_is_a_configuration_bug() -> None:
    """A nonsensical rate limit raises rather than crawling at full speed."""
    with pytest.raises(ValueError, match="non-negative"):
        RateLimiter(-1.0)


@sync
async def test_fetcher_paces_requests_through_the_limiter() -> None:
    """Every fetch goes through the limiter, not just the first."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="ok")

    clock = FakeClock()
    sleeper = RecordingSleep(clock)
    async with make_client(handler) as client:
        target = Fetcher(
            client,
            limiter=RateLimiter(1.0, clock=clock, sleep=sleeper),
            user_agent=USER_AGENT,
            respect_robots=False,
            sleep=sleeper,
        )
        await target.fetch("https://en.wikipedia.org/wiki/A")
        await target.fetch("https://en.wikipedia.org/wiki/B")

    assert sleeper.calls == [pytest.approx(1.0)]


# ─── robots.txt ──────────────────────────────────────────────────────────────


def robots_handler(body: str, status: int = 200) -> Any:
    """Build a handler serving a robots.txt and a trivial page.

    Args:
        body: The robots.txt content.
        status: Status to serve for robots.txt.

    Returns:
        A handler recording every path it was asked for on ``.seen``.
    """
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/robots.txt":
            return httpx.Response(status, text=body)
        return httpx.Response(200, text="<html>page</html>")

    handler.seen = seen  # type: ignore[attr-defined]
    return handler


@sync
async def test_robots_disallow_blocks_the_request_entirely() -> None:
    """A disallowed path is never requested, and says why."""
    handler = robots_handler("User-agent: *\nDisallow: /wiki/Special:\n")
    async with make_client(handler) as client:
        target = make_fetcher(client, respect_robots=True)
        result = await target.fetch("https://en.wikipedia.org/wiki/Special:Random")

    assert result.ok is False
    assert result.error == "robots_disallowed"
    assert result.attempts == 0
    assert handler.seen == ["/robots.txt"]


@sync
async def test_robots_allow_permits_the_request() -> None:
    """A path outside the disallow rules is fetched normally."""
    handler = robots_handler("User-agent: *\nDisallow: /wiki/Special:\n")
    async with make_client(handler) as client:
        target = make_fetcher(client, respect_robots=True)
        result = await target.fetch(ARTICLE_URL)

    assert result.ok
    assert result.status == 200
    assert handler.seen == ["/robots.txt", "/wiki/Battle_of_Cannae"]


@sync
async def test_robots_is_fetched_once_per_host() -> None:
    """The policy is cached; a crawl does not refetch robots.txt per page."""
    handler = robots_handler("User-agent: *\nAllow: /\n")
    async with make_client(handler) as client:
        target = make_fetcher(client, respect_robots=True)
        await target.fetch(ARTICLE_URL)
        await target.fetch("https://en.wikipedia.org/wiki/Battle_of_Zama")

    assert handler.seen.count("/robots.txt") == 1


@sync
async def test_missing_robots_is_treated_as_permission() -> None:
    """A 404 robots.txt means no rules were published, so fetching is fine."""
    handler = robots_handler("", status=404)
    async with make_client(handler) as client:
        target = make_fetcher(client, respect_robots=True)
        result = await target.fetch(ARTICLE_URL)

    assert result.ok


@sync
async def test_unreachable_robots_is_treated_as_refusal() -> None:
    """A 5xx robots.txt denies: an unwell host should not be hammered."""
    handler = robots_handler("", status=503)
    async with make_client(handler) as client:
        target = make_fetcher(client, respect_robots=True, max_retries=0)
        result = await target.fetch(ARTICLE_URL)

    assert result.ok is False
    assert result.error == "robots_disallowed"
    assert "/wiki/Battle_of_Cannae" not in handler.seen


@sync
async def test_robots_can_be_turned_off_for_a_self_hosted_mirror() -> None:
    """With respect_robots false no robots.txt is fetched at all."""
    handler = robots_handler("User-agent: *\nDisallow: /\n")
    async with make_client(handler) as client:
        target = make_fetcher(client, respect_robots=False)
        result = await target.fetch(ARTICLE_URL)

    assert result.ok
    assert "/robots.txt" not in handler.seen


# ─── Retry and backoff ───────────────────────────────────────────────────────


@sync
async def test_transient_failures_are_retried_with_exponential_backoff() -> None:
    """Two 503s then a 200: three attempts, backing off 2s then 4s."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, text="<html>ok</html>")

    sleeper = RecordingSleep()
    async with make_client(handler) as client:
        target = make_fetcher(client, sleep=sleeper)
        result = await target.fetch(ARTICLE_URL)

    assert result.ok
    assert result.attempts == 3
    assert sleeper.calls == [2.0, 4.0]


@sync
async def test_retries_stop_at_the_configured_maximum() -> None:
    """A permanently failing host costs four attempts and three backoffs."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="broken")

    sleeper = RecordingSleep()
    async with make_client(handler) as client:
        target = make_fetcher(client, sleep=sleeper, max_retries=3)
        result = await target.fetch(ARTICLE_URL)

    assert result.ok is False
    assert result.status == 500
    assert result.error == "HTTP 500"
    assert result.attempts == 4
    assert sleeper.calls == [2.0, 4.0, 8.0]


@sync
async def test_a_429_waits_as_long_as_the_server_asked() -> None:
    """Wikipedia's Retry-After beats the formula, which would answer 29s with 2s.

    This is the defect handover.md 5.2 recorded on the crawl side and 14.4 on
    the LLM side: blind doubling spends every attempt inside a window the
    limit was never going to lift in.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="slow down", headers={"Retry-After": "29"})

    sleeper = RecordingSleep()
    async with make_client(handler) as client:
        target = make_fetcher(client, sleep=sleeper, max_retries=2)
        result = await target.fetch(ARTICLE_URL)

    assert result.status == 429
    assert sleeper.calls == [29.0, 29.0]


@sync
async def test_a_retry_after_shorter_than_the_backoff_does_not_shorten_it() -> None:
    """The header raises the wait, never lowers it: the backoff is still a floor."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="busy", headers={"Retry-After": "1"})

    sleeper = RecordingSleep()
    async with make_client(handler) as client:
        target = make_fetcher(client, sleep=sleeper, max_retries=2)
        await target.fetch(ARTICLE_URL)

    assert sleeper.calls == [2.0, 4.0]


@sync
async def test_an_absurd_retry_after_is_capped_not_slept_through() -> None:
    """A crawl of thousands of pages cannot block an hour on one of them."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="come back tomorrow", headers={"Retry-After": "86400"})

    sleeper = RecordingSleep()
    async with make_client(handler) as client:
        target = make_fetcher(client, sleep=sleeper, max_retries=1)
        await target.fetch(ARTICLE_URL)

    assert sleeper.calls == [fetcher.MAX_SERVER_BACKOFF_S]


def test_retry_after_reads_both_forms_rfc_9110_allows() -> None:
    """Wikimedia sends seconds from the API limiter and a date from the CDN."""
    now = datetime(2026, 10, 21, 7, 28, 0, tzinfo=UTC)

    def response(headers: dict[str, str]) -> httpx.Response:
        return httpx.Response(429, headers=headers)

    assert fetcher.retry_after_seconds(response({"Retry-After": "29"})) == 29.0
    assert (
        fetcher.retry_after_seconds(
            response({"Retry-After": "Wed, 21 Oct 2026 07:28:30 GMT"}), now=now
        )
        == 30.0
    )
    # A date already past means retry now, which is an answer and not the
    # same as no header at all.
    assert (
        fetcher.retry_after_seconds(
            response({"Retry-After": "Wed, 21 Oct 2026 07:27:00 GMT"}), now=now
        )
        == 0.0
    )
    assert fetcher.retry_after_seconds(response({})) is None
    assert fetcher.retry_after_seconds(response({"Retry-After": "soon"})) is None


@sync
async def test_client_errors_are_not_retried() -> None:
    """A 404 is an answer, not a fault; retrying it would waste the budget."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404, text="missing")

    sleeper = RecordingSleep()
    async with make_client(handler) as client:
        target = make_fetcher(client, sleep=sleeper)
        result = await target.fetch(ARTICLE_URL)

    assert calls["n"] == 1
    assert result.attempts == 1
    assert result.status == 404
    assert result.error == "HTTP 404"
    assert sleeper.calls == []


@sync
async def test_network_failure_is_reported_not_raised() -> None:
    """A dropped connection is data: it comes back as a result, not an error."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    sleeper = RecordingSleep()
    async with make_client(handler) as client:
        target = make_fetcher(client, sleep=sleeper, max_retries=2)
        result = await target.fetch(ARTICLE_URL)

    assert result.ok is False
    assert result.status is None
    assert result.error is not None
    assert "ConnectError" in result.error
    assert result.attempts == 3
    assert sleeper.calls == [2.0, 4.0]


@pytest.mark.parametrize("bad", [{"user_agent": "  "}, {"max_retries": -1}, {"backoff_base": 0}])
@sync
async def test_nonsensical_fetcher_settings_raise(bad: dict[str, Any]) -> None:
    """Misconfiguration is a bug and fails loudly at construction."""
    settings: dict[str, Any] = {"user_agent": USER_AGENT, "max_retries": 3, "backoff_base": 2.0}
    settings.update(bad)

    async with make_client(lambda request: httpx.Response(200)) as client:
        with pytest.raises(ValueError):
            Fetcher(client, limiter=RateLimiter(0.0), respect_robots=False, **settings)


# ─── Content hashing ─────────────────────────────────────────────────────────


def test_content_hash_detects_change() -> None:
    """Identical bodies hash alike; a single character changes the digest."""
    assert content_hash("<html>a</html>") == content_hash(b"<html>a</html>")
    assert content_hash("<html>a</html>") != content_hash("<html>b</html>")
    assert len(content_hash("x")) == 64


@sync
async def test_successful_fetch_carries_the_body_hash() -> None:
    """A re-crawl can compare hashes because the fetch records one."""
    body = "<html>Cannae</html>"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body)

    async with make_client(handler) as client:
        result = await make_fetcher(client).fetch(ARTICLE_URL)

    assert result.content_hash == content_hash(body)


@sync
async def test_failed_fetch_has_no_hash() -> None:
    """A failure must not look like unchanged content on the next run."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    async with make_client(handler) as client:
        result = await make_fetcher(client).fetch(ARTICLE_URL)

    assert result.content_hash is None


def test_log_entry_mirrors_the_fetch_result() -> None:
    """Every crawl_log row carries status, hash and error as fetched."""
    ok = FetchResult(
        url=ARTICLE_URL,
        status=200,
        text="body",
        content_hash=content_hash("body"),
        attempts=1,
        fetched_at=fetcher.utc_now(),
    )
    failed = FetchResult(
        url=ARTICLE_URL, status=500, error="HTTP 500", attempts=4, fetched_at=fetcher.utc_now()
    )

    assert entry_from_result(ok).content_hash == content_hash("body")
    assert entry_from_result(ok).errors is None
    assert entry_from_result(failed).http_status == 500
    assert entry_from_result(failed).errors == "HTTP 500"
    assert entry_from_result(failed).content_hash is None


def test_log_entry_params_match_the_insert_columns() -> None:
    """The bound parameters name exactly the columns crawl_log expects."""
    entry = CrawlLogEntry(url=ARTICLE_URL, fetched_at=fetcher.utc_now(), http_status=200)

    assert set(entry.as_params()) == {
        "url",
        "fetched_at",
        "http_status",
        "content_hash",
        "errors",
        "battle_id",
    }


# ─── Resume state ────────────────────────────────────────────────────────────


def test_state_round_trips(tmp_path: Path) -> None:
    """What was saved is what loads, including discovery order."""
    path = tmp_path / "crawl_state.json"
    original = state.CrawlState()
    original.add_battle_urls([ARTICLE_URL, "https://en.wikipedia.org/wiki/Battle_of_Zama"])
    original.mark_list_page_done(LIST_URL)
    original.mark_fetched(ARTICLE_URL, "abc123")
    original.mark_failed("https://en.wikipedia.org/wiki/Battle_of_Zama", "HTTP 500")
    original.mark_query_done("all_battles")

    state.save_state(original, path)
    loaded = state.load_state(path)

    assert loaded.battle_urls == original.battle_urls
    assert loaded.fetched == {ARTICLE_URL: "abc123"}
    assert loaded.failed == {"https://en.wikipedia.org/wiki/Battle_of_Zama": "HTTP 500"}
    assert loaded.list_pages_done == [LIST_URL]
    assert loaded.wikidata_done == ["all_battles"]


def test_state_resume_skips_completed_work(tmp_path: Path) -> None:
    """Only unfetched battles are pending after a reload."""
    path = tmp_path / "crawl_state.json"
    first = state.CrawlState()
    first.add_battle_urls([ARTICLE_URL, "https://en.wikipedia.org/wiki/Battle_of_Zama"])
    first.mark_fetched(ARTICLE_URL, "abc123")
    state.save_state(first, path)

    resumed = state.load_state(path)

    assert resumed.is_fetched(ARTICLE_URL)
    assert resumed.pending_battles() == ["https://en.wikipedia.org/wiki/Battle_of_Zama"]
    assert resumed.pending_battles(limit=0) == []


def test_state_dedupes_urls_across_list_pages() -> None:
    """Adding a known URL again adds nothing."""
    tracked = state.CrawlState()

    assert tracked.add_battle_urls([ARTICLE_URL]) == [ARTICLE_URL]
    assert tracked.add_battle_urls([ARTICLE_URL]) == []
    assert tracked.battle_urls == [ARTICLE_URL]


def test_a_successful_refetch_clears_a_previous_failure() -> None:
    """A URL that failed then succeeded is not reported as still failing."""
    tracked = state.CrawlState()
    tracked.mark_failed(ARTICLE_URL, "HTTP 503")
    tracked.mark_fetched(ARTICLE_URL, "abc123")

    assert tracked.failed == {}
    assert tracked.is_fetched(ARTICLE_URL)


def test_corrupt_state_starts_fresh_rather_than_aborting(tmp_path: Path) -> None:
    """A half-written state file costs a re-crawl, not the run."""
    path = tmp_path / "crawl_state.json"
    path.write_text('{"version": 1, "battle_urls": [', encoding="utf-8")

    loaded = state.load_state(path)

    assert loaded.battle_urls == []
    assert loaded.fetched == {}


def test_state_of_an_unknown_version_is_discarded(tmp_path: Path) -> None:
    """An older layout is not misread as the current one."""
    path = tmp_path / "crawl_state.json"
    path.write_text(json.dumps({"version": 99, "battle_urls": ["x"]}), encoding="utf-8")

    assert state.load_state(path).battle_urls == []


def test_missing_state_file_is_the_normal_first_run(tmp_path: Path) -> None:
    """No state file means nothing has been crawled yet."""
    assert state.load_state(tmp_path / "absent.json").battle_urls == []


def test_save_is_atomic_and_leaves_no_temporary_file(tmp_path: Path) -> None:
    """The write is renamed into place, so no partial file survives."""
    path = tmp_path / "nested" / "crawl_state.json"
    state.save_state(state.CrawlState(), path)

    assert path.exists()
    assert list(path.parent.glob("*.tmp")) == []
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == state.STATE_VERSION


# ─── Citations ───────────────────────────────────────────────────────────────


def test_citation_urls_come_from_the_reference_apparatus() -> None:
    """External reference links are extracted; internal links are not."""
    urls = citations.extract_citation_urls(fixture("battle_article.html"), ARTICLE_URL)

    assert "https://www.jstor.org/stable/4436063" in urls
    assert "https://doi.org/10.1017/S0075435800000000" in urls
    assert "https://archive.org/details/cannae" in urls
    assert "https://books.google.com/books?id=cannae" in urls
    assert all("/wiki/Polybius" not in url for url in urls)
    assert all(not url.startswith("mailto:") for url in urls)
    assert len(urls) == len(set(urls))


def test_allow_list_admits_subdomains_but_not_lookalikes() -> None:
    """Matching is on domain boundaries, so a crafted host is refused."""
    allow = ["jstor.org", "doi.org", "archive.org"]

    assert citations.domain_allowed("https://www.jstor.org/stable/1", allow)
    assert citations.domain_allowed("https://jstor.org/stable/1", allow)
    assert not citations.domain_allowed("http://jstor.org.attacker.test/stable/1", allow)
    assert not citations.domain_allowed("https://example.com/notes", allow)
    assert not citations.domain_allowed("not a url", allow)


def test_citations_are_split_into_fetchable_and_skipped() -> None:
    """Off-list URLs are recorded as skipped rather than silently dropped."""
    urls = citations.extract_citation_urls(fixture("battle_article.html"), ARTICLE_URL)
    split = citations.split_by_allow_list(urls, ["jstor.org", "doi.org", "archive.org"])

    assert "https://www.jstor.org/stable/4436063" in split.allowed
    assert "http://jstor.org.attacker.test/stable/4436063" in split.skipped
    assert "https://example.com/notes" in split.skipped
    assert "https://books.google.com/books?id=cannae" in split.skipped
    assert len(split.allowed) + len(split.skipped) == len(urls)


def test_citation_records_are_keyed_by_battle(tmp_path: Path) -> None:
    """One JSONL file per battle, as the stage's output contract says."""
    path = citations.write_citation_records(
        tmp_path, ARTICLE_URL, [{"url": "https://doi.org/10.1", "http_status": 200}]
    )

    assert path.name.startswith("Battle_of_Cannae-")
    assert path.suffix == ".jsonl"
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["http_status"] == 200


# ─── Wikidata ────────────────────────────────────────────────────────────────


def test_sparql_bindings_flatten_to_rows() -> None:
    """Result values are lifted out of the SPARQL envelope."""
    rows = wikidata.parse_sparql_bindings(fixture("sparql_battles.json"))

    assert len(rows) == 4
    assert rows[0]["battleLabel"] == "Battle of Cannae"
    assert rows[0]["date"] == "-0216-08-02T00:00:00Z"


def test_entity_ids_are_distinct_and_item_shaped() -> None:
    """Repeated items collapse and a property URI is not mistaken for one."""
    rows = wikidata.parse_sparql_bindings(fixture("sparql_battles.json"))

    assert wikidata.entity_ids(rows) == ["Q184408", "Q170545"]


def test_malformed_sparql_payload_yields_no_rows() -> None:
    """A bad response from the endpoint is data, not a crash."""
    assert wikidata.parse_sparql_bindings("not json") == []
    assert wikidata.parse_sparql_bindings('{"results": {}}') == []
    assert wikidata.parse_sparql_bindings("[]") == []


def test_entity_data_url_rejects_a_non_item() -> None:
    """A malformed id means an upstream parsing bug, so it raises."""
    assert wikidata.entity_data_url("Q184408").endswith("Special:EntityData/Q184408.json")
    with pytest.raises(ValueError, match="Not a Wikidata item id"):
        wikidata.entity_data_url("P585")


@sync
async def test_sparql_query_is_sent_as_a_json_request() -> None:
    """The query goes out with the JSON format and Accept the service wants."""
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["accept"] = request.headers.get("accept")
        seen["agent"] = request.headers.get("user-agent")
        return httpx.Response(200, text=fixture("sparql_battles.json"))

    async with make_client(handler) as client:
        result = await wikidata.run_query(make_fetcher(client), "SELECT ?battle WHERE {}")

    assert result.ok
    assert "format=json" in seen["url"]
    assert seen["accept"] == "application/sparql-results+json"
    assert seen["agent"] == USER_AGENT


@sync
async def test_empty_sparql_query_is_a_config_bug() -> None:
    """An empty query in the seed file raises instead of being sent."""
    async with make_client(lambda request: httpx.Response(200)) as client:
        with pytest.raises(ValueError, match="empty"):
            await wikidata.run_query(make_fetcher(client), "   ")


def test_dbpedia_url_derives_from_the_article_title() -> None:
    """DBpedia resources are keyed by the English article title."""
    assert dbpedia.resource_json_url("Battle of Cannae") == (
        "https://dbpedia.org/data/Battle_of_Cannae.json"
    )
    with pytest.raises(ValueError):
        dbpedia.resource_json_url("  ")


# ─── Seed config ─────────────────────────────────────────────────────────────


def test_project_seed_config_loads() -> None:
    """The real config/sources_seed.yaml is valid for the crawl stage."""
    loaded = seeds.load_seeds()

    assert len(loaded.battle_lists) >= 10
    assert all(url.startswith("https://") for url in loaded.battle_lists)
    assert "all_battles" in loaded.wikidata_queries


@pytest.mark.parametrize(
    "body",
    [
        "wikipedia_battle_lists: []",
        "wikidata_queries: {}",
        "wikipedia_battle_lists:\n  - not-a-url",
        "wikipedia_battle_lists:\n  - https://ok.example\nwikidata_queries:\n  bad: '  '",
        "- just\n- a\n- list",
    ],
)
def test_malformed_seed_config_raises(tmp_path: Path, body: str) -> None:
    """A broken seed file is a bug in the run and fails before any fetching."""
    path = tmp_path / "seed.yaml"
    path.write_text(body, encoding="utf-8")

    with pytest.raises(seeds.SeedConfigError):
        seeds.load_seeds(path)


def test_missing_seed_config_raises(tmp_path: Path) -> None:
    """An absent seed file is a configuration bug, not empty data."""
    with pytest.raises(seeds.SeedConfigError, match="not found"):
        seeds.load_seeds(tmp_path / "absent.yaml")


# ─── Stage wiring ────────────────────────────────────────────────────────────


def crawl_spec(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    """Build a crawl spec pointed at a temporary workspace.

    Args:
        tmp_path: The test's temporary directory.
        **overrides: Params to replace.

    Returns:
        A spec in the shape agents/crawl.yaml produces.
    """
    params: dict[str, Any] = {
        "rate_limit_wikipedia": 0.0,
        "rate_limit_other": 0.0,
        "max_retries": 1,
        "backoff_base": 2.0,
        "respect_robots": True,
        "user_agent": USER_AGENT,
        "citation_depth": 1,
        "citation_allow_domains": ["jstor.org", "doi.org", "archive.org"],
        "fetch_dbpedia": False,
        "max_wikidata_entities": 2,
        "output_root": str(tmp_path / "raw"),
        "state_file": str(tmp_path / "raw" / "crawl_state.json"),
        "seed_path": str(tmp_path / "seed.yaml"),
    }
    params.update(overrides)
    return {"stage": "crawl", "description": "test", "params": params}


def write_seed(tmp_path: Path) -> None:
    """Write a one-list seed config into the test workspace.

    Args:
        tmp_path: The test's temporary directory.
    """
    (tmp_path / "seed.yaml").write_text(
        "wikipedia_battle_lists:\n"
        f"  - {LIST_URL}\n"
        "wikidata_queries:\n"
        "  all_battles: |\n"
        "    SELECT ?battle WHERE {}\n",
        encoding="utf-8",
    )


def crawl_handler() -> Any:
    """Build a handler serving the fixture corpus.

    Returns:
        A handler recording requested URLs on ``.seen``.
    """
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        seen.append(url)
        path = request.url.path

        if path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /w/\n")
        if url.startswith(LIST_URL):
            return httpx.Response(200, text=fixture("battle_list.html"))
        if request.url.host == "query.wikidata.org":
            return httpx.Response(200, text=fixture("sparql_battles.json"))
        if "Special:EntityData" in path:
            return httpx.Response(200, text='{"entities": {}}')
        if request.url.host == "en.wikipedia.org":
            return httpx.Response(200, text=fixture("battle_article.html"))
        if request.url.host in ("www.jstor.org", "doi.org", "archive.org"):
            return httpx.Response(200, text="citation body")
        return httpx.Response(404, text="not served")

    handler.seen = seen  # type: ignore[attr-defined]
    return handler


def test_stage_module_satisfies_the_runner_contract() -> None:
    """The orchestrator's import-time contract check passes."""
    check_conforms(crawl_stage)


def test_missing_param_is_a_bug_not_a_default(tmp_path: Path) -> None:
    """The stage refuses to invent a rate limit or a user agent."""
    spec = crawl_spec(tmp_path)
    del spec["params"]["rate_limit_wikipedia"]

    with pytest.raises(crawl_stage.CrawlConfigError, match="rate_limit_wikipedia"):
        crawl_stage.CrawlParams.from_spec(spec)


@pytest.mark.parametrize(
    "override, message",
    [
        ({"citation_allow_domains": "jstor.org"}, "list of domain strings"),
        ({"user_agent": "   "}, "user_agent"),
        ({"rate_limit_other": "soon"}, "must be a number"),
    ],
)
def test_misstated_params_raise(tmp_path: Path, override: dict[str, Any], message: str) -> None:
    """A param of the wrong shape fails before any request goes out."""
    with pytest.raises(crawl_stage.CrawlConfigError, match=message):
        crawl_stage.CrawlParams.from_spec(crawl_spec(tmp_path, **override))


def test_params_map_wikimedia_hosts_to_the_wikipedia_rate(tmp_path: Path) -> None:
    """The two spec rate limits reach the right hosts."""
    params = crawl_stage.CrawlParams.from_spec(
        crawl_spec(tmp_path, rate_limit_wikipedia=1.0, rate_limit_other=2.0)
    )
    limiter = params.build_limiter()

    assert limiter.interval_for("https://en.wikipedia.org/wiki/A") == 1.0
    assert limiter.interval_for("https://query.wikidata.org/sparql") == 1.0
    assert limiter.interval_for("https://doi.org/10.1") == 2.0


def test_dry_run_touches_neither_disk_nor_database(tmp_path: Path) -> None:
    """A dry run plans the work and writes nothing."""
    write_seed(tmp_path)

    crawl_stage.run(crawl_spec(tmp_path), StageContext(dry_run=True))

    assert not (tmp_path / "raw").exists()


@sync
async def test_crawl_writes_articles_citations_and_log(tmp_path: Path) -> None:
    """One pass over a seed list produces the stage's declared outputs."""
    write_seed(tmp_path)
    params = crawl_stage.CrawlParams.from_spec(crawl_spec(tmp_path))
    params.paths.ensure()
    loaded = seeds.load_seeds(params.seed_path)
    sink = InMemoryCrawlLog()
    handler = crawl_handler()

    async with make_client(handler) as client:
        summary = await crawl_stage.crawl(
            params, loaded, state.CrawlState(), sink, client=client, limit=2
        )

    assert summary.list_pages == 1
    assert summary.battles_discovered == 5
    assert summary.battles_fetched == 2
    # Both fixture articles cite the same works: the three allowed URLs are
    # fetched once between them, while each article records its own three
    # off-list citations as skipped.
    assert summary.citations_fetched == 3
    assert summary.citations_skipped == 6
    assert summary.wikidata_queries == 1
    assert summary.wikidata_entities == 2
    assert summary.failures == 0

    articles = sorted(p.name for p in params.paths.battles_html.glob("*.html"))
    assert len(articles) == 2
    assert articles[0].startswith("Battle_of_Cannae-")

    citation_files = list(params.paths.citations.glob("*.jsonl"))
    assert len(citation_files) == 2
    records = [
        json.loads(line)
        for line in citation_files[0].read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert any(record.get("skipped") == "domain_not_allowed" for record in records)
    assert any(record.get("http_status") == 200 for record in records)

    assert (params.paths.wikidata / "query-all_battles.json").exists()
    assert (params.paths.wikidata / "Q184408.json").exists()
    assert params.paths.state_file.exists()

    logged = sink.urls()
    assert ARTICLE_URL in logged
    assert all(not url.endswith("robots.txt") for url in logged)
    assert all(entry.content_hash for entry in sink.entries)


@sync
async def test_an_interrupted_crawl_resumes_instead_of_refetching(tmp_path: Path) -> None:
    """The second run skips everything the first one completed."""
    write_seed(tmp_path)
    params = crawl_stage.CrawlParams.from_spec(crawl_spec(tmp_path))
    params.paths.ensure()
    loaded = seeds.load_seeds(params.seed_path)
    handler = crawl_handler()

    async with make_client(handler) as client:
        await crawl_stage.crawl(
            params, loaded, state.load_state(params.paths.state_file), InMemoryCrawlLog(),
            client=client, limit=1,
        )
        after_first = len(handler.seen)

        resumed = state.load_state(params.paths.state_file)
        second = await crawl_stage.crawl(
            params, loaded, resumed, InMemoryCrawlLog(), client=client, limit=1
        )

    assert resumed.list_pages_done == [LIST_URL]
    assert second.list_pages == 0
    assert second.wikidata_queries == 0
    assert second.battles_fetched == 1
    assert LIST_URL not in handler.seen[after_first:]


@sync
async def test_a_failing_article_is_logged_and_the_crawl_continues(tmp_path: Path) -> None:
    """One dead article does not end the run; it becomes a crawl_log row."""
    write_seed(tmp_path)
    params = crawl_stage.CrawlParams.from_spec(crawl_spec(tmp_path, max_retries=0))
    params.paths.ensure()
    loaded = seeds.load_seeds(params.seed_path)
    sink = InMemoryCrawlLog()

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        if url.startswith(LIST_URL):
            return httpx.Response(200, text=fixture("battle_list.html"))
        if url == ARTICLE_URL:
            return httpx.Response(500, text="broken")
        if request.url.host == "query.wikidata.org":
            return httpx.Response(200, text='{"results": {"bindings": []}}')
        return httpx.Response(200, text=fixture("battle_article.html"))

    async with make_client(handler) as client:
        tracked = state.CrawlState()
        summary = await crawl_stage.crawl(params, loaded, tracked, sink, client=client, limit=2)

    assert summary.failures >= 1
    assert summary.battles_fetched == 1
    assert tracked.failed[ARTICLE_URL] == "HTTP 500"

    failures = [entry for entry in sink.entries if entry.errors is not None]
    assert any(entry.url == ARTICLE_URL and entry.http_status == 500 for entry in failures)
    assert not (params.paths.battles_html / wikipedia.article_filename(ARTICLE_URL)).exists()


@sync
async def test_robots_refusal_is_recorded_with_no_status(tmp_path: Path) -> None:
    """A page robots.txt forbids is logged as an attempt with no status."""
    write_seed(tmp_path)
    params = crawl_stage.CrawlParams.from_spec(crawl_spec(tmp_path))
    params.paths.ensure()
    loaded = seeds.load_seeds(params.seed_path)
    sink = InMemoryCrawlLog()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /wiki/\n")
        return httpx.Response(200, text=fixture("battle_list.html"))

    async with make_client(handler) as client:
        summary = await crawl_stage.crawl(
            params, loaded, state.CrawlState(), sink, client=client, limit=1
        )

    assert summary.list_pages == 0
    assert summary.failures >= 1
    assert sink.entries[0].http_status is None
    assert sink.entries[0].errors == "robots_disallowed"
