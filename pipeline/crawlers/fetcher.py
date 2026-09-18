"""
Polite HTTP fetching for the crawl stage.

The crawl stage pulls thousands of pages from a handful of hosts, most of
them Wikimedia's. Rate limiting, robots.txt and a truthful User-Agent are
therefore correctness requirements rather than courtesies: without them the
crawl gets the project blocked and the run produces nothing.

Everything here follows the project's split between bugs and data. A network
failure, a 404 or a robots.txt disallow is a *property of the crawl*: it is
returned as a :class:`FetchResult` carrying the reason, logged, and recorded
in ``crawl_log``. A missing or nonsensical configuration value is a *bug* and
raises immediately.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx
import structlog

__all__ = [
    "DEFAULT_TIMEOUT_S",
    "FetchResult",
    "Fetcher",
    "RateLimiter",
    "RobotsPolicy",
    "content_hash",
    "host_of",
]

logger = structlog.get_logger()

# Long enough for a slow Wikimedia response, short enough that one stuck
# request cannot hold up a crawl of thousands of pages.
DEFAULT_TIMEOUT_S = 30.0

# Statuses worth another attempt: the server is overloaded or rate limiting,
# not telling us the resource is wrong.
_RETRYABLE_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})

SleepFn = Callable[[float], Awaitable[None]]
ClockFn = Callable[[], float]


def host_of(url: str) -> str:
    """Return the lowercased host of a URL.

    Args:
        url: An absolute URL.

    Returns:
        The host, or an empty string when the URL has none.
    """
    return (urlsplit(url).hostname or "").lower()


def content_hash(content: str | bytes) -> str:
    """Hash fetched content so a re-crawl can tell changed pages from stale ones.

    Args:
        content: The response body, as text or bytes.

    Returns:
        A hex SHA-256 digest.
    """
    data = content.encode("utf-8") if isinstance(content, str) else content
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class FetchResult:
    """The outcome of one fetch, successful or not.

    Attributes:
        url: The URL fetched.
        status: HTTP status, or None when no response was received.
        text: Response body, present only on success.
        content_hash: SHA-256 of the body, present only on success.
        error: Why the fetch did not yield a usable body, else None.
        attempts: Round trips made, including retries. Zero when the fetch
            was refused before any request went out.
        fetched_at: When the attempt finished, timezone-aware UTC.
    """

    url: str
    status: int | None = None
    text: str | None = None
    content_hash: str | None = None
    error: str | None = None
    attempts: int = 0
    fetched_at: datetime = datetime(1970, 1, 1, tzinfo=UTC)

    @property
    def ok(self) -> bool:
        """Whether the fetch returned a usable body."""
        return self.error is None and self.text is not None


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime.

    Returns:
        The current UTC time.
    """
    return datetime.now(UTC)


class RateLimiter:
    """Enforces a minimum interval between requests to the same host.

    The interval is per host rather than global, so a slow citation domain
    does not throttle Wikipedia and vice versa. The clock and sleep function
    are injected so tests can assert the pacing without real delay.
    """

    def __init__(
        self,
        default_interval: float,
        intervals: Mapping[str, float] | None = None,
        *,
        clock: ClockFn = time.monotonic,
        sleep: SleepFn = asyncio.sleep,
    ) -> None:
        """Create a limiter.

        Args:
            default_interval: Seconds between requests to any host without a
                specific entry.
            intervals: Per-domain overrides. A key matches a host exactly or
                as a domain suffix, so ``wikipedia.org`` covers
                ``en.wikipedia.org``. The longest matching key wins.
            clock: Monotonic time source.
            sleep: Awaitable sleep.

        Raises:
            ValueError: If any interval is negative.
        """
        negative = [v for v in (default_interval, *(intervals or {}).values()) if v < 0]
        if negative:
            raise ValueError(f"Rate limit intervals must be non-negative, got {negative}")

        self._default = default_interval
        self._intervals = {k.lower(): v for k, v in (intervals or {}).items()}
        self._clock = clock
        self._sleep = sleep
        self._next_free: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def interval_for(self, url: str) -> float:
        """Return the interval that applies to a URL's host.

        Args:
            url: An absolute URL.

        Returns:
            Seconds to leave between requests to that host.
        """
        host = host_of(url)
        best_interval = self._default
        best_len = -1
        for suffix, interval in self._intervals.items():
            if (host == suffix or host.endswith(f".{suffix}")) and len(suffix) > best_len:
                best_interval, best_len = interval, len(suffix)
        return best_interval

    async def acquire(self, url: str) -> float:
        """Wait until it is polite to request this URL's host.

        Args:
            url: The URL about to be fetched.

        Returns:
            The seconds waited, which tests and logs use to verify pacing.
        """
        host = host_of(url)
        lock = self._locks.setdefault(host, asyncio.Lock())
        async with lock:
            interval = self.interval_for(url)
            now = self._clock()
            earliest = self._next_free.get(host, now)
            wait = max(0.0, earliest - now)
            if wait > 0:
                await self._sleep(wait)
            self._next_free[host] = max(now, earliest) + interval
            return wait


@dataclass(frozen=True)
class _HostRules:
    """Cached robots.txt decision for one host.

    Attributes:
        parser: Parsed rules, or None when the host published none.
        default_allow: What to do in the absence of parsed rules.
    """

    parser: RobotFileParser | None
    default_allow: bool

    def allows(self, url: str, user_agent: str) -> bool:
        """Report whether a URL may be fetched under these rules.

        Args:
            url: The URL in question.
            user_agent: The agent string to match rules against.

        Returns:
            True if the fetch is permitted.
        """
        if self.parser is None:
            return self.default_allow
        return self.parser.can_fetch(user_agent, url)


class RobotsPolicy:
    """Fetches, caches and applies robots.txt per host.

    An unreachable robots.txt is treated as a disallow. That is the
    conservative reading: if a host is unwell we should not hammer it with
    the very crawl its robots.txt might have forbidden. A 4xx is treated as
    permission, since that is the standard "no rules published" signal.
    """

    def __init__(
        self,
        user_agent: str,
        fetch: Callable[[str], Awaitable[FetchResult]],
    ) -> None:
        """Create a policy.

        Args:
            user_agent: The agent string rules are evaluated against.
            fetch: Raw fetch callable used to retrieve robots.txt. It must not
                itself consult this policy, or the two would recurse.
        """
        self._user_agent = user_agent
        self._fetch = fetch
        self._cache: dict[str, _HostRules] = {}

    async def allows(self, url: str) -> bool:
        """Report whether robots.txt permits fetching a URL.

        Args:
            url: The absolute URL about to be fetched.

        Returns:
            True if the fetch is permitted.
        """
        host = host_of(url)
        rules = self._cache.get(host)
        if rules is None:
            rules = await self._load(url)
            self._cache[host] = rules
        return rules.allows(url, self._user_agent)

    async def _load(self, url: str) -> _HostRules:
        """Fetch and parse a host's robots.txt.

        Args:
            url: Any URL on the host.

        Returns:
            The rules to apply to that host from now on.
        """
        parts = urlsplit(url)
        robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
        result = await self._fetch(robots_url)
        host = host_of(url)

        if result.ok and result.text is not None:
            parser = RobotFileParser()
            parser.parse(result.text.splitlines())
            logger.debug("robots_loaded", host=host)
            return _HostRules(parser=parser, default_allow=True)

        if result.status is not None and 400 <= result.status < 500:
            logger.debug("robots_absent", host=host, status=result.status)
            return _HostRules(parser=None, default_allow=True)

        logger.warning(
            "robots_unavailable_denying", host=host, status=result.status, error=result.error
        )
        return _HostRules(parser=None, default_allow=False)


class Fetcher:
    """Rate-limited, robots-aware, retrying HTTP GET.

    Never raises for network conditions: every outcome comes back as a
    :class:`FetchResult` so the caller can log it to ``crawl_log`` and carry
    on with the next URL.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        limiter: RateLimiter,
        user_agent: str,
        max_retries: int = 3,
        backoff_base: float = 2.0,
        respect_robots: bool = True,
        sleep: SleepFn = asyncio.sleep,
    ) -> None:
        """Create a fetcher.

        Args:
            client: An open httpx async client.
            limiter: Per-host rate limiter.
            user_agent: Sent on every request and used for robots matching.
            max_retries: Retries after the first attempt.
            backoff_base: Seconds for the first backoff; doubles each retry.
            respect_robots: Whether to consult robots.txt before fetching.
            sleep: Awaitable sleep, injected for tests.

        Raises:
            ValueError: If the user agent is blank or the retry settings are
                nonsensical. Both are configuration bugs, not data.
        """
        if not user_agent.strip():
            raise ValueError("user_agent is required: an anonymous crawler gets blocked")
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}")
        if backoff_base <= 0:
            raise ValueError(f"backoff_base must be > 0, got {backoff_base}")

        self._client = client
        self._limiter = limiter
        self._user_agent = user_agent
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._sleep = sleep
        self._robots = RobotsPolicy(user_agent, self.fetch_raw) if respect_robots else None

    @property
    def robots(self) -> RobotsPolicy | None:
        """The robots policy in force, or None when robots are not respected."""
        return self._robots

    async def fetch(
        self,
        url: str,
        *,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> FetchResult:
        """Fetch a URL, honouring robots.txt, rate limits and retries.

        Args:
            url: Absolute URL to fetch.
            params: Query parameters.
            headers: Extra request headers.

        Returns:
            The result, successful or not. A robots disallow comes back with
            ``attempts == 0`` and an explanatory error.
        """
        if self._robots is not None and not await self._robots.allows(url):
            logger.info("fetch_skipped_robots", url=url)
            return FetchResult(url=url, error="robots_disallowed", attempts=0, fetched_at=utc_now())
        return await self.fetch_raw(url, params=params, headers=headers)

    async def fetch_raw(
        self,
        url: str,
        *,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> FetchResult:
        """Fetch a URL without consulting robots.txt.

        Used for robots.txt itself, which cannot be gated on its own rules.

        Args:
            url: Absolute URL to fetch.
            params: Query parameters.
            headers: Extra request headers.

        Returns:
            The result, successful or not.
        """
        request_headers = {"User-Agent": self._user_agent}
        if headers:
            request_headers.update(headers)

        attempts = 0
        status: int | None = None
        error: str | None = None

        for attempt in range(self._max_retries + 1):
            await self._limiter.acquire(url)
            attempts += 1
            try:
                response = await self._client.get(
                    url, params=dict(params) if params else None, headers=request_headers
                )
            except httpx.HTTPError as exc:
                status = None
                error = f"{type(exc).__name__}: {exc}"
            else:
                status = response.status_code
                if status not in _RETRYABLE_STATUSES:
                    if response.is_success:
                        body = response.text
                        return FetchResult(
                            url=url,
                            status=status,
                            text=body,
                            content_hash=content_hash(body),
                            attempts=attempts,
                            fetched_at=utc_now(),
                        )
                    return FetchResult(
                        url=url,
                        status=status,
                        error=f"HTTP {status}",
                        attempts=attempts,
                        fetched_at=utc_now(),
                    )
                error = f"HTTP {status}"

            if attempt < self._max_retries:
                delay = self._backoff_base * (2.0**attempt)
                logger.warning(
                    "fetch_retry", url=url, attempt=attempts, error=error, backoff_s=delay
                )
                await self._sleep(delay)

        logger.warning("fetch_failed", url=url, attempts=attempts, status=status, error=error)
        return FetchResult(
            url=url,
            status=status,
            error=error or "fetch failed",
            attempts=attempts,
            fetched_at=utc_now(),
        )
