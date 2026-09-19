"""
The crawl stage against live Wikipedia.

Every systematic bug this project has found came from contact with something
real rather than from reading code: the naval mislabelling (handover.md 4.2),
the robots.txt rule that silently disabled every SPARQL query (4.7), the
alias matches thrown away after the query found them (14.1). In each case a
fixture suite agreed with the code because both had been written from the
same assumption. These tests exist so the crawl path has an answer to that,
and they assert the things a recorded fixture cannot: that Wikipedia's real
HTML still parses, that its robots.txt still permits ``/wiki/``, that its
rate limiting is survivable at the configured spacing.

**They are opt-in and skipped by default.** They make real requests to a
third party, so they must not run in CI, must not run on every ``pytest``,
and must not be something a contributor triggers by accident:

    GENERAL_WAR_LIVE_CRAWL=1 python -m pytest tests/integration/test_crawl_live.py -v

Politeness is a correctness requirement here, not a courtesy. The fetcher is
built from the shipped ``agents/crawl.yaml`` rather than from test defaults,
so what runs is the real configuration: if the spec's rate limit is too fast
for Wikipedia, that is a failure these tests should show rather than dodge.
The whole module fetches ten articles at 2s spacing, which is under a minute.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx
import pytest

from pipeline.config import load_agent_spec
from pipeline.crawlers import dbpedia, wikipedia
from pipeline.crawlers.fetcher import DEFAULT_TIMEOUT_S, Fetcher, FetchResult, RateLimiter
from pipeline.extractors import parse_article_infobox

_ENV_FLAG = "GENERAL_WAR_LIVE_CRAWL"

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.environ.get(_ENV_FLAG),
        reason=f"live network test; set {_ENV_FLAG}=1 to run",
    ),
]

# Ten battles chosen so a parse failure is legible rather than mysterious:
# two BC (the date path that cannot round-trip through Python), three naval
# (the covariate 4.2 was getting wrong), a siege, and two disambiguated
# titles whose parentheses the URL rules must survive, one of them
# non-ASCII. Every title here was checked against the live API rather than
# recalled: an earlier draft of this list invented Battle_of_Hałycz, which
# 404s, and handover.md 14.6 records the same mistake made with a Q-id.
LIVE_BATTLES: tuple[str, ...] = (
    "https://en.wikipedia.org/wiki/Battle_of_Cannae",
    "https://en.wikipedia.org/wiki/Battle_of_Actium",
    "https://en.wikipedia.org/wiki/Battle_of_Trafalgar",
    "https://en.wikipedia.org/wiki/Battle_of_Midway",
    "https://en.wikipedia.org/wiki/Battle_of_Myeongnyang",
    "https://en.wikipedia.org/wiki/Siege_of_Alesia",
    "https://en.wikipedia.org/wiki/Battle_of_Austerlitz",
    "https://en.wikipedia.org/wiki/Battle_of_Gettysburg",
    "https://en.wikipedia.org/wiki/Battle_of_%C5%81%C3%B3d%C5%BA_(1914)",
    "https://en.wikipedia.org/wiki/Sack_of_Rome_(410)",
)

# Naval engagements among the fetched set. Wikipedia serves all three through
# the generic {{Infobox military conflict}} template, which is why 4.2
# mislabelled them.
NAVAL_BATTLES: frozenset[str] = frozenset(
    {"Battle of Trafalgar", "Battle of Midway", "Battle of Myeongnyang"}
)


def live_params() -> dict[str, Any]:
    """Read the shipped crawl spec's parameters.

    Using the real spec rather than test defaults is the point of this
    module: a rate limit that is too fast for Wikipedia should fail here.

    Returns:
        The ``params`` mapping from ``agents/crawl.yaml``.
    """
    return dict(load_agent_spec("crawl").get("params") or {})


def make_live_fetcher(client: httpx.AsyncClient) -> Fetcher:
    """Build a fetcher from the shipped spec.

    Args:
        client: An open async client.

    Returns:
        A fetcher configured exactly as a real crawl would configure it.
    """
    params = live_params()
    wikimedia = float(params["rate_limit_wikipedia"])
    return Fetcher(
        client,
        limiter=RateLimiter(
            float(params["rate_limit_other"]),
            {
                "wikipedia.org": wikimedia,
                "wikimedia.org": wikimedia,
                "wikidata.org": wikimedia,
            },
        ),
        user_agent=str(params["user_agent"]),
        max_retries=int(params["max_retries"]),
        backoff_base=float(params["backoff_base"]),
        respect_robots=bool(params["respect_robots"]),
    )


async def _fetch_all(urls: tuple[str, ...]) -> list[FetchResult]:
    """Fetch several articles through one rate-limited fetcher.

    Args:
        urls: The article URLs.

    Returns:
        One result per URL, in order.
    """
    timeout = httpx.Timeout(DEFAULT_TIMEOUT_S)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        fetcher = make_live_fetcher(client)
        return [await fetcher.fetch(url) for url in urls]


@pytest.fixture(scope="module")
def live_results() -> list[FetchResult]:
    """Fetch the battle articles once and share them across the module.

    Fetching per test would multiply the load on Wikipedia by the number of
    tests for no extra evidence.

    Returns:
        One result per URL in :data:`LIVE_BATTLES`.
    """
    return asyncio.run(_fetch_all(LIVE_BATTLES))


def test_every_battle_article_is_fetched(live_results: list[FetchResult]) -> None:
    """The configured rate limit survives ten consecutive Wikipedia requests.

    This is the assertion handover.md 5.2 asked for: a smoke test at 1.5s
    spacing drew a 429, and 1.0 was the configured value. If the spec is
    still too fast, this fails with the status rather than corrupting a real
    crawl months later.
    """
    failures = [(r.url, r.status, r.error) for r in live_results if not r.ok]
    assert not failures, f"live fetch failed for {failures}"

    # A 429 the Retry-After path recovered from still costs extra attempts,
    # and is worth seeing even when the fetch ultimately succeeded.
    throttled = [(r.url, r.attempts) for r in live_results if r.attempts > 1]
    assert not throttled, (
        f"these needed retries at the configured rate limit, so the spec is "
        f"probably still too fast: {throttled}"
    )


def test_robots_still_permits_the_article_namespace() -> None:
    """en.wikipedia.org may say no at any time, and 4.7 is what that costs.

    A robots rule silently turned every SPARQL query in the project into an
    empty result. The same failure on ``/wiki/`` would empty the entire
    corpus, so it is checked against the live file rather than assumed.
    """

    async def check() -> tuple[bool, bool]:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            fetcher = make_live_fetcher(client)
            robots = fetcher.robots
            assert robots is not None, "the shipped spec should respect robots"
            article = await robots.allows(LIVE_BATTLES[0])
            # Wikipedia disallows /w/index.php with a query. If that ever
            # stops being true, this assertion is the thing to revisit.
            edit = await robots.allows(
                "https://en.wikipedia.org/w/index.php?title=X&action=edit"
            )
            return article, edit

    allowed, edit_allowed = asyncio.run(check())

    assert allowed, "robots.txt now disallows /wiki/: the crawl corpus would be empty"
    assert not edit_allowed, "robots.txt no longer disallows the edit path; rules have changed"


def test_live_articles_still_parse_into_infoboxes(live_results: list[FetchResult]) -> None:
    """Wikipedia's real HTML parses, which is what 4.2 proved fixtures cannot.

    The fixtures encoded the same assumption as the parser, so both agreed
    and both were wrong. This asserts against whatever Wikipedia serves today.
    """
    parsed: list[tuple[str, Any]] = []
    unparsed: list[str] = []

    for result in live_results:
        assert result.text is not None
        title = wikipedia.title_from_url(result.url)
        extraction = parse_article_infobox(
            result.text,
            is_html=True,
            source_ref=result.url,
            source_title=title,
        )
        if extraction is None:
            unparsed.append(title)
        else:
            parsed.append((title, extraction))

    assert not unparsed, f"no infobox found in live articles: {unparsed}"

    # Two sides is the shape the whole model assumes. An infobox that parses
    # but yields no sides is the failure that would quietly halve the corpus.
    sideless = [title for title, e in parsed if len(e.sides) < 2]
    assert not sideless, f"live infobox parsed but yielded fewer than two sides: {sideless}"


def test_no_naval_battle_is_labelled_a_land_engagement(
    live_results: list[FetchResult],
) -> None:
    """The regression test for handover.md 4.2, asserted against live HTML.

    Wikipedia uses the generic ``{{Infobox military conflict}}`` template for
    essentially every battle including naval ones, so a parser that reads
    "field" from it mislabels Trafalgar, Midway and Myeongnyang -- three of
    the articles fetched here. The generic template must leave battle_type
    unset for classify to infer, and this is the only test that can prove it
    against the templates Wikipedia actually serves.
    """
    mislabelled: list[tuple[str, str]] = []

    for result in live_results:
        assert result.text is not None
        title = wikipedia.title_from_url(result.url)
        extraction = parse_article_infobox(
            result.text, is_html=True, source_ref=result.url, source_title=title
        )
        if extraction is None:
            continue
        battle_type = extraction.facts.battle_type
        if title in NAVAL_BATTLES and battle_type is not None and battle_type != "naval":
            mislabelled.append((title, battle_type))

    assert not mislabelled, (
        f"a naval battle was given a non-naval battle_type from the infobox: "
        f"{mislabelled}. This is handover.md 4.2 recurring."
    )


def test_dbpedia_still_serves_the_articles_we_fetch() -> None:
    """DBpedia is a real dependency of extract, and it goes down.

    The extract stage treats an absent DBpedia twin as a missing source and
    carries on, which is right, but it means an outage is invisible. Two
    resources are enough to tell "DBpedia is down" from "this battle has no
    twin".
    """

    async def fetch() -> list[FetchResult]:
        async with httpx.AsyncClient(timeout=httpx.Timeout(DEFAULT_TIMEOUT_S)) as client:
            fetcher = make_live_fetcher(client)
            return [await dbpedia.fetch_resource(fetcher, url) for url in LIVE_BATTLES[:2]]

    results = asyncio.run(fetch())

    assert any(r.ok for r in results), (
        f"no DBpedia resource fetched: {[(r.url, r.status, r.error) for r in results]}"
    )
