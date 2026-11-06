"""
Review fixes for the resolve side of the infobox-wikilink signal.

A link is taken only when it is the mention's own name, and a failed lookup is
counted and warned about instead of silently changing identities between runs.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from structlog.testing import capture_logs

from pipeline.crawlers.fetcher import FetchResult
from pipeline.resolvers import Candidate
from pipeline.resolvers.wikilinks import WIKILINK_METHOD, WikilinkIndex
from tests.unit.test_resolve_wikilinks import (
    THOMAS,
    TITLES,
    WRONG,
    _mention,
    _resolve,
    _Resolved,
    _Source,
    _SparqlFetcher,
)

AUGUSTUS = Candidate(qid="Q9", label="Augustus II the Strong", birth_year=1670, death_year=1733)
LANNES = Candidate(qid="Q10", label="Jean Lannes", birth_year=1769, death_year=1809)


def _taken(run: _Resolved, name: str) -> bool:
    return run.of(name).method == WIKILINK_METHOD


@pytest.mark.parametrize(
    "name",
    [
        "Jean Lannes (son of Augustus II the Strong)",
        "Jean Lannes, aide to Augustus II the Strong",
        "Jean Lannes (succeeded Augustus II the Strong)",
    ],
)
def test_a_link_to_a_relative_or_superior_does_not_become_the_mentions_identity(name: str) -> None:
    mentions = [_mention(name, ("Augustus II the Strong",), year=1800)]
    with capture_logs() as logs:
        run = _resolve(
            mentions,
            _Source([LANNES], [AUGUSTUS, LANNES]),
            {"Augustus II the Strong": "Q9"},
        )
    assert not _taken(run, name)
    event = [e for e in logs if e["event"] == "resolve_wikilinks"][0]
    assert event["ignored_name_mismatch"] == 1
    assert event["linked"] == 0


def test_a_shortened_display_name_still_takes_its_link() -> None:
    title = "Arthur Wellesley, 1st Duke of Wellington"
    wellesley = Candidate(qid="Q129", label=title, birth_year=1769, death_year=1852)
    mentions = [_mention("Wellington", (title,), year=1815)]
    run = _resolve(mentions, _Source([], [wellesley]), {title: "Q129"})
    assert _taken(run, "Wellington")
    assert run.of("Wellington").qid == "Q129"


def test_a_longer_mention_takes_a_link_the_person_is_known_by() -> None:
    napoleon = Candidate(
        qid="Q517",
        label="Napoleon",
        aliases=("Napoleon Bonaparte",),
        birth_year=1769,
        death_year=1821,
    )
    mentions = [_mention("Napoleon Bonaparte", ("Napoleon",), year=1805)]
    run = _resolve(mentions, _Source([], [napoleon]), {"Napoleon": "Q517"})
    assert _taken(run, "Napoleon Bonaparte")


def test_initials_may_stand_for_the_linked_names() -> None:
    mentions = [_mention("G. H. Thomas", ("George Henry Thomas",))]
    run = _resolve(mentions, _Source([WRONG], [THOMAS]), TITLES)
    assert _taken(run, "G. H. Thomas")


class _FailingFetcher(_SparqlFetcher):
    async def fetch_raw(self, url: str, *, params: Any = None, headers: Any = None) -> FetchResult:
        return FetchResult(url=url, status=503, error="http_503", attempts=4)

    async def fetch(self, url: str, *, params: Any = None, headers: Any = None) -> FetchResult:
        return FetchResult(url=url, status=500, error="http_500", attempts=4)


def test_titles_whose_lookup_failed_are_counted_and_warned_about(tmp_path: Path) -> None:
    index = WikilinkIndex(["Title A", "Title B"], tmp_path / "c.json")
    with capture_logs() as logs:
        asyncio.run(index.resolve(_FailingFetcher(), batch_size=25, max_page_fetches=10))
    assert index.failed_titles == ["Title A", "Title B"]
    events = [e for e in logs if e["event"] == "wikilink_fetch_failures"]
    assert events
    assert events[0]["titles"] == 2
    assert events[0]["log_level"] == "warning"


def test_a_page_failure_is_counted_but_a_404_is_not(tmp_path: Path) -> None:
    class _Mixed(_SparqlFetcher):
        async def fetch(self, url: str, *, params: Any = None, headers: Any = None) -> FetchResult:
            if url.endswith("Flaky"):
                return FetchResult(url=url, status=500, error="http_500", attempts=4)
            return FetchResult(url=url, status=404, error="http_404", attempts=1)

    index = WikilinkIndex(["Flaky", "Gone"], tmp_path / "c.json")
    asyncio.run(index.resolve(_Mixed(), batch_size=25, max_page_fetches=10))
    assert index.failed_titles == ["Flaky"]
    assert index.qids == {"Gone": None}


def test_a_clean_run_has_no_failures_and_no_warning(tmp_path: Path) -> None:
    index = WikilinkIndex(["Dead Link"], tmp_path / "c.json")
    with capture_logs() as logs:
        asyncio.run(index.resolve(_SparqlFetcher(), batch_size=25, max_page_fetches=10))
    assert index.failed_titles == []
    assert not [e for e in logs if e["event"] == "wikilink_fetch_failures"]


def test_mentions_whose_title_was_never_answered_are_counted_separately() -> None:
    mentions = [_mention("G. H. Thomas", ("George Henry Thomas",))]
    with capture_logs() as logs:
        _resolve(mentions, _Source([WRONG], [THOMAS]), {})
    event = [e for e in logs if e["event"] == "resolve_wikilinks"][0]
    assert event["unanswered_title"] == 1
    assert event["ignored_unresolved_title"] == 0
