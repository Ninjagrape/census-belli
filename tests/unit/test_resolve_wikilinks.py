"""
Resolve side of the infobox-wikilink signal.

A commander the infobox wikilinks is identified by the link, not by the name
the editor typed. These tests drive ``resolve_groups`` with stub sources (no
network, no database) and pin the precedence: manual overrides, then the
wikilink, then name matching exactly as before.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from structlog.testing import capture_logs

from pipeline.config import load_agent_spec
from pipeline.crawlers.fetcher import FetchResult
from pipeline.overrides import OverrideReport, load_overrides
from pipeline.resolvers import (
    Candidate,
    Decision,
    Mention,
    MentionGroup,
    ResolveCounts,
    candidate_keys,
    group_mentions,
)
from pipeline.resolvers.disambiguate import prompt_parts
from pipeline.resolvers.overrides import ResolveOverrides
from pipeline.resolvers.records import METHODS, BattleContext
from pipeline.resolvers.wikilinks import (
    WIKILINK_CONFIDENCE,
    WIKILINK_METHOD,
    WikilinkIndex,
    WikilinkLinker,
    build_sitelink_query,
    fetch_redirect_qids,
    fetch_sitelink_qids,
)
from pipeline.stages.base import StageContext
from pipeline.stages.resolve import (
    LazyService,
    _build_candidate_source,
    _build_identities,
    resolve_groups,
)
from tests.unit.test_resolve import _StubService
from tests.unit.test_resolve_overrides import _link as _override_link

BATTLE = "Battle of Nashville"


def _mention(
    name: str,
    links: tuple[str, ...] = (),
    *,
    year: int | None = 1864,
    battle: str = BATTLE,
    side: str = "Union",
) -> Mention:
    slug = battle.replace(" ", "_")
    return Mention(
        battle_slug=slug,
        battle_name=battle,
        side_label=side,
        name=name,
        context=BattleContext(slug=slug, name=battle, year=year, date_text=str(year)),
        link_titles=links,
    )


class _Source:
    """Serves candidates by folded name, and entities by Q-id."""

    def __init__(self, by_name: list[Candidate], by_qid: list[Candidate] | None = None) -> None:
        self.by_name = by_name
        self.by_qid = {c.qid: c for c in by_qid or []}

    def prefetch(self, groups: Any) -> None:
        return None

    def search(self, group: MentionGroup) -> list[Candidate]:
        keys = set(group.keys)
        return [c for c in self.by_name if keys & set(candidate_keys(c))]

    def non_person_kind(self, group: MentionGroup) -> str | None:
        return None

    def candidate_by_qid(self, qid: str) -> Candidate | None:
        return self.by_qid.get(qid)


class _Resolved:
    def __init__(self, groups: list[MentionGroup], decisions: dict[str, Decision]) -> None:
        self.groups = groups
        self.decisions = decisions

    def of(self, name: str, battle: str | None = None) -> Decision:
        for group in self.groups:
            for mention in group.mentions:
                if mention.name == name and (battle is None or mention.battle_name == battle):
                    return self.decisions[group.key]
        raise AssertionError(f"no group holds {name!r}")


def _resolve(
    mentions: list[Mention],
    source: _Source,
    title_qids: dict[str, str | None] | None = None,
    *,
    service: _StubService | None = None,
    overrides: ResolveOverrides | None = None,
) -> _Resolved:
    system, template, schema = prompt_parts(load_agent_spec("resolve"))
    stub = service or _StubService({"match": "NEW_ENTITY", "confidence": 0.5})
    groups = group_mentions(mentions)
    decisions = resolve_groups(
        groups,
        source,
        LazyService(lambda: stub),  # type: ignore[arg-type,return-value]
        threshold=85.0,
        margin=6.0,
        counts=ResolveCounts(),
        system=system,
        template=template,
        schema=schema,
        overrides=overrides,
        wikilinks=None if title_qids is None else WikilinkLinker(title_qids),
    )
    return _Resolved(groups, decisions)


WRONG = Candidate(qid="Q111", label="G. H. Thomas", description="engraver")
THOMAS = Candidate(qid="Q222", label="George Henry Thomas", birth_year=1816, death_year=1870)
TITLES = {"George Henry Thomas": "Q222"}


def test_the_method_is_known() -> None:
    assert WIKILINK_METHOD == "infobox_wikilink"
    assert WIKILINK_METHOD in METHODS
    assert 0.9 <= WIKILINK_CONFIDENCE < 1.0


# ─── Precedence ──────────────────────────────────────────────────────────────


def test_a_wikilink_beats_an_exact_name_match_on_the_wrong_person() -> None:
    mentions = [_mention("G. H. Thomas", ("George Henry Thomas",))]
    service = _StubService({"match": "NEW_ENTITY", "confidence": 0.5})

    plain = _resolve(mentions, _Source([WRONG], [THOMAS]))
    assert plain.of("G. H. Thomas").qid == "Q111"

    run = _resolve(mentions, _Source([WRONG], [THOMAS]), TITLES, service=service)
    decision = run.of("G. H. Thomas")
    assert (decision.status, decision.qid, decision.method) == (
        "linked",
        "Q222",
        "infobox_wikilink",
    )
    assert decision.confidence == WIKILINK_CONFIDENCE
    assert decision.candidate is THOMAS
    assert service.calls == []


def test_a_manual_override_beats_the_wikilink(tmp_path: Path) -> None:
    mentions = [_mention("G. H. Thomas", ("George Henry Thomas",))]
    other = Candidate(qid="Q333", label="Somebody Else")
    entry = _override_link(1, "G. H. Thomas", "Q333", battle=BATTLE.replace(" ", "_"))
    entry["target"]["side"] = "Union"
    path = tmp_path / "overrides.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "entries": [entry]}), encoding="utf-8")
    loaded = load_overrides(path, required=True)
    overrides = ResolveOverrides(loaded.for_stage("resolve"), OverrideReport("resolve", loaded))

    run = _resolve(mentions, _Source([WRONG], [THOMAS, other]), TITLES, overrides=overrides)

    decision = run.of("G. H. Thomas")
    assert (decision.qid, decision.method, decision.override_id) == ("Q333", "manual", "ov-0001")


def test_a_link_to_a_non_human_is_ignored() -> None:
    mentions = [_mention("G. H. Thomas", ("Vice admiral",))]
    # Q-id known for the title, but the by-Q-id lookup returns no human.
    run = _resolve(mentions, _Source([WRONG], []), {"Vice admiral": "Q999"})
    decision = run.of("G. H. Thomas")
    assert (decision.qid, decision.method) == ("Q111", "exact_wikidata")


def test_a_rank_link_beside_a_person_link_still_yields_the_person() -> None:
    mentions = [_mention("Vice admiral G. H. Thomas", ("Vice admiral", "George Henry Thomas"))]
    run = _resolve(
        mentions,
        _Source([], [THOMAS]),
        {"Vice admiral": "Q999", "George Henry Thomas": "Q222"},
    )
    assert run.of("Vice admiral G. H. Thomas").method == "infobox_wikilink"
    assert run.of("Vice admiral G. H. Thomas").qid == "Q222"


def test_two_different_humans_give_no_signal() -> None:
    other = Candidate(qid="Q444", label="Another Man", birth_year=1800, death_year=1880)
    mentions = [_mention("G. H. Thomas", ("George Henry Thomas", "Another Man"))]
    run = _resolve(
        mentions,
        _Source([WRONG], [THOMAS, other]),
        {"George Henry Thomas": "Q222", "Another Man": "Q444"},
    )
    decision = run.of("G. H. Thomas")
    assert (decision.qid, decision.method) == ("Q111", "exact_wikidata")


def test_a_link_the_lifespan_gate_rejects_falls_back_to_name_matching() -> None:
    ancient = Candidate(qid="Q555", label="George Henry Thomas", birth_year=1000, death_year=1060)
    mentions = [_mention("G. H. Thomas", ("George Henry Thomas",), year=1864)]
    run = _resolve(mentions, _Source([WRONG], [ancient]), {"George Henry Thomas": "Q555"})
    decision = run.of("G. H. Thomas")
    assert (decision.qid, decision.method) == ("Q111", "exact_wikidata")


def test_an_undated_battle_does_not_reject_the_link() -> None:
    mentions = [_mention("G. H. Thomas", ("George Henry Thomas",), year=None)]
    run = _resolve(mentions, _Source([WRONG], [THOMAS]), TITLES)
    assert run.of("G. H. Thomas").qid == "Q222"


def test_an_unresolved_title_is_ignored() -> None:
    mentions = [_mention("G. H. Thomas", ("Nobody Known",))]
    run = _resolve(mentions, _Source([WRONG], [THOMAS]), {"Nobody Known": None})
    assert run.of("G. H. Thomas").qid == "Q111"


def test_the_agreeing_case_keeps_the_qid_and_only_changes_the_method() -> None:
    mentions = [_mention("George Henry Thomas", ("George Henry Thomas",))]
    source = _Source([THOMAS], [THOMAS])
    before = _resolve(mentions, source)
    after = _resolve(mentions, source, TITLES)
    assert before.of("George Henry Thomas").qid == "Q222"
    assert after.of("George Henry Thomas").qid == "Q222"
    assert before.of("George Henry Thomas").method == "exact_wikidata"
    assert after.of("George Henry Thomas").method == "infobox_wikilink"


def test_linked_and_unlinked_mentions_of_a_name_reunite_through_the_qid() -> None:
    mentions = [
        _mention("George Henry Thomas", ("George Henry Thomas",), battle="Battle of A"),
        _mention("George Henry Thomas", (), battle="Battle of B"),
    ]
    source = _Source([THOMAS], [THOMAS])
    run = _resolve(mentions, source, TITLES)

    methods = sorted(d.method for d in run.decisions.values())
    assert methods == ["exact_wikidata", "infobox_wikilink"]
    identities = _build_identities(run.groups, run.decisions)
    assert [i.qid for i in identities] == ["Q222"]
    assert sum(len(g.mentions) for g in identities[0].groups) == 2


def test_only_the_linked_mention_leaves_its_name_group() -> None:
    mentions = [
        _mention("G. H. Thomas", ("George Henry Thomas",), battle="Battle of A"),
        _mention("G. H. Thomas", (), battle="Battle of B"),
    ]
    run = _resolve(mentions, _Source([WRONG], [THOMAS]), TITLES)
    assert run.of("G. H. Thomas", "Battle of A").qid == "Q222"
    assert run.of("G. H. Thomas", "Battle of B").qid == "Q111"


def test_the_run_logs_what_the_wikilinks_did() -> None:
    mentions = [
        _mention("G. H. Thomas", ("George Henry Thomas",), battle="Battle of A"),
        _mention("Mystery Man", ("Nobody Known",), battle="Battle of B"),
        _mention("Plain Man", (), battle="Battle of C"),
    ]
    with capture_logs() as logs:
        _resolve(mentions, _Source([WRONG], [THOMAS]), {**TITLES, "Nobody Known": None})
    events = [e for e in logs if e["event"] == "resolve_wikilinks"]
    assert len(events) == 1
    assert events[0]["mentions_with_links"] == 2
    assert events[0]["linked"] == 1
    assert events[0]["ignored_unresolved_title"] == 1


# ─── Wikidata lookups ────────────────────────────────────────────────────────


class _SparqlFetcher:
    """Serves SPARQL from ``fetch_raw`` and Wikipedia pages from ``fetch``."""

    def __init__(
        self,
        bindings: list[dict[str, str]] | None = None,
        pages: dict[str, FetchResult] | None = None,
    ) -> None:
        self.bindings = bindings or []
        self.pages = pages or {}
        self.queries: list[str] = []
        self.raw_urls: list[str] = []
        self.page_urls: list[str] = []

    async def fetch_raw(self, url: str, *, params: Any = None, headers: Any = None) -> FetchResult:
        self.raw_urls.append(url)
        if "wikipedia.org" in url:
            raise AssertionError("Wikipedia pages must go through the robots-checked fetch")
        self.queries.append(str((params or {}).get("query", "")))
        body = {
            "results": {
                "bindings": [
                    {
                        "title": {"type": "literal", "value": row["title"]},
                        "item": {
                            "type": "uri",
                            "value": f"http://www.wikidata.org/entity/{row['qid']}",
                        },
                    }
                    for row in self.bindings
                ]
            }
        }
        return FetchResult(url=url, status=200, text=json.dumps(body), attempts=1)

    async def fetch(self, url: str, *, params: Any = None, headers: Any = None) -> FetchResult:
        self.page_urls.append(url)
        return self.pages.get(url, FetchResult(url=url, status=404, error="http_404", attempts=1))


def test_the_sitelink_query_is_byte_stable_for_a_shuffled_title_list() -> None:
    titles = [f"Person {i}" for i in range(30)] + ['Quote "Name"']
    queries = []
    for seed in range(3):
        shuffled = list(titles)
        random.Random(seed).shuffle(shuffled)
        fetcher = _SparqlFetcher()
        asyncio.run(fetch_sitelink_qids(fetcher, shuffled, batch_size=10))
        queries.append(fetcher.queries)
    assert queries[0] == queries[1] == queries[2]
    assert len(queries[0]) == 4
    assert 'VALUES ?title { "Person 0"@en' in queries[0][0]
    assert r"Quote \"Name\"" in queries[0][-1]
    assert "schema:isPartOf <https://en.wikipedia.org/>" in queries[0][0]
    assert queries[0][0] == build_sitelink_query(sorted(titles)[:10])


def test_the_sitelink_bindings_are_parsed_into_title_to_qid() -> None:
    fetcher = _SparqlFetcher(
        [
            {"title": "George Henry Thomas", "qid": "Q222"},
            {"title": "Ernst Schirlitz", "qid": "Q88983"},
        ]
    )
    answered: set[str] = set()
    found = asyncio.run(
        fetch_sitelink_qids(
            fetcher,
            ["Ernst Schirlitz", "George Henry Thomas", "Redirect Title"],
            batch_size=25,
            answered=answered,
        )
    )
    assert found == {"George Henry Thomas": "Q222", "Ernst Schirlitz": "Q88983"}
    assert answered == {"Ernst Schirlitz", "George Henry Thomas", "Redirect Title"}


def test_the_redirect_fallback_reads_a_wiki_page_through_fetch_not_fetch_raw() -> None:
    url = "https://en.wikipedia.org/wiki/Abdallah_I_of_Jordan"
    html = '<script>RLCONF={"wgWikibaseItemId":"Q192780"};</script>'
    fetcher = _SparqlFetcher(pages={url: FetchResult(url=url, status=200, text=html, attempts=1)})

    found = asyncio.run(
        fetch_redirect_qids(fetcher, ["Abdallah I of Jordan", "No Such Page"], max_fetches=10)
    )

    assert found == {"Abdallah I of Jordan": "Q192780", "No Such Page": None}
    assert fetcher.raw_urls == []
    assert fetcher.page_urls == [
        "https://en.wikipedia.org/wiki/Abdallah_I_of_Jordan",
        "https://en.wikipedia.org/wiki/No_Such_Page",
    ]


def test_the_redirect_fallback_is_capped_and_failures_are_not_cached() -> None:
    fetcher = _SparqlFetcher()
    with capture_logs() as logs:
        found = asyncio.run(fetch_redirect_qids(fetcher, ["A", "B", "C"], max_fetches=1))
    assert list(found) == ["A"]
    assert len(fetcher.page_urls) == 1
    skipped = [e for e in logs if e["event"] == "resolve_wikilink_page_fetches_skipped"]
    assert skipped and skipped[0]["skipped"] == 2


def test_the_index_resolves_and_caches_titles(tmp_path: Path) -> None:
    cache = tmp_path / "resolve" / "wikilink_qids.json"
    url = "https://en.wikipedia.org/wiki/Redirect_Title"
    page = '"wgWikibaseItemId":"Q7"'
    fetcher = _SparqlFetcher(
        [{"title": "George Henry Thomas", "qid": "Q222"}],
        pages={url: FetchResult(url=url, status=200, text=page, attempts=1)},
    )
    index = WikilinkIndex(["George Henry Thomas", "Redirect Title", "Dead Link"], cache)
    asyncio.run(index.resolve(fetcher, batch_size=25, max_page_fetches=10))

    assert index.qids == {"George Henry Thomas": "Q222", "Redirect Title": "Q7", "Dead Link": None}
    assert json.loads(cache.read_text(encoding="utf-8")) == index.qids

    again = _SparqlFetcher()
    reloaded = WikilinkIndex(["George Henry Thomas", "Dead Link"], cache)
    asyncio.run(reloaded.resolve(again, batch_size=25, max_page_fetches=10))
    assert again.queries == [] and again.page_urls == []
    assert reloaded.qids["George Henry Thomas"] == "Q222"


def test_a_local_run_reads_the_cache_and_never_touches_the_network(tmp_path: Path) -> None:
    cache = tmp_path / "wikilink_qids.json"
    cache.write_text(json.dumps({"George Henry Thomas": "Q222", "Dead Link": None}), "utf-8")
    index = WikilinkIndex(["George Henry Thomas", "Dead Link", "Never Looked Up"], cache)
    groups = group_mentions([_mention("G. H. Thomas", ("George Henry Thomas",))])

    with capture_logs() as logs:
        source = _build_candidate_source(
            "local",
            groups,
            {},
            tmp_path / "raw",
            StageContext(),
            threshold=85.0,
            margin=6.0,
            wikilinks=index,
        )

    assert source is not None
    assert index.qids == {"George Henry Thomas": "Q222", "Dead Link": None}
    assert index.required_qids() == ["Q222"]
    uncached = [e for e in logs if e["event"] == "resolve_wikilinks_uncached"]
    assert uncached and uncached[0]["titles"] == 1


# ─── Regression: the corrections made by hand ────────────────────────────────

_MERGE_RE = re.compile(
    r"The infobox mention '(?P<mention>.+?)' links (?P<title>.+?) \((?P<qid>Q\d+)\)\."
)
_LINK_RE = re.compile(r"The infobox links (?P<title>.+)\.$")
_WRONG_ROW_RE = re.compile(r"Linked row \d+ \((?P<label>.+?), (?P<qid>Q\d+):")


_KINDS: set[str] = set()


def _regression_cases() -> list[tuple[str, str, str, str, str]]:
    """(id, mention, title, wrong qid, right qid) for ov-0300..ov-0474, sampled."""
    overrides = load_overrides(Path("config/overrides.yaml"), required=True)
    cases: dict[str, list[tuple[str, str, str, str, str]]] = {
        "commander_link": [],
        "general_merge": [],
    }
    for entry in sorted(overrides.for_stage("resolve"), key=lambda e: e.id):
        if not "ov-0300" <= entry.id <= "ov-0474":
            continue
        note = entry.note or ""
        if entry.kind == "general_merge":
            found = _MERGE_RE.search(note)
            if found is None or found["qid"] != str(entry.value):
                continue
            cases["general_merge"].append(
                (
                    entry.id,
                    found["mention"],
                    found["title"],
                    str(entry.target["general"]),
                    str(entry.value),
                )
            )
        elif entry.kind == "commander_link":
            title = _LINK_RE.search(note)
            wrong = _WRONG_ROW_RE.search(note)
            if title is None or wrong is None:
                continue
            cases["commander_link"].append(
                (
                    entry.id,
                    str(entry.target["mention"]),
                    title["title"],
                    wrong["qid"],
                    str(entry.value),
                )
            )
    # Every third parseable entry of each kind: deterministic, mixed, and spread
    # over the whole range rather than the first screenful.
    sample = {kind: found[::3] for kind, found in cases.items()}
    _KINDS.update(kind for kind, found in sample.items() if found)
    return sample["general_merge"] + sample["commander_link"]


_CASES = _regression_cases()


def test_the_regression_sample_is_large_and_mixed() -> None:
    assert len(_CASES) >= 20
    assert {"commander_link", "general_merge"} == _KINDS


@pytest.mark.parametrize(
    ("override_id", "mention", "title", "wrong_qid", "right_qid"),
    _CASES,
    ids=[case[0] for case in _CASES],
)
def test_the_wikilink_alone_reproduces_a_manual_correction(
    override_id: str, mention: str, title: str, wrong_qid: str, right_qid: str
) -> None:
    """With NO override applied, the wikilink picks the override's Q-id.

    The title to Q-id map below is taken from the overrides themselves (the
    override's own value), so this proves the precedence rule: a wikilink to
    a human beats an exact-name match on the wrong person. It does not prove
    that live Wikidata sitelinks return these Q-ids.
    """
    wrong = Candidate(qid=wrong_qid, label=mention, description="namesake")
    right = Candidate(qid=right_qid, label=title, description="the linked person")
    mentions = [_mention(mention, (title,), year=None)]

    control = _resolve(mentions, _Source([wrong], [right]))
    assert control.of(mention).qid == wrong_qid, "the name alone must pick the wrong person"

    run = _resolve(mentions, _Source([wrong], [right]), {title: right_qid})
    decision = run.of(mention)
    assert (decision.qid, decision.method, override_id is not None) == (
        right_qid,
        "infobox_wikilink",
        True,
    )
