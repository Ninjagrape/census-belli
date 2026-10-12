"""
Unit tests for the non-person filter, the search fallback and the biography
fields the resolve stage derives.

Everything the 2026-10 pilot got wrong here was invisible in the output: an
organisation that became a general still looked like a commander with a record,
and a famous marshal that came back "no candidate returned" still looked like a
minor man nobody had heard of. No database and no network.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from pipeline.crawlers.fetcher import FetchResult
from pipeline.resolvers import (
    Candidate,
    Identity,
    Mention,
    MentionGroup,
    ResolveCounts,
    astronomical_year,
    clean_surface,
    fetch_candidates,
    fetch_non_person_kinds,
    group_mentions,
    match_group,
    non_person_reason,
)
from pipeline.resolvers.candidates import PrefetchedCandidateSource, collect_query_names
from pipeline.resolvers.records import BattleContext
from pipeline.resolvers.store import year_to_date_literal
from pipeline.stages.resolve import _derive_era, resolve_groups

_BINS: tuple[tuple[str, int | None], ...] = (
    ("ancient", 500),
    ("medieval", 1500),
    ("early_modern", 1792),
    ("napoleonic", 1816),
    ("industrial", 1914),
    ("modern", None),
)


def _mention(name: str, *, year: int | None = 1813) -> Mention:
    return Mention(
        battle_slug="Battle_of_Leipzig",
        battle_name="Battle of Leipzig",
        side_label="Coalition",
        name=name,
        context=BattleContext(slug="Battle_of_Leipzig", name="Battle of Leipzig", year=year),
    )


def _groups(*names: str, year: int | None = 1813) -> list[MentionGroup]:
    return group_mentions([_mention(n, year=year) for n in names])


class _Source:
    """A candidate source with a fixed answer and an optional non-person verdict."""

    def __init__(self, found: list[Candidate], kind: str | None = None) -> None:
        self.found = found
        self.kind = kind

    def prefetch(self, groups: Any) -> None:
        return None

    def search(self, group: MentionGroup) -> list[Candidate]:
        return list(self.found)

    def non_person_kind(self, group: MentionGroup) -> str | None:
        return self.kind


class _Fetcher:
    """Serves one scripted SPARQL body and records the queries sent."""

    def __init__(self, body: str) -> None:
        self.body = body
        self.queries: list[str] = []

    async def fetch_raw(self, url: str, *, params: Any = None, headers: Any = None) -> FetchResult:
        self.queries.append(str((params or {}).get("query", "")))
        return FetchResult(url=url, status=200, text=self.body, attempts=1)


def _bindings(*rows: dict[str, str]) -> str:
    return json.dumps(
        {"results": {"bindings": [{k: {"value": v} for k, v in row.items()} for row in rows]}}
    )


# ─── Deterministic filter ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name",
    [
        "Luftwaffe",
        "Stavka",
        "Army Group B",
        "Army Group Centre",
        "Panzer Group West",
        "1st Ukrainian Front",
        "Bryansk Front",
        "Allied Air Forces",
        "Allied naval force",
        "Army High Command",
        "OB West",
        "SHAEF",
        "Army of the Potomac",
        "12 March",
        "Reinforcement",
        "Detailed list",
    ],
)
def test_organisations_and_table_fragments_are_not_persons(name: str) -> None:
    assert non_person_reason(name) is not None


@pytest.mark.parametrize(
    "name",
    [
        "Napoleon",
        "Fleet Admiral Chester Nimitz",
        "General Patton",
        "Gebhard von Blücher",
        "Prince of Württemberg",
        "Von Daun",
        "Marshal Zhukov",
    ],
)
def test_a_name_that_merely_contains_a_rank_or_unit_word_is_left_alone(name: str) -> None:
    assert non_person_reason(name) is None


def test_a_filtered_group_never_becomes_a_general() -> None:
    groups = _groups("Luftwaffe", "Napoleon")
    decisions = resolve_groups(
        groups, _Source([]), None, threshold=85.0, margin=6.0, counts=ResolveCounts(), schema={}
    )
    by_name = {g.display_name: decisions[g.key] for g in groups}
    assert by_name["Luftwaffe"].status == "unresolved"
    assert by_name["Luftwaffe"].method == "non_person"
    assert by_name["Napoleon"].status == "new"


# ─── Wikidata half of the filter ─────────────────────────────────────────────


def test_a_name_that_is_only_a_non_human_item_is_refused() -> None:
    groups = _groups("Ploscânea")
    decisions = resolve_groups(
        groups,
        _Source([], kind="village in Romania"),
        None,
        threshold=85.0,
        margin=6.0,
        counts=ResolveCounts(),
        schema={},
    )
    decision = decisions[groups[0].key]
    assert decision.method == "non_person"
    assert "village in Romania" in decision.reasoning


def test_an_exact_human_is_never_overruled_by_a_place_with_the_same_name() -> None:
    # Blücher's grandson (1865) carries the exact label and fails the date
    # gate. The name must stay a person: the check is "no human with this
    # name", not "no *viable* human".
    grandson = Candidate(
        qid="Q1261581", label="Gebhard von Blücher", birth_year=1865, death_year=1931
    )
    groups = _groups("Gebhard von Blücher", year=1813)
    decisions = resolve_groups(
        groups,
        _Source([grandson], kind="family name"),
        None,
        threshold=85.0,
        margin=6.0,
        counts=ResolveCounts(),
        schema={},
    )
    assert decisions[groups[0].key].method == "new_entity"


def test_non_person_kinds_are_read_from_the_query_result() -> None:
    fetcher = _Fetcher(_bindings({"name": "Stavka", "kinds": "supreme command; military staff"}))
    kinds = asyncio.run(fetch_non_person_kinds(fetcher, ["Stavka", "Napoleon"]))  # type: ignore[arg-type]
    assert kinds == {"Stavka": "military staff; supreme command"}
    sent = fetcher.queries[0]
    assert "FILTER NOT EXISTS" in sent and "wd:Q5" in sent
    assert '"Stavka"@en' in sent and '"Stavka"@mul' in sent


# ─── Search fallback ─────────────────────────────────────────────────────────


def _search_row(name: str, qid: str, label: str, birth: str, death: str) -> dict[str, str]:
    return {
        "name": name,
        "person": f"http://www.wikidata.org/entity/{qid}",
        "personLabel": label,
        "birth": birth,
        "death": death,
    }


def test_the_search_pass_asks_for_a_ranked_search_with_plain_literals() -> None:
    fetcher = _Fetcher(_bindings())
    asyncio.run(fetch_candidates(fetcher, ["Jacques MacDonald"], search=True))  # type: ignore[arg-type]
    sent = fetcher.queries[0]
    assert 'wikibase:api "EntitySearch"' in sent
    assert '"Jacques MacDonald"' in sent
    assert '"Jacques MacDonald"@en' not in sent
    assert "wd:Q5" in sent  # a search hit must still be a human


def test_a_search_hit_is_not_given_the_searched_name_as_an_alias() -> None:
    # The exact path records the matched name as an alias because the entity
    # really carries it. A search hit only resembles the name, and an alias
    # here would be an exact-match key: the fuzzy hit would link at 0.95.
    body = _bindings(
        _search_row(
            "Emmanuel de Grouchy",
            "Q333821",
            "Emmanuel de Grouchy, Marquis de Grouchy",
            "1766-10-23T00:00:00Z",
            "1847-05-29T00:00:00Z",
        )
    )
    found = asyncio.run(
        fetch_candidates(_Fetcher(body), ["Emmanuel de Grouchy"], search=True)  # type: ignore[arg-type]
    )
    (candidate,) = found["Emmanuel de Grouchy"]
    assert candidate.aliases == ()
    assert (candidate.birth_year, candidate.death_year) == (1766, 1847)


@pytest.mark.parametrize(
    ("mention", "label", "born", "died", "year"),
    [
        # Labels as Wikidata returned them on 2026-10-11 for the pilot's names.
        ("Emmanuel de Grouchy", "Emmanuel de Grouchy, Marquis de Grouchy", 1766, 1847, 1815),
        ("Matvei Platov", "Matvey Ivanovich Platov", 1753, 1818, 1812),
        ("Gebhard Leberecht von Blücher", "Gebhard Leberecht von Blücher", 1742, 1819, 1813),
    ],
)
def test_a_search_hit_with_a_longer_label_links_through_the_fuzzy_matcher(
    mention: str, label: str, born: int, died: int, year: int
) -> None:
    group = _groups(mention, year=year)[0]
    candidate = Candidate(qid="Q9", label=label, birth_year=born, death_year=died)
    decision = match_group(group, [candidate], threshold=85.0, margin=6.0)
    assert decision.status == "linked"
    assert decision.qid == "Q9"


def test_a_search_hit_who_was_not_alive_is_still_refused() -> None:
    # The search widens the net; it must not widen what the date gate lets in.
    group = _groups("Gebhard von Blücher", year=1813)[0]
    grandson = Candidate(
        qid="Q1261581", label="Gebhard von Blücher", birth_year=1865, death_year=1931
    )
    assert match_group(group, [grandson], threshold=85.0, margin=6.0).status == "new"


def test_extending_a_source_keeps_the_first_lookup_and_drops_repeats() -> None:
    group = _groups("Jacques MacDonald")[0]
    name = collect_query_names([group])[0]
    first = Candidate(qid="Q1", label="Exact", aliases=(name,))
    source = PrefetchedCandidateSource({name: [first]})
    source.extend(
        {name: [Candidate(qid="Q1", label="Search twin"), Candidate(qid="Q2", label="New")]}
    )
    found = source.search(group)
    assert [c.qid for c in found] == ["Q1", "Q2"]
    assert found[0].aliases == (name,)


# ─── Surface cleaning and dates ──────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "cleaned"),
    [
        ("Overall: Gebhard Leberecht von Blücher", "Gebhard Leberecht von Blücher"),
        ("Craonne: Mikhail Vorontsov", "Mikhail Vorontsov"),
        ("Napoleon", "Napoleon"),
    ],
)
def test_a_sub_heading_flattened_into_a_name_is_removed(raw: str, cleaned: str) -> None:
    assert clean_surface(raw) == cleaned


@pytest.mark.parametrize(
    ("literal", "expected"),
    [
        ("52 BC", -51),
        ("October 332 BC", -331),
        ("219 BC (8 months)", -218),
        ("late 48 BC - early or mid 47 BC", -47),
        ("AD 79", 79),
        ("October 1805", 1805),
        ("0031-09-02 BC", -30),
        ("2015", None),
        ("12", None),
    ],
)
def test_prose_dates_are_read_as_astronomical_years(literal: str, expected: int | None) -> None:
    assert astronomical_year(literal) == expected


# ─── Biography ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("year", "expected"),
    [
        (1769, "1769-01-01"),
        (1, "0001-01-01"),
        (0, "0001-01-01 BC"),
        (-43, "0044-01-01 BC"),
        (None, None),
    ],
)
def test_astronomical_years_become_postgres_date_literals(
    year: int | None, expected: str | None
) -> None:
    assert year_to_date_literal(year) == expected


def _identity(*years: int, birth: int | None = None) -> Identity:
    group = _groups("Someone", year=years[0] if years else None)[0]
    group.mentions = [_mention("Someone", year=y) for y in years]
    return Identity(key="k", canonical_name="Someone", groups=[group], birth_year=birth)


def test_era_is_the_bin_of_the_median_battle_year() -> None:
    assert _derive_era(_identity(1805, 1809, 1815), _BINS) == "napoleonic"
    assert _derive_era(_identity(-216, -202), _BINS) == "ancient"


def test_era_falls_back_to_the_year_they_came_of_age() -> None:
    identity = Identity(key="k", canonical_name="Someone", birth_year=1742)
    assert _derive_era(identity, _BINS) == "early_modern"


def test_a_person_nothing_dates_gets_no_era() -> None:
    assert _derive_era(Identity(key="k", canonical_name="Someone"), _BINS) is None


def test_a_name_that_is_only_a_disambiguation_page_or_family_name_is_not_evidence() -> None:
    # "Servilius Geminus" and "Alexey Yermolov" are labels of disambiguation
    # pages; both are men. Only a class that is not about the name counts.
    fetcher = _Fetcher(
        _bindings(
            {"name": "Servilius Geminus", "kinds": "Wikimedia disambiguation page; family name"},
            {"name": "Auerstedt", "kinds": "family name; village"},
        )
    )
    kinds = asyncio.run(
        fetch_non_person_kinds(fetcher, ["Servilius Geminus", "Auerstedt"])  # type: ignore[arg-type]
    )
    assert kinds == {"Auerstedt": "village"}
