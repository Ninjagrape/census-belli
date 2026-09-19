"""
Unit tests for entity resolution.

The cases worth pinning are the ones where a plausible implementation gets it
wrong, and they are mostly about *refusing* to match. Name similarity alone
merges the five Scipios, links a commander to his own grandson, and turns
every infobox that says "Unknown" into one prolific general. Each of those has
a test here, because each would produce a ranking that looks entirely
reasonable and is wrong.

No database and no network: the candidate source and the LLM service are both
stubbed. The matcher is pure, which is the point of the module boundary.
"""

from __future__ import annotations

import asyncio
import functools
import json
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

import pytest

from pipeline.crawlers.fetcher import FetchResult
from pipeline.llm.base import CallStatus, LLMResponse
from pipeline.resolvers import (
    Candidate,
    Decision,
    Mention,
    MentionGroup,
    ResolveCounts,
    astronomical_year,
    clean_surface,
    disambiguate,
    format_candidates,
    group_mentions,
    is_placeholder,
    lifespan_verdict,
    match_against_identities,
    match_group,
    name_keys,
    person_candidates,
)
from pipeline.resolvers.candidates import (
    CandidateLookupError,
    LocalCandidateSource,
    NullCandidateSource,
    PrefetchedCandidateSource,
    _iso_to_literal,
    _sparql_literal,
    collect_query_names,
    fetch_candidates,
)
from pipeline.resolvers.records import BattleContext
from pipeline.stages.resolve import LazyService, _build_identities, resolve_groups


# ─── Helpers ─────────────────────────────────────────────────────────────────
def sync(test: Callable[..., Coroutine[Any, Any, None]]) -> Callable[..., None]:
    """Run a coroutine test on its own event loop.

    pytest-asyncio is declared in the dev extras but is not installed, so the
    coroutine tests drive their own loop, the same way tests/unit/test_crawl.py
    does. Keeps the suite runnable on the base dependencies.

    Args:
        test: The coroutine test function.

    Returns:
        A plain callable pytest will collect.
    """

    @functools.wraps(test)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        asyncio.run(test(*args, **kwargs))

    return wrapper



def _mention(
    name: str,
    *,
    battle: str = "Battle of Actium",
    year: int | None = -30,
    side: str = "Octavian",
    polity: str = "Roman Republic",
    role: str = "field_commander",
    evidence: str = "",
) -> Mention:
    """Build a mention with battle context attached."""
    slug = battle.replace(" ", "_")
    return Mention(
        battle_slug=slug,
        battle_name=battle,
        side_label=side,
        name=name,
        apparent_role=role,
        role_evidence=evidence,
        polity=polity,
        context=BattleContext(slug=slug, name=battle, year=year, war="Final War"),
    )


def _group(*names: str, year: int | None = -30, battle: str = "Battle of Actium") -> MentionGroup:
    """Build a single group from mentions of the same person."""
    groups = group_mentions([_mention(n, battle=battle, year=year) for n in names])
    assert len(groups) == 1, f"expected one group, got {[g.key for g in groups]}"
    return groups[0]


def _candidate(qid: str, label: str, born: int | None, died: int | None, **kw: Any) -> Candidate:
    """Build a Wikidata candidate."""
    return Candidate(qid=qid, label=label, birth_year=born, death_year=died, **kw)


class _StubSource:
    """A candidate source returning a fixed list for every group."""

    def __init__(self, candidates: list[Candidate]) -> None:
        self.candidates = candidates
        self.prefetched = 0

    def prefetch(self, groups: Any) -> None:
        self.prefetched += 1

    def search(self, group: MentionGroup) -> list[Candidate]:
        return list(self.candidates)


class _StubService:
    """An LLM service returning one scripted payload."""

    def __init__(self, payload: dict[str, Any] | None, status: CallStatus = CallStatus.OK) -> None:
        self.payload = payload
        self.status = status
        self.calls: list[str] = []

    def complete(
        self, system: str, user: str, json_schema: dict[str, Any], **kw: Any
    ) -> LLMResponse:
        self.calls.append(user)
        return LLMResponse(
            status=self.status,
            provider="stub",
            model="stub-1",
            request_hash="hash",
            data=self.payload,
            error=None if self.status is CallStatus.OK else "stub failure",
        )


class _StubFetcher:
    """A fetcher returning one scripted SPARQL body per call.

    Only ``fetch_raw`` is implemented. That is deliberate: WDQS robots.txt
    disallows /sparql for crawlers, so ``run_query`` bypasses the robots gate
    the way the code already does for robots.txt itself. If a future change
    routes queries back through ``fetch``, these tests fail with an
    AttributeError rather than quietly returning nothing -- which is exactly
    how that defect went unnoticed the first time.
    """

    def __init__(self, *bodies: str | None) -> None:
        self.bodies = list(bodies)
        self.queries: list[str] = []

    async def fetch_raw(
        self, url: str, *, params: Any = None, headers: Any = None
    ) -> FetchResult:
        self.queries.append(str((params or {}).get("query", "")))
        body = self.bodies[min(len(self.queries) - 1, len(self.bodies) - 1)]
        if body is None:
            return FetchResult(url=url, status=500, error="boom", attempts=1)
        return FetchResult(url=url, status=200, text=body, attempts=1)


def _entity(
    qid: str,
    label: str,
    *,
    human: bool = True,
    aliases: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Build a Wikidata entity payload as the crawl stage stores it."""
    return {
        "entities": {
            qid: {
                "id": qid,
                "labels": {"en": {"value": label}},
                "descriptions": {"en": {"value": "Roman general"}},
                "aliases": {"en": [{"value": a} for a in aliases]},
                "claims": {
                    "P31": [
                        {
                            "mainsnak": {
                                "datavalue": {"value": {"id": "Q5" if human else "Q178561"}}
                            }
                        }
                    ],
                    "P569": [
                        {
                            "mainsnak": {
                                "datavalue": {
                                    "value": {
                                        "time": "-0062-01-01T00:00:00Z",
                                        "precision": 11,
                                    }
                                }
                            }
                        }
                    ],
                },
                "sitelinks": {"enwiki": {"url": f"https://en.wikipedia.org/wiki/{label}"}},
            }
        }
    }


# ─── Name normalisation ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Gen. George S. Patton", "george s patton"),
        ("Sir Edward Codrington", "edward codrington"),
        ("Field Marshal Bernard Montgomery", "bernard montgomery"),
        ("Admiral Lord Nelson", "nelson"),
    ],
)
def test_leading_titles_produce_a_stripped_key(raw: str, expected: str) -> None:
    assert expected in name_keys(raw)


def test_regnal_numerals_are_never_stripped() -> None:
    # Ramesses II and Ramesses III are different men with different records.
    # Folding them together would fit one skill parameter to two careers.
    assert set(name_keys("Ramesses II")).isdisjoint(set(name_keys("Ramesses III")))


def test_a_title_that_is_the_name_survives() -> None:
    # "Duke of Wellington" is the common name, not an ornament on one.
    keys = name_keys("Duke of Wellington")
    assert "duke of wellington" in keys
    assert "wellington" in keys


def test_fate_markers_and_citations_are_stripped() -> None:
    assert clean_surface("Marcus Licinius Crassus †") == "Marcus Licinius Crassus"
    assert clean_surface("Publius Quinctilius Varus[1] (POW)") == "Publius Quinctilius Varus"


def test_inverted_names_are_flipped_but_titles_are_not() -> None:
    assert "horatio nelson" in name_keys("Nelson, Horatio")
    # A trailing title is not a forename; flipping here would produce nonsense.
    assert "1st duke of wellington arthur wellesley" not in name_keys(
        "Arthur Wellesley, 1st Duke of Wellington"
    )


@pytest.mark.parametrize(
    "raw",
    ["Unknown", "unknown", "various", "Several local chieftains", "", "???", "N/A"],
)
def test_placeholders_are_recognised(raw: str) -> None:
    assert is_placeholder(raw)


@pytest.mark.parametrize("raw", ["Napoleon", "Yi Sun-sin", "José de San Martín"])
def test_real_names_are_not_placeholders(raw: str) -> None:
    assert not is_placeholder(raw)


# ─── Dates ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("literal", "expected"),
    [
        ("0031-09-02 BC", -30),  # 31 BC is astronomical -30; there is a year zero
        ("0001-01-01 BC", 0),
        ("1805-12-02", 1805),
        ("", None),
        ("not a date", None),
    ],
)
def test_astronomical_year_matches_the_generated_column(literal: str, expected: int | None) -> None:
    assert astronomical_year(literal) == expected


# ─── Grouping ────────────────────────────────────────────────────────────────


def test_a_rank_or_a_fate_marker_does_not_split_a_commander_in_two() -> None:
    # Three spellings of one surname that differ only in what an editor put
    # around it. Each folds to the same key, so they resolve as one person.
    mentions = [
        _mention("Gen. Patton", battle="Battle of El Guettar", year=1943),
        _mention("Patton", battle="Operation Cobra", year=1944),
        _mention("Patton[3]", battle="Battle of the Bulge", year=1944),
    ]
    groups = group_mentions(mentions)
    assert len(groups) == 1
    assert len(groups[0].mentions) == 3


def test_grouping_needs_an_exact_shared_key_and_nothing_less() -> None:
    # "Agrippa" and "Marcus Vipsanius Agrippa" are the same man, and this step
    # deliberately does not say so: grouping is exact-key only, because a
    # wrong merge here is invisible downstream. They are reunited later, by
    # the Q-id they both resolve to -- see
    # test_two_groups_linking_to_one_entity_become_one_general.
    groups = group_mentions([_mention("Agrippa"), _mention("Marcus Vipsanius Agrippa")])
    assert len(groups) == 2


def test_different_people_do_not_group() -> None:
    assert len(group_mentions([_mention("Scipio Africanus"), _mention("Hannibal Barca")])) == 2


def test_the_canonical_name_drops_the_rank() -> None:
    # display_name becomes generals.canonical_name for a new entity, so the
    # published name should not carry a rank when a bare form was also seen.
    group = _group("Gen. Patton", "Gen. Patton", "Patton")
    assert group.display_name == "Patton"
    assert "Gen. Patton" in group.surface_forms


# ─── The date gate ───────────────────────────────────────────────────────────


def test_a_candidate_dead_before_the_battle_is_rejected() -> None:
    assert lifespan_verdict(_candidate("Q36456", "Hannibal", -246, -182), [1811]) is False


def test_a_candidate_alive_at_the_battle_is_accepted() -> None:
    assert lifespan_verdict(_candidate("Q36456", "Hannibal", -246, -182), [-201]) is True


def test_the_gate_abstains_when_it_cannot_judge() -> None:
    # No dated battle, or no dated life: abstaining is not the same as passing,
    # and the matcher treats it as "the name has to carry this alone".
    assert lifespan_verdict(_candidate("Q1", "Somebody", -246, -182), []) is None
    assert lifespan_verdict(_candidate("Q1", "Somebody", None, None), [1500]) is None


def test_an_exact_name_match_is_refused_when_the_dates_rule_it_out() -> None:
    # This is the case that name similarity alone gets wrong every time.
    group = _group("Hannibal", year=1811, battle="Battle of Lissa")
    decision = match_group(group, [_candidate("Q36456", "Hannibal", -246, -182)])
    assert decision.status == "new"
    assert decision.qid is None
    assert "date gate" in decision.reasoning


# ─── Matching ────────────────────────────────────────────────────────────────


def test_a_single_exact_match_links() -> None:
    group = _group("Marcus Vipsanius Agrippa")
    decision = match_group(group, [_candidate("Q167846", "Marcus Vipsanius Agrippa", -62, -11)])
    assert decision.status == "linked"
    assert decision.qid == "Q167846"
    assert decision.method == "exact_wikidata"
    assert decision.confidence > 0.9


def test_two_exact_matches_go_to_the_llm_rather_than_guessing() -> None:
    decision = match_group(
        _group("Scipio", year=-201),
        [_candidate("Q1", "Scipio", -235, -183), _candidate("Q2", "Scipio", -220, -180)],
    )
    assert decision.status == "ambiguous"
    assert decision.candidates_considered == 2


def test_near_ties_go_to_the_llm_rather_than_picking_the_first() -> None:
    decision = match_group(
        _group("Publius Cornelius Scipio", year=-201),
        [
            _candidate("Q1", "Publius Cornelius Scipio Africanus", -235, -183),
            _candidate("Q2", "Publius Cornelius Scipio Nasica", -227, -171),
        ],
    )
    assert decision.status == "ambiguous"


def test_an_unrelated_candidate_becomes_a_new_entity_without_an_llm_call() -> None:
    decision = match_group(
        _group("Quintus Fabius Maximus", year=-217),
        [_candidate("Q99", "Winston Churchill", 1874, 1965)],
    )
    assert decision.status == "new"
    assert decision.method == "new_entity"


def test_no_candidates_at_all_becomes_a_new_entity() -> None:
    decision = match_group(_group("Vercingetorix"), [])
    assert decision.status == "new"
    assert decision.candidates_considered == 0


# ─── Corpus-internal matching ────────────────────────────────────────────────


def _anchor(name: str, candidate: Candidate, *, year: int, battle: str) -> list[Any]:
    """Resolve one group against one candidate and build its identity."""
    group = _group(name, year=year, battle=battle)
    return _build_identities([group], {group.key: match_group(group, [candidate])})


def test_a_bare_surname_attaches_to_an_identity_already_resolved() -> None:
    anchors = _anchor(
        "Napoleon Bonaparte",
        _candidate("Q517", "Napoleon Bonaparte", 1769, 1821),
        year=1805,
        battle="Battle of Austerlitz",
    )
    assert anchors and anchors[0].qid == "Q517"

    attached = match_against_identities(
        _group("Bonaparte", year=1806, battle="Battle of Jena"), anchors
    )
    assert attached is not None
    assert attached.qid == "Q517"
    assert attached.method == "corpus_fuzzy"
    # A corpus match is never trusted as far as a Wikidata link.
    assert attached.confidence < 0.95


def test_the_career_gate_blocks_a_match_across_centuries() -> None:
    anchors = _anchor(
        "Scipio Africanus",
        _candidate("Q1", "Scipio Africanus", -235, -183),
        year=-201,
        battle="Battle of Zama",
    )
    stray = _group("Scipio Africanus", year=1500, battle="Battle of Cerignola")
    assert match_against_identities(stray, anchors) is None


# ─── Disambiguation ──────────────────────────────────────────────────────────


_CANDIDATES = [
    _candidate("Q1", "Scipio Africanus", -235, -183, description="Roman general"),
    _candidate("Q2", "Scipio Aemilianus", -185, -129),
]


def _disambiguate(
    payload: dict[str, Any] | None,
    status: CallStatus = CallStatus.OK,
) -> tuple[Decision, _StubService]:
    """Run the disambiguation step against a scripted response."""
    service = _StubService(payload, status)
    decision = disambiguate(
        _group("Scipio", year=-201),
        _CANDIDATES,
        service,  # type: ignore[arg-type]
        system="s",
        template="{name} / {candidates_formatted}",
        schema={"type": "object"},
    )
    return decision, service


def test_the_model_choosing_a_candidate_links_it() -> None:
    decision, _ = _disambiguate({"match": "Q1", "confidence": 0.8, "reasoning": "dates fit"})
    assert decision.status == "linked"
    assert decision.qid == "Q1"
    assert decision.confidence == 0.8


def test_new_entity_is_honoured() -> None:
    decision, _ = _disambiguate({"match": "NEW_ENTITY", "confidence": 0.9})
    assert decision.status == "new"
    assert decision.qid is None


def test_uncertain_splits_rather_than_merges() -> None:
    decision, _ = _disambiguate({"match": "UNCERTAIN", "confidence": 0.9})
    assert decision.status == "new"
    # A split the ranking can absorb beats a merge that pools two records.
    assert decision.confidence < 0.5


def test_a_qid_that_was_never_offered_is_not_trusted() -> None:
    decision, _ = _disambiguate({"match": "Q999999", "confidence": 1.0})
    assert decision.status == "new"
    assert "not among the candidates" in decision.reasoning


def test_a_failed_call_leaves_the_group_unresolved() -> None:
    # Not a new entity: an API failure is a known unknown, and inventing an
    # identity would put the outage into the data where nothing can see it.
    decision, _ = _disambiguate(None, CallStatus.API_ERROR)
    assert decision.status == "unresolved"
    assert decision.method == "llm_failed"


def test_the_prompt_offers_the_qids_the_model_must_answer_with() -> None:
    rendered = format_candidates(_CANDIDATES)
    assert "Q1" in rendered and "Q2" in rendered
    assert "-235" in rendered  # astronomical years, not "236 BC"


# ─── Candidate sources ───────────────────────────────────────────────────────


def test_local_source_indexes_people_and_ignores_battles(tmp_path: Path) -> None:
    (tmp_path / "Q167846.json").write_text(
        json.dumps(_entity("Q167846", "Marcus Vipsanius Agrippa", aliases=("Agrippa",))),
        encoding="utf-8",
    )
    (tmp_path / "Q185729.json").write_text(
        json.dumps(_entity("Q185729", "Battle of Actium", human=False)), encoding="utf-8"
    )

    source = LocalCandidateSource(tmp_path)
    assert [c.qid for c in source.search(_group("Agrippa"))] == ["Q167846"]
    assert source.search(_group("Battle of Actium")) == []


def test_person_candidates_reads_the_lifespan_as_an_astronomical_year() -> None:
    candidates = person_candidates(_entity("Q167846", "Marcus Vipsanius Agrippa"))
    assert candidates[0].birth_year == -61  # 62 BC


def test_null_source_resolves_everything_as_new() -> None:
    assert NullCandidateSource().search(_group("Anybody")) == []


def test_sparql_literals_escape_quotes() -> None:
    # A name carrying a quote would otherwise end the literal early and
    # change the query that gets sent.
    assert _sparql_literal('Sun "the Great"') == '"Sun \\"the Great\\""@en'


@pytest.mark.parametrize(
    ("iso", "expected"),
    [
        ("-0031-09-02T00:00:00Z", -30),
        ("1769-08-15T00:00:00Z", 1769),  # the query service's form: unsigned
        ("+1769-08-15T00:00:00Z", 1769),  # the entity dumps' form: signed
        ("", None),
    ],
)
def test_sparql_datetimes_become_astronomical_years(iso: str, expected: int | None) -> None:
    assert astronomical_year(_iso_to_literal(iso)) == expected


def test_prefetched_source_serves_by_surface_form() -> None:
    group = _group("Agrippa")
    names = collect_query_names([group])
    source = PrefetchedCandidateSource(
        {names[0]: [_candidate("Q167846", "Marcus Vipsanius Agrippa", -62, -11)]}
    )
    assert [c.qid for c in source.search(group)] == ["Q167846"]


@sync
async def test_fetch_candidates_folds_rows_into_one_candidate_per_person() -> None:
    # The query GROUPs by (name, person), so occupations arrive pre-joined and
    # one person yields one row per *matched name*. Two names hitting the same
    # entity must still fold into a single candidate, or the matcher would see
    # a duplicate and call it ambiguous.
    body = json.dumps(
        {
            "results": {
                "bindings": [
                    {
                        "name": {"value": "Napoleon"},
                        "person": {"value": "http://www.wikidata.org/entity/Q517"},
                        "personLabel": {"value": "Napoleon"},
                        "birth": {"value": "1769-08-15T00:00:00Z"},
                        "death": {"value": "1821-05-05T00:00:00Z"},
                        "occupations": {"value": "military officer, politician"},
                    },
                    {
                        "name": {"value": "Napoleon Bonaparte"},
                        "person": {"value": "http://www.wikidata.org/entity/Q517"},
                        "personLabel": {"value": "Napoleon"},
                        "birth": {"value": "1769-08-15T00:00:00Z"},
                        "death": {"value": "1821-05-05T00:00:00Z"},
                        "occupations": {"value": "military officer, politician"},
                    },
                ]
            }
        }
    )
    found = await fetch_candidates(
        _StubFetcher(body),  # type: ignore[arg-type]
        ["Napoleon", "Napoleon Bonaparte"],
    )

    assert [c.qid for c in found["Napoleon"]] == ["Q517"]
    assert [c.qid for c in found["Napoleon Bonaparte"]] == ["Q517"]

    candidate = found["Napoleon"][0]
    assert candidate.birth_year == 1769
    assert candidate.death_year == 1821
    assert candidate.occupations == ("military officer", "politician")
    assert candidate.is_military
    # The other matched name is carried as an alias; the label is not.
    assert candidate.aliases == ("Napoleon Bonaparte",)


@sync
async def test_an_entity_matched_by_alias_is_kept() -> None:
    # The query matches on label OR altLabel, so the name we asked about is
    # often not the entity's label: "Duke of Wellington" is an alias of
    # "Arthur Wellesley, 1st Duke of Wellington". The lookup used to re-derive
    # which name had matched by comparing against the label alone, which threw
    # away every alias match -- Wellington returned zero candidates from a
    # query that had just found him. Index on the ?name the endpoint reports.
    body = json.dumps(
        {
            "results": {
                "bindings": [
                    {
                        "name": {"value": "Duke of Wellington"},
                        "person": {"value": "http://www.wikidata.org/entity/Q131691"},
                        "personLabel": {"value": "Arthur Wellesley, 1st Duke of Wellington"},
                        "birth": {"value": "1769-05-01T00:00:00Z"},
                        "death": {"value": "1852-09-14T00:00:00Z"},
                    }
                ]
            }
        }
    )
    found = await fetch_candidates(
        _StubFetcher(body),  # type: ignore[arg-type]
        ["Duke of Wellington"],
    )

    candidates = found["Duke of Wellington"]
    assert [c.qid for c in candidates] == ["Q131691"]
    # The matched name is carried as an alias, which is what gives the matcher
    # an exact-match key for a commander whose label is longer than the name
    # an infobox writes.
    assert "Duke of Wellington" in candidates[0].aliases
    assert candidates[0].birth_year == 1769


@sync
async def test_a_label_match_still_works_and_is_not_its_own_alias() -> None:
    body = json.dumps(
        {
            "results": {
                "bindings": [
                    {
                        "name": {"value": "Napoleon"},
                        "person": {"value": "http://www.wikidata.org/entity/Q517"},
                        "personLabel": {"value": "Napoleon"},
                    }
                ]
            }
        }
    )
    found = await fetch_candidates(_StubFetcher(body), ["Napoleon"])  # type: ignore[arg-type]
    candidate = found["Napoleon"][0]
    assert candidate.qid == "Q517"
    assert candidate.aliases == ()


@sync
async def test_a_total_lookup_failure_raises_rather_than_resolving_everything_as_new() -> None:
    # Graceful degradation is for when *part* of a system errors. When every
    # query fails the stage has no candidates at all, and continuing would
    # write a corpus in which no commander is linked to anything -- which is
    # precisely what robots.txt silently did before run_query was exempted.
    with pytest.raises(CandidateLookupError):
        await fetch_candidates(_StubFetcher(None), ["Napoleon"])  # type: ignore[arg-type]


@sync
async def test_one_failed_batch_among_several_still_degrades_quietly() -> None:
    body = json.dumps(
        {
            "results": {
                "bindings": [
                    {
                        "person": {"value": "http://www.wikidata.org/entity/Q517"},
                        "personLabel": {"value": "Napoleon"},
                    }
                ]
            }
        }
    )
    # First batch succeeds, second fails. One bad query is data, not config.
    found = await fetch_candidates(
        _StubFetcher(body, None),  # type: ignore[arg-type]
        ["Napoleon", "Nelson"],
        batch_size=1,
    )
    assert [c.qid for c in found["Napoleon"]] == ["Q517"]
    assert found["Nelson"] == []


# ─── The stage's decision loop ───────────────────────────────────────────────


def test_placeholder_mentions_never_become_generals() -> None:
    groups = group_mentions([_mention("Unknown"), _mention("Hannibal Barca", year=-216)])
    counts = ResolveCounts()
    decisions = resolve_groups(
        groups, _StubSource([]), None, threshold=85.0, margin=6.0, counts=counts, schema={}
    )
    by_name = {g.display_name: decisions[g.key] for g in groups}
    assert by_name["Unknown"].status == "unresolved"
    assert by_name["Unknown"].method == "placeholder"
    assert by_name["Hannibal Barca"].status == "new"
    assert counts.unresolved == 1


def test_an_ambiguous_group_with_no_llm_stays_unresolved() -> None:
    groups = group_mentions([_mention("Scipio", year=-201)])
    counts = ResolveCounts()
    decisions = resolve_groups(
        groups,
        _StubSource(
            [_candidate("Q1", "Scipio", -235, -183), _candidate("Q2", "Scipio", -220, -180)]
        ),
        None,
        threshold=85.0,
        margin=6.0,
        counts=counts,
        schema={},
    )
    decision = decisions[groups[0].key]
    assert decision.status == "unresolved"
    assert "no LLM service" in decision.reasoning
    assert counts.llm_calls == 0


def test_the_llm_is_asked_once_per_group_not_once_per_mention() -> None:
    mentions = [
        _mention("Scipio", battle="Battle of Zama", year=-201),
        _mention("Scipio", battle="Battle of Ilipa", year=-205),
        _mention("Scipio", battle="Battle of Baecula", year=-207),
    ]
    service = _StubService({"match": "Q1", "confidence": 0.8})
    counts = ResolveCounts()
    resolve_groups(
        group_mentions(mentions),
        _StubSource(
            [_candidate("Q1", "Scipio", -235, -183), _candidate("Q2", "Scipio", -220, -180)]
        ),
        LazyService(lambda: service),  # type: ignore[arg-type,return-value]
        threshold=85.0,
        margin=6.0,
        counts=counts,
        system="s",
        template="{name}",
        schema={},
    )
    assert len(service.calls) == 1
    assert counts.llm_calls == 1


def test_two_groups_linking_to_one_entity_become_one_general() -> None:
    groups = group_mentions(
        [
            _mention("Marcus Vipsanius Agrippa", battle="Battle of Actium", year=-30),
            _mention("M. Agrippa", battle="Battle of Naulochus", year=-35),
        ]
    )
    assert len(groups) == 2, "the two spellings should not group on name alone"

    agrippa = _candidate(
        "Q167846", "Marcus Vipsanius Agrippa", -61, -11, aliases=("M. Agrippa", "Agrippa")
    )
    decisions = resolve_groups(
        groups,
        _StubSource([agrippa]),
        None,
        threshold=85.0,
        margin=6.0,
        counts=ResolveCounts(),
        schema={},
    )
    identities = _build_identities(groups, decisions)

    assert len(identities) == 1
    assert identities[0].qid == "Q167846"
    assert len(identities[0].groups) == 2
    assert "M. Agrippa" in identities[0].aliases
    # The canonical name is not repeated as an alias of itself.
    assert identities[0].canonical_name not in identities[0].aliases


def test_each_row_keeps_its_own_group_confidence() -> None:
    # A well-attested identity must not lend its confidence to a weakly
    # matched spelling that merged into it.
    groups = group_mentions(
        [
            _mention("Napoleon Bonaparte", battle="Battle of Austerlitz", year=1805),
            _mention("Bonaparte", battle="Battle of Jena", year=1806),
        ]
    )
    resolve_groups(
        groups,
        _StubSource([_candidate("Q517", "Napoleon Bonaparte", 1769, 1821)]),
        None,
        threshold=85.0,
        margin=6.0,
        counts=ResolveCounts(),
        schema={},
    )
    by_name = {g.display_name: g.confidence for g in groups}
    assert by_name["Napoleon Bonaparte"] > by_name["Bonaparte"]
