"""
The per-mention lifespan guard and the lifetime split in resolve.

A name group used to link as a whole when any one of its battles fell inside a
candidate's life, so one in-lifespan battle carried every namesake mention with
it. These tests use the real years of the cases a historiography review found
(astronomical years: 45 BC is -44). No network, no database.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from pipeline.config import load_agent_spec
from pipeline.resolvers import (
    Candidate,
    Decision,
    Mention,
    MentionGroup,
    ResolveCounts,
    career_clusters,
    group_mentions,
    lifespan_verdict,
    partition_by_lifespan,
)
from pipeline.resolvers.disambiguate import prompt_parts
from pipeline.resolvers.records import BattleContext
from pipeline.stages.resolve import (
    _MAX_SPLIT_DEPTH,
    LazyService,
    _build_identities,
    _GroupResolver,
    resolve_groups,
)
from tests.unit.test_resolve import _StubService

# sha256 of the user prompt the code rendered before the guard existed, for the
# group built in _unsplit_group_prompt. The request hash is the LLM cache key,
# so a group that is not split must keep rendering exactly this.
_PRE_GUARD_PROMPT_SHA256 = "3457ea0aeec45f8e76ebf546d51929023ca101e4a871dec80d4b372ef4ab4be1"


def _mention(name: str, battle: str, year: int | None, *, evidence: str = "") -> Mention:
    slug = battle.replace(" ", "_")
    return Mention(
        battle_slug=slug,
        battle_name=battle,
        side_label="Rome",
        name=name,
        role_evidence=evidence,
        context=BattleContext(
            slug=slug, name=battle, year=year, war="Second Punic War", date_text=str(year)
        ),
    )


class _Source:
    def __init__(self, candidates: list[Candidate]) -> None:
        self.candidates = candidates

    def prefetch(self, groups: Any) -> None:
        return None

    def search(self, group: MentionGroup) -> list[Candidate]:
        return list(self.candidates)

    def non_person_kind(self, group: MentionGroup) -> str | None:
        return None


def _resolve(
    mentions: list[Mention],
    candidates: list[Candidate],
    service: _StubService | None = None,
) -> tuple[list[MentionGroup], dict[str, Decision]]:
    system, template, schema = prompt_parts(load_agent_spec("resolve"))
    groups = group_mentions(mentions)
    stub = service or _StubService({"match": "NEW_ENTITY", "confidence": 0.5})
    decisions = resolve_groups(
        groups,
        _Source(candidates),
        LazyService(lambda: stub),  # type: ignore[arg-type,return-value]
        threshold=85.0,
        margin=6.0,
        counts=ResolveCounts(),
        system=system,
        template=template,
        schema=schema,
    )
    return groups, decisions


def _battles_of(groups: list[MentionGroup], decisions: dict[str, Decision], qid: str) -> set[str]:
    return {
        m.battle_name
        for g in groups
        if decisions[g.key].qid == qid and decisions[g.key].status == "linked"
        for m in g.mentions
    }


def test_munda_is_detached_from_pompey_the_great() -> None:
    pompey = Candidate(qid="Q1347", label="Pompey", birth_year=-105, death_year=-47)
    mentions = [
        _mention("Pompey", "Battle of Pharsalus", -47),
        _mention("Pompey", "Battle of Munda", -44),
    ]

    groups, decisions = _resolve(mentions, [pompey])

    assert _battles_of(groups, decisions, "Q1347") == {"Battle of Pharsalus"}
    detached = next(g for g in groups if g.split_from)
    assert [m.battle_name for m in detached.mentions] == ["Battle of Munda"]
    assert decisions[detached.key].status == "unresolved"
    assert decisions[detached.key].method == "split_unmatched"
    assert "Q1347" in detached.split_reason


def test_nasica_born_in_228_bc_cannot_command_at_the_ticinus() -> None:
    nasica = Candidate(qid="Q453860", label="Scipio Nasica", birth_year=-227, death_year=-190)

    assert lifespan_verdict(nasica, [-217]) is False
    # Fifteen years after birth, less the two years of slack, he may command.
    assert lifespan_verdict(nasica, [-214]) is True


def test_the_minimum_command_age_does_not_move_the_half_dated_upper_bound() -> None:
    born_only = Candidate(qid="Q1", label="X", birth_year=-300)

    assert lifespan_verdict(born_only, [-300 + 100 + 2]) is True
    assert lifespan_verdict(born_only, [-300 + 100 + 3]) is False
    assert lifespan_verdict(born_only, [-300 + 10]) is False


def test_prague_1757_is_detached_from_charles_of_lorraine_but_chotusitz_stays() -> None:
    charles = Candidate(
        qid="Q3659993", label="Charles of Lorraine", birth_year=1684, death_year=1751
    )
    mentions = [
        _mention("Charles of Lorraine", "Battle of Chotusitz", 1742),
        _mention("Charles of Lorraine", "Battle of Soor", 1745),
        _mention("Charles of Lorraine", "Battle of Prague", 1757),
    ]

    groups, decisions = _resolve(mentions, [charles])

    assert _battles_of(groups, decisions, "Q3659993") == {
        "Battle of Chotusitz",
        "Battle of Soor",
    }
    assert [m.battle_name for g in groups if g.split_from for m in g.mentions] == [
        "Battle of Prague"
    ]


def test_a_detached_mention_is_matched_again_on_its_own_years() -> None:
    elder = Candidate(qid="Q10", label="Charles of Lorraine", birth_year=1684, death_year=1751)
    younger = Candidate(qid="Q11", label="Charles of Lorraine", birth_year=1712, death_year=1780)
    mentions = [
        _mention("Charles of Lorraine", "Battle of Chotusitz", 1742),
        _mention("Charles of Lorraine", "Battle of Prague", 1757),
    ]
    # The elder is only confirmed for 1742, and the younger for both, so the
    # whole group is ambiguous and goes to the model, which picks the elder.
    service = _StubService({"match": "Q10", "confidence": 0.8})

    groups, decisions = _resolve(mentions, [elder, younger], service)

    assert _battles_of(groups, decisions, "Q10") == {"Battle of Chotusitz"}
    assert _battles_of(groups, decisions, "Q11") == {"Battle of Prague"}


def test_a_1948_mention_is_split_from_caesars_group() -> None:
    caesar = Candidate(qid="Q1048", label="Julius Caesar", birth_year=-99, death_year=-43)
    modern = Candidate(qid="Q9", label="Julius Caesar", birth_year=1900, death_year=1990)
    mentions = [
        _mention("Julius Caesar", "Battle of Alesia", -51),
        _mention("Julius Caesar", "Battle of Munda", -44),
        _mention("Julius Caesar", "A Modern Battle", 1948),
    ]

    groups, decisions = _resolve(mentions, [caesar, modern])

    assert _battles_of(groups, decisions, "Q1048") == {"Battle of Alesia", "Battle of Munda"}
    assert _battles_of(groups, decisions, "Q9") == {"A Modern Battle"}
    assert len({g.key for g in groups}) == len(groups) == 2


def test_clusters_keep_undated_mentions_with_the_largest_one() -> None:
    mentions = [
        _mention("X", "B1", -200),
        _mention("X", "B2", -190),
        _mention("X", "B3", 1900),
        _mention("X", "B4", None),
    ]

    clusters = career_clusters(mentions)

    assert [[m.battle_name for m in c] for c in clusters] == [["B1", "B2", "B4"], ["B3"]]


def test_a_gap_of_exactly_one_career_does_not_split() -> None:
    mentions = [_mention("X", "B1", 1000), _mention("X", "B2", 1060)]

    assert len(career_clusters(mentions)) == 1


def test_a_split_off_part_with_no_candidate_is_not_made_a_new_general() -> None:
    mentions = [_mention("Nobody Known", "B1", 1700), _mention("Nobody Known", "B2", 1900)]

    groups, decisions = _resolve(mentions, [])

    by_key = {g.key: decisions[g.key] for g in groups}
    statuses = sorted(d.status for d in by_key.values())
    assert statuses == ["new", "unresolved"]


def test_the_detach_recursion_is_bounded_and_terminates() -> None:
    # Years within one career of each other so no lifetime split applies.
    mentions = [_mention("Same Name", f"B{i}", 1000 + i) for i in range(1, 4)]
    candidates = [
        Candidate(qid="Q1", label="Same Name", birth_year=960, death_year=1002),
        Candidate(qid="Q2", label="Same Name", birth_year=980, death_year=1003),
        Candidate(qid="Q3", label="Same Name", birth_year=985, death_year=1040),
    ]

    groups, decisions = _resolve(mentions, candidates)

    assert len(groups) <= 4
    assert sum(len(g.mentions) for g in groups) == 3
    assert all(d.status in {"linked", "unresolved", "new", "ambiguous"} for d in decisions.values())


def test_an_undated_mention_stays_with_its_group() -> None:
    pompey = Candidate(qid="Q1347", label="Pompey", birth_year=-105, death_year=-47)
    mentions = [
        _mention("Pompey", "Battle of Pharsalus", -47),
        _mention("Pompey", "An Undated Battle", None),
    ]

    kept, detached = partition_by_lifespan(mentions, pompey)

    assert len(kept) == 2
    assert detached == []


def _unsplit_group_prompt() -> str:
    mentions = [
        _mention("Scipio", "Battle of Zama", -201, evidence="led the right wing"),
        _mention("Scipio", "Battle of Ilipa", -205, evidence="led the right wing"),
    ]
    candidates = [
        Candidate(
            qid="Q100",
            label="Scipio Africanus",
            aliases=("Scipio",),
            birth_year=-235,
            death_year=-182,
            description="Roman general",
            sitelinks=100,
        ),
        Candidate(
            qid="Q101",
            label="Scipio",
            aliases=("Scipio",),
            birth_year=-240,
            death_year=-180,
            description="Another",
            sitelinks=5,
        ),
    ]
    service = _StubService({"match": "NEW_ENTITY", "confidence": 0.5})
    _resolve(mentions, candidates, service)
    assert len(service.calls) == 1
    return service.calls[0]


def test_a_group_that_is_not_split_renders_the_same_prompt_as_before() -> None:
    prompt = _unsplit_group_prompt()

    assert hashlib.sha256(prompt.encode()).hexdigest() == _PRE_GUARD_PROMPT_SHA256


def _two_era_candidates() -> list[Candidate]:
    return [
        Candidate(
            qid="Q1", label="Jan Novak", aliases=("Novak",), birth_year=1680, death_year=1740
        ),
        Candidate(
            qid="Q2", label="Jan Novak", aliases=("Novak",), birth_year=1690, death_year=1750
        ),
        Candidate(
            qid="Q3", label="Jan Novak", aliases=("Novak",), birth_year=1860, death_year=1920
        ),
        Candidate(
            qid="Q4", label="Jan Novak", aliases=("Novak",), birth_year=1870, death_year=1930
        ),
    ]


@pytest.mark.parametrize("answer", ["NEW_ENTITY", "UNCERTAIN", "Q999"])
def test_a_split_off_group_the_llm_calls_new_is_left_unresolved(answer: str) -> None:
    mentions = [_mention("Jan Novak", "B1", 1700), _mention("Jan Novak", "B2", 1900)]
    stub = _StubService({"match": answer, "confidence": 0.5})

    groups, decisions = _resolve(mentions, _two_era_candidates(), stub)

    split = [g for g in groups if g.split_from]
    assert len(split) == 1
    assert any("B2" in call for call in stub.calls)
    assert decisions[split[0].key].status == "unresolved"
    assert decisions[split[0].key].method == "split_unmatched"


def test_the_depth_cap_still_detaches_the_mentions_outside_the_window() -> None:
    mentions = [
        _mention("Pompey", "Battle of Pharsalus", -47),
        _mention("Pompey", "Battle of Munda", -44),
    ]
    pompey = Candidate(qid="Q1347", label="Pompey", birth_year=-105, death_year=-47)
    groups = group_mentions(mentions)
    resolver = _GroupResolver(groups, _Source([pompey]), threshold=85.0, margin=6.0)
    resolver._depth[groups[0].key] = _MAX_SPLIT_DEPTH
    resolver.decide_from(0)

    detached = [g for g in groups if g.split_from]
    assert [m.battle_name for g in detached for m in g.mentions] == ["Battle of Munda"]
    assert resolver.decisions[detached[0].key].method == "split_unmatched"
    assert resolver.decisions[groups[0].key].status == "linked"
    assert [m.battle_name for m in groups[0].mentions] == ["Battle of Pharsalus"]


def test_a_detached_part_does_not_lend_its_surface_forms_to_the_kept_general() -> None:
    pompey = Candidate(qid="Q1347", label="Pompey", birth_year=-105, death_year=-47)
    mentions = [
        _mention("Pompey", "Battle of Pharsalus", -47),
        _mention("Gnaeus Pompeius Junior", "Battle of Munda", -44),
    ]
    groups = group_mentions(mentions)
    # One shared group, as a loose name key would build it.
    merged = groups[0]
    if len(groups) > 1:
        merged.mentions.extend(groups[1].mentions)
        merged.surface_forms = sorted({*merged.surface_forms, *groups[1].surface_forms})
        merged.keys = (*merged.keys, *groups[1].keys)
        groups = [merged]
    resolver = _GroupResolver(groups, _Source([pompey]), threshold=85.0, margin=6.0)
    resolver.decide_from(0)

    kept = groups[0]
    detached = next(g for g in groups if g.split_from)
    identities = _build_identities(groups, resolver.decisions)

    assert detached.surface_forms == ["Gnaeus Pompeius Junior"]
    assert "Gnaeus Pompeius Junior" not in kept.surface_forms
    assert all("Gnaeus Pompeius Junior" not in identity.aliases for identity in identities)
