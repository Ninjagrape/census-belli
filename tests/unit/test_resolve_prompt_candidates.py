"""
The candidates the LLM disambiguation prompt actually shows.

The prompt lists at most ``MAX_CANDIDATES_IN_PROMPT`` candidates. The stage
used to hand it the candidate source's raw order, so on the real corpus Julius
Caesar (11th of 11 for his own name) and Hannibal Barca (15th of 15) were cut
from their own prompts and the model was left choosing between modern
namesakes. These tests rebuild that shape with synthetic candidates: no
network, no database.
"""

from __future__ import annotations

from typing import Any

from pipeline.config import load_agent_spec
from pipeline.resolvers import (
    Candidate,
    Mention,
    MentionGroup,
    ResolveCounts,
    group_mentions,
    rank_for_prompt,
)
from pipeline.resolvers.candidates import _merge_rows
from pipeline.resolvers.disambiguate import MAX_CANDIDATES_IN_PROMPT, prompt_parts
from pipeline.resolvers.records import BattleContext
from pipeline.stages.resolve import LazyService, resolve_groups
from tests.unit.test_resolve import _StubService


def _mention(name: str, battle: str, year: int) -> Mention:
    slug = battle.replace(" ", "_")
    return Mention(
        battle_slug=slug,
        battle_name=battle,
        side_label="Rome",
        name=name,
        context=BattleContext(slug=slug, name=battle, year=year),
    )


def _namesakes(label: str, count: int, *, born: int, died: int | None) -> list[Candidate]:
    """Modern namesakes with low Q-ids, i.e. first in the endpoint's order."""
    return [
        Candidate(
            qid=f"Q{100 + i}",
            label=label,
            birth_year=born + i,
            death_year=None if died is None else died + i,
            description="modern namesake",
        )
        for i in range(count)
    ]


class _Source:
    def __init__(self, candidates: list[Candidate]) -> None:
        self.candidates = candidates

    def prefetch(self, groups: Any) -> None:
        return None

    def search(self, group: MentionGroup) -> list[Candidate]:
        return list(self.candidates)

    def non_person_kind(self, group: MentionGroup) -> str | None:
        return None


def _shown_qids(user_prompt: str) -> list[str]:
    return [
        line[2:].split(":", 1)[0] for line in user_prompt.splitlines() if line.startswith("- Q")
    ]


def _run_prompt(mentions: list[Mention], candidates: list[Candidate]) -> str:
    system, template, schema = prompt_parts(load_agent_spec("resolve"))
    service = _StubService({"match": "NEW_ENTITY", "confidence": 0.5})
    resolve_groups(
        group_mentions(mentions),
        _Source(candidates),
        LazyService(lambda: service),  # type: ignore[arg-type,return-value]
        threshold=85.0,
        margin=6.0,
        counts=ResolveCounts(),
        system=system,
        template=template,
        schema=schema,
    )
    assert len(service.calls) == 1, "the group should have gone to the model"
    return service.calls[0]


def test_a_famous_late_listed_candidate_survives_the_prompt_cap() -> None:
    # The group's years include a stray modern one, as the real Caesar group
    # does (1948), so the date gate keeps modern namesakes alive alongside him.
    caesar = Candidate(
        qid="Q1048",
        label="Julius Caesar",
        birth_year=-98,
        death_year=-42,
        occupations=("military leader",),
        sitelinks=250,
    )
    modern = _namesakes("Julius Caesar", 10, born=1900, died=1990)
    mentions = [
        _mention("Julius Caesar", "Battle of Alesia", -51),
        _mention("Julius Caesar", "A Modern Battle", 1948),
    ]

    prompt = _run_prompt(mentions, [*modern, caesar])

    shown = _shown_qids(prompt)
    assert len(shown) == MAX_CANDIDATES_IN_PROMPT
    assert shown[0] == "Q1048"


def test_hannibal_barca_beats_dateless_namesakes_into_the_prompt() -> None:
    barca = Candidate(
        qid="Q36456", label="Hannibal", birth_year=-245, death_year=-181, sitelinks=180
    )
    dateless = [Candidate(qid=f"Q{100 + i}", label="Hannibal") for i in range(12)]
    # The real group carries a stray 2015 access-date year, which keeps a
    # living namesake date-compatible and the group ambiguous.
    living = Candidate(qid="Q50", label="Hannibal", birth_year=1960)
    mentions = [
        _mention("Hannibal", "Battle of Ager Falernus", -216),
        _mention("Hannibal", "A Modern Battle", 2015),
    ]

    prompt = _run_prompt(mentions, [*dateless, living, barca])

    assert _shown_qids(prompt)[0] == "Q36456"


def test_the_date_gate_still_filters_before_the_cap() -> None:
    group = group_mentions([_mention("Hannibal", "Battle of Cannae", -215)])[0]
    modern = Candidate(qid="Q5", label="Hannibal", birth_year=1850, death_year=1920)
    ancient = Candidate(qid="Q9", label="Hannibal", birth_year=-247, death_year=-183)

    assert [c.qid for c in rank_for_prompt(group, [modern, ancient])] == ["Q9"]


def test_sitelinks_order_otherwise_equal_candidates() -> None:
    group = group_mentions([_mention("Sidney Smith", "Siege of Acre", 1799)])[0]
    obscure = Candidate(qid="Q1", label="Sidney Smith")
    famous = Candidate(qid="Q9999", label="Sidney Smith", sitelinks=40)

    assert [c.qid for c in rank_for_prompt(group, [obscure, famous])] == ["Q9999", "Q1"]


def test_ranking_is_stable_across_input_order() -> None:
    group = group_mentions([_mention("Hannibal", "Battle of Cannae", -215)])[0]
    pool = [Candidate(qid=f"Q{n}", label="Hannibal") for n in (7, 3, 5)]

    assert rank_for_prompt(group, pool) == rank_for_prompt(group, list(reversed(pool)))


def test_sitelinks_are_read_from_the_query_rows() -> None:
    rows = [
        {
            "person": "http://www.wikidata.org/entity/Q1048",
            "name": "Julius Caesar",
            "sitelinks": "250",
        },
        {
            "person": "http://www.wikidata.org/entity/Q2",
            "name": "Julius Caesar",
            "sitelinks": "n/a",
        },
    ]

    by_qid = {c.qid: c for c, _ in _merge_rows(rows)}

    assert by_qid["Q1048"].sitelinks == 250
    assert by_qid["Q2"].sitelinks == 0


def test_the_rendered_prompt_has_single_braces_only() -> None:
    prompt = _run_prompt(
        [_mention("Scipio", "Battle of Zama", -201)],
        [
            Candidate(qid="Q1", label="Scipio", birth_year=-235, death_year=-183),
            Candidate(qid="Q2", label="Scipio", birth_year=-220, death_year=-180),
        ],
    )

    assert '{"match":' in prompt
    assert "{{" not in prompt and "}}" not in prompt


def test_multi_valued_fields_fold_in_a_stable_order() -> None:
    # GROUP_CONCAT order is the endpoint's and varies between identical
    # queries; the prompt is hashed for the llm_calls cache.
    def fold_with(occupations: str, countries: str) -> Candidate:
        row = {
            "person": "http://www.wikidata.org/entity/Q5",
            "name": "X",
            "occupations": occupations,
            "countries": countries,
        }
        return _merge_rows([row])[0][0]

    one = fold_with("writer, general, politician", "Rome, Carthage")
    two = fold_with("politician, writer, general", "Carthage, Rome")

    assert one.occupations == two.occupations == ("general", "politician", "writer")
    assert one.country == two.country == "Carthage, Rome"


def test_the_prompt_text_is_identical_whatever_the_concat_order() -> None:
    def candidates(occupations: str) -> list[Candidate]:
        rows = [
            {
                "person": f"http://www.wikidata.org/entity/Q{n}",
                "name": "Scipio",
                "occupations": occupations,
            }
            for n in (1, 2)
        ]
        return [c for c, _ in _merge_rows(rows)]

    mentions = [_mention("Scipio", "Battle of Zama", -201)]

    assert _run_prompt(mentions, candidates("a, b, c")) == _run_prompt(
        mentions, candidates("c, a, b")
    )
