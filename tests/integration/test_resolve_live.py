"""
The resolve stage's Wikidata lookup against the live SPARQL endpoint.

Every systematic bug this stage has found came from pointing it at a real
service rather than from reading code: robots.txt silently disabling every
SPARQL query in the project (handover.md 4.7), alias matches thrown away
after the query found them (14.1), and Nelson's label having migrated to
Wikidata's ``mul`` tag, which an ``en``-only lookup cannot see (14.6, 16.1).
The last of those is exactly the shape of regression a fixture-only suite
cannot catch: the fixture would have to notice, on its own, that Wikidata
changed something out from under it. These tests query the real endpoint so
that class of drift has somewhere to show up.

**They are opt-in and skipped by default.** They make real requests to a
third party, so they must not run in CI, must not run on every ``pytest``,
and must not be something a contributor triggers by accident:

    GENERAL_WAR_LIVE_CRAWL=1 python -m pytest tests/integration/test_resolve_live.py -v

Gated on the same flag as ``test_crawl_live.py`` rather than a resolve-specific
one: both hit real third-party services, and TODO.md's own task line for this
module names ``GENERAL_WAR_LIVE_CRAWL=1``.

Politeness: every commander this module needs is looked up in one SPARQL
batch, sharing one :class:`~pipeline.crawlers.fetcher.Fetcher` and the rate
limit ``agents/resolve.yaml`` configures for the query service
(``rate_limit_sparql``, 2.0s as shipped) -- the same reasoning
``test_crawl_live.py`` gives for building its fetcher from the shipped spec
rather than test defaults.
"""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import Sequence
from typing import Any, Final

import httpx
import pytest

from pipeline.config import load_agent_spec
from pipeline.crawlers.fetcher import Fetcher, RateLimiter
from pipeline.resolvers import (
    BattleContext,
    Candidate,
    Mention,
    MentionGroup,
    fetch_candidates,
    group_mentions,
    match_group,
    viable_candidates,
)

_ENV_FLAG = "GENERAL_WAR_LIVE_CRAWL"

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.environ.get(_ENV_FLAG),
        reason=f"live network test; set {_ENV_FLAG}=1 to run",
    ),
]

# agents/resolve.yaml declares fuzzy_threshold, ambiguity_margin,
# sparql_batch_size, max_names_per_group, rate_limit_sparql and
# respect_robots, but not user_agent or a request timeout -- the stage
# (pipeline/stages/resolve.py) supplies those two itself. Mirrored here
# rather than imported, so this module states plainly what it runs under,
# the same reasoning test_crawl_live.py gives for reading the shipped spec.
_DEFAULT_USER_AGENT: Final[str] = (
    "general-war-research/0.1 (https://github.com/census-belli; entity resolution)"
)
_DEFAULT_TIMEOUT_S: Final[float] = 60.0

# A bare item id, e.g. "Q517" -- never a name. handover.md 16.2 records the
# label service returning one in place of a label it could not resolve, and
# candidates.py._QID_SHAPE is the same pattern; duplicated here rather than
# imported because it is a private module constant and this check should
# hold regardless of how the production code happens to spell it.
_QID_SHAPE: Final[re.Pattern[str]] = re.compile(r"Q\d+")

# Every Q-id below was looked up live against wikidata.org on 2026-09-20, not
# recalled from memory -- handover.md 14.6 and 16.1 both record what happens
# when one is remembered instead. Worth restating because it already bit this
# exact set once: Q47153 is not Hannibal Barca, it is a 2009 novel. The
# correct id is Q36456.
NELSON_QID: Final[str] = "Q83235"
NAPOLEON_QID: Final[str] = "Q517"
WELLINGTON_QID: Final[str] = "Q131691"
CAESAR_QID: Final[str] = "Q1048"
HANNIBAL_BARCA_QID: Final[str] = "Q36456"

# The surface form each commander is queried under. "Duke of Wellington" is
# an *alias* of his Wikidata label ("Arthur Wellesley, 1st Duke of
# Wellington"), which is the case 14.1 found being discarded after the query
# had already found it.
KNOWN_COMMANDERS: Final[dict[str, str]] = {
    "Horatio Nelson": NELSON_QID,
    "Napoleon": NAPOLEON_QID,
    "Duke of Wellington": WELLINGTON_QID,
    "Julius Caesar": CAESAR_QID,
}

_HANNIBAL_NAME: Final[str] = "Hannibal"


def live_params() -> dict[str, Any]:
    """Read the shipped resolve spec's parameters.

    Using the real spec rather than test defaults is the point: a rate limit
    or batch size too aggressive for the query service should fail here
    rather than be dodged by a gentler test-only value.

    Returns:
        The ``params`` mapping from ``agents/resolve.yaml``.
    """
    return dict(load_agent_spec("resolve").get("params") or {})


async def _fetch_all(names: Sequence[str]) -> dict[str, list[Candidate]]:
    """Query the live Wikidata endpoint for a batch of surface forms.

    Args:
        names: Every name this module's tests need candidates for.

    Returns:
        Each requested name mapped to its candidates, as
        :func:`pipeline.resolvers.fetch_candidates` returns them.
    """
    params = live_params()
    user_agent = str(params.get("user_agent", _DEFAULT_USER_AGENT))
    rate_limit = float(params.get("rate_limit_sparql", 2.0))
    batch_size = int(params.get("sparql_batch_size", 25))
    respect_robots = bool(params.get("respect_robots", True))

    async with httpx.AsyncClient(
        timeout=_DEFAULT_TIMEOUT_S,
        follow_redirects=True,
        headers={"User-Agent": user_agent},
    ) as client:
        fetcher = Fetcher(
            client,
            limiter=RateLimiter(default_interval=rate_limit),
            user_agent=user_agent,
            respect_robots=respect_robots,
        )
        return await fetch_candidates(fetcher, list(names), batch_size=batch_size)


@pytest.fixture(scope="module")
def live_candidates() -> dict[str, list[Candidate]]:
    """Query Wikidata once for every commander this module tests.

    Module-scoped so five tests cost one SPARQL batch rather than five: at
    the shipped batch size (25) every name here fits in a single query.

    Returns:
        Surface form to candidates, from the live endpoint.
    """
    names = [*KNOWN_COMMANDERS, _HANNIBAL_NAME]
    return asyncio.run(_fetch_all(names))


def _group_for(name: str, *, year: int, battle: str) -> MentionGroup:
    """Build a one-mention group with a dated battle context.

    Args:
        name: The commander's surface form, as an infobox would write it.
        year: Astronomical year of the battle.
        battle: The battle's display name.

    Returns:
        A :class:`MentionGroup` of exactly one mention, as
        :func:`pipeline.resolvers.group_mentions` would build it from
        ``commanders_raw.jsonl`` and ``battles.jsonl``.
    """
    slug = battle.lower().replace(" ", "-")
    mention = Mention(
        battle_slug=slug,
        battle_name=battle,
        side_label="side",
        name=name,
        context=BattleContext(slug=slug, name=battle, year=year),
    )
    groups = group_mentions([mention])
    return groups[0]


def test_known_commanders_have_candidates(live_candidates: dict[str, list[Candidate]]) -> None:
    """Four commanders whose identity is not in doubt still resolve to it.

    Not a claim that the matcher will link them -- only that the live query
    still returns their entity at all. A commander missing here means either
    the endpoint changed shape or a name/label went stale, and either is
    worth knowing before it shows up as a silent drop in a real run.
    """
    missing: list[tuple[str, str, list[str]]] = []

    for name, expected_qid in KNOWN_COMMANDERS.items():
        candidates = live_candidates.get(name, [])
        qids = sorted({c.qid for c in candidates})
        if expected_qid not in qids:
            missing.append((name, expected_qid, qids))

    assert not missing, f"expected commander not among live candidates: {missing}"


def test_nelson_reachable_via_mul_label(live_candidates: dict[str, list[Candidate]]) -> None:
    """The regression from handover.md 16.1: Nelson has no ``en`` label at all.

    Q83235's labels are ``en-gb`` ("Horatio Nelson, 1st Viscount Nelson") and
    ``mul`` ("Horatio Nelson") -- Wikidata's multilingual code, onto which
    person labels have been migrating since 2024. An ``en``-only literal
    lookup cannot see him; ``_NAME_LANGUAGE_TAGS`` in
    ``pipeline/resolvers/candidates.py`` asks in both ``en`` and ``mul`` for
    exactly this reason. If Wikidata ever migrates the label back to ``en``,
    or removes the ``mul`` one, this is the test that should notice.
    """
    qids = {c.qid for c in live_candidates.get("Horatio Nelson", [])}
    assert NELSON_QID in qids, (
        "Nelson (Q83235) was not among the live candidates for 'Horatio "
        "Nelson'. He carries no 'en' label, only 'mul' -- see handover.md "
        "16.1. Either the mul label has been removed, or the lookup has "
        "stopped asking for it."
    )


def test_no_candidate_label_is_qid_shaped(live_candidates: dict[str, list[Candidate]]) -> None:
    """No candidate is ever published under its own Q-id as a label.

    handover.md 16.2: the Wikidata label service returns the bare item id in
    place of a label it could not resolve in the requested languages, and an
    early version of ``_merge_rows`` accepted that string as a name. On a
    link it becomes ``generals.canonical_name`` -- a general silently
    published as "Q83235". ``_preferred_name`` now falls back to the longest
    matched surface form instead, and this is the regression test for it
    against real endpoint output, not a fixture built to already agree.
    """
    offenders = [
        (name, candidate.qid, candidate.label)
        for name, candidates in live_candidates.items()
        for candidate in candidates
        if _QID_SHAPE.fullmatch(candidate.label.strip())
    ]
    assert not offenders, (
        f"a live candidate's label is Q-id shaped rather than a name: {offenders}. "
        "See handover.md 16.2."
    )


def test_date_gate_separates_centuries(live_candidates: dict[str, list[Candidate]]) -> None:
    """"Hannibal" in an 1811 battle must not reach Hannibal Barca (247-183 BC).

    This is the assertion behind ``matcher.py``'s whole design: name
    similarity alone cannot tell a Carthaginian general from anyone else
    later named after him, and the lifespan gate is what does. It is also
    the specific case ``MAX_PLAUSIBLE_AGE_YEARS`` (handover.md 16.3) exists
    for: Hannibal Barca's death claim is an explicit "no value" on Wikidata,
    so an unbounded upper end would have judged him alive, and positively so,
    in 1811.
    """
    candidates = live_candidates.get(_HANNIBAL_NAME, [])
    assert candidates, "live query returned no candidates at all for 'Hannibal'"

    group = _group_for(_HANNIBAL_NAME, year=1811, battle="Battle of Lissa")
    viable = viable_candidates(candidates, group.years)
    viable_qids = {c.qid for c in viable}

    assert HANNIBAL_BARCA_QID not in viable_qids, (
        "Hannibal Barca (Q36456, 247-183 BC) survived the date gate for an "
        "1811 battle. The lifespan gate in matcher.py is no longer "
        "separating centuries -- see handover.md 12.2 and 16.3."
    )


def test_known_commanders_resolve_deterministically(
    live_candidates: dict[str, list[Candidate]],
) -> None:
    """Napoleon at Austerlitz and Nelson at Trafalgar link with no LLM call.

    Both are the worked examples in handover.md 13.2 and 16.1: against real
    candidates, an unbounded name match would leave each ambiguous against a
    dateless namesake (a rapper, an NFT founder), and the date-confirmed
    exact-match rule is what resolves them without spending LLM quota. This
    exercises ``match_group`` directly rather than the whole stage, so it
    needs no database and no LLM key.
    """
    cases: tuple[tuple[str, int, str, str], ...] = (
        ("Napoleon", 1805, "Battle of Austerlitz", NAPOLEON_QID),
        ("Horatio Nelson", 1805, "Battle of Trafalgar", NELSON_QID),
    )
    failures: list[tuple[str, str, str | None, str]] = []

    for name, year, battle, expected_qid in cases:
        group = _group_for(name, year=year, battle=battle)
        candidates = live_candidates.get(name, [])
        decision = match_group(group, candidates)
        if decision.status != "linked" or decision.qid != expected_qid:
            failures.append((name, decision.status, decision.qid, decision.reasoning))

    assert not failures, (
        f"expected a deterministic link with no LLM step, got: {failures}"
    )
