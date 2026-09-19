"""
Entity resolution for the resolve stage.

The stage funnels many surface mentions into few canonical people, in the
order ``agents/resolve.yaml`` lays out: group the mentions, look up
candidates, match deterministically, and ask a model only about what is left.

Each step is a module, and each is pure apart from the two at the edges --
:mod:`pipeline.resolvers.candidates`, which may talk to Wikidata, and
:mod:`pipeline.resolvers.store`, which writes rows. Everything between them
takes records and returns records, which is what lets the matcher be tested
on the cases that matter (a name shared by five Romans, a commander dead
before the battle) without a network or a database.

Typical use::

    from pipeline.resolvers import group_mentions, load_mentions, match_group

    mentions = load_mentions(processed / "commanders_raw.jsonl",
                             processed / "battles.jsonl")
    for group in group_mentions(mentions):
        decision = match_group(group, source.search(group), threshold=85.0)
"""

from __future__ import annotations

from pipeline.resolvers.battles import DuplicateBattle, find_duplicate_battles
from pipeline.resolvers.candidates import (
    CandidateLookupError,
    CandidateSource,
    LocalCandidateSource,
    NullCandidateSource,
    PrefetchedCandidateSource,
    candidate_keys,
    collect_query_names,
    fetch_candidates,
    person_candidates,
    query_names,
)
from pipeline.resolvers.disambiguate import disambiguate, format_candidates, prompt_parts
from pipeline.resolvers.matcher import (
    AMBIGUITY_MARGIN,
    best_score,
    lifespan_verdict,
    match_against_identities,
    match_group,
    viable_candidates,
)
from pipeline.resolvers.mentions import (
    astronomical_year,
    group_mentions,
    load_battle_contexts,
    load_mentions,
)
from pipeline.resolvers.names import (
    clean_surface,
    fold,
    is_placeholder,
    is_title_free,
    name_keys,
)
from pipeline.resolvers.records import (
    METHODS,
    BattleContext,
    Candidate,
    Decision,
    Identity,
    Mention,
    MentionGroup,
    ResolveCounts,
    to_jsonable,
)
from pipeline.resolvers.store import (
    UNRESOLVED_FIELD,
    load_battle_index,
    load_side_index,
    log_unresolved,
    write_identity,
)

__all__ = [
    "AMBIGUITY_MARGIN",
    "METHODS",
    "UNRESOLVED_FIELD",
    "BattleContext",
    "Candidate",
    "CandidateLookupError",
    "CandidateSource",
    "Decision",
    "DuplicateBattle",
    "Identity",
    "LocalCandidateSource",
    "Mention",
    "MentionGroup",
    "NullCandidateSource",
    "ResolveCounts",
    "PrefetchedCandidateSource",
    "astronomical_year",
    "best_score",
    "candidate_keys",
    "clean_surface",
    "collect_query_names",
    "disambiguate",
    "fetch_candidates",
    "find_duplicate_battles",
    "fold",
    "format_candidates",
    "group_mentions",
    "is_placeholder",
    "is_title_free",
    "lifespan_verdict",
    "load_battle_contexts",
    "load_battle_index",
    "load_mentions",
    "load_side_index",
    "log_unresolved",
    "match_against_identities",
    "match_group",
    "name_keys",
    "person_candidates",
    "prompt_parts",
    "query_names",
    "to_jsonable",
    "viable_candidates",
    "write_identity",
]
