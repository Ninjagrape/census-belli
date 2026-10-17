"""
The resolve stage: commander mentions to canonical people.

The extract stage left ``battle_commanders`` empty on purpose. Its
``general_id`` is NOT NULL, and until somebody decides that the "Agrippa" of
one article and the "Marcus Vipsanius Agrippa" of another are one man, there
is no row to point at. This stage decides, and it is the stage that finally
populates that table.

The order is the one ``agents/resolve.yaml`` sets out, cheapest first:

1. Group mentions by normalised name, so a commander in forty battles is
   resolved once rather than forty times.
2. Look up Wikidata candidates for each group.
3. Match deterministically -- exact label, then fuzzy with a date gate.
4. Attach whatever is left to identities the corpus has already resolved.
5. Ask the model only about groups that are still ambiguous.

Everything the stage decided lands in ``data/processed/resolution_log.jsonl``,
one row per group, whether it resolved or not. That file is the audit trail: a
published ranking rests on these merges, and a merge nobody can inspect is a
merge nobody can challenge.

What cannot be resolved is written to ``missing_data_log`` rather than
dropped, which is what stops the ``resolution_rate`` gate measuring itself.
"""

from __future__ import annotations

import asyncio
import json
import statistics
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Final

import httpx
import structlog

from pipeline.crawlers.fetcher import Fetcher, RateLimiter
from pipeline.imputers.priors import DEFAULT_PRIORS_PATH, era_of_year, load_priors
from pipeline.llm import LLMService
from pipeline.overrides import OverrideReport, clear_report, load_overrides, overrides_path
from pipeline.resolvers import (
    AMBIGUITY_MARGIN,
    Candidate,
    CandidateSource,
    Decision,
    DuplicateBattle,
    Identity,
    LocalCandidateSource,
    Mention,
    MentionGroup,
    NullCandidateSource,
    PrefetchedCandidateSource,
    ResolveCounts,
    career_clusters,
    collect_query_names,
    disambiguate,
    fetch_candidates,
    fetch_non_person_kinds,
    find_duplicate_battles,
    fold,
    group_mentions,
    is_placeholder,
    load_battle_index,
    load_mentions,
    load_side_index,
    log_unresolved,
    match_against_identities,
    match_group,
    non_person_reason,
    partition_by_lifespan,
    prompt_parts,
    prune_stale_commanders,
    rank_for_prompt,
    remove_orphan_generals,
    side_of,
    write_identity,
)
from pipeline.resolvers.battles import DEFAULT_NAME_THRESHOLD, DEFAULT_YEAR_SLACK
from pipeline.resolvers.candidates import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_NAMES_PER_GROUP,
    candidate_keys,
    fetch_candidates_by_qid,
)
from pipeline.resolvers.names import clean_surface, name_keys
from pipeline.resolvers.overrides import ResolveOverrides
from pipeline.stages.base import StageContext, StageOutcome

__all__ = ["LazyService", "resolve_groups", "run"]

logger = structlog.get_logger()

PROCESSED_ROOT: Final[Path] = Path("data/processed")
RAW_ROOT: Final[Path] = Path("data/raw")

_COMMANDERS_FILE: Final[str] = "commanders_raw.jsonl"
_BATTLES_FILE: Final[str] = "battles.jsonl"
_RESOLUTION_LOG: Final[str] = "resolution_log.jsonl"
_DUPLICATE_BATTLES: Final[str] = "duplicate_battles.jsonl"
_UNPLACED_MENTIONS: Final[str] = "unplaced_mentions.jsonl"
_WIKIDATA_DIR: Final[str] = "wikidata"

# The query service is a shared, unpaid resource and it is stricter than the
# article API. One request every two seconds is slower than handover-log §5.2's
# floor for Wikipedia, deliberately.
_DEFAULT_SPARQL_RATE_LIMIT: Final[float] = 2.0
_DEFAULT_TIMEOUT_S: Final[float] = 60.0
_DEFAULT_USER_AGENT: Final[str] = (
    "general-war-research/0.1 (https://github.com/census-belli; entity resolution)"
)

_PLACEHOLDER_CONFIDENCE: Final[float] = 0.0

# How many surface forms of a still-unplaced person the ranked-search fallback
# asks about. Fewer than the exact lookup: a search returns many hits per name
# and the later forms are mostly near-duplicates of the first.
_DEFAULT_SEARCH_NAMES_PER_GROUP: Final[int] = 2
_DEFAULT_SEARCH_BATCH_SIZE: Final[int] = 10

# A person with no dated battles and no Wikidata birth year gets no era rather
# than a guess; a person with only a birth year is placed at the age they could
# first have commanded.
_ADULT_AGE_YEARS: Final[int] = 35


class LazyService:
    """An LLM service built on first use, not on principle.

    The extract stage builds its service up front, and should: every article
    goes through the model, so a missing key would fail the whole batch and
    failing immediately is kinder. Resolve is the other case. Most groups are
    settled by an exact Wikidata label or a fuzzy match, and a corpus that
    resolves cleanly needs no model at all -- so building the service eagerly
    would make an API key a hard requirement for work that never calls an API.

    Deferring it means a run with nothing ambiguous needs no credentials, and
    a run that does need the model still fails loudly, at the point it needs
    it, rather than resolving everything as a new entity.
    """

    def __init__(self, factory: Callable[[], LLMService] | None) -> None:
        """Wrap a factory.

        Args:
            factory: Builds the service, or None when this run must not use
                one (a dry run, or a test).
        """
        self._factory = factory
        self._service: LLMService | None = None

    @property
    def available(self) -> bool:
        """Whether a service could be built if one were needed."""
        return self._factory is not None

    @property
    def built(self) -> LLMService | None:
        """The service, if one was ever built."""
        return self._service

    def get(self) -> LLMService | None:
        """Build the service, or return the one already built.

        Returns:
            The service, or None when no factory was given.

        Raises:
            LLMConfigError: If the provider or credentials are unusable. This
                propagates: the groups that reached this point are exactly
                the ones no deterministic step could settle, and resolving
                them as new entities would bury the configuration problem in
                the data.
        """
        if self._factory is None:
            return None
        if self._service is None:
            self._service = self._factory()
        return self._service


def _placeholder_decision(group: MentionGroup) -> Decision:
    """Refuse to make a general out of a mention that names nobody.

    Args:
        group: The mention group.

    Returns:
        An unresolved decision. "Unknown" appears in thousands of infoboxes;
        resolving it would create one entity credited with every battle whose
        commander nobody recorded, and that entity would rank.
    """
    return Decision(
        status="unresolved",
        canonical_name=group.display_name,
        method="placeholder",
        confidence=_PLACEHOLDER_CONFIDENCE,
        reasoning="the mention names no one",
    )


def _non_person_decision(group: MentionGroup, reason: str) -> Decision:
    """Refuse to make a general out of an organisation, formation or place.

    Args:
        group: The mention group.
        reason: Why the mention cannot be a person, for the audit trail.

    Returns:
        An unresolved decision. "Luftwaffe" and "Stavka" appear as commanders
        in infoboxes; a general created for either would be credited with every
        battle it is named in and would rank like a man.
    """
    return Decision(
        status="unresolved",
        canonical_name=group.display_name,
        method="non_person",
        confidence=_PLACEHOLDER_CONFIDENCE,
        reasoning=f"the mention {reason}",
    )


def _stranded_decision(group: MentionGroup, decision: Decision) -> Decision:
    """Close out a group that is still ambiguous with no model to ask.

    Args:
        group: The mention group.
        decision: The ambiguous decision from the matcher.

    Returns:
        An unresolved decision carrying the matcher's reasoning. Taking the
        top candidate anyway is exactly the false merge the matcher declined
        to make, so the group is left for a run that has an LLM.
    """
    return Decision(
        status="unresolved",
        canonical_name=group.display_name,
        method="llm_failed",
        score=decision.score,
        runner_up=decision.runner_up,
        candidates_considered=decision.candidates_considered,
        reasoning=f"ambiguous and no LLM service available: {decision.reasoning}",
    )


def _identity_key(decision: Decision) -> str:
    """The key that decides which groups become one person.

    Args:
        decision: A resolved decision.

    Returns:
        The Q-id when the group linked to Wikidata, otherwise a corpus-local
        key folded from the canonical name. Two groups linking to the same
        entity therefore merge; two unlinked groups merge only if they
        settled on the same name.
    """
    if decision.qid:
        return decision.qid
    return f"new:{fold(decision.canonical_name)}"


def _build_identities(
    groups: list[MentionGroup],
    decisions: dict[str, Decision],
) -> list[Identity]:
    """Fold resolved groups into one identity per person.

    Args:
        groups: Every group the run resolved.
        decisions: Each group's final decision, keyed by group key. Groups
            absent from the mapping are skipped, which is how the anchor
            pass asks for linked identities only.

    Returns:
        One identity per canonical person, in first-seen order. Unresolved
        groups contribute nothing.
    """
    identities: dict[str, Identity] = {}

    for group in groups:
        decision = decisions.get(group.key)
        if decision is None or decision.status not in ("linked", "new"):
            continue

        key = _identity_key(decision)
        identity = identities.get(key)
        if identity is None:
            identity = Identity(
                key=key,
                canonical_name=decision.canonical_name,
                qid=decision.qid,
                wikipedia_url=decision.candidate.wikipedia_url if decision.candidate else "",
                nationality=decision.candidate.country if decision.candidate else "",
                confidence=decision.confidence,
                method=decision.method,
                birth_year=decision.candidate.birth_year if decision.candidate else None,
                death_year=decision.candidate.death_year if decision.candidate else None,
                candidate=decision.candidate,
            )
            identities[key] = identity
        elif decision.confidence > identity.confidence:
            # The identity is described by its best-evidenced group; each
            # row's own confidence still comes from its group's decision.
            identity.confidence = decision.confidence
            identity.method = decision.method

        identity.groups.append(group)
        identity.uncertain_split = identity.uncertain_split or decision.method == "llm_uncertain"
        for alias in group.surface_forms:
            if alias and alias not in identity.aliases:
                identity.aliases.append(alias)

    for identity in identities.values():
        identity.aliases = [a for a in identity.aliases if a != identity.canonical_name]

    return list(identities.values())


def _refuse_non_person(
    group: MentionGroup,
    found: list[Candidate],
    decision: Decision,
    source: CandidateSource,
) -> Decision:
    """Turn a would-be new entity into a refusal when Wikidata says it is no man.

    Args:
        group: The mention group.
        found: The human candidates the source returned for it.
        decision: The matcher's ``new`` decision.
        source: Where candidates came from; asked what the name is.

    Returns:
        A non-person decision when the name is the exact label or alias of an
        item that is not a human and no human with that exact name exists,
        otherwise the decision unchanged. The exact-human exception matters:
        "Gebhard von Blücher" is the label of a 1865 grandson who fails the
        date gate, and that must not make the name a non-person; the same goes
        for any commander who shares a label with a place.
    """
    group_keys = set(group.keys)
    if any(group_keys & set(candidate_keys(c)) for c in found):
        return decision
    kind = source.non_person_kind(group)
    if kind is None:
        return decision
    return _non_person_decision(group, f"is the name of a non-human Wikidata item ({kind})")


# Detached mentions are matched again on their own years. Each round must be
# viable for a candidate the previous rounds were not, so this ends on its own;
# the bound is a backstop against a candidate source that misbehaves.
_MAX_SPLIT_DEPTH: Final[int] = 4


def _battle_years(mentions: list[Mention]) -> list[int]:
    """Distinct astronomical years of the dated mentions, ascending."""
    return sorted({m.context.year for m in mentions if m.context and m.context.year is not None})


def _split_decision(group: MentionGroup, decision: Decision) -> Decision:
    """Leave a split-off part with no candidate unresolved, never a new general.

    Args:
        group: A group split off from a larger one.
        decision: The matcher's ``new`` decision for it.

    Returns:
        An unresolved decision. ``_upsert_general`` finds a corpus-local
        general by name, so a new entity for the detached part would be
        written onto the same row as its namesake and the split undone.
        Handover 9.3 prefers a visible split to an invisible merge.
    """
    return Decision(
        status="unresolved",
        canonical_name=group.display_name,
        method="split_unmatched",
        score=decision.score,
        runner_up=decision.runner_up,
        candidates_considered=decision.candidates_considered,
        reasoning=f"{group.split_reason}; no candidate fits these battles ({decision.reasoning})",
    )


def _part_of(group: MentionGroup, mentions: list[Mention], **changes: Any) -> MentionGroup:
    """Make a split part carry only its own mentions' names.

    Args:
        group: The group being split.
        mentions: The mentions this part keeps.
        **changes: Further fields to replace on the part.

    Returns:
        A copy of ``group`` whose ``surface_forms`` and ``keys`` come from
        ``mentions`` alone, so a general linked to one part is not given the
        other part's name forms as aliases. ``display_name`` is left alone:
        the LLM prompt is rendered from it and must stay byte-stable.
    """
    forms = {clean_surface(m.name) or m.name.strip() for m in mentions}
    keys: list[str] = []
    for mention in mentions:
        for key in name_keys(mention.name):
            if key not in keys:
                keys.append(key)
    return replace(
        group,
        mentions=mentions,
        surface_forms=sorted(forms),
        keys=tuple(keys) or group.keys,
        **changes,
    )


def _split_lifetimes(groups: list[MentionGroup]) -> list[MentionGroup]:
    """Separate a name's mentions that are further apart than one career.

    Args:
        groups: The grouped mentions.

    Returns:
        The groups with each over-long one replaced by its clusters. The
        largest cluster keeps the group's key and the others take
        ``<key>#era<n>``. A group that does not split is returned as the same
        object, so its prompt and request hash are unchanged.
    """
    result: list[MentionGroup] = []
    for group in groups:
        clusters = career_clusters(group.mentions)
        if len(clusters) == 1:
            result.append(group)
            continue
        largest = max(range(len(clusters)), key=lambda i: (len(clusters[i]), -i))
        for index, mentions in enumerate(clusters):
            years = _battle_years(mentions)
            reason = f"lifetime split of {group.display_name!r}: battles {years[0]}..{years[-1]}"
            if index == largest:
                part = _part_of(group, mentions, split_reason=reason)
            else:
                part = _part_of(
                    group,
                    mentions,
                    key=f"{group.key}#era{index + 1}",
                    split_from=group.key,
                    split_reason=reason,
                )
            logger.info(
                "resolve_group_split",
                group=group.display_name,
                part=part.key,
                mentions=len(mentions),
                years=years,
                reason="dated battles further apart than one career",
            )
            result.append(part)
    return result


class _GroupResolver:
    """Matches groups, and re-matches the mentions a lifespan check detaches.

    Holds the per-run state ``resolve_groups`` threads between its passes. The
    group list is extended in place, so the caller sees every subgroup.
    """

    def __init__(
        self,
        groups: list[MentionGroup],
        source: CandidateSource,
        *,
        threshold: float,
        margin: float,
    ) -> None:
        self.groups = groups
        self.source = source
        self.threshold = threshold
        self.margin = margin
        self.decisions: dict[str, Decision] = {}
        self.candidates: dict[str, list[Candidate]] = {}
        self._excluded: dict[str, frozenset[str]] = {}
        self._depth: dict[str, int] = {}

    def decide_from(self, start: int) -> None:
        """Run the deterministic match on every group from ``start`` on.

        Args:
            start: Index of the first group not yet matched. Groups that a
                guard appends while this runs are matched too.
        """
        index = start
        while index < len(self.groups):
            self._decide(index)
            index += 1

    def guard(self, index: int) -> None:
        """Detach the mentions a linked candidate could not have commanded in.

        Args:
            index: Position of a group whose final decision is in place.
        """
        group = self.groups[index]
        decision = self.decisions[group.key]
        candidate = decision.candidate
        if decision.status != "linked" or candidate is None:
            return
        kept, detached = partition_by_lifespan(group.mentions, candidate)
        if not detached:
            return

        depth = self._depth.get(group.key, 0)
        years = _battle_years(detached)
        reason = (
            f"lifespan guard: {candidate.label} ({candidate.qid}, "
            f"{candidate.birth_year}..{candidate.death_year}) cannot have commanded "
            f"in {years}"
        )
        logger.info(
            "resolve_mentions_detached",
            group=group.display_name,
            qid=candidate.qid,
            years=years,
            mentions=len(detached),
            reason="battle outside the candidate's command window",
        )
        sub_key = f"{group.key}#not-{candidate.qid}"
        subgroup = _part_of(
            group,
            detached,
            key=sub_key,
            confidence=0.0,
            method="new_entity",
            split_from=group.split_from or group.key,
            split_reason=reason,
        )
        self.groups[index] = _part_of(group, kept)
        if depth >= _MAX_SPLIT_DEPTH:
            # Still detached, so the namesake's mentions do not stay linked to
            # the candidate; no further search is made for them.
            logger.warning("resolve_split_depth_exhausted", group=group.key, qid=candidate.qid)
            exhausted = Decision(
                status="new",
                canonical_name=subgroup.display_name,
                method="new_entity",
                reasoning="split depth exhausted",
            )
            self.decisions[sub_key] = _split_decision(subgroup, exhausted)
            self.groups.append(subgroup)
            return
        self._excluded[sub_key] = self._excluded.get(group.key, frozenset()) | {candidate.qid}
        self._depth[sub_key] = depth + 1
        self.groups.append(subgroup)

    def final_decision(self, group: MentionGroup, decision: Decision) -> Decision:
        """Pass a final decision through the split-off rule.

        Args:
            group: The group the decision is for.
            decision: Its final decision, from the matcher or the LLM.

        Returns:
            ``split_unmatched`` for a split-off group that came out ``new``
            by any route; every other decision unchanged. A corpus-local
            general for a split-off part would merge back by name.
        """
        if group.split_from and decision.status == "new":
            return _split_decision(group, decision)
        return decision

    def _decide(self, index: int) -> None:
        group = self.groups[index]
        if group.key in self.decisions:
            return
        if not group.split_from and is_placeholder(group.display_name):
            self.decisions[group.key] = _placeholder_decision(group)
            return
        reason = None if group.split_from else non_person_reason(group.display_name)
        if reason is not None:
            self.decisions[group.key] = _non_person_decision(group, reason)
            return

        excluded = self._excluded.get(group.key, frozenset())
        found = [c for c in self.source.search(group) if c.qid not in excluded]
        self.candidates[group.key] = found
        decision = match_group(group, found, threshold=self.threshold, margin=self.margin)
        if decision.status == "new" and not group.split_from:
            decision = _refuse_non_person(group, found, decision, self.source)
        decision = self.final_decision(group, decision)
        self.decisions[group.key] = decision
        self.guard(index)


def resolve_groups(
    groups: list[MentionGroup],
    source: CandidateSource,
    service: LazyService | None,
    *,
    threshold: float,
    margin: float,
    counts: ResolveCounts,
    system: str = "",
    template: str = "",
    schema: dict[str, Any] | None = None,
    overrides: ResolveOverrides | None = None,
) -> dict[str, Decision]:
    """Decide what every group refers to.

    Two guards run around the matcher. Before it, a name whose dated battles
    are further apart than one career is split into clusters. After it, every
    mention of a linked group whose battle falls outside the candidate's
    command window is detached and matched again on its own years. Both edit
    ``groups`` in place; the caller must use the list afterwards.

    Args:
        groups: The grouped mentions. Replaced in place by the split groups.
        source: Where Wikidata candidates come from.
        service: The stage's deferred LLM service, or None to skip the LLM
            step entirely. It is only built if a group needs it.
        threshold: The spec's ``fuzzy_threshold``.
        margin: How far a match must beat its runner-up.
        counts: Counters to accumulate into.
        system: ``prompt.system`` from the spec.
        template: ``prompt.user_template`` from the spec.
        schema: ``prompt.output_schema`` from the spec.
        overrides: The run's manual overrides. ``commander_link`` mentions are
            taken out of their groups first and resolved as fixed decisions
            that skip the guards, the matcher and the LLM; ``general_merge``
            relinks the final decisions at the end.

    Returns:
        A final decision per group, keyed by group key.
    """
    manual_groups: list[MentionGroup] = []
    fixed: dict[str, Decision] = {}
    if overrides is not None:
        manual_groups, fixed = overrides.take_links(groups, source)
    groups[:] = _split_lifetimes(groups)
    groups.extend(manual_groups)
    resolver = _GroupResolver(groups, source, threshold=threshold, margin=margin)
    # _decide leaves a group that already has a decision alone, which is what
    # keeps a manual link away from the lifespan guard and the matcher.
    resolver.decisions.update(fixed)
    resolver.decide_from(0)
    decisions = resolver.decisions

    # Step 2, second half: a bare surname Wikidata did not answer for often
    # belongs to somebody this corpus has already pinned to an entity. Only
    # Wikidata-anchored identities are offered as targets, so corpus matches
    # cannot chain off each other. A manual link is not an anchor: a correction
    # for one mention must not pull other mentions in by name.
    anchors = _build_identities(
        groups,
        {key: d for key, d in decisions.items() if d.status == "linked" and d.method != "manual"},
    )
    if anchors:
        position = 0
        while position < len(groups):
            group = groups[position]
            current = decisions.get(group.key)
            if current is not None and current.status == "new":
                attached = match_against_identities(
                    group, anchors, threshold=threshold, margin=margin
                )
                if attached is not None:
                    decisions[group.key] = attached
                    # The same per-mention guard as for a candidate link: the
                    # identity's Wikidata lifespan binds the corpus match too.
                    before = len(groups)
                    resolver.guard(position)
                    resolver.decide_from(before)
            position += 1

    index = 0
    while index < len(groups):
        group = groups[index]
        decision = decisions[group.key]
        if decision.status == "ambiguous":
            client = service.get() if service is not None else None
            if client is None:
                decisions[group.key] = _stranded_decision(group, decision)
            else:
                counts.llm_calls += 1
                before = len(groups)
                decisions[group.key] = resolver.final_decision(
                    group,
                    disambiguate(
                        group,
                        # Date-gated and ranked: the prompt shows only the first few, and
                        # source order cut Caesar and Hannibal from their own lists.
                        rank_for_prompt(group, resolver.candidates.get(group.key, [])),
                        client,
                        system=system,
                        template=template,
                        schema=schema or {},
                    ),
                )
                resolver.guard(index)
                resolver.decide_from(before)
        index += 1

    # A corpus link can detach every mention of its group, leaving a husk.
    for group in [g for g in groups if not g.mentions]:
        decisions.pop(group.key, None)
    groups[:] = [g for g in groups if g.mentions]

    if overrides is not None:
        overrides.apply_merges(groups, decisions, source)

    for group in groups:
        decision = decisions[group.key]
        group.confidence = decision.confidence
        group.method = decision.method
        if decision.status == "linked":
            counts.linked += 1
        elif decision.status == "new":
            counts.new_entities += 1
        else:
            counts.unresolved += 1

    return decisions


def _log_row(group: MentionGroup, decision: Decision, general_id: int | None) -> dict[str, Any]:
    """Render one resolution-log row.

    Args:
        group: The mention group.
        decision: Its final decision.
        general_id: The row it was written to, when it was written.

    Returns:
        A JSON-serialisable mapping.
    """
    split = (
        {"split_from": group.split_from, "split_reason": group.split_reason}
        if group.split_from or group.split_reason
        else {}
    )
    manual = {"override_id": decision.override_id} if decision.override_id else {}
    return {
        "group_key": group.key,
        "display_name": group.display_name,
        "surface_forms": group.surface_forms,
        "mentions": len(group.mentions),
        "battles": sorted({m.battle_name for m in group.mentions})[:50],
        "years": sorted(set(group.years)),
        "status": decision.status,
        "method": decision.method,
        "wikidata_id": decision.qid,
        "canonical_name": decision.canonical_name,
        "confidence": decision.confidence,
        "score": decision.score,
        "runner_up": decision.runner_up,
        "candidates_considered": decision.candidates_considered,
        "reasoning": decision.reasoning,
        "general_id": general_id,
        **split,
        **manual,
    }


def _write_resolution_log(
    path: Path,
    groups: list[MentionGroup],
    decisions: dict[str, Decision],
    identities: list[Identity],
) -> None:
    """Write the audit trail of every merge and link decision.

    Args:
        path: ``data/processed/resolution_log.jsonl``.
        groups: Every group the run resolved.
        decisions: Their final decisions.
        identities: The identities built from them, carrying the row ids.
    """
    general_ids: dict[str, int | None] = {}
    for identity in identities:
        for group in identity.groups:
            general_ids[group.key] = identity.general_id

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for group in groups:
            row = _log_row(group, decisions[group.key], general_ids.get(group.key))
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")

    logger.info("resolution_log_written", path=str(path), rows=len(groups))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to a JSON lines file, creating the directory if needed.

    An empty list still writes an empty file: "the scan found nothing" and
    "the scan never ran" must not look the same to whoever reads it next.

    Args:
        path: The destination file.
        rows: The rows to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    logger.info("processed_file_written", path=str(path), rows=len(rows))


def _is_lookup_candidate(group: MentionGroup) -> bool:
    """Whether a group is worth asking Wikidata about.

    Args:
        group: The mention group.

    Returns:
        False for a placeholder and for a name the deterministic filter has
        already refused. They never reach the matcher, so spending a query on
        them only slows the run and floods the VALUES block.
    """
    return not is_placeholder(group.display_name) and non_person_reason(group.display_name) is None


def _is_unsettled(
    group: MentionGroup, source: PrefetchedCandidateSource, threshold: float, margin: float
) -> bool:
    """Whether the matcher would still make this group a new entity.

    Args:
        group: The mention group.
        source: The candidates fetched so far.
        threshold: The spec's ``fuzzy_threshold``.
        margin: The spec's ``ambiguity_margin``.

    Returns:
        True when the lookups so far leave the group with no link and no
        ambiguity for the model to settle: the groups a wider search might
        still place.
    """
    decision = match_group(group, source.search(group), threshold=threshold, margin=margin)
    return decision.status == "new"


async def _gather_sparql_candidates(
    groups: list[MentionGroup],
    *,
    user_agent: str,
    rate_limit: float,
    batch_size: int,
    respect_robots: bool,
    max_names_per_group: int,
    search_names_per_group: int,
    search_batch_size: int,
    threshold: float,
    margin: float,
    qids: Sequence[str] = (),
) -> PrefetchedCandidateSource:
    """Open one client and ask the query service about every group.

    Three passes, each only for what the one before left unplaced:

    1. **Exact**: humans whose label or alias is the surface form.
    2. **Search**: a ranked name search, for a commander whose Wikidata label
       is longer than the name an infobox writes ("Grouchy" against "Emmanuel
       de Grouchy, Marquis de Grouchy"). The hits are candidates only; the
       matcher's threshold, date gate and margin still decide.
    3. **Non-person**: items that are not human but carry the name, for
       groups still unplaced after both ("Stavka", "Jena").

    Args:
        groups: The groups to look up.
        user_agent: Sent on every request.
        rate_limit: Seconds between requests to the query service.
        batch_size: Names per exact query.
        respect_robots: Whether to consult robots.txt.
        max_names_per_group: Surface forms asked about per group.
        search_names_per_group: Surface forms asked about in the search pass.
        search_batch_size: Names per search query; searches cost more each.
        threshold: The spec's ``fuzzy_threshold``.
        margin: The spec's ``ambiguity_margin``.
        qids: Entities manual overrides name outright, fetched by item id.

    Returns:
        A source holding the candidates and non-person verdicts found.
    """
    lookup = [group for group in groups if _is_lookup_candidate(group)]

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
        names = collect_query_names(lookup, max_names_per_group)
        by_name = await fetch_candidates(fetcher, names, batch_size=batch_size)
        source = PrefetchedCandidateSource(by_name, max_names_per_group=max_names_per_group)

        unsettled = [g for g in lookup if _is_unsettled(g, source, threshold, margin)]
        search_names = collect_query_names(unsettled, search_names_per_group)
        if search_names:
            logger.info("resolve_search_fallback", groups=len(unsettled), names=len(search_names))
            source.extend(
                await fetch_candidates(
                    fetcher, search_names, batch_size=search_batch_size, search=True
                )
            )

        still_new = [g for g in unsettled if _is_unsettled(g, source, threshold, margin)]
        kind_names = collect_query_names(still_new, max_names_per_group)
        if kind_names:
            source.set_non_person(
                await fetch_non_person_kinds(fetcher, kind_names, batch_size=batch_size)
            )
        if qids:
            source.add_by_qid(await fetch_candidates_by_qid(fetcher, qids, batch_size=batch_size))
        return source


def _build_candidate_source(
    kind: str,
    groups: list[MentionGroup],
    params: dict[str, Any],
    raw_root: Path,
    ctx: StageContext,
    *,
    threshold: float,
    margin: float,
    qids: Sequence[str] = (),
) -> CandidateSource:
    """Construct the candidate source the spec asks for.

    Args:
        kind: ``params.candidate_source``.
        groups: The groups to resolve, for the SPARQL prefetch.
        params: The spec's params.
        raw_root: The ``data/raw`` directory.
        ctx: The stage context.
        threshold: The spec's ``fuzzy_threshold``, for deciding which groups
            the search fallback should try.
        margin: The spec's ``ambiguity_margin``.
        qids: Q-ids manual overrides name, which the SPARQL source fetches
            directly; the local source indexes them from disk.

    Returns:
        A source. A dry run always gets the null source, because the point of
        a dry run is to cost the work without doing it.

    Raises:
        ValueError: If the spec names a source that does not exist. A typo
            here would otherwise resolve the whole corpus as new entities and
            look like a successful run.
    """
    if ctx.dry_run:
        logger.info("resolve_dry_run_skipping_candidate_lookup")
        return NullCandidateSource()

    if kind == "local":
        return LocalCandidateSource(raw_root / _WIKIDATA_DIR)

    if kind == "none":
        return NullCandidateSource()

    if kind != "sparql":
        raise ValueError(
            f"agents/resolve.yaml params.candidate_source is {kind!r}; "
            "expected 'sparql', 'local' or 'none'"
        )

    return asyncio.run(
        _gather_sparql_candidates(
            groups,
            user_agent=str(params.get("user_agent", _DEFAULT_USER_AGENT)),
            rate_limit=float(params.get("rate_limit_sparql", _DEFAULT_SPARQL_RATE_LIMIT)),
            batch_size=int(params.get("sparql_batch_size", DEFAULT_BATCH_SIZE)),
            respect_robots=bool(params.get("respect_robots", True)),
            max_names_per_group=int(params.get("max_names_per_group", DEFAULT_MAX_NAMES_PER_GROUP)),
            search_names_per_group=int(
                params.get("search_names_per_group", _DEFAULT_SEARCH_NAMES_PER_GROUP)
            ),
            search_batch_size=int(params.get("search_batch_size", _DEFAULT_SEARCH_BATCH_SIZE)),
            threshold=threshold,
            margin=margin,
            qids=qids,
        )
    )


def _unresolved_mentions(
    groups: list[MentionGroup],
    decisions: dict[str, Decision],
    *,
    non_person: bool = False,
) -> list[Mention]:
    """Collect the mentions of every group that got no general.

    Args:
        groups: Every group the run resolved.
        decisions: Their final decisions.
        non_person: Collect the groups refused as non-persons instead of the
            groups that are merely unresolved. They are logged separately and
            kept out of the ``resolution_rate`` denominator.

    Returns:
        The mentions to record in ``missing_data_log``.
    """
    return [
        mention
        for group in groups
        if decisions[group.key].status == "unresolved"
        and (decisions[group.key].method == "non_person") == non_person
        for mention in group.mentions
    ]


def _derive_era(identity: Identity, bins: tuple[tuple[str, int | None], ...]) -> str | None:
    """Bucket a person into an era with the impute stage's bins.

    Args:
        identity: The resolved person.
        bins: ``config/imputation_priors.yaml``'s ``(name, exclusive_upper)``
            era bins, so a general's era and a battle's era mean the same
            thing.

    Returns:
        The era of the median year of the battles they appear in, since that
        is when this corpus saw them command. Failing that, the year they
        turned 35. None when neither is known: an unplaced person is left
        unplaced rather than filed under the 'undated' bin.
    """
    years = identity.years
    if years:
        return era_of_year(int(statistics.median(years)), bins)
    if identity.birth_year is not None:
        return era_of_year(identity.birth_year + _ADULT_AGE_YEARS, bins)
    return None


def _load_era_bins(params: dict[str, Any]) -> tuple[tuple[str, int | None], ...]:
    """Read the shared era bins, tolerating their absence.

    Args:
        params: The spec's params.

    Returns:
        The bins, or an empty tuple when the priors file cannot be read.
        Every general's era is then left NULL: resolve does not need the
        impute config to run, and inventing boundaries here would put two
        definitions of 'napoleonic' in one pipeline.
    """
    path = Path(str(params.get("imputation_priors", DEFAULT_PRIORS_PATH)))
    try:
        return load_priors(path).era_bins
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("resolve_era_bins_unavailable", path=str(path), error=str(exc))
        return ()


# An unresolved group with one of these methods is pending, not refused: the
# answer may still link it, so its side keeps every existing row.
_PENDING_METHODS: Final = frozenset({"llm_deferred", "llm_failed"})


def _prune_stale_commanders(
    ctx: StageContext,
    groups: list[MentionGroup],
    decisions: dict[str, Decision],
    battles: dict[str, int],
    sides: dict[tuple[int, str], int],
    linked: dict[tuple[int, int, int], str],
    failed_sides: set[tuple[int, int]],
    counts: ResolveCounts,
) -> None:
    """Remove the commander links a re-run no longer makes.

    Args:
        ctx: The stage context, carrying the connection.
        groups: Every group the run resolved.
        decisions: Their final decisions.
        battles: Battle slug and name to id.
        sides: Side id by (battle id, label).
        linked: Every row written this run, keyed by battle, side and general.
        failed_sides: Sides of identities whose write failed.
        counts: Counters to accumulate into.
    """
    if ctx.limit is not None:
        # A truncated group list does not cover every mention of a side, so a
        # row it did not relink may belong to a group that was cut.
        logger.info("resolve_stale_commander_prune_skipped", limit=ctx.limit)
        return

    processed: set[tuple[int, int]] = set()
    protected: set[tuple[int, int]] = set(failed_sides)
    for group in groups:
        pending = decisions[group.key].method in _PENDING_METHODS
        for mention in group.mentions:
            placed = side_of(mention, battles, sides)
            if placed is None:
                continue
            processed.add(placed)
            if pending:
                protected.add(placed)

    counts.commanders_removed, counts.sides_reset = prune_stale_commanders(
        ctx.db_conn, processed, protected, linked
    )


def _write_database(
    ctx: StageContext,
    groups: list[MentionGroup],
    decisions: dict[str, Decision],
    identities: list[Identity],
    counts: ResolveCounts,
    params: dict[str, Any],
) -> tuple[list[DuplicateBattle], list[Mention]]:
    """Write generals, aliases, commanders and the missing-data rows.

    Args:
        ctx: The stage context, carrying the connection.
        groups: Every group the run resolved.
        decisions: Their final decisions.
        identities: The identities to write.
        counts: Counters to accumulate into.
        params: The spec's params, for the duplicate-battle thresholds.

    Returns:
        Suspected duplicate battles, for the report file, which are reported
        and never merged (see :mod:`pipeline.resolvers.battles`); and the
        mentions that could not be logged in ``missing_data_log`` because
        their battle is not in the database at all.
    """
    conn = ctx.db_conn
    battles = load_battle_index(conn)
    sides = load_side_index(conn)

    if not battles:
        logger.warning(
            "resolve_no_battles_in_database",
            hint="run the extract stage with a database connection first",
        )

    era_bins = _load_era_bins(params)
    unplaced: list[Mention] = []
    source_cache: dict[tuple[str, str], int] = {}
    linked: dict[tuple[int, int, int], str] = {}
    failed_sides: set[tuple[int, int]] = set()
    for identity in identities:
        before_counts = counts.as_dict()
        before_cache = dict(source_cache)
        before_unplaced = len(unplaced)
        if era_bins:
            identity.era = _derive_era(identity, era_bins)
        try:
            # A savepoint per identity: on PostgreSQL one failed statement
            # aborts the whole transaction, and every later write would fail.
            with conn.begin_nested():
                write_identity(
                    conn, identity, battles, sides, counts, source_cache, unplaced, linked
                )
        except Exception as exc:
            # One person who will not go in is a data problem, not a reason
            # to lose the rest of the run's writes.
            for name, value in before_counts.items():
                setattr(counts, name, value)
            source_cache.clear()
            source_cache.update(before_cache)
            # Whatever the failed write collected is stale, and every mention
            # of the person is now one that never reached the table.
            del unplaced[before_unplaced:]
            unplaced.extend(identity.mentions)
            failed_sides.update(
                placed
                for placed in (side_of(m, battles, sides) for m in identity.mentions)
                if placed is not None
            )
            counts.failed_identities += 1
            logger.error(
                "resolve_write_failed",
                identity=identity.canonical_name,
                wikidata_id=identity.qid,
                error=str(exc),
                error_type=type(exc).__name__,
            )

    homeless = log_unresolved(
        conn,
        _unresolved_mentions(groups, decisions),
        battles,
        sides,
        counts,
        unplaced=unplaced,
        non_person=_unresolved_mentions(groups, decisions, non_person=True),
    )

    _prune_stale_commanders(ctx, groups, decisions, battles, sides, linked, failed_sides, counts)

    # Last write, so a general this run stranded is swept too, and before the
    # quality gates, which count an orphan beside its live namesake.
    counts.orphan_generals_removed = len(remove_orphan_generals(conn))

    duplicates = find_duplicate_battles(
        conn,
        threshold=float(params.get("battle_duplicate_threshold", DEFAULT_NAME_THRESHOLD)),
        year_slack=int(params.get("battle_duplicate_year_slack", DEFAULT_YEAR_SLACK)),
    )
    counts.duplicate_battles = len(duplicates)
    return duplicates, homeless


def _unplaced_row(mention: Mention) -> dict[str, Any]:
    """Render a mention that has no battle in the database.

    Args:
        mention: A commander mention whose battle could not be found.

    Returns:
        A JSON-serialisable mapping.
    """
    return {
        "battle_slug": mention.battle_slug,
        "battle_name": mention.battle_name,
        "side_label": mention.side_label,
        "name": mention.name,
    }


def _report_unplaced(homeless: list[Mention], counts: ResolveCounts) -> None:
    """Say loudly how many mentions could not reach the database at all.

    ``missing_data_log.battle_id`` is NOT NULL, so a mention whose battle is
    absent cannot be recorded there and the SQL ``resolution_rate`` gate cannot
    see it. They go to ``unplaced_mentions.jsonl`` instead, and this warning is
    what stops that being a quiet second place for them.

    Args:
        homeless: The mentions with no battle row.
        counts: The run's counters, for the denominator.
    """
    if not homeless:
        return
    logger.warning(
        "resolve_mentions_without_a_battle",
        mentions=len(homeless),
        of=counts.mentions,
        fraction=round(len(homeless) / max(1, counts.mentions), 4),
        hint="the extract stage did not write these battles; the resolution_rate gate "
        "cannot count them (missing_data_log.battle_id is NOT NULL)",
    )


def run(spec: dict[str, Any], context: StageContext | None = None) -> StageOutcome | None:
    """Run the resolve stage.

    Args:
        spec: The loaded ``agents/resolve.yaml``, with overrides applied.
        context: Shared services. Without a database connection the stage
            still resolves and still writes the resolution log, and skips the
            row writes.

    Returns:
        A :class:`StageOutcome` naming the batch file when disambiguation
        requests were deferred to offline processing, otherwise None.

    Raises:
        ValueError: If the spec is incomplete or names an unknown candidate
            source. Both would otherwise produce a run that looks successful
            and resolves nothing.
    """
    ctx = context or StageContext()
    params = spec.get("params") or {}

    processed_root = Path(str(params.get("processed_root", PROCESSED_ROOT)))
    raw_root = Path(str(params.get("raw_root", RAW_ROOT)))
    threshold = float(params.get("fuzzy_threshold", 85))
    margin = float(params.get("ambiguity_margin", AMBIGUITY_MARGIN))
    source_kind = str(params.get("candidate_source", "sparql")).strip().lower()

    # Read before any work is done: an incomplete spec should fail before the
    # stage spends a single query.
    system, template, schema = prompt_parts(spec)

    path, explicit = overrides_path(params)
    overrides = load_overrides(path, required=explicit)
    clear_report("resolve", processed_root)
    report = OverrideReport("resolve", overrides, limited=ctx.limit is not None)
    resolve_overrides = ResolveOverrides(overrides.for_stage("resolve"), report)

    mentions = load_mentions(
        processed_root / _COMMANDERS_FILE,
        processed_root / _BATTLES_FILE,
    )
    if not mentions:
        logger.warning(
            "resolve_found_no_mentions",
            path=str(processed_root / _COMMANDERS_FILE),
            hint="run the extract stage first",
        )
        return None

    groups = group_mentions(mentions)
    if ctx.limit is not None:
        groups = groups[: ctx.limit]

    counts = ResolveCounts()
    counts.mentions = sum(len(g.mentions) for g in groups)
    counts.groups = len(groups)

    source = _build_candidate_source(
        source_kind,
        groups,
        params,
        raw_root,
        ctx,
        threshold=threshold,
        margin=margin,
        qids=resolve_overrides.required_qids(groups),
    )
    source.prefetch(groups)

    service = LazyService(
        None if ctx.dry_run else lambda: LLMService.from_spec(spec, db_conn=ctx.db_conn)
    )

    decisions = resolve_groups(
        groups,
        source,
        service,
        threshold=threshold,
        margin=margin,
        counts=counts,
        system=system,
        template=template,
        schema=schema,
        overrides=resolve_overrides,
    )

    counts.groups = len(groups)
    identities = _build_identities(groups, decisions)
    counts.identities = len(identities)
    counts.non_person_mentions = len(_unresolved_mentions(groups, decisions, non_person=True))

    duplicates: list[DuplicateBattle] = []
    homeless: list[Mention] = []
    if ctx.db_conn is not None and not ctx.dry_run:
        duplicates, homeless = _write_database(ctx, groups, decisions, identities, counts, params)
        _report_unplaced(homeless, counts)
    else:
        logger.info("resolve_database_write_skipped", dry_run=ctx.dry_run)

    report.write(processed_root)
    _write_resolution_log(processed_root / _RESOLUTION_LOG, groups, decisions, identities)
    _write_jsonl(processed_root / _DUPLICATE_BATTLES, [pair.as_dict() for pair in duplicates])
    _write_jsonl(processed_root / _UNPLACED_MENTIONS, [_unplaced_row(m) for m in homeless])

    outcome: StageOutcome | None = None
    if service.built is not None:
        batch_file = service.built.flush_pending()
        service.built.log_summary()
        if service.built.usage.deferred:
            outcome = StageOutcome(
                deferred_requests=service.built.usage.deferred, batch_file=batch_file
            )

    logger.info("resolve_complete", **counts.as_dict())
    return outcome
