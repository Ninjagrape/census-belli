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
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import httpx
import structlog

from pipeline.crawlers.fetcher import Fetcher, RateLimiter
from pipeline.llm import LLMService
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
    collect_query_names,
    disambiguate,
    fetch_candidates,
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
    prompt_parts,
    write_identity,
)
from pipeline.resolvers.battles import DEFAULT_NAME_THRESHOLD, DEFAULT_YEAR_SLACK
from pipeline.resolvers.candidates import DEFAULT_BATCH_SIZE, DEFAULT_MAX_NAMES_PER_GROUP
from pipeline.stages.base import StageContext

__all__ = ["LazyService", "resolve_groups", "run"]

logger = structlog.get_logger()

PROCESSED_ROOT: Final[Path] = Path("data/processed")
RAW_ROOT: Final[Path] = Path("data/raw")

_COMMANDERS_FILE: Final[str] = "commanders_raw.jsonl"
_BATTLES_FILE: Final[str] = "battles.jsonl"
_RESOLUTION_LOG: Final[str] = "resolution_log.jsonl"
_DUPLICATE_BATTLES: Final[str] = "duplicate_battles.jsonl"
_WIKIDATA_DIR: Final[str] = "wikidata"

# The query service is a shared, unpaid resource and it is stricter than the
# article API. One request every two seconds is slower than handover §5.2's
# floor for Wikipedia, deliberately.
_DEFAULT_SPARQL_RATE_LIMIT: Final[float] = 2.0
_DEFAULT_TIMEOUT_S: Final[float] = 60.0
_DEFAULT_USER_AGENT: Final[str] = (
    "general-war-research/0.1 (https://github.com/census-belli; entity resolution)"
)

_PLACEHOLDER_CONFIDENCE: Final[float] = 0.0


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
            )
            identities[key] = identity
        elif decision.confidence > identity.confidence:
            # The identity is described by its best-evidenced group; each
            # row's own confidence still comes from its group's decision.
            identity.confidence = decision.confidence
            identity.method = decision.method

        identity.groups.append(group)
        for alias in group.surface_forms:
            if alias and alias not in identity.aliases:
                identity.aliases.append(alias)

    for identity in identities.values():
        identity.aliases = [a for a in identity.aliases if a != identity.canonical_name]

    return list(identities.values())


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
) -> dict[str, Decision]:
    """Decide what every group refers to.

    Args:
        groups: The grouped mentions.
        source: Where Wikidata candidates come from.
        service: The stage's deferred LLM service, or None to skip the LLM
            step entirely. It is only built if a group needs it.
        threshold: The spec's ``fuzzy_threshold``.
        margin: How far a match must beat its runner-up.
        counts: Counters to accumulate into.
        system: ``prompt.system`` from the spec.
        template: ``prompt.user_template`` from the spec.
        schema: ``prompt.output_schema`` from the spec.

    Returns:
        A final decision per group, keyed by group key.
    """
    decisions: dict[str, Decision] = {}
    candidates_by_group: dict[str, list[Candidate]] = {}

    for group in groups:
        if is_placeholder(group.display_name):
            decisions[group.key] = _placeholder_decision(group)
            continue
        found = source.search(group)
        candidates_by_group[group.key] = found
        decisions[group.key] = match_group(group, found, threshold=threshold, margin=margin)

    # Step 2, second half: a bare surname Wikidata did not answer for often
    # belongs to somebody this corpus has already pinned to an entity. Only
    # Wikidata-anchored identities are offered as targets, so corpus matches
    # cannot chain off each other.
    anchors = _build_identities(
        groups,
        {key: d for key, d in decisions.items() if d.status == "linked"},
    )
    if anchors:
        for group in groups:
            if decisions[group.key].status != "new":
                continue
            attached = match_against_identities(group, anchors, threshold=threshold, margin=margin)
            if attached is not None:
                decisions[group.key] = attached

    for group in groups:
        decision = decisions[group.key]
        if decision.status != "ambiguous":
            continue
        client = service.get() if service is not None else None
        if client is None:
            decisions[group.key] = _stranded_decision(group, decision)
            continue
        counts.llm_calls += 1
        decisions[group.key] = disambiguate(
            group,
            candidates_by_group.get(group.key, []),
            client,
            system=system,
            template=template,
            schema=schema or {},
        )

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


async def _gather_sparql_candidates(
    names: list[str],
    *,
    user_agent: str,
    rate_limit: float,
    batch_size: int,
    respect_robots: bool,
) -> dict[str, list[Candidate]]:
    """Open one client and ask the query service about every name.

    Args:
        names: Distinct surface forms to look up.
        user_agent: Sent on every request.
        rate_limit: Seconds between requests to the query service.
        batch_size: Names per query.
        respect_robots: Whether to consult robots.txt.

    Returns:
        Surface form to candidates.
    """
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
        return await fetch_candidates(fetcher, names, batch_size=batch_size)


def _build_candidate_source(
    kind: str,
    groups: list[MentionGroup],
    params: dict[str, Any],
    raw_root: Path,
    ctx: StageContext,
) -> CandidateSource:
    """Construct the candidate source the spec asks for.

    Args:
        kind: ``params.candidate_source``.
        groups: The groups to resolve, for the SPARQL prefetch.
        params: The spec's params.
        raw_root: The ``data/raw`` directory.
        ctx: The stage context.

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

    limit = int(params.get("max_names_per_group", DEFAULT_MAX_NAMES_PER_GROUP))
    names = collect_query_names(groups, limit)
    by_name = asyncio.run(
        _gather_sparql_candidates(
            names,
            user_agent=str(params.get("user_agent", _DEFAULT_USER_AGENT)),
            rate_limit=float(params.get("rate_limit_sparql", _DEFAULT_SPARQL_RATE_LIMIT)),
            batch_size=int(params.get("sparql_batch_size", DEFAULT_BATCH_SIZE)),
            respect_robots=bool(params.get("respect_robots", True)),
        )
    )
    return PrefetchedCandidateSource(by_name, max_names_per_group=limit)


def _unresolved_mentions(
    groups: list[MentionGroup],
    decisions: dict[str, Decision],
) -> list[Mention]:
    """Collect the mentions of every group that got no general.

    Args:
        groups: Every group the run resolved.
        decisions: Their final decisions.

    Returns:
        The mentions to record in ``missing_data_log``.
    """
    return [
        mention
        for group in groups
        if decisions[group.key].status == "unresolved"
        for mention in group.mentions
    ]


def _write_database(
    ctx: StageContext,
    groups: list[MentionGroup],
    decisions: dict[str, Decision],
    identities: list[Identity],
    counts: ResolveCounts,
    params: dict[str, Any],
) -> list[DuplicateBattle]:
    """Write generals, aliases, commanders and the missing-data rows.

    Args:
        ctx: The stage context, carrying the connection.
        groups: Every group the run resolved.
        decisions: Their final decisions.
        identities: The identities to write.
        counts: Counters to accumulate into.
        params: The spec's params, for the duplicate-battle thresholds.

    Returns:
        Suspected duplicate battles, for the report file. They are reported
        and never merged; see :mod:`pipeline.resolvers.battles`.
    """
    conn = ctx.db_conn
    battles = load_battle_index(conn)
    sides = load_side_index(conn)

    if not battles:
        logger.warning(
            "resolve_no_battles_in_database",
            hint="run the extract stage with a database connection first",
        )

    source_cache: dict[tuple[str, str], int] = {}
    for identity in identities:
        try:
            write_identity(conn, identity, battles, sides, counts, source_cache)
        except Exception as exc:
            # One person who will not go in is a data problem, not a reason
            # to lose the rest of the run's writes.
            logger.error(
                "resolve_write_failed",
                identity=identity.canonical_name,
                wikidata_id=identity.qid,
                error=str(exc),
                error_type=type(exc).__name__,
            )

    log_unresolved(conn, _unresolved_mentions(groups, decisions), battles, sides, counts)

    duplicates = find_duplicate_battles(
        conn,
        threshold=float(params.get("battle_duplicate_threshold", DEFAULT_NAME_THRESHOLD)),
        year_slack=int(params.get("battle_duplicate_year_slack", DEFAULT_YEAR_SLACK)),
    )
    counts.duplicate_battles = len(duplicates)
    return duplicates


def run(spec: dict[str, Any], context: StageContext | None = None) -> None:
    """Run the resolve stage.

    Args:
        spec: The loaded ``agents/resolve.yaml``, with overrides applied.
        context: Shared services. Without a database connection the stage
            still resolves and still writes the resolution log, and skips the
            row writes.

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
        return

    groups = group_mentions(mentions)
    if ctx.limit is not None:
        groups = groups[: ctx.limit]

    counts = ResolveCounts()
    counts.mentions = sum(len(g.mentions) for g in groups)
    counts.groups = len(groups)

    source = _build_candidate_source(source_kind, groups, params, raw_root, ctx)
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
    )

    identities = _build_identities(groups, decisions)
    counts.identities = len(identities)

    duplicates: list[DuplicateBattle] = []
    if ctx.db_conn is not None and not ctx.dry_run:
        duplicates = _write_database(ctx, groups, decisions, identities, counts, params)
    else:
        logger.info("resolve_database_write_skipped", dry_run=ctx.dry_run)

    _write_resolution_log(processed_root / _RESOLUTION_LOG, groups, decisions, identities)
    _write_jsonl(processed_root / _DUPLICATE_BATTLES, [pair.as_dict() for pair in duplicates])

    if service.built is not None:
        service.built.log_summary()

    logger.info("resolve_complete", **counts.as_dict())
