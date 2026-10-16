"""
The classify stage: command roles, battle types, and missingness mechanisms.

Two unrelated jobs share this stage because ``agents/classify.yaml`` says so
(see its module docstring): turning each side's already-resolved commanders
into a command hierarchy, and labelling why a missing field is missing. Both
read what earlier stages wrote and touch nothing upstream of them.

**Task A, command role.** :func:`pipeline.classifiers.roles.classify_side_roles`
settles most sides deterministically -- one commander, or roles already
consistent. What is left (the spec's ``default_split`` fallback) is exactly
the set of sides worth an LLM call, and only those ever reach one. Two modes,
set by ``params.llm_mode``:

- ``offline`` (the default): look up each pending side's request in
  ``llm_calls`` by its exact hash. A hit is applied; a miss is left at the
  deterministic fallback, counted, and written to a batch file under
  ``data/llm_requests/`` for ``/process-llm-batch``. No network, no API
  client is built, and the orchestrator stops after the stage.
- ``api``: build an :class:`pipeline.llm.LLMService` lazily -- most runs
  settle everything deterministically or from the cache and never need
  one -- and send the request live. A failed call leaves the deterministic
  decision in place, per the project's extraction-failure rule.

**Task B, battle type and missingness.** :func:`pipeline.classifiers
.battle_type.infer_battle_type` runs for every battle still 'unknown', reading
troop-report branches, strength text, and the raw article when one exists on
disk. Run *before* the missingness pass, deliberately: a battle_type just
inferred moves its ``missing_data_log`` row to 'observed', which is what keeps
the missingness pass -- which only ever touches an 'unclassified' row -- from
reconsidering it. :func:`pipeline.classifiers.missingness.classify_missingness`
then runs over what remains.

Every write is a DB write; there is no partial "planned but not connected"
mode. Without a connection this fails loudly rather than silently doing
nothing, per the brief -- there is no data-quality reason to degrade here, and
degrading would look like a successful, empty run.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, Final

import structlog

from pipeline.classifiers import (
    HTML_SUFFIXES,
    BattleTypeDecision,
    ClassifyCounts,
    MissingnessContext,
    MissingnessDecision,
    RoleDecision,
    SideGroup,
    apply_llm_classification,
    apply_llm_missingness,
    build_missingness_request,
    build_role_request,
    classify_missingness,
    classify_side_roles,
    extract_categories,
    find_article_path,
    find_populated_unclassified_log_ids,
    find_unclassified_battle_type_log_ids,
    infer_battle_type,
    infer_campaign,
    infer_siege_defenders,
    infer_terrain,
    llm_defender_flag,
    load_battle_source_slugs,
    load_battle_wikipedia_urls,
    load_battles_needing_terrain,
    load_battles_needing_type,
    load_missingness_contexts,
    load_missingness_inputs,
    load_side_groups,
    load_siege_sides,
    load_troop_evidence,
    log_unlogged_troop_totals,
    mark_populated_fields_observed,
    read_article_text,
    select_excerpt,
    write_battle_type_decisions,
    write_defender_flags,
    write_missingness_decisions,
    write_role_decisions,
    write_terrain,
)
from pipeline.db import get_connection
from pipeline.extractors.article import clean_article_text
from pipeline.llm import LLMService
from pipeline.llm.base import LLMConfigError, LLMRequest, request_hash
from pipeline.llm.call_log import find_completed_call
from pipeline.llm.factory import llm_params
from pipeline.llm.offline import OfflineProvider
from pipeline.stages.base import StageContext, StageOutcome

__all__ = [
    "LazyService",
    "build_pending_missingness_requests",
    "build_pending_role_requests",
    "run",
]

logger = structlog.get_logger()

PROCESSED_ROOT: Final[Path] = Path("data/processed")
RAW_ROOT: Final[Path] = Path("data/raw")

_DEFAULT_EXCERPT_MAX_CHARS: Final[int] = 6000
_PARAGRAPH_RE: Final = re.compile(r"\n{2,}")


class LazyService:
    """An LLM service built on first use, only when ``llm_mode`` asks for one.

    Most sides settle deterministically or from the offline cache, so a run
    with nothing left ambiguous -- or one running in ``offline`` mode at all
    -- needs no credentials. See ``pipeline.stages.resolve.LazyService``,
    which the same reasoning is copied from; the two are not shared because
    resolve's factory always builds a service (its LLM step is unconditional
    once a group is ambiguous) while classify's is conditioned on the mode
    param as well as on there being any pending side at all.
    """

    def __init__(self, factory: Callable[[], LLMService] | None) -> None:
        """Wrap a factory.

        Args:
            factory: Builds the service, or None when this run must not use
                one (``llm_mode == "offline"``, or a dry run).
        """
        self._factory = factory
        self._service: LLMService | None = None

    @property
    def built(self) -> LLMService | None:
        """The service, if one was ever built."""
        return self._service

    def get(self) -> LLMService:
        """Build the service, or return the one already built.

        Returns:
            The service.

        Raises:
            LLMConfigError: If no factory was given, or the provider or
                credentials are unusable.
        """
        if self._factory is None:
            raise LLMConfigError(
                "classify stage reached the LLM step with no factory configured "
                "(params.llm_mode is not 'api', or this is a dry run)"
            )
        if self._service is None:
            self._service = self._factory()
        return self._service


def _deterministic_roles(
    groups: list[SideGroup],
) -> tuple[dict[int, list[RoleDecision]], list[SideGroup]]:
    """Apply the deterministic role rule to every side.

    Args:
        groups: Every side to classify.

    Returns:
        Decisions keyed by ``side_id``, and the sides whose decision was the
        spec's ``default_split`` fallback on every commander -- exactly the
        sides :func:`build_pending_role_requests` and the LLM step consider.
    """
    decisions: dict[int, list[RoleDecision]] = {}
    pending: list[SideGroup] = []
    for side in groups:
        if side.commanders and all(c.attribution_method == "llm" for c in side.commanders):
            # Already read by the LLM on an earlier run. Its roles now look
            # consistent, so the deterministic rule would recompute them from
            # the fixed weight table and relabel them rule_roles -- replacing
            # the model's judgement of the article with a constant. Leave the
            # rows exactly as they are; nothing is written for this side.
            continue
        side_decisions = classify_side_roles(side)
        decisions[side.side_id] = side_decisions
        if side_decisions and all(d.attribution_method == "default_split" for d in side_decisions):
            pending.append(side)
    return decisions, pending


def _load_article(
    raw_root: Path, wikipedia_url: str | None, source_slug: str | None = None
) -> tuple[str, list[str]]:
    """Read and clean a battle's raw article, if one exists on disk.

    Args:
        raw_root: The ``data/raw`` directory, or a test fixture root.
        wikipedia_url: The battle's stored URL, or None.
        source_slug: The battle's crawl slug, or None.

    Returns:
        ``(cleaned text, categories)``, both empty when no article file was
        found or it could not be read -- a legitimate "no evidence" outcome
        for both the role excerpt and the battle-type inference that read it.
    """
    path = find_article_path(raw_root, wikipedia_url, source_slug)
    if path is None:
        return "", []
    raw = read_article_text(path)
    if raw is None:
        return "", []
    is_html = path.suffix.lower() in HTML_SUFFIXES
    return clean_article_text(raw, is_html=is_html) or "", extract_categories(raw)


def _build_excerpt(article_text: str, names: list[str], max_chars: int) -> str:
    """Split a cleaned article into paragraphs and hand them to ``select_excerpt``.

    Args:
        article_text: The battle's cleaned article text, or "" when none.
        names: The side's commander names.
        max_chars: The spec's ``params.excerpt_max_chars``.

    Returns:
        The chosen excerpt; "" when there is no article at all.
    """
    if not article_text:
        return ""
    passages = [p.strip() for p in _PARAGRAPH_RE.split(article_text) if p.strip()]
    if not passages:
        passages = [article_text]
    return select_excerpt(passages, names, max_chars)


def _build_request_pairs(
    pending: list[SideGroup],
    battle_urls: dict[int, str | None],
    raw_root: Path,
    spec: dict[str, Any],
    excerpt_max_chars: int,
    battle_slugs: dict[int, str | None] | None = None,
) -> list[tuple[SideGroup, LLMRequest]]:
    """Build one command-role request per already-identified pending side.

    Args:
        pending: Sides whose deterministic decision was the spec's
            ``default_split`` fallback on every commander.
        battle_urls: Every battle's URL, keyed by battle id.
        raw_root: The ``data/raw`` directory, or a test fixture root.
        spec: The loaded ``agents/classify.yaml``.
        excerpt_max_chars: The spec's ``params.excerpt_max_chars``.
        battle_slugs: Every battle's crawl slug, keyed by battle id; preferred
            over the URL when locating the article.

    Returns:
        One ``(side, request)`` pair per pending side, in the same order.
        An article shared by several sides of the same battle is read once.
    """
    article_cache: dict[int, str] = {}
    pairs: list[tuple[SideGroup, LLMRequest]] = []
    for side in pending:
        article_text = article_cache.get(side.battle_id)
        if article_text is None:
            article_text, _categories = _load_article(
                raw_root,
                battle_urls.get(side.battle_id),
                (battle_slugs or {}).get(side.battle_id),
            )
            article_cache[side.battle_id] = article_text

        excerpt = _build_excerpt(article_text, [c.name for c in side.commanders], excerpt_max_chars)
        pairs.append((side, build_role_request(side, excerpt, spec)))

    return pairs


def build_pending_role_requests(
    conn: Any,
    raw_root: Path,
    spec: dict[str, Any],
    *,
    excerpt_max_chars: int = _DEFAULT_EXCERPT_MAX_CHARS,
) -> list[tuple[SideGroup, LLMRequest]]:
    """Build the command-role request for every side that needs one.

    The single code path both the stage's own LLM step and
    ``scripts/llm_offline.py``'s exporter call, so a request built here and a
    request built there are always byte-identical and hash the same way --
    the correlation ``export`` -> process -> ``import`` -> re-run depends on.

    Args:
        conn: An open database connection.
        raw_root: The ``data/raw`` directory, or a test fixture root.
        spec: The loaded ``agents/classify.yaml``.
        excerpt_max_chars: The spec's ``params.excerpt_max_chars``.

    Returns:
        One ``(side, request)`` pair per side whose deterministic decision
        was the spec's ``default_split`` fallback on every commander, in
        ``side_id`` order.
    """
    groups = load_side_groups(conn)
    _decisions, pending = _deterministic_roles(groups)
    if not pending:
        return []

    battle_urls = load_battle_wikipedia_urls(conn)
    return _build_request_pairs(
        pending,
        battle_urls,
        raw_root,
        spec,
        excerpt_max_chars,
        load_battle_source_slugs(conn),
    )


def _tally_role_method(decision: RoleDecision, counts: ClassifyCounts) -> None:
    """Fold one role decision's method into the run's counters.

    Args:
        decision: A final role decision, deterministic or LLM-derived.
        counts: The run's counters.
    """
    if decision.attribution_method == "rule_single":
        counts.single_commander += 1
    elif decision.attribution_method == "rule_roles":
        counts.rule_roles += 1
    elif decision.attribution_method == "llm":
        counts.llm_classified += 1
    else:
        # Still 'default_split': either no LLM step ran, or it ran and found
        # nothing -- a cache miss in 'offline' mode, or a failed call in
        # 'api' mode. Either way the side is exactly what the brief calls
        # "awaiting_llm"; ClassifyCounts has no field of that name, so this
        # count is what stands in for it, and is reported as such.
        counts.default_split += 1
    if decision.needs_review:
        counts.needs_review += 1


def _resolve_pending_roles(
    pairs: list[tuple[SideGroup, LLMRequest]],
    conn: Any,
    *,
    llm_mode: str,
    provider: str,
    model: str,
    service: LazyService,
    counts: ClassifyCounts,
    dry_run: bool,
    offline: OfflineProvider | None = None,
    defender_flags: dict[int, bool] | None = None,
) -> dict[int, list[RoleDecision]]:
    """Resolve every pending side's request, offline or live.

    Args:
        pairs: Sides and their built requests, from
            :func:`build_pending_role_requests`.
        conn: An open database connection, for the offline cache lookup.
        llm_mode: ``params.llm_mode``, already validated as 'offline' or 'api'.
        provider: ``params.llm_provider`` (via :func:`pipeline.llm.factory
            .llm_params`), for the offline hash lookup.
        model: ``params.llm_model``, likewise.
        service: The stage's deferred LLM client, used only in 'api' mode.
        counts: The run's counters; every request built increments
            ``llm_calls``, whether it is served from the cache, sent live, or
            (in a dry run) not attempted at all.
        dry_run: When True, requests are counted but never looked up or sent.
        offline: Collects the offline-mode cache misses for the batch file.
            Without one, a miss is only left at the fallback.
        defender_flags: When given, filled with ``side_id`` to the LLM's
            ``is_defender`` answer for each side whose answer was accepted
            and stated a boolean.

    Returns:
        Final role decisions for every side an answer was found for, keyed
        by ``side_id``. A side with no answer is absent, leaving its
        deterministic ``default_split`` decision as the final one.
    """
    resolved: dict[int, list[RoleDecision]] = {}

    for side, request in pairs:
        counts.llm_calls += 1
        if dry_run:
            continue

        if llm_mode == "offline":
            digest = request_hash(request, provider, model)
            cached = find_completed_call(conn, digest)
            if cached is not None:
                _record_answer(side, cached, resolved, defender_flags)
            elif offline is not None:
                offline.complete(request)
            continue

        client = service.get()
        response = client.complete(
            system=request.system,
            user=request.user,
            json_schema=request.json_schema,
            schema_name=request.schema_name,
            metadata=request.metadata,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        if response.ok and response.data is not None:
            _record_answer(side, response.data, resolved, defender_flags)

    return resolved


def _record_answer(
    side: SideGroup,
    data: dict[str, Any],
    resolved: dict[int, list[RoleDecision]],
    defender_flags: dict[int, bool] | None,
) -> None:
    """Apply one parsed role answer and note its defender flag.

    Args:
        side: The side the answer is for.
        data: The parsed response body.
        resolved: Role decisions by ``side_id``, updated in place.
        defender_flags: ``side_id`` to flag, updated in place; None to ignore flags.
    """
    decisions = apply_llm_classification(side, data)
    resolved[side.side_id] = decisions
    if defender_flags is None:
        return
    flag = llm_defender_flag(decisions, data)
    if flag is not None:
        defender_flags[side.side_id] = flag


def _usable_llm_defender_flags(
    flags: dict[int, bool], pairs: list[tuple[SideGroup, LLMRequest]]
) -> dict[int, bool]:
    """Drop LLM defender flags that contradict each other within a battle.

    Args:
        flags: ``side_id`` to the LLM's flag.
        pairs: The sides the requests were built for, to find each side's battle.

    Returns:
        ``flags`` without any battle in which more than one side was answered
        True: two defenders is not a reading of one engagement, and which of
        them is wrong is not recoverable, so neither is written.
    """
    battle_of = {side.side_id: side.battle_id for side, _request in pairs}
    defenders_per_battle: dict[int, int] = {}
    for side_id, flag in flags.items():
        if flag:
            battle = battle_of[side_id]
            defenders_per_battle[battle] = defenders_per_battle.get(battle, 0) + 1
    return {
        side_id: flag
        for side_id, flag in flags.items()
        if defenders_per_battle.get(battle_of[side_id], 0) <= 1
    }


def _tally_battle_type(decision: BattleTypeDecision, counts: ClassifyCounts) -> None:
    """Fold one battle-type decision into the run's counters.

    Args:
        decision: A battle-type decision.
        counts: The run's counters.
    """
    if decision.battle_type == "naval":
        counts.battle_type_naval += 1
    elif decision.battle_type.startswith("siege"):
        counts.battle_type_siege += 1
    elif decision.battle_type == "aerial":
        counts.battle_type_aerial += 1
    elif decision.battle_type == "amphibious":
        counts.battle_type_amphibious += 1
    elif decision.battle_type == "field":
        counts.battle_type_field += 1
    elif decision.battle_type == "campaign":
        counts.battle_type_campaign += 1
    elif decision.battle_type == "guerrilla":
        counts.battle_type_guerrilla += 1
    else:
        counts.battle_type_unknown += 1


def _infer_battle_types(
    conn: Any,
    raw_root: Path,
    counts: ClassifyCounts,
) -> list[BattleTypeDecision]:
    """Infer a type for every battle still 'unknown', and spot campaigns.

    Args:
        conn: An open database connection.
        raw_root: The ``data/raw`` directory, or a test fixture root.
        counts: The run's counters.

    Returns:
        One decision per battle read. A battle an earlier run typed by rule is
        revisited only for the campaign rule (title and dates, no article
        read); the decision is marked ``replaces_rule_type`` and no other
        decision is produced for it.
    """
    decisions: list[BattleTypeDecision] = []
    for battle in load_battles_needing_type(conn):
        battle_id = battle["battle_id"]
        if battle["current_type"] != "unknown":
            campaign = infer_campaign(
                battle_id, battle["name"], battle["span_days"], battle["date_precision"]
            )
            if campaign is None:
                continue
            decision = replace(campaign, replaces_rule_type=True)
        else:
            article_text, categories = _load_article(
                raw_root, battle["wikipedia_url"], battle.get("source_slug")
            )
            branches, strengths = load_troop_evidence(conn, battle_id)
            decision = infer_battle_type(
                battle_id,
                battle["name"],
                article_text or None,
                categories,
                branches,
                strengths,
                battle["span_days"],
                battle["date_precision"],
            )
        decisions.append(decision)
        counts.battles_typed += 1
        _tally_battle_type(decision, counts)
    return decisions


def _infer_terrain(conn: Any, raw_root: Path) -> dict[int, tuple[list[str], str]]:
    """Read terrain tags for every battle whose ``terrain`` is still empty.

    Args:
        conn: An open database connection.
        raw_root: The ``data/raw`` directory, or a test fixture root.

    Returns:
        ``battle_id`` to ``(tags, evidence)`` for battles whose article
        supports at least one tag.
    """
    found: dict[int, tuple[list[str], str]] = {}
    for battle in load_battles_needing_terrain(conn):
        article_text, _categories = _load_article(
            raw_root, battle["wikipedia_url"], battle.get("source_slug")
        )
        tags, evidence = infer_terrain(article_text)
        if tags:
            found[battle["battle_id"]] = (tags, evidence)
    return found


def _siege_defender_flags(conn: Any) -> dict[int, bool]:
    """Work out ``is_defender`` for every siege whose title names the besieged side.

    Args:
        conn: An open database connection. Read after the battle-type write,
            so a siege typed in this same run is included.

    Returns:
        ``side_id`` to flag; sides the title does not settle are absent and
        stay NULL.
    """
    flags: dict[int, bool] = {}
    for name, sides in load_siege_sides(conn).values():
        flags.update(infer_siege_defenders(name, sides))
    return flags


def _missingness_exclusions(
    conn: Any,
    battle_type_decisions: list[BattleTypeDecision],
    terrain_by_battle: dict[int, tuple[list[str], str]],
) -> set[int]:
    """Collect the ``missing_data_log`` rows the missingness pass must skip.

    One computation for the stage and for the offline export
    (:func:`build_pending_missingness_requests`), so the exported requests
    cannot drift from the ones the stage builds.

    Args:
        conn: An open database connection.
        battle_type_decisions: This run's battle-type decisions.
        terrain_by_battle: This run's terrain inference.

    Returns:
        The ``log_id`` values of rows this run settles through another route:
        battle types and terrain it infers, and covariate rows whose column
        is already populated.
    """
    observed_battle_ids = {d.battle_id for d in battle_type_decisions if d.battle_type != "unknown"}
    exclude = find_unclassified_battle_type_log_ids(conn, observed_battle_ids)
    exclude |= find_unclassified_battle_type_log_ids(conn, set(terrain_by_battle), "terrain")
    exclude |= find_populated_unclassified_log_ids(
        conn, {d.battle_id for d in battle_type_decisions if d.fortified is not None}
    )
    return exclude


def _classify_missingness_rows(
    conn: Any, counts: ClassifyCounts, *, exclude_log_ids: set[int]
) -> tuple[list[MissingnessDecision], list[int]]:
    """Classify every still-unclassified, in-scope ``missing_data_log`` row.

    Args:
        conn: An open database connection.
        counts: The run's counters.
        exclude_log_ids: ``battle_type`` and ``terrain`` rows this same run's
            passes already resolved (see :func:`pipeline.classifiers.load
            .find_unclassified_battle_type_log_ids`). Both passes read the
            database as it was before either wrote anything, so without this
            a row the battle-type pass is about to mark 'observed' would
            still read 'unclassified' here and get a heuristic decision
            computed for it -- one the write step then silently discards,
            since it guards on the same 'unclassified' state. Filtering here
            keeps the counters honest rather than relying on write order.

    Returns:
        ``(decisions, undecided)``: one decision per row the heuristics
        reached a conclusion for, and the ``log_id`` of each row they left at
        None (:func:`pipeline.classifiers.missingness.classify_missingness`),
        which the LLM step may still answer. A row nobody answers stays
        'unclassified'.
    """
    decisions: list[MissingnessDecision] = []
    undecided: list[int] = []
    inputs = [
        record for record in load_missingness_inputs(conn) if record.log_id not in exclude_log_ids
    ]
    counts.missing_fields_seen = len(inputs)
    for record in inputs:
        decision = classify_missingness(record)
        if decision is None:
            undecided.append(record.log_id)
            continue
        decisions.append(decision)
        counts.missingness_classified += 1
    return decisions, undecided


def _build_missingness_pairs(
    conn: Any, log_ids: list[int], spec: dict[str, Any]
) -> list[tuple[MissingnessContext, LLMRequest]]:
    """Build one missingness request per row the heuristics left open.

    Args:
        conn: An open database connection.
        log_ids: The undecided ``missing_data_log`` rows.
        spec: The loaded ``agents/classify.yaml``.

    Returns:
        ``(context, request)`` pairs in ``log_id`` order.
    """
    return [
        (context, build_missingness_request(context, spec))
        for context in load_missingness_contexts(conn, log_ids)
    ]


def build_pending_missingness_requests(
    conn: Any, spec: dict[str, Any], raw_root: Path = Path("data/raw")
) -> list[tuple[MissingnessContext, LLMRequest]]:
    """Build the missingness request for every row the heuristics cannot settle.

    The path ``scripts/llm_offline.py`` calls, so an exported request and the
    one the stage builds hash alike. It applies the stage's own exclusions
    (rows the battle-type and terrain inference settle, and populated
    covariate rows), computed read-only from the same inference.

    Args:
        conn: An open database connection.
        spec: The loaded ``agents/classify.yaml``.
        raw_root: The ``data/raw`` directory the inference reads articles from.

    Returns:
        ``(context, request)`` pairs in ``log_id`` order.
    """
    exclude = _missingness_exclusions(
        conn,
        _infer_battle_types(conn, raw_root, ClassifyCounts()),
        _infer_terrain(conn, raw_root),
    )
    undecided = [
        record.log_id
        for record in load_missingness_inputs(conn)
        if record.log_id not in exclude and classify_missingness(record) is None
    ]
    return _build_missingness_pairs(conn, undecided, spec)


def _resolve_pending_missingness(
    pairs: list[tuple[MissingnessContext, LLMRequest]],
    conn: Any,
    *,
    llm_mode: str,
    provider: str,
    model: str,
    service: LazyService,
    counts: ClassifyCounts,
    dry_run: bool,
    offline: OfflineProvider | None,
) -> list[MissingnessDecision]:
    """Answer the open missingness rows from the cache, offline or live.

    Args:
        pairs: Contexts and their requests.
        conn: An open database connection, for the offline cache lookup.
        llm_mode: 'offline' or 'api'.
        provider: ``params.llm_provider``, for the hash.
        model: ``params.llm_model``, for the hash.
        service: The stage's deferred LLM client, used only in 'api' mode.
        counts: The run's counters.
        dry_run: When True, requests are counted and never looked up or sent.
        offline: Collects offline cache misses for the batch file.

    Returns:
        A decision per row an accepted answer named a mechanism for. Every
        other row stays 'unclassified'.
    """
    decisions: list[MissingnessDecision] = []
    for context, request in pairs:
        counts.llm_calls += 1
        if dry_run:
            continue
        data: dict[str, Any] | None = None
        if llm_mode == "offline":
            data = find_completed_call(conn, request_hash(request, provider, model))
            if data is None and offline is not None:
                offline.complete(request)
        else:
            response = service.get().complete(
                system=request.system,
                user=request.user,
                json_schema=request.json_schema,
                schema_name=request.schema_name,
                metadata=request.metadata,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            if response.ok:
                data = response.data
        if data is None:
            continue
        decision = apply_llm_missingness(context.log_id, data)
        if decision is not None:
            decisions.append(decision)
            counts.missingness_llm_classified += 1
    return decisions


def _run(spec: dict[str, Any], ctx: StageContext, conn: Any) -> StageOutcome | None:
    """Run the stage against an already-open connection.

    Args:
        spec: The loaded ``agents/classify.yaml``.
        ctx: The stage context (dry run, row limit).
        conn: An open database connection.

    Returns:
        A :class:`StageOutcome` naming the batch file when offline requests
        were deferred, otherwise None.

    Raises:
        ValueError: If ``params.llm_mode`` is neither 'offline' nor 'api'.
    """
    params = spec.get("params") or {}
    raw_root = Path(str(params.get("raw_root", RAW_ROOT)))
    excerpt_max_chars = int(params.get("excerpt_max_chars", _DEFAULT_EXCERPT_MAX_CHARS))

    llm_mode = str(params.get("llm_mode", "offline")).strip().lower()
    if llm_mode not in ("offline", "api"):
        raise ValueError(
            f"agents/classify.yaml params.llm_mode is {llm_mode!r}; expected 'offline' or 'api'"
        )

    llm_config = llm_params(spec)
    provider = str(llm_config["provider"])
    model = str(llm_config["model"] or "")

    counts = ClassifyCounts()

    groups = load_side_groups(conn)
    if ctx.limit is not None:
        groups = groups[: ctx.limit]
    counts.sides_processed = len(groups)

    decisions_by_side, pending = _deterministic_roles(groups)

    battle_urls = load_battle_wikipedia_urls(conn)
    pairs = _build_request_pairs(
        pending,
        battle_urls,
        raw_root,
        spec,
        excerpt_max_chars,
        load_battle_source_slugs(conn),
    )

    service = LazyService(
        None
        if ctx.dry_run or llm_mode != "api"
        else lambda: LLMService.from_spec(spec, db_conn=conn)
    )

    offline = OfflineProvider(provider, model) if llm_mode == "offline" else None
    llm_defenders: dict[int, bool] = {}
    resolved = _resolve_pending_roles(
        pairs,
        conn,
        llm_mode=llm_mode,
        provider=provider,
        model=model,
        service=service,
        counts=counts,
        dry_run=ctx.dry_run,
        offline=offline,
        defender_flags=llm_defenders,
    )
    decisions_by_side.update(resolved)

    all_role_decisions = [d for decisions in decisions_by_side.values() for d in decisions]
    for decision in all_role_decisions:
        _tally_role_method(decision, counts)

    battle_type_decisions = _infer_battle_types(conn, raw_root, counts)
    terrain_by_battle = _infer_terrain(conn, raw_root)
    exclude_log_ids = _missingness_exclusions(conn, battle_type_decisions, terrain_by_battle)
    if not ctx.dry_run:
        # Before the missingness read, so a newly logged gap is classified in
        # this same run rather than waiting for the next.
        counts.troop_gaps_logged = log_unlogged_troop_totals(conn)
    missingness_decisions, undecided = _classify_missingness_rows(
        conn, counts, exclude_log_ids=exclude_log_ids
    )
    missingness_decisions += _resolve_pending_missingness(
        _build_missingness_pairs(conn, undecided, spec),
        conn,
        llm_mode=llm_mode,
        provider=provider,
        model=model,
        service=service,
        counts=counts,
        dry_run=ctx.dry_run,
        offline=offline,
    )
    counts.missingness_left_unclassified = len(undecided) - counts.missingness_llm_classified

    if ctx.dry_run:
        counts.terrain_set = len(terrain_by_battle)
        logger.info("classify_dry_run", **counts.as_dict())
        return None

    write_role_decisions(conn, all_role_decisions)
    write_battle_type_decisions(conn, battle_type_decisions)
    counts.terrain_set = write_terrain(conn, terrain_by_battle)
    mark_populated_fields_observed(conn)
    write_missingness_decisions(conn, missingness_decisions)
    # The siege-title rule first: it reads the besieged place off the title,
    # which beats the model's reading, and the writer only fills NULLs.
    counts.defenders_set = write_defender_flags(conn, _siege_defender_flags(conn))
    counts.defenders_set += write_defender_flags(
        conn, _usable_llm_defender_flags(llm_defenders, pairs)
    )

    if service.built is not None:
        service.built.log_summary()

    outcome: StageOutcome | None = None
    if offline is not None and offline.pending:
        pending_count = offline.pending
        batch_file = offline.flush("classify")
        logger.info(
            "llm_requests_deferred", stage="classify", count=pending_count, path=str(batch_file)
        )
        outcome = StageOutcome(deferred_requests=pending_count, batch_file=batch_file)

    logger.info("classify_complete", **counts.as_dict())
    return outcome


def run(spec: dict[str, Any], context: StageContext | None = None) -> StageOutcome | None:
    """Run the classify stage.

    Args:
        spec: The loaded ``agents/classify.yaml``, with overrides applied.
        context: Shared services. Every output of this stage is a database
            write, so unlike crawl or resolve there is no reduced mode for
            missing a connection: one is opened from ``DATABASE_URL`` when
            ``context.db_conn`` is None, and failing to reach one fails the
            stage rather than silently doing nothing.

    Raises:
        ValueError: If ``params.llm_mode`` is neither 'offline' nor 'api'.
        DatabaseConfigError: If no database connection was given and none
            can be opened.

    Returns:
        A :class:`StageOutcome` when requests were deferred, else None.
    """
    ctx = context or StageContext()
    if ctx.db_conn is not None:
        return _run(spec, ctx, ctx.db_conn)
    with get_connection() as conn:
        return _run(spec, ctx, conn)
