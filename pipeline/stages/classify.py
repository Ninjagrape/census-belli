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
  deterministic fallback and counted, not guessed at. No network, no API
  client is built.
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
from pathlib import Path
from typing import Any, Final

import structlog

from pipeline.classifiers import (
    HTML_SUFFIXES,
    BattleTypeDecision,
    ClassifyCounts,
    MissingnessDecision,
    RoleDecision,
    SideGroup,
    apply_llm_classification,
    build_role_request,
    classify_missingness,
    classify_side_roles,
    extract_categories,
    find_article_path,
    find_unclassified_battle_type_log_ids,
    infer_battle_type,
    load_battle_wikipedia_urls,
    load_battles_needing_type,
    load_missingness_inputs,
    load_side_groups,
    load_troop_evidence,
    read_article_text,
    select_excerpt,
    write_battle_type_decisions,
    write_missingness_decisions,
    write_role_decisions,
)
from pipeline.db import get_connection
from pipeline.extractors.article import clean_article_text
from pipeline.llm import LLMService
from pipeline.llm.base import LLMConfigError, LLMRequest, request_hash
from pipeline.llm.call_log import find_completed_call
from pipeline.llm.factory import llm_params
from pipeline.stages.base import StageContext

__all__ = ["LazyService", "build_pending_role_requests", "run"]

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


def _load_article(raw_root: Path, wikipedia_url: str | None) -> tuple[str, list[str]]:
    """Read and clean a battle's raw article, if one exists on disk.

    Args:
        raw_root: The ``data/raw`` directory, or a test fixture root.
        wikipedia_url: The battle's stored URL, or None.

    Returns:
        ``(cleaned text, categories)``, both empty when no article file was
        found or it could not be read -- a legitimate "no evidence" outcome
        for both the role excerpt and the battle-type inference that read it.
    """
    path = find_article_path(raw_root, wikipedia_url)
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
) -> list[tuple[SideGroup, LLMRequest]]:
    """Build one command-role request per already-identified pending side.

    Args:
        pending: Sides whose deterministic decision was the spec's
            ``default_split`` fallback on every commander.
        battle_urls: Every battle's URL, keyed by battle id.
        raw_root: The ``data/raw`` directory, or a test fixture root.
        spec: The loaded ``agents/classify.yaml``.
        excerpt_max_chars: The spec's ``params.excerpt_max_chars``.

    Returns:
        One ``(side, request)`` pair per pending side, in the same order.
        An article shared by several sides of the same battle is read once.
    """
    article_cache: dict[int, str] = {}
    pairs: list[tuple[SideGroup, LLMRequest]] = []
    for side in pending:
        article_text = article_cache.get(side.battle_id)
        if article_text is None:
            article_text, _categories = _load_article(raw_root, battle_urls.get(side.battle_id))
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
    return _build_request_pairs(pending, battle_urls, raw_root, spec, excerpt_max_chars)


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
                resolved[side.side_id] = apply_llm_classification(side, cached)
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
            resolved[side.side_id] = apply_llm_classification(side, response.data)

    return resolved


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
    else:
        counts.battle_type_unknown += 1


def _infer_battle_types(
    conn: Any,
    raw_root: Path,
    counts: ClassifyCounts,
) -> list[BattleTypeDecision]:
    """Infer a type for every battle still 'unknown'.

    Args:
        conn: An open database connection.
        raw_root: The ``data/raw`` directory, or a test fixture root.
        counts: The run's counters.

    Returns:
        One decision per battle read.
    """
    decisions: list[BattleTypeDecision] = []
    for battle in load_battles_needing_type(conn):
        battle_id = battle["battle_id"]
        article_text, categories = _load_article(raw_root, battle["wikipedia_url"])
        branches, strengths = load_troop_evidence(conn, battle_id)
        decision = infer_battle_type(
            battle_id, battle["name"], article_text or None, categories, branches, strengths
        )
        decisions.append(decision)
        counts.battles_typed += 1
        _tally_battle_type(decision, counts)
    return decisions


def _classify_missingness_rows(
    conn: Any, counts: ClassifyCounts, *, exclude_log_ids: set[int]
) -> list[MissingnessDecision]:
    """Classify every still-unclassified, in-scope ``missing_data_log`` row.

    Args:
        conn: An open database connection.
        counts: The run's counters.
        exclude_log_ids: ``battle_type`` rows this same run's battle-type
            pass already resolved (see :func:`pipeline.classifiers.load
            .find_unclassified_battle_type_log_ids`). Both passes read the
            database as it was before either wrote anything, so without this
            a row the battle-type pass is about to mark 'observed' would
            still read 'unclassified' here and get a heuristic decision
            computed for it -- one the write step then silently discards,
            since it guards on the same 'unclassified' state. Filtering here
            keeps the counters honest rather than relying on write order.

    Returns:
        One decision per row the heuristics reached a conclusion for. A row
        left at None by :func:`pipeline.classifiers.missingness
        .classify_missingness` is not represented, and stays 'unclassified'.
    """
    decisions: list[MissingnessDecision] = []
    inputs = [
        record for record in load_missingness_inputs(conn) if record.log_id not in exclude_log_ids
    ]
    counts.missing_fields_seen = len(inputs)
    for record in inputs:
        decision = classify_missingness(record)
        if decision is None:
            counts.missingness_left_unclassified += 1
            continue
        decisions.append(decision)
        counts.missingness_classified += 1
    return decisions


def _run(spec: dict[str, Any], ctx: StageContext, conn: Any) -> None:
    """Run the stage against an already-open connection.

    Args:
        spec: The loaded ``agents/classify.yaml``.
        ctx: The stage context (dry run, row limit).
        conn: An open database connection.

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
    pairs = _build_request_pairs(pending, battle_urls, raw_root, spec, excerpt_max_chars)

    service = LazyService(
        None
        if ctx.dry_run or llm_mode != "api"
        else lambda: LLMService.from_spec(spec, db_conn=conn)
    )

    resolved = _resolve_pending_roles(
        pairs,
        conn,
        llm_mode=llm_mode,
        provider=provider,
        model=model,
        service=service,
        counts=counts,
        dry_run=ctx.dry_run,
    )
    decisions_by_side.update(resolved)

    all_role_decisions = [d for decisions in decisions_by_side.values() for d in decisions]
    for decision in all_role_decisions:
        _tally_role_method(decision, counts)

    battle_type_decisions = _infer_battle_types(conn, raw_root, counts)
    observed_battle_ids = {d.battle_id for d in battle_type_decisions if d.battle_type != "unknown"}
    exclude_log_ids = find_unclassified_battle_type_log_ids(conn, observed_battle_ids)
    missingness_decisions = _classify_missingness_rows(
        conn, counts, exclude_log_ids=exclude_log_ids
    )

    if ctx.dry_run:
        logger.info("classify_dry_run", **counts.as_dict())
        return

    write_role_decisions(conn, all_role_decisions)
    write_battle_type_decisions(conn, battle_type_decisions)
    write_missingness_decisions(conn, missingness_decisions)

    if service.built is not None:
        service.built.log_summary()

    logger.info("classify_complete", **counts.as_dict())


def run(spec: dict[str, Any], context: StageContext | None = None) -> None:
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
    """
    ctx = context or StageContext()
    if ctx.db_conn is not None:
        _run(spec, ctx, ctx.db_conn)
    else:
        with get_connection() as conn:
            _run(spec, ctx, conn)
