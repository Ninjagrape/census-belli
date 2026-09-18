"""
The crawl stage: fetch raw content, interpret nothing.

Order of work, and why:

1. Battle-list pages are parsed for article links. The fourteen seed lists
   overlap, so links are deduplicated before anything is fetched.
2. Each battle article is fetched once, written to ``data/raw/battles_html``,
   and mined for citations. Allowed citation domains are fetched; the rest
   are recorded as skipped so the trail stays complete.
3. Wikidata SPARQL queries run last. They are slow and independent, so a
   crawl interrupted before them still leaves the Wikipedia corpus usable.

Every fetch attempt, including failures and robots refusals, becomes a
``crawl_log`` row, and progress is written to ``data/raw/crawl_state.json``
after each batch so an interrupted run resumes instead of re-requesting
thousands of pages from Wikimedia.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import structlog

from pipeline.crawlers import citations as citations_mod
from pipeline.crawlers import dbpedia as dbpedia_mod
from pipeline.crawlers import wikidata as wikidata_mod
from pipeline.crawlers import wikipedia as wikipedia_mod
from pipeline.crawlers.fetcher import (
    DEFAULT_TIMEOUT_S,
    Fetcher,
    FetchResult,
    RateLimiter,
)
from pipeline.crawlers.log import CrawlLogWriter, SqlCrawlLog, entry_from_result
from pipeline.crawlers.seeds import DEFAULT_SEED_PATH, Seeds, load_seeds
from pipeline.crawlers.state import DEFAULT_STATE_PATH, CrawlState, load_state, save_state
from pipeline.db import get_connection
from pipeline.stages.base import StageContext

__all__ = [
    "CrawlConfigError",
    "CrawlParams",
    "CrawlPaths",
    "CrawlSummary",
    "crawl",
    "run",
]

logger = structlog.get_logger()

# Spec params the stage refuses to guess at. A missing one is a bug in the
# spec, not a property of the data, so it raises rather than defaulting.
_REQUIRED_PARAMS = (
    "rate_limit_wikipedia",
    "rate_limit_other",
    "max_retries",
    "backoff_base",
    "respect_robots",
    "user_agent",
    "citation_allow_domains",
)

# Save resume state this often, trading a little IO for losing at most this
# many articles' progress to an interrupt.
_STATE_SAVE_EVERY = 25

# Entity fetches are one request each at the non-Wikipedia rate, so the full
# 50k result set of a seed query would take days. The cap keeps the stage
# finite; raise it with --set max_wikidata_entities=N.
_DEFAULT_MAX_WIKIDATA_ENTITIES = 2000


class CrawlConfigError(ValueError):
    """Raised when the crawl spec is missing or misstates a parameter."""


@dataclass(frozen=True)
class CrawlPaths:
    """Where the stage writes its raw output.

    Attributes:
        battles_html: One .html per battle article.
        wikidata: One .json per SPARQL query and per entity.
        dbpedia: One .json per DBpedia resource.
        citations: One .jsonl per battle, holding its fetched citations.
        state_file: Resume state.
    """

    battles_html: Path
    wikidata: Path
    dbpedia: Path
    citations: Path
    state_file: Path

    @classmethod
    def under(cls, root: Path | str, state_file: Path | str = DEFAULT_STATE_PATH) -> CrawlPaths:
        """Build the standard layout under a raw-data root.

        Args:
            root: The raw data directory, normally ``data/raw``.
            state_file: Where resume state lives.

        Returns:
            The paths, not yet created on disk.
        """
        base = Path(root)
        return cls(
            battles_html=base / "battles_html",
            wikidata=base / "wikidata",
            dbpedia=base / "dbpedia",
            citations=base / "citations",
            state_file=Path(state_file),
        )

    def ensure(self) -> None:
        """Create the output directories.

        Raises:
            OSError: If a directory cannot be created.
        """
        for directory in (self.battles_html, self.wikidata, self.dbpedia, self.citations):
            directory.mkdir(parents=True, exist_ok=True)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class CrawlParams:
    """The crawl stage's runtime settings, validated.

    Attributes:
        rate_limit_wikipedia: Seconds between requests to Wikimedia hosts.
        rate_limit_other: Seconds between requests to any other host.
        max_retries: Retries after the first attempt.
        backoff_base: Seconds for the first backoff; doubles each retry.
        respect_robots: Whether robots.txt is consulted.
        user_agent: Sent on every request.
        citation_allow_domains: Domains whose citations may be fetched.
        citation_depth: Hops from a battle article. 0 disables citations.
        fetch_dbpedia: Whether to fetch the DBpedia twin of each article.
        max_wikidata_entities: Cap on per-entity Wikidata fetches.
        seed_path: Path to the seed config.
        paths: Where output is written.
    """

    rate_limit_wikipedia: float
    rate_limit_other: float
    max_retries: int
    backoff_base: float
    respect_robots: bool
    user_agent: str
    citation_allow_domains: tuple[str, ...]
    citation_depth: int
    fetch_dbpedia: bool
    max_wikidata_entities: int
    seed_path: Path
    paths: CrawlPaths

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> CrawlParams:
        """Read and validate the params section of the crawl spec.

        Args:
            spec: The loaded agent spec.

        Returns:
            The validated params.

        Raises:
            CrawlConfigError: If a required param is absent or has a type the
                stage cannot use.
        """
        params = spec.get("params")
        if not isinstance(params, dict):
            raise CrawlConfigError(
                "agents/crawl.yaml must define a params mapping; got "
                f"{type(params).__name__}"
            )

        missing = [key for key in _REQUIRED_PARAMS if params.get(key) is None]
        if missing:
            raise CrawlConfigError(f"agents/crawl.yaml is missing required param(s): {missing}")

        allow_domains = params["citation_allow_domains"]
        if not isinstance(allow_domains, list) or not all(
            isinstance(domain, str) for domain in allow_domains
        ):
            raise CrawlConfigError("citation_allow_domains must be a list of domain strings")

        user_agent = str(params["user_agent"]).strip()
        if not user_agent:
            raise CrawlConfigError("user_agent must be a non-empty string")

        root = params.get("output_root", "data/raw")
        state_file = params.get("state_file", DEFAULT_STATE_PATH)

        return cls(
            rate_limit_wikipedia=_as_float(params, "rate_limit_wikipedia"),
            rate_limit_other=_as_float(params, "rate_limit_other"),
            max_retries=_as_int(params, "max_retries"),
            backoff_base=_as_float(params, "backoff_base"),
            respect_robots=bool(params["respect_robots"]),
            user_agent=user_agent,
            citation_allow_domains=tuple(allow_domains),
            citation_depth=int(params.get("citation_depth", 1)),
            fetch_dbpedia=bool(params.get("fetch_dbpedia", True)),
            max_wikidata_entities=int(
                params.get("max_wikidata_entities", _DEFAULT_MAX_WIKIDATA_ENTITIES)
            ),
            seed_path=Path(params.get("seed_path", DEFAULT_SEED_PATH)),
            paths=CrawlPaths.under(root, state_file),
        )

    def build_limiter(self) -> RateLimiter:
        """Create the per-domain rate limiter these params describe.

        Returns:
            A limiter pacing Wikimedia hosts at the Wikipedia rate and
            everything else at the other rate.
        """
        return RateLimiter(
            self.rate_limit_other,
            {
                "wikipedia.org": self.rate_limit_wikipedia,
                "wikimedia.org": self.rate_limit_wikipedia,
                "wikidata.org": self.rate_limit_wikipedia,
            },
        )


def _as_float(params: dict[str, Any], key: str) -> float:
    """Read a numeric param.

    Args:
        params: The params mapping.
        key: The param name.

    Returns:
        The value as a float.

    Raises:
        CrawlConfigError: If the value is not numeric.
    """
    value = params[key]
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise CrawlConfigError(f"{key} must be a number, got {type(value).__name__}")
    try:
        return float(value)
    except ValueError as exc:
        raise CrawlConfigError(f"{key} must be a number, got {value!r}") from exc


def _as_int(params: dict[str, Any], key: str) -> int:
    """Read an integer param.

    Args:
        params: The params mapping.
        key: The param name.

    Returns:
        The value as an int.

    Raises:
        CrawlConfigError: If the value is not an integer.
    """
    return int(_as_float(params, key))


@dataclass
class CrawlSummary:
    """What a crawl run accomplished.

    Attributes:
        list_pages: Battle-list pages fetched this run.
        battles_discovered: New battle URLs found this run.
        battles_fetched: Battle articles successfully fetched.
        citations_fetched: Citation URLs fetched.
        citations_skipped: Citation URLs off the allow list.
        dbpedia_fetched: DBpedia resources fetched.
        wikidata_queries: SPARQL queries that returned results.
        wikidata_entities: Wikidata entity documents fetched.
        failures: Fetch attempts that yielded no usable body.
    """

    list_pages: int = 0
    battles_discovered: int = 0
    battles_fetched: int = 0
    citations_fetched: int = 0
    citations_skipped: int = 0
    dbpedia_fetched: int = 0
    wikidata_queries: int = 0
    wikidata_entities: int = 0
    failures: int = 0
    failed_urls: list[str] = field(default_factory=list)

    def as_log_fields(self) -> dict[str, int]:
        """Return the counts for a structlog line.

        Returns:
            Scalar counters only.
        """
        return {
            "list_pages": self.list_pages,
            "battles_discovered": self.battles_discovered,
            "battles_fetched": self.battles_fetched,
            "citations_fetched": self.citations_fetched,
            "citations_skipped": self.citations_skipped,
            "dbpedia_fetched": self.dbpedia_fetched,
            "wikidata_queries": self.wikidata_queries,
            "wikidata_entities": self.wikidata_entities,
            "failures": self.failures,
        }


def _record(
    writer: CrawlLogWriter,
    state: CrawlState,
    summary: CrawlSummary,
    result: FetchResult,
) -> bool:
    """Log one fetch attempt to crawl_log and resume state.

    Args:
        writer: The crawl_log sink.
        state: Resume state.
        summary: Counters for this run.
        result: The attempt to record.

    Returns:
        True if the fetch produced a usable body.
    """
    writer.record(entry_from_result(result))
    if result.ok:
        state.mark_fetched(result.url, result.content_hash)
        return True

    state.mark_failed(result.url, result.error or "unknown error")
    summary.failures += 1
    if len(summary.failed_urls) < 100:
        summary.failed_urls.append(result.url)
    return False


async def _crawl_list_pages(
    fetcher: Fetcher,
    seeds: Seeds,
    state: CrawlState,
    writer: CrawlLogWriter,
    summary: CrawlSummary,
) -> None:
    """Fetch and parse the battle-list pages, discovering article URLs.

    Args:
        fetcher: The configured fetcher.
        seeds: Seed config.
        state: Resume state, updated in place.
        writer: The crawl_log sink.
        summary: Counters for this run.
    """
    for list_url in seeds.battle_lists:
        if list_url in state.list_pages_done:
            logger.debug("list_page_already_done", url=list_url)
            continue

        result = await fetcher.fetch(list_url)
        if not _record(writer, state, summary, result) or result.text is None:
            continue

        summary.list_pages += 1
        links = wikipedia_mod.parse_battle_links(result.text, list_url)
        added = state.add_battle_urls(links)
        summary.battles_discovered += len(added)
        state.mark_list_page_done(list_url)

        logger.info(
            "list_page_parsed",
            url=list_url,
            links=len(links),
            new=len(added),
            known=len(state.battle_urls),
        )


async def _crawl_citations(
    fetcher: Fetcher,
    params: CrawlParams,
    battle_url: str,
    html: str,
    state: CrawlState,
    writer: CrawlLogWriter,
    summary: CrawlSummary,
) -> None:
    """Fetch the allowed citations of one battle article.

    Args:
        fetcher: The configured fetcher.
        params: Validated stage params.
        battle_url: The article the citations belong to.
        html: The article's HTML.
        state: Resume state, updated in place.
        writer: The crawl_log sink.
        summary: Counters for this run.
    """
    urls = citations_mod.extract_citation_urls(html, battle_url)
    if not urls:
        return

    split = citations_mod.split_by_allow_list(urls, params.citation_allow_domains)
    summary.citations_skipped += len(split.skipped)

    records: list[dict[str, object]] = [
        {
            "citation_id": citations_mod.citation_id(url),
            "url": url,
            "battle_url": battle_url,
            "skipped": "domain_not_allowed",
        }
        for url in split.skipped
    ]

    for url in split.allowed:
        if state.is_fetched(url):
            continue
        result = await fetcher.fetch(url)
        if _record(writer, state, summary, result):
            summary.citations_fetched += 1
        records.append(
            {
                "citation_id": citations_mod.citation_id(url),
                "url": url,
                "battle_url": battle_url,
                "http_status": result.status,
                "content_hash": result.content_hash,
                "error": result.error,
                "fetched_at": result.fetched_at.isoformat(),
                "content": result.text,
            }
        )

    if records:
        citations_mod.write_citation_records(params.paths.citations, battle_url, records)


async def _crawl_battles(
    fetcher: Fetcher,
    params: CrawlParams,
    state: CrawlState,
    writer: CrawlLogWriter,
    summary: CrawlSummary,
    limit: int | None,
) -> None:
    """Fetch battle articles, their citations and their DBpedia twins.

    Args:
        fetcher: The configured fetcher.
        params: Validated stage params.
        state: Resume state, updated in place.
        writer: The crawl_log sink.
        summary: Counters for this run.
        limit: Process at most this many articles.
    """
    pending = state.pending_battles(limit)
    logger.info("battles_pending", count=len(pending), known=len(state.battle_urls))

    for index, battle_url in enumerate(pending, start=1):
        result = await wikipedia_mod.fetch_article(fetcher, battle_url)
        if not _record(writer, state, summary, result) or result.text is None:
            continue

        wikipedia_mod.write_article(params.paths.battles_html, battle_url, result.text)
        summary.battles_fetched += 1

        if params.citation_depth > 0:
            await _crawl_citations(
                fetcher, params, battle_url, result.text, state, writer, summary
            )

        if params.fetch_dbpedia:
            dbpedia_result = await dbpedia_mod.fetch_resource(fetcher, battle_url)
            if _record(writer, state, summary, dbpedia_result) and dbpedia_result.text is not None:
                summary.dbpedia_fetched += 1
                _write_json(
                    params.paths.dbpedia,
                    wikipedia_mod.article_filename(battle_url, suffix=".json"),
                    dbpedia_result.text,
                )

        if index % _STATE_SAVE_EVERY == 0:
            save_state(state, params.paths.state_file)
            writer.flush()
            logger.info("crawl_progress", done=index, of=len(pending), **summary.as_log_fields())


async def _crawl_wikidata(
    fetcher: Fetcher,
    params: CrawlParams,
    seeds: Seeds,
    state: CrawlState,
    writer: CrawlLogWriter,
    summary: CrawlSummary,
) -> None:
    """Run the seed SPARQL queries and fetch the entities they name.

    Args:
        fetcher: The configured fetcher.
        params: Validated stage params.
        seeds: Seed config.
        state: Resume state, updated in place.
        writer: The crawl_log sink.
        summary: Counters for this run.
    """
    qids: list[str] = []

    for name, query in seeds.wikidata_queries.items():
        if name in state.wikidata_done:
            logger.debug("sparql_query_already_done", query=name)
            continue

        result = await wikidata_mod.run_query(fetcher, query)
        if not _record(writer, state, summary, result) or result.text is None:
            continue

        _write_json(params.paths.wikidata, f"query-{name}.json", result.text)
        state.mark_query_done(name)
        summary.wikidata_queries += 1

        rows = wikidata_mod.parse_sparql_bindings(result.text)
        for variable in ("battle", "person"):
            qids.extend(wikidata_mod.entity_ids(rows, variable))
        logger.info("sparql_query_done", query=name, rows=len(rows))

    budget = params.max_wikidata_entities
    for qid in dict.fromkeys(qids):
        if summary.wikidata_entities >= budget:
            logger.info("wikidata_entity_budget_reached", budget=budget)
            break
        url = wikidata_mod.entity_data_url(qid)
        if state.is_fetched(url):
            continue

        result = await fetcher.fetch(url)
        if _record(writer, state, summary, result) and result.text is not None:
            summary.wikidata_entities += 1
            _write_json(params.paths.wikidata, f"{qid}.json", result.text)

        if summary.wikidata_entities % _STATE_SAVE_EVERY == 0:
            save_state(state, params.paths.state_file)


def _write_json(directory: Path, filename: str, payload: str) -> Path:
    """Write a raw JSON payload verbatim.

    Args:
        directory: Target directory, created if absent.
        filename: File name within it.
        payload: The body exactly as fetched; no reserialisation, so the raw
            record stays byte-faithful to what the source served.

    Returns:
        The path written.

    Raises:
        OSError: If the file cannot be written.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text(payload, encoding="utf-8")
    return path


async def crawl(
    params: CrawlParams,
    seeds: Seeds,
    state: CrawlState,
    writer: CrawlLogWriter,
    *,
    client: httpx.AsyncClient | None = None,
    limit: int | None = None,
) -> CrawlSummary:
    """Run the whole crawl against an HTTP client.

    Args:
        params: Validated stage params.
        seeds: Seed config.
        state: Resume state, updated in place and saved as work completes.
        writer: The crawl_log sink.
        client: An open httpx client. One is created and closed here when
            omitted; tests pass a client with a mock transport.
        limit: Process at most this many battle articles.

    Returns:
        Counters describing what the run accomplished.
    """
    summary = CrawlSummary()
    owned = client is None
    http = client or httpx.AsyncClient(
        timeout=DEFAULT_TIMEOUT_S, follow_redirects=True, headers={"User-Agent": params.user_agent}
    )

    fetcher = Fetcher(
        http,
        limiter=params.build_limiter(),
        user_agent=params.user_agent,
        max_retries=params.max_retries,
        backoff_base=params.backoff_base,
        respect_robots=params.respect_robots,
    )

    try:
        await _crawl_list_pages(fetcher, seeds, state, writer, summary)
        save_state(state, params.paths.state_file)

        await _crawl_battles(fetcher, params, state, writer, summary, limit)
        save_state(state, params.paths.state_file)

        await _crawl_wikidata(fetcher, params, seeds, state, writer, summary)
    finally:
        save_state(state, params.paths.state_file)
        writer.flush()
        if owned:
            await http.aclose()

    return summary


def _describe_plan(params: CrawlParams, seeds: Seeds, state: CrawlState, limit: int | None) -> None:
    """Log what a run would do, without doing it.

    Args:
        params: Validated stage params.
        seeds: Seed config.
        state: Resume state as loaded.
        limit: The record cap in force, if any.
    """
    logger.info(
        "crawl_dry_run",
        list_pages=len(seeds.battle_lists),
        list_pages_done=len(state.list_pages_done),
        battles_known=len(state.battle_urls),
        battles_pending=len(state.pending_battles(limit)),
        sparql_queries=len(seeds.wikidata_queries),
        rate_limit_wikipedia=params.rate_limit_wikipedia,
        rate_limit_other=params.rate_limit_other,
        respect_robots=params.respect_robots,
        output_root=str(params.paths.battles_html.parent),
    )


def run(spec: dict[str, Any], context: StageContext | None = None) -> None:
    """Execute the crawl stage.

    Args:
        spec: The loaded agent spec, with overrides applied.
        context: Shared services. When it carries a database connection that
            connection is used for ``crawl_log``; otherwise one is opened
            from DATABASE_URL.

    Raises:
        CrawlConfigError: If the spec omits or misstates a param.
        SeedConfigError: If the seed config is missing or malformed.
        DatabaseConfigError: If no database is configured for a live run.
    """
    ctx = context or StageContext()
    params = CrawlParams.from_spec(spec)
    seeds = load_seeds(params.seed_path)
    state = load_state(params.paths.state_file)

    if ctx.dry_run:
        _describe_plan(params, seeds, state, ctx.limit)
        return

    params.paths.ensure()

    if ctx.db_conn is not None:
        writer: CrawlLogWriter = SqlCrawlLog(ctx.db_conn)
        summary = asyncio.run(crawl(params, seeds, state, writer, limit=ctx.limit))
    else:
        with get_connection() as conn:
            summary = asyncio.run(crawl(params, seeds, state, SqlCrawlLog(conn), limit=ctx.limit))

    logger.info("crawl_complete", **summary.as_log_fields())
    if summary.failed_urls:
        logger.warning("crawl_failures", count=summary.failures, sample=summary.failed_urls[:10])
