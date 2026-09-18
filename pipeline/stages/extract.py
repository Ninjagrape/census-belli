"""
The extract stage: raw crawl output to structured battle records.

What the stage does per battle is: parse the infobox deterministically, map
the Wikidata entity, run the LLM over the article body and any fetched
citations, then merge the four into one record. Structured sources win
scalar fields; every troop number survives as its own report.

The prompt, the output schema, the provider, the model and the batch size
all come from ``agents/extract.yaml`` at run time. Nothing about the
extraction is hardcoded here, so changing what the model is asked is a spec
edit, and the request hash the LLM layer caches on moves with it.

Failure policy follows the project rule. A configuration problem -- an
unknown provider, a missing API key -- raises, because every article in the
batch would fail the same way. An extraction problem flags the battle for
review and the batch continues.

Inputs, laid down by the crawl stage:

- ``data/raw/battles_html/<Title>-<hash>.html`` -- one rendered article per
  battle, named by :func:`pipeline.crawlers.wikipedia.article_filename`. A
  ``.wikitext`` sibling is preferred when one exists, because the template
  names the infobox variant and the campaignboxes are readable.
- ``data/raw/citations/<Title>-<hash>.jsonl`` -- the same stem, one JSON
  object per fetched citation.
- ``data/raw/wikidata/<QID>.json`` -- keyed by entity, not by article, so
  entities are joined back to their battle through their ``enwiki``
  sitelink.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import structlog

from pipeline.crawlers.wikipedia import article_filename
from pipeline.extractors import (
    ArticleExtraction,
    BattleExtraction,
    ExtractionFailure,
    ExtractWriteCounts,
    SourceExtraction,
    build_passages,
    extract_from_passages,
    iter_batches,
    map_entity,
    merge_battle,
    parse_article_infobox,
    to_jsonable,
    write_battle,
)
from pipeline.extractors.article import DEFAULT_MAX_PASSAGE_CHARS
from pipeline.llm import LLMService
from pipeline.stages.base import StageContext

__all__ = ["RawBattle", "discover_battles", "extract_battle", "run"]

logger = structlog.get_logger()

RAW_ROOT: Final[Path] = Path("data/raw")
PROCESSED_ROOT: Final[Path] = Path("data/processed")

_ARTICLE_DIR: Final[str] = "battles_html"
_WIKIDATA_DIR: Final[str] = "wikidata"
_CITATIONS_DIR: Final[str] = "citations"

# Preferred first: mwparserfromhell reads templates, so a wikitext sibling
# gives a better infobox than the rendered HTML the crawl stage stores.
_ARTICLE_SUFFIXES: Final[tuple[str, ...]] = (".wikitext", ".wiki", ".txt", ".html", ".htm")
_HTML_SUFFIXES: Final[tuple[str, ...]] = (".html", ".htm")

# article_filename appends a 10-character digest of the URL to a sanitised
# title, because sanitising can map two articles onto one name. Stripping it
# back off recovers a readable battle name.
_URL_DIGEST_RE: Final = re.compile(r"-[0-9a-f]{10}$")

_TITLE_TAG_RE: Final = re.compile(
    r"<title[^>]*>(?P<title>.*?)</title>", re.IGNORECASE | re.DOTALL
)


@dataclass
class RawBattle:
    """The raw files the crawl stage left for one battle."""

    slug: str
    article_path: Path | None = None
    wikidata_path: Path | None = None
    citation_paths: list[Path] = field(default_factory=list)
    title: str = ""

    @property
    def name(self) -> str:
        """A human-readable battle name.

        Returns:
            The title read from the article when there was one, otherwise the
            slug with the crawl stage's URL digest and underscores removed.
        """
        if self.title:
            return self.title
        return _URL_DIGEST_RE.sub("", self.slug).replace("_", " ").strip()

    @property
    def article_is_html(self) -> bool:
        """Whether the stored article is rendered HTML rather than wikitext."""
        return (
            self.article_path is not None
            and self.article_path.suffix.lower() in _HTML_SUFFIXES
        )


def _article_title(raw: str) -> str:
    """Read a battle name out of a stored article.

    Args:
        raw: The article's stored text.

    Returns:
        The page title with Wikipedia's suffix removed, or an empty string.
    """
    match = _TITLE_TAG_RE.search(raw[:8000])
    if match is None:
        return ""
    title = re.sub(r"\s+", " ", match.group("title")).strip()
    return re.sub(r"\s*[-–—]\s*Wikipedia\s*$", "", title).strip()


def _wikidata_index(wikidata_dir: Path) -> dict[str, Path]:
    """Map each Wikidata entity file onto the article slug it belongs to.

    Wikidata files are keyed by Q-id, not by article, so the join runs
    through the entity's ``enwiki`` sitelink and the crawl stage's own
    filename rule. An entity with no English sitelink has no battle to
    attach to and is skipped.

    Args:
        wikidata_dir: The ``data/raw/wikidata`` directory.

    Returns:
        Article slug to entity file.
    """
    index: dict[str, Path] = {}

    for path in sorted(wikidata_dir.glob("*.json")):
        payload = _read_json(path)
        if payload is None:
            continue
        for url in _enwiki_urls(payload):
            index[Path(article_filename(url)).stem] = path

    return index


def _enwiki_urls(payload: dict[str, Any]) -> list[str]:
    """List the English Wikipedia URLs a Wikidata payload points at.

    Args:
        payload: A ``wbgetentities`` response or a bare entity.

    Returns:
        Article URLs, one per entity carrying an ``enwiki`` sitelink.
    """
    entities = payload.get("entities")
    candidates = (
        list(entities.values()) if isinstance(entities, dict) and entities else [payload]
    )

    urls: list[str] = []
    for entity in candidates:
        if not isinstance(entity, dict):
            continue
        sitelinks = entity.get("sitelinks")
        if not isinstance(sitelinks, dict):
            continue
        entry = sitelinks.get("enwiki")
        if not isinstance(entry, dict):
            continue
        if entry.get("url"):
            urls.append(str(entry["url"]))
        elif entry.get("title"):
            title = str(entry["title"]).replace(" ", "_")
            urls.append(f"https://en.wikipedia.org/wiki/{title}")

    return urls


def discover_battles(raw_root: Path = RAW_ROOT) -> list[RawBattle]:
    """Find every battle with raw data on disk.

    Args:
        raw_root: The ``data/raw`` directory the crawl stage writes to.

    Returns:
        One record per battle, sorted by slug, each pointing at whichever of
        the article, Wikidata and citation files exist. A battle with only
        some of them is still returned: the extractors handle a missing
        source and the merger logs it.
    """
    battles: dict[str, RawBattle] = {}

    article_dir = raw_root / _ARTICLE_DIR
    if article_dir.is_dir():
        for path in sorted(article_dir.iterdir()):
            if path.suffix.lower() not in _ARTICLE_SUFFIXES or not path.is_file():
                continue
            entry = battles.setdefault(path.stem, RawBattle(slug=path.stem))
            current = entry.article_path
            if current is None or _ARTICLE_SUFFIXES.index(
                path.suffix.lower()
            ) < _ARTICLE_SUFFIXES.index(current.suffix.lower()):
                entry.article_path = path

    citations_dir = raw_root / _CITATIONS_DIR
    if citations_dir.is_dir():
        for path in sorted(citations_dir.glob("*.jsonl")):
            battles.setdefault(path.stem, RawBattle(slug=path.stem)).citation_paths.append(path)

    wikidata_dir = raw_root / _WIKIDATA_DIR
    if wikidata_dir.is_dir():
        matched = 0
        for slug, path in _wikidata_index(wikidata_dir).items():
            battles.setdefault(slug, RawBattle(slug=slug)).wikidata_path = path
            matched += 1
        logger.info("wikidata_entities_joined_to_articles", joined=matched)

    for battle in battles.values():
        if battle.article_path is None:
            continue
        head = _read_text(battle.article_path)
        if head:
            battle.title = _article_title(head)

    return [battles[slug] for slug in sorted(battles)]


def _read_text(path: Path) -> str | None:
    """Read a file, treating an unreadable one as absent.

    Args:
        path: The file to read.

    Returns:
        Its contents, or None when it could not be read.
    """
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("raw_file_unreadable", path=str(path), error=str(exc))
        return None


def _read_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON file, treating a malformed one as absent.

    Args:
        path: The file to read.

    Returns:
        The decoded object, or None when it could not be read or parsed.
    """
    raw = _read_text(path)
    if raw is None:
        return None
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("raw_json_malformed", path=str(path), error=str(exc))
        return None
    return decoded if isinstance(decoded, dict) else None


def _citation_records(path: Path) -> list[dict[str, Any]]:
    """Read one battle's fetched citations from a JSON lines file.

    Args:
        path: The ``.jsonl`` file.

    Returns:
        One mapping per readable line; malformed lines are logged and skipped.
    """
    raw = _read_text(path)
    if raw is None:
        return []

    records: list[dict[str, Any]] = []
    for number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("citation_line_malformed", path=str(path), line=number)
            continue
        if isinstance(decoded, dict):
            records.append(decoded)
    return records


def _prompt_parts(spec: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    """Read the prompt and schema from the agent spec.

    Args:
        spec: The loaded ``agents/extract.yaml``.

    Returns:
        The system prompt, the user template and the output schema, exactly
        as the spec defines them.

    Raises:
        ValueError: If the spec defines no prompt. Running the LLM pass with
            an improvised prompt would produce data that does not match what
            the spec says was asked, which is worse than not running.
    """
    prompt = spec.get("prompt") or {}
    system = prompt.get("system")
    template = prompt.get("user_template")
    schema = prompt.get("output_schema")

    if not system or not template or not isinstance(schema, dict):
        raise ValueError(
            "agents/extract.yaml must define prompt.system, prompt.user_template and "
            "prompt.output_schema; the stage loads them at run time rather than "
            "carrying a copy."
        )

    return str(system), str(template), schema


def _llm_sources(
    battle: RawBattle,
    service: LLMService | None,
    *,
    system: str,
    template: str,
    schema: dict[str, Any],
    max_passage_chars: int,
) -> ArticleExtraction:
    """Run the LLM pass over a battle's body text and citations.

    Args:
        battle: The battle's raw files.
        service: The stage's LLM service, or None to skip the pass.
        system: ``prompt.system`` from the spec.
        template: ``prompt.user_template`` from the spec.
        schema: ``prompt.output_schema`` from the spec.
        max_passage_chars: Chunk size for long articles.

    Returns:
        The extractions and the failures from every passage.
    """
    combined = ArticleExtraction()
    if service is None:
        return combined

    if battle.article_path is not None:
        raw = _read_text(battle.article_path)
        if raw:
            passages = build_passages(
                battle.name,
                raw,
                source_title=battle.name,
                max_chars=max_passage_chars,
                is_html=battle.article_is_html,
            )
            result = extract_from_passages(
                service,
                passages,
                system_prompt=system,
                user_template=template,
                json_schema=schema,
                source_ref=str(battle.article_path),
            )
            combined.extractions.extend(result.extractions)
            combined.failures.extend(result.failures)

    for path in battle.citation_paths:
        for record in _citation_records(path):
            # The crawl stage records URLs it deliberately did not fetch, so
            # that the trail is complete; those rows carry no content.
            if record.get("skipped"):
                continue
            body = str(record.get("content") or record.get("text") or "")
            if not body.strip():
                continue
            passages = build_passages(
                battle.name,
                body,
                source_type="web_secondary",
                source_title=str(record.get("title") or record.get("url") or path.name),
                max_chars=max_passage_chars,
            )
            result = extract_from_passages(
                service,
                passages,
                system_prompt=system,
                user_template=template,
                json_schema=schema,
                source_ref=str(record.get("url") or path),
            )
            combined.extractions.extend(result.extractions)
            combined.failures.extend(result.failures)

    return combined


def extract_battle(
    battle: RawBattle,
    service: LLMService | None,
    *,
    system: str,
    template: str,
    schema: dict[str, Any],
    max_passage_chars: int = DEFAULT_MAX_PASSAGE_CHARS,
) -> BattleExtraction:
    """Extract and merge every source for one battle.

    Args:
        battle: The battle's raw files.
        service: The stage's LLM service, or None to run the deterministic
            extractors only.
        system: ``prompt.system`` from the spec.
        template: ``prompt.user_template`` from the spec.
        schema: ``prompt.output_schema`` from the spec.
        max_passage_chars: Chunk size for long articles.

    Returns:
        The merged record, carrying any extraction failures so that the
        battle is flagged for review rather than silently thinned.
    """
    sources: list[SourceExtraction] = []
    failures: list[ExtractionFailure] = []

    if battle.article_path is not None:
        raw = _read_text(battle.article_path)
        if raw:
            infobox = parse_article_infobox(
                raw,
                is_html=battle.article_is_html,
                source_ref=str(battle.article_path),
                source_title=battle.name,
            )
            if infobox is not None:
                sources.append(infobox)

    if battle.wikidata_path is not None:
        payload = _read_json(battle.wikidata_path)
        if payload is None:
            failures.append(
                ExtractionFailure(
                    battle_name=battle.name,
                    source_ref=str(battle.wikidata_path),
                    status="parse_error",
                    error="wikidata payload unreadable or not a JSON object",
                )
            )
        else:
            mapped = map_entity(
                payload,
                source_ref=str(battle.wikidata_path),
                fallback_name=battle.name,
            )
            if mapped is not None:
                sources.append(mapped)

    llm_result = _llm_sources(
        battle,
        service,
        system=system,
        template=template,
        schema=schema,
        max_passage_chars=max_passage_chars,
    )
    sources.extend(llm_result.extractions)
    failures.extend(llm_result.failures)

    return merge_battle(battle.slug, sources, failures=failures, fallback_name=battle.name)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to a JSON lines file, creating the directory if needed.

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


def _write_outputs(battles: list[BattleExtraction], processed_root: Path) -> None:
    """Write the stage's four processed files.

    Args:
        battles: Every merged battle from this run.
        processed_root: The ``data/processed`` directory.
    """
    battle_rows: list[dict[str, Any]] = []
    commander_rows: list[dict[str, Any]] = []
    troop_rows: list[dict[str, Any]] = []
    casualty_rows: list[dict[str, Any]] = []

    for battle in battles:
        battle_rows.append(
            {
                "slug": battle.slug,
                "name": battle.name,
                **to_jsonable(battle.facts),
                "sides": [
                    {"label": s.label, "aliases": s.aliases, "polity": s.polity,
                     "outcome": s.outcome}
                    for s in battle.sides
                ],
                "disagreements": [to_jsonable(d) for d in battle.disagreements],
                "missing": [to_jsonable(m) for m in battle.missing],
                "failures": [to_jsonable(f) for f in battle.failures],
                "notes": battle.notes,
                "n_sources": battle.n_sources,
                "needs_review": battle.needs_review,
                "review_notes": battle.review_notes,
            }
        )
        for side in battle.sides:
            for commander in side.commanders:
                commander_rows.append(
                    {"battle_slug": battle.slug, "battle_name": battle.name,
                     **to_jsonable(commander)}
                )
            for report in side.troop_reports:
                troop_rows.append(
                    {"battle_slug": battle.slug, **to_jsonable(report)}
                )
            for casualty in side.casualty_reports:
                casualty_rows.append(
                    {"battle_slug": battle.slug, **to_jsonable(casualty)}
                )

    _write_jsonl(processed_root / "battles.jsonl", battle_rows)
    _write_jsonl(processed_root / "commanders_raw.jsonl", commander_rows)
    _write_jsonl(processed_root / "troop_reports.jsonl", troop_rows)
    _write_jsonl(processed_root / "casualty_reports.jsonl", casualty_rows)


def _build_service(spec: dict[str, Any], context: StageContext) -> LLMService | None:
    """Construct the stage's LLM service, unless the run does not need one.

    Args:
        spec: The loaded agent spec.
        context: The stage context.

    Returns:
        A configured service, or None for a dry run.

    Raises:
        LLMConfigError: If the spec's provider or credentials are unusable.
            This halts the stage deliberately: every article would fail the
            same way, and a stage that silently downgraded to infobox-only
            extraction would look like it succeeded.
    """
    if context.dry_run:
        logger.info("extract_dry_run_skipping_llm_pass")
        return None
    return LLMService.from_spec(spec, db_conn=context.db_conn)


def run(spec: dict[str, Any], context: StageContext | None = None) -> None:
    """Run the extract stage.

    Args:
        spec: The loaded ``agents/extract.yaml``, with overrides applied.
        context: Shared services. Without one the stage runs against the
            default paths, skips the database writes and still produces the
            processed files.
    """
    ctx = context or StageContext()
    params = spec.get("params") or {}

    raw_root = Path(str(params.get("raw_root", RAW_ROOT)))
    processed_root = Path(str(params.get("processed_root", PROCESSED_ROOT)))
    batch_size = int(params.get("batch_size", 50))
    max_passage_chars = int(params.get("max_passage_chars", DEFAULT_MAX_PASSAGE_CHARS))

    system, template, schema = _prompt_parts(spec)

    discovered = discover_battles(raw_root)
    if ctx.limit is not None:
        discovered = discovered[: ctx.limit]

    if not discovered:
        logger.warning(
            "extract_found_no_raw_input",
            raw_root=str(raw_root),
            hint="run the crawl stage first",
        )
        return

    service = _build_service(spec, ctx)
    merged: list[BattleExtraction] = []

    for batch_number, batch in enumerate(iter_batches(discovered, batch_size), start=1):
        logger.info(
            "extract_batch_start",
            batch=batch_number,
            battles=len(batch),
            batch_size=batch_size,
        )
        for battle in batch:
            try:
                merged.append(
                    extract_battle(
                        battle,
                        service,
                        system=system,
                        template=template,
                        schema=schema,
                        max_passage_chars=max_passage_chars,
                    )
                )
            except Exception as exc:
                # One article must not take the batch down with it. The
                # battle is skipped, loudly, and the run continues.
                logger.error(
                    "extract_battle_failed",
                    slug=battle.slug,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )

    _write_outputs(merged, processed_root)

    if ctx.db_conn is not None and not ctx.dry_run:
        counts = ExtractWriteCounts()
        cache: dict[tuple[str, str], int] = {}
        for record in merged:
            try:
                write_battle(ctx.db_conn, record, counts, cache)
            except Exception as exc:
                # A row that will not go in is a data problem, not a reason
                # to lose the rest of the run's writes.
                logger.error("extract_write_failed", slug=record.slug, error=str(exc))
        logger.info("extract_rows_written", **counts.as_dict())
    else:
        logger.info("extract_database_write_skipped", dry_run=ctx.dry_run)

    if service is not None:
        service.log_summary()

    flagged = [b for b in merged if b.needs_review]
    logger.info(
        "extract_complete",
        battles=len(merged),
        flagged_for_review=len(flagged),
        troop_reports=sum(len(s.troop_reports) for b in merged for s in b.sides),
        missing_fields=sum(len(b.missing) for b in merged),
    )
