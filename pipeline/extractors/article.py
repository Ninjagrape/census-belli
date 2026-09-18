"""
Article body cleaning and the LLM extraction batcher.

The infobox gives the skeleton; the body gives everything the infobox had no
row for -- which commander actually directed the fighting, whether a troop
figure is what was on the field or what was on the establishment, what the
ground was like. That has to go through a model, so this module's job is to
hand the model passages it can actually read and to turn what comes back
into the same records the deterministic extractors produce.

Long articles are chunked on section boundaries. Splitting mid-paragraph
would hand the model a number whose qualifying clause is in the next chunk,
which is exactly how a theatre-strength figure gets recorded as engaged.

The prompt is never written here. It is read from ``agents/extract.yaml`` at
run time and passed in, so the spec stays the single source of truth for
what the model is asked and the request hash stays stable across runs.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, TypeVar

import structlog
from bs4 import BeautifulSoup

from pipeline.extractors.records import (
    BattleFacts,
    CasualtyReport,
    CommanderMention,
    ExtractionFailure,
    Provenance,
    SideExtraction,
    SourceExtraction,
    TroopReport,
    coerce_branch,
    coerce_casualty_type,
)
from pipeline.extractors.textnorm import resolve_scope, strip_markup
from pipeline.llm import LLMService

__all__ = [
    "DEFAULT_MAX_PASSAGE_CHARS",
    "ArticleExtraction",
    "Passage",
    "build_passages",
    "clean_article_text",
    "extract_from_passages",
    "iter_batches",
    "payload_to_extraction",
    "render_template",
    "split_sections",
]

logger = structlog.get_logger()

# Roughly 3k tokens of English prose. Well inside every model's context, and
# small enough that a passage's qualifying clauses stay with its numbers.
DEFAULT_MAX_PASSAGE_CHARS: Final[int] = 12_000

# Sections that never carry battle facts. Dropping them cuts token spend
# without touching anything the extraction needs.
_SKIP_SECTIONS: Final[frozenset[str]] = frozenset(
    {
        "references",
        "notes",
        "citations",
        "further reading",
        "external links",
        "see also",
        "bibliography",
        "sources",
        "footnotes",
    }
)

_HTML_HINT_RE: Final = re.compile(r"<(?:html|body|div|p|table)\b", re.IGNORECASE)
_HEADING_RE: Final = re.compile(r"^\s*(={2,6})\s*(.+?)\s*\1\s*$", re.MULTILINE)
_HEADING_MARK: Final = "␟"
_HEADING_MARK_RE: Final = re.compile(f"{_HEADING_MARK}(.+?){_HEADING_MARK}")
_PLACEHOLDER_RE: Final = re.compile(r"\{(\w+)\}")
_SENTENCE_RE: Final = re.compile(r"(?<=[.!?])\s+")

T = TypeVar("T")


@dataclass(frozen=True)
class Passage:
    """One chunk of article text, sized for a single LLM call."""

    battle_name: str
    text: str
    section: str = "lead"
    index: int = 0
    total: int = 1
    source_type: str = "wikipedia_body"
    source_title: str = ""


@dataclass
class ArticleExtraction:
    """What the LLM pass produced for one article."""

    extractions: list[SourceExtraction] = field(default_factory=list)
    failures: list[ExtractionFailure] = field(default_factory=list)


# ─── Cleaning ────────────────────────────────────────────────────────────────


def _clean_html(raw: str) -> str:
    """Reduce rendered article HTML to headed plain text.

    Args:
        raw: The HTML of a Wikipedia article.

    Returns:
        Plain text with ``== Heading ==`` markers kept so sections survive.
    """
    soup = BeautifulSoup(raw, "html.parser")

    for tag in soup.find_all(["script", "style", "sup", "table", "figure", "figcaption"]):
        tag.decompose()

    blocks: list[str] = []
    for node in soup.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = node.get_text(" ", strip=True)
        if not text:
            continue
        if node.name in ("h1", "h2", "h3", "h4"):
            blocks.append(f"== {text} ==")
        else:
            blocks.append(text)

    if not blocks:
        blocks = [soup.get_text(" ", strip=True)]

    return "\n\n".join(blocks).strip()


def clean_article_text(raw: str, *, is_html: bool | None = None) -> str:
    """Strip markup from an article while keeping its section structure.

    Args:
        raw: Article source, either MediaWiki wikitext or rendered HTML.
        is_html: Force the input format. When None the format is sniffed.

    Returns:
        Plain text with headings preserved as ``== Heading ==`` lines.
    """
    if not raw.strip():
        return ""

    html_input = _HTML_HINT_RE.search(raw[:4000]) is not None if is_html is None else is_html
    if html_input:
        return _clean_html(raw)

    # Protect headings behind a sentinel before strip_markup collapses the
    # '=' runs, then restore them in the same shape _clean_html emits.
    marked = _HEADING_RE.sub(lambda m: f"\n{_HEADING_MARK}{m.group(2)}{_HEADING_MARK}\n", raw)
    text = strip_markup(marked)
    return _HEADING_MARK_RE.sub(r"== \1 ==", text).strip()


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split cleaned article text into (section title, body) pairs.

    Args:
        text: Cleaned article text with ``== Heading ==`` markers.

    Returns:
        Sections in document order, with reference and navigation sections
        removed. Text before the first heading is returned as ``"lead"``.
    """
    pattern = re.compile(r"^\s*==\s*(.+?)\s*==\s*$", re.MULTILINE)

    sections: list[tuple[str, str]] = []
    cursor = 0
    title = "lead"

    for match in pattern.finditer(text):
        body = text[cursor : match.start()].strip()
        if body:
            sections.append((title, body))
        title = match.group(1).strip()
        cursor = match.end()

    tail = text[cursor:].strip()
    if tail:
        sections.append((title, tail))

    return [(t, b) for t, b in sections if t.strip().lower() not in _SKIP_SECTIONS]


def _pack(body: str, max_chars: int) -> list[str]:
    """Pack a section body into chunks no larger than the limit.

    Args:
        body: One section's text.
        max_chars: Maximum characters per chunk.

    Returns:
        Chunks split on paragraph boundaries, falling back to sentence
        boundaries only for a single paragraph that exceeds the limit.
    """
    chunks: list[str] = []
    current = ""

    for paragraph in re.split(r"\n{2,}", body):
        piece = paragraph.strip()
        if not piece:
            continue

        if len(piece) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            sentence_buffer = ""
            for sentence in _SENTENCE_RE.split(piece):
                if len(sentence_buffer) + len(sentence) + 1 > max_chars and sentence_buffer:
                    chunks.append(sentence_buffer.strip())
                    sentence_buffer = ""
                sentence_buffer = f"{sentence_buffer} {sentence}".strip()
            if sentence_buffer:
                chunks.append(sentence_buffer.strip())
            continue

        if len(current) + len(piece) + 2 > max_chars and current:
            chunks.append(current)
            current = piece
        else:
            current = f"{current}\n\n{piece}".strip()

    if current:
        chunks.append(current)

    return chunks


def build_passages(
    battle_name: str,
    raw: str,
    *,
    source_type: str = "wikipedia_body",
    source_title: str = "",
    max_chars: int = DEFAULT_MAX_PASSAGE_CHARS,
    is_html: bool | None = None,
) -> list[Passage]:
    """Turn one raw article into the passages the LLM will be asked about.

    Args:
        battle_name: The battle the article is about.
        raw: Article source, wikitext or HTML.
        source_type: A ``source_type`` enum value for the resulting reports.
        source_title: Human-readable title of the document.
        max_chars: Maximum characters per passage.
        is_html: Force the input format; None sniffs it.

    Returns:
        Passages in document order. An article with no usable prose yields
        an empty list.
    """
    text = clean_article_text(raw, is_html=is_html)
    if not text:
        return []

    pending: list[tuple[str, str]] = []
    for section, body in split_sections(text):
        pending.extend((section, chunk) for chunk in _pack(body, max_chars))

    total = len(pending)
    return [
        Passage(
            battle_name=battle_name,
            text=chunk,
            section=section,
            index=i,
            total=total,
            source_type=source_type,
            source_title=source_title or battle_name,
        )
        for i, (section, chunk) in enumerate(pending)
    ]


def iter_batches(items: Sequence[T], batch_size: int) -> Iterator[list[T]]:
    """Yield successive batches of a sequence.

    Args:
        items: The items to batch.
        batch_size: Items per batch; values below 1 are treated as 1.

    Yields:
        Lists of at most ``batch_size`` items.
    """
    size = max(1, batch_size)
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


# ─── Prompting ───────────────────────────────────────────────────────────────


def render_template(template: str, values: dict[str, str]) -> str:
    """Fill ``{placeholder}`` slots in a spec's user template.

    Substitution is done by pattern rather than ``str.format`` so that a
    brace in article text cannot raise, and an unrecognised placeholder in
    the spec is logged and left visible rather than aborting the batch.

    Args:
        template: The template from ``agents/extract.yaml``.
        values: Replacements by placeholder name.

    Returns:
        The rendered prompt.
    """

    def substitute(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            logger.warning("prompt_template_placeholder_unknown", placeholder=key)
            return match.group(0)
        return values[key]

    return _PLACEHOLDER_RE.sub(substitute, template)


def _passage_prompt(template: str, passage: Passage) -> str:
    """Render the user prompt for one passage.

    Args:
        template: The spec's ``prompt.user_template``.
        passage: The passage to extract from.

    Returns:
        The rendered prompt.
    """
    return render_template(
        template,
        {
            "battle_name": passage.battle_name,
            "source_type": passage.source_type,
            "source_title": passage.source_title,
            "text_passage": passage.text,
        },
    )


# ─── LLM payload to records ──────────────────────────────────────────────────


def _troop_reports(
    payload: dict[str, Any], provenance: Provenance, section: str
) -> list[TroopReport]:
    """Convert a payload's ``troop_reports`` array into records.

    Args:
        payload: One LLM response object.
        provenance: Where the passage came from.
        section: The article section the passage was taken from.

    Returns:
        Troop reports, with scope settled against the quoted context.
    """
    reports: list[TroopReport] = []

    for item in payload.get("troop_reports") or []:
        if not isinstance(item, dict):
            continue
        raw_value = item.get("value")
        if not isinstance(raw_value, int | float):
            continue
        quote = str(item.get("context_quote") or "")
        reports.append(
            TroopReport(
                side_label=str(item.get("side") or "").strip(),
                reported_value=float(raw_value),
                provenance=provenance,
                branch=coerce_branch(item.get("branch")),
                # The model is asked to judge scope and usually does, but the
                # sentence it quoted is the evidence. Where the two conflict
                # the wording wins; see textnorm.resolve_scope.
                scope=resolve_scope(item.get("scope"), quote),
                is_estimate=bool(item.get("is_estimate")),
                is_upper_bound=bool(item.get("is_upper_bound")),
                is_lower_bound=bool(item.get("is_lower_bound")),
                extracted_context=quote,
                page_or_section=section,
            )
        )

    return reports


def _casualty_reports(payload: dict[str, Any], provenance: Provenance) -> list[CasualtyReport]:
    """Convert a payload's ``casualty_reports`` array into records.

    Args:
        payload: One LLM response object.
        provenance: Where the passage came from.

    Returns:
        Casualty reports.
    """
    reports: list[CasualtyReport] = []

    for item in payload.get("casualty_reports") or []:
        if not isinstance(item, dict):
            continue
        raw_value = item.get("value")
        if not isinstance(raw_value, int | float):
            continue
        reports.append(
            CasualtyReport(
                side_label=str(item.get("side") or "").strip(),
                reported_value=float(raw_value),
                provenance=provenance,
                casualty_type=coerce_casualty_type(item.get("casualty_type")),
                is_estimate=bool(item.get("is_estimate")),
            )
        )

    return reports


def _commanders(payload: dict[str, Any], provenance: Provenance) -> list[CommanderMention]:
    """Convert a payload's ``commanders`` array into records.

    Args:
        payload: One LLM response object.
        provenance: Where the passage came from.

    Returns:
        Commander mentions, pre entity resolution.
    """
    mentions: list[CommanderMention] = []

    for item in payload.get("commanders") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        mentions.append(
            CommanderMention(
                side_label=str(item.get("side") or "").strip(),
                name=name,
                provenance=provenance,
                apparent_role=str(item.get("apparent_role") or "unclear"),
                role_evidence=str(item.get("role_evidence") or ""),
            )
        )

    return mentions


def payload_to_extraction(
    payload: dict[str, Any],
    passage: Passage,
    provenance: Provenance,
) -> SourceExtraction:
    """Turn one validated LLM response into a source extraction.

    Args:
        payload: The response object, already schema-validated by the LLM layer.
        passage: The passage it was produced from.
        provenance: Where the passage came from.

    Returns:
        The same record shape the deterministic extractors produce, so the
        merger treats every source identically.
    """
    outcome = payload.get("outcome")
    outcome_map = outcome if isinstance(outcome, dict) else {}

    terrain_raw = payload.get("terrain")
    terrain = [str(t).strip() for t in terrain_raw if str(t).strip()] \
        if isinstance(terrain_raw, list) else []

    facts = BattleFacts(
        name=str(payload.get("battle_name") or passage.battle_name) or None,
        terrain=terrain,
        fortified=payload.get("fortified") if isinstance(payload.get("fortified"), bool) else None,
        weather=str(payload["weather"]) if payload.get("weather") else None,
        victor=str(outcome_map["victor"]) if outcome_map.get("victor") else None,
        outcome_level=str(outcome_map["outcome_level"])
        if outcome_map.get("outcome_level")
        else None,
        outcome_evidence=str(outcome_map["evidence"]) if outcome_map.get("evidence") else None,
    )

    by_side: dict[str, SideExtraction] = {}

    def side_for(label: str) -> SideExtraction:
        key = label.strip().lower()
        if key not in by_side:
            by_side[key] = SideExtraction(label=label.strip() or "unattributed")
        return by_side[key]

    for mention in _commanders(payload, provenance):
        side_for(mention.side_label).commanders.append(mention)
    for report in _troop_reports(payload, provenance, passage.section):
        side_for(report.side_label).troop_reports.append(report)
    for casualty in _casualty_reports(payload, provenance):
        side_for(casualty.side_label).casualty_reports.append(casualty)

    return SourceExtraction(
        provenance=provenance,
        facts=facts,
        sides=list(by_side.values()),
    )


# ─── The batched LLM pass ────────────────────────────────────────────────────


def extract_from_passages(
    service: LLMService,
    passages: Iterable[Passage],
    *,
    system_prompt: str,
    user_template: str,
    json_schema: dict[str, Any],
    source_ref: str = "",
    battle_id: int | None = None,
) -> ArticleExtraction:
    """Run the LLM extraction over an article's passages.

    A passage that fails is recorded and the rest of the article continues,
    per the project's extraction-failure rule. The service handles retries,
    schema validation, the audit row and the resume cache, so nothing here
    re-implements any of that.

    Args:
        service: The stage's LLM service.
        passages: Passages to extract from.
        system_prompt: ``prompt.system`` from the spec, unmodified.
        user_template: ``prompt.user_template`` from the spec.
        json_schema: ``prompt.output_schema`` from the spec.
        source_ref: URL or path identifying the document.
        battle_id: Database id, recorded on the audit row when known.

    Returns:
        The extractions and the failures, both possibly empty.
    """
    result = ArticleExtraction()

    for passage in passages:
        provenance = Provenance(
            source_type=passage.source_type,
            extraction_method="llm_extraction",
            source_ref=source_ref,
            source_title=passage.source_title,
        )

        metadata: dict[str, Any] = {"passage": f"{passage.index + 1}/{passage.total}"}
        if battle_id is not None:
            metadata["battle_id"] = battle_id

        response = service.complete(
            system=system_prompt,
            user=_passage_prompt(user_template, passage),
            json_schema=json_schema,
            metadata=metadata,
        )

        if not response.ok or response.data is None:
            result.failures.append(
                ExtractionFailure(
                    battle_name=passage.battle_name,
                    source_ref=source_ref or passage.source_title,
                    status=response.status.value,
                    error=response.error or "no data returned",
                    passage_index=passage.index,
                )
            )
            continue

        result.extractions.append(payload_to_extraction(response.data, passage, provenance))

    if result.failures:
        logger.warning(
            "article_passages_flagged_for_review",
            source=source_ref,
            failed=len(result.failures),
            succeeded=len(result.extractions),
        )

    return result
