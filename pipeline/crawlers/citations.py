"""
Citation extraction and fetching from battle articles.

A Wikipedia infobox's troop numbers are only as good as what they cite, and
the reconciliation stage needs the underlying reports, not Wikipedia's
summary of them. This module pulls the external links out of an article's
reference apparatus and fetches the ones on an allowed domain.

The allow list is a hard boundary. A battle article's references point at
hundreds of hosts, most of them irrelevant, some hostile, and fetching them
all would turn a research crawler into an open web crawler. Everything else
is recorded as skipped rather than silently dropped, so the missing-data
trail stays complete.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import structlog
from bs4 import BeautifulSoup, Tag

from pipeline.crawlers.fetcher import host_of
from pipeline.crawlers.wikipedia import article_filename

__all__ = [
    "CitationSplit",
    "citation_filename",
    "citation_id",
    "domain_allowed",
    "extract_citation_urls",
    "split_by_allow_list",
    "write_citation_records",
]

logger = structlog.get_logger()

# Where MediaWiki keeps an article's reference apparatus.
_REFERENCE_SELECTORS = (
    "ol.references",
    "div.reflist",
    "div.refbegin",
    "span.reference-text",
    "cite",
    "div.citation",
)


@dataclass(frozen=True)
class CitationSplit:
    """Citation URLs partitioned by the allow list.

    Attributes:
        allowed: URLs on an allowed domain, to be fetched.
        skipped: URLs on any other domain, recorded but not fetched.
    """

    allowed: list[str]
    skipped: list[str]


def extract_citation_urls(html: str, base_url: str | None = None) -> list[str]:
    """Extract external citation URLs from an article's references.

    Args:
        html: The article HTML.
        base_url: The article's URL, used only for logging context.

    Returns:
        Absolute http(s) URLs in document order, deduplicated.
    """
    soup = BeautifulSoup(html, "html.parser")

    containers: list[Tag] = []
    for selector in _REFERENCE_SELECTORS:
        containers.extend(tag for tag in soup.select(selector) if isinstance(tag, Tag))

    seen: set[str] = set()
    urls: list[str] = []
    for container in containers:
        for anchor in container.find_all("a", href=True):
            if not isinstance(anchor, Tag):
                continue
            href = anchor.get("href")
            if not isinstance(href, str):
                continue
            url = _clean_external_url(href)
            if url is None or url in seen:
                continue
            seen.add(url)
            urls.append(url)

    logger.debug("citations_extracted", article=base_url, count=len(urls))
    return urls


def _clean_external_url(href: str) -> str | None:
    """Reduce a reference link to an absolute external URL.

    Args:
        href: The raw href.

    Returns:
        The URL, or None when it is internal, relative or not http(s).
    """
    candidate = href.strip()
    if candidate.startswith("//"):
        candidate = f"https:{candidate}"

    parts = urlsplit(candidate)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return None
    return candidate


def domain_allowed(url: str, allow_domains: Iterable[str]) -> bool:
    """Report whether a URL's host is on the allow list.

    Matching is on domain boundaries, so ``www.jstor.org`` is allowed by
    ``jstor.org`` while ``jstor.org.attacker.test`` is not.

    Args:
        url: The candidate URL.
        allow_domains: Domains from the spec's ``citation_allow_domains``.

    Returns:
        True if the URL may be fetched.
    """
    host = host_of(url)
    if not host:
        return False
    for domain in allow_domains:
        allowed = domain.strip().lower().lstrip(".")
        if allowed and (host == allowed or host.endswith(f".{allowed}")):
            return True
    return False


def split_by_allow_list(urls: Sequence[str], allow_domains: Iterable[str]) -> CitationSplit:
    """Partition citation URLs into fetchable and skipped.

    Args:
        urls: Candidate citation URLs.
        allow_domains: Domains from the spec's ``citation_allow_domains``.

    Returns:
        The partition. Both halves preserve input order.
    """
    domains = list(allow_domains)
    allowed: list[str] = []
    skipped: list[str] = []
    for url in urls:
        (allowed if domain_allowed(url, domains) else skipped).append(url)
    return CitationSplit(allowed=allowed, skipped=skipped)


def citation_filename(battle_url: str, suffix: str = ".jsonl") -> str:
    """Return the filename holding one battle's fetched citations.

    Args:
        battle_url: The battle article URL the citations belong to.
        suffix: File extension.

    Returns:
        A filesystem-safe filename keyed by battle, as the spec's outputs
        require.
    """
    return article_filename(battle_url, suffix=suffix)


def write_citation_records(
    directory: Path | str,
    battle_url: str,
    records: Sequence[dict[str, object]],
) -> Path:
    """Append fetched citation records for one battle as JSON lines.

    Args:
        directory: Target directory, created if absent.
        battle_url: The battle the citations belong to.
        records: One dict per citation, already JSON-serialisable.

    Returns:
        The path written.

    Raises:
        OSError: If the file cannot be written.
    """
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    path = target / citation_filename(battle_url)
    lines = [json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def citation_id(url: str) -> str:
    """Return a stable short identifier for a citation URL.

    Args:
        url: The citation URL.

    Returns:
        A hex digest prefix, used to key citation records.
    """
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
