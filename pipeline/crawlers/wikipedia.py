"""
Wikipedia list-page parsing and article fetching.

The seed config names fourteen battle-list pages. They overlap heavily --
Cannae appears on the chronological list, the Roman list and the list of
battles before 301 -- so discovery has to deduplicate, or the crawl would
fetch the same article several times and inflate every downstream count.

Link selection is deliberately conservative. A list page is mostly links that
are not battles: navigation boxes, references, year articles, polities,
commanders. Taking every ``/wiki/`` link would pull in tens of thousands of
irrelevant pages, so a link is kept only when its title reads like a battle
and its namespace is the article namespace.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit

import structlog
from bs4 import BeautifulSoup, Tag

from pipeline.crawlers.fetcher import Fetcher, FetchResult

__all__ = [
    "WIKIPEDIA_DOMAIN",
    "article_filename",
    "dedupe_key",
    "dedupe_urls",
    "fetch_article",
    "is_battle_title",
    "normalise_url",
    "parse_battle_links",
    "title_from_url",
    "write_article",
]

logger = structlog.get_logger()

WIKIPEDIA_DOMAIN = "wikipedia.org"

_ARTICLE_PREFIX = "/wiki/"

# Non-article namespaces. A title containing a colon is only a namespace when
# the part before it is one of these; "Siege of Vienna: 1683" is an article.
_NAMESPACES = frozenset(
    {
        "file",
        "image",
        "category",
        "template",
        "help",
        "portal",
        "wikipedia",
        "special",
        "talk",
        "user",
        "draft",
        "module",
        "mediawiki",
        "book",
        "wikt",
        "s",
        "commons",
    }
)

# Meta-articles that are indexes rather than battles. Without these the
# fourteen seed lists would discover each other endlessly.
_INDEX_PREFIXES = (
    "list of",
    "lists of",
    "timeline of",
    "timelines of",
    "outline of",
    "index of",
    "chronology of",
    "category of",
    "bibliography of",
    "order of battle",
)

# Titles that read like a single military engagement. Tuned for recall on the
# seed lists; the extract stage discards anything that turns out not to be a
# battle, so a false positive costs one fetch and a false negative costs a
# whole commander's record.
_BATTLE_TITLE_RE = re.compile(
    r"""(?xi)
    \b(
        battles?
      | sieges?
      | blockade
      | bombardment
      | raid
      | raids
      | skirmish
      | ambush
      | massacre
      | mutiny
      | naval\s+action
      | action\s+of
      | assault\s+on
      | attack\s+on
      | capture\s+of
      | sack\s+of
      | storming\s+of
      | relief\s+of
      | fall\s+of
      | defen[cs]e\s+of
      | landing\s+at
      | operation
      | engagement
    )\b
    """
)

# Page furniture whose links are navigation, not content.
_NAVIGATIONAL_CLASSES = frozenset(
    {
        "navbox",
        "navbox-inner",
        "vertical-navbox",
        "sidebar",
        "metadata",
        "reflist",
        "reference",
        "refbegin",
        "hatnote",
        "toc",
        "mw-editsection",
        "noprint",
        "catlinks",
        "authority-control",
    }
)


def normalise_url(href: str, base_url: str) -> str | None:
    """Resolve a link and reduce it to a canonical article URL.

    Args:
        href: The raw ``href`` from the page.
        base_url: The URL of the page the link was found on.

    Returns:
        An absolute URL with fragment and query removed, or None when the
        link does not point at a Wikipedia article.
    """
    if not href or href.startswith(("#", "mailto:", "javascript:", "tel:")):
        return None

    parts = urlsplit(urljoin(base_url, href))
    if parts.scheme not in ("http", "https"):
        return None

    host = (parts.hostname or "").lower()
    if host != WIKIPEDIA_DOMAIN and not host.endswith(f".{WIKIPEDIA_DOMAIN}"):
        return None

    if not parts.path.startswith(_ARTICLE_PREFIX) or parts.path == _ARTICLE_PREFIX:
        return None

    return urlunsplit((parts.scheme, (parts.netloc or "").lower(), parts.path, "", ""))


def title_from_url(url: str) -> str:
    """Return the human-readable article title a URL points at.

    Args:
        url: A Wikipedia article URL.

    Returns:
        The decoded title with underscores turned back into spaces.
    """
    path = urlsplit(url).path
    raw = path[len(_ARTICLE_PREFIX) :] if path.startswith(_ARTICLE_PREFIX) else path
    return unquote(raw).replace("_", " ").strip()


def dedupe_key(url: str) -> str:
    """Return the identity a URL should be deduplicated on.

    Percent-encoding differs between list pages: one links to
    ``Battle_of_Nan%E2%80%99an`` and another to the decoded form. Both name
    the same article, so the key is the decoded title.

    Args:
        url: A Wikipedia article URL.

    Returns:
        A canonical key for the article.
    """
    title = title_from_url(url)
    if not title:
        return url
    # MediaWiki capitalises the first letter of every title; the rest is
    # case-sensitive, so only that character is folded.
    return (title[0].upper() + title[1:]).replace(" ", "_")


def is_battle_title(title: str) -> bool:
    """Report whether an article title names a single military engagement.

    Args:
        title: A decoded article title.

    Returns:
        True if the title should be crawled as a battle article.
    """
    cleaned = title.strip()
    if not cleaned:
        return False

    prefix, separator, _ = cleaned.partition(":")
    if separator and prefix.strip().lower() in _NAMESPACES:
        return False

    lowered = cleaned.lower()
    if lowered.startswith(_INDEX_PREFIXES):
        return False

    return bool(_BATTLE_TITLE_RE.search(lowered))


def _is_navigational(anchor: Tag) -> bool:
    """Report whether a link sits in page furniture rather than content.

    Args:
        anchor: The anchor tag.

    Returns:
        True if any ancestor is a navbox, reference list or similar.
    """
    for parent in anchor.parents:
        if not isinstance(parent, Tag):
            continue
        classes = parent.get("class")
        if isinstance(classes, list) and _NAVIGATIONAL_CLASSES.intersection(classes):
            return True
        if parent.name == "sup":
            return True
    return False


def parse_battle_links(html: str, base_url: str) -> list[str]:
    """Extract links to individual battle articles from a battle-list page.

    Args:
        html: The list page's HTML.
        base_url: The URL the HTML came from, used to resolve relative links.

    Returns:
        Deduplicated absolute article URLs, in the order they appear.
    """
    soup = BeautifulSoup(html, "html.parser")
    content = soup.select_one("div.mw-parser-output") or soup.select_one("#mw-content-text") or soup

    found: list[str] = []
    for anchor in content.find_all("a", href=True):
        if not isinstance(anchor, Tag):
            continue
        href = anchor.get("href")
        if not isinstance(href, str):
            continue
        if _is_navigational(anchor):
            continue

        url = normalise_url(href, base_url)
        if url is None or not is_battle_title(title_from_url(url)):
            continue
        found.append(url)

    links = dedupe_urls(found)
    logger.debug("battle_links_parsed", source=base_url, anchors=len(found), unique=len(links))
    return links


def dedupe_urls(urls: Iterable[str]) -> list[str]:
    """Remove duplicate article URLs while preserving discovery order.

    Args:
        urls: Candidate article URLs, possibly differing only in encoding.

    Returns:
        One URL per distinct article, first spelling seen.
    """
    seen: set[str] = set()
    unique: list[str] = []
    for url in urls:
        key = dedupe_key(url)
        if key in seen:
            continue
        seen.add(key)
        unique.append(url)
    return unique


def article_filename(url: str, suffix: str = ".html") -> str:
    """Return a filesystem-safe filename for an article's raw content.

    The title is kept readable so the raw directory can be browsed by hand,
    with a hash of the full URL appended because sanitising titles can map
    two distinct articles onto the same name.

    Args:
        url: The article URL.
        suffix: File extension to append.

    Returns:
        A filename valid on Windows and POSIX alike.
    """
    title = title_from_url(url) or url
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", title).strip("._") or "article"
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
    return f"{safe[:100]}-{digest}{suffix}"


async def fetch_article(fetcher: Fetcher, url: str) -> FetchResult:
    """Fetch one battle article.

    Args:
        fetcher: The configured fetcher.
        url: The article URL.

    Returns:
        The fetch result, successful or not.
    """
    result = await fetcher.fetch(url)
    if not result.ok:
        logger.info("article_fetch_failed", url=url, status=result.status, error=result.error)
    return result


def write_article(directory: Path | str, url: str, html: str) -> Path:
    """Write an article's HTML into the raw data directory.

    Args:
        directory: Target directory, created if absent.
        url: The article URL, used to derive the filename.
        html: The raw HTML.

    Returns:
        The path written.

    Raises:
        OSError: If the file cannot be written.
    """
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    path = target / article_filename(url)
    path.write_text(html, encoding="utf-8")
    return path
