"""
DBpedia resource fetching.

DBpedia re-publishes Wikipedia's infoboxes as RDF, already typed and with the
template mess resolved. Where an infobox parse is ambiguous, having DBpedia's
reading of the same infobox gives the extract stage a second opinion that
costs one cheap request per battle.

Every DBpedia resource is keyed by the English Wikipedia article title, so no
separate entity resolution is needed at this stage.
"""

from __future__ import annotations

from urllib.parse import quote

import structlog

from pipeline.crawlers.fetcher import Fetcher, FetchResult
from pipeline.crawlers.wikipedia import title_from_url

__all__ = ["DBPEDIA_DATA_TEMPLATE", "fetch_resource", "resource_json_url"]

logger = structlog.get_logger()

DBPEDIA_DATA_TEMPLATE = "https://dbpedia.org/data/{title}.json"


def resource_json_url(article_title: str) -> str:
    """Return the DBpedia JSON URL for a Wikipedia article title.

    Args:
        article_title: The article title, with spaces or underscores.

    Returns:
        The DBpedia data URL.

    Raises:
        ValueError: If the title is blank, which means a parsing bug upstream.
    """
    cleaned = article_title.strip().replace(" ", "_")
    if not cleaned:
        raise ValueError("Cannot build a DBpedia URL from a blank title")
    return DBPEDIA_DATA_TEMPLATE.format(title=quote(cleaned, safe="_(),'-"))


async def fetch_resource(fetcher: Fetcher, battle_url: str) -> FetchResult:
    """Fetch the DBpedia resource matching a Wikipedia battle article.

    Args:
        fetcher: The configured fetcher.
        battle_url: The Wikipedia article URL.

    Returns:
        The fetch result, successful or not.

    Raises:
        ValueError: If no title can be derived from the URL.
    """
    url = resource_json_url(title_from_url(battle_url))
    result = await fetcher.fetch(url)
    if not result.ok:
        logger.debug("dbpedia_fetch_failed", url=url, status=result.status, error=result.error)
    return result
