"""
Wikidata SPARQL and entity fetching.

The Wikipedia lists miss battles that have a Wikidata item but no list entry,
and Wikidata carries structured dates, coordinates and participant links that
no infobox parse can match for reliability. The queries themselves live in
``config/sources_seed.yaml`` so that broadening the corpus is a config change
rather than a code change.

SPARQLWrapper is the obvious client here, but it is synchronous and would sit
outside the rate limiter and retry policy that the rest of the crawl runs
under. The endpoint is a plain HTTP GET, so it is issued through the same
:class:`~pipeline.crawlers.fetcher.Fetcher` as everything else and every
attempt lands in ``crawl_log`` like any other fetch.
"""

from __future__ import annotations

import json
import re
from typing import Any

import structlog

from pipeline.crawlers.fetcher import Fetcher, FetchResult

__all__ = [
    "ENTITY_DATA_TEMPLATE",
    "SPARQL_ENDPOINT",
    "entity_data_url",
    "entity_ids",
    "parse_sparql_bindings",
    "qid_from_uri",
    "run_query",
]

logger = structlog.get_logger()

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
ENTITY_DATA_TEMPLATE = "https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"

_QID_RE = re.compile(r"/(Q\d+)$")


def qid_from_uri(uri: str) -> str | None:
    """Extract a Wikidata item id from an entity URI.

    Args:
        uri: An entity URI such as ``http://www.wikidata.org/entity/Q178561``.

    Returns:
        The Q-id, or None when the URI does not name an item.
    """
    match = _QID_RE.search(uri.strip())
    return match.group(1) if match else None


def entity_data_url(qid: str) -> str:
    """Return the JSON URL for a Wikidata entity.

    Args:
        qid: A Wikidata item id, e.g. ``Q178561``.

    Returns:
        The Special:EntityData JSON URL.

    Raises:
        ValueError: If the id is not a well-formed Q-id. A malformed id here
            means a parsing bug upstream, not bad remote data.
    """
    cleaned = qid.strip()
    if not re.fullmatch(r"Q\d+", cleaned):
        raise ValueError(f"Not a Wikidata item id: {qid!r}")
    return ENTITY_DATA_TEMPLATE.format(qid=cleaned)


def parse_sparql_bindings(payload: str) -> list[dict[str, str]]:
    """Flatten a SPARQL JSON result into plain string rows.

    Args:
        payload: The endpoint's JSON response body.

    Returns:
        One dict per row, mapping variable name to its literal or URI value.
        An unparseable or unexpected payload yields an empty list: a bad
        response from a remote service is data, not a crash.
    """
    try:
        parsed: Any = json.loads(payload)
    except json.JSONDecodeError as exc:
        logger.warning("sparql_payload_unparseable", error=str(exc))
        return []

    if not isinstance(parsed, dict):
        logger.warning("sparql_payload_not_object", type=type(parsed).__name__)
        return []

    results = parsed.get("results")
    bindings = results.get("bindings") if isinstance(results, dict) else None
    if not isinstance(bindings, list):
        logger.warning("sparql_payload_missing_bindings")
        return []

    rows: list[dict[str, str]] = []
    for binding in bindings:
        if not isinstance(binding, dict):
            continue
        row = {
            str(name): str(cell.get("value", ""))
            for name, cell in binding.items()
            if isinstance(cell, dict)
        }
        if row:
            rows.append(row)
    return rows


def entity_ids(rows: list[dict[str, str]], variable: str = "battle") -> list[str]:
    """Collect distinct Wikidata ids from query rows.

    Args:
        rows: Rows as returned by :func:`parse_sparql_bindings`.
        variable: The SPARQL variable holding entity URIs.

    Returns:
        Q-ids in first-seen order, without duplicates.
    """
    seen: set[str] = set()
    ids: list[str] = []
    for row in rows:
        uri = row.get(variable)
        if not uri:
            continue
        qid = qid_from_uri(uri)
        if qid is None or qid in seen:
            continue
        seen.add(qid)
        ids.append(qid)
    return ids


async def run_query(
    fetcher: Fetcher,
    query: str,
    *,
    endpoint: str = SPARQL_ENDPOINT,
) -> FetchResult:
    """Run one SPARQL query against the Wikidata endpoint.

    Args:
        fetcher: The configured fetcher, which supplies rate limiting,
            robots handling and retries.
        query: The SPARQL text from the seed config.
        endpoint: Query service URL.

    Returns:
        The fetch result. ``result.text`` is the raw JSON body on success.

    Raises:
        ValueError: If the query is empty, which is a seed-config bug.
    """
    if not query.strip():
        raise ValueError("SPARQL query is empty")

    result = await fetcher.fetch(
        endpoint,
        params={"query": query, "format": "json"},
        headers={"Accept": "application/sparql-results+json"},
    )
    if not result.ok:
        logger.warning("sparql_query_failed", status=result.status, error=result.error)
    return result


async def fetch_entity(fetcher: Fetcher, qid: str) -> FetchResult:
    """Fetch one Wikidata entity's JSON.

    Args:
        fetcher: The configured fetcher.
        qid: The item id.

    Returns:
        The fetch result, successful or not.

    Raises:
        ValueError: If the id is malformed.
    """
    return await fetcher.fetch(entity_data_url(qid))
