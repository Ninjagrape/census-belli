"""
Crawlers for the General WAR pipeline's crawl stage.

Each module owns one source: Wikipedia list pages and articles, the Wikidata
query service, DBpedia resources, and the external citations an article
points at. They share one :class:`~pipeline.crawlers.fetcher.Fetcher`, so
rate limiting, robots.txt, retries and ``crawl_log`` recording behave the
same whichever source is being read.

The stage runner in ``pipeline.stages.crawl`` sequences them; nothing here
reaches for configuration or the database on its own.
"""

from __future__ import annotations

from pipeline.crawlers.fetcher import (
    Fetcher,
    FetchResult,
    RateLimiter,
    RobotsPolicy,
    content_hash,
    host_of,
)
from pipeline.crawlers.log import (
    CrawlLogEntry,
    CrawlLogWriter,
    InMemoryCrawlLog,
    SqlCrawlLog,
    entry_from_result,
)
from pipeline.crawlers.seeds import SeedConfigError, Seeds, load_seeds
from pipeline.crawlers.state import CrawlState, load_state, save_state

__all__ = [
    "CrawlLogEntry",
    "CrawlLogWriter",
    "CrawlState",
    "FetchResult",
    "Fetcher",
    "InMemoryCrawlLog",
    "RateLimiter",
    "RobotsPolicy",
    "SeedConfigError",
    "Seeds",
    "SqlCrawlLog",
    "content_hash",
    "entry_from_result",
    "host_of",
    "load_seeds",
    "load_state",
    "save_state",
]
