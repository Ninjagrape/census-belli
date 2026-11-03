"""
Identifying a commander by the article the infobox links, not the name it shows.

An infobox editor who writes ``[[George Henry Thomas|G. H. Thomas]]`` has
already said who the commander is. Matching the display text "G. H. Thomas"
against Wikidata labels ignores that, and on corpus3 about 240 linked rows
pointed at a namesake (an engraver, a sultan of Anjouan) while the infobox
linked the right man; fixing them took 175 manual overrides. This module uses
the link.

**Where the link becomes a Q-id.** English Wikipedia titles map to Wikidata
items through sitelinks, queried in batches from the SPARQL endpoint
(:func:`fetch_sitelink_qids`). A title the query misses is usually a
redirect, which has no sitelink of its own; those are read from the article
page itself (:func:`fetch_redirect_qids`). Answers, including "no item", are
cached in ``data/processed/resolve/wikilink_qids.json`` so a re-run asks
nothing twice (:class:`WikilinkIndex`).

**Precedence** (per mention): a manual override wins; else a wikilink to
exactly one human who could have commanded in that battle decides; else the
name matcher runs unchanged. :class:`WikilinkLinker` implements the middle
step as a pre-pass that mirrors how overrides carve mentions out of name
groups: the linked mentions leave their name group for a group per Q-id with a
fixed decision, and the rest of the name group is matched as before.

A rank link such as ``[[Vice admiral]]`` resolves to a non-human item and is
ignored, so ``[[Vice admiral]] [[Ernst Schirlitz]]`` still yields one person.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Final
from urllib.parse import quote

import structlog

from pipeline.crawlers.fetcher import Fetcher
from pipeline.crawlers.wikidata import (
    parse_sparql_bindings,
    qid_from_article_html,
    qid_from_uri,
    run_query,
)
from pipeline.resolvers.candidates import (
    DEFAULT_BATCH_SIZE,
    CandidateSource,
    QidLookupSource,
    escape_sparql,
    is_unlabelled,
)
from pipeline.resolvers.matcher import lifespan_verdict, match_group
from pipeline.resolvers.overrides import _name_forms, _with_mentions
from pipeline.resolvers.records import Candidate, Decision, Mention, MentionGroup

__all__ = [
    "DEFAULT_MAX_PAGE_FETCHES",
    "WIKILINK_CONFIDENCE",
    "WIKILINK_METHOD",
    "WikilinkIndex",
    "WikilinkLinker",
    "build_sitelink_query",
    "collect_link_titles",
    "fetch_redirect_qids",
    "fetch_sitelink_qids",
    "wikilink_cache_path",
]

logger = structlog.get_logger()

WIKILINK_METHOD: Final[str] = "infobox_wikilink"

# High but below a manual link's 1.0: the infobox is an editor's claim about a
# Wikipedia article, and the article's Wikidata item is a second hop.
WIKILINK_CONFIDENCE: Final[float] = 0.95

DEFAULT_MAX_PAGE_FETCHES: Final[int] = 2000

_CACHE_FILE: Final[str] = "wikilink_qids.json"
_WIKIPEDIA_ARTICLE_PREFIX: Final[str] = "https://en.wikipedia.org/wiki/"
_GROUP_KEY_PREFIX: Final[str] = "wikilink:"

_SITELINK_QUERY: Final[str] = (
    "SELECT ?title ?item WHERE { VALUES ?title { %s } "
    "?article schema:about ?item ; "
    "schema:isPartOf <https://en.wikipedia.org/> ; "
    "schema:name ?title . }"
)


def wikilink_cache_path(processed_root: Path) -> Path:
    """Where title to Q-id answers are cached.

    Args:
        processed_root: The ``data/processed`` directory.

    Returns:
        ``<processed_root>/resolve/wikilink_qids.json``.
    """
    return processed_root / "resolve" / _CACHE_FILE


def collect_link_titles(groups: Iterable[MentionGroup]) -> list[str]:
    """List every distinct article title the groups' mentions link.

    Args:
        groups: The grouped mentions.

    Returns:
        Sorted titles, so every request built from them is byte-stable.
    """
    return sorted({t for g in groups for m in g.mentions for t in m.link_titles})


def build_sitelink_query(titles: Sequence[str]) -> str:
    """Render the sitelink query for one batch of titles.

    Args:
        titles: English Wikipedia article titles, already in the order to send.

    Returns:
        A SPARQL SELECT binding each title to the item whose enwiki sitelink
        it is.
    """
    values = " ".join(f'"{escape_sparql(t)}"@en' for t in titles)
    return _SITELINK_QUERY % values


async def fetch_sitelink_qids(
    fetcher: Fetcher,
    titles: Iterable[str],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    answered: set[str] | None = None,
) -> dict[str, str]:
    """Map English Wikipedia titles to Wikidata items through their sitelinks.

    Args:
        fetcher: A configured fetcher. The query goes through ``fetch_raw``
            (via :func:`run_query`) on purpose: robots.txt disallows /sparql.
        titles: Titles to look up; sorted and deduplicated before batching so
            the requests are byte-stable.
        batch_size: Titles per query.
        answered: When given, receives every title whose batch succeeded, so
            the caller can tell "no sitelink" from "the query failed".

    Returns:
        Title to Q-id for each title that is an enwiki sitelink. A redirect or
        a missing page is simply absent. A failed batch is logged and skipped.
    """
    ordered = sorted(set(titles))
    size = max(1, batch_size)
    found: dict[str, str] = {}

    for start in range(0, len(ordered), size):
        batch = ordered[start : start + size]
        result = await run_query(fetcher, build_sitelink_query(batch))
        if not result.ok or not result.text:
            logger.warning(
                "sparql_sitelink_batch_failed",
                titles=len(batch),
                status=result.status,
                error=result.error,
            )
            continue
        for row in parse_sparql_bindings(result.text):
            qid = qid_from_uri(row.get("item", ""))
            title = row.get("title", "")
            if qid is not None and title:
                found.setdefault(title, qid)
        if answered is not None:
            answered.update(batch)

    logger.info("sparql_sitelink_prefetch_complete", asked=len(ordered), found=len(found))
    return found


def _article_url(title: str) -> str:
    """The ``/wiki/`` URL of an article, which robots.txt allows crawling."""
    return _WIKIPEDIA_ARTICLE_PREFIX + quote(title.replace(" ", "_"), safe="_(),:")


async def fetch_redirect_qids(
    fetcher: Fetcher,
    titles: Iterable[str],
    *,
    max_fetches: int = DEFAULT_MAX_PAGE_FETCHES,
) -> dict[str, str | None]:
    """Find the item of titles the sitelink query missed, from the article page.

    A wikilink often targets a redirect ("Abdallah I of Jordan"), and a
    redirect has no sitelink; Wikipedia serves the target article, whose own
    markup names the item. The page goes through the ordinary robots-checked
    ``fetch`` (robots.txt allows /wiki/ and disallows /w/ and /api/), never
    the Action API.

    Args:
        fetcher: A configured fetcher.
        titles: Titles to read, in the order to try them.
        max_fetches: Page fetches to spend; the rest are skipped and logged.

    Returns:
        Title to Q-id, or None when Wikipedia has no such page or the page
        carries no item. A title whose fetch failed for any other reason is
        absent, so it is not cached as an answer and is asked again next run.
    """
    ordered = sorted(set(titles))
    budget = max(0, max_fetches)
    found: dict[str, str | None] = {}

    for title in ordered[:budget]:
        result = await fetcher.fetch(_article_url(title))
        if result.ok and result.text is not None:
            found[title] = qid_from_article_html(result.text)
        elif result.status == 404:
            found[title] = None

    skipped = len(ordered) - min(len(ordered), budget)
    if skipped:
        logger.warning(
            "resolve_wikilink_page_fetches_skipped",
            skipped=skipped,
            max_fetches=budget,
            hint="raise params.wikilink_max_page_fetches; the skipped titles are not cached",
        )
    logger.info("wikilink_redirect_fetch_complete", fetched=min(len(ordered), budget))
    return found


def _read_cache(path: Path | None) -> dict[str, str | None]:
    """Read the title to Q-id cache; an absent or unreadable file is empty."""
    if path is None or not path.is_file():
        return {}
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("wikilink_cache_unreadable", path=str(path), error=str(exc))
        return {}
    if not isinstance(raw, dict):
        return {}
    return {
        str(title): (qid if isinstance(qid, str) else None)
        for title, qid in raw.items()
        if qid is None or isinstance(qid, str)
    }


class WikilinkIndex:
    """The titles a run needs, and the Q-id each one is known to have.

    Args:
        titles: The article titles the run's mentions link.
        cache_path: The JSON cache, or None for no cache.
    """

    def __init__(self, titles: Iterable[str], cache_path: Path | None) -> None:
        self.titles = sorted(set(titles))
        self.cache_path = cache_path
        self._cache = _read_cache(cache_path)
        # None means looked up and found to have no item.
        self.qids: dict[str, str | None] = {
            t: self._cache[t] for t in self.titles if t in self._cache
        }

    def uncached(self) -> list[str]:
        """Titles with no answer yet."""
        return [t for t in self.titles if t not in self.qids]

    def required_qids(self) -> list[str]:
        """The distinct Q-ids the run must fetch as candidates, sorted."""
        return sorted({q for q in self.qids.values() if q})

    def log_uncached(self) -> None:
        """Say how many titles a cache-only run could not answer."""
        missing = self.uncached()
        if missing:
            logger.info(
                "resolve_wikilinks_uncached",
                titles=len(missing),
                of=len(self.titles),
                hint="run with candidate_source: sparql once to fill the cache",
            )

    async def resolve(self, fetcher: Fetcher, *, batch_size: int, max_page_fetches: int) -> None:
        """Answer every uncached title and save the cache.

        Args:
            fetcher: A configured fetcher.
            batch_size: Titles per sitelink query.
            max_page_fetches: Redirect page fetches to spend.
        """
        todo = self.uncached()
        if not todo:
            return
        answered: set[str] = set()
        sitelinks = await fetch_sitelink_qids(
            fetcher, todo, batch_size=batch_size, answered=answered
        )
        self.qids.update(sitelinks)
        # Only a title whose query succeeded may be chased through its page: if
        # the query failed, the title may well have a sitelink.
        misses = [t for t in todo if t in answered and t not in sitelinks]
        self.qids.update(await fetch_redirect_qids(fetcher, misses, max_fetches=max_page_fetches))
        self.save()

    def save(self) -> None:
        """Write the cache, keeping answers for titles this run did not need."""
        if self.cache_path is None:
            return
        merged = {**self._cache, **self.qids}
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(dict(sorted(merged.items())), ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        logger.info("wikilink_cache_written", path=str(self.cache_path), titles=len(merged))


def _group_for(qid: str, mentions: list[Mention]) -> MentionGroup:
    """Build the group a Q-id's wikilinked mentions are resolved as."""
    forms, keys = _name_forms(mentions)
    counts = Counter(forms)
    spelled = Counter(m.name.strip() for m in mentions)
    display = max(forms, key=lambda f: (spelled.get(f, 0) + counts[f], len(f), f))
    key = f"{_GROUP_KEY_PREFIX}{qid}"
    return MentionGroup(
        key=key,
        keys=keys or (key,),
        display_name=display,
        surface_forms=forms,
        mentions=mentions,
        confidence=WIKILINK_CONFIDENCE,
        method=WIKILINK_METHOD,
    )


class WikilinkLinker:
    """Links mentions to the person their infobox wikilink names.

    Args:
        title_qids: Article title to Q-id, None where the title has no item.
    """

    def __init__(self, title_qids: Mapping[str, str | None]) -> None:
        self._title_qids = title_qids

    def take_links(
        self,
        groups: list[MentionGroup],
        source: CandidateSource,
        *,
        threshold: float | None = None,
        margin: float | None = None,
    ) -> tuple[list[MentionGroup], dict[str, Decision]]:
        """Pull wikilink-identified mentions out of their name groups.

        Args:
            groups: The name groups, after manual overrides took theirs.
                Edited in place: taken mentions are removed and groups left
                empty are dropped.
            source: The candidate source, asked for each Q-id's entity. A
                source that cannot fetch by Q-id links nothing.
            threshold: The spec's ``fuzzy_threshold``. With ``margin``, lets
                the log say how many links differ from what name matching
                alone would have chosen.
            margin: The spec's ``ambiguity_margin``.

        Returns:
            One new group per Q-id, and their fixed decisions keyed by group
            key. The groups skip the matcher, the lifespan guard and the LLM;
            each mention was already judged against the person's lifespan.
        """
        counts: Counter[str] = Counter()
        lookups: dict[str, Candidate | None] = {}
        taken: dict[str, list[Mention]] = {}
        titles_for: dict[str, str] = {}
        taken_ids: set[int] = set()

        def candidate_of(qid: str) -> Candidate | None:
            if qid not in lookups:
                lookups[qid] = (
                    source.candidate_by_qid(qid) if isinstance(source, QidLookupSource) else None
                )
            return lookups[qid]

        for group in groups:
            preview = None
            for mention in group.mentions:
                if not mention.link_titles:
                    continue
                counts["mentions_with_links"] += 1
                picked = self._identify(mention, candidate_of, counts)
                if picked is None:
                    continue
                candidate, title = picked
                taken.setdefault(candidate.qid, []).append(mention)
                titles_for.setdefault(candidate.qid, title)
                taken_ids.add(id(mention))
                if threshold is not None and margin is not None:
                    if preview is None:
                        found = source.search(group)
                        preview = match_group(group, found, threshold=threshold, margin=margin)
                    if preview.status != "linked":
                        counts["name_match_undecided"] += 1
                    elif preview.qid == candidate.qid:
                        counts["agrees_with_name_match"] += 1
                    else:
                        counts["differs_from_name_match"] += 1

        new_groups: list[MentionGroup] = []
        decisions: dict[str, Decision] = {}
        for qid in sorted(taken):
            person = candidate_of(qid)
            assert person is not None
            if is_unlabelled(person):
                # The label service had no English label; the linked article's
                # own title names the person, and a Q-id must never be a name.
                person = replace(person, label=titles_for[qid])
            group = _group_for(qid, taken[qid])
            new_groups.append(group)
            decisions[group.key] = Decision(
                status="linked",
                canonical_name=person.label,
                method=WIKILINK_METHOD,
                qid=person.qid,
                confidence=WIKILINK_CONFIDENCE,
                reasoning=f"the infobox links the article {titles_for[qid]!r} ({qid})",
                candidate=person,
            )
            counts["linked"] += len(taken[qid])

        rebuilt: list[MentionGroup] = []
        for group in groups:
            kept = [m for m in group.mentions if id(m) not in taken_ids]
            if len(kept) == len(group.mentions):
                rebuilt.append(group)
            elif kept:
                rebuilt.append(_with_mentions(group, kept))
        groups[:] = rebuilt

        if counts["mentions_with_links"]:
            logger.info(
                "resolve_wikilinks",
                mentions_with_links=counts["mentions_with_links"],
                linked=counts["linked"],
                ignored_unresolved_title=counts["ignored_unresolved_title"],
                ignored_non_human=counts["ignored_non_human"],
                ignored_multi_human=counts["ignored_multi_human"],
                ignored_date_rejected=counts["ignored_date_rejected"],
                agrees_with_name_match=counts["agrees_with_name_match"],
                differs_from_name_match=counts["differs_from_name_match"],
                name_match_undecided=counts["name_match_undecided"],
            )
        return new_groups, decisions

    def _identify(
        self,
        mention: Mention,
        candidate_of: Any,
        counts: Counter[str],
    ) -> tuple[Candidate, str] | None:
        """Decide whether one mention's links name exactly one usable human.

        Args:
            mention: A mention with at least one link title.
            candidate_of: Q-id to the human entity, or None.
            counts: Receives the reason when the links are ignored.

        Returns:
            The person and the title that named them, or None.
        """
        titled = [(t, self._title_qids.get(t)) for t in mention.link_titles]
        resolved = [(t, q) for t, q in titled if q]
        if not resolved:
            counts["ignored_unresolved_title"] += 1
            return None

        humans: dict[str, tuple[Candidate, str]] = {}
        for title, qid in resolved:
            candidate = candidate_of(qid)
            if candidate is not None:
                humans.setdefault(qid, (candidate, title))
        if not humans:
            counts["ignored_non_human"] += 1
            return None
        if len(humans) > 1:
            counts["ignored_multi_human"] += 1
            return None

        candidate, title = next(iter(humans.values()))
        year = mention.context.year if mention.context else None
        if lifespan_verdict(candidate, [] if year is None else [year]) is False:
            counts["ignored_date_rejected"] += 1
            return None
        return candidate, title
