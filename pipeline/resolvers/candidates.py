"""
Where candidate Wikidata entities for a commander mention come from.

Two sources implement :class:`CandidateSource`, and which one a run uses is a
spec setting rather than a code change:

``sparql``
    Asks the live Wikidata query service for humans whose label or alias
    matches a surface form. This is what ``agents/resolve.yaml`` step 1
    describes, and it is the only source that can resolve a commander whose
    battle article the crawl stage never fetched an entity for.

``local``
    Indexes the entity JSON already in ``data/raw/wikidata``. Offline,
    free, and the source the tests use. It only knows entities the crawl
    stage happened to fetch, so on a real corpus it is a supplement rather
    than a substitute.

Both are queried with *surface forms*, not with the folded keys from
:mod:`pipeline.resolvers.names`: Wikidata literals are cased and accented, so
"Jose de San Martin" would miss where "José de San Martín" hits. Folding
happens afterwards, on both sides, in the matcher.

The SPARQL side is a coroutine, :func:`fetch_candidates`, rather than a source
that reaches out per group: one HTTP client, one rate limiter and one event
loop for the whole run. It goes through the crawl stage's
:class:`~pipeline.crawlers.fetcher.Fetcher`, so it inherits the rate limiting
and the retry policy the rest of the project's HTTP already has, and what it
returns is wrapped in :class:`PrefetchedCandidateSource`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Final, Protocol

import structlog

from pipeline.crawlers.fetcher import Fetcher
from pipeline.crawlers.wikidata import parse_sparql_bindings, qid_from_uri, run_query
from pipeline.extractors.wikidata_mapper import wikidata_time_to_date
from pipeline.resolvers.mentions import astronomical_year
from pipeline.resolvers.names import fold
from pipeline.resolvers.records import Candidate, MentionGroup

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_MAX_NAMES_PER_GROUP",
    "CandidateLookupError",
    "CandidateSource",
    "LocalCandidateSource",
    "NullCandidateSource",
    "PrefetchedCandidateSource",
    "candidate_keys",
    "collect_query_names",
    "fetch_candidates",
    "person_candidates",
    "query_names",
]

logger = structlog.get_logger()

_Q_HUMAN: Final[str] = "Q5"
_P_INSTANCE_OF: Final[str] = "P31"
_P_BIRTH: Final[str] = "P569"
_P_DEATH: Final[str] = "P570"

# One query asks for every name in a batch at once. Kept modest because the
# query service times out on large VALUES blocks joined against altLabel.
DEFAULT_BATCH_SIZE: Final[int] = 25

# The query GROUPs rather than joining country, occupation and article
# directly, and that is a performance decision, not a tidiness one. Joined
# plainly, one person yields |countries| x |occupations| x |articles| rows;
# measured on an eight-name batch that was 90 rows and 30 seconds, against 36
# rows and 1.9 seconds aggregated -- the same 35 people either way. At corpus
# scale the difference is between usable and not.

# Escaped into a SPARQL string literal; a name carrying a quote or a newline
# would otherwise end the literal early and change the query.
_SPARQL_ESCAPES: Final[tuple[tuple[str, str], ...]] = (
    ("\\", "\\\\"),
    ('"', '\\"'),
    ("\n", "\\n"),
    ("\r", "\\r"),
    ("\t", "\\t"),
)

# ?name is selected, not merely bound. The query matches on label OR altLabel,
# so the requested name often is not the entity's label -- "Duke of
# Wellington" is an alias of "Arthur Wellesley, 1st Duke of Wellington". Ask
# the endpoint which name matched rather than working it out again afterwards
# from the label alone, which silently discarded every alias-matched entity:
# Wellington and Nelson both, until 2026-09-19.
_LABEL_QUERY: Final[str] = """
SELECT ?name ?person ?personLabel ?personDescription ?birth ?death
       (GROUP_CONCAT(DISTINCT ?countryLabel; separator=", ") AS ?countries)
       (GROUP_CONCAT(DISTINCT ?occupationLabel; separator=", ") AS ?occupations)
       (SAMPLE(?articleUrl) AS ?article) WHERE {{
  VALUES ?name {{ {names} }}
  VALUES ?labelProp {{ rdfs:label skos:altLabel }}
  ?person ?labelProp ?name .
  ?person wdt:P31 wd:Q5 .
  OPTIONAL {{ ?person wdt:P569 ?birth . }}
  OPTIONAL {{ ?person wdt:P570 ?death . }}
  OPTIONAL {{
    ?person wdt:P27 ?country .
    ?country rdfs:label ?countryLabel .
    FILTER(LANG(?countryLabel) = "en")
  }}
  OPTIONAL {{
    ?person wdt:P106 ?occupation .
    ?occupation rdfs:label ?occupationLabel .
    FILTER(LANG(?occupationLabel) = "en")
  }}
  OPTIONAL {{
    ?articleUrl schema:about ?person ;
                schema:isPartOf <https://en.wikipedia.org/> .
  }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
}}
GROUP BY ?name ?person ?personLabel ?personDescription ?birth ?death
LIMIT 400
"""


class CandidateLookupError(RuntimeError):
    """Every candidate query failed, so the lookup produced nothing at all.

    Distinct from "these names matched no entity", which is ordinary and
    common. This means the stage never got an answer -- a disallowed path, a
    malformed query, an unreachable endpoint -- and every name would resolve
    as a new entity if the run continued.

    The project's rule is graceful degradation when *part* of a system errors.
    A total failure is not that, and swallowing it is how robots.txt disabled
    every SPARQL query in the project without anyone noticing.
    """


def candidate_keys(candidate: Candidate) -> tuple[str, ...]:
    """Every folded name a candidate can be matched on.

    Args:
        candidate: The entity.

    Returns:
        The folded label followed by the folded aliases, deduplicated and
        with empties dropped.
    """
    keys: list[str] = []
    for value in (candidate.label, *candidate.aliases):
        folded = fold(value)
        if folded and folded not in keys:
            keys.append(folded)
    return tuple(keys)


class CandidateSource(Protocol):
    """Somewhere candidate entities for a mention can be looked up."""

    def prefetch(self, groups: Sequence[MentionGroup]) -> None:
        """Fetch candidates for many groups at once.

        Called before the first :meth:`search`. A source with nothing to
        batch may do nothing.

        Args:
            groups: Every group the run will resolve.
        """
        ...

    def search(self, group: MentionGroup) -> list[Candidate]:
        """Return the entities a group's names could refer to.

        Args:
            group: The mention group to look up.

        Returns:
            Candidates, unordered and ungated. Filtering on lifespan and
            scoring both belong to the matcher.
        """
        ...


class NullCandidateSource:
    """A source that knows nothing, for dry runs and deterministic tests."""

    def prefetch(self, groups: Sequence[MentionGroup]) -> None:
        """Do nothing.

        Args:
            groups: Ignored.
        """
        return None

    def search(self, group: MentionGroup) -> list[Candidate]:
        """Return no candidates.

        Args:
            group: Ignored.

        Returns:
            An empty list, so every named group resolves to a new entity.
        """
        return []


def _entities(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """List the entity objects inside a crawled Wikidata payload.

    Args:
        payload: A ``wbgetentities`` response or a bare entity.

    Returns:
        Every entity mapping the payload carries.
    """
    entities = payload.get("entities")
    if isinstance(entities, dict) and entities:
        return [value for value in entities.values() if isinstance(value, dict)]
    return [payload]


def _claim_values(entity: dict[str, Any], prop: str) -> list[dict[str, Any]]:
    """Read the data values of one property's claims.

    Args:
        entity: A Wikidata entity mapping.
        prop: The property id.

    Returns:
        Each claim's ``datavalue.value`` mapping, skipping malformed claims.
    """
    claims = entity.get("claims")
    if not isinstance(claims, dict):
        return []

    values: list[dict[str, Any]] = []
    for claim in claims.get(prop, []) or []:
        if not isinstance(claim, dict):
            continue
        mainsnak = claim.get("mainsnak")
        if not isinstance(mainsnak, dict):
            continue
        datavalue = mainsnak.get("datavalue")
        if not isinstance(datavalue, dict):
            continue
        value = datavalue.get("value")
        if isinstance(value, dict):
            values.append(value)
    return values


def _is_human(entity: dict[str, Any]) -> bool:
    """Whether an entity is an instance of human (Q5).

    Args:
        entity: A Wikidata entity mapping.

    Returns:
        True when any ``P31`` claim names Q5. A battle entity is not a
        candidate commander, and the raw directory holds mostly battles.
    """
    return any(value.get("id") == _Q_HUMAN for value in _claim_values(entity, _P_INSTANCE_OF))


def _claim_year(entity: dict[str, Any], prop: str) -> int | None:
    """Read a time-valued claim as an astronomical year.

    Args:
        entity: A Wikidata entity mapping.
        prop: A time-valued property id, e.g. ``"P569"``.

    Returns:
        The astronomical year, or None when the claim is absent or coarser
        than the mapper will convert.
    """
    for value in _claim_values(entity, prop):
        literal, _ = wikidata_time_to_date(value)
        year = astronomical_year(literal)
        if year is not None:
            return year
    return None


def _english(block: Any) -> str:
    """Read the English value out of a labels or descriptions block.

    Args:
        block: The entity's ``labels`` or ``descriptions`` mapping.

    Returns:
        The English string, or an empty string.
    """
    if not isinstance(block, dict):
        return ""
    english = block.get("en")
    if not isinstance(english, dict):
        return ""
    return str(english.get("value") or "")


def person_candidates(payload: dict[str, Any]) -> list[Candidate]:
    """Build candidates from one crawled Wikidata payload.

    Args:
        payload: A ``wbgetentities`` response or a bare entity.

    Returns:
        One candidate per human entity in the payload. Non-human entities
        yield nothing.

    Note:
        Country and occupation come back as Q-ids in a raw entity dump, not
        as labels, so they are left empty here. They are tiebreaks only, and
        a bare Q-id in a field the LLM prompt renders would be noise.
    """
    candidates: list[Candidate] = []

    for entity in _entities(payload):
        if not _is_human(entity):
            continue

        qid = str(entity.get("id") or "")
        label = _english(entity.get("labels"))
        if not qid or not label:
            continue

        aliases: list[str] = []
        alias_block = entity.get("aliases")
        if isinstance(alias_block, dict):
            for item in alias_block.get("en", []) or []:
                if isinstance(item, dict) and item.get("value"):
                    aliases.append(str(item["value"]))

        wikipedia_url = ""
        sitelinks = entity.get("sitelinks")
        if isinstance(sitelinks, dict):
            enwiki = sitelinks.get("enwiki")
            if isinstance(enwiki, dict):
                wikipedia_url = str(enwiki.get("url") or "")

        candidates.append(
            Candidate(
                qid=qid,
                label=label,
                description=_english(entity.get("descriptions")),
                aliases=tuple(aliases),
                birth_year=_claim_year(entity, _P_BIRTH),
                death_year=_claim_year(entity, _P_DEATH),
                wikipedia_url=wikipedia_url,
            )
        )

    return candidates


class LocalCandidateSource:
    """Candidates indexed from Wikidata entity JSON already on disk.

    Attributes:
        directory: The ``data/raw/wikidata`` directory.
    """

    def __init__(self, directory: Path) -> None:
        """Index every human entity in a directory.

        Args:
            directory: Where the crawl stage wrote entity JSON. A missing
                directory indexes nothing and logs, rather than raising: a
                corpus crawled without person entities is a thin run, not a
                broken one.
        """
        self.directory = directory
        self._by_key: dict[str, list[Candidate]] = {}
        self._load()

    def _load(self) -> None:
        """Read and index the directory's entity files."""
        if not self.directory.is_dir():
            logger.warning("local_candidates_directory_missing", path=str(self.directory))
            return

        people = 0
        for path in sorted(self.directory.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("wikidata_entity_unreadable", path=str(path), error=str(exc))
                continue
            if not isinstance(payload, dict):
                continue
            for candidate in person_candidates(payload):
                people += 1
                for key in candidate_keys(candidate):
                    self._by_key.setdefault(key, []).append(candidate)

        logger.info(
            "local_candidates_indexed",
            path=str(self.directory),
            people=people,
            keys=len(self._by_key),
        )

    def prefetch(self, groups: Sequence[MentionGroup]) -> None:
        """Do nothing; the index is built at construction.

        Args:
            groups: Ignored.
        """
        return None

    def search(self, group: MentionGroup) -> list[Candidate]:
        """Look a group's keys up in the index.

        Args:
            group: The mention group.

        Returns:
            Every indexed candidate matching any of the group's keys,
            deduplicated by Q-id.
        """
        found: dict[str, Candidate] = {}
        for key in group.keys:
            for candidate in self._by_key.get(key, []):
                found.setdefault(candidate.qid, candidate)
        return list(found.values())


def _sparql_literal(value: str) -> str:
    """Render a name as an English-tagged SPARQL string literal.

    Args:
        value: A surface form.

    Returns:
        The literal, e.g. ``'"Horatio Nelson"@en'``, with quotes, backslashes
        and newlines escaped so a name cannot terminate the literal early.
    """
    escaped = value
    for char, replacement in _SPARQL_ESCAPES:
        escaped = escaped.replace(char, replacement)
    return f'"{escaped}"@en'


def _iso_to_literal(value: str) -> str:
    """Convert a SPARQL xsd:dateTime into the mapper's date literal form.

    Args:
        value: A value such as ``"-0031-09-02T00:00:00Z"``, or an empty
            string when the binding was absent.

    Returns:
        A literal :func:`astronomical_year` understands, or an empty string.
    """
    raw = value.strip()
    if not raw:
        return ""

    # The query service writes an AD year unsigned ("1758-09-29T00:00:00Z")
    # while Wikidata's entity dumps sign it ("+1769-05-01T00:00:00Z"). Accept
    # either: an unhandled "+" makes the year fail isdigit() below, and the
    # candidate then silently loses the lifespan the date gate depends on.
    negative = raw.startswith("-")
    body = raw[1:] if raw[0] in "+-" else raw
    date_part = body.split("T", 1)[0]
    if date_part.count("-") != 2:
        return ""

    year, month, day = date_part.split("-")
    if not year.isdigit():
        return ""

    # The query service zeroes months and days it does not know; the year is
    # the only part the lifespan gate reads, so the rest is made valid.
    month = month if month.isdigit() and month != "00" else "01"
    day = day if day.isdigit() and day != "00" else "01"
    literal = f"{int(year):04d}-{month}-{day}"
    return f"{literal} BC" if negative else literal


def _merge_rows(rows: Iterable[dict[str, str]]) -> list[tuple[Candidate, set[str]]]:
    """Fold SPARQL result rows into one candidate per person.

    One person yields a row per country-occupation combination, so the rows
    have to be folded rather than read one to one.

    Args:
        rows: Flattened bindings from :func:`parse_sparql_bindings`.

    Returns:
        One (candidate, matched names) pair per distinct person, in first-seen
        order. The matched names are the requested strings that actually hit
        this entity's label or one of its aliases, and they are what the
        caller indexes on -- recomputing the match from the label alone loses
        every entity found through ``skos:altLabel``.

        Those names also become the candidate's ``aliases``, which is how the
        matcher gets an exact-match key for a commander whose Wikidata label
        is longer than the name an infobox writes.
    """
    merged: dict[str, dict[str, Any]] = {}

    for row in rows:
        qid = qid_from_uri(row.get("person", ""))
        if qid is None:
            continue
        entry = merged.setdefault(
            qid,
            {
                "label": row.get("personLabel", "") or qid,
                "description": row.get("personDescription", ""),
                "birth": astronomical_year(_iso_to_literal(row.get("birth", ""))),
                "death": astronomical_year(_iso_to_literal(row.get("death", ""))),
                "countries": [],
                "occupations": [],
                "article": row.get("article", ""),
                "names": set(),
            },
        )
        # GROUP_CONCAT already collapsed these, so a row carries the whole
        # set as one comma-joined string. Several rows per person can still
        # arrive when more than one requested name matched them.
        for field, value in (
            ("countries", row.get("countries", "")),
            ("occupations", row.get("occupations", "")),
        ):
            for item in (part.strip() for part in value.split(",")):
                if item and item not in entry[field]:
                    entry[field].append(item)
        matched = row.get("name", "")
        if matched:
            entry["names"].add(matched)

    results: list[tuple[Candidate, set[str]]] = []
    for qid, entry in merged.items():
        label = str(entry["label"])
        names: set[str] = entry["names"]
        results.append(
            (
                Candidate(
                    qid=qid,
                    label=label,
                    description=str(entry["description"]),
                    aliases=tuple(sorted(n for n in names if n != label)),
                    birth_year=entry["birth"],
                    death_year=entry["death"],
                    country=", ".join(entry["countries"]),
                    occupations=tuple(entry["occupations"]),
                    wikipedia_url=str(entry["article"]),
                ),
                names,
            )
        )
    return results


DEFAULT_MAX_NAMES_PER_GROUP: Final[int] = 3


def query_names(group: MentionGroup, limit: int = DEFAULT_MAX_NAMES_PER_GROUP) -> list[str]:
    """Choose which of a group's surface forms to ask Wikidata about.

    Args:
        group: The mention group.
        limit: How many forms to keep.

    Returns:
        The display name first, then the other forms, capped. The full set is
        mostly near-duplicates and each one widens the VALUES block.
    """
    unique: list[str] = []
    for name in (group.display_name, *group.surface_forms):
        if name and name not in unique:
            unique.append(name)
    return unique[:limit]


def collect_query_names(
    groups: Sequence[MentionGroup],
    limit: int = DEFAULT_MAX_NAMES_PER_GROUP,
) -> list[str]:
    """Gather every name a run needs to ask about, once each.

    Args:
        groups: Every group the run will resolve.
        limit: Forms per group.

    Returns:
        Distinct names in first-seen order.
    """
    wanted: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for name in query_names(group, limit):
            if name not in seen:
                seen.add(name)
                wanted.append(name)
    return wanted


async def fetch_candidates(
    fetcher: Fetcher,
    names: Sequence[str],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, list[Candidate]]:
    """Ask the query service about many names, in batches.

    One coroutine for the whole run, so the caller opens a single HTTP client
    and a single event loop. A failed batch is logged and skipped: the groups
    it covered then find no candidates and resolve as new entities, which is
    wrong but recoverable on a re-run, and better than aborting a stage that
    has already resolved thousands of mentions.

    Args:
        fetcher: A configured fetcher, supplying rate limiting and retries.
            Handover 5.2 applies here too: the query service is stricter than
            the article API, not laxer.
        names: The surface forms to ask about.
        batch_size: Names per query.

    Returns:
        Each requested name mapped to its candidates. A name that was asked
        about and matched nothing maps to an empty list, which is different
        from a name that was never asked about at all.

    Raises:
        CandidateLookupError: If every batch failed. One failed batch among
            several is a bad query or a blip and degrades quietly; all of
            them failing is a configuration problem, and continuing would
            resolve the whole corpus as new entities.
    """
    by_name: dict[str, list[Candidate]] = {}
    size = max(1, batch_size)

    if not names:
        return by_name

    logger.info(
        "sparql_candidate_prefetch_start",
        names=len(names),
        batches=(len(names) + size - 1) // size,
    )

    batches = 0
    failed = 0
    last_error = ""

    for start in range(0, len(names), size):
        batches += 1
        batch = list(names[start : start + size])
        for name in batch:
            by_name.setdefault(name, [])

        query = _LABEL_QUERY.format(names=" ".join(_sparql_literal(n) for n in batch))
        result = await run_query(fetcher, query)

        if not result.ok or not result.text:
            failed += 1
            last_error = result.error or f"HTTP {result.status}"
            logger.warning(
                "sparql_candidate_batch_failed",
                names=len(batch),
                status=result.status,
                error=result.error,
            )
            continue

        candidates = _merge_rows(parse_sparql_bindings(result.text))
        indexed = 0
        requested = {fold(name): name for name in batch}
        for candidate, matched_names in candidates:
            # Index on the names the endpoint reported matching. Deriving it
            # from the label instead drops every alias match, which is most
            # of the commanders known by a title.
            targets = {
                requested[folded]
                for folded in (fold(n) for n in matched_names)
                if folded in requested
            }
            if not targets:
                targets = {
                    name for name in batch if fold(name) in set(candidate_keys(candidate))
                }
            for name in targets:
                by_name[name].append(candidate)
                indexed += 1

        logger.debug(
            "sparql_candidate_batch",
            names=len(batch),
            entities=len(candidates),
            indexed=indexed,
        )

    if batches and failed == batches:
        logger.error(
            "sparql_candidate_lookup_total_failure",
            batches=batches,
            names=len(names),
            error=last_error,
        )
        raise CandidateLookupError(
            f"every one of {batches} candidate queries failed ({last_error}). "
            "Nothing would resolve to a Wikidata entity, so the run is stopping "
            "rather than writing a corpus of new entities."
        )

    matched = sum(1 for found in by_name.values() if found)
    logger.info(
        "sparql_candidate_prefetch_complete",
        names=len(by_name),
        with_candidates=matched,
        failed_batches=failed,
    )
    return by_name


class PrefetchedCandidateSource:
    """Candidates already fetched, served by name.

    The SPARQL side of the stage is one coroutine (:func:`fetch_candidates`)
    rather than a source that reaches out per group, so the whole run shares
    one HTTP client, one rate limiter and one event loop. What comes back is
    wrapped in this, which is what the rest of the stage sees.
    """

    def __init__(
        self,
        by_name: dict[str, list[Candidate]],
        *,
        max_names_per_group: int = DEFAULT_MAX_NAMES_PER_GROUP,
    ) -> None:
        """Wrap a prefetched mapping as a candidate source.

        Args:
            by_name: Surface form to candidates, from
                :func:`fetch_candidates`.
            max_names_per_group: Must match what the prefetch used, or a
                group's later forms will be looked up under keys that were
                never fetched.
        """
        self._by_name = by_name
        self._limit = max(1, max_names_per_group)

    def prefetch(self, groups: Sequence[MentionGroup]) -> None:
        """Do nothing; the fetching already happened.

        Args:
            groups: Ignored.
        """
        return None

    def search(self, group: MentionGroup) -> list[Candidate]:
        """Return the prefetched candidates for a group.

        Args:
            group: The mention group.

        Returns:
            Candidates for any of the group's queried names, deduplicated by
            Q-id.
        """
        found: dict[str, Candidate] = {}
        for name in query_names(group, self._limit):
            for candidate in self._by_name.get(name, []):
                found.setdefault(candidate.qid, candidate)
        return list(found.values())
