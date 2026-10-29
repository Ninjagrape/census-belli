"""Measure and freeze corpus3: corpus2 plus every battle sharing a Wikidata parent with it.

corpus2 holds the battles of Ethan Arsht's roster. Commanders outside that
roster (Montgomery, Zhukov) have only a handful of their battles in it, so
their ranking would rest on fragments of campaigns. The rule used here to
complete the campaigns is mechanical and checkable:

    corpus3 = corpus2 + every battle or siege with a Wikidata P361 ("part of")
              parent in common with a corpus2 battle.

This script reads the corpus2 database (read-only), asks Wikidata for the P361
parents of each corpus2 battle and the children of each parent, resolves the
English Wikipedia titles to canonical URLs, and writes (a) a JSON measurement
report and (b) with ``--write-seed`` a frozen seed YAML in the format of
``config/sources_seed_arsht.yaml``. The seed names URLs only: once written,
corpus membership does not change when Wikidata does.

Only Wikidata (SPARQL and the MediaWiki API on en.wikipedia.org, for title
resolution) is contacted. Etiquette: a descriptive User-Agent, one request at a
time, at most two per second, ``Retry-After`` and ``maxlag`` honoured, and
exponential backoff capped at three retries.

Usage::

    python -m scripts.campaign_seed --report path/to/report.json --write-seed
    python -m scripts.campaign_seed --report report.json            # measure only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import mwparserfromhell
import sqlalchemy as sa

__all__ = [
    "DEFAULT_DB_URL",
    "SEED_OUT",
    "Client",
    "astronomical_to_historical",
    "apply_commander_filter",
    "build_report",
    "main",
    "parse_commanders",
    "parse_wdqs_year",
    "select_shared_commander",
    "title_from_url",
    "url_from_title",
    "write_seed",
]

DEFAULT_DB_URL = "postgresql+psycopg://general_war:general_war@127.0.0.1:5432/general_war_corpus2"
SEED_OUT = Path("config/sources_seed_corpus3.yaml")

_WDQS = "https://query.wikidata.org/sparql"
_WIKI_API = "https://en.wikipedia.org/w/api.php"
_USER_AGENT = (
    f"GeneralWARBot/0.1 (https://github.com/Ninjagrape/census-belli) httpx/{httpx.__version__}"
)

_MIN_INTERVAL_SECONDS = 0.5
_MAX_RETRIES = 3
_BACKOFF_BASE_SECONDS = 2.0
_WDQS_TIMEOUT_SECONDS = 70.0
_MAXLAG_SECONDS = 5
_TITLES_PER_QUERY = 50
_PARENTS_PER_CHILD_QUERY = 10
_ITEMS_PER_VALUES = 100
_MAX_REDIRECT_HOPS = 5

_BATTLE_CLASS = "Q178561"
_SIEGE_CLASS = "Q188055"

# Parents whose P31 labels or sheer size mark them as a war or theatre rather
# than a campaign: adding them would add a whole war's battles at once.
_BROAD_TYPE_WORDS = ("war", "theatre", "theater", "front", "conflict")
_BROAD_NEW_THRESHOLDS = (50, 100)

_COMMANDER_PROPERTIES = ("P4791", "P710")  # commanded by, participant
_PARTICIPANT_IN = "P1344"

_FOCUS_GENERALS = {"Montgomery": "Q152025", "Zhukov": "Q124617", "Rommel": "Q14060"}

_DISAMBIGUATION_MARK = "disambiguation"


class WikidataError(RuntimeError):
    """A request kept failing after the allowed retries."""


class Client:
    """A single-threaded, rate-limited HTTP client that counts its requests."""

    def __init__(self) -> None:
        self._http = httpx.Client(
            headers={"User-Agent": _USER_AGENT},
            timeout=_WDQS_TIMEOUT_SECONDS,
            follow_redirects=True,
        )
        self._last_request = 0.0
        self.requests = 0
        self.retries = 0

    def close(self) -> None:
        """Close the underlying connection pool."""
        self._http.close()

    def _pace(self) -> None:
        wait = _MIN_INTERVAL_SECONDS - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def request_json(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, str] | None = None,
        data: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Send a request and return its JSON body, retrying politely.

        Honours ``Retry-After`` on 429 and 503 and on MediaWiki ``maxlag``
        errors, otherwise doubles the wait from a 2s base. At most three
        retries.

        Args:
            method: HTTP method.
            url: Endpoint.
            params: Query-string parameters.
            data: Form body (SPARQL goes by POST so large VALUES lists fit).

        Returns:
            The decoded JSON object.

        Raises:
            WikidataError: If the last attempt still failed.
        """
        delay = _BACKOFF_BASE_SECONDS
        reason = "unknown"
        for attempt in range(_MAX_RETRIES + 1):
            self._pace()
            self.requests += 1
            wait = delay
            try:
                response = self._http.request(method, url, params=params, data=data)
            except httpx.HTTPError as exc:
                reason = f"{type(exc).__name__}: {exc}"
            else:
                retry_after = response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait = float(retry_after)
                if response.status_code == 200:
                    body: dict[str, Any] = response.json()
                    error = body.get("error") if isinstance(body, dict) else None
                    if not (error and error.get("code") == "maxlag"):
                        return body
                    reason = "maxlag"
                else:
                    reason = f"HTTP {response.status_code}"
                    if response.status_code not in (429, 500, 502, 503, 504):
                        raise WikidataError(f"{url}: {reason}: {response.text[:200]}")
            if attempt == _MAX_RETRIES:
                break
            self.retries += 1
            print(f"  {reason}; waiting {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)
            delay *= 2
        raise WikidataError(f"{url}: gave up after {_MAX_RETRIES} retries ({reason})")

    def sparql(self, query: str) -> list[dict[str, str]]:
        """Run a SPARQL query and flatten the bindings to ``{var: value}`` dicts."""
        body = self.request_json("POST", _WDQS, data={"query": query, "format": "json"})
        return [
            {name: cell["value"] for name, cell in row.items()}
            for row in body["results"]["bindings"]
        ]

    def mediawiki(self, params: dict[str, str]) -> dict[str, Any]:
        """Call the en.wikipedia.org API with ``maxlag`` set."""
        full = {"format": "json", "formatversion": "2", "maxlag": str(_MAXLAG_SECONDS), **params}
        return self.request_json("GET", _WIKI_API, params=full)


def url_from_title(title: str) -> str:
    """Build the canonical en.wikipedia.org URL for an article title."""
    return f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"


def title_from_url(url: str) -> str:
    """Recover the article title from an en.wikipedia.org URL."""
    tail = url.rsplit("/wiki/", 1)[-1]
    return urllib.parse.unquote(tail).replace("_", " ")


def qid(uri: str) -> str:
    """Take the Q-id off the end of a Wikidata entity URI."""
    return uri.rsplit("/", 1)[-1]


def parse_wdqs_year(value: str) -> int | None:
    """Read the historical year (negative for BC) from a WDQS dateTime literal.

    The query service writes years in the XSD 1.1 astronomical convention
    (year 0 exists and is 1 BC), unlike the Wikidata JSON API; ``0000`` is
    therefore 1 BC and ``-0489`` is 490 BC.

    Args:
        value: A literal such as ``1944-06-06T00:00:00Z``.

    Returns:
        The year in historical numbering, or None if unparseable.
    """
    match = re.match(r"^([+-]?)(\d{4,})-", value)
    if match is None:
        return None
    year = int(match.group(2)) * (-1 if match.group(1) == "-" else 1)
    return year - 1 if year <= 0 else year


def astronomical_to_historical(year: int) -> int:
    """Convert an astronomical year (the DB column) to historical numbering."""
    return year - 1 if year <= 0 else year


def year_label(year: int | None) -> str:
    """Render a historical year as ``490 BC``, ``1805`` or ``year unknown``."""
    if year is None:
        return "year unknown"
    return f"{abs(year)} BC" if year < 0 else str(year)


def _values(prefix: str, items: list[str]) -> str:
    return " ".join(f"{prefix}:{item}" for item in items)


def _chunks(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


@dataclass
class Corpus2:
    """The corpus2 battles as read from the database."""

    rows: list[dict[str, Any]]
    commanders: dict[str, list[str]] = field(default_factory=dict)


def read_corpus2(db_url: str) -> Corpus2:
    """Read corpus2 battles and the focus generals' battles, read-only."""
    engine = sa.create_engine(db_url)
    with engine.connect() as conn:
        rows = [
            dict(r._mapping)
            for r in conn.execute(
                sa.text(
                    "SELECT battle_id, name, wikidata_id, wikipedia_url, year_astronomical "
                    "FROM battles ORDER BY battle_id"
                )
            )
        ]
        commanders: dict[str, list[str]] = {}
        for label, general_qid in _FOCUS_GENERALS.items():
            found = conn.execute(
                sa.text(
                    "SELECT DISTINCT b.name FROM battle_commanders bc "
                    "JOIN generals g ON g.general_id = bc.general_id "
                    "JOIN battles b ON b.battle_id = bc.battle_id "
                    "WHERE g.wikidata_id = :q ORDER BY b.name"
                ),
                {"q": general_qid},
            )
            commanders[label] = [r[0] for r in found]
    engine.dispose()
    return Corpus2(rows=rows, commanders=commanders)


def _follow(title: str, normalised: Mapping[str, str], redirects: Mapping[str, str]) -> str:
    """Apply the API's normalisation, then follow redirect hops (with a loop guard)."""
    current = normalised.get(title, title)
    for _ in range(_MAX_REDIRECT_HOPS):
        if current not in redirects:
            break
        current = redirects[current]
    return current


def resolve_titles(client: Client, titles: list[str]) -> dict[str, dict[str, Any]]:
    """Resolve titles through redirects; flag missing and disambiguation pages.

    Args:
        client: The rate-limited client.
        titles: Article titles as given (spaces, not underscores).

    Returns:
        ``{input_title: {title, exists, disambiguation, qid}}``.
    """
    out: dict[str, dict[str, Any]] = {}
    for chunk in _chunks(sorted(set(titles)), _TITLES_PER_QUERY):
        payload = client.mediawiki(
            {
                "action": "query",
                "titles": "|".join(chunk),
                "prop": "pageprops",
                "ppprop": "disambiguation|wikibase_item",
                "redirects": "1",
            }
        ).get("query", {})
        normalised = {n["from"]: n["to"] for n in payload.get("normalized", [])}
        redirects = {r["from"]: r["to"] for r in payload.get("redirects", [])}
        pages = {p["title"]: p for p in payload.get("pages", [])}
        for title in chunk:
            final = _follow(title, normalised, redirects)
            page = pages.get(final, {})
            props = page.get("pageprops") or {}
            out[title] = {
                "title": final,
                "exists": bool(page) and not page.get("missing"),
                "disambiguation": _DISAMBIGUATION_MARK in props,
                "qid": props.get("wikibase_item"),
            }
    return out


def fetch_parents(client: Client, battle_qids: list[str]) -> dict[str, list[str]]:
    """Map each corpus2 battle Q-id to its P361 parent Q-ids."""
    parents: dict[str, list[str]] = defaultdict(list)
    for chunk in _chunks(sorted(battle_qids), _ITEMS_PER_VALUES):
        rows = client.sparql(
            f"SELECT ?b ?p WHERE {{ VALUES ?b {{ {_values('wd', chunk)} }} ?b wdt:P361 ?p . }}"
        )
        for row in rows:
            parents[qid(row["b"])].append(qid(row["p"]))
    return {k: sorted(set(v)) for k, v in parents.items()}


def fetch_parent_info(client: Client, parent_qids: list[str]) -> dict[str, dict[str, Any]]:
    """Fetch label and P31 types for each parent."""
    info: dict[str, dict[str, Any]] = {p: {"label": p, "types": []} for p in parent_qids}
    for chunk in _chunks(sorted(parent_qids), _ITEMS_PER_VALUES):
        rows = client.sparql(
            "SELECT ?p ?pl ?t ?tl WHERE { "
            f"VALUES ?p {{ {_values('wd', chunk)} }} "
            "OPTIONAL { ?p rdfs:label ?pl FILTER(lang(?pl) = 'en') } "
            "OPTIONAL { ?p wdt:P31 ?t . ?t rdfs:label ?tl FILTER(lang(?tl) = 'en') } }"
        )
        for row in rows:
            entry = info[qid(row["p"])]
            entry["label"] = row.get("pl", entry["label"])
            if "tl" in row:
                entry["types"].append(f"{row['tl']} ({qid(row['t'])})")
    for entry in info.values():
        entry["types"] = sorted(set(entry["types"]))
    return info


def _children_query(parents: list[str]) -> str:
    return f"""
SELECT ?p ?b ?title
       (MAX(?isBattle) AS ?battle)
       (MIN(?pt) AS ?pointInTime) (MIN(?st) AS ?startTime)
       (GROUP_CONCAT(DISTINCT ?tl; separator="|") AS ?typeLabels)
WHERE {{
  VALUES ?p {{ {_values("wd", parents)} }}
  ?b wdt:P361 ?p .
  OPTIONAL {{ ?s schema:about ?b ; schema:isPartOf <https://en.wikipedia.org/> ;
              schema:name ?title . }}
  BIND(EXISTS {{ ?b wdt:P31/wdt:P279* ?c .
                VALUES ?c {{ wd:{_BATTLE_CLASS} wd:{_SIEGE_CLASS} }} }} AS ?isBattle)
  OPTIONAL {{ ?b wdt:P585 ?pt }}
  OPTIONAL {{ ?b wdt:P580 ?st }}
  OPTIONAL {{ ?b wdt:P31 ?t . ?t rdfs:label ?tl FILTER(lang(?tl) = 'en') }}
}}
GROUP BY ?p ?b ?title
"""


def fetch_children(client: Client, parent_qids: list[str]) -> list[dict[str, str]]:
    """Fetch every P361 child of each parent, with type flag, title and date.

    A query that fails (timeout on a very large parent) is split in half and
    retried, down to single parents.
    """
    rows: list[dict[str, str]] = []
    pending = _chunks(sorted(parent_qids), _PARENTS_PER_CHILD_QUERY)
    while pending:
        chunk = pending.pop(0)
        try:
            rows.extend(client.sparql(_children_query(chunk)))
        except WikidataError:
            if len(chunk) == 1:
                raise
            mid = len(chunk) // 2
            pending = [chunk[:mid], chunk[mid:], *pending]
    return rows


def fetch_focus_links(client: Client) -> dict[str, set[str]]:
    """Battles Wikidata links to each focus general by P4791, P710 or P1344."""
    found: dict[str, set[str]] = {}
    for label, general_qid in _FOCUS_GENERALS.items():
        rows = client.sparql(
            "SELECT DISTINCT ?b WHERE { "
            f"{{ ?b wdt:{_COMMANDER_PROPERTIES[0]} wd:{general_qid} }} UNION "
            f"{{ ?b wdt:{_COMMANDER_PROPERTIES[1]} wd:{general_qid} }} UNION "
            f"{{ wd:{general_qid} wdt:{_PARTICIPANT_IN} ?b }} }}"
        )
        found[label] = {qid(r["b"]) for r in rows}
    return found


def _year_of(row: dict[str, str]) -> int | None:
    for key in ("pointInTime", "startTime"):
        if key in row:
            year = parse_wdqs_year(row[key])
            if year is not None:
                return year
    return None


def _is_broad(info: dict[str, Any]) -> bool:
    labels = " ".join(info["types"]).lower()
    return any(re.search(rf"\b{word}", labels) for word in _BROAD_TYPE_WORDS)


def build_report(client: Client, db_url: str) -> dict[str, Any]:
    """Run the whole measurement and return the report dict."""
    corpus = read_corpus2(db_url)
    rows = corpus.rows

    title_by_battle = {
        r["battle_id"]: title_from_url(r["wikipedia_url"]) for r in rows if r["wikipedia_url"]
    }
    resolved2 = resolve_titles(client, list(title_by_battle.values()))
    corpus2_urls: dict[int, str] = {}
    corpus2_qids: dict[int, str] = {}
    qid_recovered = 0
    for r in rows:
        title = title_by_battle.get(r["battle_id"])
        res = resolved2.get(title) if title else None
        if res and res["exists"]:
            corpus2_urls[r["battle_id"]] = url_from_title(res["title"])
        found = r["wikidata_id"] or (res["qid"] if res else None)
        if found:
            corpus2_qids[r["battle_id"]] = found
            if not r["wikidata_id"]:
                qid_recovered += 1

    no_qid = [r["name"] for r in rows if not r["wikidata_id"]]
    no_qid_unrecovered = [r["name"] for r in rows if r["battle_id"] not in corpus2_qids]

    parents_of = fetch_parents(client, sorted(set(corpus2_qids.values())))
    no_parent = [
        r["name"]
        for r in rows
        if r["battle_id"] not in corpus2_qids or corpus2_qids[r["battle_id"]] not in parents_of
    ]
    parent_qids = sorted({p for ps in parents_of.values() for p in ps})
    parent_info = fetch_parent_info(client, parent_qids)
    child_rows = fetch_children(client, parent_qids)

    corpus2_qid_set = set(corpus2_qids.values())
    corpus2_url_set = set(corpus2_urls.values())
    qid_to_parents: dict[str, list[str]] = defaultdict(list)
    for bq, ps in parents_of.items():
        for p in ps:
            qid_to_parents[p].append(bq)

    # Per-child aggregation across parents.
    children: dict[str, dict[str, Any]] = {}
    per_parent: dict[str, dict[str, Any]] = {
        p: {
            "children": set(),
            "battle_children": set(),
            "no_sitelink": 0,
            "excluded_types": defaultdict(int),
        }
        for p in parent_qids
    }
    for row in child_rows:
        parent, child = qid(row["p"]), qid(row["b"])
        stats = per_parent[parent]
        stats["children"].add(child)
        is_battle = row.get("battle") in ("true", "1")
        if not is_battle:
            for label in (row.get("typeLabels") or "(no type)").split("|"):
                stats["excluded_types"][label or "(no type)"] += 1
            continue
        stats["battle_children"].add(child)
        entry = children.setdefault(
            child,
            {"qid": child, "title": row.get("title"), "year": _year_of(row), "parents": set()},
        )
        entry["parents"].add(parent)
        if entry["year"] is None:
            entry["year"] = _year_of(row)
    for stats in per_parent.values():
        stats["no_sitelink"] = sum(1 for c in stats["battle_children"] if not children[c]["title"])

    candidates = [c for c in children.values() if c["title"] and c["qid"] not in corpus2_qid_set]
    no_sitelink_total = sum(1 for c in children.values() if not c["title"])
    resolved_new = resolve_titles(client, [c["title"] for c in candidates])

    new_battles: dict[str, dict[str, Any]] = {}
    dropped = {
        "missing": 0,
        "disambiguation": 0,
        "already_in_corpus2_by_url": 0,
        "duplicate_url": 0,
    }
    for cand in sorted(candidates, key=lambda c: c["qid"]):
        res = resolved_new[cand["title"]]
        if not res["exists"]:
            dropped["missing"] += 1
            continue
        if res["disambiguation"]:
            dropped["disambiguation"] += 1
            continue
        url = url_from_title(res["title"])
        if url in corpus2_url_set:
            dropped["already_in_corpus2_by_url"] += 1
            continue
        if url in new_battles:
            dropped["duplicate_url"] += 1
            new_battles[url]["parents"] |= cand["parents"]
            continue
        new_battles[url] = {**cand, "url": url, "title": res["title"]}

    parent_table = []
    for p in parent_qids:
        stats = per_parent[p]
        new_here = {b["qid"] for b in new_battles.values() if p in b["parents"]}
        parent_table.append(
            {
                "qid": p,
                "label": parent_info[p]["label"],
                "types": parent_info[p]["types"],
                "broad": _is_broad(parent_info[p]) or len(new_here) > _BROAD_NEW_THRESHOLDS[0],
                "corpus2_battles": len(qid_to_parents[p]),
                "total_children": len(stats["children"]),
                "battle_children": len(stats["battle_children"]),
                "excluded_by_type": len(stats["children"]) - len(stats["battle_children"]),
                "excluded_types": dict(
                    sorted(stats["excluded_types"].items(), key=lambda kv: (-kv[1], kv[0]))
                ),
                "new_children": len(new_here),
            }
        )
    parent_table.sort(key=lambda r: (-r["new_children"], r["label"]))

    totals = {"all": len(new_battles)}
    for limit in _BROAD_NEW_THRESHOLDS:
        big = {r["qid"] for r in parent_table if r["new_children"] > limit}
        kept = {
            b["qid"] for b in new_battles.values() if not (b["parents"] and b["parents"] <= big)
        }
        totals[f"excluding_parents_over_{limit}"] = len(kept)

    focus_links = fetch_focus_links(client)
    focus = {}
    new_by_qid = {b["qid"]: b for b in new_battles.values()}
    for label, general_qid in _FOCUS_GENERALS.items():
        in_new = sorted(new_by_qid[q]["title"] for q in focus_links[label] if q in new_by_qid)
        in_c2_wd = sorted(q for q in focus_links[label] if q in corpus2_qid_set)
        focus[label] = {
            "qid": general_qid,
            "corpus2_battles_db": corpus.commanders[label],
            "corpus2_battles_linked_in_wikidata": len(in_c2_wd),
            "new_battles_linked_in_wikidata": len(in_new),
            "new_battle_titles": in_new,
        }

    dated = [b for b in new_battles.values() if b["year"] is not None]
    dates = {
        "new_total": len(new_battles),
        "with_p585_or_p580": len(dated),
        "bc": sum(1 for b in dated if b["year"] < 0),
        "ad": sum(1 for b in dated if b["year"] >= 0),
    }

    return {
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "corpus2": {
            "battles": len(rows),
            "without_wikidata_id_in_db": len(no_qid),
            "qid_recovered_via_enwiki_sitelink": qid_recovered,
            "still_without_qid": no_qid_unrecovered,
            "without_p361_parent_or_qid": len(no_parent),
            "without_p361_parent_names": no_parent,
            "with_p361_parent": len(rows) - len(no_parent),
        },
        "parents": parent_table,
        "totals": totals,
        "dropped_candidates": dropped,
        "battle_children_without_enwiki_sitelink": no_sitelink_total,
        "focus_generals": focus,
        "dates": dates,
        "_corpus2_entries": [
            {
                "url": corpus2_urls[r["battle_id"]],
                "year": astronomical_to_historical(r["year_astronomical"])
                if r["year_astronomical"] is not None
                else None,
                "parents": sorted(
                    parent_info[p]["label"]
                    for p in parents_of.get(corpus2_qids.get(r["battle_id"], ""), [])
                ),
            }
            for r in rows
            if r["battle_id"] in corpus2_urls
        ],
        "_new_entries": [
            {
                "url": b["url"],
                "year": b["year"],
                "qid": b["qid"],
                "parents": sorted(parent_info[p]["label"] for p in b["parents"]),
            }
            for b in new_battles.values()
        ],
    }


_INFOBOX_NAMES = frozenset({"infobox military conflict", "infobox battle"})
_COMMANDER_FIELD = re.compile(r"^commander[123][a-z]?$")
_FLAG_TEMPLATE = re.compile(r"^(flag|country data|noflag)")
_SKIP_NAMESPACES = ("file:", "image:", "category:", "template:", "help:", "wikipedia:")
_WIKITEXT_BATCH = 50

NO_INFOBOX = "no_infobox"
NO_NAMED_COMMANDER = "no_named_commander"
NO_SHARED_COMMANDER = "no_shared_commander"
KEPT = "kept"

_FOCUS_COMMANDERS = (
    "Bernard Montgomery",
    "Georgy Zhukov",
    "Erwin Rommel",
    "Napoleon",
    "Arthur Wellesley, 1st Duke of Wellington",
    "Ulysses S. Grant",
    "Robert E. Lee",
    "Khalid ibn al-Walid",
    "Pyrrhus of Epirus",
)


def normalise_target(target: str) -> str:
    """Normalise a wikilink target: no fragment, spaces for underscores, capitalised."""
    name = target.split("#", 1)[0].replace("_", " ")
    name = re.sub(r"\s+", " ", name).strip().lstrip(":").strip()
    return name[:1].upper() + name[1:]


def _strip_noise(value: Any) -> Any:
    """Drop flag templates and ``<ref>`` tags from a parsed field value."""
    code = mwparserfromhell.parse(str(value))
    doomed: list[Any] = [
        t for t in code.filter_templates() if _FLAG_TEMPLATE.match(str(t.name).strip().lower())
    ]
    doomed += [t for t in code.filter_tags() if str(t.tag).strip().lower() == "ref"]
    for node in doomed:
        try:
            code.remove(node)
        except ValueError:
            continue  # already removed along with its parent
    return code


def parse_commanders(wikitext: str) -> list[str] | None:
    """Read the commander wikilink targets from the first military-conflict infobox.

    Args:
        wikitext: An article's wikitext.

    Returns:
        None if the article has no such infobox, otherwise the sorted distinct
        normalised link targets in ``commander1..3`` and their lettered variants.
    """
    code = mwparserfromhell.parse(wikitext)
    for template in code.filter_templates():
        if str(template.name).strip().lower().replace("_", " ") not in _INFOBOX_NAMES:
            continue
        targets: set[str] = set()
        for param in template.params:
            if not _COMMANDER_FIELD.match(str(param.name).strip().lower()):
                continue
            for link in _strip_noise(param.value).filter_wikilinks():
                target = normalise_target(str(link.title))
                if target and not target.lower().startswith(_SKIP_NAMESPACES):
                    targets.add(target)
        return sorted(targets)
    return None


def select_shared_commander(
    corpus2: Mapping[str, frozenset[str]],
    candidates: Mapping[str, frozenset[str] | None],
) -> dict[str, str]:
    """Decide which candidates share a commander with enough kept battles.

    A candidate is kept while at least one of its commanders is listed by two or
    more kept battles (itself included). Corpus2 battles are never dropped.
    A dropped candidate's commanders were each listed once, so dropping it
    cannot lower any other count below two and the loop settles in one pass; it
    stays a loop so the contract holds if the rule is ever tightened.

    Args:
        corpus2: Commander keys per corpus2 battle (anchors, never dropped).
        candidates: Commander keys per candidate; None means no infobox.

    Returns:
        ``{candidate: reason}`` with reason one of ``kept``, ``no_infobox``,
        ``no_named_commander`` or ``no_shared_commander``.
    """
    decision: dict[str, str] = {}
    alive: dict[str, frozenset[str]] = {}
    for name, commanders in candidates.items():
        if commanders is None:
            decision[name] = NO_INFOBOX
        elif not commanders:
            decision[name] = NO_NAMED_COMMANDER
        else:
            alive[name] = commanders
    while True:
        counts: dict[str, int] = defaultdict(int)
        for commanders in [*corpus2.values(), *alive.values()]:
            for key in commanders:
                counts[key] += 1
        failing = [n for n, cs in alive.items() if not any(counts[k] >= 2 for k in cs)]
        if not failing:
            break
        for name in failing:
            decision[name] = NO_SHARED_COMMANDER
            del alive[name]
    decision.update({name: KEPT for name in alive})
    return decision


def fetch_wikitext(client: Client, titles: list[str]) -> dict[str, str | None]:
    """Fetch current wikitext for each title, 50 per request, following redirects.

    Returns:
        ``{input_title: wikitext, or None if the page is missing}``.
    """
    out: dict[str, str | None] = {}
    for chunk in _chunks(sorted(set(titles)), _WIKITEXT_BATCH):
        params = {
            "action": "query",
            "titles": "|".join(chunk),
            "prop": "revisions",
            "rvprop": "content",
            "rvslots": "main",
            "redirects": "1",
        }
        normalised: dict[str, str] = {}
        redirects: dict[str, str] = {}
        content: dict[str, str] = {}
        cont: dict[str, str] = {}
        while True:
            payload = client.mediawiki({**params, **cont})
            query = payload.get("query", {})
            normalised.update({n["from"]: n["to"] for n in query.get("normalized", [])})
            redirects.update({r["from"]: r["to"] for r in query.get("redirects", [])})
            for page in query.get("pages", []):
                if page.get("revisions"):
                    content[page["title"]] = page["revisions"][0]["slots"]["main"]["content"]
            cont = payload.get("continue") or {}
            if not cont:
                break
        for title in chunk:
            out[title] = content.get(_follow(title, normalised, redirects))
    return out


def _commander_map(
    client: Client, titles: Mapping[str, str]
) -> tuple[dict[str, frozenset[str] | None], int]:
    """Map each key to its canonical commander titles (None when no infobox).

    Returns the map and the number of distinct commander targets resolved.
    """
    wikitext = fetch_wikitext(client, list(titles.values()))
    raw: dict[str, list[str] | None] = {}
    for key, title in titles.items():
        text = wikitext[title]
        raw[key] = None if text is None else parse_commanders(text)
    targets = sorted({t for ts in raw.values() for t in ts or []})
    canonical = {t: info["title"] for t, info in resolve_titles(client, targets).items()}
    mapped = {
        key: None if ts is None else frozenset(canonical.get(t, t) for t in ts)
        for key, ts in raw.items()
    }
    return mapped, len(targets)


def _count_ge(universe: Mapping[str, frozenset[str]], minimum: int) -> int:
    counts: dict[str, int] = defaultdict(int)
    for commanders in universe.values():
        for key in commanders:
            counts[key] += 1
    return sum(1 for n in counts.values() if n >= minimum)


def _battles_of(key: str, universe: Mapping[str, frozenset[str]]) -> int:
    return sum(1 for cs in universe.values() if key in cs)


def _filter_summary(
    client: Client,
    decisions: list[dict[str, Any]],
    commanders: Mapping[str, frozenset[str] | None],
    corpus2: Mapping[str, frozenset[str]],
    kept_all: Mapping[str, frozenset[str]],
    n_targets: int,
) -> dict[str, Any]:
    focus = resolve_titles(client, list(_FOCUS_COMMANDERS))
    return {
        "pool": len(decisions),
        "by_reason": {
            r: sum(1 for d in decisions if d["decision"] == r)
            for r in (KEPT, NO_INFOBOX, NO_NAMED_COMMANDER, NO_SHARED_COMMANDER)
        },
        "kept_anchored_to_corpus2": sum(
            1 for d in decisions if d["decision"] == KEPT and d["anchored_to_corpus2"]
        ),
        "distinct_commander_targets_resolved": n_targets,
        "corpus2_articles_without_infobox": sorted(u for u in corpus2 if commanders[u] is None),
        "corpus2_articles_without_commanders": sorted(
            u for u in corpus2 if commanders[u] is not None and not commanders[u]
        ),
        "focus_commanders": {
            name: {
                "canonical": focus[name]["title"],
                "corpus2": _battles_of(focus[name]["title"], corpus2),
                "corpus3": _battles_of(focus[name]["title"], kept_all),
            }
            for name in _FOCUS_COMMANDERS
        },
        "commanders_with_3_plus_battles": {
            "corpus2": _count_ge(corpus2, 3),
            "corpus3": _count_ge(kept_all, 3),
        },
    }


def apply_commander_filter(client: Client, report: dict[str, Any]) -> dict[str, Any]:
    """Apply the shared-commander rule to a report's candidate pool.

    Args:
        client: The rate-limited client.
        report: A report holding ``_corpus2_entries`` and the candidate pool
            (``_candidates_all`` if present, else ``_new_entries``).

    Returns:
        A new report whose ``_new_entries`` holds only kept candidates, with
        ``_candidates_all``, ``candidate_decisions`` and ``commander_filter``
        added.
    """
    pool = report.get("_candidates_all") or report["_new_entries"]
    c2_entries = report["_corpus2_entries"]
    titles = {e["url"]: title_from_url(e["url"]) for e in [*c2_entries, *pool]}
    commanders, n_targets = _commander_map(client, titles)
    c2_urls = {e["url"] for e in c2_entries}
    corpus2 = {u: commanders[u] or frozenset() for u in c2_urls}
    cands = {e["url"]: commanders[e["url"]] for e in pool if e["url"] not in c2_urls}
    decision = select_shared_commander(corpus2, cands)
    c2_keys = {k for cs in corpus2.values() for k in cs}

    decisions = [
        {
            "url": e["url"],
            "decision": decision[e["url"]],
            "commanders": sorted(cands[e["url"]] or []),
            "anchored_to_corpus2": bool(c2_keys & (cands[e["url"]] or frozenset())),
            "parents": e["parents"],
        }
        for e in pool
        if e["url"] in decision
    ]
    kept_urls = {d["url"] for d in decisions if d["decision"] == KEPT}
    kept_all = {**corpus2, **{u: cands[u] or frozenset() for u in kept_urls}}
    summary = _filter_summary(client, decisions, commanders, corpus2, kept_all, n_targets)
    totals = {**report["totals"], "candidate_pool": len(pool), "kept_after_filter": len(kept_urls)}
    return {
        **report,
        "totals": totals,
        "commander_filter": summary,
        "candidate_decisions": decisions,
        "_candidates_all": pool,
        "_new_entries": [e for e in pool if e["url"] in kept_urls],
    }


def write_seed(report: dict[str, Any], path: Path) -> int:
    """Write the frozen corpus3 seed file from a report.

    Args:
        report: The dict returned by :func:`build_report`.
        path: Output YAML path.

    Returns:
        The number of distinct article URLs written.
    """
    by_url: dict[str, dict[str, Any]] = {}
    for entry in [*report["_corpus2_entries"], *report["_new_entries"]]:
        by_url.setdefault(entry["url"], entry)
    ordered = sorted(by_url.values(), key=lambda e: (e["year"] is None, e["year"] or 0, e["url"]))
    cf = report["commander_filter"]
    reasons = cf["by_reason"]
    header = [
        "# corpus3 seed: corpus2 plus the battles or sieges (Wikidata Q178561 / Q188055,",
        "# including subclasses) that share a P361 (part of) parent with a corpus2",
        "# battle AND pass the shared-commander rule, so that campaigns of commanders",
        "# outside Arsht's roster are complete without unrelated battles.",
        "#",
        "# Shared-commander rule: parse the first Infobox military conflict of each",
        "# article (current wikitext); commanders are the wikilinks in commander1..3 and",
        "# lettered variants, redirects resolved. Kept candidates have a commander listed",
        "# in at least 2 kept battles (corpus2 battles are never dropped; iterated to a",
        "# fixed point). Dropped: no infobox, no linked commander, no shared commander.",
        "#",
        "# Frozen draft. Generated by scripts/campaign_seed.py from Wikidata SPARQL and",
        "# the MediaWiki API on " + date.today().isoformat() + "; corpus2 entries come from the",
        "# corpus2 database. Titles are canonical (redirects followed, disambiguation",
        "# pages dropped). Membership does not change when Wikidata does; to change it,",
        "# regenerate deliberately:",
        "#   python -m scripts.campaign_seed --report <path> --write-seed",
        "#",
        f"# Candidate pool {cf['pool']}: kept {reasons[KEPT]}, dropped"
        f" {reasons[NO_INFOBOX]} {NO_INFOBOX}, {reasons[NO_NAMED_COMMANDER]}"
        f" {NO_NAMED_COMMANDER}, {reasons[NO_SHARED_COMMANDER]} {NO_SHARED_COMMANDER}.",
        f"# Battles: {len(ordered)} ({len(report['_corpus2_entries'])} from corpus2,"
        f" {len(report['_new_entries'])} new).",
        "",
        "wikipedia_battle_articles:",
    ]
    lines = list(header)
    for entry in ordered:
        parents = "; ".join(entry["parents"]) if entry["parents"] else "corpus2, no parent"
        lines.append(f"  - {entry['url']}  # {year_label(entry['year'])}; {parents}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(ordered)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Measure and freeze the corpus3 seed.")
    parser.add_argument("--report", type=Path, required=True, help="JSON report output path")
    parser.add_argument("--write-seed", action="store_true", help=f"write {SEED_OUT}")
    parser.add_argument("--seed-out", type=Path, default=SEED_OUT)
    parser.add_argument("--db-url", default=DEFAULT_DB_URL)
    parser.add_argument(
        "--pool",
        type=Path,
        help="reuse the corpus2 entries and candidate pool of an earlier report"
        " instead of querying Wikidata (only the commander filter is re-run)",
    )
    args = parser.parse_args(argv)

    started = time.monotonic()
    client = Client()
    try:
        if args.pool:
            report = json.loads(args.pool.read_text(encoding="utf-8"))
        else:
            report = build_report(client, args.db_url)
        report = apply_commander_filter(client, report)
    finally:
        client.close()
    report["run"] = {
        "requests": client.requests,
        "retries": client.retries,
        "seconds": round(time.monotonic() - started, 1),
        "user_agent": _USER_AGENT,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"report: {args.report}")
    if args.write_seed:
        count = write_seed(report, args.seed_out)
        print(f"seed: {args.seed_out} ({count} URLs)")
    print(f"requests: {client.requests}, seconds: {report['run']['seconds']}")
    print(f"candidate pool: {report['totals']['candidate_pool']}")
    print(f"kept new battles: {report['totals']['kept_after_filter']}")
    print(f"by reason: {report['commander_filter']['by_reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
