"""Harvest Ethan Arsht's published per-battle WAR figures as a reference set.

The project's stated purpose is to improve on Arsht's 2018 analysis
(https://towardsdatascience.com/napoleon-was-the-best-general-ever-and-the-math-proves-it-86efed303eeb/).
That analysis shipped its results: ``ethanarsht/military_rankings`` on GitHub
carries one small HTML page per general, each embedding a Bokeh
``ColumnDataSource`` with the general's battle list, per-battle WAR, running
cumulative WAR, win/loss outcome and year. This script decodes those pages for
a curated roster and writes them to ``tests/fixtures/gold/arsht/``.

**This is a reference set, not a gold set.** Arsht's numbers are the output of
a different model over a differently-built corpus, and several of the defects
this project exists to fix are visible in them -- campaigns entered as single
battles, mojibake in article titles, and no command attribution at all. Nothing
here is an assertion about the world. What it is good for:

- a corpus roster with real coverage of the hard cases, so crawl, extract and
  resolve can be exercised on battles whose right answers are checkable
- a comparison target for the model stage: our WAR should correlate with his
  and should differ in ways we can explain. An uncorrelated result means one
  of us has a bug; an identical result means we have not changed anything.

The roster is curated, not scraped wholesale. Every entry names why it is in
the set, because a test set assembled without that reason drifts into whatever
was easy to fetch.

Usage::

    python -m scripts.arsht_reference                      # fetch and write
    python -m scripts.arsht_reference --verify-titles      # + resolve on Wikipedia
    python -m scripts.arsht_reference --cache-dir ./cache  # reuse downloads
    python -m scripts.arsht_reference --verify-titles --with-infobox --write-seed
    python -m scripts.arsht_reference --list-roster
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import re
import struct
import sys
import time
import unicodedata
import urllib.parse
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import httpx

__all__ = [
    "OUT_DIR",
    "SEED_OUT",
    "ROSTER",
    "BattleRow",
    "HarvestReport",
    "RosterEntry",
    "decode_column_source",
    "demojibake",
    "fetch_battle_dates",
    "fetch_general_page",
    "fetch_infobox_scrape",
    "fetch_wikidata_items",
    "harvest",
    "parse_page",
    "parse_wikidata_year",
    "resolve_titles",
    "to_astronomical",
    "wikipedia_url",
    "write_seed_config",
    "year_label",
]

OUT_DIR = Path("tests/fixtures/gold/arsht")
SEED_OUT = Path("config/sources_seed_arsht.yaml")

_RAW_BASE = "https://raw.githubusercontent.com/ethanarsht/military_rankings/master"
_WIKI_API = "https://en.wikipedia.org/w/api.php"
_WIKIDATA_API = "https://www.wikidata.org/w/api.php"
_USER_AGENT = "GeneralWAR/0.1 (academic research; reference-set build)"

# Arsht's pages are static files on raw.githubusercontent.com, not an API, but
# the roster is small enough that politeness costs nothing.
_FETCH_DELAY_SECONDS = 0.4

# The MediaWiki API takes up to 50 titles per query for anonymous callers.
_TITLES_PER_QUERY = 50

# His CSV cells hold whole infobox paragraphs, well past csv's default limit.
_CSV_FIELD_LIMIT = 10_000_000

# Wikidata rate-limits batched entity reads; these mirror the crawl stage's
# retry policy, which CLAUDE.md fixes at 3 retries from a 2s base.
_MAX_RETRIES = 3
_BACKOFF_BASE_SECONDS = 2.0

# Wikidata answers 429 far more readily at 50 ids per call than the MediaWiki
# title API does, so entity reads go in smaller batches.
_ENTITIES_PER_QUERY = 25

# Wikidata's time precision scale: 11 is a day, 10 a month, 9 a year, 8 a
# decade, 7 a century. Below 9 the year digits are padding, not a year.
_MIN_YEAR_PRECISION = 9

# Arsht's year column carries -5000 where his pipeline found no year, and it is
# not a rare case: it covers every battle of seven generals in this roster,
# Caesar, Alexander, Augustus, Agrippa, Themistocles, Belisarius and Khalid
# among them. Carried through as a number it would place Khalid ibn al-Walid,
# who died in 642 AD, five millennia BC, and an era covariate read straight off
# it would be wrong for precisely the ancient commanders whose era matters
# most. It is recorded here as missing, with the raw value kept beside it.
_YEAR_MISSING_SENTINEL = -5000

# Four of Arsht's battle titles are bare enough that Wikipedia now answers them
# with a disambiguation page: a page that exists, does not redirect, and holds
# nothing but links. Crawled, each would have yielded an article with no
# infobox, no commanders and no date. "Siege of Gaza" is the sharpest case --
# the title today offers Alexander's siege alongside three 21st-century ones.
#
# Each target below was chosen by hand from the disambiguation page itself,
# using the general and year the row already carries, and each is recorded so
# the choice can be argued with. This is deliberately a table and not a
# heuristic: picking the nearest year automatically would be right four times
# here and wrong silently somewhere else.
_DISAMBIGUATION_TARGETS: dict[str, str] = {
    # Alexander the Great, 332 BC, on the way to Egypt after Tyre.
    "Siege of Gaza": "Siege of Gaza (332 BC)",
    # Rommel, 1940: the British counterattack at Arras. The page also offers
    # sieges in 1640 and 1654 and battles in 1914, 1915, 1917 and 1918.
    "Battle of Arras": "Battle of Arras (1940)",
    # Napoleon, 1813, against the Russo-Prussian army; not the 1945 battle.
    "Battle of Bautzen": "Battle of Bautzen (1813)",
    # Frederick the Great, 1762, in the Seven Years' War; not 1866.
    "Battle of Burkersdorf": "Battle of Burkersdorf (1762)",
}

_NDARRAY_FORMATS: dict[str, tuple[str, int]] = {
    "float64": ("d", 8),
    "float32": ("f", 4),
    "int64": ("q", 8),
    "int32": ("i", 4),
}


@dataclass(frozen=True)
class RosterEntry:
    """One general in the reference roster.

    Attributes:
        name: The general's name exactly as Arsht's repository spells it. The
            page filename is this plus ``.html``; a mismatch fetches nothing,
            which is why misses are reported rather than skipped.
        reason: Why this general is in the set. Coverage is the point of a
            test set, so an entry that cannot say what it covers does not
            belong in one.
    """

    name: str
    reason: str


# Two groups, kept separate on purpose.
#
# The first is everyone the article itself names, with the scores it quotes.
# Those are the figures a reader can check us against without downloading
# anything, so a divergence there is the one that needs an explanation.
#
# The second is chosen for the failure modes this project's own handover
# records: naval battles the infobox parser mislabelled (§4.2), BC dates that
# cannot round-trip through Python's date type (§5.1), shared command that the
# battle_commanders attribution columns exist to model, and names whose
# Wikidata label has migrated to `mul` (§16.1).
ROSTER: tuple[RosterEntry, ...] = (
    # --- named in the article ---
    RosterEntry("Napoleon", "article's headline result, WAR 16.679, 43 battles"),
    RosterEntry("Julius Caesar", "article's 2nd place, WAR 7.445"),
    RosterEntry("Hannibal", "article's 6th place, WAR 5.519; Cannae is its worked example"),
    RosterEntry("Alexander the Great", "article's 10th place, WAR 4.391"),
    RosterEntry("Robert E. Lee", "article's negative-WAR case, -1.89; 2nd most battles at 27"),
    RosterEntry("Erwin Rommel", "article's negative-WAR case, -1.953"),
    RosterEntry("George S. Patton", "article quotes WAR 0.9"),
    RosterEntry("Pyrrhus of Epirus", "article quotes WAR -0.53; the eponymous victory problem"),
    RosterEntry("Moshe Dayan", "article quotes WAR 2.109, 60th; modern combined-arms"),
    RosterEntry("Ariel Sharon", "article quotes WAR 2.171, 58th"),
    # --- chosen for coverage of this project's known failure modes ---
    RosterEntry("Yi Sun-sin", "naval-only career; battle_type must not default to field (§4.2)"),
    RosterEntry("Themistocles", "naval, BC dates, and a genuinely ambiguous Wikidata match"),
    RosterEntry("Scipio Africanus", "BC dates; Zama pairs him against Hannibal in one battle"),
    RosterEntry("Augustus", "Actium: nominal command with Agrippa commanding tactically"),
    RosterEntry("Marcus Vipsanius Agrippa", "the other half of the Octavian/Agrippa problem"),
    RosterEntry("Trajan", "BC/AD boundary era, small battle count, ancient troop inflation"),
    RosterEntry("Ulysses S. Grant", "Lee's opponent; the pair tests relative skill directly"),
    RosterEntry("Georgy Zhukov", "Rommel's era; vast troop numbers, Soviet source disagreement"),
    RosterEntry("Khalid ibn al-Walid", "non-European corpus; transliterated name variants"),
    RosterEntry("Subutai", "Mongol corpus; commanded under Genghis Khan, so hierarchy matters"),
    RosterEntry("Frederick the Great", "18th century; head of state and field commander at once"),
    RosterEntry("Belisarius", "Byzantine; frequently outnumbered, so force ratio matters"),
)


@dataclass(frozen=True)
class BattleRow:
    """One (general, battle) pair as Arsht's published figures give it.

    Attributes:
        general: The general, as his repository spells the name.
        battle: The battle's display name, as his plot labels it.
        year: The year his dataset assigns, negative for BC, or None where his
            missing-year sentinel stood.
        year_raw: The sentinel value itself where one was present, so the
            missingness stays auditable rather than becoming a silent null.
        outcome: ``V`` for victory or ``D`` for defeat, his coding.
        war: The WAR his model credits this general for this battle.
        cumulative: His running career total after this battle.
    """

    general: str
    battle: str
    year: int | None
    outcome: str
    war: float
    cumulative: float
    year_raw: int | None = None


@dataclass
class HarvestReport:
    """What a harvest run found and failed to find.

    Attributes:
        rows: Every parsed battle row.
        missing: Roster names whose page did not exist.
        empty: Roster names whose page held no decodable battle data.
    """

    rows: list[BattleRow] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    empty: list[str] = field(default_factory=list)


def decode_column_source(value: Any) -> Any:
    """Decode a Bokeh column, which may be a list or a base64 ndarray.

    Bokeh serialises numeric columns as ``{"__ndarray__": "<base64>",
    "dtype": "float64", "shape": [n]}`` and string columns as plain lists.
    Decoding uses ``struct`` rather than numpy so the reference set can be
    rebuilt without the modelling dependencies installed.

    Args:
        value: A raw column value from the serialised data block.

    Returns:
        A list of Python scalars, or the value unchanged if it is not a
        recognised ndarray wrapper.

    Raises:
        ValueError: If the wrapper names a dtype this function cannot decode.
    """
    if not isinstance(value, dict) or "__ndarray__" not in value:
        return value

    dtype = str(value.get("dtype", ""))
    if dtype not in _NDARRAY_FORMATS:
        raise ValueError(f"Unsupported Bokeh ndarray dtype: {dtype!r}")
    code, size = _NDARRAY_FORMATS[dtype]
    raw = base64.b64decode(str(value["__ndarray__"]))
    return list(struct.unpack(f"<{len(raw) // size}{code}", raw))


def _iter_data_blocks(html: str) -> list[dict[str, Any]]:
    """Pull every serialised Bokeh data block out of a page.

    A regex cannot do this: the blocks nest braces. The scan finds each
    ``"data":{`` and walks forward counting depth.

    Args:
        html: The page source.

    Returns:
        Every block that parsed as JSON, in document order. Blocks that do not
        parse are skipped rather than raising, because a page may carry
        unrelated serialised objects and one bad block should not lose the
        page.
    """
    blocks: list[dict[str, Any]] = []
    cursor = 0
    marker = '"data":{'
    while True:
        start = html.find(marker, cursor)
        if start < 0:
            return blocks
        open_brace = start + len('"data":')
        depth = 0
        end = -1
        for index in range(open_brace, len(html)):
            char = html[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = index
                    break
        if end < 0:
            return blocks
        cursor = end
        try:
            parsed = json.loads(html[open_brace : end + 1])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            blocks.append(parsed)


def _fold(name: str) -> str:
    """Normalise a name for comparison across mojibake and accent differences.

    Arsht's pipeline mangled some non-ASCII titles on the way through, so a
    byte-for-byte comparison of names would drop real matches.

    Args:
        name: The name to fold.

    Returns:
        A lowercase, accent-stripped, whitespace-collapsed form.
    """
    decomposed = unicodedata.normalize("NFKD", name)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", stripped).strip().lower()


def parse_page(html: str, expected_general: str) -> list[BattleRow]:
    """Extract the battle rows a general's page carries.

    Args:
        html: The page source.
        expected_general: The roster name, used to pick the right block when a
            page carries several generals' series.

    Returns:
        One row per battle, in the page's own order, which is chronological.
        Empty if the page has no block for this general.
    """
    wanted = _fold(expected_general)
    for block in _iter_data_blocks(html):
        if not {"battle", "value", "cumulative", "outcome"} <= set(block):
            continue
        generals = block.get("general") or []
        if generals and _fold(str(generals[0])) != wanted:
            continue

        battles = [str(name) for name in decode_column_source(block["battle"])]
        values = [float(v) for v in decode_column_source(block["value"])]
        cumulative = [float(v) for v in decode_column_source(block["cumulative"])]
        outcomes = [str(o) for o in decode_column_source(block["outcome"])]
        years_raw = decode_column_source(block.get("year", []))

        rows: list[BattleRow] = []
        for index, battle in enumerate(battles):
            year_value = years_raw[index] if index < len(years_raw) else None
            year = int(year_value) if isinstance(year_value, (int, float)) else None
            sentinel = year if year == _YEAR_MISSING_SENTINEL else None
            rows.append(
                BattleRow(
                    general=expected_general,
                    battle=battle,
                    year=None if sentinel is not None else year,
                    year_raw=sentinel,
                    outcome=outcomes[index],
                    war=values[index],
                    cumulative=cumulative[index],
                )
            )
        return rows
    return []


def _safe_filename(name: str) -> str:
    """Make a filename that survives Windows' reserved characters.

    Args:
        name: The general's name.

    Returns:
        The name with path-hostile characters replaced.
    """
    return re.sub(r'[<>:"/\\|?*]', "_", name)


def fetch_general_page(client: httpx.Client, name: str, cache_dir: Path | None) -> str | None:
    """Fetch one general's page, using the cache if it holds it.

    Args:
        client: An open HTTP client.
        name: The general's name as the repository spells it.
        cache_dir: Where to read and write cached pages, or None to always
            fetch.

    Returns:
        The page source, or None if the repository has no page for this name.
    """
    cached = None if cache_dir is None else cache_dir / f"{_safe_filename(name)}.html"
    if cached is not None and cached.exists():
        return cached.read_text(encoding="utf-8", errors="replace")

    url = f"{_RAW_BASE}/{urllib.parse.quote(f'{name}.html')}"
    response = client.get(url)
    time.sleep(_FETCH_DELAY_SECONDS)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    text = response.text
    if cached is not None:
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(text, encoding="utf-8")
    return text


def wikipedia_url(battle: str) -> str:
    """Build the Wikipedia article URL a battle name implies.

    This is the naive mapping, and it is meant to be. Where it is wrong --
    ``Battle of Gallipoli`` redirecting to ``Gallipoli campaign``, a campaign
    entered as a battle -- that is a finding about the corpus, not a bug to
    paper over, so ``--verify-titles`` records the redirect rather than this
    function guessing.

    Args:
        battle: The battle's display name.

    Returns:
        An absolute en.wikipedia.org article URL.
    """
    return f"https://en.wikipedia.org/wiki/{urllib.parse.quote(battle.replace(' ', '_'))}"


def resolve_titles(client: httpx.Client, battles: list[str]) -> dict[str, dict[str, Any]]:
    """Ask Wikipedia what each battle title actually resolves to.

    Args:
        client: An open HTTP client.
        battles: Battle display names.

    A title landing on a disambiguation page is the case worth catching. Such a
    page exists, does not redirect, and has no infobox, no commanders and no
    date, so every check short of asking passes it and the crawl fetches a list
    of links as though it were a battle. Four of Arsht's titles do this. Where
    ``_DISAMBIGUATION_TARGETS`` names the article he meant, that article is
    queried instead; anything still landing on a disambiguation page is flagged
    and kept out of the corpus.

    Args:
        client: An open HTTP client.
        battles: Battle display names.

    Returns:
        A mapping from the queried name to a record holding the canonical
        title, whether the query was redirected or disambiguated, and whether
        the page exists.
    """
    resolved: dict[str, dict[str, Any]] = {}
    for start in range(0, len(battles), _TITLES_PER_QUERY):
        chunk = battles[start : start + _TITLES_PER_QUERY]
        asked = {name: _DISAMBIGUATION_TARGETS.get(name, name.replace("_", " ")) for name in chunk}
        response = _get_with_retry(
            client,
            _WIKI_API,
            {
                "action": "query",
                "titles": "|".join(asked.values()),
                "prop": "pageprops",
                "redirects": "1",
                "format": "json",
                "formatversion": "2",
            },
        )
        payload = response.json().get("query", {})

        redirects = {r["from"]: r["to"] for r in payload.get("redirects", [])}
        normalised = {n["from"]: n["to"] for n in payload.get("normalized", [])}
        existing = {p["title"] for p in payload.get("pages", []) if not p.get("missing")}
        disambiguations = {
            page["title"]
            for page in payload.get("pages", [])
            if "disambiguation" in (page.get("pageprops") or {})
        }

        for name in chunk:
            queried = asked[name]
            canonical = normalised.get(queried, queried)
            canonical = redirects.get(canonical, canonical)
            resolved[name] = {
                "queried_title": queried,
                "canonical_title": canonical,
                "redirected": canonical != name.replace("_", " "),
                "exists": canonical in existing,
                "is_disambiguation": canonical in disambiguations,
                "disambiguated_from": name if name in _DISAMBIGUATION_TARGETS else None,
                "url": wikipedia_url(canonical),
            }
    return resolved


def demojibake(text: str) -> str:
    """Undo the UTF-8-read-as-Latin-1 damage in Arsht's CSVs where it is safe.

    His scrape wrote ``Guantánamo`` as ``GuantÃ¡namo``. The repair is exact
    when the round trip succeeds and a no-op when it does not, so a string that
    was never damaged is returned unchanged rather than mangled a second time.

    Args:
        text: Possibly damaged text.

    Returns:
        The repaired text, or the input if repair is not possible.
    """
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text


def fetch_infobox_scrape(
    client: httpx.Client, battles: set[str], cache_dir: Path | None
) -> list[dict[str, Any]]:
    """Pull Arsht's own infobox parse for the roster battles.

    ``final_vd_fill.csv`` in his repository is one row per battle-commander
    pair, carrying the raw ``strength`` text from each side of the infobox
    alongside the numbers he parsed out of it. That makes it the one thing in
    this reference set that the extract stage can be compared against directly:
    same source article, same field, a different parser.

    His ``Ships`` column matters more than its size suggests. It is the only
    machine-readable naval signal available without re-reading every article,
    and this project mislabelled every naval battle as a land engagement once
    already (handover §4.2).

    Args:
        client: An open HTTP client.
        battles: Battle display names to keep, folded for comparison here.
        cache_dir: Where to cache the download, or None to always fetch.

    Returns:
        One record per matching battle-commander row.
    """
    cached = None if cache_dir is None else cache_dir / "final_vd_fill.csv"
    if cached is not None and cached.exists():
        raw = cached.read_text(encoding="utf-8", errors="replace")
    else:
        response = client.get(f"{_RAW_BASE}/final_vd_fill.csv")
        response.raise_for_status()
        raw = response.text
        if cached is not None:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_text(raw, encoding="utf-8")

    wanted = {_fold(name) for name in battles}
    csv.field_size_limit(_CSV_FIELD_LIMIT)
    rows: list[dict[str, Any]] = []
    for row in csv.DictReader(io.StringIO(raw)):
        title = demojibake(row.get("Battle") or "").replace("_", " ")
        if _fold(title) not in wanted:
            continue
        rows.append(
            {
                "battle": title,
                "commander": demojibake(row.get("belligerent") or "").strip(),
                "side": (row.get("pos") or "").strip(),
                "outcome": (row.get("VorD") or "").strip(),
                "result_text": _collapse(demojibake(row.get("Result") or "")),
                "own_strength_text": _collapse(demojibake(row.get("own") or "")),
                "opp_strength_text": _collapse(demojibake(row.get("opp") or "")),
                "date_text": _collapse(demojibake(row.get("Date") or "")),
                "location_text": _collapse(demojibake(row.get("Location") or "")),
                "parsed": {
                    field_name.lower(): _as_number(row.get(field_name))
                    for field_name in ("Infantry", "Cavalry", "Artillery", "Ships", "Airforce")
                },
            }
        )
    return rows


def _collapse(text: str) -> str:
    """Flatten the embedded newlines Wikipedia infobox text carries.

    Args:
        text: Raw cell text.

    Returns:
        The text on one line, with runs of whitespace collapsed.
    """
    return re.sub(r"\s+", " ", text).strip()


def _as_number(value: str | None) -> float | None:
    """Read one of his parsed strength columns.

    Args:
        value: The raw cell.

    Returns:
        The number, or None where the cell is blank or unparseable.
    """
    if not value or not value.strip():
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _get_with_retry(client: httpx.Client, url: str, params: dict[str, str]) -> httpx.Response:
    """GET a URL, backing off when the server says to slow down.

    Wikidata answers a 50-item ``wbgetentities`` call with 429 often enough
    that a fixed delay is not enough, and an unhandled 429 loses the whole run
    a few hundred requests in. The backoff mirrors what the crawl stage already
    does: honour ``Retry-After`` when it is given, otherwise double the wait.

    Args:
        client: An open HTTP client.
        url: The endpoint.
        params: Query parameters.

    Returns:
        The successful response.

    Raises:
        httpx.HTTPStatusError: If the last attempt still failed.
    """
    delay = _BACKOFF_BASE_SECONDS
    for attempt in range(_MAX_RETRIES + 1):
        response = client.get(url, params=params)
        if response.status_code != 429:
            response.raise_for_status()
            time.sleep(_FETCH_DELAY_SECONDS)
            return response
        if attempt == _MAX_RETRIES:
            response.raise_for_status()
        retry_after = response.headers.get("Retry-After")
        wait = float(retry_after) if retry_after and retry_after.isdigit() else delay
        print(f"  rate limited, waiting {wait:.0f}s", file=sys.stderr)
        time.sleep(wait)
        delay *= 2
    raise RuntimeError("unreachable")


def parse_wikidata_year(time_value: str) -> int | None:
    """Read the year out of a Wikidata time literal.

    Wikidata writes ``-0334-05-00T00:00:00Z`` for May 334 BC: the sign is
    applied to the historical year number, and the day may be ``00`` where the
    precision is coarser than a day. Neither ``datetime`` nor ``date`` can hold
    that -- ``date.MINYEAR`` is 1, and month precision has no valid day -- so
    the year is taken with a regex and kept as an integer, which is the same
    conclusion handover §5.1 reached for this project's own dates.

    Args:
        time_value: The raw ``time`` field of a Wikidata time datavalue.

    Returns:
        The year in historical numbering, negative for BC, or None if the
        literal is not shaped as expected.
    """
    match = re.match(r"^([+-])(\d{4,})-", time_value)
    if match is None:
        return None
    year = int(match.group(2))
    return -year if match.group(1) == "-" else year


def to_astronomical(historical_year: int) -> int:
    """Convert a historical year to this project's astronomical numbering.

    Historical numbering has no year zero: 1 BC is followed by 1 AD.
    Astronomical numbering inserts one, so every BC year shifts by one while AD
    years are unchanged. 31 BC is ``-31`` historically and ``-30``
    astronomically.

    This is not a detail. ``battles.year_astronomical`` is the column the
    pipeline indexes, joins and builds the era covariate from, and both sources
    feeding this reference set -- Arsht's own data and Wikidata -- write the
    historical convention. Carried across unconverted, every BC battle in the
    corpus would sit one year off the thing it is compared against.

    Args:
        historical_year: The year as Wikidata and Arsht write it, negative for
            BC.

    Returns:
        The same year in astronomical numbering.
    """
    return historical_year + 1 if historical_year < 0 else historical_year


def year_label(historical_year: int | None) -> str:
    """Render a year the way a reader expects to see it.

    Args:
        historical_year: The year in historical numbering, or None.

    Returns:
        ``334 BC``, ``1805``, or ``year unknown``.
    """
    if historical_year is None:
        return "year unknown"
    return f"{abs(historical_year)} BC" if historical_year < 0 else str(historical_year)


def fetch_wikidata_items(client: httpx.Client, titles: list[str]) -> dict[str, str]:
    """Map Wikipedia titles to their Wikidata item ids.

    The ids are worth keeping for their own sake: the resolve stage links
    battles to canonical Wikidata ids, and these are those ids, taken from the
    sitelink rather than matched by name.

    Args:
        client: An open HTTP client.
        titles: Canonical article titles.

    Returns:
        A mapping from title to Q-id, omitting titles that have no item.
    """
    items: dict[str, str] = {}
    for start in range(0, len(titles), _TITLES_PER_QUERY):
        chunk = titles[start : start + _TITLES_PER_QUERY]
        response = _get_with_retry(
            client,
            _WIKI_API,
            {
                "action": "query",
                "titles": "|".join(chunk),
                "prop": "pageprops",
                "ppprop": "wikibase_item",
                "redirects": "1",
                "format": "json",
                "formatversion": "2",
            },
        )
        for page in response.json().get("query", {}).get("pages", []):
            qid = page.get("pageprops", {}).get("wikibase_item")
            if qid:
                items[str(page["title"])] = str(qid)
    return items


def fetch_battle_dates(client: httpx.Client, qids: list[str]) -> dict[str, dict[str, Any]]:
    """Read each battle's date from its Wikidata item.

    Prefers P585 (point in time) and falls back to P580 (start time) for the
    engagements that ran long enough to carry one. A battle with neither is
    left without a date rather than given a guessed one.

    Args:
        client: An open HTTP client.
        qids: Wikidata item ids.

    Returns:
        A mapping from Q-id to a record carrying the historical year, the
        astronomical year, the Wikidata precision and the property it came
        from.
    """
    dates: dict[str, dict[str, Any]] = {}
    for start in range(0, len(qids), _ENTITIES_PER_QUERY):
        chunk = qids[start : start + _ENTITIES_PER_QUERY]
        response = _get_with_retry(
            client,
            _WIKIDATA_API,
            {
                "action": "wbgetentities",
                "ids": "|".join(chunk),
                "props": "claims",
                "format": "json",
            },
        )
        entities = response.json().get("entities", {})

        for qid in chunk:
            claims = entities.get(qid, {}).get("claims", {})
            for prop in ("P585", "P580"):
                # Wikidata items carry competing claims, and the first in the
                # list is not the one the item means. Deprecated statements are
                # there precisely because they are wrong, and preferred ones
                # are the editors' verdict where several compete.
                statements = [
                    statement
                    for statement in (claims.get(prop) or [])
                    if statement.get("rank") != "deprecated"
                ]
                statements.sort(key=lambda s: s.get("rank") != "preferred")
                if not statements:
                    continue
                value = statements[0].get("mainsnak", {}).get("datavalue", {}).get("value", {})
                historical = parse_wikidata_year(str(value.get("time", "")))
                if historical is None:
                    continue
                # Precision below 9 means the item is dated to a decade,
                # century or millennium, and the year digits are padding. The
                # 1948 Arab-Israeli War carries P585 = +1940-00-00 at decade
                # precision, which read as a year says 1940. Fall through to
                # P580, which dates it to the day.
                if int(value.get("precision", 0)) < _MIN_YEAR_PRECISION:
                    continue
                dates[qid] = {
                    "year": historical,
                    "year_astronomical": to_astronomical(historical),
                    "precision": value.get("precision"),
                    "property": prop,
                }
                break
    return dates


def harvest(roster: tuple[RosterEntry, ...], cache_dir: Path | None) -> HarvestReport:
    """Fetch and parse every page in the roster.

    Args:
        roster: The generals to harvest.
        cache_dir: Download cache directory, or None to always fetch.

    Returns:
        The rows found, plus the roster names that produced nothing.
    """
    report = HarvestReport()
    headers = {"User-Agent": _USER_AGENT}
    with httpx.Client(headers=headers, timeout=30.0, follow_redirects=True) as client:
        for entry in roster:
            page = fetch_general_page(client, entry.name, cache_dir)
            if page is None:
                report.missing.append(entry.name)
                continue
            rows = parse_page(page, entry.name)
            if not rows:
                report.empty.append(entry.name)
                continue
            report.rows.extend(rows)
    return report


def _career_totals(rows: list[BattleRow]) -> list[dict[str, Any]]:
    """Reduce battle rows to one career total per general.

    Args:
        rows: Every harvested row.

    Returns:
        One record per general with battle counts, win rate and final WAR.
    """
    by_general: dict[str, list[BattleRow]] = {}
    for row in rows:
        by_general.setdefault(row.general, []).append(row)

    totals: list[dict[str, Any]] = []
    for general, general_rows in by_general.items():
        wins = sum(1 for row in general_rows if row.outcome.upper().startswith("V"))
        known_years = [row.year for row in general_rows if row.year is not None]
        totals.append(
            {
                "general": general,
                "battles": len(general_rows),
                "wins": wins,
                "losses": len(general_rows) - wins,
                "arsht_war_total": round(general_rows[-1].cumulative, 6),
                "arsht_war_per_battle": round(general_rows[-1].cumulative / len(general_rows), 6),
                "first_year": min(known_years) if known_years else None,
                "last_year": max(known_years) if known_years else None,
                "battles_missing_year": len(general_rows) - len(known_years),
            }
        )
    return totals


def _battle_index(
    rows: list[BattleRow],
    titles: dict[str, dict[str, Any]] | None,
    dates: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Collapse the rows to one record per distinct battle.

    Where Wikidata knows a battle's date, it wins. Arsht's year is kept beside
    it under ``arsht_year`` so a disagreement stays visible instead of being
    quietly overwritten, and a battle he had no year for -- every battle of
    Alexander, Caesar and five other ancient generals -- gets a real one.

    Args:
        rows: Every harvested row.
        titles: Resolved Wikipedia titles, or None.
        dates: Wikidata dates keyed by Q-id, or None.

    Returns:
        A mapping from battle name to its record, carrying every general the
        set has on that battle.
    """
    battles: dict[str, dict[str, Any]] = {}
    for row in rows:
        record = battles.setdefault(
            row.battle,
            {
                "battle": row.battle,
                "year": row.year,
                "generals": [],
                "url": wikipedia_url(row.battle),
            },
        )
        if row.general not in record["generals"]:
            record["generals"].append(row.general)
        # One side of a battle may carry a year where the other has the
        # sentinel, so take whichever row knows it.
        if record["year"] is None and row.year is not None:
            record["year"] = row.year
    if titles is not None:
        for name, record in battles.items():
            record.update(titles.get(name, {}))

    for record in battles.values():
        arsht_year = record.get("year")
        found = (dates or {}).get(str(record.get("wikidata_qid") or ""))
        if found is not None:
            record["year"] = found["year"]
            record["year_astronomical"] = found["year_astronomical"]
            record["date_precision"] = found["precision"]
            record["date_property"] = found["property"]
            record["year_source"] = "wikidata"
        else:
            historical = record.get("year")
            record["year_astronomical"] = (
                to_astronomical(int(historical)) if historical is not None else None
            )
            record["year_source"] = "arsht" if historical is not None else None
        record["arsht_year"] = arsht_year
        record["year_label"] = year_label(record["year"])
    return battles


def _write_outputs(
    report: HarvestReport,
    roster: tuple[RosterEntry, ...],
    out_dir: Path,
    titles: dict[str, dict[str, Any]] | None,
    infobox: list[dict[str, Any]] | None = None,
    dates: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Write the reference set and its manifest.

    Args:
        report: The harvest result.
        roster: The roster that produced it, for the manifest.
        out_dir: Destination directory.
        titles: Resolved Wikipedia titles, or None if title verification was
            not run.
        infobox: His own infobox parse for these battles, or None if it was
            not requested.
        dates: Wikidata dates keyed by Q-id, or None.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if infobox is not None:
        infobox_path = out_dir / "arsht_infobox.jsonl"
        with infobox_path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in sorted(infobox, key=lambda r: (r["battle"], r["commander"])):
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    war_path = out_dir / "arsht_war.jsonl"
    with war_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in report.rows:
            handle.write(
                json.dumps(
                    {
                        "general": row.general,
                        "battle": row.battle,
                        "year": row.year,
                        "year_missing": row.year_raw is not None,
                        "outcome": row.outcome,
                        "war": round(row.war, 6),
                        "cumulative": round(row.cumulative, 6),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    battles = _battle_index(report.rows, titles, dates)
    battles_path = out_dir / "battles.jsonl"
    with battles_path.open("w", encoding="utf-8", newline="\n") as handle:
        ordered = sorted(
            battles.values(),
            key=lambda record: (record["year"] is None, record["year"] or 0, record["battle"]),
        )
        for record in ordered:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    careers = _career_totals(report.rows)
    careers_path = out_dir / "career_totals.jsonl"
    with careers_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in sorted(careers, key=lambda r: -float(r["arsht_war_total"])):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    (out_dir / "MANIFEST.md").write_text(
        _manifest(report, roster, battles, careers, titles), encoding="utf-8"
    )


def _manifest(
    report: HarvestReport,
    roster: tuple[RosterEntry, ...],
    battles: dict[str, dict[str, Any]],
    careers: list[dict[str, Any]],
    titles: dict[str, dict[str, Any]] | None,
) -> str:
    """Render the manifest that travels with the reference set.

    Args:
        report: The harvest result.
        roster: The roster used.
        battles: The deduplicated battle records.
        careers: Career totals per general.
        titles: Resolved titles, or None.

    Returns:
        Markdown.
    """
    lines = [
        "# Arsht Reference Set",
        "",
        "Decoded from the published output of Ethan Arsht's 2018 analysis,",
        "`ethanarsht/military_rankings` on GitHub. Each general's page there embeds a",
        "Bokeh `ColumnDataSource` carrying his battle list, per-battle WAR, running",
        "cumulative WAR, outcome and year; this set is those columns, decoded.",
        "",
        "## What this is not",
        "",
        "**Not a gold set. Not ground truth.** These are one model's outputs over one",
        "corpus, and this project exists because that corpus and that model have known",
        "problems. Treat a disagreement as a question, not a failure.",
        "",
        "Specifically, do not assert against these values in a unit test as though they",
        "were facts. They are useful as:",
        "",
        "1. a roster of battles and commanders with real coverage of hard cases",
        "2. a correlation target for the model stage -- our WAR should track his and",
        "   diverge in ways we can name",
        "3. a source of corpus-hygiene findings: where his battle titles redirect to",
        "   campaigns, his corpus counted a campaign as a battle",
        "",
        "## Provenance",
        "",
        f"- Built: {date.today().isoformat()}",
        "- Source: https://github.com/ethanarsht/military_rankings (per-general HTML pages)",
        "- Article: https://towardsdatascience.com/napoleon-was-the-best-general-ever-and-the-math-proves-it-86efed303eeb/",
        "- Rebuild with: `python -m scripts.arsht_reference --verify-titles`",
        "- Reviewed by: **not yet reviewed by a human**",
        "",
        "## Contents",
        "",
        "| File | Rows | What it holds |",
        "|---|---|---|",
        f"| `arsht_war.jsonl` | {len(report.rows)} | one row per (general, battle) with his WAR |",
        f"| `battles.jsonl` | {len(battles)} | deduplicated battles with Wikipedia URLs |",
        f"| `career_totals.jsonl` | {len(careers)} | his career WAR per general |",
        "",
        "## Roster, and why each name is in it",
        "",
        "| General | Battles | Arsht WAR | Why in the set |",
        "|---|---|---|---|",
    ]

    totals_by_general = {record["general"]: record for record in careers}
    for entry in roster:
        record = totals_by_general.get(entry.name)
        if record is None:
            status = "**no page found**" if entry.name in report.missing else "**page had no data**"
            lines.append(f"| {entry.name} | — | {status} | {entry.reason} |")
        else:
            lines.append(
                f"| {entry.name} | {record['battles']} | {record['arsht_war_total']} | "
                f"{entry.reason} |"
            )

    sentinel_rows = [row for row in report.rows if row.year_raw is not None]
    sentinel_generals = sorted({row.general for row in sentinel_rows})
    lines += [
        "",
        "## Missing years in the source",
        "",
        f"{len(sentinel_rows)} of {len(report.rows)} rows carry Arsht's "
        f"`{_YEAR_MISSING_SENTINEL}` missing-year sentinel, written here as a null year "
        "with `year_missing: true`.",
        "",
        "It is not scattered noise. It covers every battle of these generals:",
        "",
    ]
    lines += [f"- {name}" for name in sentinel_generals]
    lines += [
        "",
        "All of them ancient or early medieval. Anything that read that column as a "
        "number would date Khalid ibn al-Walid, who died in 642 AD, to 5000 BC, and an "
        "era covariate built from it would be wrong for exactly the commanders whose "
        "era matters most. This project's own `missing_data_log` exists for this.",
        "",
    ]

    from_wikidata = [r for r in battles.values() if r.get("year_source") == "wikidata"]
    from_arsht = [r for r in battles.values() if r.get("year_source") == "arsht"]
    undated = [r for r in battles.values() if r.get("year") is None]
    disagreements = [
        r
        for r in battles.values()
        if r.get("arsht_year") is not None
        and r.get("year") is not None
        and r["arsht_year"] != r["year"]
    ]
    lines += [
        "",
        "## Dates",
        "",
        "Arsht's own years are kept as `arsht_year`, but the authoritative year comes",
        "from the battle's Wikidata item (P585 point in time, falling back to P580",
        "start time), reached through the article's sitelink. That fills every gap his",
        "`-5000` sentinel left, including the whole of Alexander's career.",
        "",
        f"- From Wikidata: {len(from_wikidata)}",
        f"- From Arsht, no Wikidata date: {len(from_arsht)}",
        f"- Still undated: {len(undated)}",
        "",
        "### Two conventions, and why `year_astronomical` exists",
        "",
        "Wikidata and Arsht both write **historical** years: 31 BC is `-31`. This",
        "project's `battles.year_astronomical` column writes **astronomical** years,",
        "where a year zero exists and 31 BC is `-30`. Every record therefore carries",
        "both. Join on `year_astronomical`; every BC battle in the corpus would",
        "otherwise sit one year off the column it is compared against.",
        "",
        "Statements whose Wikidata precision is coarser than a year are rejected rather",
        "than read. The 1948 Arab-Israeli War carries P585 = `+1940-00-00` at decade",
        "precision, and taken as a year that says 1940.",
        "",
    ]
    if disagreements:
        lines += [
            "### Where the two sources disagree",
            "",
            "| Battle | Arsht | Wikidata |",
            "|---|---|---|",
        ]
        lines += [
            f"| {r['battle']} | {r['arsht_year']} | {r['year']} |" for r in disagreements
        ]
        lines.append("")
    disambiguated = [r for r in battles.values() if r.get("disambiguated_from")]
    still_ambiguous = [r for r in battles.values() if r.get("is_disambiguation")]
    if disambiguated or still_ambiguous:
        lines += [
            "### Titles that landed on a disambiguation page",
            "",
            "These exist and do not redirect, so title verification alone passes them,",
            "but they carry no infobox, no commanders and no date. Each was repointed by",
            "hand using the general and year on the row; the mapping is in",
            "`_DISAMBIGUATION_TARGETS` in the harvester.",
            "",
            "| Arsht's title | Article actually meant |",
            "|---|---|",
        ]
        lines += [
            f"| {r['disambiguated_from']} | {r['canonical_title']} |" for r in disambiguated
        ]
        lines.append("")
        if still_ambiguous:
            lines += [
                "**Still ambiguous, and excluded from the corpus:**",
                "",
            ]
            lines += [f"- {r['battle']}" for r in still_ambiguous]
            lines.append("")

    if undated:
        lines += ["### Still undated", ""]
        lines += [
            f"- {r['battle']} ({r.get('wikidata_qid') or 'no Wikidata item'})" for r in undated
        ]
        lines.append("")

    lines += ["", "## Title verification", ""]
    if titles is None:
        lines += [
            "Not run. Re-run with `--verify-titles` to record which battle names",
            "resolve on Wikipedia and which redirect elsewhere.",
        ]
    else:
        missing_titles = sorted(name for name, t in titles.items() if not t["exists"])
        redirected = sorted(
            name for name, t in titles.items() if t["redirected"] and t["exists"]
        )
        lines += [
            f"- Resolved: {len(titles) - len(missing_titles)} of {len(titles)}",
            f"- Redirected to a different title: {len(redirected)}",
            f"- Not found on Wikipedia: {len(missing_titles)}",
            "",
            "Redirects matter. A battle name that redirects to a campaign article is a",
            "campaign his corpus counted as a battle, which is exactly the kind of unit",
            "error the reconcile stage's troop numbers cannot survive.",
            "",
        ]
        if redirected:
            lines += ["### Redirected", "", "| Queried | Resolves to |", "|---|---|"]
            lines += [f"| {name} | {titles[name]['canonical_title']} |" for name in redirected]
            lines.append("")
        if missing_titles:
            lines += ["### Not found", ""]
            lines += [f"- {name}" for name in missing_titles]
            lines.append("")

    lines += [
        "## How each stage uses this",
        "",
        "| Stage | Use |",
        "|---|---|",
        "| crawl | `battles.jsonl` URLs are the pilot corpus; every one must fetch 200 |",
        "| extract | commanders parsed from each infobox must include the roster general |",
        "| resolve | roster generals must link to one Wikidata id each, not several |",
        "| classify | Actium must not attribute Agrippa's tactical command to Augustus |",
        "| reconcile | ancient battles here are where source inflation shows up |",
        "| model | our career WAR should correlate with `career_totals.jsonl` |",
        "",
    ]
    return "\n".join(lines) + "\n"


def write_seed_config(battles: dict[str, dict[str, Any]], path: Path) -> int:
    """Write the curated corpus as a crawl seed file.

    Only battles whose title was verified against the live MediaWiki API are
    written, and each is written at its canonical title, so the crawl does not
    spend its budget following redirects his corpus carried. Two of his entries
    -- ``First Battle of Philippi`` and ``Second Battle of Philippi`` -- resolve
    to the same article and collapse to one URL here, which is itself the
    finding: his corpus counted one battle twice.

    The file names no list pages and no SPARQL queries. An evaluation corpus
    whose membership changes because somebody edited a Wikipedia list is not an
    evaluation corpus.

    Args:
        battles: The deduplicated battle records, title-verified.
        path: Where to write the YAML.

    Returns:
        The number of distinct article URLs written.

    Raises:
        ValueError: If no battle in the set carries a verified title, which
            means ``--verify-titles`` was not run and the URLs are guesses.
    """
    verified = [record for record in battles.values() if record.get("exists")]
    if not verified:
        raise ValueError("No verified titles: run with --verify-titles before writing a seed")

    # A disambiguation page is not a battle article. One that reached the seed
    # would crawl to a list of links, extract to nothing, and sit in the corpus
    # as a battle with no commanders on either side.
    unresolved = [record for record in verified if record.get("is_disambiguation")]
    verified = [record for record in verified if not record.get("is_disambiguation")]
    for record in unresolved:
        print(
            f"  excluded, still a disambiguation page: {record['battle']} "
            f"-> {record.get('canonical_title')}",
            file=sys.stderr,
        )

    by_url: dict[str, dict[str, Any]] = {}
    for record in sorted(
        verified, key=lambda r: (r["year"] is None, r["year"] or 0, r["battle"])
    ):
        by_url.setdefault(str(record["url"]), record)

    header = [
        "# Curated pilot corpus: the battles of the generals named in Ethan Arsht's",
        "# 2018 analysis, the work this project sets out to improve on, plus the",
        "# commanders chosen to cover this project's known failure modes -- naval",
        "# battles (handover §4.2), BC dates (§5.1) and shared command.",
        "#",
        "# Every title here was resolved against the live MediaWiki API, so these are",
        "# canonical article URLs, not the redirects his corpus carried.",
        "#",
        f"# Battles: {len(by_url)}. Generated {date.today().isoformat()}.",
        "# Regenerate with:",
        "#   python -m scripts.arsht_reference --verify-titles --write-seed",
        "#",
        "# No list pages and no SPARQL queries on purpose: the membership of an",
        "# evaluation corpus must not change because somebody edited a Wikipedia list.",
        "",
        "wikipedia_battle_articles:",
    ]
    lines = list(header)
    for url, record in by_url.items():
        when = str(record.get("year_label") or year_label(record.get("year")))
        lines.append(f"  - {url}  # {when}; {', '.join(record['generals'])}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(by_url)


def main(argv: list[str] | None = None) -> int:
    """Run the harvest from the command line.

    Args:
        argv: Argument vector, defaulting to ``sys.argv[1:]``.

    Returns:
        0 on success, 1 if the roster produced no rows at all.
    """
    parser = argparse.ArgumentParser(description="Harvest Arsht's published WAR figures")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="where to write the set")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="reuse downloaded pages from here, and write new ones into it",
    )
    parser.add_argument(
        "--verify-titles",
        action="store_true",
        help=(
            "resolve each battle title, its Wikidata item and its date "
            "(three requests per 50 battles)"
        ),
    )
    parser.add_argument(
        "--with-infobox",
        action="store_true",
        help="also pull his own infobox parse for these battles (a 4.5MB download)",
    )
    parser.add_argument(
        "--write-seed",
        nargs="?",
        const=SEED_OUT,
        default=None,
        type=Path,
        metavar="PATH",
        help=f"also write the corpus as a crawl seed file (default {SEED_OUT})",
    )
    parser.add_argument(
        "--list-roster", action="store_true", help="print the roster and exit without fetching"
    )
    args = parser.parse_args(argv)

    if args.list_roster:
        for entry in ROSTER:
            print(f"{entry.name:42} {entry.reason}")
        return 0

    report = harvest(ROSTER, args.cache_dir)
    if not report.rows:
        print("No rows harvested. The repository layout may have changed.", file=sys.stderr)
        return 1

    names = sorted({row.battle for row in report.rows})
    titles: dict[str, dict[str, Any]] | None = None
    dates: dict[str, dict[str, Any]] | None = None
    infobox: list[dict[str, Any]] | None = None
    if args.verify_titles or args.with_infobox:
        with httpx.Client(headers={"User-Agent": _USER_AGENT}, timeout=60.0) as client:
            if args.verify_titles:
                titles = resolve_titles(client, names)
                canonical = sorted(
                    {str(record["canonical_title"]) for record in titles.values()
                     if record["exists"]}
                )
                items = fetch_wikidata_items(client, canonical)
                for record in titles.values():
                    qid = items.get(str(record["canonical_title"]))
                    if qid:
                        record["wikidata_qid"] = qid
                dates = fetch_battle_dates(client, sorted(set(items.values())))
            if args.with_infobox:
                infobox = fetch_infobox_scrape(client, set(names), args.cache_dir)

    _write_outputs(report, ROSTER, args.out_dir, titles, infobox, dates)

    if args.write_seed is not None:
        try:
            written = write_seed_config(
                _battle_index(report.rows, titles, dates), args.write_seed
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(f"Wrote {args.write_seed} with {written} battle URLs")

    generals = len({row.general for row in report.rows})
    battles = len({row.battle for row in report.rows})
    print(f"Harvested {len(report.rows)} rows: {generals} generals, {battles} distinct battles")
    if report.missing:
        print(f"No page found for: {', '.join(report.missing)}")
    if report.empty:
        print(f"Page held no battle data for: {', '.join(report.empty)}")
    if titles is not None:
        missing_titles = [name for name, t in titles.items() if not t["exists"]]
        redirected = [name for name, t in titles.items() if t["redirected"] and t["exists"]]
        print(
            f"Titles: {len(titles) - len(missing_titles)}/{len(titles)} resolve, "
            f"{len(redirected)} redirect, {len(missing_titles)} missing"
        )
    print(f"Wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
