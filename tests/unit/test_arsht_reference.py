"""Tests for the Arsht reference-set harvester.

The decode is checked against a synthetic page rather than a captured one, so
these tests prove the parser agrees with the Bokeh wire format, not that the
live repository still serves it.

The sentinel test is the one to keep. Arsht's ``-5000`` missing-year marker
covers every battle of seven ancient generals, and a parser that let it through
as a number would date Khalid ibn al-Walid to five millennia before he lived.
"""

from __future__ import annotations

import base64
import json
import struct
from pathlib import Path

import pytest

from scripts import arsht_reference as ref


def bokeh_page(
    general: str,
    battles: list[str],
    values: list[float],
    outcomes: list[str],
    years: list[float],
) -> str:
    """Build a page carrying one serialised Bokeh data block.

    Args:
        general: The general the block belongs to.
        battles: Battle labels.
        values: Per-battle WAR.
        outcomes: ``V`` or ``D`` per battle.
        years: Year per battle, using -5000 for missing.

    Returns:
        HTML with one embedded data block, shaped as Bokeh serialises one.
    """

    def ndarray(numbers: list[float]) -> dict[str, object]:
        packed = struct.pack(f"<{len(numbers)}d", *numbers)
        return {
            "__ndarray__": base64.b64encode(packed).decode("ascii"),
            "dtype": "float64",
            "shape": [len(numbers)],
        }

    running: list[float] = []
    total = 0.0
    for value in values:
        total += value
        running.append(total)

    block = {
        "battle": battles,
        "general": [general] * len(battles),
        "outcome": outcomes,
        "value": ndarray(values),
        "cumulative": ndarray(running),
        "year": ndarray(years),
    }
    return f'<html><script>{{"data":{json.dumps(block)},"selected":null}}</script></html>'


def test_decodes_a_bokeh_page_into_battle_rows() -> None:
    """The harvester reads battles, WAR, outcome and year off one block."""
    page = bokeh_page(
        "Hannibal",
        ["Battle of the Trebia", "Battle of Cannae", "Battle of Zama"],
        [0.5, 0.6, -0.4],
        ["V", "V", "D"],
        [-218.0, -216.0, -202.0],
    )

    rows = ref.parse_page(page, "Hannibal")

    assert [row.battle for row in rows] == [
        "Battle of the Trebia",
        "Battle of Cannae",
        "Battle of Zama",
    ]
    assert [row.outcome for row in rows] == ["V", "V", "D"]
    assert [row.year for row in rows] == [-218, -216, -202]
    assert rows[-1].cumulative == pytest.approx(0.7)


def test_missing_year_sentinel_becomes_null_not_a_bc_date() -> None:
    """-5000 is Arsht's 'no year', and must never reach an era covariate."""
    page = bokeh_page("Khalid ibn al-Walid", ["Battle of Yarmouk"], [0.5], ["V"], [-5000.0])

    rows = ref.parse_page(page, "Khalid ibn al-Walid")

    assert rows[0].year is None
    assert rows[0].year_raw == -5000


def test_block_for_a_different_general_is_not_claimed() -> None:
    """A page may carry several series; the wrong one must not be taken."""
    page = bokeh_page("Trajan", ["Battle of Adamclisi"], [0.5], ["V"], [101.0])

    assert ref.parse_page(page, "Napoleon") == []


def test_page_without_a_data_block_yields_nothing() -> None:
    """A page that is not a plot is reported empty rather than raising."""
    assert ref.parse_page("<html><body>no plot here</body></html>", "Napoleon") == []


def test_unparseable_json_block_does_not_lose_the_page() -> None:
    """One malformed block must not discard a good one later in the page."""
    good = bokeh_page("Trajan", ["Battle of Adamclisi"], [0.5], ["V"], [101.0])
    page = '<script>"data":{not json at all}</script>' + good

    rows = ref.parse_page(page, "Trajan")

    assert [row.battle for row in rows] == ["Battle of Adamclisi"]


def test_unknown_ndarray_dtype_raises() -> None:
    """An unrecognised dtype is a format change, not something to guess at."""
    with pytest.raises(ValueError, match="dtype"):
        ref.decode_column_source({"__ndarray__": "AAAA", "dtype": "float16", "shape": [1]})


def test_demojibake_repairs_double_encoded_text() -> None:
    """His scrape wrote Guantánamo as GuantÃ¡namo; the repair is exact."""
    assert ref.demojibake("GuantÃ¡namo") == "Guantánamo"


def test_demojibake_leaves_undamaged_text_alone() -> None:
    """Text that was never damaged must not be mangled by the repair."""
    assert ref.demojibake("Battle of Cannae") == "Battle of Cannae"


def test_wikipedia_url_escapes_titles() -> None:
    """Titles with spaces and accents must produce a fetchable URL."""
    assert ref.wikipedia_url("Battle of the Rhône Crossing") == (
        "https://en.wikipedia.org/wiki/Battle_of_the_Rh%C3%B4ne_Crossing"
    )


def test_seed_config_refuses_unverified_titles(tmp_path: Path) -> None:
    """Writing a corpus from guessed URLs would bake redirects into it."""
    battles = {"Battle of Cannae": {"battle": "Battle of Cannae", "year": -216, "generals": []}}

    with pytest.raises(ValueError, match="verify-titles"):
        ref.write_seed_config(battles, tmp_path / "seed.yaml")


def test_seed_config_collapses_battles_sharing_one_article(tmp_path: Path) -> None:
    """Two of his entries resolve to one article; the corpus holds it once."""
    url = "https://en.wikipedia.org/wiki/Battle_of_Philippi"
    battles = {
        "First Battle of Philippi": {
            "battle": "First Battle of Philippi",
            "year": -42,
            "generals": ["Augustus"],
            "url": url,
            "exists": True,
        },
        "Second Battle of Philippi": {
            "battle": "Second Battle of Philippi",
            "year": -42,
            "generals": ["Marcus Vipsanius Agrippa"],
            "url": url,
            "exists": True,
        },
    }
    path = tmp_path / "seed.yaml"

    written = ref.write_seed_config(battles, path)

    assert written == 1
    assert path.read_text(encoding="utf-8").count(url) == 1


def _load(name: str) -> list[dict[str, object]]:
    """Read one committed reference file.

    Args:
        name: The file's name inside the reference directory.

    Returns:
        Its rows.
    """
    text = (ref.OUT_DIR / name).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_reference_set_career_totals_match_its_battle_rows() -> None:
    """The committed set parses, and its two views agree with each other."""
    rows = _load("arsht_war.jsonl")
    careers = _load("career_totals.jsonl")

    assert len(rows) > 200
    by_general: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_general.setdefault(str(row["general"]), []).append(row)

    for career in careers:
        general_rows = by_general[str(career["general"])]
        assert career["battles"] == len(general_rows)
        assert float(str(career["arsht_war_total"])) == pytest.approx(
            float(str(general_rows[-1]["cumulative"])), abs=1e-6
        )
        assert int(str(career["wins"])) + int(str(career["losses"])) == career["battles"]


def test_napoleon_reference_total_matches_the_published_article() -> None:
    """The article prints 16.679 for Napoleon; a decode bug would not.

    This is the one value in the set with an independent published source, so
    it is the check that the harvest read the columns it thinks it read.
    """
    careers = {str(record["general"]): record for record in _load("career_totals.jsonl")}

    assert careers["Napoleon"]["battles"] == 43
    assert float(str(careers["Napoleon"]["arsht_war_total"])) == pytest.approx(16.679, abs=0.001)


def test_ancient_generals_carry_no_year_in_the_source() -> None:
    """The missing-year gap is recorded, not silently filled.

    Seven ancient and early-medieval generals have no year on any battle. That
    is a property of the source worth failing on if it ever changes shape,
    because the era covariate depends on it.
    """
    rows = _load("arsht_war.jsonl")
    missing = {str(row["general"]) for row in rows if row["year_missing"]}

    assert "Julius Caesar" in missing
    assert "Khalid ibn al-Walid" in missing
    assert all(row["year"] is None for row in rows if row["year_missing"])


# ─── Dates: two conventions, one of which is a trap ──────────────────────────


@pytest.mark.parametrize(
    "literal, expected",
    [
        ("+1805-12-02T00:00:00Z", 1805),
        ("-0334-05-00T00:00:00Z", -334),
        ("-0031-09-02T00:00:00Z", -31),
        ("+0636-08-20T00:00:00Z", 636),
    ],
)
def test_wikidata_year_is_read_in_historical_numbering(literal: str, expected: int) -> None:
    """Wikidata negates the historical year: -0334 is 334 BC, not 333 BC.

    Granicus is the case that makes this concrete, and it is checked against
    the live endpoint in the manifest build: Wikidata gives -0334-05-00 and the
    battle is May 334 BC.
    """
    assert ref.parse_wikidata_year(literal) == expected


def test_month_precision_day_zero_does_not_break_the_parse() -> None:
    """A day of 00 is legal in Wikidata and fatal to datetime, so avoid both."""
    assert ref.parse_wikidata_year("-0334-05-00T00:00:00Z") == -334


def test_unparseable_time_literal_is_reported_as_no_year() -> None:
    """A shape we do not recognise must not become a guessed year."""
    assert ref.parse_wikidata_year("sometime in the third century") is None


@pytest.mark.parametrize(
    "historical, astronomical",
    [
        (-334, -333),  # Granicus
        (-31, -30),  # Actium, the worked example in handover §5.1
        (-1, 0),  # 1 BC is astronomical year zero
        (1, 1),
        (1805, 1805),
    ],
)
def test_astronomical_conversion_matches_the_schema_column(
    historical: int, astronomical: int
) -> None:
    """`battles.year_astronomical` inserts a year zero; both sources do not.

    Getting this wrong puts every BC battle one year off the column the
    pipeline indexes, joins and builds the era covariate from, and nothing
    downstream would notice.
    """
    assert ref.to_astronomical(historical) == astronomical


def test_year_label_reads_as_a_historian_would_write_it() -> None:
    """The seed file's comments are for humans, so BC is spelled out."""
    assert ref.year_label(-334) == "334 BC"
    assert ref.year_label(1805) == "1805"
    assert ref.year_label(None) == "year unknown"


def test_every_bc_battle_carries_both_conventions() -> None:
    """The committed corpus holds historical and astronomical years together."""
    battles = _load("battles.jsonl")
    bc = [record for record in battles if record["year"] is not None and int(record["year"]) < 0]

    assert len(bc) > 50
    for record in bc:
        assert record["year_astronomical"] == int(record["year"]) + 1


def test_granicus_is_dated_despite_arsht_having_no_year() -> None:
    """The gap that prompted this: Alexander's battles had no year at all."""
    battles = {record["battle"]: record for record in _load("battles.jsonl")}
    granicus = battles["Battle of the Granicus"]

    assert granicus["arsht_year"] is None
    assert granicus["year"] == -334
    assert granicus["year_astronomical"] == -333
    assert granicus["year_label"] == "334 BC"
    assert granicus["year_source"] == "wikidata"
    assert granicus["wikidata_qid"] == "Q192938"


def test_the_corpus_is_almost_entirely_dated_now() -> None:
    """Arsht left 59 rows undated; this set leaves one battle undated."""
    battles = _load("battles.jsonl")
    undated = [record for record in battles if record["year"] is None]

    assert len(undated) <= 2


# ─── Disambiguation pages, which pass every other check ──────────────────────


def test_no_disambiguation_page_reaches_the_corpus() -> None:
    """A disambiguation page exists and does not redirect, so it slips through.

    It has no infobox, no commanders and no date. Crawled, it yields a list of
    links recorded as a battle.
    """
    battles = _load("battles.jsonl")

    assert [r["battle"] for r in battles if r.get("is_disambiguation")] == []


@pytest.mark.parametrize(
    "arsht_title, article, year",
    [
        ("Siege of Gaza", "Siege of Gaza (332 BC)", -332),
        ("Battle of Arras", "Battle of Arras (1940)", 1940),
        ("Battle of Bautzen", "Battle of Bautzen (1813)", 1813),
        ("Battle of Burkersdorf", "Battle of Burkersdorf (1762)", 1762),
    ],
)
def test_ambiguous_titles_point_at_the_battle_actually_meant(
    arsht_title: str, article: str, year: int
) -> None:
    """Each was repointed by hand from the general and year on the row.

    Gaza is the one that shows why it matters: the bare title today offers
    Alexander's siege alongside three 21st-century ones.
    """
    battles = {str(record["battle"]): record for record in _load("battles.jsonl")}
    record = battles[arsht_title]

    assert record["canonical_title"] == article
    assert record["disambiguated_from"] == arsht_title
    assert record["is_disambiguation"] is False
    assert record["year"] == year


def test_seed_config_excludes_an_unresolved_disambiguation_page(tmp_path: Path) -> None:
    """If Wikipedia turns a title into a disambiguation page, it drops out."""
    battles = {
        "Battle of Cannae": {
            "battle": "Battle of Cannae",
            "year": -216,
            "generals": ["Hannibal"],
            "url": "https://en.wikipedia.org/wiki/Battle_of_Cannae",
            "exists": True,
            "is_disambiguation": False,
        },
        "Battle of Somewhere": {
            "battle": "Battle of Somewhere",
            "year": 1800,
            "generals": ["Nobody"],
            "url": "https://en.wikipedia.org/wiki/Battle_of_Somewhere",
            "exists": True,
            "is_disambiguation": True,
        },
    }
    path = tmp_path / "seed.yaml"

    written = ref.write_seed_config(battles, path)

    assert written == 1
    assert "Battle_of_Somewhere" not in path.read_text(encoding="utf-8")


def test_every_battle_in_the_corpus_now_has_a_year() -> None:
    """Arsht left 59 rows undated; after the Wikidata pass none are."""
    battles = _load("battles.jsonl")

    assert [record["battle"] for record in battles if record["year"] is None] == []
