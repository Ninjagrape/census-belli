"""Merger guards against year-shaped non-years (archive dates) from DBpedia."""

from __future__ import annotations

import pytest

from pipeline.extractors import BattleFacts, merge_battle
from pipeline.extractors.dates import astronomical_year, title_year_span
from pipeline.extractors.records import Provenance, SourceExtraction


def _src(source_type: str, start: str | None, precision: str | None) -> SourceExtraction:
    return SourceExtraction(
        provenance=Provenance(
            source_type=source_type, extraction_method="infobox_parser", source_ref=source_type
        ),
        facts=BattleFacts(date_start=start, date_precision=precision),
    )


def test_title_helpers_read_years_ranges_and_bc() -> None:
    assert astronomical_year("0210-01-01 BC") == -209
    assert astronomical_year("1813-05-02") == 1813
    assert title_year_span("Battle of Herdonia (210 BC)") == (-209, -209)
    assert title_year_span("Siege of X (537–538)") == (537, 538)
    assert title_year_span("Battle of X (210–209 BC)") == (-209, -208)
    assert title_year_span("Battle of Cannae") is None


def test_dbpedia_archive_date_loses_to_the_infobox_herdonia() -> None:
    battle = merge_battle(
        "h",
        [_src("dbpedia", "2015-12-19", "day"), _src("wikipedia_infobox", "0210-01-01 BC", "year")],
        fallback_name="Battle of Herdonia (210 BC)",
    )

    assert battle.facts.date_start == "0210-01-01 BC"
    assert battle.facts.date_precision == "year"
    assert battle.disagreements[0].rejected["dbpedia"] == "2015-12-19"


def test_dbpedia_wrong_century_loses_to_the_infobox_day_precision_dyrrhachium() -> None:
    battle = merge_battle(
        "d",
        [_src("dbpedia", "1948-07-10", "day"), _src("wikipedia_infobox", "0048-07-10 BC", "day")],
        fallback_name="Battle of Dyrrhachium (48 BC)",
    )

    assert battle.facts.date_start == "0048-07-10 BC"
    assert battle.facts.date_precision == "day"


def test_agreement_within_a_year_keeps_the_usual_priority() -> None:
    battle = merge_battle(
        "a",
        [_src("dbpedia", "1813-10-19", "day"), _src("wikipedia_infobox", "1814-01-01", "year")],
        fallback_name="Battle of Leipzig",
    )

    assert battle.facts.date_start == "1813-10-19"
    assert battle.facts.date_precision == "day"


def test_title_year_replaces_a_contradicting_date_when_no_infobox_date_exists() -> None:
    battle = merge_battle(
        "t", [_src("dbpedia", "2015-12-19", "day")], fallback_name="Battle of Herdonia (210 BC)"
    )

    assert battle.facts.date_start == "0210-01-01 BC"
    assert battle.facts.date_precision == "year"
    assert battle.facts.date_end is None


def test_a_year_inside_a_range_title_is_accepted() -> None:
    battle = merge_battle(
        "r", [_src("wikipedia_infobox", "0537-06-01", "month")], fallback_name="Siege (537–538)"
    )

    assert battle.facts.date_start == "0537-06-01"


@pytest.mark.parametrize("title", ["Battle of Cannae", "Battle of X (1813 campaign)"])
def test_titles_without_an_explicit_year_never_override(title: str) -> None:
    battle = merge_battle("n", [_src("dbpedia", "2015-12-19", "day")], fallback_name=title)

    assert battle.facts.date_start == "2015-12-19"
