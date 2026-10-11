"""
Unit tests for infobox date parsing.

The raw strings are the ones a real 211-battle run failed on: 56 battles did
not insert because the editor-typed text was passed to a DATE column.
"""

from __future__ import annotations

import pytest

from pipeline.extractors import parse_infobox
from pipeline.extractors.dates import parse_date_text, sanitise_date_fields

# (raw text, date_start, date_end, precision)
CASES = [
    ("212 BC", "0212-01-01 BC", None, "year"),
    ("58 BC", "0058-01-01 BC", None, "year"),
    ("52 BC", "0052-01-01 BC", None, "year"),
    ("57 BC", "0057-01-01 BC", None, "year"),
    ("Spring 208 BC", "0208-01-01 BC", None, "year"),
    ("September 217 BC", "0217-09-01 BC", None, "month"),
    ("June 530 AD", "0530-06-01", None, "month"),
    ("Winter 102", "0102-01-01", None, "year"),
    ("106", "0106-01-01", None, "year"),
    ("21–22 May 1809", "1809-05-21", "1809-05-22", "day"),
    ("21-22 May 1809", "1809-05-21", "1809-05-22", "day"),
    (
        "5–6 June 1967 ( 1967-06-05 – 1967-06-06 )",
        "1967-06-05",
        "1967-06-06",
        "day",
    ),
    (
        "15 November [ O.S. 3 November ] 1812 – 18 November [ O.S. 6 November ] 1812",
        "1812-11-15",
        "1812-11-18",
        "day",
    ),
    (
        "15 November [ O.S. 3 November ] 1812 – 18 November...",
        "1812-11-15",
        "1812-11-18",
        "day",
    ),
    ("1st day, 9th month of 1592", "1592-01-01", None, "year"),
    ("January 635 AD (13 AH)", "0635-01-01", None, "month"),
    ("0216-08-02 BC", "0216-08-02 BC", None, "day"),
    ("1948-05-15", "1948-05-15", None, "day"),
    ("6 May 1757", "1757-05-06", None, "day"),
    ("September 29–30, 1864", "1864-09-29", "1864-09-30", "day"),
    ("April 6-7, 1862", "1862-04-06", "1862-04-07", "day"),
    ("June–July 634 CE", "0634-06-01", "0634-07-31", "month"),
    ("June–August 49 BC", "0049-06-01 BC", "0049-08-31 BC", "month"),
    ("Summer to Autumn 217 BC", "0217-01-01 BC", None, "year"),
    ("late 48 BC – early or mid 47 BC", "0048-01-01 BC", "0047-12-31 BC", "year"),
    ("Late September 218 BC", "0218-09-01 BC", None, "month"),
    ("Early 209 BC", "0209-01-01 BC", None, "year"),
    ("May 633 A.D.", "0633-05-01", None, "month"),
    ("September 634 A.D", "0634-09-01", None, "month"),
    ("July 6–16, 1863", "1863-07-06", "1863-07-16", "day"),
    ("February 217 BC", "0217-02-01 BC", None, "month"),
    # Rendered infoboxes append age text or a second calendar after the date.
    ("18 June 1815 ; 211 years ago ( 1815-06-18 )", "1815-06-18", None, "day"),
    ("April 9, 1865 ; 161 years ago ( 1865-04-09 )", "1865-04-09", None, "day"),
    (
        "16 December 1598 ( Gregorian Calendar )\nKeichō 3, 19th day of 11 month ( Lunar )",
        "1598-12-16",
        None,
        "day",
    ),
    ("3 and 23 October 42 BC", "0042-10-03 BC", "0042-10-23 BC", "day"),
    ("12 and 13 March 1814", "1814-03-12", "1814-03-13", "day"),
    ("26 or 27 September, 480 BC", "0480-09-26 BC", "0480-09-27 BC", "day"),
    ("22/23 December 218 BC", "0218-12-22 BC", "0218-12-23 BC", "day"),
]

UNPARSEABLE = [
    "",
    "   ",
    "unknown",
    "the Middle Ages",
    "31 February 1800",
    "0 BC",
    "5 May 1900 and 6 June 1901",
]


@pytest.mark.parametrize(("raw", "start", "end", "precision"), CASES)
def test_raw_date_text_becomes_a_postgres_literal(
    raw: str, start: str, end: str | None, precision: str
) -> None:
    parsed = parse_date_text(raw)

    assert parsed is not None
    assert (parsed.date_start, parsed.date_end, parsed.precision) == (start, end, precision)


@pytest.mark.parametrize("raw", UNPARSEABLE)
def test_unreadable_date_text_returns_none(raw: str) -> None:
    assert parse_date_text(raw) is None


def test_none_input_returns_none() -> None:
    assert parse_date_text(None) is None


def test_leap_day_is_valid_only_in_leap_years() -> None:
    assert parse_date_text("29 February 1804") is not None
    assert parse_date_text("29 February 1805") is None
    # 217 BC is astronomical -216, a leap year.
    assert parse_date_text("29 February 217 BC") is not None
    assert parse_date_text("29 February 216 BC") is None


def test_sanitise_keeps_a_structured_sources_own_precision() -> None:
    assert sanitise_date_fields("0031-01-01 BC", None, "year") == ("0031-01-01 BC", None, "year")


def test_sanitise_parses_raw_text_and_ignores_the_sources_precision() -> None:
    assert sanitise_date_fields("Spring 208 BC", None, "day") == ("0208-01-01 BC", None, "year")


def test_sanitise_nulls_everything_when_the_start_is_unreadable() -> None:
    assert sanitise_date_fields("sometime", "later", "day") == (None, None, None)


def _wikitext(date: str) -> str:
    return (
        "{{Infobox military conflict\n"
        "| conflict = Battle of Test\n"
        f"| date = {date}\n"
        "| place = Somewhere\n"
        "| result = Roman victory\n"
        "| combatant1 = Rome\n"
        "| combatant2 = Carthage\n"
        "}}\n"
    )


@pytest.mark.parametrize(("raw", "start", "end", "precision"), CASES)
def test_the_infobox_parser_emits_parsed_dates_not_raw_text(
    raw: str, start: str, end: str | None, precision: str
) -> None:
    source = parse_infobox(_wikitext(raw), source_ref="x", source_title="Battle of Test")

    assert source is not None
    assert source.facts.date_start == start
    assert source.facts.date_end == end
    assert source.facts.date_precision == precision


def test_an_unparseable_infobox_date_is_null_and_remembered() -> None:
    source = parse_infobox(_wikitext("sometime"), source_ref="x", source_title="Battle of Test")

    assert source is not None
    assert source.facts.date_start is None
    assert source.facts.date_precision is None
    assert source.unparsed_date_text == "sometime"
