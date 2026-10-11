"""Parse the free-text date an infobox carries into a Postgres date literal.

Infobox ``date`` fields are typed by editors: ``21–22 May 1809``, ``Spring 208
BC``, ``15 November [ O.S. 3 November ] 1812``. The ``battles`` columns are
``DATE``, so raw text reaching them is a database error, and one such error
used to abort the whole run's writes.

BC dates are returned as Postgres literals (``0216-08-02 BC``), never as
``datetime.date``, which starts at year 1. A historical year ``N BC`` is
written ``NNNN-... BC``; Postgres itself maps it to astronomical ``1 - N`` in
``battles.year_astronomical``. There is no year 0.

The parser is deliberately conservative: anything it cannot read in full
returns ``None`` rather than a guess, and the caller logs and flags it.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from typing import Final

import structlog

logger = structlog.get_logger(__name__)

PRECISIONS: Final[frozenset[str]] = frozenset({"day", "month", "year", "decade", "century"})

_MAX_YEAR: Final[int] = 9999
_MAX_DAY: Final[int] = 31

_MONTHS: Final[dict[str, int]] = {
    name: number
    for number, names in enumerate(
        (
            ("january", "jan"),
            ("february", "feb"),
            ("march", "mar"),
            ("april", "apr"),
            ("may",),
            ("june", "jun"),
            ("july", "jul"),
            ("august", "aug"),
            ("september", "sept", "sep"),
            ("october", "oct"),
            ("november", "nov"),
            ("december", "dec"),
        ),
        start=1,
    )
    for name in names
}
_SEASONS: Final[frozenset[str]] = frozenset({"spring", "summer", "autumn", "fall", "winter"})
_QUALIFIERS: Final[frozenset[str]] = frozenset({"early", "mid", "late"})
_CIRCA: Final[frozenset[str]] = frozenset({"c", "ca", "circa", "about"})
_FILLER: Final[frozenset[str]] = frozenset({"or", "of", "the"})

_ISO_RE: Final = re.compile(r"(\d{4})-(\d{2})-(\d{2})(\s+BC)?", re.IGNORECASE)
_ISO_FULL_RE: Final = re.compile(
    rf"^{_ISO_RE.pattern}(?:\s*-\s*{_ISO_RE.pattern})?$", re.IGNORECASE
)
_ORDINAL_MONTH_RE: Final = re.compile(
    r"\d+(?:st|nd|rd|th)\s+day,?\s+\d+(?:st|nd|rd|th)\s+month\s+of\s+(\d{1,4})\b",
    re.IGNORECASE,
)
_PAREN_RE: Final = re.compile(r"\([^)]*\)|\[[^\]]*\]")
_DASH_RE: Final = re.compile("[‐-―−�]")
# "12 and 13 March", "26 or 27 September", "22/23 December": a span of days,
# read as first to last.
_ALTERNATIVE_DAYS_RE: Final = re.compile(r"\b(\d{1,2})\s*(?:/|\band\b|\bor\b)\s*(\d{1,2})\b")
_ERA_RE: Final = re.compile(
    r"\b(B\.?\s?C\.?(?:\s?E\.?)?|A\.?\s?D\.?|C\.?\s?E\.?)(?![A-Za-z])", re.I
)
_TOKEN_RE: Final = re.compile(r"[A-Za-z]+\.?|\d+(?:st|nd|rd|th)?|[,/-]")
_DAY_RANGE_DMY_RE: Final = re.compile(r"^(\d{1,2})\s*-\s*(\d{1,2})\s+([A-Za-z.]+)\s+(\d+)$")
_DAY_RANGE_MDY_RE: Final = re.compile(r"^([A-Za-z.]+)\s+(\d{1,2})\s*-\s*(\d{1,2}),?\s+(\d+)$")


@dataclass(frozen=True)
class ParsedDate:
    """A date read from text, as Postgres literals.

    Attributes:
        date_start: First day, as ``YYYY-MM-DD`` with a `` BC`` suffix for BC.
        date_end: Last day of a range, or None for a single date.
        precision: One of :data:`PRECISIONS`.
    """

    date_start: str
    date_end: str | None
    precision: str


@dataclass(frozen=True)
class _Part:
    """One side of a possible range, with whatever fields it names."""

    year: int | None = None
    month: int | None = None
    day: int | None = None
    bc: bool | None = None


def _literal(year: int, month: int, day: int, bc: bool) -> str:
    """Format a Postgres date literal."""
    text = f"{year:04d}-{month:02d}-{day:02d}"
    return f"{text} BC" if bc else text


def _days_in_month(year: int, month: int, bc: bool) -> int:
    """Return the length of a month in the proleptic Gregorian calendar.

    Args:
        year: Historical year (BC years count down, no year 0).
        month: Month number.
        bc: Whether the year is BC.

    Returns:
        The number of days.
    """
    astronomical = 1 - year if bc else year
    return calendar.monthrange(2000 + (astronomical % 400), month)[1]


def _valid(year: int, month: int, day: int, bc: bool) -> bool:
    """Check a calendar date, with historical (no year 0) numbering."""
    if not 1 <= year <= _MAX_YEAR or not 1 <= month <= 12:
        return False
    return 1 <= day <= _days_in_month(year, month, bc)


def _parse_iso(text: str) -> ParsedDate | None:
    """Read one or two already-ISO dates, e.g. ``0216-08-02 BC``."""
    match = _ISO_FULL_RE.match(text.strip())
    if match is None:
        return None
    first = _iso_group(match.group(1), match.group(2), match.group(3), match.group(4))
    if first is None:
        return None
    if match.group(5) is None:
        return ParsedDate(first, None, "day")
    last = _iso_group(match.group(5), match.group(6), match.group(7), match.group(8))
    return ParsedDate(first, last, "day") if last else None


def _iso_group(year: str, month: str, day: str, bc: str | None) -> str | None:
    """Validate and re-emit one ISO date's captured groups."""
    is_bc = bc is not None
    if not _valid(int(year), int(month), int(day), is_bc):
        return None
    return _literal(int(year), int(month), int(day), is_bc)


def _clean(text: str) -> str:
    """Strip markup, brackets, parenthesised extras and normalise dashes."""
    cleaned = text.replace("\xa0", " ")
    cleaned = _PAREN_RE.sub(" ", cleaned)
    # Rendered infoboxes append "; 161 years ago" or a second calendar's line
    # after the date proper; only the first segment is the date.
    cleaned = re.split(r"[;\n]", cleaned.strip(), maxsplit=1)[0]
    cleaned = _DASH_RE.sub("-", cleaned)
    cleaned = _ALTERNATIVE_DAYS_RE.sub(r"\1-\2", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip(" ,;.")


def _extract_flags(text: str) -> tuple[str, bool | None, bool, bool] | None:
    """Remove era markers, seasons, qualifiers and filler words.

    Args:
        text: Cleaned date text.

    Returns:
        (remaining text, bc, season_or_vague, circa), or None when a word the
        parser does not know is left over.
    """
    bc: bool | None = None
    for era in _ERA_RE.findall(text):
        is_bc = era.upper().replace(".", "").replace(" ", "").startswith("B")
        if bc is not None and bc != is_bc:
            return None
        bc = is_bc
    text = _ERA_RE.sub(" ", text)

    vague = False
    circa = False
    kept: list[str] = []
    for token in _TOKEN_RE.findall(text):
        word = token.lower().rstrip(".")
        if word in _SEASONS:
            vague = True
        elif word in _QUALIFIERS:
            continue
        elif word in _CIRCA:
            circa = True
        elif word == "to" or token == "-":
            kept.append("-")
        elif word in _FILLER or token in {",", "/"}:
            continue
        elif word in _MONTHS or token[0].isdigit():
            kept.append(token)
        else:
            return None
    return " ".join(kept), bc, vague, circa


def _ordinal_free(token: str) -> int:
    """Convert ``3`` or ``3rd`` to an int."""
    return int(re.sub(r"\D", "", token))


def _parse_part(text: str, *, bare_is_day: bool = False) -> _Part | None:
    """Read the month, day and year named in one fragment.

    Args:
        text: Tokens (month names and numbers) from one side of a range.
        bare_is_day: Read a lone small number as a day, not a year.

    Returns:
        The named fields, or None when the fragment is unreadable.
    """
    month: int | None = None
    numbers: list[int] = []
    for token in text.split():
        word = token.lower().rstrip(".")
        if word in _MONTHS:
            if month is not None:
                return None
            month = _MONTHS[word]
        else:
            numbers.append(_ordinal_free(token))

    if len(numbers) > 2 or (not numbers and month is None):
        return None
    if not numbers:
        return _Part(month=month)

    big = [n for n in numbers if n > _MAX_DAY]
    if len(big) > 1:
        return None
    if len(numbers) == 2:
        # Day then year in both "d Month y" and "Month d, y".
        if month is None or numbers[0] > _MAX_DAY or numbers[1] == 0:
            return None
        return _Part(year=numbers[1], month=month, day=numbers[0])
    number = numbers[0]
    if number > _MAX_DAY or (month is None and not bare_is_day):
        return _Part(year=number, month=month)
    return _Part(month=month, day=number) if month is not None else _Part(day=number)


def _single_day_text(part: _Part, bc: bool) -> str | None:
    """Render a fully specified part as a literal, or None when invalid."""
    if part.year is None or part.month is None or part.day is None:
        return None
    if not _valid(part.year, part.month, part.day, bc):
        return None
    return _literal(part.year, part.month, part.day, bc)


def _resolve_range(
    first: _Part, last: _Part | None, bc: bool, *, vague: bool, circa: bool
) -> ParsedDate | None:
    """Combine the parts of a date or range into literals and a precision."""
    if last is not None:
        # An end part carries the year (and era) the start part leaves out.
        first = _Part(
            year=first.year if first.year is not None else last.year,
            month=first.month,
            day=first.day,
            bc=bc,
        )
        last = _Part(
            year=last.year if last.year is not None else first.year,
            month=last.month if last.month is not None or first.month is None else first.month,
            day=last.day,
            bc=bc,
        )
    if first.year is None or not 1 <= first.year <= _MAX_YEAR:
        return None

    has_end = last is not None
    year_only = vague or circa or first.month is None
    if year_only and not has_end:
        return ParsedDate(_literal(first.year, 1, 1, bc), None, "year")

    month = first.month or 1
    day = first.day
    if not has_end and day is None:
        if not _valid(first.year, month, 1, bc):
            return None
        return ParsedDate(_literal(first.year, month, 1, bc), None, "month")
    if not has_end:
        text = _single_day_text(first, bc)
        return ParsedDate(text, None, "day") if text else None

    assert last is not None
    if last.year is None:
        return None
    start_precision = "day" if day is not None else ("month" if first.month else "year")
    end_precision = "day" if last.day is not None else ("month" if last.month else "year")
    start = _literal(first.year, month, day or 1, bc)
    end_month = last.month or 12
    end_day = last.day or _days_in_month(last.year, end_month, bc)
    if not _valid(first.year, month, day or 1, bc) or not _valid(last.year, end_month, end_day, bc):
        return None
    precision = (
        "day"
        if start_precision == end_precision == "day"
        else ("year" if "year" in (start_precision, end_precision) else "month")
    )
    return ParsedDate(start, _literal(last.year, end_month, end_day, bc), precision)


def _parse_compact_day_range(text: str, bc: bool) -> ParsedDate | None:
    """Read ``21-22 May 1809`` and ``May 5-7, 1864`` same-month day ranges."""
    match = _DAY_RANGE_DMY_RE.match(text)
    if match is not None:
        day_a, day_b, month_name, year = match.groups()
    else:
        match = _DAY_RANGE_MDY_RE.match(text)
        if match is None:
            return None
        month_name, day_a, day_b, year = match.groups()
    month = _MONTHS.get(month_name.lower().rstrip("."))
    if month is None:
        return None
    if not (_valid(int(year), month, int(day_a), bc) and _valid(int(year), month, int(day_b), bc)):
        return None
    return ParsedDate(
        _literal(int(year), month, int(day_a), bc),
        _literal(int(year), month, int(day_b), bc),
        "day",
    )


def parse_date_text(raw: str | None) -> ParsedDate | None:
    """Parse infobox date text into Postgres date literals.

    Args:
        raw: The raw text, e.g. ``"Spring 208 BC"`` or ``"21–22 May 1809"``.

    Returns:
        The literals and precision, or None when the text is empty or cannot
        be read in full. The caller flags the battle and logs the gap.
    """
    if raw is None or not raw.strip():
        return None

    iso = _parse_iso(raw)
    if iso is not None:
        return iso

    ordinal = _ORDINAL_MONTH_RE.search(raw)
    if ordinal is not None:
        year = int(ordinal.group(1))
        return ParsedDate(_literal(year, 1, 1, False), None, "year") if year >= 1 else None

    cleaned = _clean(raw)
    if not cleaned:
        return None

    flags = _extract_flags(cleaned)
    if flags is None:
        return None
    body, era, vague, circa = flags
    bc = bool(era)

    return _parse_body(body, bc, vague=vague, circa=circa)


def _parse_body(body: str, bc: bool, *, vague: bool, circa: bool) -> ParsedDate | None:
    """Parse the era- and word-stripped body of a date text."""
    if not (vague or circa):
        compact = _parse_compact_day_range(body, bc)
        if compact is not None:
            return compact

    pieces = [p.strip() for p in re.split(r"\s*-\s*", body) if p.strip()]
    if not pieces or len(pieces) > 2:
        return None

    last = _parse_part(pieces[-1])
    if last is None:
        return None
    if len(pieces) == 1:
        return _resolve_range(last, None, bc, vague=vague, circa=circa)

    first = _parse_part(pieces[0], bare_is_day=last.month is not None and last.year is not None)
    if first is None:
        return None
    return _resolve_range(first, last, bc, vague=vague, circa=circa)


def sanitise_date_fields(
    date_start: str | None, date_end: str | None, precision: str | None
) -> tuple[str | None, str | None, str | None]:
    """Make date fields safe to cast to DATE, parsing raw text when needed.

    Already-valid literals pass through unchanged. Anything else is parsed;
    what cannot be parsed becomes NULL so raw text never reaches the database.

    Args:
        date_start: A literal, raw text, or None.
        date_end: A literal, raw text, or None.
        precision: The source's precision, or None.

    Returns:
        (date_start, date_end, precision), all NULL together when the start is
        unusable.
    """
    if date_start is None:
        return None, None, None

    is_literal = _parse_iso(date_start) is not None
    parsed = parse_date_text(date_start)
    if parsed is None:
        logger.warning("date_text_unparseable", value=date_start[:80])
        return None, None, None

    end = parsed.date_end
    if end is None and date_end is not None:
        end_parsed = parse_date_text(date_end)
        end = (end_parsed.date_end or end_parsed.date_start) if end_parsed else None
    # A literal came from a structured source whose own precision is the
    # authority (Wikidata writes year-precision dates as January 1st).
    keep_source = is_literal and precision in PRECISIONS
    return parsed.date_start, end, precision if keep_source else parsed.precision
