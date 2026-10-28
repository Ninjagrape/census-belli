"""Load-time plausibility rules for troop reports.

The extract stage defaults ``branch`` to ``total``, so a figure only has to be
an unlabelled number in an infobox to arrive as a side's whole army. On the
pilot corpus 80 of 641 usable rows were below 400, and they were day numbers
("13 March:"), vessel and equipment counts ("42 submarines", "140
helicopters") and the individual contingents listed beside a stated total
("30,000 allies" beside "Total: 86,400"). Fed to the model as independent
observations of the side they drag its estimate toward noise, and a hierarchy
with a free per-source term absorbs the rest.

These rules are deliberately about the *text and value of one row*, not about
what the model would make of it, so each can be tested on a context string and
counted by reason. They are conservative by design: a rule that wrongly drops
a real total costs one observation, while one that lets junk through can move
a side by an order of magnitude.

Everything here is pure; :mod:`pipeline.reconcilers.load` applies it.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Final

from pipeline.extractors.store import OVERRIDE_SOURCE_PREFIX
from pipeline.reconcilers.records import ReconcileCounts, Report, SourceKey

__all__ = [
    "NON_PERSONNEL_TERMS",
    "drop_components_of_stated_totals",
    "drop_orphan_parenthetical_rows",
    "is_override_report",
    "exclusion_reason",
    "filter_troop_reports",
    "is_stated_total",
]

# Words that, directly after a figure, say it counts things that are not people.
# Plurals are spelled out in the pattern so "guns" matches and "gunners" does
# not. The list is wider than the ships-and-aircraft core on
# purpose: every extra term here was read off a junk row in the pilot corpus
# ("57 small junks", "150 APCs", "85-200 war elephants").
NON_PERSONNEL_TERMS: Final[tuple[str, ...]] = (
    "ships?",
    "boats?",
    "vessels?",
    "galleys?",
    "galleons?",
    "warships?",
    "battleships?",
    "ironclads?",
    "timberclads?",
    "junks?",
    "transports?",
    "submarines?",
    "panokseons?",
    "aircraft",
    "planes?",
    "helicopters?",
    "tanks?",
    "tankettes?",
    "stug'?s?",
    "assault guns?",
    "apcs?",
    "trucks?",
    "guns?",
    "cannons?",
    "artillery pieces?",
    "batteries",
    "elephants?",
    "horses?",
    "mortars?",
    "camels?",
    "luftflotte",
)

_NUMBER: Final[str] = r"\d+(?:,\d{3})*(?:\.\d+)?"

# Words that cannot sit between a figure and an equipment word without the
# figure belonging to the people instead: "20,000 men on ships" counts men.
_GAP_STOPWORDS: Final[str] = (
    r"men|soldiers|troops|infantry|cavalry|personnel|marines|archers|warriors"
    r"|crews?|crewmen|sailors|seamen|rowers|oarsmen|on|in|aboard|of|and|with|from|for"
)

# An equipment word that is really a modifier of people ("2,000 tank crews",
# "a ship's crew") does not make the figure a count of equipment.
_PEOPLE_AFTER: Final[str] = (
    r"(?:'s)?\s+(?:crews?|crewmen|sailors|seamen|marines|men|personnel|soldiers|troops"
    r"|infantry|gunners|rowers|oarsmen)\b"
)

# The equipment word has to follow the figure within two words ("85-200 war
# elephants", "150 surface-to-air missile batteries"). A bare "context contains
# 'guns'" test also drops real totals: "30,000 men, 100 cannon" is 30,000 men,
# and the 100 is the cannon's.
_FIGURE_THEN_EQUIPMENT = re.compile(
    rf"(?P<a>{_NUMBER})(?:\s*(?:[-‐-―−]|to)\s*(?P<b>{_NUMBER}))?\s*"
    rf"(?:(?!(?:{_GAP_STOPWORDS})\b)[^\W\d][\w'-]*\s+){{0,2}}"
    rf"(?:{'|'.join(NON_PERSONNEL_TERMS)})\b(?!{_PEOPLE_AFTER})",
    re.IGNORECASE,
)

# "Luftflotte 4" names an air fleet; the number is its ordinal, not a strength.
_AIR_FLEET_NUMBER = re.compile(rf"\bluftflotte\s*(?P<a>{_NUMBER})", re.IGNORECASE)


def _as_number(text: str) -> float:
    """Parse a figure written with thousands separators.

    Args:
        text: A match of :data:`_NUMBER`.

    Returns:
        The value.
    """
    return float(text.replace(",", ""))


def _is_non_personnel_figure(value: float, context: str) -> bool:
    """Whether the row's value is a count of equipment or vessels.

    Args:
        value: The row's reported value.
        context: The row's ``extracted_context``.

    Returns:
        True when ``value`` is the figure (or an end of the range) in front of
        an equipment word, or the ordinal after "Luftflotte".
    """
    for match in _FIGURE_THEN_EQUIPMENT.finditer(context):
        figures = {_as_number(g) for g in (match["a"], match["b"]) if g is not None}
        if value in figures:
            return True
    return any(value == _as_number(m["a"]) for m in _AIR_FLEET_NUMBER.finditer(context))


_MONTHS: Final[str] = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
    r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)

# "13 March:", "16-17 October:", "On 1 March 1944 :", "As of 1 October 1941:".
# Anchored at the start and ending on the colon, so a label that is followed by
# a figure ("13 March: 28,000") is left to its own value.
_DATE_LABEL = re.compile(
    r"^\W*(?:(?:on|as of|by|from|at)\s+)?"
    r"(?P<day>\d{1,2})(?:\s*[-‐-―−]\s*(?P<day2>\d{1,2}))?"
    rf"\s+(?:{_MONTHS})\.?(?:\s+(?P<year>\d{{4}}))?\s*:\s*$",
    re.IGNORECASE,
)

# A figure stated as the side's total: "Total: 86,400", "Total : 118,000-120,000".
# Only a context that *starts* with the word counts. "78,000 in total" is also a
# total, but "6,000 skirmishers, total 40,000" is a component that mentions one,
# and the start of the string is the reliable infobox convention.
_STATED_TOTAL = re.compile(r"^\W*total\b", re.IGNORECASE)


def is_stated_total(context: str) -> bool:
    """Whether a context marks its figure as the side's stated total.

    Args:
        context: The report's ``extracted_context``.

    Returns:
        True when the context begins with the word "total".
    """
    return _STATED_TOTAL.match(context) is not None


def _date_label_numbers(context: str) -> set[float] | None:
    """The numbers in a date label, or None when the context is not one.

    Args:
        context: The report's ``extracted_context``.

    Returns:
        The day, second day and year of a leading date label, or None.
    """
    match = _DATE_LABEL.match(context)
    if match is None:
        return None
    return {float(g) for g in (match["day"], match["day2"], match["year"]) if g is not None}


_PARENTHETICAL = re.compile(r"\([^()]*\)")
_ANY_NUMBER = re.compile(_NUMBER)


def _split_parenthetical(context: str) -> tuple[set[float], set[float]]:
    """The figures inside parentheses and the figures outside them.

    Args:
        context: The report's ``extracted_context``.

    Returns:
        ``(inside, outside)`` sets of numbers.
    """
    inside = {
        _as_number(n) for g in _PARENTHETICAL.findall(context) for n in _ANY_NUMBER.findall(g)
    }
    outside_text = _PARENTHETICAL.sub(" ", context)
    return inside, {_as_number(n) for n in _ANY_NUMBER.findall(outside_text)}


def _is_parenthetical_subfigure(value: float, context: str) -> bool:
    """Whether the value is a bracketed qualifier of a larger figure in its context.

    "90,000 men (60,000 Italian, 30,000 German)" and "23,000 troops (3,000
    crossed)": the bracketed figure is a part or a phase of the stated one, not
    a second observation of the same quantity.

    Args:
        value: The row's reported value.
        context: The row's ``extracted_context``.

    Returns:
        True when the value appears only inside parentheses and a larger figure
        stands outside them.
    """
    inside, outside = _split_parenthetical(context)
    return value in inside and value not in outside and any(o > value for o in outside)


def exclusion_reason(value: float, context: str, *, min_plausible_troops: float) -> str | None:
    """Say why one troop row cannot be a head count, if it cannot.

    The checks run from most to least specific so the counter names the real
    cause: a day number is a date, not merely small, and "2 Panokseons" is a
    vessel count whatever its size.

    Args:
        value: The row's reported value.
        context: The row's ``extracted_context``.
        min_plausible_troops: Values below this are not an army's strength.

    Returns:
        ``"date_context"``, ``"non_personnel"``, ``"parenthetical_subfigure"``,
        ``"implausible_small"``, or None when the row passes.
    """
    date_numbers = _date_label_numbers(context)
    if date_numbers is not None and value in date_numbers:
        return "date_context"
    if _is_non_personnel_figure(value, context):
        return "non_personnel"
    if _is_parenthetical_subfigure(value, context):
        return "parenthetical_subfigure"
    if value < min_plausible_troops:
        return "implausible_small"
    return None


def drop_orphan_parenthetical_rows(reports: list[Report], counts: ReconcileCounts) -> list[Report]:
    """Drop a bracketed figure with nothing larger beside it in its own context.

    A context such as "(3,000 crossed)" has no outside figure to compare with, so
    the row is dropped only when the same source states a larger figure for the
    same side.

    Args:
        reports: Troop reports that already passed the per-row rules.
        counts: Counters; increments ``excluded_parenthetical_subfigure``.

    Returns:
        The reports in their original order, minus the orphans.
    """
    largest: dict[tuple[int, SourceKey], float] = defaultdict(float)
    for r in reports:
        key = (r.side_id, r.source_key)
        largest[key] = max(largest[key], r.reported_value)
    kept: list[Report] = []
    for r in reports:
        inside, outside = _split_parenthetical(r.extracted_context)
        if (
            r.reported_value in inside
            and not outside
            and largest[(r.side_id, r.source_key)] > r.reported_value
        ):
            counts.excluded_parenthetical_subfigure += 1
            continue
        kept.append(r)
    return kept


def drop_components_of_stated_totals(
    reports: list[Report], counts: ReconcileCounts
) -> list[Report]:
    """Drop a source's other rows for a side when it also states that side's total.

    "Total: 86,400" beside "25,000 Romans" and "30,000 allies" is one figure and
    its parts. Keeping the parts as separate observations of the side's total
    teaches the model that this source reports about a third of the truth.

    Args:
        reports: Troop reports that already passed the per-row rules.
        counts: Counters; increments ``excluded_component_of_stated_total``.

    Returns:
        The reports in their original order, minus the components.
    """
    has_total: set[tuple[int, SourceKey]] = {
        (r.side_id, r.source_key) for r in reports if is_stated_total(r.extracted_context)
    }
    kept: list[Report] = []
    for report in reports:
        if (report.side_id, report.source_key) in has_total and not is_stated_total(
            report.extracted_context
        ):
            counts.excluded_component_of_stated_total += 1
            continue
        kept.append(report)
    return kept


def is_override_report(report: Report) -> bool:
    """Whether a report is a figure a ``troop_strength`` override wrote.

    Args:
        report: Any loaded report.

    Returns:
        True when its source url carries the override prefix.
    """
    return report.source_key.url.startswith(OVERRIDE_SOURCE_PREFIX)


def filter_troop_reports(
    reports: list[Report], counts: ReconcileCounts, *, min_plausible_troops: float
) -> list[Report]:
    """Apply every plausibility rule to a set of troop reports.

    Args:
        reports: Troop reports that already passed the branch and positivity
            rules.
        counts: Counters; each exclusion is counted under its own reason.
        min_plausible_troops: The small-value floor.

    Returns:
        The reports that survive, in their original order.

    Raises:
        ValueError: If a ``troop_strength`` override is below the floor. A curator
            entered it on purpose, so dropping it as implausible would discard the
            correction without a trace.
    """
    for report in reports:
        if is_override_report(report) and report.reported_value < min_plausible_troops:
            entry_id = report.source_key.url.removeprefix(OVERRIDE_SOURCE_PREFIX)
            raise ValueError(
                f"troop_strength override {entry_id}"
                f" gives {report.reported_value:g} troops, below min_plausible_troops "
                f"({min_plausible_troops:g}); fix the override or lower the floor"
            )
    reason_counter = {
        "date_context": "excluded_date_context",
        "parenthetical_subfigure": "excluded_parenthetical_subfigure",
        "non_personnel": "excluded_non_personnel",
        "implausible_small": "excluded_implausible_small",
    }
    survivors: list[Report] = []
    tally: dict[str, int] = defaultdict(int)
    for report in reports:
        reason = exclusion_reason(
            report.reported_value,
            report.extracted_context,
            min_plausible_troops=min_plausible_troops,
        )
        if reason is None:
            survivors.append(report)
        else:
            tally[reason_counter[reason]] += 1
    for field_name, n in tally.items():
        setattr(counts, field_name, getattr(counts, field_name) + n)
    return drop_components_of_stated_totals(
        drop_orphan_parenthetical_rows(survivors, counts), counts
    )
