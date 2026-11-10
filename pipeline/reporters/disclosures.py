"""The caveats every output carries, merged with evaluate's and the run's own."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any, Final

import structlog

from pipeline.reporters.coverage import (
    ANCIENT,
    CENTURY,
    REPORT_ERAS,
    CoverageStats,
    FameBin,
)

logger = structlog.get_logger()

__all__ = [
    "INTERVAL_METHOD_NOTE",
    "OVERLAP_METHOD_NOTE",
    "STATIC_DISCLOSURES",
    "coverage_disclosures",
    "interval_method_note",
    "merge_disclosures",
    "p_above_next_note",
    "run_disclosures",
]

INTERVAL_METHOD_NOTE: Final[str] = (
    "WAR intervals are the model's stored 95% credible intervals on total WAR, from the pooled "
    "posterior draws widened for imputation uncertainty. Rank overlap flags compare each "
    "general's interval only with the next listed general's."
)

OVERLAP_METHOD_NOTE: Final[str] = (
    "Rank overlap flags compare each general's interval, total WAR plus or minus z times the "
    "stored Rubin-pooled WAR sd, only with the next listed general's. This run stored no WAR "
    "credible interval, so this is a normal approximation that ignores skew in the WAR "
    "posterior and the correlation between generals' WAR (shared replacement level, shared "
    "battles). It is a rough guide, not a probability that the order is right."
)

STATIC_DISCLOSURES: Final[tuple[str, ...]] = (
    "Cross-era comparison rests on the shared skill prior. Generals who never faced a common "
    "opponent chain are linked only through that prior, so their relative order is weakly "
    "identified.",
    "Technology is not modelled. A commander from a period with a large technological edge "
    "is compared on the same scale as one without it.",
    "Naval and aerial battles share one population and one force slope with land battles.",
    "Total WAR is volume weighted: it sums expected WAR over credited battles, so a gap in a "
    "general's era coverage in the corpus is also a gap in their rank. WAR per battle is shown "
    "beside it.",
    "Attribution weights are fixed data from the classify stage. Where several commanders "
    "share a side the default split is equal, and the model does not learn who actually "
    "commanded.",
    "WAR is shared side performance, not individual battlefield credit: a side's expected "
    "surplus over a replacement-level commander is split by attribution weight, so a "
    "subordinate inherits the result of the leader above them. Effective battles are shown "
    "beside rank for that reason; the skill interval width shows how much a general's own "
    "record pins down their skill.",
    "Expected score is P(win) plus half of P(indecisive) under the model. It is shown as "
    "expected score, never as a win probability, and WAR is expected WAR.",
    INTERVAL_METHOD_NOTE,
)


def interval_method_note(*, uses_normal_approx: bool) -> str:
    """The sentence describing how WAR intervals were obtained.

    Args:
        uses_normal_approx: Whether any listed interval is the fallback
            normal approximation.

    Returns:
        The disclosure text.
    """
    return OVERLAP_METHOD_NOTE if uses_normal_approx else INTERVAL_METHOD_NOTE


_P_NEXT_TAIL: Final[str] = (
    " It uses the unwidened mixture of pooled posterior draws, while the WAR interval is "
    "widened by Rubin's total variance T, so it is slightly overconfident relative to that "
    "interval (a small effect)."
)


def p_above_next_note(source: str) -> str:
    """The sentence saying what P(above next ranked) is in this report.

    Args:
        source: ``war`` or ``skill`` (recomputed over the headline ordering from
            the pooled draws), ``skill_unavailable`` (recomputed on WAR because
            the run ranks by skill but the draws file holds none) or ``stored``
            (no draws file, so the model stage's column is shown).

    Returns:
        The disclosure text.
    """
    if source == "stored":
        return (
            "P(above next ranked) is the model stage's stored value: the share of pooled "
            "posterior draws in which a general's WAR exceeds that of the general ranked "
            "immediately below in the full ranking. This run has no pooled draws file, so it "
            "could not be recomputed, and where a general below min_battles sits between two "
            "listed generals it refers to that hidden general, not the next listed row."
            + _P_NEXT_TAIL
        )
    basis = "skill" if source == "skill" else "WAR"
    note = (
        f"P(above next ranked) is recomputed from the pooled draws over the headline "
        f"ordering: the share of draws in which a general's {basis} exceeds that of the next "
        "listed general, so it accounts for the correlation between them."
    )
    if source == "skill_unavailable":
        note += (
            " The run ranks by skill but its draws file holds no skill draws, so the "
            "comparison is on total WAR and may disagree with the skill-based order."
        )
    return note + _P_NEXT_TAIL


_SPACES = re.compile(r"\s+")


def _key(sentence: str) -> str:
    return _SPACES.sub(" ", sentence).strip().casefold()


def merge_disclosures(*groups: Iterable[str]) -> tuple[str, ...]:
    """Concatenate disclosure groups, dropping repeats and blanks.

    Repeats are matched ignoring case and whitespace. The first occurrence
    keeps its position, so the report's own list leads.

    Args:
        *groups: Disclosure lists in priority order.

    Returns:
        The merged disclosures.
    """
    seen: set[str] = set()
    out: list[str] = []
    for group in groups:
        for sentence in group:
            key = _key(sentence)
            if key and key not in seen:
                seen.add(key)
                out.append(sentence.strip())
    return tuple(out)


def _count(counts: Mapping[str, Any], key: str) -> int | None:
    value = counts.get(key)
    return int(value) if isinstance(value, int | float) else None


def run_disclosures(
    counts: Mapping[str, Any],
    *,
    min_battles: int,
    n_under_determined: int,
    n_ranks_in_gaps: int = 0,
) -> tuple[str, ...]:
    """Disclosures that quote this run's own counts.

    Args:
        counts: ``model_runs.diagnostics.counts`` of the reported run.
        min_battles: The headline threshold.
        n_under_determined: Generals listed without a rank.
        n_ranks_in_gaps: Generals below ``min_battles`` whose stored rank lies
            inside the headline's rank range.

    Returns:
        Sentences for the counts that are present.
    """
    out: list[str] = []
    scope, used = _count(counts, "battles_in_scope"), _count(counts, "battles_included")
    if scope is not None and used is not None:
        parts = [
            f"{n} with {label}"
            for key, label in (
                ("battles_no_creditable_side", "no creditable side"),
                ("battles_no_outcome", "no recorded outcome"),
            )
            if (n := _count(counts, key))
        ]
        detail = f" ({', '.join(parts)})" if parts else ""
        out.append(
            f"The model credits {used} of {scope} in-scope battles; {scope - used} "
            f"{_verb(scope - used, 'is', 'are')} "
            f"excluded{detail}. Generals are ranked only on the battles that are credited."
        )
    multi, split = (
        _count(counts, "sides_multi_commander_all"),
        _count(counts, "sides_default_split_all"),
    )
    if multi:
        out.append(
            f"{split or 0} of {multi} multi-commander sides use the default equal split of credit."
        )
    out.append(
        f"Generals with fewer than {min_battles} credited battles are listed separately "
        f"without a rank ({n_under_determined} in this run)."
    )
    if n_ranks_in_gaps:
        out.append(
            "Rank counts headline generals from 1; the stored position among all credited "
            f"generals is shown separately. {_plural(n_ranks_in_gaps, 'general')} with fewer "
            f"than {min_battles} credited battles {_verb(n_ranks_in_gaps, 'holds', 'hold')} "
            "stored positions within the headline "
            "range and are listed without a rank."
        )
    return tuple(out)


_MAX_SPANS: Final[int] = 8


def _plural(n: int, noun: str) -> str:
    """``1 answer``, ``2 answers``."""
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _verb(n: int, singular: str, plural: str) -> str:
    """Pick the verb form that agrees with a count: ``1 side rests``, ``2 sides rest``."""
    return singular if n == 1 else plural


def _bc(year: int) -> int:
    """Years before the common era as a reader counts them (astronomical 0 is 1 BC)."""
    return 1 - year


def _span(first: int, last: int) -> str:
    """An inclusive span of astronomical years, with BC and AD only where they are needed.

    Args:
        first: First astronomical year.
        last: Last astronomical year, not before ``first``.

    Returns:
        ``1 BC to AD 1799`` when the span crosses zero, ``401 to 102 BC`` when it is
        entirely BC, ``1300 to 1499`` when entirely AD. A single year stands alone.
    """
    if last <= 0:
        if first == last:
            return f"{_bc(first)} BC"
        return f"{_bc(first)} to {_bc(last)} BC"
    if first <= 0:
        return f"{_bc(first)} BC to AD {last}"
    return str(first) if first == last else f"{first} to {last}"


def _pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.1f}%"


def _rule_single(c: CoverageStats) -> str | None:
    if c.rule_single_sides is None or not c.credited_sides:
        return None
    return (
        f"{c.rule_single_sides} of {c.credited_sides} credited sides use the rule_single "
        "attribution: only one commander is named in the infobox, so that commander is "
        "credited with the whole side (weight 1.0) without any check of the article text. "
        "A commander who was in fact absent from the field would be credited anyway."
    )


def _components(c: CoverageStats) -> str | None:
    if c.n_components is None:
        return None
    text = f"The comparison graph has {_plural(c.n_components, 'connected component')}."
    if c.top_outside_largest is not None and c.top_component_known:
        text += (
            f" {c.top_outside_largest} of the {c.top_component_known} leading generals sit "
            "outside the largest one. Generals in different components never faced a common "
            "opponent chain, so their relative order rests on the shared skill prior alone."
        )
    return text


def _passage_pass(c: CoverageStats) -> str | None:
    if c.extract_answers is None:
        return None
    if c.extract_answers == 0:
        return (
            "No answers from the extract stage's article-passage pass are stored, so troop "
            "scope and prose evidence come from infoboxes only."
        )
    return (
        f"{_plural(c.extract_answers, 'answer')} from the extract stage's article-passage "
        f"pass {'is' if c.extract_answers == 1 else 'are'} stored. "
        "How many passage requests are still unanswered is not recorded in the database, "
        "so it is not stated here; troop scope and prose evidence are missing for those."
    )


def _corpus_shape(c: CoverageStats) -> list[str]:
    out: list[str] = []
    if c.corpus_battles and c.first_century is not None and c.last_century is not None:
        first = c.first_year if c.first_year is not None else c.first_century
        last = c.last_year if c.last_year is not None else c.last_century + CENTURY - 1
        text = f"The {c.corpus_battles} credited battles span {_span(first, last)}."
        if c.empty_spans:
            shown = [_span(a, b) for a, b in c.empty_spans[:_MAX_SPANS]]
            more = len(c.empty_spans) - len(shown)
            tail = f" and {_plural(more, 'more span')}" if more > 0 else ""
            text += (
                f" No credited battle falls in {', '.join(shown)}{tail}, so no general from "
                "those periods can be ranked."
            )
        out.append(text)
    if c.headline_by_era and c.n_headline:
        lead, *rest = c.headline_by_era
        listed = ", ".join(
            [
                f"{lead.n} fought their first credited battle in the {lead.era} era "
                f"({_era_range(lead.era)})",
                *(f"{e.n} in the {e.era} era ({_era_range(e.era)})" for e in rest),
            ]
        )
        undated = c.n_headline - sum(e.n for e in c.headline_by_era)
        if undated > 0:
            listed += f", and {undated} {_verb(undated, 'has', 'have')} no dated credited battle"
        out.append(
            f"Of the {c.n_headline} headline generals, {listed}. Which periods the corpus "
            "covers shapes who can reach the headline."
        )
    fame = _fame_sentence(c.fame_bins)
    if fame:
        out.append(fame)
    return out


def _era_range(label: str) -> str:
    """The years an era covers, for the sentence that quotes its count."""
    cuts = [start for start, _ in REPORT_ERAS]
    names = [name for _, name in REPORT_ERAS]
    if label == ANCIENT:
        return f"before {cuts[0]}"
    i = names.index(label)
    return f"{cuts[i]} onwards" if i == len(cuts) - 1 else f"{cuts[i]} to {cuts[i + 1] - 1}"


def _pool(bins: dict[str, FameBin], labels: tuple[str, ...]) -> tuple[int, int] | None:
    found = [bins[k] for k in labels if k in bins]
    return (sum(b.n_credited for b in found), sum(b.n_wins for b in found)) if found else None


def _fame_sentence(fame_bins: tuple[FameBin, ...]) -> str | None:
    bins = {b.label: b for b in fame_bins}
    high, low = _pool(bins, ("5-9", "10+")), _pool(bins, ("1", "2"))
    if not high or not low or not high[0] or not low[0]:
        return None
    return (
        f"Generals who led a side in 5 or more credited battles won "
        f"{_pct(high[1], high[0])} of those {high[0]} battles, against "
        f"{_pct(low[1], low[0])} of {low[0]} for generals with 1 or 2. "
        "Commanders who win appear in more articles, so a high battle count is partly "
        "fame, and volume-weighted WAR rewards it."
    )


def _troops(c: CoverageStats) -> str | None:
    if not c.troop_eras:
        return None
    multi = sum(e.multi_source for e in c.troop_eras)
    single = sum(e.single_report for e in c.troop_eras)
    imputed = sum(e.imputed for e in c.troop_eras)
    other = sum(e.other for e in c.troop_eras)
    bound = sum(e.bound_only for e in c.troop_eras)
    total = multi + single + imputed + other + bound
    if not total:
        return None
    by_era = "; ".join(f"{e.era} {e.weak} of {e.total}" for e in c.troop_eras if e.total)
    bound_text = (
        f" {_plural(bound, 'side')} {_verb(bound, 'rests', 'rest')} only on an upper or lower "
        "bound ('up to' or 'at least')."
        if bound
        else ""
    )
    other_text = (
        f" {_plural(other, 'side')} {_verb(other, 'carries', 'carry')} a total from another "
        f"reconcile method and {_verb(other, 'is', 'are')} counted in neither group."
        if other
        else ""
    )
    weak_kinds = "single report, bound only or imputed" if bound else "single report or imputed"
    return (
        f"Of {total} sides in credited battles, {multi} "
        f"{_verb(multi, 'has', 'have')} a troop total reconciled from two "
        f"or more independent sources, {single} {_verb(single, 'rests', 'rest')} on a single "
        f"report, and {imputed} {_verb(imputed, 'has', 'have')} no reported total and "
        f"{_verb(imputed, 'is', 'are')} imputed.{bound_text}{other_text} Sides without a "
        f"multi-source total ({weak_kinds}) by era: {by_era}."
    )


def _rank_numbering(c: CoverageStats, min_battles: int) -> str | None:
    if not c.n_headline or not c.n_generals_total:
        return None
    return (
        f"Rank in this report counts the {c.n_headline} generals with at least {min_battles} "
        f"credited battles from 1 to {c.n_headline}. The stored rank, a position among all "
        f"{c.n_generals_total} credited generals, is shown in its own column and differs "
        "where generals below the threshold sit between listed ones."
    )


def coverage_disclosures(stats: CoverageStats, *, min_battles: int) -> tuple[str, ...]:
    """Sentences quoting the data-coverage numbers of the reported run.

    A sentence whose numbers are undetermined is left out and logged.

    Args:
        stats: The numbers from :func:`pipeline.reporters.coverage.load_coverage`.
        min_battles: The headline threshold.

    Returns:
        The sentences, in report order.
    """
    candidates = (
        ("rule_single", [_rule_single(stats)]),
        ("components", [_components(stats)]),
        ("passage_pass", [_passage_pass(stats)]),
        ("corpus_shape", _corpus_shape(stats)),
        ("troop_provenance", [_troops(stats)]),
        ("rank_numbering", [_rank_numbering(stats, min_battles)]),
    )
    out: list[str] = []
    for item, sentences in candidates:
        kept = [s for s in sentences if s]
        if not kept:
            logger.warning("report_disclosure_omitted", item=item)
        out.extend(kept)
    return tuple(out)
