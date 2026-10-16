"""The caveats every output carries, merged with evaluate's and the run's own."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any, Final

__all__ = [
    "INTERVAL_METHOD_NOTE",
    "OVERLAP_METHOD_NOTE",
    "STATIC_DISCLOSURES",
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
            f"The model credits {used} of {scope} in-scope battles; {scope - used} are "
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
            "Ranks are positions among all credited generals; "
            f"{n_ranks_in_gaps} general(s) with fewer than {min_battles} credited battles hold "
            "ranks within the headline range and are listed without a rank."
        )
    return tuple(out)
