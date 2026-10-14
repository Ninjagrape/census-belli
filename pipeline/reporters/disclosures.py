"""The caveats every output carries, merged with evaluate's and the run's own."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any, Final

__all__ = [
    "OVERLAP_METHOD_NOTE",
    "STATIC_DISCLOSURES",
    "merge_disclosures",
    "run_disclosures",
]

OVERLAP_METHOD_NOTE: Final[str] = (
    "Rank overlap flags compare each general's interval, total WAR plus or minus z times the "
    "stored Rubin-pooled WAR sd, only with the next listed general's. This is a normal "
    "approximation that ignores skew in the WAR posterior and the correlation between "
    "generals' WAR (shared replacement level, shared battles), so it is a rough guide, not a "
    "probability that the order is right. The model stores no WAR credible interval."
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
    OVERLAP_METHOD_NOTE,
)

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
