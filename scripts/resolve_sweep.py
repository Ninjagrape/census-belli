"""Sweep resolve thresholds against the gold set.

``pipeline/resolvers/matcher.py`` decides most of the corpus with two
parameters, ``fuzzy_threshold`` and ``ambiguity_margin``, and the module's own
docstring says which mistake to fear: a false merge pools two commanders'
battles into one skill estimate and nothing downstream can detect it, while a
false split only costs power and is visible as two near-identical names. This
script exists to pick threshold/margin values with that asymmetry in mind,
against a hand-labelled gold set rather than by eye.

It spends zero LLM quota. A group the matcher would hand to the LLM step is
reported here as ``ambiguous`` and left there -- never resolved by any other
means -- so every combo's ``ambiguous_count`` is a faithful projection of how
many LLM calls that setting would cost in a real run.

Usage::

    python -m scripts.resolve_sweep
    python -m scripts.resolve_sweep --threshold-range 70 95 --threshold-step 5 \\
        --margin-range 2 12 --margin-step 2
    python -m scripts.resolve_sweep --json sweep_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline.resolvers.matcher import match_group
from pipeline.resolvers.records import BattleContext, Candidate, Mention, MentionGroup

__all__ = [
    "GOLD_DIR",
    "GoldCase",
    "SweepResult",
    "build_group",
    "float_range",
    "load_candidates",
    "load_gold_cases",
    "recommend",
    "score_case",
    "sweep",
]

GOLD_DIR = Path("tests/fixtures/gold/resolve")

_DEFAULT_THRESHOLD_RANGE: tuple[float, float] = (70.0, 95.0)
_DEFAULT_THRESHOLD_STEP: float = 5.0
_DEFAULT_MARGIN_RANGE: tuple[float, float] = (2.0, 12.0)
_DEFAULT_MARGIN_STEP: float = 2.0


@dataclass(frozen=True)
class GoldCase:
    """One mention with its expected resolution.

    Attributes:
        name: The commander's surface form, as an infobox would write it.
        battle_slug: The battle's stable identifier.
        battle_name: The battle's display name.
        year: Astronomical year of the battle, or None when undated.
        side_label: The side the commander fought on.
        polity: The side's polity, used as a matcher tiebreak.
        expected_qid: The Wikidata id this mention should link to, when the
            expected verdict is ``linked`` or a correctly resolved
            ``ambiguous_ok``.
        expected_verdict: One of ``linked``, ``new``, ``ambiguous_ok``,
            ``placeholder``.
    """

    name: str
    battle_slug: str
    battle_name: str
    year: int | None
    side_label: str
    polity: str
    expected_qid: str | None
    expected_verdict: str


@dataclass(frozen=True)
class SweepResult:
    """Metrics for one threshold/margin combo.

    Attributes:
        threshold: The ``fuzzy_threshold`` tried.
        margin: The ``ambiguity_margin`` tried.
        correct_links: Linked to the expected Wikidata entity.
        wrong_links: Linked to the wrong entity, or linked when the gold
            verdict was ``new`` -- a false merge. The number to minimise.
        correct_new: Correctly left as a new, corpus-local entity.
        missed_links: Expected ``linked``, matcher returned ``new`` -- a false
            split. Costs power but is visible, so it is not the number to
            minimise first.
        correct_placeholders: Placeholder mentions, always correct by
            construction since they are never run through the matcher.
        ambiguous_count: Groups the matcher would hand to the LLM step. Never
            resolved here; reported as a projected cost.
        total: Every gold case scored, including placeholders.
    """

    threshold: float
    margin: float
    correct_links: int
    wrong_links: int
    correct_new: int
    missed_links: int
    correct_placeholders: int
    ambiguous_count: int
    total: int

    @property
    def wrong_link_rate(self) -> float:
        """Wrong links as a fraction of every case scored."""
        return self.wrong_links / max(1, self.total)

    @property
    def link_rate(self) -> float:
        """Correct links as a fraction of the non-placeholder cases."""
        return self.correct_links / max(1, self.total - self.correct_placeholders)

    @property
    def ambiguous_rate(self) -> float:
        """Ambiguous cases as a fraction of the non-placeholder cases."""
        return self.ambiguous_count / max(1, self.total - self.correct_placeholders)

    def as_dict(self) -> dict[str, float | int]:
        """Render this result for JSON output.

        Returns:
            A flat mapping of every field and derived rate.
        """
        return {
            "threshold": self.threshold,
            "margin": self.margin,
            "correct_links": self.correct_links,
            "wrong_links": self.wrong_links,
            "correct_new": self.correct_new,
            "missed_links": self.missed_links,
            "correct_placeholders": self.correct_placeholders,
            "ambiguous_count": self.ambiguous_count,
            "total": self.total,
            "wrong_link_rate": round(self.wrong_link_rate, 4),
            "link_rate": round(self.link_rate, 4),
            "ambiguous_rate": round(self.ambiguous_rate, 4),
        }


def _candidate_from_row(row: dict[str, Any]) -> Candidate:
    """Build a candidate from one row of ``candidates.jsonl``.

    Args:
        row: One entry of a name's ``candidates`` list.

    Returns:
        The candidate, with tuple fields converted from JSON lists.
    """
    return Candidate(
        qid=str(row.get("qid") or ""),
        label=str(row.get("label") or ""),
        description=str(row.get("description") or ""),
        aliases=tuple(row.get("aliases") or ()),
        birth_year=row.get("birth_year"),
        death_year=row.get("death_year"),
        country=str(row.get("country") or ""),
        occupations=tuple(row.get("occupations") or ()),
        wikipedia_url=str(row.get("wikipedia_url") or ""),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON lines file into a list of mappings.

    Args:
        path: The file to read.

    Returns:
        One mapping per non-blank line.

    Raises:
        SystemExit: If the file is missing. The sweep has nothing to measure
            without it, so failing loudly here is better than reporting an
            empty, misleadingly clean sweep.
    """
    if not path.is_file():
        print(
            f"gold fixture missing: {path}\n"
            "expected tests/fixtures/gold/resolve/mentions.jsonl and "
            "candidates.jsonl -- see TODO.md's resolve gold-set task.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_gold_cases(gold_dir: Path = GOLD_DIR) -> list[GoldCase]:
    """Load every mention in the gold set.

    Args:
        gold_dir: The directory holding ``mentions.jsonl``.

    Returns:
        One :class:`GoldCase` per line, in file order.
    """
    rows = _read_jsonl(gold_dir / "mentions.jsonl")
    cases: list[GoldCase] = []
    for row in rows:
        cases.append(
            GoldCase(
                name=str(row.get("name") or ""),
                battle_slug=str(row.get("battle_slug") or ""),
                battle_name=str(row.get("battle_name") or row.get("battle_slug") or ""),
                year=row.get("year"),
                side_label=str(row.get("side_label") or "side"),
                polity=str(row.get("polity") or ""),
                expected_qid=row.get("expected_qid"),
                expected_verdict=str(row.get("expected_verdict") or ""),
            )
        )
    return cases


def load_candidates(gold_dir: Path = GOLD_DIR) -> dict[str, list[Candidate]]:
    """Load the cached candidates for every gold-set surface form.

    Args:
        gold_dir: The directory holding ``candidates.jsonl``.

    Returns:
        Surface form to candidates, as :func:`pipeline.resolvers.fetch_candidates`
        would have returned it, read from the cache instead of the network.
    """
    rows = _read_jsonl(gold_dir / "candidates.jsonl")
    by_name: dict[str, list[Candidate]] = {}
    for row in rows:
        name = str(row.get("name") or "")
        if not name:
            continue
        candidates = [_candidate_from_row(c) for c in row.get("candidates") or []]
        by_name[name] = candidates
    return by_name


def build_group(case: GoldCase) -> MentionGroup:
    """Build a one-mention group for a gold case, the way ``group_mentions`` would.

    Args:
        case: The gold case.

    Returns:
        A :class:`MentionGroup` of exactly one mention, matching the shape
        ``test_resolve_live.py``'s ``_group_for`` builds for a live matcher
        call.
    """
    from pipeline.resolvers.mentions import group_mentions

    context = BattleContext(slug=case.battle_slug, name=case.battle_name, year=case.year)
    mention = Mention(
        battle_slug=case.battle_slug,
        battle_name=case.battle_name,
        side_label=case.side_label,
        name=case.name,
        polity=case.polity,
        context=context,
    )
    groups = group_mentions([mention])
    return groups[0]


def score_case(
    case: GoldCase,
    candidates: list[Candidate],
    *,
    threshold: float,
    margin: float,
) -> str:
    """Classify one gold case's outcome under one threshold/margin combo.

    Args:
        case: The gold case.
        candidates: The cached candidates for this mention's surface form.
        threshold: The ``fuzzy_threshold`` to try.
        margin: The ``ambiguity_margin`` to try.

    Returns:
        One of ``correct_link``, ``wrong_link``, ``correct_new``,
        ``missed_link``, ``ambiguous``, ``correct_placeholder``, or
        ``unresolved`` for a shape score_case has no bucket for.
    """
    if case.expected_verdict == "placeholder":
        return "correct_placeholder"

    group = build_group(case)
    decision = match_group(group, candidates, threshold=threshold, margin=margin)

    if case.expected_verdict == "linked":
        if decision.status == "linked":
            return "correct_link" if decision.qid == case.expected_qid else "wrong_link"
        if decision.status == "new":
            return "missed_link"
        if decision.status == "ambiguous":
            return "ambiguous"
        return "unresolved"

    if case.expected_verdict == "new":
        if decision.status == "new":
            return "correct_new"
        if decision.status == "linked":
            return "wrong_link"
        if decision.status == "ambiguous":
            return "ambiguous"
        return "unresolved"

    if case.expected_verdict == "ambiguous_ok":
        if decision.status == "ambiguous":
            return "correct_link"
        if decision.status == "linked" and decision.qid == case.expected_qid:
            return "correct_link"
        if decision.status == "linked":
            return "wrong_link"
        if decision.status == "new":
            return "missed_link"
        return "unresolved"

    return "unresolved"


def float_range(low: float, high: float, step: float) -> list[float]:
    """Build an inclusive grid of floats.

    Args:
        low: The first value.
        high: The last value, included when it lands on the step.
        step: The spacing between values. Must be positive.

    Returns:
        Values from ``low`` to ``high``, inclusive, rounded to one decimal
        place to avoid float-accumulation drift in the printed table.

    Raises:
        ValueError: If ``step`` is not positive.
    """
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")

    values: list[float] = []
    current = low
    # A half-step epsilon so a high end that lands exactly on the step
    # (95.0 from 70.0 by 5.0) is not dropped by float accumulation.
    while current <= high + step / 2:
        values.append(round(current, 1))
        current += step
    return values


def sweep(
    cases: list[GoldCase],
    candidates_by_name: dict[str, list[Candidate]],
    *,
    thresholds: list[float],
    margins: list[float],
) -> list[SweepResult]:
    """Score every threshold/margin combo against the gold set.

    Args:
        cases: Every gold case.
        candidates_by_name: Cached candidates, keyed by surface form.
        thresholds: The ``fuzzy_threshold`` grid.
        margins: The ``ambiguity_margin`` grid.

    Returns:
        One :class:`SweepResult` per combo, in grid order (threshold-major).
    """
    results: list[SweepResult] = []

    for threshold in thresholds:
        for margin in margins:
            counts = {
                "correct_link": 0,
                "wrong_link": 0,
                "correct_new": 0,
                "missed_link": 0,
                "correct_placeholder": 0,
                "ambiguous": 0,
                "unresolved": 0,
            }
            for case in cases:
                candidates = candidates_by_name.get(case.name, [])
                outcome = score_case(case, candidates, threshold=threshold, margin=margin)
                counts[outcome] = counts.get(outcome, 0) + 1

            results.append(
                SweepResult(
                    threshold=threshold,
                    margin=margin,
                    correct_links=counts["correct_link"],
                    wrong_links=counts["wrong_link"],
                    correct_new=counts["correct_new"],
                    missed_links=counts["missed_link"],
                    correct_placeholders=counts["correct_placeholder"],
                    ambiguous_count=counts["ambiguous"],
                    total=len(cases),
                )
            )

    return results


def recommend(results: list[SweepResult]) -> SweepResult | None:
    """Pick the best combo: zero wrong links first, then fewest ambiguous.

    Args:
        results: Every combo the sweep scored.

    Returns:
        The best result, or None when ``results`` is empty.
    """
    if not results:
        return None
    return min(results, key=lambda r: (r.wrong_links, r.ambiguous_count, r.missed_links))


def _print_table(results: list[SweepResult]) -> None:
    """Print the sweep as a fixed-width table, flagging zero-wrong-link rows.

    Args:
        results: Every combo the sweep scored.
    """
    columns = (
        f"{'threshold':>9}  {'margin':>6}  {'wrong':>5}  "
        f"{'correct':>7}  {'missed':>6}  {'ambig':>5}  {'total':>5}"
    )
    print(columns)
    for result in results:
        marker = "  <-- zero wrong links" if result.wrong_links == 0 else ""
        print(
            f"{result.threshold:>9.1f}  {result.margin:>6.1f}  "
            f"{result.wrong_links:>5}  {result.correct_links:>7}  "
            f"{result.missed_links:>6}  {result.ambiguous_count:>5}  "
            f"{result.total:>5}{marker}"
        )


def _print_summary(results: list[SweepResult]) -> None:
    """Print the recommended combo and warn if none is safe.

    Args:
        results: Every combo the sweep scored.
    """
    best = recommend(results)
    if best is None:
        print("\nNo results to summarise -- the gold set is empty.")
        return

    zero_wrong = [r for r in results if r.wrong_links == 0]
    print()
    if not zero_wrong:
        print(
            "WARNING: no threshold/margin combo in this grid achieves zero wrong "
            "links. A false merge is undetectable downstream -- widen the grid or "
            "review the gold set before picking a default."
        )
    print(
        f"Recommended: threshold={best.threshold:.1f}, margin={best.margin:.1f} "
        f"(wrong_links={best.wrong_links}, ambiguous={best.ambiguous_count}, "
        f"missed_links={best.missed_links})"
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Arguments to parse, or None to read ``sys.argv``.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Sweep resolve fuzzy_threshold and ambiguity_margin against the gold set."
    )
    parser.add_argument(
        "--threshold-range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=list(_DEFAULT_THRESHOLD_RANGE),
        help="fuzzy_threshold range, inclusive (default: 70 95)",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=_DEFAULT_THRESHOLD_STEP,
        help="fuzzy_threshold step (default: 5)",
    )
    parser.add_argument(
        "--margin-range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=list(_DEFAULT_MARGIN_RANGE),
        help="ambiguity_margin range, inclusive (default: 2 12)",
    )
    parser.add_argument(
        "--margin-step",
        type=float,
        default=_DEFAULT_MARGIN_STEP,
        help="ambiguity_margin step (default: 2)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        metavar="PATH",
        help="also write the full results as JSON to this path",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the sweep and print its results.

    Args:
        argv: Arguments to parse, or None to read ``sys.argv``.
    """
    args = _parse_args(argv)

    cases = load_gold_cases(GOLD_DIR)
    candidates_by_name = load_candidates(GOLD_DIR)

    threshold_low, threshold_high = args.threshold_range
    margin_low, margin_high = args.margin_range
    thresholds = float_range(threshold_low, threshold_high, args.threshold_step)
    margins = float_range(margin_low, margin_high, args.margin_step)

    results = sweep(cases, candidates_by_name, thresholds=thresholds, margins=margins)

    _print_table(results)
    _print_summary(results)

    if args.json is not None:
        payload = [result.as_dict() for result in results]
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {len(payload)} results to {args.json}")


if __name__ == "__main__":
    main()
