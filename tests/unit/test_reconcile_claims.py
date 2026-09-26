"""
Unit tests for the reconcile stage's deterministic classification.

The cases worth pinning are the ones where a plausible implementation
quietly gets it wrong: guessing a claim's era from the battle's date instead
of leaving it unlabelled, reading a missing year as year zero, or merging two
independent sources into one lineage because their numbers happen to agree.
Each of those would understate the reconcile model's uncertainty in a way
nothing downstream would notice, which is exactly the failure mode this
package exists to avoid.

No database and no network: everything under test is pure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pipeline.reconcilers import (
    Report,
    SourceKey,
    assign_lineages,
    classify_regime,
    era_flag,
    roundness,
)

_GOLD_ARSHT_INFOBOX = (
    Path(__file__).resolve().parents[1] / "fixtures" / "gold" / "arsht" / "arsht_infobox.jsonl"
)


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _report(
    report_id: int,
    *,
    side_id: int = 1,
    battle_id: int = 1,
    source_id: int = 1,
    source_type: str = "wikipedia_infobox",
    url: str = "",
    quantity: str = "troops",
    branch: str = "total",
    reported_value: float = 10_000.0,
) -> Report:
    """Build a minimal, valid Report for the field(s) one test cares about."""
    return Report(
        report_id=report_id,
        side_id=side_id,
        battle_id=battle_id,
        source_id=source_id,
        source_key=SourceKey.from_row(source_type, url),
        source_type=source_type,
        quantity=quantity,  # type: ignore[arg-type]
        reported_value=reported_value,
        branch=branch,
    )


# ─── classify_regime ────────────────────────────────────────────────────────
def test_an_infobox_field_naming_ancient_sources_is_classified_as_an_ancient_claim() -> None:
    text = "250,000-1,000,000 (ancient sources) (See Size of Persian army)"
    assert classify_regime(text) == "ancient_claim"


def test_an_infobox_field_naming_modern_estimates_is_classified_as_modern_scholarly() -> None:
    text = "52,930-100,000 (modern estimates)"
    assert classify_regime(text) == "modern_scholarly"


def test_a_named_ancient_author_is_enough_without_the_words_ancient_sources() -> None:
    text = "Livy gives the figure as 25,000."
    # No occurrence of "ancient" or "source" anywhere in the string -- the
    # author's name alone must carry the classification.
    assert "ancient" not in text.lower()
    assert "source" not in text.lower()
    assert classify_regime(text) == "ancient_claim"


def test_citation_brackets_do_not_prevent_a_match() -> None:
    # A citation ref sits directly between the two words of the keyword
    # phrase. Without stripping brackets first, "modern" is never followed
    # by whitespace-then-"estimates", so the match would fail.
    text = "Modern[1] estimates 15,000"
    assert classify_regime(text) == "modern_scholarly"


def test_an_unmatched_context_is_unlabelled_rather_than_guessed_from_the_era() -> None:
    assert classify_regime("20,000 men") == "unlabelled"
    assert classify_regime(None) == "unlabelled"
    assert classify_regime("") == "unlabelled"


def test_the_real_infobox_corpus_classifies_without_error() -> None:
    # This is what keeps the classifier honest against real Wikipedia text
    # rather than against hand-picked examples: it must not raise on any of
    # the 382 distinct strength strings in the gold corpus, and enough of
    # them must carry a real marker that the patterns are doing something,
    # not just returning "unlabelled" for everything.
    rows = [json.loads(line) for line in _GOLD_ARSHT_INFOBOX.open(encoding="utf-8")]
    labelled = 0
    checked = 0
    for row in rows:
        for field in ("own_strength_text", "opp_strength_text"):
            text: Any = row.get(field)
            if not text or str(text).strip().lower() in ("nan", "none", ""):
                continue
            checked += 1
            regime = classify_regime(str(text))
            assert regime in (
                "modern_scholarly",
                "ancient_claim",
                "chronicle",
                "administrative_partisan",
                "staff_return",
                "unlabelled",
            )
            if regime != "unlabelled":
                labelled += 1
    assert checked > 0
    # 45 of 2,086 on the corpus as it stands. The figure is low because most
    # infoboxes state a number without saying whose it is, and because a field
    # naming two traditions at once now abstains rather than picking the first
    # marker it sees. Both are correct; the threshold is set just under the
    # measured value so a real loss of matching power fails the test while
    # ordinary corpus churn does not.
    assert labelled >= 40, (
        f"only {labelled} of {checked} strength fields carried a regime marker; "
        "the patterns have stopped matching real text"
    )


# ─── era_flag ───────────────────────────────────────────────────────────────
def test_a_battle_with_a_null_year_has_no_era_flag_rather_than_a_false_one() -> None:
    # (year or 0) < cutoff would read a NULL year as 1 BC and call an
    # undated battle ancient. Most undated battles are undated obscure
    # modern skirmishes, not lost antiquity, so that would be backwards.
    assert era_flag(None, 500) is None


def test_a_bc_year_is_ancient_and_an_ad_year_is_not() -> None:
    assert era_flag(-30, 500) == 1.0  # 31 BC, astronomical year -30
    assert era_flag(1900, 500) == 0.0


# ─── roundness ──────────────────────────────────────────────────────────────
def test_a_round_million_scores_higher_roundness_than_a_precise_five_figure_number() -> None:
    assert roundness(1_000_000) > roundness(52_930)


# ─── assign_lineages ────────────────────────────────────────────────────────
def test_a_wikidata_row_repeating_a_wikipedia_value_joins_its_lineage() -> None:
    wikipedia = _report(1, source_type="wikipedia_infobox", reported_value=100_000.0)
    wikidata = _report(2, source_type="wikidata", reported_value=100_500.0)  # within 1%
    lineages = assign_lineages([wikipedia, wikidata])
    assert lineages[1] == lineages[2]


def test_two_genuinely_independent_sources_that_happen_to_agree_stay_separate_lineages() -> None:
    # Over-merging is the opposite failure from under-merging, and just as
    # damaging: two real, independent sources agreeing is exactly the
    # evidence the reconcile model needs, and folding them into one lineage
    # would throw it away.
    first = _report(1, source_type="peer_reviewed", reported_value=100_000.0)
    second = _report(2, source_type="peer_reviewed", reported_value=100_000.0)
    lineages = assign_lineages([first, second])
    assert lineages[1] != lineages[2]


def test_values_differing_by_more_than_the_tolerance_stay_separate_lineages() -> None:
    wikipedia = _report(1, source_type="wikipedia_infobox", reported_value=100_000.0)
    wikidata = _report(2, source_type="wikidata", reported_value=200_000.0)
    lineages = assign_lineages([wikipedia, wikidata], relative_tolerance=0.01)
    assert lineages[1] != lineages[2]


# ─── Multi-regime fields ─────────────────────────────────────────────────────

# A real Wikipedia strength field, verbatim apart from the dash. One field,
# two traditions, and the extract stage turns it into two troop_reports that
# may share it as their context.
_GAUGAMELA_FIELD = (
    "52,930-100,000 (modern estimates) 250,000-1,000,000 (ancient sources) "
    "(See Size of Persian army)"
)
_YARMOUK_FIELD = (
    "15,000-150,000 (modern estimates) 100,000-200,000 (primary Arab sources) "
    "140,000 (primary Roman sources)"
)


def test_a_field_naming_two_traditions_is_unlabelled_rather_than_modern() -> None:
    # Returning the first match would label Herodotus' million
    # modern_scholarly -- the anchor, pinned to exactly zero bias -- so no
    # parameter would be free to absorb it and the side's estimate would be
    # dragged up an order of magnitude. Ambiguity must fall back to the era.
    assert classify_regime(_GAUGAMELA_FIELD) == "unlabelled"
    assert classify_regime(_YARMOUK_FIELD) == "unlabelled"


def test_the_ancient_figure_in_a_two_tradition_field_resolves_to_ancient() -> None:
    assert classify_regime(_GAUGAMELA_FIELD, value=1_000_000) == "ancient_claim"


def test_the_modern_figure_in_a_two_tradition_field_resolves_to_modern() -> None:
    assert classify_regime(_GAUGAMELA_FIELD, value=52_930) == "modern_scholarly"


def test_a_value_that_does_not_appear_leaves_a_multi_regime_field_unlabelled() -> None:
    # Guessing from a number the text does not contain would be inventing an
    # attribution, which is the error this whole function exists to avoid.
    assert classify_regime(_GAUGAMELA_FIELD, value=7) == "unlabelled"


def test_a_value_does_not_override_an_unambiguous_single_regime_field() -> None:
    single = "250,000 according to Herodotus"
    assert classify_regime(single) == "ancient_claim"
    assert classify_regime(single, value=250_000) == "ancient_claim"
