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

import pytest

from pipeline.reconcilers import (
    Report,
    SourceKey,
    assign_lineages,
    classify_regime,
    era_flag,
    merge_cross_regime_duplicates,
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
    claim_regime: str = "unlabelled",
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
        claim_regime=claim_regime,
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


# ─── merge_cross_regime_duplicates ──────────────────────────────────────────
def test_an_unlabelled_restatement_of_a_labelled_figure_is_dropped() -> None:
    attributed = _report(1, reported_value=1_000_000.0, claim_regime="ancient_claim")
    infobox = _report(2, source_id=2, reported_value=1_000_000.0)
    kept = merge_cross_regime_duplicates([attributed, infobox])
    assert [r.report_id for r in kept] == [1]


def test_a_near_match_within_one_percent_is_still_a_duplicate() -> None:
    attributed = _report(1, reported_value=1_000_000.0, claim_regime="ancient_claim")
    infobox = _report(2, source_id=2, reported_value=1_005_000.0)
    assert [r.report_id for r in merge_cross_regime_duplicates([attributed, infobox])] == [1]


def test_an_unlabelled_figure_with_no_labelled_match_is_kept() -> None:
    attributed = _report(1, reported_value=1_000_000.0, claim_regime="ancient_claim")
    other = _report(2, source_id=2, reported_value=47_000.0)
    assert len(merge_cross_regime_duplicates([attributed, other])) == 2


def test_the_merge_does_not_cross_sides_or_quantities() -> None:
    attributed = _report(1, reported_value=1_000_000.0, claim_regime="ancient_claim")
    other_side = _report(2, side_id=2, source_id=2, reported_value=1_000_000.0)
    casualties = _report(3, source_id=3, quantity="casualties", reported_value=1_000_000.0)
    assert len(merge_cross_regime_duplicates([attributed, other_side, casualties])) == 3


def test_two_agreeing_unlabelled_rows_stay_independent() -> None:
    first = _report(1, reported_value=50_000.0)
    second = _report(2, source_id=2, reported_value=50_000.0)
    assert len(merge_cross_regime_duplicates([first, second])) == 2


def test_two_agreeing_labelled_rows_stay_independent() -> None:
    first = _report(1, reported_value=50_000.0, claim_regime="chronicle")
    second = _report(2, source_id=2, reported_value=50_000.0, claim_regime="ancient_claim")
    assert len(merge_cross_regime_duplicates([first, second])) == 2


# --- Modern wording variants and ancient-tradition wording ----------------------
@pytest.mark.parametrize(
    "text",
    [
        "30,000 (modern estimation)",
        "5,000 (modern est.)",
        "Modern estimate: 108,000–158,000",
        "~7,000 (Modern estimate)",
    ],
)
def test_modern_wording_variants_are_modern_scholarly(text: str) -> None:
    assert classify_regime(text) == "modern_scholarly"


@pytest.mark.parametrize(
    "text",
    [
        "70,000 (Muslim source)",
        "80,000 (Islamic estimates)",
        "70,000 (Arab estimates)",
        "90,000 (Byzantine accounts)",
        "50,000 (Roman tradition)",
        "20,000 (Greek sources)",
        "100,000 (Persian estimate)",
    ],
)
def test_ancient_tradition_wording_is_an_ancient_claim(text: str) -> None:
    assert classify_regime(text) == "ancient_claim"


def test_a_bare_tradition_word_without_source_wording_is_not_ancient() -> None:
    assert classify_regime("40,000 Roman soldiers") == "unlabelled"


def test_hazir_field_labels_each_figure_by_the_marker_that_follows_it() -> None:
    field = "70,000 (Muslim source) ~7,000 (Modern estimate)"
    assert classify_regime(field, value=70_000) == "ancient_claim"
    assert classify_regime(field, value=7_000) == "modern_scholarly"


def test_chains_field_labels_both_range_bounds_modern_and_the_total_ancient() -> None:
    field = "80,000 (Islamic estimates) 15,000–20,000 (modern estimates)"
    assert classify_regime(field, value=80_000) == "ancient_claim"
    assert classify_regime(field, value=15_000) == "modern_scholarly"
    assert classify_regime(field, value=20_000) == "modern_scholarly"


# ─── classify_regime: per-figure bracket binding ────────────────────────────
def test_a_bracket_naming_two_regimes_binds_nothing_and_the_nearer_marker_wins() -> None:
    text = "30,000 (Plutarch; modern historians doubt this)"
    assert classify_regime(text, value=30_000.0) == "ancient_claim"


def test_gaugamela_figures_each_take_their_own_bracket() -> None:
    text = "52,930-100,000 (modern estimates) 250,000-1,000,000 (ancient sources)"
    assert classify_regime(text, value=100_000.0) == "modern_scholarly"
    assert classify_regime(text, value=52_930.0) == "modern_scholarly"
    assert classify_regime(text, value=250_000.0) == "ancient_claim"
    assert classify_regime(text, value=1_000_000.0) == "ancient_claim"


def test_a_yarmouk_like_field_binds_each_figure_to_its_bracket() -> None:
    text = "80,000 (Islamic estimates) 15,000-40,000 (modern estimates)"
    assert classify_regime(text, value=80_000.0) == "ancient_claim"
    assert classify_regime(text, value=15_000.0) == "modern_scholarly"
    assert classify_regime(text, value=40_000.0) == "modern_scholarly"


def test_a_yarmouk_like_field_without_brackets_stays_unlabelled_without_a_value() -> None:
    text = "Muslim sources 80,000 modern estimates 15,000-40,000"
    assert classify_regime(text) == "unlabelled"


# ─── classify_regime: a figure's own unrecognised source label ──────────────
def test_a_figure_with_its_own_unrecognised_source_label_borrows_no_other_marker() -> None:
    text = "100,000 (Carthaginian claim) 20,000 (modern estimates)"
    assert classify_regime(text, value=100_000.0) == "unlabelled"
    assert classify_regime(text, value=20_000.0) == "modern_scholarly"


@pytest.mark.parametrize(
    "label", ["Chinese sources", "Carthaginian accounts", "Frankish chronicles", "Gallic tradition"]
)
def test_a_capitalised_people_source_label_in_brackets_is_an_ancient_claim(label: str) -> None:
    text = f"100,000 ({label}) 20,000 (modern estimates)"
    assert classify_regime(text, value=100_000.0) in ("ancient_claim", "chronicle")
    assert classify_regime(text, value=20_000.0) == "modern_scholarly"


def test_chinese_sources_are_ancient_claim_and_frankish_chronicles_a_chronicle() -> None:
    assert classify_regime("100,000 (Chinese sources)", value=100_000.0) == "ancient_claim"
    assert classify_regime("5,000 (Frankish chronicles)", value=5_000.0) == "chronicle"


def test_modern_sources_in_brackets_stay_modern_not_ancient() -> None:
    assert classify_regime("100,000 (Modern sources)", value=100_000.0) == "modern_scholarly"


def test_an_unlabelling_bracket_does_not_swallow_a_later_modern_marker_for_an_estimate_note() -> (
    None
):
    assert classify_regime("30,000 (est.) per modern historians", value=30_000.0) == (
        "modern_scholarly"
    )
