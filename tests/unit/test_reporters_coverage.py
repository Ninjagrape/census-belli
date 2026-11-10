"""Coverage disclosures: sentence formatting, graceful omission and the gate. No database."""

from __future__ import annotations

from dataclasses import replace

from structlog.testing import capture_logs

from pipeline.reporters.coverage import (
    CoverageStats,
    EraCount,
    FameBin,
    TroopEra,
    _empty_spans,
    era_label,
    with_component_counts,
)
from pipeline.reporters.disclosures import coverage_disclosures
from pipeline.reporters.manifest import disclosures_match_db_gate
from pipeline.reporters.records import RubinInfo
from pipeline.reporters.tables import split_generals
from tests.unit.reporters_helpers import estimate

FULL = CoverageStats(
    rule_single_sides=7,
    credited_sides=20,
    n_components=3,
    top_outside_largest=4,
    top_component_known=10,
    extract_answers=0,
    corpus_battles=40,
    first_century=-500,
    last_century=1800,
    empty_spans=((-400, -101), (0, 1799)),
    n_headline=9,
    n_generals_total=15,
    headline_by_era=(EraCount("ancient", 6), EraCount("Revolutionary and Napoleonic", 3)),
    fame_bins=(FameBin("5-9", 5, 4), FameBin("10+", 3, 2), FameBin("1", 6, 2), FameBin("2", 4, 1)),
    troop_eras=(TroopEra("ancient", 2, 3, 1), TroopEra("medieval", 1, 0, 4)),
)


def _joined(stats: CoverageStats) -> str:
    return "\n".join(coverage_disclosures(stats, min_battles=3))


def test_each_sentence_states_its_number() -> None:
    text = _joined(FULL)

    assert "7 of 20 credited sides use the rule_single attribution" in text
    assert "3 connected components" in text
    assert "4 of the 10 leading generals sit outside the largest one" in text
    assert "infoboxes only" in text
    assert "The 40 credited battles span 501 BC to AD 1899." in text
    assert "401 to 102 BC" in text and "1 BC to AD 1799" in text
    assert (
        "Of the 9 headline generals, 6 fought their first credited battle in the ancient era "
        "(before 500), 3 in the Revolutionary and Napoleonic era (1789 to 1815)"
    ) in text
    assert "75.0% of those 8 battles, against 30.0% of 10" in text
    assert "Of 11 sides in credited battles, 3 have a troop total reconciled" in text
    assert "ancient 4 of 6; medieval 4 of 5" in text
    assert "the 9 generals with at least 3 credited battles from 1 to 9" in text
    assert "all 15 credited generals" in text


def test_no_sentence_uses_a_dash_other_than_a_hyphen() -> None:
    text = _joined(FULL)

    assert not any(ch in text for ch in "–—‒―")


def test_an_empty_stats_object_yields_no_sentences_and_logs_each_omission() -> None:
    with capture_logs() as logs:
        out = coverage_disclosures(CoverageStats(), min_battles=3)

    assert out == ()
    omitted = {entry["item"] for entry in logs if entry["event"] == "report_disclosure_omitted"}
    assert omitted == {
        "rule_single",
        "components",
        "passage_pass",
        "corpus_shape",
        "troop_provenance",
        "rank_numbering",
    }


def test_an_undetermined_quantity_drops_only_its_own_sentence() -> None:
    stats = replace(FULL, rule_single_sides=None, fame_bins=(FameBin("5 or more", 8, 6),))

    text = _joined(stats)

    assert "rule_single" not in text
    assert "against" not in text
    assert "connected component" in text
    assert "troop total" in text


def test_the_components_sentence_omits_the_top_share_when_it_is_unknown() -> None:
    text = _joined(replace(FULL, top_outside_largest=None, top_component_known=None))

    assert "3 connected components." in text
    assert "leading generals" not in text


def test_the_passage_pass_sentence_does_not_invent_a_pending_count() -> None:
    text = _joined(replace(FULL, extract_answers=12))

    assert "12 answers from the extract stage" in text
    assert "not recorded in the database" in text
    assert "infoboxes only" not in text


def test_many_empty_spans_are_truncated_with_a_count() -> None:
    spans = tuple((i * 200, i * 200 + 99) for i in range(12))

    text = _joined(replace(FULL, empty_spans=spans))

    assert "and 4 more spans" in text


def test_empty_spans_are_the_unoccupied_runs_between_the_first_and_last_century() -> None:
    counts = {-500: 2, -100: 1, 0: 0, 300: 4}

    assert _empty_spans(counts) == ((-400, -101), (0, 299))
    assert _empty_spans({}) == ()


def test_era_label_cuts() -> None:
    assert era_label(-489) == "ancient"
    assert era_label(500) == "medieval"
    assert era_label(1788) == "early modern"
    assert era_label(1789) == "Revolutionary and Napoleonic"
    assert era_label(1816) == "later nineteenth century"
    assert era_label(1914) == "twentieth century and later"


def test_headline_rank_counts_headline_rows_from_one_and_stored_rank_is_kept() -> None:
    ests = [
        estimate(1, rank=1, n_battles=5),
        estimate(2, rank=2, n_battles=2),
        estimate(3, rank=3, n_battles=3),
        estimate(4, rank=723, n_battles=4),
    ]

    headline, under = split_generals(
        ests, {}, min_battles=3, z=1.0, overlap_threshold=0.5, show_overlap=False
    )

    assert [(g.headline_rank, g.rank) for g in headline] == [(1, 1), (2, 3), (3, 723)]
    assert [g.headline_rank for g in under] == [None]


def test_component_counts_use_component_zero_as_the_largest() -> None:
    ests = [estimate(i, rank=i) for i in range(1, 5)]
    rubin = {
        1: RubinInfo(component_id=0),
        2: RubinInfo(component_id=1),
        3: RubinInfo(component_id=0),
    }
    headline, _ = split_generals(
        ests, rubin, min_battles=3, z=1.0, overlap_threshold=0.5, show_overlap=False
    )

    out = with_component_counts(CoverageStats(), headline)

    assert (out.top_outside_largest, out.top_component_known) == (1, 3)


def test_component_counts_stay_unset_without_component_ids() -> None:
    headline, _ = split_generals(
        [estimate(1, rank=1)], {}, min_battles=3, z=1.0, overlap_threshold=0.5, show_overlap=False
    )

    assert with_component_counts(CoverageStats(), headline) == CoverageStats()


def test_the_gate_passes_when_fresh_reported_and_written_agree() -> None:
    sentences = coverage_disclosures(FULL, min_battles=3)

    gate = disclosures_match_db_gate(
        fresh=sentences, reported=sentences, in_json=list(sentences), in_markdown=list(sentences)
    )

    assert gate["passed"] is True and gate["name"] == "disclosures_match_db"


def test_the_gate_fails_when_a_reported_number_differs_from_the_recomputation() -> None:
    fresh = coverage_disclosures(FULL, min_battles=3)
    stale = coverage_disclosures(replace(FULL, rule_single_sides=8), min_battles=3)

    gate = disclosures_match_db_gate(
        fresh=fresh, reported=stale, in_json=list(stale), in_markdown=list(stale)
    )

    assert gate["passed"] is False
    assert len(gate["actual"]["stale"]) == 1 and "8 of 20" in gate["actual"]["stale"][0]


def test_the_gate_fails_when_a_sentence_did_not_reach_a_file() -> None:
    fresh = coverage_disclosures(FULL, min_battles=3)

    gate = disclosures_match_db_gate(
        fresh=fresh, reported=fresh, in_json=list(fresh[1:]), in_markdown=list(fresh)
    )

    assert gate["passed"] is False and gate["actual"]["missing_json"] == 1


def test_singular_counts_are_not_pluralised() -> None:
    one = replace(FULL, n_components=1, extract_answers=1, empty_spans=((-400, -101),) * 10)
    text = _joined(one)

    assert "1 connected component." in text
    assert "1 answer from the extract stage's article-passage pass is stored" in text
    assert "and 2 more spans" in text
    assert "(s)" not in _joined(FULL)


def test_spans_are_written_with_bc_and_ad_only_where_needed() -> None:
    from pipeline.reporters.disclosures import _span

    assert _span(0, 1799) == "1 BC to AD 1799"
    assert _span(-400, -101) == "401 to 102 BC"
    assert _span(1300, 1499) == "1300 to 1499"
    assert _span(-5, 5) == "6 BC to AD 5"
    assert _span(0, 0) == "1 BC"
    assert _span(1815, 1815) == "1815"


def test_the_era_sentence_lists_only_occupied_eras_and_quotes_each_edge_once() -> None:
    text = _joined(
        replace(FULL, headline_by_era=(EraCount("medieval", 2), EraCount("early modern", 1)))
    )

    assert (
        "2 fought their first credited battle in the medieval era (500 to 1499), "
        "1 in the early modern era (1500 to 1788)" in text
    )
    assert "ancient" not in text.split("headline generals")[1].split(".")[0]


def test_report_eras_are_a_named_constant_distinct_from_the_reconcile_bins() -> None:
    from pipeline.reporters.coverage import REPORT_ERAS

    assert [start for start, _ in REPORT_ERAS] == [500, 1500, 1789, 1816, 1914]
