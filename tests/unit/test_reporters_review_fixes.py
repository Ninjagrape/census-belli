"""Code-review fixes to the report stage's coverage work. No database."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine, text
from structlog.testing import capture_logs

from pipeline.reporters.coverage import CoverageStats, EraCount, TroopEra, _troop_slot
from pipeline.reporters.coverage_check import differences, used_counts
from pipeline.reporters.coverage_panels import build_panels
from pipeline.reporters.coverage_series import TimelineBin, guarded
from pipeline.reporters.disclosures import coverage_disclosures
from pipeline.reporters.explorer_coverage import COVERAGE_JS
from pipeline.reporters.explorer_script import EXPLORER_JS
from pipeline.reporters.figures import write_forest_plot
from pipeline.reporters.manifest import coverage_figures_gate, disclosures_match_db_gate
from pipeline.reporters.tables import split_generals
from tests.unit.reporters_helpers import estimate

COUNTS = {"credited_sides": 20, "rule_single_sides": 7, "n_headline": 9, "credited_battles": 40}


def _joined(stats: CoverageStats) -> str:
    return "\n".join(coverage_disclosures(stats, min_battles=3))


def test_gate_fails_when_an_independent_recount_differs() -> None:
    gate = disclosures_match_db_gate(
        fresh=["a"],
        reported=["a"],
        in_json=["a"],
        in_markdown=["a"],
        used_counts=COUNTS,
        recounted={**COUNTS, "rule_single_sides": 8},
    )

    assert gate["passed"] is False
    assert gate["actual"]["count_differences"] == {"rule_single_sides": [7, 8]}


def test_gate_passes_when_recounts_agree() -> None:
    gate = disclosures_match_db_gate(
        fresh=["a"],
        reported=["a"],
        in_json=["a"],
        in_markdown=["a"],
        used_counts=COUNTS,
        recounted=COUNTS,
    )

    assert gate["passed"] is True


def test_gate_fails_on_credited_battles_without_any_coverage_sentence() -> None:
    gate = disclosures_match_db_gate(
        fresh=[], reported=[], in_json=[], in_markdown=[], used_counts=COUNTS, recounted=COUNTS
    )

    assert gate["passed"] is False and gate["actual"]["no_sentences_despite_battles"] is True


def test_used_counts_and_differences_handle_missing_stats() -> None:
    assert used_counts(None)["credited_sides"] is None
    assert differences({"credited_sides": None}, {"credited_sides": 2}) == {
        "credited_sides": (None, 2)
    }


def test_figures_gate_checks_explorer_payload_panels_and_hosts() -> None:
    base: dict[str, Any] = {
        "expected": ["a.png"],
        "on_disk": ["a.png"],
        "in_markdown": ["a.png"],
        "explorer_has_tab": True,
        "written_panels": ["timeline", "components"],
    }

    ok = coverage_figures_gate(
        **base,
        explorer_panels=["components", "timeline"],
        explorer_hosts=["timeline", "components"],
    )
    payload = coverage_figures_gate(
        **base, explorer_panels=["timeline"], explorer_hosts=["timeline", "components"]
    )
    host = coverage_figures_gate(
        **base, explorer_panels=["timeline", "components"], explorer_hosts=["timeline"]
    )

    assert ok["passed"] is True
    assert payload["passed"] is False
    assert payload["actual"]["explorer_payload_mismatch"] == ["components"]
    assert host["passed"] is False
    assert host["actual"]["explorer_missing_hosts"] == ["components"]


def test_forest_plot_labels_use_the_headline_rank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import matplotlib.axes

    seen: list[list[str]] = []
    real = matplotlib.axes.Axes.set_yticklabels

    def spy(self: Any, labels: Any, *a: Any, **k: Any) -> Any:
        seen.append(list(labels))
        return real(self, labels, *a, **k)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_yticklabels", spy)
    ests = [
        estimate(1, "A", rank=1),
        estimate(2, "Thin", rank=2, n_battles=1),
        estimate(3, "C", rank=3),
    ]
    headline, _ = split_generals(
        ests, {}, min_battles=3, z=1.0, overlap_threshold=0.5, show_overlap=False
    )

    write_forest_plot(headline, tmp_path, 0.95)

    assert seen[0] == ["1. A", "2. C"]


def test_explorer_ranks_use_headline_rank_and_resize_hidden_charts() -> None:
    assert "g.headline_rank" in EXPLORER_JS.split("function rankText")[1].split("\n")[0]
    assert "Plotly.Plots.resize" in EXPLORER_JS and "resizeRankingCharts" in EXPLORER_JS
    assert "Plotly.Plots.resize" in COVERAGE_JS and "resizeRankingCharts" in COVERAGE_JS


def test_timeline_bins_take_the_era_of_their_midpoint() -> None:
    stats = CoverageStats(
        timeline=(TimelineBin(475, 1), TimelineBin(1500, 1), TimelineBin(1775, 1))
    )

    eras = [b["era"] for b in build_panels(stats, ())["timeline"]["bins"]]

    # Bin 475 has midpoint 487 (before the 500 cut); bin 1775 has midpoint 1787 (before 1789).
    assert eras == ["ancient", "early modern", "early modern"]


def test_span_uses_the_first_and_last_credited_year() -> None:
    stats = CoverageStats(
        corpus_battles=3, first_century=-500, last_century=1800, first_year=-489, last_year=1815
    )

    assert "span 490 BC to AD 1815." in _joined(stats)


def test_undated_headline_generals_are_counted_so_the_eras_add_up() -> None:
    stats = CoverageStats(n_headline=9, headline_by_era=(EraCount("ancient", 6),))

    assert "and 3 have no dated credited battle" in _joined(stats)
    full = replace(stats, headline_by_era=(EraCount("ancient", 9),))
    assert "no dated credited battle" not in _joined(full)


@pytest.mark.parametrize(
    ("method", "has_total", "slot"),
    [
        ("source_disagreement", True, 0),
        ("single_report_debiased", True, 1),
        (None, False, 2),
        ("source_disagreement", False, 2),
        ("something_new", True, 3),
        (None, True, 3),
        ("bound_only", True, 4),
        ("bound_only", False, 2),
    ],
)
def test_troop_provenance_buckets(method: str | None, has_total: bool, slot: int) -> None:
    assert _troop_slot(method, has_total) == slot


def test_other_method_sides_get_their_own_clause_only_when_present() -> None:
    plain = CoverageStats(troop_eras=(TroopEra("ancient", 2, 1, 1),))
    other = CoverageStats(troop_eras=(TroopEra("ancient", 2, 1, 1, 2),))

    assert "another reconcile method" not in _joined(plain)
    shown = _joined(other)
    assert "Of 6 sides" in shown
    assert "2 sides carry a total from another reconcile method" in shown
    assert build_panels(other, ())["troop_provenance"]["eras"][0]["other"] == 2


def test_bound_only_sides_get_their_own_clause_only_when_present() -> None:
    plain = CoverageStats(troop_eras=(TroopEra("ancient", 2, 1, 1),))
    bound = CoverageStats(troop_eras=(TroopEra("ancient", 2, 1, 1, bound_only=1),))

    assert "rest only on an upper or lower bound" not in _joined(plain)
    assert "(single report or imputed)" in _joined(plain)
    shown = _joined(bound)
    assert "Of 5 sides" in shown
    assert "1 side rests only on an upper or lower bound ('up to' or 'at least')" in shown
    assert "(single report, bound only or imputed)" in shown
    assert build_panels(bound, ())["troop_provenance"]["eras"][0]["bound"] == 1


def test_troop_clauses_agree_with_a_count_of_one() -> None:
    era = TroopEra("ancient", multi_source=1, single_report=1, imputed=1, other=1, bound_only=1)
    stats = CoverageStats(troop_eras=(era,))

    shown = _joined(stats)
    assert "1 side rests only on an upper or lower bound" in shown
    assert "1 side carries a total from another reconcile method and is counted" in shown
    assert "1 has a troop total reconciled from two or more independent sources" in shown
    assert "1 rests on a single report" in shown
    assert "1 has no reported total and is imputed" in shown


def test_troop_clauses_agree_with_a_count_of_two() -> None:
    era = TroopEra("ancient", multi_source=2, single_report=2, imputed=2, other=2, bound_only=2)
    stats = CoverageStats(troop_eras=(era,))

    shown = _joined(stats)
    assert "2 sides rest only on an upper or lower bound" in shown
    assert "2 sides carry a total from another reconcile method and are counted" in shown
    assert "2 have a troop total reconciled from two or more independent sources" in shown
    assert "2 rest on a single report" in shown
    assert "2 have no reported total and are imputed" in shown


def test_guarded_logs_and_returns_the_default_on_a_database_error() -> None:
    engine = create_engine("sqlite://")
    with engine.connect() as conn, capture_logs() as logs:
        value = guarded(
            conn, "missing_table", lambda: conn.execute(text("SELECT * FROM nope")).all(), ()
        )
        assert conn.execute(text("SELECT 1")).scalar() == 1

    assert value == ()
    event = next(e for e in logs if e["event"] == "report_coverage_unavailable")
    assert event["query"] == "missing_table" and event["error"]
