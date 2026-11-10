"""Coverage panels: data-to-series transforms, skip logic, figures, gate and explorer tab."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from structlog.testing import capture_logs

from pipeline.reporters.coverage import (
    CoverageStats,
    FameBin,
    TroopEra,
    with_component_sizes,
)
from pipeline.reporters.coverage_figures import write_coverage_figures
from pipeline.reporters.coverage_panels import (
    PANEL_FILES,
    build_panels,
    contraction,
    expected_figures,
)
from pipeline.reporters.coverage_series import (
    ComponentMarker,
    CreditSource,
    ReportedPoint,
    TimelineBin,
    credit_by_general,
)
from pipeline.reporters.explorer import render_explorer
from pipeline.reporters.exports import build_payload
from pipeline.reporters.manifest import coverage_figures_gate
from pipeline.reporters.markdown import render_markdown
from pipeline.reporters.records import RubinInfo
from pipeline.reporters.tables import split_generals
from tests.unit.reporters_helpers import estimate

STUB_JS = "/* plotly stub */"


def _headline(n: int = 3) -> Any:
    ests = [estimate(i, f"General {i}", rank=i, n_battles=5) for i in range(1, n + 1)]
    rubin = {
        i: RubinInfo(component_id=0 if i < n else 1, effective_battles=4.5) for i in range(1, n + 1)
    }
    headline, _ = split_generals(
        ests, rubin, min_battles=3, z=1.0, overlap_threshold=0.5, show_overlap=False
    )
    return headline, rubin


def _stats() -> CoverageStats:
    return CoverageStats(
        timeline=(TimelineBin(-500, 2), TimelineBin(1800, 3), TimelineBin(1825, 1)),
        empty_spans=((-400, 1799),),
        credit_source=(
            CreditSource(1, "llm", 2.0),
            CreditSource(1, "rule_single", 1.0),
            CreditSource(2, "weird_value", 1.5),
        ),
        sigma_skill=1.0,
        troop_eras=(
            TroopEra("ancient", 2, 1, 0),
            TroopEra("Revolutionary and Napoleonic", 0, 1, 1),
        ),
        reported_vs_reconciled=(
            ReportedPoint(-489, 20000.0, 15000.0),
            ReportedPoint(1815, 90000.0, 70000.0),
        ),
        fame_bins=(FameBin("5-9", 5, 4), FameBin("1", 4, 1), FameBin("3-4", 6, 3)),
    )


def _full() -> tuple[CoverageStats, Any]:
    headline, rubin = _headline()
    return with_component_sizes(_stats(), rubin, headline), headline


def test_contraction_is_one_minus_the_squared_sd_over_sigma() -> None:
    assert contraction(0.5, 1.0) == pytest.approx(0.75)
    assert contraction(1.0, 1.0) == pytest.approx(0.0)
    assert contraction(0.5, 0.0) is None
    assert contraction(float("nan"), 1.0) is None


def test_component_sizes_count_generals_per_component_and_mark_the_leaders() -> None:
    stats, _ = _full()

    assert stats.component_sizes == (2, 1)
    assert stats.top_markers == (
        ComponentMarker("General 1", 1, 0),
        ComponentMarker("General 2", 2, 0),
        ComponentMarker("General 3", 3, 1),
    )


def test_every_panel_is_built_from_full_stats() -> None:
    stats, headline = _full()

    panels = build_panels(stats, headline)

    assert list(panels) == list(PANEL_FILES)
    assert panels["timeline"]["bins"][0] == {"start": -500, "n": 2, "era": "ancient"}
    assert panels["timeline"]["empty_spans"] == [[-400, 1799]]
    assert panels["components"]["sizes"] == [2, 1]
    row = panels["credit_source"]["rows"][0]
    assert row["weights"] == {"llm": 2.0, "rule_single": 1.0}
    other = panels["credit_source"]["rows"][1]["weights"]
    assert other == {"other": 1.5}
    assert panels["credit_source"]["methods"] == ["llm", "rule_single", "other"]
    points = panels["info_vs_rank"]["points"]
    assert points[0]["contraction"] == pytest.approx(0.75)
    assert panels["info_vs_rank"]["line"] == 0.25
    assert panels["troop_provenance"]["order"] == ["ancient", "Revolutionary and Napoleonic"]
    assert [b["label"] for b in panels["fame_bias"]["bins"]] == ["1", "3-4", "5-9"]
    assert panels["fame_bias"]["bins"][2]["rate"] == pytest.approx(0.8)


def test_a_panel_without_data_is_skipped_with_a_warning_and_the_rest_survive() -> None:
    stats, headline = _full()
    gone = replace(stats, timeline=(), fame_bins=(), component_sizes=())

    with capture_logs() as logs:
        panels = build_panels(gone, headline)

    assert set(PANEL_FILES) - set(panels) == {"timeline", "fame_bias", "components"}
    skipped = {e["panel"] for e in logs if e["event"] == "report_coverage_panel_skipped"}
    assert skipped == {"timeline", "fame_bias", "components"}
    assert expected_figures(panels) == (
        PANEL_FILES["credit_source"],
        PANEL_FILES["info_vs_rank"],
        PANEL_FILES["troop_provenance"],
    )


def test_missing_sigma_draws_effective_battles_only_and_logs() -> None:
    stats, headline = _full()

    with capture_logs() as logs:
        panels = build_panels(replace(stats, sigma_skill=None), headline)

    assert all(p["contraction"] is None for p in panels["info_vs_rank"]["points"])
    assert any(e["event"] == "report_coverage_contraction_unavailable" for e in logs)


def test_missing_scatter_omits_only_that_subpanel() -> None:
    stats, headline = _full()

    with capture_logs() as logs:
        panels = build_panels(replace(stats, reported_vs_reconciled=()), headline)

    assert "scatter" not in panels["troop_provenance"]
    assert any(e["event"] == "report_coverage_subpanel_skipped" for e in logs)


def test_empty_stats_build_no_panels() -> None:
    assert build_panels(CoverageStats(), []) == {}


def test_credit_rows_are_grouped_by_general() -> None:
    rows = (
        CreditSource(1, "llm", 1.0),
        CreditSource(1, "manual", 0.5),
        CreditSource(2, "llm", 2.0),
    )

    assert credit_by_general(rows) == {1: {"llm": 1.0, "manual": 0.5}, 2: {"llm": 2.0}}


def test_figures_are_written_for_every_panel_and_a_bad_panel_is_skipped(tmp_path: Path) -> None:
    stats, headline = _full()
    panels = build_panels(stats, headline)

    names = write_coverage_figures(panels, tmp_path)

    assert names == expected_figures(panels)
    assert all((tmp_path / n).read_bytes().startswith(b"\x89PNG") for n in names)

    broken = {**panels, "fame_bias": {"bins": [{"nonsense": 1}]}}
    with capture_logs() as logs:
        written = write_coverage_figures(broken, tmp_path / "again")
    assert PANEL_FILES["fame_bias"] not in written
    assert len(written) == len(names) - 1
    assert any(e["event"] == "report_coverage_figure_failed" for e in logs)


def test_the_gate_passes_only_when_every_expected_figure_is_written_and_embedded() -> None:
    names = ["a.png", "b.png"]

    ok = coverage_figures_gate(
        expected=names, on_disk=names, in_markdown=names, explorer_has_tab=True
    )
    unwritten = coverage_figures_gate(
        expected=names, on_disk=["a.png"], in_markdown=names, explorer_has_tab=True
    )
    unembedded = coverage_figures_gate(
        expected=names, on_disk=names, in_markdown=["a.png"], explorer_has_tab=True
    )
    no_tab = coverage_figures_gate(
        expected=names, on_disk=names, in_markdown=names, explorer_has_tab=False
    )
    nothing = coverage_figures_gate(expected=[], on_disk=[], in_markdown=[], explorer_has_tab=True)

    assert ok["passed"] and nothing["passed"]
    assert not unwritten["passed"] and unwritten["actual"]["not_written"] == ["b.png"]
    assert not unembedded["passed"] and unembedded["actual"]["not_embedded"] == ["b.png"]
    assert not no_tab["passed"]


def test_markdown_embeds_coverage_figures_under_their_own_heading() -> None:
    from tests.unit.reporters_helpers import view

    v = view([estimate(1, rank=1)])

    text = render_markdown(v, [], ["coverage_timeline.png"])

    assert "## Data coverage and caveats" in text
    assert "![coverage_timeline](figures/coverage_timeline.png)" in text
    assert "No coverage data is available." in render_markdown(v, [])


def test_the_explorer_carries_the_coverage_tab_and_data_and_stays_offline() -> None:
    from tests.unit.reporters_helpers import view

    stats, headline = _full()
    panels = build_panels(stats, headline)
    payload = build_payload(view([estimate(1, rank=1)]), coverage=panels)

    page = render_explorer(payload, plotly_js=STUB_JS)

    assert 'id="tab-coverage"' in page and 'id="tabbtn-coverage"' in page
    assert "Data coverage and caveats" in page
    for key in PANEL_FILES:
        assert f"chart-cov-{key}" in page
    assert "http://" not in page and "https://" not in page
    data = page.split('id="report-data">', 1)[1].split("</script>", 1)[0]
    assert set(json.loads(data)["coverage"]) == set(PANEL_FILES)
