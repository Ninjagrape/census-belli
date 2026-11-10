# ruff: noqa: F811  (pytest fixtures imported from test_report are redefined as arguments)
"""Coverage disclosures against the report fixture database.

Reuses the module fixtures of ``test_report`` (an evaluable run over the
inspector seed: three credited battles at -489, -30 and 1815, one commander
each, all sides reconciled from two sources). Each test changes rows inside the
``conn`` fixture's transaction, which is rolled back afterwards.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection

from pipeline.reporters.coverage import CoverageStats, EraCount, load_coverage
from pipeline.reporters.disclosures import coverage_disclosures
from pipeline.reporters.markdown import _esc
from tests.integration.test_report import (  # noqa: F401
    MIN_BATTLES,
    _csv_rows,
    _execute,
    _load,
    conn,
    engine,
    pytestmark,
    world,
)


def _coverage(conn: Connection, world: dict[str, Any]) -> CoverageStats:
    return load_coverage(conn, world["run_id"], min_battles=MIN_BATTLES)


def test_queries_on_the_fixture_return_the_hand_counted_numbers(
    conn: Connection, world: dict[str, Any]
) -> None:
    stats = _coverage(conn, world)

    assert (stats.rule_single_sides, stats.credited_sides) == (0, 3)
    assert stats.n_components == 2
    assert stats.extract_answers == 1
    assert (stats.corpus_battles, stats.first_century, stats.last_century) == (3, -500, 1800)
    assert stats.empty_spans == ((-400, -101), (0, 1799))
    assert (stats.n_headline, stats.n_generals_total) == (4, 6)
    assert stats.headline_by_era == (EraCount("ancient", 4),)
    assert [(e.era, e.multi_source, e.single_report, e.imputed) for e in stats.troop_eras] == [
        ("ancient", 4, 0, 0),
        ("Revolutionary and Napoleonic", 2, 0, 0),
    ]
    assert [(b.label, b.n_credited, b.n_wins) for b in stats.fame_bins] == [("3-4", 3, 3)]


def test_changed_rows_move_the_numbers_they_feed(conn: Connection, world: dict[str, Any]) -> None:
    gid = world["general_ids"]
    conn.execute(text("UPDATE battle_commanders SET attribution_method = 'rule_single'"))
    conn.execute(
        text("UPDATE general_skill_estimates SET n_battles = 5 WHERE general_id = :g"),
        {"g": gid["Miltiades"]},
    )
    conn.execute(
        text("UPDATE general_skill_estimates SET n_battles = 2 WHERE general_id = :g"),
        {"g": gid["Wellington"]},
    )
    conn.execute(
        text("UPDATE battle_war_details SET actual_outcome = 0.0 WHERE general_id = :g"),
        {"g": gid["Wellington"]},
    )
    conn.execute(
        text("UPDATE battle_sides SET est_troops_total = NULL WHERE battle_id = :b"),
        {"b": world["ids"]["b2"]},
    )
    conn.execute(
        text(
            "UPDATE battle_sides SET est_troops_method = 'single_report_debiased' "
            "WHERE battle_id = :b"
        ),
        {"b": world["ids"]["b0"]},
    )

    stats = _coverage(conn, world)

    assert stats.rule_single_sides == 3
    assert [(b.label, b.n_credited, b.n_wins) for b in stats.fame_bins] == [
        ("5-9", 1, 1),
        ("3-4", 1, 1),
        ("2", 1, 0),
    ]
    assert stats.n_headline == 3
    by_era = {e.era: (e.multi_source, e.single_report, e.imputed) for e in stats.troop_eras}
    assert by_era == {"ancient": (2, 2, 0), "Revolutionary and Napoleonic": (0, 0, 2)}
    text_out = "\n".join(coverage_disclosures(stats, min_battles=MIN_BATTLES))
    assert "3 of 3 credited sides use the rule_single attribution" in text_out
    assert "100.0% of those 1 battles, against 0.0% of 1" in text_out


def test_an_unknown_run_degrades_to_no_numbers_without_raising(
    conn: Connection, world: dict[str, Any]
) -> None:
    stats = load_coverage(conn, world["run_id"] + 10_000, min_battles=MIN_BATTLES)

    assert stats.rule_single_sides is None and stats.credited_sides is None
    assert stats.n_components is None and stats.corpus_battles is None
    assert stats.headline_by_era == () and stats.troop_eras == () and stats.fame_bins == ()
    sentences = coverage_disclosures(stats, min_battles=MIN_BATTLES)
    assert not any("credited sides" in s or "troop total" in s for s in sentences)


def test_the_stage_writes_the_sentences_and_the_gate_passes(
    conn: Connection, world: dict[str, Any], tmp_path: Path
) -> None:
    results, out = _execute(conn, tmp_path, world)

    gate = results["disclosures_match_db"]
    assert gate.passed, gate.message
    manifest = _load(out / "manifest.json")
    entry = next(g for g in manifest["gates"] if g["name"] == "disclosures_match_db")
    assert entry["passed"] and entry["actual"]["fresh"] >= 6
    payload = _load(out / "rankings.json")
    markdown = (out / "report.md").read_text(encoding="utf-8")
    block = [d for d in payload["disclosures"] if "credited sides" in d or "component" in d]
    assert any("0 of 3 credited sides use the rule_single attribution" in d for d in block)
    assert any("2 connected components" in d for d in block)
    assert all(_esc(d) in markdown for d in payload["disclosures"])


def test_the_gate_catches_a_database_that_changed_after_the_view_was_built(
    conn: Connection, world: dict[str, Any], tmp_path: Path, monkeypatch: Any
) -> None:
    from pipeline.stages import report as report_stage

    real = report_stage._fresh_coverage_text

    def drifted(c: Connection, view: Any) -> tuple[str, ...]:
        c.execute(text("UPDATE battle_commanders SET attribution_method = 'rule_single'"))
        return real(c, view)

    monkeypatch.setattr(report_stage, "_fresh_coverage_text", drifted)

    results, _ = _execute(conn, tmp_path, world)

    assert not results["disclosures_match_db"].passed


def test_rank_numbering_counts_headline_rows_and_keeps_the_stored_rank(
    conn: Connection, world: dict[str, Any], tmp_path: Path
) -> None:
    _, out = _execute(conn, tmp_path, world)

    rows = _csv_rows(out / "rankings.csv")
    headline = [r for r in rows if r["headline"] == "true"]
    assert [int(r["headline_rank"]) for r in headline] == [1, 2, 3, 4]
    assert [int(r["rank"]) for r in headline] == [1, 2, 3, 5]
    assert all(r["headline_rank"] == "" for r in rows if r["headline"] == "false")
    markdown = (out / "report.md").read_text(encoding="utf-8")
    assert any(line.startswith("| 4 | Scipio | 5 |") for line in markdown.splitlines())
    payload = _load(out / "rankings.json")
    scipio = next(g for g in payload["generals"] if g["name"] == "Scipio")
    assert (scipio["headline_rank"], scipio["rank"]) == (4, 5)


def test_the_stage_writes_the_coverage_figures_and_the_tab_and_the_gate_passes(
    conn: Connection, world: dict[str, Any], tmp_path: Path
) -> None:
    from pipeline.reporters.coverage_panels import PANEL_FILES

    results, out = _execute(conn, tmp_path, world)

    assert results["coverage_figures_written"].passed, results["coverage_figures_written"].message
    manifest = _load(out / "manifest.json")
    entry = next(g for g in manifest["gates"] if g["name"] == "coverage_figures_written")
    expected = entry["actual"]["expected"]
    assert expected and set(expected) <= set(PANEL_FILES.values())
    markdown = (out / "report.md").read_text(encoding="utf-8")
    assert "## Data coverage and caveats" in markdown
    for name in expected:
        assert (out / "figures" / name).read_bytes()[1:4] == b"PNG"
        assert f"figures/{name}" in markdown
    explorer = (out / "explorer.html").read_text(encoding="utf-8")
    assert 'id="tab-coverage"' in explorer
    assert "coverage_timeline.png" in expected


def test_a_missing_figure_fails_the_coverage_gate(
    conn: Connection, world: dict[str, Any], tmp_path: Path, monkeypatch: Any
) -> None:
    from pipeline.stages import report as report_stage

    monkeypatch.setattr(report_stage, "write_coverage_figures", lambda panels, directory: ())

    results, _ = _execute(conn, tmp_path, world)

    assert not results["coverage_figures_written"].passed


def test_a_general_listed_on_both_sides_of_a_battle_is_counted_once(
    conn: Connection, world: dict[str, Any]
) -> None:
    before = _coverage(conn, world)
    battle = world["ids"]["b0"]
    gid = world["general_ids"]["Miltiades"]
    own = conn.execute(
        text("SELECT side_id FROM battle_commanders WHERE battle_id = :b AND general_id = :g"),
        {"b": battle, "g": gid},
    ).scalar_one()
    other = conn.execute(
        text("SELECT side_id FROM battle_sides WHERE battle_id = :b AND side_id <> :s"),
        {"b": battle, "s": own},
    ).scalar_one()
    conn.execute(
        text(
            "INSERT INTO battle_commanders (battle_id, side_id, general_id, hierarchy_rank, "
            "attribution_weight, attribution_method) VALUES (:b, :s, :g, 0, 0.5, 'rule_single')"
        ),
        {"b": battle, "s": other, "g": gid},
    )

    after = _coverage(conn, world)

    assert after.credited_sides == before.credited_sides
    assert after.rule_single_sides == before.rule_single_sides
    assert after.fame_bins == before.fame_bins
    assert after.credit_source == before.credit_source


def test_a_failing_query_degrades_that_quantity_and_leaves_the_transaction_usable(
    conn: Connection, world: dict[str, Any], monkeypatch: Any
) -> None:
    from pipeline.reporters import coverage as coverage_module

    monkeypatch.setattr(
        coverage_module, "_TROOPS", text("SELECT year_astronomical FROM no_such_table")
    )

    stats = _coverage(conn, world)

    assert stats.troop_eras == ()
    assert stats.credited_sides == 3 and stats.fame_bins
    assert conn.execute(text("SELECT 1")).scalar() == 1


def test_the_gate_fails_when_the_independent_recount_disagrees(
    conn: Connection, world: dict[str, Any], tmp_path: Path, monkeypatch: Any
) -> None:
    from pipeline.stages import report as report_stage

    real = report_stage.recount

    def off_by_one(c: Connection, run_id: int, *, min_battles: int) -> dict[str, int | None]:
        counts = real(c, run_id, min_battles=min_battles)
        return {**counts, "credited_sides": (counts["credited_sides"] or 0) + 1}

    monkeypatch.setattr(report_stage, "recount", off_by_one)

    results, out = _execute(conn, tmp_path, world)

    assert not results["disclosures_match_db"].passed
    entry = next(
        g for g in _load(out / "manifest.json")["gates"] if g["name"] == "disclosures_match_db"
    )
    assert "credited_sides" in entry["actual"]["count_differences"]


def test_the_explorer_payload_carries_every_written_panel(
    conn: Connection, world: dict[str, Any], tmp_path: Path
) -> None:
    results, out = _execute(conn, tmp_path, world)

    gate = results["coverage_figures_written"]
    assert gate.passed, gate.message
    entry = next(
        g for g in _load(out / "manifest.json")["gates"] if g["name"] == "coverage_figures_written"
    )
    assert entry["actual"]["explorer_payload_mismatch"] == []
    assert entry["actual"]["explorer_missing_hosts"] == []
