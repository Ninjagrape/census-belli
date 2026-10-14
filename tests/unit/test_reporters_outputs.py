"""Exports, the explorer page, report.md and the manifest, with no database."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

import pytest

from pipeline.quality import Severity, _check_evaluation_gate
from pipeline.reporters.explorer import embed_json, render_explorer
from pipeline.reporters.exports import (
    CSV_COLUMNS,
    build_payload,
    read_csv_rows,
    write_rankings_csv,
    write_rankings_json,
)
from pipeline.reporters.manifest import (
    build_manifest,
    evaluation_current_gate,
    export_matches_db_gate,
    write_manifest,
)
from pipeline.reporters.markdown import _esc, render_markdown
from tests.unit.reporters_helpers import battle, estimate, evaluation, params, view

HOSTILE = "</script><b>x</b>"
STUB_JS = "/* plotly stub */"


def _data_block(page: str) -> str:
    match = re.search(
        r'<script type="application/json" id="report-data">(.*?)</script>', page, re.S
    )
    assert match is not None
    return match.group(1)


def _view(**kwargs: Any):  # type: ignore[no-untyped-def]
    ests = [
        estimate(1, HOSTILE, rank=1, total_war=1.23456789, war_sd=0.123456),
        estimate(2, "Plain", rank=2, total_war=0.9, n_battles=4),
        estimate(3, "Thin", rank=3, n_battles=1),
    ]
    battles = {1: (battle(1, 10, 0.5), battle(1, 11, 0.73456789))}
    return view(ests, battles=battles, evaluation_info=evaluation(Path(".")), **kwargs)


# ─── explorer ────────────────────────────────────────────────────────────────


def test_a_hostile_general_name_cannot_break_out_of_the_inlined_json() -> None:
    page = render_explorer(build_payload(_view()), plotly_js=STUB_JS)

    assert "<b>x</b>" not in page
    assert html.escape(HOSTILE) not in page
    # the three script elements (data, plotly, explorer) and no extra closer
    assert page.count("</script>") == 3
    names = [g["name"] for g in json.loads(_data_block(page))["generals"]]
    assert HOSTILE in names


def test_embed_json_escapes_angle_brackets_and_ampersands() -> None:
    out = embed_json({"a": "<!-- & </script>"})

    assert "<" not in out and ">" not in out and "&" not in out
    assert json.loads(out) == {"a": "<!-- & </script>"}


def test_every_disclosure_appears_in_the_explorer_and_the_report() -> None:
    v = _view()
    page = render_explorer(build_payload(v), plotly_js=STUB_JS)
    report = render_markdown(v, [])

    assert v.disclosures
    for disclosure in v.disclosures:
        assert html.escape(disclosure, quote=False) in page
        assert _esc(disclosure) in report
        assert disclosure in json.dumps(build_payload(v), ensure_ascii=False)


def test_the_explorer_builds_text_with_textcontent_only() -> None:
    from pipeline.reporters.explorer_script import EXPLORER_JS

    assert "innerHTML" not in EXPLORER_JS
    assert "document.write" not in EXPLORER_JS


def test_the_provisional_banner_is_static_html_with_each_reason() -> None:
    v = _view(evaluation_reason="stale evaluation report: evaluated run 3, reported run 7")
    page = render_explorer(build_payload(v), plotly_js=STUB_JS)

    assert "PROVISIONAL" in page
    assert "stale evaluation report" in page


def test_a_clean_view_has_no_banner() -> None:
    v = view([estimate(1, rank=1)], evaluation_info=evaluation(Path(".")))

    assert "PROVISIONAL" not in render_explorer(build_payload(v), plotly_js=STUB_JS)
    assert "PROVISIONAL" not in render_markdown(v, [])


# ─── report.md ───────────────────────────────────────────────────────────────


def test_report_md_escapes_markdown_and_html_in_names() -> None:
    report = render_markdown(_view(), [])

    assert "<b>" not in report
    assert "&lt;" in report


def test_report_md_shows_no_rank_for_under_determined_generals() -> None:
    report = render_markdown(_view(), [])
    thin_section = report.split("## Generals without a rank")[1].split("## Evaluation")[0]

    assert "| Thin | 1 |" in thin_section
    assert "Rank" not in thin_section


def test_report_md_labels_say_expected_not_win_probability() -> None:
    report = render_markdown(_view(), [])

    assert "Expected WAR" in report
    assert "win probability" in report  # only in the disclosure that rules it out


def test_report_md_marks_evaluation_unavailable_instead_of_dropping_it() -> None:
    v = view([estimate(1, rank=1)], evaluation_reason="no evaluation report for model run 7")

    report = render_markdown(v, [])

    assert "## Evaluation" in report
    assert "Not available" in report


def test_report_md_embeds_figures_with_forward_slash_paths() -> None:
    report = render_markdown(_view(), ["forest_top.png", "calibration.png"])

    assert "(figures/forest_top.png)" in report
    assert "\\" not in re.findall(r"!\[[^\]]*\]\(([^)]*)\)", report)[0]


def test_user_facing_text_has_no_long_dashes() -> None:
    v = _view()
    text = render_markdown(v, []) + render_explorer(build_payload(v), plotly_js=STUB_JS)

    assert "—" not in text and "–" not in text


# ─── exports ─────────────────────────────────────────────────────────────────


def test_csv_rounds_floats_and_blanks_the_rank_of_under_determined_rows(tmp_path: Path) -> None:
    v = view(
        [estimate(1, rank=1, total_war=1.23456789), estimate(2, rank=2, n_battles=1)],
        report_params=params(export_float_precision=2),
    )

    rows_written = write_rankings_csv(tmp_path / "r.csv", v)
    rows = read_csv_rows(tmp_path / "r.csv")

    assert rows_written == 2
    assert list(rows[0]) == list(CSV_COLUMNS)
    assert rows[0]["total_war"] == "1.23"
    assert (rows[0]["headline"], rows[0]["rank"]) == ("true", "1")
    assert (rows[1]["headline"], rows[1]["rank"]) == ("false", "")


def test_csv_missing_values_are_empty_cells(tmp_path: Path) -> None:
    v = view([estimate(1, rank=1, total_war=None, war_sd=None)])

    write_rankings_csv(tmp_path / "r.csv", v)

    row = read_csv_rows(tmp_path / "r.csv")[0]
    assert row["total_war"] == "" and row["war_normal_approx_lo"] == ""


def test_json_round_trips_at_full_float_precision(tmp_path: Path) -> None:
    v = _view()
    payload = build_payload(v)

    write_rankings_json(tmp_path / "r.json", payload)
    back = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))

    first = back["generals"][0]
    assert back == json.loads(json.dumps(payload))
    assert first["total_war"] == 1.23456789
    assert first["name"] == HOSTILE
    assert sum(b["expected_war"] for b in first["battles"]) == pytest.approx(0.5 + 0.73456789)


def test_json_keeps_every_battle_and_the_disclosures() -> None:
    payload = build_payload(_view())

    assert len(payload["generals"][0]["battles"]) == 2
    assert payload["disclosures"]
    assert payload["provisional"] is False
    assert payload["n_headline"] == 2 and payload["n_under_determined"] == 1


def test_the_json_holds_every_general_and_only_the_explorer_is_capped(tmp_path: Path) -> None:
    ests = [estimate(i, rank=i) for i in range(1, 6)]
    v = view(ests, report_params=params(explorer_max_generals=2))

    assert len(build_payload(v)["generals"]) == 5
    capped = build_payload(v, limit=v.params.explorer_max_generals)
    assert len(capped["generals"]) == 2
    assert (capped["n_generals_shown"], capped["n_generals_total"]) == (2, 5)
    assert write_rankings_csv(tmp_path / "r.csv", v) == 5


def test_the_explorer_says_how_many_generals_it_shows_only_when_capped() -> None:
    ests = [estimate(i, rank=i) for i in range(1, 6)]
    v = view(ests, report_params=params(explorer_max_generals=2))

    capped = render_explorer(build_payload(v, limit=2), plotly_js=STUB_JS)
    full = render_explorer(build_payload(v), plotly_js=STUB_JS)

    assert "Showing 2 of 5 generals" in capped
    assert "Showing" not in full.split("</header>")[1].split("<script")[0]


def test_overlap_is_null_in_json_and_blank_in_csv_when_not_comparable(tmp_path: Path) -> None:
    v = view([estimate(1, rank=1, total_war=1.0), estimate(2, rank=2, total_war=5.0)])

    payload = build_payload(v)
    write_rankings_csv(tmp_path / "r.csv", v)
    rows = read_csv_rows(tmp_path / "r.csv")

    assert payload["generals"][0]["overlaps_next_rank"] is False
    assert payload["generals"][1]["overlaps_next_rank"] is None
    assert [r["overlaps_next_rank"] for r in rows] == ["false", ""]
    assert "war_normal_approx_lo" in payload["generals"][0]
    assert "war_interval_lo" not in payload["generals"][0]


def test_csv_text_cells_that_look_like_formulas_are_quoted_but_numbers_are_not(
    tmp_path: Path,
) -> None:
    v = view(
        [
            estimate(1, "=HYPERLINK(1)", rank=1, total_war=-2.5),
            estimate(2, "-Rex", rank=2),
            estimate(3, "@home", rank=3),
            estimate(4, "Plus+", rank=4),
        ]
    )

    write_rankings_csv(tmp_path / "r.csv", v)
    rows = read_csv_rows(tmp_path / "r.csv")

    assert [r["name"] for r in rows] == ["'=HYPERLINK(1)", "'-Rex", "'@home", "Plus+"]
    assert rows[0]["total_war"] == "-2.5"


def test_the_report_marks_overlap_as_n_a_for_the_last_row_and_never_prints_none() -> None:
    v = view([estimate(1, rank=1, total_war=1.0), estimate(2, rank=None, total_war=5.0)])

    report = render_markdown(v, [])
    unranked = [line for line in report.splitlines() if line.startswith("| n/a | General 2")]

    assert len(unranked) == 1 and unranked[0].rstrip().endswith("| n/a |")
    assert "None" not in report


def test_rank_overlap_text_does_not_present_the_interval_as_credible() -> None:
    v = _view()
    report = render_markdown(v, [])

    assert "normal approximation" in report
    assert "is on skill, and is shown beside it" not in report
    assert "derived here from the stored sd" in report
    assert "next listed general" in report


def test_the_explorer_axis_and_labels_name_the_approximation() -> None:
    from pipeline.reporters.explorer_script import EXPLORER_JS

    assert "Expected total WAR (bars: normal approximation)" in EXPLORER_JS
    assert "war_interval" not in EXPLORER_JS


def test_sensitivity_ranks_are_restricted_to_generals_in_the_payload() -> None:
    payload = build_payload(_view())

    assert payload["sensitivity"]["ranks"]["observed_only"] == {"1": 2, "2": 1}


def test_the_payload_says_evaluation_is_unavailable_when_it_is() -> None:
    v = view([estimate(1, rank=1)], evaluation_reason="no evaluation report for model run 7")
    payload = build_payload(v)

    assert payload["sensitivity"] is None
    assert payload["evaluation_unavailable"] == "no evaluation report for model run 7"


# ─── manifest and the spec's gate reader ─────────────────────────────────────


def _manifest(tmp_path: Path, *, csv_rows: int, db_rows: int, reason: str = "") -> Path:
    v = _view(evaluation_reason=reason)
    gates = [
        export_matches_db_gate(csv_headline_rows=csv_rows, db_headline_rows=db_rows),
        evaluation_current_gate(evaluation_reason=reason, evaluated_run_id=7),
    ]
    manifest = build_manifest(v, csv_rows=3, json_generals=3, gates=gates, files={})
    return write_manifest(tmp_path / "manifest.json", manifest)


def _read_gate(path: Path, gate: str):  # type: ignore[no-untyped-def]
    check = {"params": {"report": str(path), "gate": gate}, "threshold": ""}
    return _check_evaluation_gate(gate, check, Severity.ERROR, None)


def test_the_manifest_records_the_run_counts_and_provisional_flag(tmp_path: Path) -> None:
    path = _manifest(tmp_path, csv_rows=2, db_rows=2, reason="no report")
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["model_run_id"] == 7
    assert data["provisional"] is True
    assert data["reasons"] and data["counts"]["headline"] == 2
    assert data["counts"]["under_determined"] == 1


def test_matching_counts_pass_through_the_quality_gate_reader(tmp_path: Path) -> None:
    path = _manifest(tmp_path, csv_rows=2, db_rows=2)

    export = _read_gate(path, "export_matches_db")
    current = _read_gate(path, "evaluation_current")

    assert export.passed, export.message
    assert current.passed, current.message
    assert export.severity is Severity.ERROR


def test_a_count_mismatch_fails_the_export_gate_with_a_message(tmp_path: Path) -> None:
    result = _read_gate(_manifest(tmp_path, csv_rows=2, db_rows=5), "export_matches_db")

    assert not result.passed
    assert "5" in result.message and "2" in result.message


def test_an_empty_export_fails_even_when_the_counts_agree(tmp_path: Path) -> None:
    assert not _read_gate(_manifest(tmp_path, csv_rows=0, db_rows=0), "export_matches_db").passed


def test_a_stale_evaluation_fails_the_evaluation_current_gate(tmp_path: Path) -> None:
    path = _manifest(tmp_path, csv_rows=2, db_rows=2, reason="stale evaluation report")

    result = _read_gate(path, "evaluation_current")

    assert not result.passed
    assert "stale evaluation report" in result.message
