"""Disclosure merging, provisional reasons and the evaluation loader."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from pipeline.reporters.disclosures import (
    STATIC_DISCLOSURES,
    merge_disclosures,
    run_disclosures,
)
from pipeline.reporters.load import load_evaluate_severities, load_evaluation, load_model_gates
from pipeline.reporters.records import ModelGate
from pipeline.reporters.status import provisional_reasons
from tests.unit.reporters_helpers import estimate, evaluation, view

# ─── disclosures ─────────────────────────────────────────────────────────────


def test_merge_drops_repeats_ignoring_case_and_whitespace() -> None:
    merged = merge_disclosures(["A  thing.", "Other."], ["a thing.", " Other. ", "New."])

    assert merged == ("A  thing.", "Other.", "New.")


def test_merge_skips_blank_entries_and_keeps_first_order() -> None:
    assert merge_disclosures(["", "  ", "x"], ["y"]) == ("x", "y")


def test_the_view_merges_static_evaluate_and_run_disclosures() -> None:
    v = view(
        [estimate(1, rank=1), estimate(2, rank=2, n_battles=1)],
        evaluation_info=evaluation(Path(".")),
    )

    assert set(STATIC_DISCLOSURES) <= set(v.disclosures)
    assert "Listing order predicts the winner in this corpus." in v.disclosures
    assert any("credits 6 of 10 in-scope battles" in d for d in v.disclosures)
    assert len(v.disclosures) == len(set(v.disclosures))


def test_the_evaluate_technology_disclosure_is_not_repeated() -> None:
    v = view([estimate(1, rank=1)], evaluation_info=evaluation(Path(".")))

    assert sum("Technology is not modelled" in d for d in v.disclosures) == 1


def test_run_disclosures_quote_only_the_counts_present() -> None:
    assert len(run_disclosures({}, min_battles=3, n_under_determined=4)) == 1
    out = run_disclosures(
        {"sides_multi_commander_all": 254, "sides_default_split_all": 254},
        min_battles=3,
        n_under_determined=0,
    )
    assert any("254 of 254" in d for d in out)


def test_disclosures_use_no_long_dashes() -> None:
    v = view([estimate(1, rank=1)], evaluation_info=evaluation(Path(".")))

    assert not any("—" in d or "–" in d for d in v.disclosures)


# ─── provisional reasons ─────────────────────────────────────────────────────


def test_a_clean_run_with_a_current_evaluation_is_not_provisional() -> None:
    assert provisional_reasons([ModelGate("fmi", True, "warning", "")], "") == ()


def test_a_stale_evaluation_report_is_a_reason() -> None:
    reasons = provisional_reasons([], "stale evaluation report: evaluated run 3, reported run 7")

    assert reasons == ("stale evaluation report: evaluated run 3, reported run 7",)


def test_a_missing_model_gate_report_is_a_reason() -> None:
    reasons = provisional_reasons(None, "")

    assert len(reasons) == 1
    assert "model gate report" in reasons[0]


def test_a_failed_error_severity_model_gate_is_a_reason() -> None:
    reasons = provisional_reasons([ModelGate("convergence", False, "error", "rhat 1.2")], "")

    assert reasons == ("model gate convergence failed: rhat 1.2",)


def test_a_failed_warning_gate_is_not_a_reason_unless_it_is_blocking() -> None:
    gates = [
        ModelGate("circularity_check", False, "warning", "x"),
        ModelGate("exclusion_concentration", False, "warning", "most battles uncredited"),
    ]

    reasons = provisional_reasons(gates, "")

    assert reasons == ("model gate exclusion_concentration failed: most battles uncredited",)


def test_disabling_require_passing_diagnostics_ignores_model_gates_only() -> None:
    gates = [ModelGate("exclusion_concentration", False, "warning", "")]

    assert provisional_reasons(gates, "", require_passing_diagnostics=False) == ()
    assert provisional_reasons(gates, "no report", require_passing_diagnostics=False) == (
        "no report",
    )


def test_every_reason_reaches_the_view_and_flips_provisional() -> None:
    v = view(
        [estimate(1, rank=1)],
        evaluation_reason="no evaluation report for model run 7",
        gates=None,
    )

    assert v.provisional
    assert len(v.reasons) == 2


# ─── loading ─────────────────────────────────────────────────────────────────


def _write_evaluation(root: Path, run_dir: int, model_run_id: int) -> None:
    directory = root / f"run{run_dir}"
    (directory / "figures").mkdir(parents=True)
    (directory / "figures" / "calibration.png").write_bytes(b"png")
    report = {
        "model_run_id": model_run_id,
        "disclosures": ["d1"],
        "gates": [{"name": "held_out_accuracy", "passed": True}],
        "figures": ["data\\processed\\evaluation\\run7\\figures\\calibration.png", "..\\x.png"],
        "sensitivity": {"variants": [{"name": "v", "status": "ok", "vs_baseline": None}]},
    }
    (directory / "evaluation_report.json").write_text(json.dumps(report), encoding="utf-8")


def test_a_current_evaluation_report_is_read_with_its_figures_by_name(tmp_path: Path) -> None:
    _write_evaluation(tmp_path / "evaluation", 7, 7)

    info, reason = load_evaluation(tmp_path / "evaluation", 7)

    assert reason == ""
    assert info is not None
    assert [p.name for p in info.figure_files] == ["calibration.png"]


def test_a_report_for_another_run_is_stale_and_not_used(tmp_path: Path) -> None:
    _write_evaluation(tmp_path / "evaluation", 7, 3)

    info, reason = load_evaluation(tmp_path / "evaluation", 7)

    assert info is None
    assert "stale evaluation report" in reason


def test_a_missing_evaluation_report_gives_a_reason(tmp_path: Path) -> None:
    info, reason = load_evaluation(tmp_path / "evaluation", 7)

    assert info is None
    assert "no evaluation report" in reason


def test_model_gates_are_read_from_the_run_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "run7"
    run_dir.mkdir()
    (run_dir / "gate_report.json").write_text(
        json.dumps({"gates": [{"name": "g", "passed": False, "severity": "error"}]}),
        encoding="utf-8",
    )

    gates = load_model_gates(tmp_path, 7)

    assert gates is not None
    assert (gates[0].name, gates[0].passed, gates[0].severity) == ("g", False, "error")
    assert load_model_gates(tmp_path, 8) is None


# ─── evaluation gates, rank gaps, malformed input ───────────────────────────


def test_a_failed_error_gate_in_the_evaluation_report_is_a_reason() -> None:
    gates = (
        {"name": "named_checks", "passed": False, "severity": "error", "message": "not_evaluable"},
        {"name": "calibration", "passed": False, "severity": "warning", "message": "loose"},
        {"name": "held_out_accuracy", "passed": True, "severity": "error", "message": "ok"},
    )

    reasons = provisional_reasons([], "", evaluation_gates=gates)

    assert reasons == ("evaluation gate named_checks failed: not_evaluable",)


def test_a_gate_entry_severity_beats_the_spec_severity() -> None:
    gates = ({"name": "calibration", "passed": False, "severity": "error", "message": "loose"},)

    reasons = provisional_reasons(
        [], "", evaluation_gates=gates, evaluation_severities={"calibration": "warning"}
    )

    assert reasons == ("evaluation gate calibration failed: loose",)


def test_the_view_is_provisional_when_an_evaluation_error_gate_failed() -> None:
    info = replace(
        evaluation(Path(".")),
        gates=({"name": "named_checks", "passed": False, "severity": "error", "message": "m"},),
    )

    v = view([estimate(1, rank=1)], evaluation_info=info)

    assert v.provisional
    assert v.reasons == ("evaluation gate named_checks failed: m",)


def test_a_rank_held_by_a_thin_general_inside_the_headline_range_is_disclosed() -> None:
    ests = [
        estimate(1, rank=1),
        estimate(2, rank=2, n_battles=1),
        estimate(3, rank=3),
        estimate(4, rank=9, n_battles=1),
    ]

    v = view(ests)

    gap = [d for d in v.disclosures if d.startswith("Ranks are positions among all credited")]
    assert len(gap) == 1
    assert "1 general(s) with fewer than 3 credited battles" in gap[0]
    assert "—" not in gap[0]


def test_contiguous_ranks_add_no_rank_gap_disclosure() -> None:
    v = view([estimate(1, rank=1), estimate(2, rank=2), estimate(3, rank=3, n_battles=1)])

    assert not any(d.startswith("Ranks are positions") for d in v.disclosures)


def test_a_malformed_evaluation_report_degrades_to_unavailable_with_a_reason(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "evaluation" / "run7"
    directory.mkdir(parents=True)
    report = {"model_run_id": 7, "cv": "not a mapping", "sensitivity": 5}
    (directory / "evaluation_report.json").write_text(json.dumps(report), encoding="utf-8")

    info, reason = load_evaluation(tmp_path / "evaluation", 7)

    assert info is None
    assert "malformed" in reason and "run 7" in reason


def test_an_unreadable_sensitivity_csv_degrades_to_unavailable(tmp_path: Path) -> None:
    _write_evaluation(tmp_path / "evaluation", 7, 7)
    sensitivity = tmp_path / "evaluation" / "run7" / "sensitivity"
    sensitivity.mkdir()
    (sensitivity / "observed_only.csv").write_bytes(b"general_id,rank\n\xff\xfe,\x80\n")

    info, reason = load_evaluation(tmp_path / "evaluation", 7)

    assert info is None
    assert "malformed" in reason


# ─── evaluate gate severity from the spec ────────────────────────────────────

_EVALUATE_SPEC = """
quality_checks:
  - name: held_out_accuracy
    method: evaluation_gate
    params: {gate: held_out_accuracy}
    severity: warning
  - name: named_checks
    method: evaluation_gate
    params: {gate: named_checks}
    severity: error
  - name: unrelated
    method: sql
    severity: error
"""


def _spec(tmp_path: Path) -> Path:
    path = tmp_path / "evaluate.yaml"
    path.write_text(_EVALUATE_SPEC, encoding="utf-8")
    return path


def test_a_failed_gate_with_spec_severity_warning_is_not_a_reason(tmp_path: Path) -> None:
    gates = ({"name": "held_out_accuracy", "passed": False, "message": "0.51"},)
    severities = load_evaluate_severities(_spec(tmp_path))

    reasons = provisional_reasons([], "", evaluation_gates=gates, evaluation_severities=severities)

    assert reasons == ()


def test_a_failed_gate_with_spec_severity_error_is_a_reason(tmp_path: Path) -> None:
    gates = ({"name": "named_checks", "passed": False, "message": "not_evaluable"},)
    severities = load_evaluate_severities(_spec(tmp_path))

    reasons = provisional_reasons([], "", evaluation_gates=gates, evaluation_severities=severities)

    assert reasons == ("evaluation gate named_checks failed: not_evaluable",)


def test_a_failed_gate_absent_from_the_spec_is_a_reason(tmp_path: Path) -> None:
    gates = ({"name": "new_gate", "passed": False, "message": "m"},)
    severities = load_evaluate_severities(_spec(tmp_path))

    reasons = provisional_reasons([], "", evaluation_gates=gates, evaluation_severities=severities)

    assert reasons == ("evaluation gate new_gate failed: m",)


def test_the_json_severity_overrides_the_spec(tmp_path: Path) -> None:
    gates = (
        {"name": "held_out_accuracy", "passed": False, "severity": "error", "message": "x"},
        {"name": "named_checks", "passed": False, "severity": "warning", "message": "y"},
    )
    severities = load_evaluate_severities(_spec(tmp_path))

    reasons = provisional_reasons([], "", evaluation_gates=gates, evaluation_severities=severities)

    assert reasons == ("evaluation gate held_out_accuracy failed: x",)


def test_an_unreadable_spec_makes_every_unknown_failed_gate_block(tmp_path: Path) -> None:
    gates = ({"name": "held_out_accuracy", "passed": False, "message": "m"},)
    severities = load_evaluate_severities(tmp_path / "missing.yaml")

    reasons = provisional_reasons([], "", evaluation_gates=gates, evaluation_severities=severities)

    assert severities == {}
    assert reasons == ("evaluation gate held_out_accuracy failed: m",)


def test_the_spec_loader_uses_the_check_name_when_no_gate_param(tmp_path: Path) -> None:
    path = tmp_path / "evaluate.yaml"
    path.write_text(
        "quality_checks:\n  - name: bare\n    method: evaluation_gate\n    severity: warning\n",
        encoding="utf-8",
    )

    assert load_evaluate_severities(path) == {"bare": "warning"}
