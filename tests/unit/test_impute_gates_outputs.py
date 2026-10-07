"""Unit tests for the impute gates, summaries, files and the quality handler."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from pipeline.imputers.conditional import RoundSet
from pipeline.imputers.design import build_design
from pipeline.imputers.gates import (
    distributional_check,
    imputation_completeness,
    imputed_ratio_plausible,
    imputed_values_plausible,
    run_gates,
)
from pipeline.imputers.outputs import config_hash, write_gate_report, write_rounds
from pipeline.imputers.priors import load_priors
from pipeline.imputers.records import BattleLikelihood, SideRow
from pipeline.imputers.summarise import is_converged, summarise_imputed
from pipeline.quality import QualityRunner, Severity


def test_completeness_requires_the_right_shape_and_finite_values() -> None:
    good = np.zeros((3, 4))
    holes = good.copy()
    holes[1, 2] = np.nan

    assert imputation_completeness(good, n_sides=4, n_rounds=3).passed
    assert not imputation_completeness(holes, n_sides=4, n_rounds=3).passed
    assert not imputation_completeness(good, n_sides=5, n_rounds=3).passed
    assert not imputation_completeness(good, n_sides=4, n_rounds=4).passed


def test_ratio_gate_fails_an_imputed_battle_whose_median_ratio_is_implausible() -> None:
    battle_ids = np.array([1, 1, 2, 2])
    is_imputed = np.array([False, True, False, False])
    ok = np.array([[9.0, 10.0, 8.0, 8.5]] * 3)
    bad = ok.copy()
    bad[:, 1] = 9.0 + 4.5  # battle 1 imputed, median |log ratio| 4.5 in every round

    assert imputed_ratio_plausible(ok, battle_ids, is_imputed, limit=4.0).passed
    result = imputed_ratio_plausible(bad, battle_ids, is_imputed, limit=4.0)
    assert not result.passed
    assert result.severity == "error"
    assert "[1]" in result.message


def test_ratio_gate_ignores_a_single_tail_round_and_fully_observed_battles() -> None:
    battle_ids = np.array([1, 1, 2, 2])
    is_imputed = np.array([False, True, False, False])
    values = np.array([[9.0, 10.0, 4.0, 8.6]] * 5)  # battle 2 observed at log ratio 4.6
    values[2, 1] = 9.0 + 6.0  # one extreme round for the imputed battle

    assert imputed_ratio_plausible(values, battle_ids, is_imputed, limit=4.0).passed


def test_ratio_gate_ignores_single_sided_battles() -> None:
    result = imputed_ratio_plausible(
        np.array([[1.0, 50.0]]), np.array([1, 2]), np.array([True, True]), limit=4.0
    )

    assert result.passed


def test_value_range_gate_is_published_scale_only() -> None:
    is_imputed = np.array([True, False])
    values = np.array([[np.log(10.0), np.log(1e9)]])  # imputed 10 is out of range

    published = imputed_values_plausible(values, is_imputed, scale="published", lower=50, upper=1e7)
    neutral = imputed_values_plausible(values, is_imputed, scale="neutral", lower=50, upper=1e7)

    assert not published.passed
    assert neutral.passed and "not applicable" in neutral.actual
    ok = imputed_values_plausible(
        np.array([[np.log(5000.0), 0.0]]), is_imputed, scale="published", lower=50, upper=1e7
    )
    assert ok.passed


def test_distributional_check_is_a_warning_that_fires_on_a_shifted_era() -> None:
    rng = np.random.default_rng(0)
    era = np.array(["ancient"] * 120 + ["modern"] * 120)
    imputed = np.tile([True, False], 120)
    base = rng.normal(9.0, 1.0, size=(5, 240))
    base[:, (era == "modern") & imputed] += 2.5  # imputed modern sides far too big

    result = distributional_check(base, imputed, era, alpha=0.01, min_n=5)

    assert result.severity == "warning"
    assert not result.passed
    assert "modern" in result.actual and "ancient" not in result.actual.split("differing=")[1]
    assert "MAR" in result.message

    same = rng.normal(9.0, 1.0, size=(5, 240))
    assert distributional_check(same, imputed, era, alpha=0.01, min_n=5).passed


def test_distributional_check_skips_thin_eras() -> None:
    result = distributional_check(
        np.zeros((2, 4)),
        np.array([True, False, False, False]),
        np.array(["a"] * 4),
        alpha=0.01,
        min_n=5,
    )

    assert result.passed and "tested=0" in result.actual


def test_run_gates_returns_the_four_dataset_gates_in_spec_order() -> None:
    values = np.random.default_rng(1).normal(9.0, 0.3, size=(3, 4))

    gates = run_gates(
        values,
        is_imputed=np.array([False, True, False, True]),
        battle_ids=np.array([1, 1, 2, 2]),
        era_of_side=np.array(["modern"] * 4),
        scale="neutral",
        n_rounds=3,
        ratio_limit=4.0,
        value_min=50,
        value_max=1e7,
        ks_alpha=0.01,
        ks_min_n=5,
    )

    assert [g.name for g in gates] == [
        "imputation_completeness",
        "imputed_ratio_plausible",
        "imputed_values_plausible",
        "distributional_check",
    ]
    assert all(g.passed for g in gates)


@pytest.mark.parametrize(
    ("worst", "expected"),
    [
        ({"max_rhat": 1.005, "min_ess_bulk": 900.0, "divergences": 0}, True),
        ({"max_rhat": 1.02, "min_ess_bulk": 900.0, "divergences": 0}, False),
        ({"max_rhat": 1.005, "min_ess_bulk": 399.0, "divergences": 0}, False),
        ({"max_rhat": 1.005, "min_ess_bulk": 900.0, "divergences": 1}, False),
        ({"max_rhat": None, "min_ess_bulk": 900.0, "divergences": 0}, False),
    ],
)
def test_convergence_write_rule(worst: dict[str, float | int | None], expected: bool) -> None:
    assert is_converged(worst, max_rhat=1.01, min_ess_bulk=400, max_divergences=0) is expected


def _tiny_design() -> tuple[object, RoundSet]:
    sides = [
        SideRow(10, 1, 1805, None, None, None, "unknown", "unknown"),
        SideRow(11, 1, 1805, None, None, None, "unknown", "unknown", mnar=True),
    ]
    lik = BattleLikelihood(1, (10,), np.array([9.0]), np.array([[4.0]]), "joint")
    design = build_design(sides, [lik], load_priors("config/imputation_priors.yaml"))
    values = np.array([[9.0, 10.0], [9.1, 11.0], [8.9, 12.0]])
    rounds = RoundSet(values=values, is_imputed=design.is_observed == False, draw_index=(1, 2, 3))  # noqa: E712
    return design, rounds


def test_summarise_imputed_reports_median_and_interval_on_the_exp_scale() -> None:
    design, rounds = _tiny_design()

    summaries = summarise_imputed(rounds, design)  # type: ignore[arg-type]

    assert [s.side_id for s in summaries] == [11]
    assert summaries[0].value == pytest.approx(np.exp(11.0))
    assert summaries[0].lo < summaries[0].value < summaries[0].hi
    assert summaries[0].lo > np.exp(10.0)


def test_write_rounds_writes_csv_and_manifest_and_clears_stale_rounds(tmp_path: Path) -> None:
    design, rounds = _tiny_design()
    stale = tmp_path / "neutral" / "round_9"
    stale.mkdir(parents=True)
    (stale / "sides.csv").write_text("old", encoding="utf-8")
    keep = tmp_path / "neutral" / "notes"
    keep.mkdir()

    dirs = write_rounds(
        tmp_path,
        scale="neutral",
        rounds=rounds,
        design=design,  # type: ignore[arg-type]
        run_id=42,
        seed=7,
        reconcile_run_id=5,
        config_digest="abc",
    )

    assert [d.name for d in dirs] == ["round_0", "round_1", "round_2"]
    assert not stale.exists()
    assert keep.exists()
    with (dirs[1] / "sides.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0] == {
        "side_id": "10",
        "battle_id": "1",
        "log_troops": "9.1",
        "is_imputed": "False",
        "scale": "neutral",
    }
    assert rows[1]["is_imputed"] == "True"
    manifest = json.loads((dirs[1] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == 42
    assert manifest["scale"] == "neutral"
    assert manifest["seed"] == 7
    assert manifest["draw_index"] == 2
    assert manifest["reconcile_run_id"] == 5
    assert manifest["config_hash"] == "abc"


def test_config_hash_is_stable_and_sensitive() -> None:
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def _check(gate: str, tmp_path: Path, severity: str = "error") -> dict[str, object]:
    return {
        "name": gate,
        "method": "imputation_gate",
        "params": {"gate": gate, "root": str(tmp_path)},
        "severity": severity,
    }


def test_quality_handler_reports_recorded_gates(tmp_path: Path) -> None:
    values = np.zeros((2, 2))
    gates = run_gates(
        values,
        is_imputed=np.array([True, False]),
        battle_ids=np.array([1, 1]),
        era_of_side=np.array(["modern", "modern"]),
        scale="neutral",
        n_rounds=2,
        ratio_limit=4.0,
        value_min=50,
        value_max=1e7,
        ks_alpha=0.01,
        ks_min_n=5,
    )
    write_gate_report(tmp_path, scale="neutral", run_id=1, written=True, gates=gates)
    runner = QualityRunner(None)

    passed = runner.run_one(_check("imputation_completeness", tmp_path))
    unknown = runner.run_one(_check("no_such_gate", tmp_path))

    assert passed.passed and passed.severity is Severity.ERROR
    assert not unknown.passed and "absent" in unknown.message
    assert (tmp_path / "neutral" / "gate_report.json").exists()
    assert not (tmp_path / "gate_report.json").exists()  # no scale-less copy


def test_quality_handler_surfaces_a_failed_gate_and_a_missing_report(tmp_path: Path) -> None:
    runner = QualityRunner(None)
    missing = runner.run_one(_check("imputation_completeness", tmp_path))
    assert not missing.passed and "Run the impute stage" in missing.message

    bad = imputation_completeness(np.full((1, 1), np.nan), n_sides=1, n_rounds=1)
    write_gate_report(tmp_path, scale="neutral", run_id=1, written=True, gates=[bad])
    failed = runner.run_one(_check("imputation_completeness", tmp_path))
    assert not failed.passed and "not finite" in failed.message


def test_unconverged_report_has_no_gates_and_fails_every_check(tmp_path: Path) -> None:
    write_gate_report(tmp_path, scale="neutral", run_id=1, written=False, gates=[])

    result = QualityRunner(None).run_one(_check("imputed_ratio_plausible", tmp_path))

    assert not result.passed and "did not converge" in result.message


def test_a_published_report_does_not_overwrite_the_neutral_one(tmp_path: Path) -> None:
    ok = imputation_completeness(np.zeros((1, 1)), n_sides=1, n_rounds=1)
    bad = imputation_completeness(np.full((1, 1), np.nan), n_sides=1, n_rounds=1)
    write_gate_report(tmp_path, scale="neutral", run_id=1, written=True, gates=[ok])
    write_gate_report(tmp_path, scale="published", run_id=2, written=True, gates=[bad])
    runner = QualityRunner(None)

    default = runner.run_one(_check("imputation_completeness", tmp_path))
    published_check = _check("imputation_completeness", tmp_path)
    published_check["params"]["scale"] = "published"  # type: ignore[index]
    published = runner.run_one(published_check)

    assert default.passed
    assert not published.passed
