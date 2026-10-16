"""Evaluate stage, cross-validation, sensitivity and report with the sampler stubbed."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml

from pipeline.evaluators import cv as cv_module
from pipeline.evaluators.config import EvaluateParams, load_eval_model_params
from pipeline.evaluators.cv import run_cross_validation
from pipeline.evaluators.folds import train_weight
from pipeline.evaluators.report import build_gates, gate_severities
from pipeline.evaluators.sensitivity import (
    RankTable,
    SensitivityResult,
    VariantResult,
)
from pipeline.evaluators.target import PublishedRun
from pipeline.evaluators.variants import parse_variants
from pipeline.modelers.config import load_model_params
from pipeline.modelers.records import ModelDesign
from pipeline.stages import evaluate as stage
from pipeline.stages.base import StageContext
from tests.unit.evaluators_helpers import (
    N_ROUNDS,
    PRIORS,
    ROOT,
    FakeFitter,
    FakeJointFitter,
    _design,
    _measured,
    _mp,
    _params,
    _patch_sensitivity,
    _published,
    _rounds,
    _sides,
)

# ─── config ──────────────────────────────────────────────────────────────────


def test_params_read_the_spec_defaults() -> None:
    raw = yaml.safe_load((ROOT / "agents" / "evaluate.yaml").read_text(encoding="utf-8"))
    params = EvaluateParams.from_mapping(raw["params"])

    assert (params.held_out_folds, params.cv_rounds, params.eval_draws) == (5, 3, 1500)
    assert (params.top_n, params.min_top_overlap, params.seed) == (20, 12, 20261014)
    assert params.eval_min_ess_bulk == 400
    mp = load_eval_model_params(params)
    assert mp.min_ess_bulk == 400
    model_mp = load_model_params(
        yaml.safe_load((ROOT / "agents" / "model.yaml").read_text("utf-8"))["params"]
    )
    assert (mp.max_rhat, mp.max_divergences) == (model_mp.max_rhat, model_mp.max_divergences)
    assert model_mp.min_ess_bulk > mp.min_ess_bulk
    assert len(parse_variants(params.sensitivity_variants)) == 11


def test_eval_floor_never_raises_a_model_threshold_below_400(tmp_path: Path) -> None:
    spec = tmp_path / "model.yaml"
    spec.write_text(yaml.safe_dump({"params": {"write": {"min_ess_bulk": 0}}}), encoding="utf-8")
    params = _params(tmp_path, model_spec=str(spec))

    assert params.eval_min_ess_bulk == 400
    assert load_eval_model_params(params).min_ess_bulk == 0


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"held_out_folds": 1}, "held_out_folds"),
        ({"cv_rounds": 0}, "cv_rounds"),
        ({"min_top_overlap": 30}, "min_top_overlap"),
    ],
)
def test_params_validate(override: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        EvaluateParams.from_mapping(override)


def test_eval_model_params_lower_the_draws_and_keep_the_models_numbers(tmp_path: Path) -> None:
    params = _params(tmp_path)

    mp = load_eval_model_params(params)

    assert mp.sampler.draws == 20
    assert mp.sampler.tune > 0
    assert mp.scale == "neutral" and mp.max_rhat == 1.02


def test_eval_model_params_refuse_a_published_scale_model_spec(tmp_path: Path) -> None:
    spec = tmp_path / "model.yaml"
    spec.write_text(yaml.safe_dump({"params": {"scale": "published"}}), encoding="utf-8")

    with pytest.raises(ValueError, match="primary ranking only"):
        load_eval_model_params(_params(tmp_path, model_spec=str(spec)))


# ─── cross-validation ────────────────────────────────────────────────────────


def test_cv_holds_out_each_fold_and_predicts_without_the_imputed_x_diff(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    design = _design()
    fitter = FakeFitter()
    monkeypatch.setattr(cv_module, "fit_pooled", fitter)
    joint = FakeJointFitter(x_value=50.0)  # saturates P(A wins) if the draws' own x_diff is used
    monkeypatch.setattr(cv_module, "fit_joint", joint)
    seen: list[Any] = []
    real = cv_module.heldout_x_diff_draws

    def spy(*args: Any, **kwargs: Any) -> np.ndarray:
        seen.append(args[1])
        return real(*args, **kwargs)

    monkeypatch.setattr(cv_module, "heldout_x_diff_draws", spy)
    x_diffs = [np.full(design.n_battles, 99.0)] * N_ROUNDS  # absurd: must never reach a prediction

    result = run_cross_validation(design, x_diffs, _mp(), _params(tmp_path), _measured(design))

    assert [c["label"] for c in fitter.calls] == ["cv_fold_0", "cv_fold_1", "cv_fold_2"]
    for fold, call in enumerate(fitter.calls):
        np.testing.assert_array_equal(call["loglik_weight"], train_weight(result.fold_of, fold))
        assert len(call["x_diffs"]) == 2  # cv_rounds
    assert len({c["seed"] for c in fitter.calls}) == 3
    assert len(seen) == 3 * 2  # one measurement-prior draw per fold and round
    assert result.predicted.all()
    np.testing.assert_allclose(result.probs.sum(axis=1), 1.0)
    assert result.metrics["n_predicted"] == design.n_battles
    assert result.baselines["chance"]["brier"] == 0.25
    assert result.baselines["force_only"]["n"] == result.metrics["binary"]["n"]
    assert result.metrics["on_force_baseline_mask"]["n"] == result.baselines["force_only"]["n"]
    # Joint arm: one fit per fold, its own seeds and mask, predictions from x_diff_latent.
    assert [c["label"] for c in joint.calls] == [f"cv_joint_fold_{i}" for i in range(3)]
    for fold, call in enumerate(joint.calls):
        np.testing.assert_array_equal(call["loglik_weight"], train_weight(result.fold_of, fold))
    assert not {c["seed"] for c in joint.calls} & {c["seed"] for c in fitter.calls}
    assert len({c["seed"] for c in joint.calls}) == 3
    assert result.joint.predicted.all()
    assert (result.joint.probs[:, 4] > 0.99).all()
    assert result.joint.metrics["n_predicted"] == design.n_battles
    assert result.joint.baselines["force_only"]["n"] == result.baselines["force_only"]["n"]
    assert len(seen) == 3 * 2  # the joint arm never calls the measurement-prior generator


def test_cv_leaves_unconverged_and_failed_folds_unpredicted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    design = _design()
    monkeypatch.setattr(
        cv_module, "fit_pooled", FakeFitter(unconverged=("cv_fold_0",), fail_label="cv_fold_2")
    )
    monkeypatch.setattr(cv_module, "fit_joint", FakeJointFitter())
    x_diffs = [np.zeros(design.n_battles)] * N_ROUNDS

    result = run_cross_validation(design, x_diffs, _mp(), _params(tmp_path), _measured(design))

    assert [f.converged for f in result.folds] == [False, True, False]
    assert result.folds[2].error == "RuntimeError: sampler exploded"
    np.testing.assert_array_equal(result.predicted, result.fold_of == 1)
    assert result.metrics["n_predicted"] == int((result.fold_of == 1).sum())
    assert np.isnan(result.probs[~result.predicted]).all()
    assert result.joint.predicted.all()  # the arms fail independently


@pytest.mark.parametrize("mode", [{"unconverged": True}, {"fail": True}])
def test_cv_joint_arm_failure_is_recorded_and_leaves_the_imputed_arm_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: dict[str, bool]
) -> None:
    design = _design()
    monkeypatch.setattr(cv_module, "fit_pooled", FakeFitter())
    monkeypatch.setattr(cv_module, "fit_joint", FakeJointFitter(**mode))
    x_diffs = [np.zeros(design.n_battles)] * N_ROUNDS

    result = run_cross_validation(design, x_diffs, _mp(), _params(tmp_path), _measured(design))

    assert result.predicted.all() and all(f.converged for f in result.folds)
    assert not any(f.converged for f in result.joint.folds)
    assert not result.joint.predicted.any()
    assert result.joint.metrics["n_predicted"] == 0
    if "fail" in mode:
        assert result.joint.folds[0].error == "RuntimeError: joint exploded"


# ─── gates ───────────────────────────────────────────────────────────────────

NAMED_PASS = {"name": "named_checks", "passed": True, "actual": {}, "threshold": "", "message": ""}


def _cv(accuracy: float | None, brier: float | None, converged: tuple[bool, ...]) -> Any:
    n = 0 if accuracy is None else 40
    binary = {"n": n, "accuracy": accuracy, "brier": brier, "calibration": []}
    folds = [SimpleNamespace(fold=i, converged=c) for i, c in enumerate(converged)]
    return SimpleNamespace(metrics={"binary": binary}, folds=folds)


def _variant(name: str, status: str, overlap: int = 0, effective: int = 20) -> VariantResult:
    spec = parse_variants([{"name": "weak_hyperpriors", "kind": "priors"}])[0]
    spec = type(spec)(name, spec.kind, "", spec.options)
    cmp = {"score": {"top_overlap": overlap, "top_n_effective": effective}}
    return VariantResult(spec, status, None, {}, 0.0, None, cmp if status == "ok" else None, None)


def _sens(
    *variants: VariantResult, base: bool = True, noise_floor: dict[str, Any] | None = None
) -> SensitivityResult:
    table = None
    if base:
        ids = np.arange(5)
        table = RankTable.of(ids, ids.astype(float), ids.astype(float))
    return SensitivityResult(
        None,
        None if base else "baseline refit failed",
        table,
        None,
        list(variants),
        0.0,
        noise_floor,
    )


def _gates(cv: Any, sens: SensitivityResult, tmp_path: Path) -> dict[str, dict[str, Any]]:
    gates = build_gates(cv, sens, NAMED_PASS, _params(tmp_path, top_n=20, min_top_overlap=12))
    return {g["name"]: g for g in gates}


def test_gates_pass_on_good_numbers(tmp_path: Path) -> None:
    gates = _gates(
        _cv(0.62, 0.21, (True, True)),
        _sens(_variant("a", "ok", 15), _variant("b", "no_op")),
        tmp_path,
    )

    assert list(gates) == [
        "held_out_accuracy",
        "calibration",
        "cv_converged",
        "ranking_stability",
        "named_checks",
    ]
    assert all(g["passed"] for g in gates.values())
    assert all(
        set(g) == {"name", "passed", "actual", "threshold", "message"} for g in gates.values()
    )


def test_gates_carry_the_spec_severity(tmp_path: Path) -> None:
    spec = yaml.safe_load((ROOT / "agents" / "evaluate.yaml").read_text(encoding="utf-8"))
    severities = gate_severities(spec["quality_checks"])
    cv = _cv(0.62, 0.21, (True,))
    params = _params(tmp_path, top_n=20, min_top_overlap=12)

    gates = build_gates(cv, _sens(), NAMED_PASS, params, severities)

    assert severities["calibration"] == "warning"
    assert {g["name"]: g["severity"] for g in gates} == severities
    assert all("severity" not in g for g in build_gates(cv, _sens(), NAMED_PASS, params))


def test_gate_severities_ignore_other_methods_and_use_params_gate() -> None:
    checks = [
        {"name": "a", "method": "evaluation_gate", "params": {"gate": "g"}, "severity": "error"},
        {"name": "b", "method": "evaluation_gate", "severity": "warning"},
        {"name": "c", "method": "model_gate", "severity": "error"},
        {"name": "d", "method": "evaluation_gate"},
        "junk",
    ]

    assert gate_severities(checks) == {"g": "error", "b": "warning"}


def test_accuracy_and_brier_gates_use_strict_thresholds(tmp_path: Path) -> None:
    gates = _gates(_cv(0.55, 0.25, (True,)), _sens(), tmp_path)

    assert not gates["held_out_accuracy"]["passed"]
    assert not gates["calibration"]["passed"]


def test_gates_fail_without_binary_predictions(tmp_path: Path) -> None:
    gates = _gates(_cv(None, None, (False, False)), _sens(), tmp_path)

    assert not gates["held_out_accuracy"]["passed"] and not gates["calibration"]["passed"]
    assert "no held-out binary predictions" in gates["held_out_accuracy"]["message"]
    assert gates["cv_converged"]["actual"] == {"converged": 0, "folds": 2}


def test_stability_gate_names_failing_and_unconverged_variants(tmp_path: Path) -> None:
    sens = _sens(
        _variant("low_overlap", "ok", overlap=11),
        _variant("broken", "error"),
        _variant("stuck", "not_converged"),
        _variant("fine", "ok", overlap=12),
    )

    gate = _gates(_cv(0.6, 0.2, (True,)), sens, tmp_path)["ranking_stability"]

    assert not gate["passed"]
    for expected in ("low_overlap (overlap 11 < 12)", "broken (error)", "stuck (not_converged)"):
        assert expected in gate["message"]
    assert "fine" not in gate["message"]


def test_stability_gate_reports_not_evaluable_without_failing(tmp_path: Path) -> None:
    sens = _sens(_variant("a", "ok", 15), _variant("g_unlabelled_plus_1sd", "not_evaluable"))

    gate = _gates(_cv(0.6, 0.2, (True,)), sens, tmp_path)["ranking_stability"]

    assert gate["passed"]
    assert gate["actual"]["g_unlabelled_plus_1sd"] == "not_evaluable"
    assert "not evaluable: g_unlabelled_plus_1sd" in gate["message"]


def test_stability_gate_reports_the_noise_floor_without_changing_the_pass_rule(
    tmp_path: Path,
) -> None:
    floor = {"status": "ok", "score": {"top_overlap": 17, "top_n_effective": 20}}
    passing = _sens(_variant("a", "ok", 15), noise_floor=floor)
    failing = _sens(_variant("a", "ok", 11), noise_floor=floor)
    # A noise floor below the threshold does not fail the gate on its own.
    low_floor = _sens(
        _variant("a", "ok", 15), noise_floor={"status": "ok", "score": {"top_overlap": 2}}
    )

    gate = _gates(_cv(0.6, 0.2, (True,)), passing, tmp_path)["ranking_stability"]
    assert gate["passed"]
    assert "noise floor top-20 overlap 17" in gate["message"]
    assert gate["actual"]["noise_floor"] == {"top_overlap": 17}
    assert not _gates(_cv(0.6, 0.2, (True,)), failing, tmp_path)["ranking_stability"]["passed"]
    assert _gates(_cv(0.6, 0.2, (True,)), low_floor, tmp_path)["ranking_stability"]["passed"]


def test_stability_gate_names_a_failed_noise_floor(tmp_path: Path) -> None:
    sens = _sens(_variant("a", "ok", 15), noise_floor={"status": "error", "message": "x"})

    gate = _gates(_cv(0.6, 0.2, (True,)), sens, tmp_path)["ranking_stability"]

    assert gate["passed"] and "noise floor error" in gate["message"]


def test_stability_threshold_scales_with_a_short_list(tmp_path: Path) -> None:
    sens = _sens(_variant("short", "ok", overlap=6, effective=10))

    assert _gates(_cv(0.6, 0.2, (True,)), sens, tmp_path)["ranking_stability"]["passed"]
    sens = _sens(_variant("short", "ok", overlap=5, effective=10))
    assert not _gates(_cv(0.6, 0.2, (True,)), sens, tmp_path)["ranking_stability"]["passed"]


def test_stability_gate_fails_when_the_baseline_is_unusable(tmp_path: Path) -> None:
    gate = _gates(_cv(0.6, 0.2, (True,)), _sens(base=False), tmp_path)["ranking_stability"]

    assert not gate["passed"] and "baseline refit failed" in gate["message"]


# ─── the stage ───────────────────────────────────────────────────────────────


def _stage_spec(tmp_path: Path) -> dict[str, Any]:
    experts = tmp_path / "experts.yaml"
    experts.write_text(
        yaml.safe_dump(
            {
                "generals": {
                    "one": {"qid": "Q101"},
                    "two": {"qid": "Q102"},
                    "absent": {"qid": "Q999"},
                },
                "rank_checks": [{"general": "one", "rank_range": [1, 2]}],
                "pairwise_checks": [{"higher": "one", "lower": "two"}],
            }
        ),
        encoding="utf-8",
    )
    params = _params(tmp_path, expert_rankings=str(experts)).raw
    return {"params": dict(params)}


def _patch_stage(
    monkeypatch: pytest.MonkeyPatch, fitter: FakeFitter, *, published: PublishedRun | None = None
) -> ModelDesign:
    design = _design()
    monkeypatch.setattr(stage, "load_target_run", lambda conn: published or _published(design))
    monkeypatch.setattr(stage, "load_side_inputs", lambda conn: _sides())
    monkeypatch.setattr(stage, "read_rounds", lambda *a, **k: _rounds(design))
    monkeypatch.setattr(stage, "measured_log_troops", lambda conn, d: _measured(d))
    monkeypatch.setattr(stage, "load_battle_types", lambda conn: {1: "naval"})
    monkeypatch.setattr(stage, "load_battle_quality", lambda conn: {int(design.battle_ids[0]): 0.1})
    monkeypatch.setattr(cv_module, "fit_pooled", fitter)
    monkeypatch.setattr(cv_module, "fit_joint", FakeJointFitter())
    _patch_sensitivity(monkeypatch, fitter)
    return design


def _strict_json(path: Path) -> dict[str, Any]:
    def reject(token: str) -> None:
        raise AssertionError(f"non-strict JSON constant {token}")

    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
    return loaded


def test_stage_writes_the_report_files_and_figures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_stage(monkeypatch, FakeFitter())
    spec = _stage_spec(tmp_path)

    stage.run(spec, StageContext(db_conn=object()))

    run_dir = tmp_path / "evaluation" / "run7"
    report = _strict_json(run_dir / "evaluation_report.json")
    assert set(report) == {
        "schema_version",
        "model_run_id",
        "imputation_run_id",
        "created_at",
        "counts",
        "settings",
        "disclosures",
        "cv",
        "baselines",
        "sensitivity",
        "named_checks",
        "references",
        "ppc",
        "figures",
        "gates",
        "timings_seconds",
    }
    assert (report["model_run_id"], report["imputation_run_id"]) == (7, 5)
    assert [g["name"] for g in report["gates"]] == [
        "held_out_accuracy",
        "calibration",
        "cv_converged",
        "ranking_stability",
        "named_checks",
    ]
    assert report["counts"]["n_battles"] == 20
    assert any("Residual leakage" in d for d in report["disclosures"])
    assert any("Listing order predicts the winner" in d for d in report["disclosures"])
    assert report["settings"]["evaluate"]["eval_min_ess_bulk"] == 400
    for fragment in ("equal_attribution also converts", "weak_hyperpriors keeps", "prior-only"):
        assert any(fragment in d for d in report["disclosures"])
    assert set(report["cv"]) == {"imputed", "joint"}
    for arm in report["cv"].values():
        assert {"binary", "on_force_baseline_mask", "folds", "baselines"} <= set(arm)
    assert report["sensitivity"]["noise_floor"]["status"] == "ok"
    assert report["sensitivity"]["observed_only_imputed_subset"]["status"] == "ok"
    stability = next(g for g in report["gates"] if g["name"] == "ranking_stability")
    assert "noise floor" in stability["message"]
    assert set(report["timings_seconds"]) >= {"cv", "sensitivity", "ppc", "figures"}
    assert report["ppc"]["per_general"], "top generals should be checked"
    assert (tmp_path / "evaluation_report.json").read_bytes() == (
        run_dir / "evaluation_report.json"
    ).read_bytes()
    header = (run_dir / "cv_predictions.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header == (
        "battle_id,fold,p_level0,p_level1,p_level2,p_level3,p_level4,k_lo,k_hi,p_a_win_binary"
    )
    sens_csv = run_dir / "sensitivity" / f"{PRIORS}.csv"
    assert sens_csv.read_text(encoding="utf-8").splitlines()[0] == (
        "general_id,name,wikidata_id,score,rank,skill_mean"
    )
    assert (run_dir / "sensitivity" / "baseline.csv").exists()
    assert not (run_dir / "sensitivity" / "equal_attribution.csv").exists()  # no_op
    for figure in ("calibration", "rank_stability", "ci_width", "skill_vs_battles"):
        assert (run_dir / "figures" / f"{figure}.png").stat().st_size > 0


def test_stage_survives_a_variant_error_and_records_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_stage(monkeypatch, FakeFitter(fail_label=f"sens_{PRIORS}"))

    stage.run(_stage_spec(tmp_path), StageContext(db_conn=object()))

    report = _strict_json(tmp_path / "evaluation_report.json")
    variants = {v["name"]: v for v in report["sensitivity"]["variants"]}
    assert variants[PRIORS]["status"] == "error"
    stability = next(g for g in report["gates"] if g["name"] == "ranking_stability")
    assert not stability["passed"] and f"{PRIORS} (error)" in stability["message"]


def test_stage_without_a_target_run_logs_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fitter = FakeFitter()
    _patch_stage(monkeypatch, fitter)
    monkeypatch.setattr(stage, "load_target_run", lambda conn: None)

    stage.run(_stage_spec(tmp_path), StageContext(db_conn=object()))

    assert not (tmp_path / "evaluation").exists()
    assert not (tmp_path / "evaluation_report.json").exists()
    assert fitter.calls == []


def test_stage_refuses_when_the_database_moved_on(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fitter = FakeFitter()
    design = _patch_stage(monkeypatch, fitter)
    stale = _published(design)
    stale = PublishedRun(**{**stale.__dict__, "general_ids": stale.general_ids + 1000})
    monkeypatch.setattr(stage, "load_target_run", lambda conn: stale)

    stage.run(_stage_spec(tmp_path), StageContext(db_conn=object()))

    assert fitter.calls == [] and not (tmp_path / "evaluation_report.json").exists()


def test_stage_dry_run_fits_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fitter = FakeFitter()
    _patch_stage(monkeypatch, fitter)

    stage.run(_stage_spec(tmp_path), StageContext(db_conn=object(), dry_run=True))

    assert fitter.calls == [] and not (tmp_path / "evaluation").exists()


def test_stage_reports_a_broken_expert_file_as_a_failed_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_stage(monkeypatch, FakeFitter())
    spec = _stage_spec(tmp_path)
    spec["params"]["expert_rankings"] = str(tmp_path / "missing.yaml")

    stage.run(spec, StageContext(db_conn=object()))

    report = _strict_json(tmp_path / "evaluation_report.json")
    gate = next(g for g in report["gates"] if g["name"] == "named_checks")
    assert not gate["passed"] and "cannot load" in gate["message"]


def test_stage_names_checks_by_wikidata_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_stage(monkeypatch, FakeFitter())

    stage.run(_stage_spec(tmp_path), StageContext(db_conn=object()))

    report = _strict_json(tmp_path / "evaluation_report.json")
    by_name = {c["name"]: c["status"] for c in report["named_checks"]}
    assert by_name == {"one_rank_1_2": "pass", "one_above_two": "pass"}
    assert next(g for g in report["gates"] if g["name"] == "named_checks")["passed"]


def test_stage_conforms_to_the_stage_contract() -> None:
    from pipeline.stages.base import check_conforms

    check_conforms(stage)
    assert not getattr(stage, "OWNS_DB_CONNECTION", False)
