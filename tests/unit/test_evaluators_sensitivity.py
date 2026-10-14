"""Evaluate stage sensitivity refits, the noise floor and the observed_only companion."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from pipeline.evaluators.sensitivity import (
    run_sensitivity,
)
from pipeline.evaluators.variants import parse_variants
from tests.unit.evaluators_helpers import (
    N_ROUNDS,
    PRIORS,
    FakeFitter,
    _design,
    _inputs,
    _measured,
    _patch_sensitivity,
)


def test_sensitivity_reports_every_variant_status_and_shares_one_seed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    design = _design()
    fitter = FakeFitter(fail_label=f"sens_{PRIORS}", unconverged=("sens_three_level_outcome",))
    _patch_sensitivity(monkeypatch, fitter)
    inputs = _inputs(tmp_path, design)

    result = run_sensitivity(inputs, parse_variants(inputs.params.sensitivity_variants))

    status = {v.spec.name: v.status for v in result.variants}
    assert status == {
        "drop_low_quality": "ok",
        "observed_only": "ok",
        PRIORS: "error",
        "equal_attribution": "no_op",
        "field_commanders_only": "no_op",
        "exclude_extreme_ratio": "no_op",
        "three_level_outcome": "not_converged",
        "replacement_50th": "ok",
        "realised_war": "ok",
    }
    flat = next(v for v in result.variants if v.spec.name == PRIORS)
    assert flat.message == "RuntimeError: sampler exploded"
    assert {c["seed"] for c in fitter.calls if c["label"] != "sens_noise_floor"} == {inputs.seed}
    floor = next(c for c in fitter.calls if c["label"] == "sens_noise_floor")
    assert floor["seed"] != inputs.seed
    assert result.noise_floor is not None and result.noise_floor["status"] == "ok"
    assert fitter.calls[0]["label"] == "sens_baseline"
    assert all(len(c["x_diffs"]) <= 2 for c in fitter.calls)
    dropped = next(v for v in result.variants if v.spec.name == "drop_low_quality")
    assert dropped.counts["battles_dropped"] == 1
    assert dropped.vs_baseline is not None and dropped.vs_published is not None
    assert set(dropped.vs_baseline) == {"score", "skill_mean"}
    assert result.baseline_vs_published is not None


def test_observed_only_refits_on_one_fixed_x_diff_from_measured_battles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    design = _design()
    fitter = FakeFitter()
    _patch_sensitivity(monkeypatch, fitter)
    inputs = _inputs(tmp_path, design)
    spec = [
        s for s in parse_variants(inputs.params.sensitivity_variants) if s.name == "observed_only"
    ]

    run_sensitivity(inputs, spec)

    call = next(c for c in fitter.calls if c["label"] == "sens_observed_only")
    assert len(call["x_diffs"]) == 1
    np.testing.assert_allclose(call["x_diffs"][0], 0.0)  # measured means are equal


def test_observed_only_is_also_compared_with_its_imputed_subset_companion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    design = _design()
    fitter = FakeFitter()
    _patch_sensitivity(monkeypatch, fitter)
    inputs = _inputs(tmp_path, design)
    specs = [
        s for s in parse_variants(inputs.params.sensitivity_variants) if s.name == "observed_only"
    ]

    result = run_sensitivity(inputs, specs)

    observed_call = next(c for c in fitter.calls if c["label"] == "sens_observed_only")
    companion_call = next(
        c for c in fitter.calls if c["label"] == "sens_observed_only_imputed_subset"
    )
    assert len(observed_call["x_diffs"]) == 1
    assert len(companion_call["x_diffs"]) == inputs.params.sensitivity_rounds
    np.testing.assert_array_equal(
        companion_call["design"].battle_ids, observed_call["design"].battle_ids
    )
    assert result.imputed_subset is not None and result.imputed_subset.status == "ok"
    observed = result.variants[0]
    assert observed.vs_imputed_subset is not None
    assert set(observed.vs_imputed_subset) == {"score", "skill_mean"}
    summary = result.summary()
    assert summary["observed_only_imputed_subset"]["status"] == "ok"
    assert summary["variants"][0]["vs_imputed_subset"] is not None


def test_no_companion_without_observed_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fitter = FakeFitter()
    _patch_sensitivity(monkeypatch, fitter)
    inputs = _inputs(tmp_path, _design())
    specs = [s for s in parse_variants(inputs.params.sensitivity_variants) if s.name == PRIORS]

    result = run_sensitivity(inputs, specs)

    assert result.imputed_subset is None
    assert result.summary()["observed_only_imputed_subset"] is None


def test_noise_floor_uses_disjoint_rounds_when_enough_exist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    design = _design()
    fitter = FakeFitter()
    _patch_sensitivity(monkeypatch, fitter)
    inputs = dataclasses.replace(
        _inputs(tmp_path, design),
        x_diffs=[np.full(design.n_battles, float(i)) for i in range(4)],
    )

    result = run_sensitivity(inputs, [])

    base = next(c for c in fitter.calls if c["label"] == "sens_baseline")
    floor = next(c for c in fitter.calls if c["label"] == "sens_noise_floor")
    assert [x[0] for x in base["x_diffs"]] == [0.0, 1.0]
    assert [x[0] for x in floor["x_diffs"]] == [2.0, 3.0]
    assert result.noise_floor is not None
    assert result.noise_floor["disjoint_rounds"] is True
    assert set(result.noise_floor) >= {"score", "skill_mean", "rounds_used"}


def test_noise_floor_reuses_the_rounds_when_there_are_too_few(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fitter = FakeFitter()
    _patch_sensitivity(monkeypatch, fitter)

    result = run_sensitivity(_inputs(tmp_path, _design()), [])  # 3 rounds, 2 per fit

    assert result.noise_floor is not None and result.noise_floor["disjoint_rounds"] is False


def test_noise_floor_failure_is_recorded_not_raised(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_sensitivity(monkeypatch, FakeFitter(fail_label="sens_noise_floor"))

    result = run_sensitivity(_inputs(tmp_path, _design()), [])

    assert result.noise_floor is not None and result.noise_floor["status"] == "error"
    assert result.baseline_table is not None


def test_exclude_extreme_ratio_filters_on_the_measured_difference_not_the_imputed_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    design = _design()
    _patch_sensitivity(monkeypatch, FakeFitter())
    mean, sd = _measured(design)
    mean = mean.copy()
    mean[design.side_a[0]] = 12.0  # |diff| = 3 > 2, both sides measured
    mean[design.side_a[1]] = np.nan  # unmeasured side: kept even though x_diff is huge
    sd = sd.copy()
    sd[design.side_a[1]] = np.nan
    inputs = dataclasses.replace(
        _inputs(tmp_path, design),
        measured=(mean, sd),
        x_diffs=[np.full(design.n_battles, 99.0)] * N_ROUNDS,  # imputed values must not be read
    )
    spec = [
        s
        for s in parse_variants(inputs.params.sensitivity_variants)
        if s.name == "exclude_extreme_ratio"
    ]

    result = run_sensitivity(inputs, spec)

    variant = result.variants[0]
    assert variant.status == "ok"
    assert variant.counts["battles_dropped"] == 1
    assert variant.counts["n_battles"] == design.n_battles - 1


def test_post_hoc_realised_variant_ranks_by_realised_mean(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    design = _design()
    _patch_sensitivity(monkeypatch, FakeFitter())
    inputs = _inputs(tmp_path, design)
    spec = [
        s for s in parse_variants(inputs.params.sensitivity_variants) if s.name == "realised_war"
    ]

    result = run_sensitivity(inputs, spec)

    table = result.variants[0].table
    assert table is not None
    np.testing.assert_array_equal(table.score, design.general_ids.astype(float))
    assert table.rank[-1] == 1  # the largest id has the largest realised mean


def test_unconverged_baseline_marks_every_variant_without_refitting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fitter = FakeFitter(unconverged=("sens_baseline",))
    _patch_sensitivity(monkeypatch, fitter)
    inputs = _inputs(tmp_path, _design())

    result = run_sensitivity(inputs, parse_variants(inputs.params.sensitivity_variants))

    assert [c["label"] for c in fitter.calls] == ["sens_baseline"]
    assert {v.status for v in result.variants} == {"not_converged"}
    assert result.baseline_table is None
