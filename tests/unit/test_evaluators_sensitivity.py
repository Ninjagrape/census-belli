"""Evaluate stage sensitivity refits, the noise floor and the observed_only companion."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from pipeline.evaluators.sensitivity import (
    run_sensitivity,
)
from pipeline.evaluators.target import round_x_diffs
from pipeline.evaluators.variants import (
    NotEvaluableError,
    fill_weak_slopes,
    g_unlabelled_slopes,
    parse_variants,
    read_imputed_side_ids,
    shift_rounds,
)
from pipeline.imputers.likelihood import NeutralDraws
from pipeline.modelers.load import ImputedRound
from pipeline.modelers.records import SideInput
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
        "g_unlabelled_plus_1sd": "not_evaluable",
        "g_unlabelled_minus_1sd": "not_evaluable",
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


def _neutral(
    design, *, with_g: bool = True, g_scale: float = 0.5, slope: float = 2.0
) -> NeutralDraws:
    """Draws whose side-s column is exactly intercept + slope_s * g + noise-free a_era term."""
    rng = np.random.default_rng(3)
    n = 400
    g = rng.normal(0.0, g_scale, n)
    a = rng.normal(0.0, 0.2, n)
    sides = design.side_ids.astype(np.int64)
    slopes = np.where(np.arange(sides.size) % 2 == 0, slope, 0.0)
    draws = 9.0 + g[:, None] * slopes[None, :] + 0.7 * a[:, None]
    return NeutralDraws(
        run_id=1,
        side_id=sides,
        battle_id=np.arange(sides.size, dtype=np.int64),
        draws=draws,
        m0_mean=9.0,
        s0_mean=1.0,
        g_unlabelled=g if with_g else None,
        a_era=a if with_g else None,
    )


def test_g_unlabelled_variants_shift_x_diff_by_slope_times_sd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    design = _design()
    neutral = _neutral(design)
    fitter = FakeFitter()
    _patch_sensitivity(monkeypatch, fitter)
    inputs = _inputs(tmp_path, design, load_neutral=lambda: neutral)
    specs = [s for s in parse_variants(inputs.params.sensitivity_variants) if "g_unl" in s.name]

    result = run_sensitivity(inputs, specs)

    sd = float(np.std(neutral.g_unlabelled, ddof=1))
    by_name = {v.spec.name: v for v in result.variants}
    assert {v.status for v in by_name.values()} == {"ok"}
    plus = next(c for c in fitter.calls if c["label"] == "sens_g_unlabelled_plus_1sd")
    minus = next(c for c in fitter.calls if c["label"] == "sens_g_unlabelled_minus_1sd")
    base = round_x_diffs(design, inputs.rounds[: inputs.n_rounds])
    slope_of = dict(zip(neutral.side_id.tolist(), [2.0, 0.0] * design.n_sides, strict=False))
    expected = np.array(
        [
            slope_of[int(design.side_ids[a])] - slope_of[int(design.side_ids[b])]
            for a, b in zip(design.side_a, design.side_b, strict=True)
        ]
    )
    for x_plus, x_minus, x0 in zip(plus["x_diffs"], minus["x_diffs"], base, strict=True):
        np.testing.assert_allclose(x_plus - x0, expected * sd, atol=1e-6)
        np.testing.assert_allclose(x_minus - x0, -expected * sd, atol=1e-6)
    assert by_name["g_unlabelled_plus_1sd"].counts["g_unlabelled_sd"] == pytest.approx(sd)
    cmp = by_name["g_unlabelled_plus_1sd"].vs_baseline["score"]
    assert "max_abs_rank_change_top" in cmp and "spearman_rho" in cmp and "top_overlap" in cmp


def test_g_unlabelled_slopes_recover_known_b_with_a_era_present() -> None:
    design = _design()
    neutral = _neutral(design, slope=2.0)

    slopes = g_unlabelled_slopes(neutral)

    got = np.array([slopes[int(s)] for s in design.side_ids])
    np.testing.assert_allclose(got[::2], 2.0, atol=1e-9)
    np.testing.assert_allclose(got[1::2], 0.0, atol=1e-9)


def test_shift_rounds_leaves_absent_sides_and_inputs_unchanged() -> None:
    rounds = [ImputedRound(0, 5, "neutral", {1: 9.0, 2: 8.0, 3: 7.0})]

    out = shift_rounds(rounds, {1: 2.0, 2: -1.0}, 0.5)

    assert out[0].log_troops == {1: 10.0, 2: 7.5, 3: 7.0}
    assert rounds[0].log_troops == {1: 9.0, 2: 8.0, 3: 7.0}


@pytest.mark.parametrize("case", ["old_file", "zero_sd", "loader_raises"])
def test_g_unlabelled_variants_not_evaluable_with_a_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str
) -> None:
    design = _design()
    if case == "old_file":
        neutral = _neutral(design, with_g=False)
    else:
        neutral = _neutral(design, g_scale=0.0)

    def loader() -> NeutralDraws:
        if case == "loader_raises":
            raise NotEvaluableError("neutral draws file x.npz not found")
        return neutral

    fitter = FakeFitter()
    _patch_sensitivity(monkeypatch, fitter)
    inputs = _inputs(tmp_path, design, load_neutral=loader)
    specs = [s for s in parse_variants(inputs.params.sensitivity_variants) if "g_unl" in s.name]

    result = run_sensitivity(inputs, specs)

    assert [v.status for v in result.variants] == ["not_evaluable"] * 2
    assert all(v.message for v in result.variants)
    assert not any(c["label"].startswith("sens_g_unl") for c in fitter.calls)
    assert {
        "g_unlabelled" in (v.message or "") or "not found" in (v.message or "")
        for v in result.variants
    } == {True}


# ─── sides without their own usable slope ────────────────────────────────────

NL = chr(10)
_BINS = (("early", 0), ("late", None))


def _weak_neutral(side_ids: list[int], sds: list[float]) -> NeutralDraws:
    rng = np.random.default_rng(5)
    base = rng.normal(0.0, 1.0, (200, len(side_ids)))
    draws = 9.0 + base / base.std(axis=0, ddof=1) * np.array(sds)
    return NeutralDraws(
        run_id=1,
        side_id=np.array(side_ids, dtype=np.int64),
        battle_id=np.arange(len(side_ids), dtype=np.int64),
        draws=draws,
        m0_mean=9.0,
        s0_mean=1.0,
    )


def _side(side_id: int, year: int | None) -> SideInput:
    return SideInput(side_id, side_id, year, None, None, ())


def test_weak_sides_take_the_precision_weighted_era_mean_slope() -> None:
    # Early era: sides 1 (slope 1, sd 1) and 2 (slope 3, sd 0.5, so 4x the precision).
    neutral = _weak_neutral([1, 2, 3, 4], [1.0, 0.5, 1.0, 1.0])
    slopes = {1: 1.0, 2: 3.0, 3: 10.0, 4: 7.0}
    sides = [
        _side(1, -50),
        _side(2, -40),
        _side(3, -30),
        _side(4, 100),
        _side(5, -20),
        _side(6, 900),
    ]

    out, counts = fill_weak_slopes(slopes, neutral, sides, frozenset({3}), _BINS)

    assert out[1] == 1.0 and out[2] == 3.0 and out[4] == 7.0
    assert out[3] == pytest.approx((1.0 * 1.0 + 4.0 * 3.0) / 5.0)  # imputed: not its own 10.0
    assert out[5] == pytest.approx(out[3])  # missing from the npz, same era
    assert out[6] == pytest.approx(7.0)  # late era: the one observed side
    assert counts == {"filled_from_era": 3, "filled_from_overall": 0, "left_at_zero": 0}


def test_an_era_without_observed_sides_uses_the_overall_mean_then_zero() -> None:
    neutral = _weak_neutral([1], [1.0])
    sides = [_side(1, -50), _side(2, 100)]

    out, counts = fill_weak_slopes({1: 2.0}, neutral, sides, frozenset(), _BINS)

    assert out[2] == pytest.approx(2.0) and counts["filled_from_overall"] == 1
    none, counts = fill_weak_slopes({1: 2.0}, neutral, sides, frozenset({1}), _BINS)
    assert none == {1: 0.0, 2: 0.0} and counts["left_at_zero"] == 2


def test_regime_shift_moves_imputed_and_missing_sides_by_the_era_slope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    design = _design()
    neutral = _neutral(design)
    flagged = int(design.side_ids[1])  # odd position: own slope 0, flagged imputed
    keep = neutral.side_id != int(design.side_ids[3])  # absent from the npz
    thin = dataclasses.replace(neutral, side_id=neutral.side_id[keep], draws=neutral.draws[:, keep])
    fitter = FakeFitter()
    _patch_sensitivity(monkeypatch, fitter)
    inputs = dataclasses.replace(
        _inputs(tmp_path, design, load_neutral=lambda: thin),
        load_imputed_sides=lambda: frozenset({flagged}),
        era_bins=(("all", None),),
    )
    specs = [s for s in parse_variants(inputs.params.sensitivity_variants) if "plus" in s.name]

    result = run_sensitivity(inputs, specs)

    sd = float(np.std(neutral.g_unlabelled, ddof=1))
    plus = next(c for c in fitter.calls if c["label"] == "sens_g_unlabelled_plus_1sd")
    base = round_x_diffs(design, inputs.rounds[: inputs.n_rounds])
    slopes, _ = fill_weak_slopes(
        g_unlabelled_slopes(thin),
        thin,
        inputs.sides,
        frozenset({flagged}),
        (("all", None),),
    )
    expected = np.array(
        [
            slopes[int(design.side_ids[a])] - slopes[int(design.side_ids[b])]
            for a, b in zip(design.side_a, design.side_b, strict=True)
        ]
    )
    assert slopes[flagged] != 0.0 and slopes[int(design.side_ids[3])] != 0.0
    for x_plus, x0 in zip(plus["x_diffs"], base, strict=True):
        np.testing.assert_allclose(x_plus - x0, expected * sd, atol=1e-6)
    counts = result.variants[0].counts
    assert counts["sides_imputed_flagged"] == 1 and counts["sides_filled_from_era"] >= 2


def test_read_imputed_side_ids_unions_the_rounds_and_reports_a_bad_file(tmp_path: Path) -> None:
    header = "side_id,battle_id,log_troops,is_imputed,scale"
    rounds = [
        ["1,10,9.0,False,neutral", "2,11,9.1,True,neutral"],
        ["3,12,9.2,true,neutral"],
    ]
    for n, rows in enumerate(rounds):
        d = tmp_path / "neutral" / f"round_{n}"
        d.mkdir(parents=True)
        (d / "sides.csv").write_text(NL.join([header, *rows]) + NL)

    assert read_imputed_side_ids(tmp_path, "neutral", [0, 1]) == frozenset({2, 3})
    with pytest.raises(NotEvaluableError):
        read_imputed_side_ids(tmp_path, "neutral", [5])
