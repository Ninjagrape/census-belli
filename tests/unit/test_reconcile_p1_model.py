"""P1 troop side prior: structure of the built model, no sampling.

Troops take ``mu_side = m0 + e[era_bin] + b[battle] + s_d * zd[side]`` (``b``
non-centred, ``e`` rotated by default) with a Normal likelihood, informative noise and no
``sigma_round`` / ``sigma_estimate`` / ``s0``. Casualties keep their original model exactly.
Unlabelled rows from 1800 on are read at face value (the level anchor), which the compiled graph
must honour.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from pipeline.reconcilers.design import UNDATED_ERA_BIN, Design, build_design
from pipeline.reconcilers.model import (
    InflationPrior,
    ModelPriors,
    _unlabelled_era_levels,
    _zero_sum_basis,
    build_model,
    level_anchor_mask,
)
from pipeline.reconcilers.records import Report, SourceKey

pytest.importorskip("pymc")

AGENTS = Path(__file__).resolve().parents[2] / "agents"


def _report(
    report_id: int,
    *,
    side_id: int,
    year: int | None,
    regime: str = "unlabelled",
    quantity: str = "troops",
    source_type: str = "wikipedia_infobox",
    battle_id: int | None = None,
) -> Report:
    return Report(
        report_id=report_id,
        side_id=side_id,
        battle_id=side_id if battle_id is None else battle_id,
        source_id=report_id,
        source_key=SourceKey(source_type, f"https://example/{source_type}/{report_id}"),
        source_type=source_type,
        quantity=quantity,  # type: ignore[arg-type]
        reported_value=5_000.0 + 137.0 * report_id,
        scope="engaged" if quantity == "troops" else "unknown",
        year_astronomical=year,
        claim_regime=regime,
        lineage_id=report_id,
    )


# (year, regime) per row; one side per row. Row order is the report order.
_ANCHOR_CASES: list[tuple[int | None, str]] = [
    (1850, "unlabelled"),  # 0 masked
    (-200, "unlabelled"),  # 1 pre-1800: keeps g and a_era
    (None, "unlabelled"),  # 2 undated: keeps g and a_era
    (1850, "chronicle"),  # 3 labelled: untouched
    (1850, "modern_scholarly"),  # 4 anchor regime
    (1800, "unlabelled"),  # 5 boundary year is masked
    (1799, "unlabelled"),  # 6 one year earlier is not
]


def _anchor_design(quantity: str = "troops") -> Design:
    types = ("wikipedia_infobox", "wikidata")
    reports = [
        _report(
            i + 1,
            side_id=i + 1,
            year=year,
            regime=regime,
            quantity=quantity,
            source_type=types[i % 2],
        )
        for i, (year, regime) in enumerate(_ANCHOR_CASES)
    ]
    return build_design(reports, quantity=quantity, ancient_cutoff_year=500)


def _eval(model: Any, outputs: list[Any], point: dict[str, np.ndarray]) -> list[np.ndarray]:
    import pytensor

    graphs = model.replace_rvs_by_values(outputs)
    compiled = pytensor.function(model.value_vars, graphs, on_unused_input="ignore")
    return [np.asarray(x) for x in compiled(*[point[v.name] for v in model.value_vars])]


def _free_names(model: Any) -> set[str]:
    return {rv.name for rv in model.free_RVs}


def _row_order(design: Design) -> dict[int, int]:
    """Design row index of each side id (one row per side in these fixtures)."""
    return {design.side_ids[int(design.side_index[i])]: i for i in range(design.n_obs)}


# ─── the anchor mask ─────────────────────────────────────────────────────────


def test_the_anchor_mask_flags_unlabelled_rows_from_1800_only() -> None:
    design = _anchor_design()
    mask = level_anchor_mask(design)
    by_side = {s: mask[i] for s, i in _row_order(design).items()}

    assert [by_side[s] for s in range(1, 8)] == [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def test_casualties_have_no_anchor_mask() -> None:
    design = _anchor_design("casualties")

    assert not level_anchor_mask(design).any()


def test_a_masked_row_ignores_the_unlabelled_offset_and_the_era_term() -> None:
    design = _anchor_design()
    legacy = ModelPriors(era_effect="legacy")
    model = build_model(design, priors=legacy, inflation=InflationPrior())
    base = model.initial_point()
    shifted = {k: v.copy() for k, v in base.items()}
    shifted["g_free"] = shifted["g_free"] + 1.0
    shifted["a_era"] = shifted["a_era"] + 1.0

    eta_a = _eval(model, [model.row_eta], base)[0]
    eta_b = _eval(model, [model.row_eta], shifted)[0]
    raw = {s: eta_b[i] - eta_a[i] for s, i in _row_order(design).items()}
    # m0 is derived from the sampled corpus level, so moving g shifts every row by the
    # same amount; the anchor-regime row (no g, no era term) measures that shift.
    moved = {s: v - raw[5] for s, v in raw.items()}

    assert moved[1] == pytest.approx(0.0, abs=1e-12)  # unlabelled 1850
    assert moved[6] == pytest.approx(0.0, abs=1e-12)  # unlabelled 1800
    assert abs(moved[2]) > 0.5  # unlabelled 200 BC: g_unlabelled and a_era both move it
    assert abs(moved[3]) > 0.5  # undated unlabelled
    assert moved[4] == pytest.approx(1.0)  # chronicle: g only, no era term
    assert moved[5] == pytest.approx(0.0, abs=1e-12)  # anchor regime
    assert moved[7] == pytest.approx(0.5)  # unlabelled 1799: g (+1) and the recentred era (-0.5)


def test_the_rotation_weights_count_only_the_offsets_actually_applied() -> None:
    # offset_bar is the corpus average of the offsets the rows carry; with the
    # anchor, a masked row carries no regime offset, so it must not be averaged in.
    design = _anchor_design()
    model = build_model(design, priors=ModelPriors(era_effect="legacy"), inflation=InflationPrior())
    point = model.initial_point()
    point["g_free"] = point["g_free"] + 0.7
    g, delta, bar = _eval(
        model, [model["g_regime"], model["delta_level"], model["offset_bar"]], point
    )
    keep = 1.0 - level_anchor_mask(design)

    applied = np.mean(g[design.regime_index] * keep + delta[design.level_index])

    assert float(bar) == pytest.approx(float(applied), abs=1e-10)


def test_the_rotated_offset_bar_drops_unlabelled_weight_and_adds_the_era_mean() -> None:
    # offset_bar = mean(labelled g * keep + delta) + hbar, hbar = mean_k(h_k): the
    # unlabelled regime's offset enters only through hbar and u_c.
    design = _anchor_design()
    model = build_model(design, priors=ModelPriors(), inflation=InflationPrior())
    point = model.initial_point()
    point["g_free"] = point["g_free"] + 0.7
    point["a_era"] = point["a_era"] + 0.3
    g, delta, bar, a_era = _eval(
        model, [model["g_regime"], model["delta_level"], model["offset_bar"], model["a_era"]], point
    )
    levels, pre = _unlabelled_era_levels(design, 500)
    unl = design.regimes.index("unlabelled")
    labelled = np.array([r != unl for r in design.regime_index], dtype=float)
    keep = 1.0 - level_anchor_mask(design)
    rows = g[design.regime_index] * keep * labelled + delta[design.level_index]
    hbar = np.mean(g[unl] * pre + float(a_era) * levels)

    assert float(bar) == pytest.approx(float(rows.mean() + hbar), abs=1e-10)


# ─── graph structure ─────────────────────────────────────────────────────────


def _mixed_design(quantity: str) -> Design:
    years = [-400, -400, 1200, 1700, 1815, 1815, 1870, 1944, None, None]
    reports = [
        _report(
            i + 1,
            side_id=i + 1,
            battle_id=(i // 2) + 1,
            year=years[i],
            quantity=quantity,
            source_type=("wikipedia_infobox", "wikidata")[i % 2],
        )
        for i in range(10)
    ]
    return build_design(reports, quantity=quantity, ancient_cutoff_year=500)


def test_troops_use_the_era_battle_deviation_prior_and_drop_the_old_terms() -> None:
    model = build_model(_mixed_design("troops"), priors=ModelPriors(), inflation=InflationPrior())

    free = _free_names(model)
    assert {"u_c_free", "zb", "zd", "s_b", "s_d"} <= free
    assert not {"tau_e", "z_era", "e"} & free
    assert "e_prior" in model.named_vars
    assert "b" in model.named_vars
    assert not {"s0", "z_side", "sigma_round", "sigma_estimate"} & set(model.named_vars)
    assert "mu_side" in model.named_vars


def test_legacy_era_effect_keeps_todays_free_variables() -> None:
    model = build_model(
        _mixed_design("troops"),
        priors=ModelPriors(era_effect="legacy"),
        inflation=InflationPrior(),
    )

    free = _free_names(model)
    assert {"tau_e", "z_era", "zb", "zd", "s_b", "s_d", "a_era", "corpus_level"} <= free
    assert not {"u_c_free", "u_c"} & set(model.named_vars)
    assert "e" not in free
    assert "e_prior" not in model.named_vars


def test_none_era_effect_has_no_period_mean() -> None:
    design = _mixed_design("troops")
    model = build_model(design, priors=ModelPriors(era_effect="none"), inflation=InflationPrior())

    assert not {"e", "tau_e", "z_era", "u_c", "u_c_free", "e_prior"} & set(model.named_vars)
    point = model.initial_point()
    point["zb"] = np.zeros_like(point["zb"])
    point["zd"] = np.zeros_like(point["zd"])
    mu, m0 = _eval(model, [model["mu_side"], model["m0"]], point)
    assert mu == pytest.approx(np.full(design.n_sides, float(m0)))


def test_an_unknown_era_effect_raises() -> None:
    with pytest.raises(ValueError, match="era_effect"):
        ModelPriors.from_params({"era_effect": "rotatd"}, quantity="troops")
    with pytest.raises(ValueError, match="era_effect"):
        build_model(
            _mixed_design("troops"),
            priors=ModelPriors(era_effect="bogus"),
            inflation=InflationPrior(),
        )


def test_era_effect_defaults_to_rotated_and_reads_from_the_spec() -> None:
    assert ModelPriors().era_effect == "rotated"
    params = yaml.safe_load((AGENTS / "reconcile.yaml").read_text(encoding="utf-8"))["params"]
    assert ModelPriors.from_params(params, quantity="troops").era_effect == "rotated"
    legacy = ModelPriors.from_params({"era_effect": "legacy"}, quantity="troops")
    assert legacy.era_effect == "legacy"


def test_casualties_keep_s0_z_side_and_their_dispersion_terms() -> None:
    model = build_model(
        _mixed_design("casualties"),
        priors=ModelPriors.from_params({}, quantity="casualties"),
        inflation=InflationPrior(),
    )

    free = _free_names(model)
    assert {"s0", "z_side", "sigma_round", "sigma_estimate"} <= free
    assert not {"tau_e", "zb", "zd", "s_b", "s_d", "u_c"} & set(model.named_vars)
    assert "e" not in model.named_vars


def test_era_effects_sum_to_zero_over_the_five_bins() -> None:
    model = build_model(_mixed_design("troops"), priors=ModelPriors(), inflation=InflationPrior())
    rng = np.random.default_rng(3)
    point = {
        k: v + rng.normal(0.0, 1.0, size=np.shape(v)) for k, v in model.initial_point().items()
    }

    (e,) = _eval(model, [model["e"]], point)

    assert e.shape == (5,)
    assert float(e.sum()) == pytest.approx(0.0, abs=1e-10)


def test_an_undated_side_sits_at_the_corpus_level_with_no_era_effect() -> None:
    design = _mixed_design("troops")
    model = build_model(design, priors=ModelPriors(), inflation=InflationPrior())
    rng = np.random.default_rng(4)
    point = {
        k: v + rng.normal(0.0, 1.0, size=np.shape(v)) for k, v in model.initial_point().items()
    }
    point["zb"] = np.zeros_like(point["zb"])
    point["zd"] = np.zeros_like(point["zd"])

    mu, m0, e = _eval(model, [model["mu_side"], model["m0"], model["e"]], point)

    for position in range(design.n_sides):
        bin_ = int(design.era_bin_of_side[position])
        expected = float(m0) + (0.0 if bin_ == UNDATED_ERA_BIN else float(e[bin_]))
        assert mu[position] == pytest.approx(expected, abs=1e-10)
    assert (design.era_bin_of_side == UNDATED_ERA_BIN).any()


def test_battle_effects_are_shared_by_the_two_sides_of_a_battle() -> None:
    # Two battles in one era bin, two sides each: centring leaves their effects nonzero
    # (opposite signs), and the two sides of one battle still share theirs.
    reports = [
        _report(i + 1, side_id=i + 1, battle_id=i // 2 + 1, year=1200, source_type="wikidata")
        for i in range(4)
    ]
    design = build_design(reports, quantity="troops", ancient_cutoff_year=500)
    model = build_model(design, priors=ModelPriors(), inflation=InflationPrior())
    point = model.initial_point()
    point["zd"] = np.zeros_like(point["zd"])
    point["zb"] = np.array([-1.0, 1.0])

    (mu,) = _eval(model, [model["mu_side"]], point)
    by_id = dict(zip(design.side_ids, mu, strict=True))

    assert by_id[1] == pytest.approx(by_id[2])
    assert by_id[3] == pytest.approx(by_id[4])
    assert by_id[1] != pytest.approx(by_id[3])


def test_the_default_noise_priors_come_from_the_spec_and_the_family_is_normal() -> None:
    params = yaml.safe_load((AGENTS / "reconcile.yaml").read_text(encoding="utf-8"))["params"]
    troops = ModelPriors.from_params(params, quantity="troops")
    casualties = ModelPriors.from_params(params, quantity="casualties")

    assert troops.likelihood_family == "normal"
    assert troops.noise_modern_median == pytest.approx(0.22)
    assert troops.noise_modern_log_sd == pytest.approx(0.2)
    assert troops.noise_ancient_median == pytest.approx(0.40)
    assert troops.noise_ancient_log_sd == pytest.approx(0.2)
    # Casualties keep their Student-t likelihood: the redesign is troops only.
    assert casualties.likelihood_family == "student_t"


def test_troop_noise_priors_are_lognormal_at_the_stated_medians() -> None:
    priors = ModelPriors(
        likelihood_family="normal",
        noise_modern_median=0.22,
        noise_modern_log_sd=0.2,
        noise_ancient_median=0.40,
        noise_ancient_log_sd=0.2,
    )
    model = build_model(_mixed_design("troops"), priors=priors, inflation=InflationPrior())

    for name, median in (("sigma_type_modern", 0.22), ("sigma_type_ancient", 0.40)):
        mu, sigma = model.named_vars[name].owner.inputs[-2:]
        assert float(np.mean(mu.eval())) == pytest.approx(np.log(median))
        assert float(np.mean(sigma.eval())) == pytest.approx(0.2)


def _logp_of(model: Any, name: str, point: dict[str, Any]) -> float:
    return float(model.compile_logp(vars=[model[name]])(point))


def test_the_legacy_era_effect_stays_non_centred_because_centring_it_failed_to_converge() -> None:
    model = build_model(
        _mixed_design("troops"),
        priors=ModelPriors(era_effect="legacy"),
        inflation=InflationPrior(),
    )
    point = model.initial_point()
    point["z_era_zerosum__"] = np.array([0.4, -0.2, 0.3, 0.1])
    point["tau_e_log__"] = np.asarray(0.0)
    (at_one,) = _eval(model, [model["e"]], point)
    point["tau_e_log__"] = np.asarray(np.log(2.0))
    (at_two,) = _eval(model, [model["e"]], point)

    # e is tau_e times a fixed zero-sum vector, so it has no density of its own.
    assert at_two.sum() == pytest.approx(0.0, abs=1e-9)
    assert at_two == pytest.approx(2.0 * at_one)


def test_the_battle_effect_stays_non_centred_because_centring_it_diverged() -> None:
    # Non-centred in the scale: b is s_b times a fixed (bin-centred) vector, so it has no
    # density of its own at fixed zb and doubles when s_b does.
    model = build_model(_mixed_design("troops"), priors=ModelPriors(), inflation=InflationPrior())
    point = model.initial_point()
    point["zb"] = np.linspace(-1.0, 1.0, np.size(point["zb"]))
    point["s_b_log__"] = np.asarray(0.0)
    (at_one,) = _eval(model, [model["b"]], point)
    point["s_b_log__"] = np.asarray(np.log(2.0))
    (at_two,) = _eval(model, [model["b"]], point)

    assert at_two == pytest.approx(2.0 * at_one)


def test_the_independent_sides_variant_drops_the_battle_effect() -> None:
    from scipy import stats

    priors = ModelPriors(
        side_prior_independent=True, battle_effect_prior_sd=1.0, side_deviation_prior_sd=0.5
    )
    model = build_model(_mixed_design("troops"), priors=priors, inflation=InflationPrior())

    assert not {"b", "zb", "s_b"} & set(model.named_vars)
    assert {"e", "zd", "s_d", "mu_side"} <= set(model.named_vars)
    # s_d ~ HalfNormal(sqrt(1.0**2 + 0.5**2)): total side variance matches P1's.
    point = model.initial_point()
    point["s_d_log__"] = np.asarray(0.0)  # s_d = 1
    scale = float(np.sqrt(1.0**2 + 0.5**2))
    expected = float(stats.halfnorm.logpdf(1.0, scale=scale)) + 0.0  # + log jacobian (s_d = 1)
    assert _logp_of(model, "s_d", point) == pytest.approx(expected, abs=1e-6)


def test_side_prior_independent_reads_from_the_spec_and_defaults_off() -> None:
    assert ModelPriors().side_prior_independent is False
    on = ModelPriors.from_params({"side_prior_independent": True}, quantity="troops")
    assert on.side_prior_independent is True


# ─── the rotated era effect ──────────────────────────────────────────────────


def _rotation_design(years: list[int | None], regimes: list[str], *, cutoff: int = 500) -> Design:
    types = ("wikipedia_infobox", "wikidata")
    reports = [
        _report(i + 1, side_id=i + 1, year=year, regime=regime, source_type=types[i % 2])
        for i, (year, regime) in enumerate(zip(years, regimes, strict=True))
    ]
    return build_design(reports, quantity="troops", ancient_cutoff_year=cutoff)


# Bins 0 to 4 all have unlabelled rows, plus an undated unlabelled row, labelled rows in
# pre-1800 bins and an anchor-regime row.
_ROT_YEARS: list[int | None] = [-400, 1200, 1700, 1815, 1950, None, -300, 1200, 1850, 1700]
_ROT_REGIMES = [
    "unlabelled",
    "unlabelled",
    "unlabelled",
    "unlabelled",
    "unlabelled",
    "unlabelled",
    "chronicle",
    "chronicle",
    "modern_scholarly",
    "unlabelled",
]


def _rot_point(model: Any, design: Design, *, dg: float, da: float) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(11)
    point = {
        k: v + rng.normal(0.0, 0.5, size=np.shape(v)) for k, v in model.initial_point().items()
    }
    free = [r for r in design.regimes if r != "modern_scholarly"]
    point["g_free"] = point["g_free"].copy()
    point["g_free"][free.index("unlabelled")] += dg
    point["a_era"] = point["a_era"] + da
    return point


def _unlabelled_dated_rows(design: Design) -> list[int]:
    return [
        i
        for i in range(design.n_obs)
        if design.regimes[design.regime_index[i]] == "unlabelled"
        and not np.isnan(design.year_of_row[i])
    ]


def test_dated_unlabelled_rows_do_not_move_with_g_unlabelled_or_a_era_at_fixed_coordinates() -> (
    None
):
    design = _rotation_design(_ROT_YEARS, _ROT_REGIMES)
    model = build_model(design, priors=ModelPriors(), inflation=InflationPrior())

    eta_a = _eval(model, [model.row_eta], _rot_point(model, design, dg=0.0, da=0.0))[0]
    eta_b = _eval(model, [model.row_eta], _rot_point(model, design, dg=0.9, da=0.7))[0]
    diff = eta_b - eta_a
    by_side = {s: diff[i] for s, i in _row_order(design).items()}

    dated = _unlabelled_dated_rows(design)
    assert len(dated) == 6
    assert diff[dated] == pytest.approx(np.zeros(len(dated)), abs=1e-10)
    assert abs(by_side[6]) > 1e-3  # undated unlabelled
    assert abs(by_side[7]) > 1e-3  # chronicle in bin 0
    assert abs(by_side[8]) > 1e-3  # chronicle in bin 1


def test_the_same_invariance_holds_when_no_row_is_undated() -> None:
    keep = [i for i, y in enumerate(_ROT_YEARS) if y is not None]
    design = _rotation_design([_ROT_YEARS[i] for i in keep], [_ROT_REGIMES[i] for i in keep])
    model = build_model(design, priors=ModelPriors(), inflation=InflationPrior())

    eta_a = _eval(model, [model.row_eta], _rot_point(model, design, dg=0.0, da=0.0))[0]
    eta_b = _eval(model, [model.row_eta], _rot_point(model, design, dg=-1.3, da=0.4))[0]
    dated = _unlabelled_dated_rows(design)

    assert dated
    assert (eta_b - eta_a)[dated] == pytest.approx(np.zeros(len(dated)), abs=1e-10)


def test_the_e_prior_potential_is_the_zero_sum_normal_logp_of_e() -> None:
    import pymc as pm

    sigma = 0.7
    design = _rotation_design(_ROT_YEARS, _ROT_REGIMES)
    model = build_model(
        design, priors=ModelPriors(era_effect_prior_sd=sigma), inflation=InflationPrior()
    )
    point = _rot_point(model, design, dg=0.3, da=-0.2)

    e, potential = _eval(model, [model["e"], model["e_prior"]], point)
    direct = float(pm.logp(pm.ZeroSumNormal.dist(sigma=sigma, shape=5), e).eval())
    closed = float(-0.5 * np.sum(e**2) / sigma**2 - 4.0 * (np.log(sigma) + 0.5 * np.log(2 * np.pi)))

    assert float(e.sum()) == pytest.approx(0.0, abs=1e-10)
    assert float(potential) == pytest.approx(direct, abs=1e-9)
    assert float(potential) == pytest.approx(closed, abs=1e-9)


def test_the_zero_sum_basis_is_orthonormal_and_sums_to_zero() -> None:
    basis = _zero_sum_basis(5)

    assert basis.shape == (5, 4)
    assert basis.T @ basis == pytest.approx(np.eye(4), abs=1e-12)
    assert basis.sum(axis=0) == pytest.approx(np.zeros(4), abs=1e-12)


def test_the_neutral_fit_builds_with_the_rotated_era_effect() -> None:
    design = _rotation_design(_ROT_YEARS, _ROT_REGIMES)
    model = build_model(design, priors=ModelPriors(), inflation=InflationPrior(), neutral=True)

    assert {"u_c_free", "a_era", "g_free"} <= _free_names(model)
    assert "e_prior" in model.named_vars
    eta_a = _eval(model, [model.row_eta], _rot_point(model, design, dg=0.0, da=0.0))[0]
    eta_b = _eval(model, [model.row_eta], _rot_point(model, design, dg=0.9, da=0.7))[0]
    dated = _unlabelled_dated_rows(design)
    assert (eta_b - eta_a)[dated] == pytest.approx(np.zeros(len(dated)), abs=1e-10)


def test_era_levels_come_from_the_design_and_empty_pre_anchor_bins_take_the_flag() -> None:
    # Unlabelled rows only in bins 0 and 2: the ancient share is 1/2, so c = (0.5, ?, -0.5)
    # and the empty bin 1 takes flag 0 minus that mean.
    design = _rotation_design([-400, 1700, 1850], ["unlabelled"] * 3)

    levels, pre = _unlabelled_era_levels(design, 500)

    assert pre.tolist() == [1.0, 1.0, 1.0, 0.0, 0.0]
    assert levels.tolist() == pytest.approx([0.5, -0.5, -0.5, 0.0, 0.0])


def test_era_levels_default_to_the_bare_flag_when_no_unlabelled_row_applies() -> None:
    design = _rotation_design([-400, 1850], ["chronicle", "unlabelled"])

    levels, _ = _unlabelled_era_levels(design, 500)

    assert levels.tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0, 0.0])


def test_a_cutoff_that_differs_from_the_first_era_edge_raises() -> None:
    design = _rotation_design(_ROT_YEARS, _ROT_REGIMES)

    with pytest.raises(ValueError, match="ancient_cutoff_year"):
        build_model(
            design, priors=ModelPriors(), inflation=InflationPrior(), ancient_cutoff_year=1000
        )
    for mode in ("legacy", "none"):
        build_model(
            design,
            priors=ModelPriors(era_effect=mode),
            inflation=InflationPrior(),
            ancient_cutoff_year=1000,
        )


def test_an_era_flag_that_varies_within_a_bin_raises() -> None:
    # Cutoff 1000 splits the 500-1499 bin: 600 flags 1, 1200 flags 0.
    design = _rotation_design([600, 1200, 1700], ["unlabelled"] * 3, cutoff=1000)

    with pytest.raises(ValueError, match="not constant within era bin"):
        build_model(design, priors=ModelPriors(), inflation=InflationPrior())


def test_the_side_prior_diagnostics_record_the_era_effect_mode() -> None:
    az = pytest.importorskip("arviz")
    from pipeline.reconcilers.summarise import _side_prior_stats

    rng = np.random.default_rng(5)
    e = rng.normal(size=(2, 50, 5))
    e -= e.mean(axis=-1, keepdims=True)
    posterior = {
        "s_d": np.abs(rng.normal(size=(2, 50))),
        "s_b": np.abs(rng.normal(size=(2, 50))),
        "e": e,
        "a_era": rng.normal(size=(2, 50)),
    }
    stats = _side_prior_stats(az.from_dict(posterior=posterior), ModelPriors())
    assert stats is not None
    era = stats["era"]

    assert era["era_effect"] == "rotated"
    assert era["sigma_e"] == 1.0
    assert "tau_e" not in stats
    contrast = e[..., :3].mean(-1) - e[..., 3:].mean(-1)
    assert era["e_contrast_pre_vs_post_1800"]["mean"] == pytest.approx(float(contrast.mean()))
    assert "a_era" in era

    legacy = {**posterior, "tau_e": np.abs(rng.normal(size=(2, 50)))}
    legacy_stats = _side_prior_stats(az.from_dict(posterior=legacy))
    assert legacy_stats is not None
    assert "tau_e" in legacy_stats
    assert legacy_stats["era"]["era_effect"] == "legacy"


def test_u_c_has_a_flat_density_and_a_gradient_that_compiles() -> None:
    design = _rotation_design(_ROT_YEARS, _ROT_REGIMES)
    model = build_model(design, priors=ModelPriors(), inflation=InflationPrior())
    point = _rot_point(model, design, dg=0.0, da=0.0)
    shifted = {**point, "u_c_free": point["u_c_free"] + 0.5}

    own = model.compile_logp(vars=[model["u_c_free"]])
    assert own(point) == pytest.approx(0.0)
    assert own(shifted) == pytest.approx(0.0)
    grad = model.compile_dlogp()(point)
    assert np.all(np.isfinite(grad))
