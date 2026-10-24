"""P1 troop side prior: structure of the built model, no sampling.

Troops take ``mu_side = m0 + e[era_bin] + s_b * zb[battle] + s_d * zd[side]`` with a
Normal likelihood, informative noise and no ``sigma_round`` / ``sigma_estimate`` /
``s0``. Casualties keep their original model exactly. Unlabelled rows from 1800 on
are read at face value (the level anchor), which the compiled graph must honour.
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
    model = build_model(design, priors=ModelPriors(), inflation=InflationPrior())
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
    assert abs(moved[7]) > 0.5  # unlabelled 1799


def test_the_rotation_weights_count_only_the_offsets_actually_applied() -> None:
    # offset_bar is the corpus average of the offsets the rows carry; with the
    # anchor, a masked row carries no regime offset, so it must not be averaged in.
    design = _anchor_design()
    model = build_model(design, priors=ModelPriors(), inflation=InflationPrior())
    point = model.initial_point()
    point["g_free"] = point["g_free"] + 0.7
    g, delta, bar = _eval(
        model, [model["g_regime"], model["delta_level"], model["offset_bar"]], point
    )
    keep = 1.0 - level_anchor_mask(design)

    applied = np.mean(g[design.regime_index] * keep + delta[design.level_index])

    assert float(bar) == pytest.approx(float(applied), abs=1e-10)


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
    assert {"tau_e", "zb", "zd", "s_b", "s_d"} <= free
    assert "e" in model.named_vars
    assert not {"s0", "z_side", "sigma_round", "sigma_estimate"} & set(model.named_vars)
    assert "mu_side" in model.named_vars


def test_casualties_keep_s0_z_side_and_their_dispersion_terms() -> None:
    model = build_model(
        _mixed_design("casualties"), priors=ModelPriors(), inflation=InflationPrior()
    )

    free = _free_names(model)
    assert {"s0", "z_side", "sigma_round", "sigma_estimate"} <= free
    assert not {"tau_e", "zb", "zd", "s_b", "s_d"} & set(model.named_vars)
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
    design = _mixed_design("troops")
    model = build_model(design, priors=ModelPriors(), inflation=InflationPrior())
    point = model.initial_point()
    point["zd"] = np.zeros_like(point["zd"])
    point["zb"] = np.linspace(-1.0, 1.0, np.size(point["zb"]))
    point["s_b_log__"] = np.asarray(0.0)  # s_b = 1

    (mu,) = _eval(model, [model["mu_side"]], point)
    by_id = dict(zip(design.side_ids, mu, strict=True))

    # Sides 1 and 2 are one battle: same era bin, same zb, zd zero, so the same mu.
    assert by_id[1] == pytest.approx(by_id[2])
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
