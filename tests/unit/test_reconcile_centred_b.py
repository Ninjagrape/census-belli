"""Opt-in centred battle effect: ``b ~ StudentT(0, s_b)`` sampled, not ``s_b * zb``."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from pipeline.reconcilers.design import UNDATED_ERA_BIN, Design, build_design
from pipeline.reconcilers.model import InflationPrior, ModelPriors, build_model
from pipeline.reconcilers.prior_check import prior_predictive_check
from pipeline.reconcilers.records import Report, SourceKey

pytest.importorskip("pymc")

_NORMAL = ModelPriors(likelihood_family="normal")
_CENTRED = dataclasses.replace(_NORMAL, battle_effect_centred=True)
_SPEC = Path(__file__).resolve().parents[2] / "agents" / "reconcile.yaml"


def _design() -> Design:
    years = [-400, -400, -380, 1200, 1700, 1700, None, None]
    battles = [1, 1, 2, 3, 4, 4, 5, 6]
    reports = [
        Report(
            report_id=i + 1,
            side_id=i + 1,
            battle_id=battles[i],
            source_id=i + 1,
            source_key=SourceKey("wikipedia_infobox", f"https://example/{i}"),
            source_type="wikipedia_infobox",
            quantity="troops",
            reported_value=5_000.0 + 731.0 * (i + 1),
            scope="engaged",
            year_astronomical=years[i],
            claim_regime="unlabelled",
            lineage_id=i + 1,
        )
        for i in range(8)
    ]
    return build_design(reports, quantity="troops", ancient_cutoff_year=500)


def _build(design: Design, priors: ModelPriors) -> Any:
    return build_model(design, priors=priors, inflation=InflationPrior())


def _free(model: Any) -> set[str]:
    return {rv.name for rv in model.free_RVs}


def _battle_bins(design: Design) -> np.ndarray:
    bins = np.full(design.n_battles, UNDATED_ERA_BIN)
    bins[design.battle_of_side] = design.era_bin_of_side
    return bins


def _compiled(model: Any, outputs: list[Any]) -> Any:
    import pytensor

    graphs = model.replace_rvs_by_values(outputs)
    fn = pytensor.function(model.value_vars, graphs, on_unused_input="ignore")

    def run(point: dict[str, np.ndarray]) -> list[np.ndarray]:
        return [np.asarray(x) for x in fn(*[point[v.name] for v in model.value_vars])]

    return run


def _point(model: Any, seed: int = 3) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {k: v + rng.normal(0.0, 0.6, size=np.shape(v)) for k, v in model.initial_point().items()}


def test_param_defaults_off_and_parses_booleans_and_strings() -> None:
    assert ModelPriors().battle_effect_centred is False
    assert ModelPriors.from_params({}, quantity="troops").battle_effect_centred is False
    for raw in (True, "true", "True"):
        priors = ModelPriors.from_params({"battle_effect_centred": raw}, quantity="troops")
        assert priors.battle_effect_centred is True
    off = ModelPriors.from_params({"battle_effect_centred": "false"}, quantity="troops")
    assert off.battle_effect_centred is False
    with pytest.raises(ValueError):
        ModelPriors.from_params({"battle_effect_centred": "maybe"}, quantity="troops")


def test_the_spec_value_parses_and_is_off() -> None:
    params = yaml.safe_load(_SPEC.read_text(encoding="utf-8"))["params"]
    assert params["battle_effect_centred"] is False
    priors = ModelPriors.from_params(params, quantity="troops")
    assert priors.battle_effect_centred is False


def test_default_mode_keeps_zb_and_has_no_b_raw() -> None:
    model = _build(_design(), _NORMAL)
    free = _free(model)
    assert "zb" in free
    assert "b_raw" not in free
    assert "b" in model.named_vars


def test_centred_mode_samples_b_raw_and_has_no_zb() -> None:
    model = _build(_design(), _CENTRED)
    free = _free(model)
    assert "b_raw" in free
    assert "zb" not in free
    assert "zb" not in model.named_vars
    assert "b" in model.named_vars
    assert "b" not in free


def test_both_modes_give_the_same_b_at_a_matched_point() -> None:
    design = _design()
    nc = _build(design, _NORMAL)
    c = _build(design, _CENTRED)
    point_nc = _point(nc)
    s_b, zb, b_nc = _compiled(nc, [nc["s_b"], nc["zb"], nc["b"]])(point_nc)
    point_c = {v.name: point_nc.get(v.name) for v in c.value_vars if v.name in point_nc}
    point_c["b_raw"] = float(s_b) * zb
    (b_c,) = _compiled(c, [c["b"]])(point_c)
    assert b_c == pytest.approx(b_nc, abs=1e-10)
    assert (_battle_bins(design) == UNDATED_ERA_BIN).any()


def test_centred_b_has_zero_mean_per_bin_and_undated_battles_are_untouched() -> None:
    design = _design()
    model = _build(design, _CENTRED)
    point = _point(model)
    b, raw = _compiled(model, [model["b"], model["b_raw"]])(point)
    bins = _battle_bins(design)
    for k in sorted(set(bins.tolist()) - {UNDATED_ERA_BIN}):
        assert b[bins == k].mean() == pytest.approx(0.0, abs=1e-10)
    assert b[bins == UNDATED_ERA_BIN] == pytest.approx(raw[bins == UNDATED_ERA_BIN])


def test_logp_differs_from_the_non_centred_only_by_the_scale_jacobian() -> None:
    design = _design()
    nc = _build(design, _NORMAL)
    c = _build(design, _CENTRED)
    base = _point(nc)
    shared = {k: v for k, v in base.items() if k != "zb"}
    logp_nc = nc.compile_logp()
    logp_c = c.compile_logp()
    n = design.n_battles
    rng = np.random.default_rng(11)
    for s_b in (0.4, 1.3):
        # s_b enters the point on the log scale; zb is the same draw rescaled.
        point = {**shared, "s_b_log__": np.log(s_b)}
        b_raw = rng.normal(0.0, 1.0, size=n)
        got_c = float(logp_c({**point, "b_raw": b_raw}))
        got_nc = float(logp_nc({**point, "zb": b_raw / s_b}))
        assert got_c - got_nc == pytest.approx(-n * np.log(s_b), abs=1e-8)


def test_prior_check_runs_in_centred_mode() -> None:
    design = _design()
    block = prior_predictive_check(_build(design, _CENTRED), design, draws=100, seed=1)
    assert block["draws"] == 100
    q = block["strength_quantiles"]
    assert q["p01"] < q["p50"] < q["p99"]
