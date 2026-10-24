"""The informative observation-noise prior, the nu floor and the era scale.

Build-time checks only: prior draws come from the unsampled model's random
variables (``pm.draw``), so nothing here runs NUTS.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pipeline.reconcilers.design import build_design
from pipeline.reconcilers.model import InflationPrior, ModelPriors, build_model
from pipeline.reconcilers.records import Report, SourceKey

pm = pytest.importorskip("pymc")

FAITHFUL_SD = 0.2223
N_DRAWS = 6000


def _report(report_id: int, *, side_id: int, year: int | None, quantity: str = "troops") -> Report:
    return Report(
        report_id=report_id,
        side_id=side_id,
        battle_id=side_id,
        source_id=report_id,
        source_key=SourceKey("wikipedia_infobox", f"https://example/{report_id % 2}"),
        source_type="wikipedia_infobox",
        quantity=quantity,  # type: ignore[arg-type]
        reported_value=5_000.0 + 100.0 * report_id,
        scope="engaged" if quantity == "troops" else "unknown",
        year_astronomical=year,
        claim_regime="unlabelled",
        lineage_id=report_id,
    )


def _reports(
    quantity: str = "troops", years: tuple[int | None, ...] = (-200, 1800)
) -> list[Report]:
    return [
        _report(i, side_id=1 + i % 4, year=years[i % len(years)], quantity=quantity)
        for i in range(12)
    ]


def _design(quantity: str = "troops", **kwargs: object):  # type: ignore[no-untyped-def]
    years = kwargs.pop("years", (-200, 1800))
    return build_design(
        _reports(quantity, years),  # type: ignore[arg-type]
        quantity=quantity,
        ancient_cutoff_year=500,
        **kwargs,  # type: ignore[arg-type]
    )


def _inflation() -> InflationPrior:
    return InflationPrior(faithful_sd=FAITHFUL_SD)


def _draw(model: object, name: str) -> np.ndarray:
    with model:  # type: ignore[attr-defined]
        return np.asarray(pm.draw(model[name], draws=N_DRAWS, random_seed=1))  # type: ignore[index]


def _free(model: object) -> set[str]:
    return {rv.name for rv in model.free_RVs}  # type: ignore[attr-defined]


def test_the_defaults_are_the_approved_prior() -> None:
    priors = ModelPriors.from_params({}, quantity="troops")

    assert priors.observation_noise_prior == "informative"
    assert priors.noise_modern_median is None
    assert (priors.noise_modern_log_sd, priors.noise_ancient_median) == (0.3, 0.40)
    assert priors.noise_ancient_log_sd == 0.35
    assert priors.student_t_nu_floor == 3.0
    assert (priors.estimate_dispersion_sd, priors.roundness_dispersion_sd) == (0.1, 0.1)


def test_casualties_keep_their_dispersion_scales() -> None:
    priors = ModelPriors.from_params({}, quantity="casualties")

    assert (priors.estimate_dispersion_sd, priors.roundness_dispersion_sd) == (0.3, 0.3)
    explicit = ModelPriors.from_params(
        {"casualty_estimate_dispersion_sd": 0.2}, quantity="casualties"
    )
    assert explicit.estimate_dispersion_sd == 0.2


def test_an_unknown_noise_prior_name_is_refused() -> None:
    with pytest.raises(ValueError, match="observation_noise_prior"):
        ModelPriors.from_params({"observation_noise_prior": "informativ"}, quantity="troops")


def test_the_modern_scale_is_lognormal_around_the_gold_set_faithful_sd() -> None:
    model = build_model(_design(), priors=ModelPriors(), inflation=_inflation())

    draws = _draw(model, "sigma_type_modern")

    assert "sigma_type_modern" in _free(model)
    assert math.isclose(float(np.median(draws)), FAITHFUL_SD, rel_tol=0.03)
    assert math.isclose(float(np.std(np.log(draws))), 0.3, abs_tol=0.015)


def test_the_ancient_scale_is_wider_and_centred_at_point_four() -> None:
    model = build_model(_design(), priors=ModelPriors(), inflation=_inflation())

    draws = _draw(model, "sigma_type_ancient")

    assert math.isclose(float(np.median(draws)), 0.40, rel_tol=0.03)
    assert math.isclose(float(np.std(np.log(draws))), 0.35, abs_tol=0.02)


def test_the_sensitivity_switch_recentres_only_the_modern_scale() -> None:
    priors = ModelPriors.from_params({"noise_modern_median": 0.35}, quantity="troops")
    model = build_model(_design(), priors=priors, inflation=_inflation())

    modern = _draw(model, "sigma_type_modern")
    ancient = _draw(model, "sigma_type_ancient")

    assert math.isclose(float(np.median(modern)), 0.35, rel_tol=0.03)
    assert math.isclose(float(np.median(ancient)), 0.40, rel_tol=0.03)


def test_no_ancient_scale_exists_when_no_row_is_ancient() -> None:
    design = _design(years=(1800, None))
    model = build_model(design, priors=ModelPriors(), inflation=_inflation())

    assert "sigma_type_ancient" not in model.named_vars
    assert not design.ancient_rows.any()


def test_the_nu_floor_is_three() -> None:
    model = build_model(_design(), priors=ModelPriors(), inflation=_inflation())

    nu = _draw(model, "nu")

    assert float(nu.min()) >= 3.0
    assert float(np.mean(nu)) == pytest.approx(7.0, abs=0.3)


def test_the_nu_floor_is_read_from_the_spec() -> None:
    priors = ModelPriors.from_params({"student_t_nu_floor": 5.0}, quantity="troops")
    model = build_model(_design(), priors=priors, inflation=_inflation())

    assert float(_draw(model, "nu").min()) >= 5.0


def test_troops_have_no_dispersion_terms_and_casualties_keep_theirs() -> None:
    troops = build_model(_design(), priors=ModelPriors(), inflation=_inflation())
    assert not {"sigma_round", "sigma_estimate"} & set(troops.named_vars)

    priors = ModelPriors.from_params({}, quantity="casualties")
    casualties = build_model(_design("casualties"), priors=priors, inflation=_inflation())
    for name in ("sigma_round", "sigma_estimate"):
        draws = _draw(casualties, name)
        assert float(draws.mean()) == pytest.approx(0.3 * math.sqrt(2 / math.pi), abs=0.012)


def test_casualties_and_the_legacy_switch_keep_the_inverse_gamma() -> None:
    casualties = build_model(_design("casualties"), priors=ModelPriors(), inflation=_inflation())
    legacy = build_model(
        _design(),
        priors=ModelPriors(observation_noise_prior="inverse_gamma"),
        inflation=_inflation(),
    )

    for model in (casualties, legacy):
        assert "sigma_sq_type" in _free(model)
        assert "sigma_type_modern" not in model.named_vars
        assert "sigma_source_ancient" not in model.named_vars


def test_the_ancient_flag_follows_the_noise_cutoff_year() -> None:
    reports = [
        _report(0, side_id=1, year=-200),
        _report(1, side_id=2, year=1200),
        _report(2, side_id=3, year=1800),
        _report(3, side_id=4, year=None),
    ]
    default = build_design(reports, quantity="troops", ancient_cutoff_year=500)
    medieval = build_design(
        reports, quantity="troops", ancient_cutoff_year=500, noise_cutoff_year=1500
    )

    assert list(default.ancient_rows) == [1.0, 0.0, 0.0, 0.0]
    assert list(medieval.ancient_rows) == [1.0, 1.0, 0.0, 0.0]


def test_each_row_uses_its_own_era_scale() -> None:
    import pytensor

    design = _design()
    model = build_model(design, priors=ModelPriors(), inflation=_inflation())
    point = {
        name: value + np.random.default_rng(4).normal(0.0, 0.05, size=np.shape(value))
        for name, value in model.initial_point().items()
    }
    graphs = model.replace_rvs_by_values(
        [model["sigma_source"], model["sigma_source_ancient"], model.row_sd]
    )
    fn = pytensor.function(model.value_vars, graphs, on_unused_input="ignore")
    modern, ancient, row_sd = fn(*[point[v.name] for v in model.value_vars])

    assert float(np.mean(ancient)) > float(np.mean(modern))
    is_ancient = design.ancient_rows.astype(bool)
    assert is_ancient.any() and (~is_ancient).any()
    # No hedged or round figures here, so the row sd is the source scale of its own era.
    keys = design.source_index
    assert row_sd[is_ancient].mean() > row_sd[~is_ancient].mean()
    np.testing.assert_allclose(row_sd[is_ancient], ancient[keys][is_ancient], rtol=0.5)
