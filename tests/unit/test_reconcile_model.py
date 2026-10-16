"""Structural checks on the reconcile model that need no sampling.

Building a ``pm.Model`` takes a fraction of a second; sampling one takes
minutes. Anything that can be caught at build time belongs here, not in
``tests/model``.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from pipeline.reconcilers.design import build_design
from pipeline.reconcilers.model import InflationPrior, ModelPriors, build_model
from pipeline.reconcilers.records import Report, SourceKey

pytest.importorskip("pymc")


def _report(
    report_id: int,
    *,
    side_id: int,
    quantity: str,
    source_type: str,
    scope: str = "engaged",
    casualty_type: str = "total",
    regime: str = "unlabelled",
) -> Report:
    """Build a report varied along the axes that become model dimensions."""
    return Report(
        report_id=report_id,
        side_id=side_id,
        battle_id=side_id,
        source_id=report_id,
        source_key=SourceKey(source_type, f"https://example/{source_type}/{report_id % 2}"),
        source_type=source_type,
        quantity=quantity,  # type: ignore[arg-type]
        reported_value=5_000.0 + 100.0 * report_id,
        scope=scope if quantity == "troops" else "unknown",
        casualty_type=casualty_type if quantity == "casualties" else "total",
        year_astronomical=-200 if side_id % 2 else 1800,
        claim_regime=regime,
        lineage_id=report_id,
    )


def _reports(quantity: str) -> list[Report]:
    """A small corpus touching every dimension the model declares."""
    scopes = ("engaged", "available", "unknown")
    casualty_types = ("total", "killed", "wounded")
    regimes = ("modern_scholarly", "chronicle", "unlabelled")
    source_types = ("wikipedia_infobox", "wikidata", "wikipedia_infobox", "dbpedia")
    return [
        _report(
            i,
            side_id=1 + i % 4,
            quantity=quantity,
            source_type=source_types[i % len(source_types)],
            scope=scopes[i % len(scopes)],
            casualty_type=casualty_types[i % len(casualty_types)],
            regime=regimes[i % len(regimes)],
        )
        for i in range(12)
    ]


@pytest.mark.parametrize("quantity", ["troops", "casualties"])
def test_no_model_variable_shares_a_name_with_a_dimension(quantity: str) -> None:
    # ArviZ resolves idata.posterior[name] to the coordinate array when a
    # variable and a dimension share a name, so the samples become unreachable
    # and the summary crashes or, worse, reads coordinate labels. A free RV
    # called "level" beside the scope dimension "level" did exactly that, and
    # a hand-built InferenceData in the summarise tests could never show it.
    design = build_design(_reports(quantity), quantity=quantity, ancient_cutoff_year=500)
    model = build_model(design, priors=ModelPriors(), inflation=InflationPrior())

    collisions = set(model.named_vars) & set(model.coords)

    assert not collisions, f"variables named like dimensions: {sorted(collisions)}"


def test_the_neutral_option_changes_only_the_unlabelled_and_era_priors() -> None:
    import numpy as np

    design = build_design(_reports("troops"), quantity="troops", ancient_cutoff_year=500)
    inflation = InflationPrior(faithful_mu=0.0, faithful_sd=0.2, rhetorical_mu=1.5)

    published = build_model(design, priors=ModelPriors(), inflation=inflation)
    neutral = build_model(design, priors=ModelPriors(), inflation=inflation, neutral=True)

    # Same variables either way: the option is a prior change, not a new model.
    assert set(published.named_vars) == set(neutral.named_vars)

    def prior_params(model: object, name: str) -> tuple[float, float]:
        rv = model.named_vars[name]  # type: ignore[attr-defined]
        mu, sigma = rv.owner.inputs[-2:]
        return float(np.mean(mu.eval())), float(np.mean(sigma.eval()))

    assert prior_params(neutral, "a_era") == (0.0, 0.25)
    free = list(neutral.coords["regime_free"])
    g_mu, g_sd = neutral.named_vars["g_free"].owner.inputs[-2:]
    position = free.index("unlabelled")
    assert g_mu.eval()[position] == 0.0
    assert g_sd.eval()[position] == 0.2
    published_mu = published.named_vars["g_free"].owner.inputs[-2].eval()
    assert published_mu[position] > 0.0


def test_the_neutral_option_is_refused_for_casualties() -> None:
    import pytest

    design = build_design(_reports("casualties"), quantity="casualties", ancient_cutoff_year=500)
    with pytest.raises(ValueError, match="troops only"):
        build_model(design, priors=ModelPriors(), inflation=InflationPrior(), neutral=True)


def _free_names(model: object) -> set[str]:
    return {rv.name for rv in model.free_RVs}  # type: ignore[attr-defined]


def _troop_design() -> object:
    return build_design(_reports("troops"), quantity="troops", ancient_cutoff_year=500)


def test_the_default_family_is_student_t_with_nu_and_no_mixing_weight() -> None:
    model = build_model(_troop_design(), priors=ModelPriors(), inflation=InflationPrior())

    free = _free_names(model)
    assert "nu_raw" in free
    assert "pi_contam" not in free


def test_the_contaminated_family_builds_with_a_mixing_weight_and_no_nu() -> None:
    priors = ModelPriors(likelihood_family="contaminated")

    model = build_model(_troop_design(), priors=priors, inflation=InflationPrior())

    free = _free_names(model)
    assert "pi_contam" in free
    assert not {"nu", "nu_raw"} & free
    assert "obs_point" in model.named_vars


def test_the_student_t_family_still_builds_with_nu_and_no_mixing_weight() -> None:
    priors = ModelPriors(likelihood_family="student_t")

    model = build_model(_troop_design(), priors=priors, inflation=InflationPrior())

    assert "nu_raw" in _free_names(model)
    assert "pi_contam" not in _free_names(model)


def test_the_normal_family_builds_without_either() -> None:
    priors = ModelPriors(likelihood_family="normal")

    model = build_model(_troop_design(), priors=priors, inflation=InflationPrior())

    assert not {"nu_raw", "pi_contam"} & _free_names(model)


def test_contamination_settings_are_read_from_the_spec_params() -> None:
    priors = ModelPriors.from_params(
        {"contamination_pi_alpha": 2.0, "contamination_pi_beta": 5.0, "contamination_sd": 3.0},
        quantity="troops",
    )

    assert (priors.contamination_pi_alpha, priors.contamination_pi_beta) == (2.0, 5.0)
    assert priors.contamination_sd == 3.0
    assert ModelPriors.from_params({}, quantity="troops").likelihood_family == "student_t"


def test_summarises_row_location_and_scale_match_the_compiled_model_at_a_test_point() -> None:
    import numpy as np
    import pytensor

    from pipeline.reconcilers.summarise import _row_eta_and_var

    reports = [
        replace(r, is_estimate=i % 3 == 0, roundness=0.2 * (i % 4), range_var=0.05 * (i % 2))
        for i, r in enumerate(_reports("troops"))
    ]
    design = build_design(reports, quantity="troops", ancient_cutoff_year=500)
    model = build_model(design, priors=ModelPriors(), inflation=InflationPrior())

    rng = np.random.default_rng(1)
    point = {
        name: value + rng.normal(0.3, 0.5, size=np.shape(value))
        for name, value in model.initial_point().items()
    }
    names = [
        n
        for n in (
            "mu_side",
            "beta_source",
            "g_regime",
            "delta_level",
            "a_era",
            "sigma_source",
            "sigma_estimate",
            "sigma_round",
            "sigma_unknown",
        )
        if n in model.named_vars
    ]
    assert {"sigma_unknown", "sigma_estimate", "sigma_round"} <= set(names)
    graphs = model.replace_rvs_by_values([model[n] for n in names] + [model.row_eta, model.row_sd])
    compiled = pytensor.function(model.value_vars, graphs, on_unused_input="ignore")
    outputs = compiled(*[point[v.name] for v in model.value_vars])
    values = {n: np.asarray(v)[None] for n, v in zip(names, outputs[: len(names)], strict=True)}

    eta, var = _row_eta_and_var(lambda n: values[n], design, np.arange(design.n_obs))

    np.testing.assert_allclose(eta[0], outputs[-2], rtol=0, atol=1e-10)
    np.testing.assert_allclose(np.sqrt(var[0]), outputs[-1], rtol=0, atol=1e-10)


def test_the_summarise_rebuild_includes_the_casualty_ratio_offsets() -> None:
    import numpy as np
    import pytensor

    from pipeline.reconcilers.summarise import _row_eta_and_var

    reports = [replace(r, is_estimate=i % 3 == 0) for i, r in enumerate(_reports("casualties"))]
    design = build_design(reports, quantity="casualties", ancient_cutoff_year=500)
    model = build_model(design, priors=ModelPriors(), inflation=InflationPrior())

    rng = np.random.default_rng(2)
    point = {
        name: value + rng.normal(0.3, 0.5, size=np.shape(value))
        for name, value in model.initial_point().items()
    }
    names = [
        n
        for n in (
            "mu_side",
            "beta_source",
            "g_regime",
            "delta_level",
            "a_era",
            "sigma_source",
            "sigma_estimate",
            "sigma_round",
            "sigma_ratio",
            "z_ratio",
        )
        if n in model.named_vars
    ]
    assert {"sigma_ratio", "z_ratio"} <= set(names)
    graphs = model.replace_rvs_by_values([model[n] for n in names] + [model.row_eta, model.row_sd])
    compiled = pytensor.function(model.value_vars, graphs, on_unused_input="ignore")
    outputs = compiled(*[point[v.name] for v in model.value_vars])
    values = {n: np.asarray(v)[None] for n, v in zip(names, outputs[: len(names)], strict=True)}

    eta, var = _row_eta_and_var(lambda n: values[n], design, np.arange(design.n_obs))

    np.testing.assert_allclose(eta[0], outputs[-2], rtol=0, atol=1e-10)
    np.testing.assert_allclose(np.sqrt(var[0]), outputs[-1], rtol=0, atol=1e-10)


def test_the_ratio_variance_applies_to_casualty_components_only() -> None:
    casualties = build_design(
        _reports("casualties"), quantity="casualties", ancient_cutoff_year=500
    )
    troops = build_design(_reports("troops"), quantity="troops", ancient_cutoff_year=500)

    casualty_model = build_model(casualties, priors=ModelPriors(), inflation=InflationPrior())
    troop_model = build_model(troops, priors=ModelPriors(), inflation=InflationPrior())

    assert {"sigma_ratio", "z_ratio"} <= _free_names(casualty_model)
    assert not {"sigma_ratio", "z_ratio"} & _free_names(troop_model)


def test_the_ratio_scale_is_read_from_the_spec_params() -> None:
    priors = ModelPriors.from_params({"casualty_ratio_dispersion_sd": 0.8}, quantity="casualties")

    assert priors.casualty_ratio_dispersion_sd == 0.8
    assert ModelPriors.from_params({}, quantity="casualties").casualty_ratio_dispersion_sd == 0.5


def test_the_variance_prior_has_finite_variance_and_an_unchanged_mode() -> None:
    priors = ModelPriors.from_params({}, quantity="troops")

    assert priors.precision_prior_alpha > 2.0
    mode = priors.precision_prior_beta / (priors.precision_prior_alpha + 1.0)
    assert mode == pytest.approx(1.0 / 3.0)


def _bounded_design() -> object:
    reports = [
        replace(r, is_upper_bound=i % 5 == 1, is_lower_bound=i % 5 == 2)
        for i, r in enumerate(_reports("troops"))
    ]
    return build_design(reports, quantity="troops", ancient_cutoff_year=500)


def test_the_exgaussian_bound_likelihood_is_the_default() -> None:
    assert ModelPriors.from_params({}, quantity="troops").bound_likelihood == "exgaussian"
    model = build_model(_bounded_design(), priors=ModelPriors(), inflation=InflationPrior())

    assert "obs_bound_upper" in model.named_vars
    assert "obs_bound_lower" in model.named_vars
    assert {"lambda_upper", "lambda_lower"} <= _free_names(model)


def test_the_censored_bound_likelihood_stays_available() -> None:
    priors = ModelPriors.from_params({"bound_likelihood": "censored"}, quantity="troops")
    model = build_model(_bounded_design(), priors=priors, inflation=InflationPrior())

    assert "obs_censor_lower" in model.named_vars
    assert "obs_censor_upper" in model.named_vars
    assert not {"lambda_upper", "lambda_lower"} & _free_names(model)


def _bound_logp_at_offsets(
    model: object, name: str, offsets: tuple[float, float]
) -> tuple[float, float]:
    """Log density of one bound term with the report placed at ``eta + offset``."""
    import numpy as np
    import pymc as pm
    import pytensor

    point = model.initial_point()  # type: ignore[attr-defined]
    graph = model.replace_rvs_by_values([model.row_eta])  # type: ignore[attr-defined]
    eta = np.asarray(
        pytensor.function(model.value_vars, graph, on_unused_input="ignore")(  # type: ignore[attr-defined]
            *[point[v.name] for v in model.value_vars]  # type: ignore[attr-defined]
        )[0]
    )
    logp_fn = model.compile_logp(vars=[model[name]])  # type: ignore[attr-defined]
    results = []
    for offset in offsets:
        pm.set_data({"data_log_y": eta + offset}, model=model)  # type: ignore[arg-type]
        results.append(float(logp_fn(point)))
    return results[0], results[1]


def test_an_upper_bound_prefers_the_truth_below_the_figure_and_a_lower_bound_above() -> None:
    # "Up to X": the figure sits above the level, so log y - eta > 0 is the likely
    # side. "At least X": the opposite. A flipped sign would still sample and give
    # a plausible-looking fit, so the direction is pinned on the density itself.
    model = build_model(_bounded_design(), priors=ModelPriors(), inflation=InflationPrior())

    upper_below, upper_above = _bound_logp_at_offsets(model, "obs_bound_upper", (0.5, -0.5))
    lower_above, lower_below = _bound_logp_at_offsets(model, "obs_bound_lower", (-0.5, 0.5))

    # Offsets are log y - eta: eta below log X is +0.5 for an upper bound.
    assert upper_below > upper_above
    # eta above log X (offset -0.5) is the favoured side of a lower bound.
    assert lower_above > lower_below


def test_the_exgaussian_bound_likelihood_replaces_the_censoring() -> None:
    import numpy as np

    model = build_model(_bounded_design(), priors=ModelPriors(), inflation=InflationPrior())

    assert {"lambda_upper", "lambda_lower"} <= _free_names(model)
    assert "obs_bound_upper" in model.named_vars
    assert "obs_bound_lower" in model.named_vars
    assert "obs_censor_lower" not in model.named_vars
    assert "obs_censor_upper" not in model.named_vars
    assert np.isfinite(model.compile_logp()(model.initial_point()))


def test_an_unknown_bound_likelihood_is_refused() -> None:
    priors = ModelPriors(bound_likelihood="gaussian")

    with pytest.raises(ValueError, match="bound_likelihood"):
        build_model(_bounded_design(), priors=priors, inflation=InflationPrior())


def test_a_ratio_offset_exists_only_for_side_level_pairs_that_occur() -> None:
    from pipeline.reconcilers.model import _ratio_level_mask, _ratio_pairs

    design = build_design(_reports("casualties"), quantity="casualties", ancient_cutoff_year=500)

    pair_of_row, labels = _ratio_pairs(design)
    ratio_rows = _ratio_level_mask(design) > 0

    assert len(labels) == len(set(labels)) > 0
    assert len(set(pair_of_row[ratio_rows].tolist())) == len(labels)
    assert all(not label.endswith(":total") for label in labels)
    troops = build_design(_reports("troops"), quantity="troops", ancient_cutoff_year=500)
    assert _ratio_pairs(troops)[1] == []
