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
