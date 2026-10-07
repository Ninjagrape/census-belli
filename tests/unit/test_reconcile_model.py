"""Structural checks on the reconcile model that need no sampling.

Building a ``pm.Model`` takes a fraction of a second; sampling one takes
minutes. Anything that can be caught at build time belongs here, not in
``tests/model``.
"""

from __future__ import annotations

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
