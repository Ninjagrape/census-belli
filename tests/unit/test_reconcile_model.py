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
