"""The manual_entry source type's bias is pinned to exactly zero.

A ``troop_strength`` override writes ``manual_entry`` rows that supersede every other
total on their side, so the type never co-reports and its bias would otherwise be
prior-only or learned from the very battles the overrides fix.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.reconcilers.design import Design, build_design
from pipeline.reconcilers.model import (
    PINNED_BIAS_SOURCE_TYPE,
    InflationPrior,
    ModelPriors,
    build_model,
)
from tests.unit.test_reconcile_p1_model import _eval, _free_names, _report

pytest.importorskip("pymc")


def _design(with_manual: bool) -> Design:
    types = ["wikipedia_infobox", "wikidata", "wikipedia_infobox", "wikidata"]
    if with_manual:
        types += [PINNED_BIAS_SOURCE_TYPE, PINNED_BIAS_SOURCE_TYPE]
    reports = [
        _report(i + 1, side_id=i + 1, battle_id=(i // 2) + 1, year=1700 + 10 * i, source_type=t)
        for i, t in enumerate(types)
    ]
    return build_design(reports, quantity="troops", ancient_cutoff_year=500)


def test_the_manual_entry_bias_is_a_constant_zero_not_a_free_variable() -> None:
    design = _design(with_manual=True)
    model = build_model(design, priors=ModelPriors(), inflation=InflationPrior())
    rng = np.random.default_rng(1)
    point = {
        k: v + rng.normal(0.0, 1.0, size=np.shape(v)) for k, v in model.initial_point().items()
    }

    (b_type, beta_source) = _eval(model, [model["b_type"], model["beta_source"]], point)

    position = design.source_types.index(PINNED_BIAS_SOURCE_TYPE)
    assert "b_type" not in _free_names(model)
    assert b_type[position] == 0.0
    assert float(np.delete(b_type, position).sum()) == pytest.approx(0.0, abs=1e-10)
    assert (np.delete(b_type, position) != 0.0).all()
    for i, key in enumerate(design.source_keys):
        if key.source_type == PINNED_BIAS_SOURCE_TYPE:
            assert beta_source[i] == 0.0


def test_the_manual_entry_type_gets_no_per_source_effect() -> None:
    design = _design(with_manual=True)
    model = build_model(design, priors=ModelPriors(), inflation=InflationPrior())

    assert not any(
        PINNED_BIAS_SOURCE_TYPE in name for name in _free_names(model) if name != "b_type"
    )


def test_without_manual_entry_the_model_keeps_its_free_bias_variable() -> None:
    model = build_model(
        _design(with_manual=False), priors=ModelPriors(), inflation=InflationPrior()
    )

    free = _free_names(model)
    assert "b_type" in free
    assert not any("b_type_" in name for name in free)
