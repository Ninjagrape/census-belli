"""The prior predictive check: its output block, determinism and isolation from the fit."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from pipeline.reconcilers.design import build_design
from pipeline.reconcilers.model import InflationPrior, ModelPriors, build_model
from pipeline.reconcilers.prior_check import prior_predictive_check
from pipeline.reconcilers.records import Report, SourceKey

pytest.importorskip("pymc")


def _design(quantity: str = "troops"):  # type: ignore[no-untyped-def]
    reports = [
        Report(
            report_id=i,
            side_id=1 + i % 3,
            battle_id=1 + i % 3,
            source_id=i,
            source_key=SourceKey("wikipedia_infobox", f"https://example/{i}"),
            source_type="wikipedia_infobox",
            quantity=quantity,  # type: ignore[arg-type]
            reported_value=10_000.0 + 500.0 * i,
            scope="engaged" if quantity == "troops" else "unknown",
            year_astronomical=1800,
            claim_regime="unlabelled",
            lineage_id=i,
        )
        for i in range(1, 7)
    ]
    return build_design(reports, quantity=quantity, ancient_cutoff_year=500)


def _model(design):  # type: ignore[no-untyped-def]
    return build_model(design, priors=ModelPriors(), inflation=InflationPrior())


def test_block_reports_quantiles_and_the_outside_share() -> None:
    design = _design()

    block = prior_predictive_check(_model(design), design, draws=200, seed=1)

    quantiles = block["strength_quantiles"]
    assert 0 < quantiles["p01"] < quantiles["p50"] < quantiles["p99"]
    assert 0.0 <= block["share_outside"] <= 1.0
    assert block["draws"] == 200
    assert block["bounds"] == [50.0, 5_000_000.0]
    json.dumps(block)


def test_same_seed_reproduces_the_block_exactly() -> None:
    design = _design()

    first = prior_predictive_check(_model(design), design, draws=200, seed=7)
    second = prior_predictive_check(_model(design), design, draws=200, seed=7)

    assert first == second


def test_the_check_does_not_advance_the_numpy_global_generator() -> None:
    design = _design()
    np.random.seed(123)
    expected = np.random.random()
    np.random.seed(123)

    prior_predictive_check(_model(design), design, draws=100, seed=7)

    assert np.random.random() == expected


def test_tighter_bounds_raise_the_outside_share() -> None:
    design = _design()
    model = _model(design)

    wide = prior_predictive_check(model, design, draws=200, seed=3)
    narrow = prior_predictive_check(
        model, design, draws=200, seed=3, troop_bounds=(9_000.0, 11_000.0)
    )

    assert narrow["share_outside"] > wide["share_outside"]
    assert math.isfinite(narrow["share_outside"])


def test_draws_zero_skips_and_casualties_have_no_share() -> None:
    design = _design()
    assert prior_predictive_check(_model(design), design, draws=0) == {"skipped": "disabled"}

    cas = _design("casualties")
    block = prior_predictive_check(_model(cas), cas, draws=100, seed=1)
    assert block["share_outside"] is None
    assert "strength_quantiles" in block


def test_a_sampler_exception_is_recorded_as_failed_not_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pymc as pm

    def boom(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("no prior for you")

    monkeypatch.setattr(pm, "sample_prior_predictive", boom)
    design = _design()

    block = prior_predictive_check(_model(design), design, draws=50, seed=1)

    assert block["status"] == "failed"
    assert "no prior for you" in block["error"]
    assert "skipped" not in block
    json.dumps(block)


def test_the_warning_gates_fail_on_failed_and_pass_on_disabled() -> None:
    import yaml

    spec = yaml.safe_load(Path("agents/reconcile.yaml").read_text(encoding="utf-8"))
    gates = {g["name"]: g for g in spec["quality_checks"]}
    for name in ("prior_predictive_is_plausible", "prior_predictive_neutral_is_plausible"):
        sql = gates[name]["check"]
        assert "'failed' THEN 1::float" in sql
        assert "'skipped' IS NOT NULL THEN 0::float" in sql
        assert gates[name]["severity"] == "warning"
    assert "prior_predictive_neutral" in gates["prior_predictive_neutral_is_plausible"]["check"]
