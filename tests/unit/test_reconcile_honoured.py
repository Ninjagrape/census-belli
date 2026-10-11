"""reports_honoured and the NULL bias of pooled keys, on hand-built posteriors."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from pipeline.reconcilers.design import Design, build_design
from pipeline.reconcilers.records import Report, SourceKey
from pipeline.reconcilers.summarise import _source_biases, reports_honoured

az = pytest.importorskip("arviz")


def _report(report_id: int, side_id: int, value: float, *, url: str = "https://x/a") -> Report:
    return Report(
        report_id=report_id,
        side_id=side_id,
        battle_id=side_id,
        source_id=report_id,
        source_key=SourceKey("wikipedia_infobox", url),
        source_type="wikipedia_infobox",
        quantity="troops",
        reported_value=value,
        year_astronomical=1800,
        claim_regime="modern_scholarly",
        lineage_id=report_id,
    )


def _idata(design: Design, mu_by_side: dict[int, float], *, sd: float = 0.1) -> Any:
    rng = np.random.default_rng(0)
    n_chains, n_draws = 2, 200
    mu = np.stack(
        [mu_by_side[s] + sd * rng.normal(size=(n_chains, n_draws)) for s in design.side_ids],
        axis=-1,
    )
    zeros = lambda *shape: np.zeros((n_chains, n_draws, *shape))  # noqa: E731
    return az.from_dict(
        posterior={
            "mu_side": mu,
            "beta_source": zeros(design.n_sources),
            "sigma_source": zeros(design.n_sources) + 0.3,
            "g_regime": zeros(len(design.regimes)),
            "delta_level": zeros(len(design.levels)),
            "a_era": zeros(),
        },
        coords=design.coords(),
        dims={
            "mu_side": ["side"],
            "beta_source": ["source_key"],
            "sigma_source": ["source_key"],
            "g_regime": ["regime"],
            "delta_level": ["level"],
        },
    )


def _design(reports: list[Report]) -> Design:
    return build_design(reports, quantity="troops", ancient_cutoff_year=500)


def test_a_side_pulled_away_from_agreeing_reports_is_a_violation() -> None:
    reports = [_report(1, 1, 1_000_000.0), _report(2, 1, 1_200_000.0), _report(3, 2, 5_000.0)]
    design = _design(reports)
    # Side 1's reports agree near 1.1M but the fit sits at 35,000.
    idata = _idata(design, {1: math.log(35_000.0), 2: math.log(5_000.0)})

    result = reports_honoured(idata, design)

    assert result["n_checked"] == 1
    assert result["violating_side_ids"] == [1]
    assert result["n_violations"] == 1


def test_a_side_whose_interval_contains_one_report_is_honoured() -> None:
    reports = [_report(1, 1, 1_000_000.0), _report(2, 1, 1_200_000.0)]
    design = _design(reports)
    idata = _idata(design, {1: math.log(1_100_000.0)}, sd=0.3)

    result = reports_honoured(idata, design)

    assert result["n_checked"] == 1
    assert result["n_violations"] == 0


def test_sides_whose_reports_disagree_by_more_than_a_factor_of_two_are_not_checked() -> None:
    reports = [_report(1, 1, 1_000.0), _report(2, 1, 50_000.0), _report(3, 2, 5_000.0)]
    design = _design(reports)
    idata = _idata(design, {1: math.log(100.0), 2: math.log(5_000.0)})

    result = reports_honoured(idata, design)

    assert result["n_checked"] == 0
    assert result["n_violations"] == 0
    assert result["violating_side_ids"] == []


def test_the_within_side_mad_is_reported() -> None:
    reports = [_report(1, 1, 1_000.0), _report(2, 1, 2_000.0)]
    design = _design(reports)
    idata = _idata(design, {1: math.log(1_400.0)})

    result = reports_honoured(idata, design)

    assert result["within_side_mad"] == pytest.approx(math.log(2.0) / 2.0)


def test_a_pooled_key_in_a_single_type_corpus_has_no_bias_to_report() -> None:
    reports = [_report(1, 1, 1_000.0, url="https://x/a"), _report(2, 2, 2_000.0, url="https://x/b")]
    design = _design(reports)
    idata = _idata(design, {1: 7.0, 2: 7.6})

    biases = _source_biases(idata, design, reports, run_id=1, computed_at=None)  # type: ignore[arg-type]

    assert len(biases) == 2
    for bias in biases:
        assert bias.bias_mu is None and bias.bias_sd is None
        # The noise scale is the type-level one and is kept.
        assert bias.sigma_mu == pytest.approx(0.3)
        assert bias.n_reports == 1
