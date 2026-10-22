"""The per-side likelihood that reconcile's neutral fit exports for impute.

Posteriors are built by hand; nothing samples.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from pipeline.reconcilers.design import build_design
from pipeline.reconcilers.likelihood_export import (
    combine_rows,
    fisher_information,
    regress_loadings,
)
from pipeline.reconcilers.neutral import summarise_neutral, write_neutral_draws
from pipeline.reconcilers.records import Report
from tests.unit.test_reconcile_neutral import NU, SIGMA, _report, _shared_summary


def test_the_fisher_information_of_a_student_t_location_is_the_textbook_value() -> None:
    sd = np.array([0.2, 0.5])

    info = fisher_information(sd, nu=5.0)

    np.testing.assert_allclose(info, (5.0 + 1.0) / ((5.0 + 3.0) * sd**2))
    # A Normal row carries more information than a t row of the same scale.
    assert np.all(fisher_information(sd, None) > info)


def test_rows_combine_by_precision() -> None:
    adjusted = np.array([10.0, 12.0, 7.0])
    info = np.array([1.0, 3.0, 2.0])
    side_of_row = np.array([0, 0, 1])

    mean, precision, n_rows = combine_rows(adjusted, info, side_of_row, n_sides=3)

    assert mean[0] == pytest.approx((10.0 * 1 + 12.0 * 3) / 4)
    assert mean[1] == pytest.approx(7.0)
    assert math.isnan(mean[2])
    np.testing.assert_allclose(precision, [4.0, 2.0, 0.0])
    np.testing.assert_array_equal(n_rows, [2, 1, 0])


def test_rows_combine_per_draw() -> None:
    adjusted = np.array([[1.0, 3.0], [2.0, 4.0]])
    info = np.array([1.0, 1.0])

    mean, _, _ = combine_rows(adjusted, info, np.array([0, 0]), n_sides=1)

    np.testing.assert_allclose(mean[:, 0], [2.0, 3.0])


def test_regress_loadings_recovers_known_slopes_and_drops_flat_parameters() -> None:
    rng = np.random.default_rng(0)
    n = 500
    g = rng.normal(0.0, 1.0, n)
    a = rng.normal(0.0, 0.5, n)
    flat = np.full(n, 3.0)
    side_a = 9.0 - 0.8 * g + 0.1 * rng.normal(size=n)
    side_b = 7.0 - 0.5 * a + 0.1 * rng.normal(size=n)
    missing = np.full(n, np.nan)
    means = np.column_stack([side_a, side_b, missing])

    loadings, names = regress_loadings(
        means, np.column_stack([g, flat, a]), ("g_unlabelled", "offset_bar", "a_era")
    )

    assert names == ("g_unlabelled", "a_era")
    assert loadings.shape == (3, 2)
    assert loadings[0, 0] == pytest.approx(-0.8, abs=0.02)
    assert loadings[1, 1] == pytest.approx(-0.5, abs=0.03)
    assert not loadings[2].any()


def test_a_run_with_no_varying_parameter_exports_no_loadings() -> None:
    loadings, names = regress_loadings(
        np.zeros((50, 2)), np.ones((50, 3)), ("g_unlabelled", "a_era", "offset_bar")
    )

    assert names == ()
    assert loadings.shape == (2, 0)


def test_the_one_row_side_mean_is_the_adjusted_report_on_an_unlabelled_row() -> None:
    design, summary = _shared_summary()
    lik = summary.likelihood
    assert lik is not None
    side = list(summary.side_ids).index(30)
    row = int(np.flatnonzero(design.side_index == design.side_ids.index(30))[0])
    g_mean = 2.0 * float(np.mean(np.arange(4000) / 1000.0))
    a_mean = 3.0 * float(np.mean(np.arange(4000) / 1000.0))
    expected = float(design.log_y[row]) - g_mean - a_mean * float(design.era[row])
    variance = SIGMA**2 + 0.1**2 * float(design.roundness[row])

    assert lik.mean[side] == pytest.approx(expected)
    assert lik.sd[side] == pytest.approx(math.sqrt(variance * (NU + 3.0) / (NU + 1.0)))


def test_the_one_row_side_with_a_modern_report_is_its_report_at_the_fisher_sd() -> None:
    design, summary = _shared_summary()
    lik = summary.likelihood
    assert lik is not None
    side = list(summary.side_ids).index(10)
    row = int(np.flatnonzero(design.side_index == design.side_ids.index(10))[0])
    variance = SIGMA**2 + 0.1**2 * float(design.roundness[row])

    assert lik.has_point[side]
    assert lik.n_point_rows[side] == 1
    assert lik.mean[side] == pytest.approx(float(design.log_y[row]))
    assert lik.sd[side] == pytest.approx(math.sqrt(variance * (NU + 3.0) / (NU + 1.0)))


def test_two_rows_combine_by_precision_with_the_unlabelled_offset_removed() -> None:
    design, summary = _shared_summary()
    lik = summary.likelihood
    assert lik is not None
    side = list(summary.side_ids).index(20)
    rows = np.flatnonzero(design.side_index == design.side_ids.index(20))
    unlabelled = design.regime_index == design.regimes.index("unlabelled")
    index_mean = float(np.mean(np.arange(4000) / 1000.0))
    adjusted = [
        float(design.log_y[r])
        - (2.0 * index_mean + 3.0 * index_mean * float(design.era[r]) if unlabelled[r] else 0.0)
        for r in rows
    ]
    variance = np.array([SIGMA**2 + 0.1**2 * float(design.roundness[r]) for r in rows])
    info = (NU + 1.0) / (NU + 3.0) / variance

    assert lik.n_point_rows[side] == 2
    assert lik.mean[side] == pytest.approx(float(np.sum(info * adjusted) / np.sum(info)))
    assert lik.sd[side] == pytest.approx(float(1.0 / np.sqrt(np.sum(info))))


def test_the_loadings_reproduce_the_shift_a_shared_draw_makes_to_a_sides_mean() -> None:
    _, summary = _shared_summary()
    lik = summary.likelihood
    assert lik is not None
    assert summary.g_unlabelled is not None
    assert summary.a_era is not None
    assert summary.offset_bar is not None
    columns = {
        "g_unlabelled": summary.g_unlabelled,
        "a_era": summary.a_era,
        "offset_bar": summary.offset_bar,
    }
    # The three draws are exactly collinear in this fixture, so individual
    # loadings are not unique; the shift they imply is.
    theta = np.column_stack([columns[n] for n in lik.loading_names])
    centred = theta - theta.mean(axis=0)
    side = list(summary.side_ids).index(30)
    assert summary.side_era is not None
    index = summary.draws[:, 0]
    expected = -(2.0 + 3.0 * float(summary.side_era[side])) * (index - index.mean())

    np.testing.assert_allclose(centred @ lik.loadings[side], expected, atol=1e-8)
    modern = list(summary.side_ids).index(10)
    np.testing.assert_allclose(centred @ lik.loadings[modern], 0.0, atol=1e-8)


def test_the_likelihood_arrays_are_written_and_read_back(tmp_path: Path) -> None:
    from pipeline.imputers.likelihood import load_neutral_draws

    _, summary = _shared_summary()
    path = write_neutral_draws(
        summary, processed_root=tmp_path, run_id=9, database="db_a", started_at="t0"
    )

    with np.load(path) as data:
        for key in ("run_id", "database", "started_at", "side_id", "battle_id", "draws", "m0_mean"):
            assert key in data.files
        required = {"lik_mean", "lik_sd", "lik_has_point", "lik_loadings", "lik_loading_names"}
        assert required <= set(data.files)
        assert data["lik_has_point"].dtype == bool
        assert data["lik_mean"].shape == (3,)
    loaded = load_neutral_draws(path)
    assert loaded.lik_mean is not None
    assert summary.likelihood is not None
    np.testing.assert_array_equal(loaded.lik_mean, summary.likelihood.mean)
    assert loaded.lik_loading_names == summary.likelihood.loading_names


def test_a_bound_only_side_has_no_point_likelihood() -> None:
    import arviz as az

    first = _report(1, 10, 1)
    second = Report(**{**_report(2, 20, 1).__dict__, "is_upper_bound": True})
    design = build_design([first, second], quantity="troops", ancient_cutoff_year=500)
    assert design.point_rows.size == 1
    chains, per_chain = 2, 200
    index = np.arange(chains * per_chain, dtype=np.float64).reshape(chains, per_chain) / 100.0
    regimes = design.coords()["regime"]
    posterior = {
        "mu_side": np.repeat(index[:, :, None], design.n_sides, axis=2),
        "m0": 9.0 - index,
        "s0": np.full((chains, per_chain), 1.0),
        "g_regime": np.zeros((chains, per_chain, len(regimes))),
        "a_era": index,
        "offset_bar": index,
        "beta_source": np.zeros((chains, per_chain, design.n_sources)),
        "delta_level": np.zeros((chains, per_chain, len(design.levels))),
        "sigma_source": np.full((chains, per_chain, design.n_sources), SIGMA),
        "sigma_estimate": np.full((chains, per_chain), 0.1),
        "sigma_round": np.full((chains, per_chain), 0.1),
    }
    idata = az.from_dict(
        posterior=posterior,
        coords={
            "side": [str(s) for s in design.side_ids],
            "regime": regimes,
            "source_key": design.coords()["source_key"],
            "level": list(design.levels),
        },
        dims={
            "mu_side": ["side"],
            "g_regime": ["regime"],
            "beta_source": ["source_key"],
            "delta_level": ["level"],
            "sigma_source": ["source_key"],
        },
    )

    summary = summarise_neutral(idata, design, {10: 1, 20: 1}, run_id=3, ci_mass=0.95)

    lik = summary.likelihood
    assert lik is not None
    assert dict(zip(summary.side_ids, lik.has_point, strict=True)) == {10: True, 20: False}
    bound = list(summary.side_ids).index(20)
    assert math.isnan(lik.mean[bound])
    assert math.isnan(lik.sd[bound])
    assert not lik.loadings[bound].any()
