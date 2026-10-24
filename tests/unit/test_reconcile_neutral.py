"""Tests for the neutral fit's summary and the draws file the impute stage reads.

The posterior is built by hand with ``az.from_dict``; nothing here samples.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from pipeline.reconcilers.design import build_design
from pipeline.reconcilers.neutral import (
    MAX_NEUTRAL_DRAWS,
    neutral_draws_path,
    summarise_neutral,
    write_neutral_draws,
)
from pipeline.reconcilers.records import Report, SourceKey


def _report(
    report_id: int,
    side_id: int,
    battle_id: int,
    regime: str = "modern_scholarly",
    year: int = 1800,
) -> Report:
    return Report(
        report_id=report_id,
        side_id=side_id,
        battle_id=battle_id,
        source_id=report_id,
        source_key=SourceKey.from_row("peer_reviewed", f"https://example/{report_id}"),
        source_type="peer_reviewed",
        quantity="troops",
        reported_value=10_000.0 * report_id,
        scope="engaged",
        claim_regime=regime,
        year_astronomical=year,
    )


def _summary(n_draws: int = 1500):  # type: ignore[no-untyped-def]
    import arviz as az

    # Side ids deliberately out of numeric order, so a positional lookup
    # instead of a label lookup would scramble the columns.
    reports = [_report(1, 30, 3), _report(2, 10, 1), _report(3, 20, 1)]
    design = build_design(reports, quantity="troops", ancient_cutoff_year=500)
    rng = np.random.default_rng(0)
    means = {side: 8.0 + i for i, side in enumerate(design.side_ids)}
    mu = np.stack(
        [rng.normal(means[s], 0.2, size=(2, n_draws // 2)) for s in design.side_ids], axis=-1
    )
    posterior = {
        "mu_side": mu,
        "m0": rng.normal(9.0, 0.1, size=(2, n_draws // 2)),
        "s0": np.abs(rng.normal(1.2, 0.05, size=(2, n_draws // 2))),
    }
    idata = az.from_dict(
        posterior=posterior,
        coords={"side": [str(s) for s in design.side_ids]},
        dims={"mu_side": ["side"]},
    )
    battles = {30: 3, 10: 1, 20: 1}
    return design, means, summarise_neutral(idata, design, battles, run_id=7, ci_mass=0.95)


def test_estimates_are_the_log_scale_posterior_summaries() -> None:
    design, means, summary = _summary()

    assert [e.side_id for e in summary.estimates] == list(design.side_ids)
    for estimate in summary.estimates:
        assert math.isclose(estimate.log_mean, means[estimate.side_id], abs_tol=0.05)
        assert math.isclose(estimate.log_sd, 0.2, abs_tol=0.03)
        assert estimate.lo < estimate.value < estimate.hi
        assert estimate.run_id == 7


def test_the_diagnostics_block_carries_worst_and_the_hyperparameter_means() -> None:
    _, _, summary = _summary()
    block = summary.diagnostics

    assert set(block["worst"]) == {
        "max_rhat",
        "min_ess_bulk",
        "min_ess_bulk_side",
        "divergences",
        "n_sides_mcse_fail",
        "mcse_failed_side_ids",
        "mcse_failed_limits",
        "worst_side",
    }
    assert math.isclose(block["m0"]["mean"], 9.0, abs_tol=0.05)
    assert math.isclose(block["s0"]["mean"], 1.2, abs_tol=0.02)


def test_draws_are_thinned_to_the_cap_and_aligned_with_side_ids() -> None:
    design, means, summary = _summary(n_draws=4000)

    assert summary.draws.shape == (MAX_NEUTRAL_DRAWS, design.n_sides)
    assert summary.draws.dtype == np.float64
    column_means = summary.draws.mean(axis=0)
    for position, side_id in enumerate(summary.side_ids):
        assert math.isclose(float(column_means[position]), means[side_id], abs_tol=0.05)


def test_the_npz_has_the_documented_arrays_and_dtypes(tmp_path: Path) -> None:
    _, _, summary = _summary()

    path = write_neutral_draws(
        summary,
        processed_root=tmp_path,
        run_id=7,
        database="db_a",
        started_at="2026-01-01T00:00:00+00:00",
    )

    assert path == neutral_draws_path(tmp_path, 7)
    assert path.name == "neutral_draws_run7.npz"
    with np.load(path) as data:
        assert set(data.files) == {
            "run_id",
            "database",
            "started_at",
            "side_id",
            "battle_id",
            "draws",
            "m0_mean",
            "s0_mean",
            "side_era",
            "side_unlabelled_share",
        }
        assert int(data["run_id"]) == 7
        assert str(data["database"]) == "db_a"
        assert str(data["started_at"]) == "2026-01-01T00:00:00+00:00"
        assert data["side_id"].dtype == np.int64
        assert data["battle_id"].dtype == np.int64
        assert data["draws"].dtype == np.float64
        assert data["draws"].shape == (summary.draws.shape[0], 3)
        assert list(data["side_id"]) == list(summary.side_ids)
        assert list(data["battle_id"]) == [{30: 3, 10: 1, 20: 1}[s] for s in summary.side_ids]
        assert math.isclose(float(data["m0_mean"]), 9.0, abs_tol=0.05)
    assert not list(tmp_path.rglob("*.tmp.npz"))


def test_neutral_estimates_are_stamped_with_their_own_run_id() -> None:
    from datetime import UTC, datetime
    from types import SimpleNamespace

    from pipeline.reconcilers.records import SideEstimate
    from pipeline.reconcilers.store import write_neutral_estimates, write_side_estimates

    class _Conn:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        def execute(self, statement: object, params: dict[str, object]) -> SimpleNamespace:
            self.calls.append((str(statement), params))
            return SimpleNamespace(rowcount=0)

    _, _, summary = _summary()
    conn = _Conn()
    assert write_neutral_estimates(conn, summary.estimates) == 3
    assert all("est_troops_neutral_run_id" in sql for sql, _ in conn.calls)
    assert all("est_troops_run_id" not in sql for sql, _ in conn.calls)
    assert {params["run_id"] for _, params in conn.calls} == {7}

    # The published write must not touch the neutral family, so a run where
    # only one of the two passes its thresholds leaves independent provenance.
    published = SideEstimate(
        side_id=10,
        quantity="troops",
        value=1.0,
        lo=0.5,
        hi=2.0,
        run_id=9,
        n_reports=1,
        n_sources=1,
        method="single_report_debiased",
        updated_at=datetime(2026, 10, 6, tzinfo=UTC),
    )
    other = _Conn()
    write_side_estimates(other, [published])
    assert "neutral" not in other.calls[0][0]


SIGMA = 0.25
NU = 5.0


def _shared_summary(n_draws: int = 4000):  # type: ignore[no-untyped-def]
    """A neutral summary whose shared-parameter draws are exact multiples of a row index.

    Side 20 carries one unlabelled and one modern report, side 30 only an
    unlabelled one dated 300 BC, side 10 only a modern one. The unlabelled rows are
    dated before 1800 (side 20's in 1700) because from 1800 on the level anchor
    removes both shared offsets from them.
    """
    import arviz as az

    reports = [
        _report(1, 30, 3, "unlabelled", year=-300),
        _report(2, 10, 1),
        _report(3, 20, 1, "unlabelled", year=1700),
        _report(4, 20, 1),
    ]
    design = build_design(reports, quantity="troops", ancient_cutoff_year=500)
    chains, per_chain = 2, n_draws // 2
    index = (np.arange(n_draws, dtype=np.float64) / 1000.0).reshape(chains, per_chain)
    regimes = design.coords()["regime"]
    g_regime = np.zeros((chains, per_chain, len(regimes)))
    g_regime[:, :, regimes.index("unlabelled")] = 2.0 * index
    posterior = {
        "mu_side": np.repeat(index[:, :, None], design.n_sides, axis=2),
        # m0 = corpus_level - offset_bar, so it falls as offset_bar rises.
        "m0": 9.0 - 4.0 * index,
        "s0": np.full((chains, per_chain), 1.0),
        "g_regime": g_regime,
        "a_era": 3.0 * index,
        "offset_bar": 4.0 * index,
        # The pieces the exported likelihood rebuilds each row's offset and noise from.
        "beta_source": np.zeros((chains, per_chain, design.n_sources)),
        "delta_level": np.zeros((chains, per_chain, len(design.levels))),
        "sigma_source": np.full((chains, per_chain, design.n_sources), SIGMA),
        "nu": np.full((chains, per_chain), NU),
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
    battles = {30: 3, 10: 1, 20: 1}
    return design, summarise_neutral(idata, design, battles, run_id=9, ci_mass=0.95)


def test_shared_regime_draws_are_row_aligned_with_the_mu_draws() -> None:
    _, summary = _shared_summary()

    assert summary.g_unlabelled is not None
    assert summary.a_era is not None
    assert summary.offset_bar is not None
    assert summary.m0_draws is not None
    assert summary.draws.shape[0] == MAX_NEUTRAL_DRAWS
    for shared, factor in (
        (summary.g_unlabelled, 2.0),
        (summary.a_era, 3.0),
        (summary.offset_bar, 4.0),
    ):
        assert shared.shape == (MAX_NEUTRAL_DRAWS,)
        np.testing.assert_allclose(shared, factor * summary.draws[:, 0])
    assert summary.m0_draws.shape == (MAX_NEUTRAL_DRAWS,)
    np.testing.assert_allclose(summary.m0_draws, 9.0 - 4.0 * summary.draws[:, 0])


def test_per_side_era_and_unlabelled_share_follow_the_design_rows() -> None:
    design, summary = _shared_summary()

    assert summary.side_era is not None
    assert summary.side_unlabelled_share is not None
    share = dict(zip(summary.side_ids, summary.side_unlabelled_share, strict=True))
    era = dict(zip(summary.side_ids, summary.side_era, strict=True))
    assert share == {10: 0.0, 20: 0.5, 30: 1.0}
    assert era[10] == 0.0
    unlabelled_rows = design.regime_index == design.regimes.index("unlabelled")
    side_30 = design.side_index == design.side_ids.index(30)
    assert era[30] == pytest.approx(float(design.era[unlabelled_rows & side_30][0]))


def test_the_shared_arrays_round_trip_through_the_file(tmp_path: Path) -> None:
    from pipeline.imputers.likelihood import load_neutral_draws

    _, summary = _shared_summary()
    path = write_neutral_draws(
        summary, processed_root=tmp_path, run_id=9, database="db_a", started_at="t0"
    )

    with np.load(path) as data:
        assert data["g_unlabelled"].shape == (MAX_NEUTRAL_DRAWS,)
        assert data["a_era"].shape == (MAX_NEUTRAL_DRAWS,)
        assert data["offset_bar"].shape == (MAX_NEUTRAL_DRAWS,)
        assert data["m0_draws"].shape == (MAX_NEUTRAL_DRAWS,)
        assert data["side_era"].shape == (3,)
        assert data["side_unlabelled_share"].shape == (3,)
    loaded = load_neutral_draws(path)
    assert loaded.database == "db_a"
    names = ("g_unlabelled", "a_era", "offset_bar", "m0_draws", "side_era", "side_unlabelled_share")
    for name in names:
        np.testing.assert_array_equal(getattr(loaded, name), getattr(summary, name))
