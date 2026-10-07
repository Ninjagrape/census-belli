"""Tests for the neutral fit's summary and the draws file the impute stage reads.

The posterior is built by hand with ``az.from_dict``; nothing here samples.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from pipeline.reconcilers.design import build_design
from pipeline.reconcilers.neutral import (
    MAX_NEUTRAL_DRAWS,
    neutral_draws_path,
    summarise_neutral,
    write_neutral_draws,
)
from pipeline.reconcilers.records import Report, SourceKey


def _report(report_id: int, side_id: int, battle_id: int) -> Report:
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
        claim_regime="modern_scholarly",
        year_astronomical=1800,
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

    assert set(block["worst"]) == {"max_rhat", "min_ess_bulk", "divergences"}
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

    path = write_neutral_draws(summary, processed_root=tmp_path, run_id=7)

    assert path == neutral_draws_path(tmp_path, 7)
    assert path.name == "neutral_draws_run7.npz"
    with np.load(path) as data:
        assert set(data.files) == {"run_id", "side_id", "battle_id", "draws", "m0_mean", "s0_mean"}
        assert int(data["run_id"]) == 7
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

    from pipeline.reconcilers.records import SideEstimate
    from pipeline.reconcilers.store import write_neutral_estimates, write_side_estimates

    class _Conn:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        def execute(self, statement: object, params: dict[str, object]) -> None:
            self.calls.append((str(statement), params))

    _, _, summary = _summary()
    conn = _Conn()
    assert write_neutral_estimates(conn, summary.estimates) == 3
    assert all("est_troops_neutral_run_id" in sql for sql, _ in conn.calls)
    assert all("est_troops_run_id" not in sql for sql, _ in conn.calls)
    assert {params["run_id"] for _, params in conn.calls} == {7}

    # The published write must not touch the neutral family, so a run where
    # only one of the two passes its thresholds leaves independent provenance.
    published = SideEstimate(
        side_id=10, quantity="troops", value=1.0, lo=0.5, hi=2.0, run_id=9,
        n_reports=1, n_sources=1, method="single_report_debiased",
        updated_at=datetime(2026, 10, 6, tzinfo=UTC),
    )
    other = _Conn()
    write_side_estimates(other, [published])
    assert "neutral" not in other.calls[0][0]
