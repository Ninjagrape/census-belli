"""The frozen-chain diagnostic: marginalised zd must look iid N(0, 1) in every chain."""

from __future__ import annotations

from typing import Any

import arviz as az
import numpy as np
import xarray as xr

from pipeline.reconcilers.summarise import chain_health
from pipeline.stages.reconcile import _global_failures

N_DRAWS = 400
N_SIDES = 6


def _idata(
    zd: np.ndarray | None,
    *,
    step_size: np.ndarray | None = None,
    step_stat: str = "step_size",
) -> az.InferenceData:
    """Build a fit with an optional zd posterior and step size statistic."""
    n_chains = 3 if zd is None else zd.shape[0]
    noise = np.random.default_rng(1).normal(size=(n_chains, N_DRAWS))
    kwargs: dict[str, Any] = {"posterior": {"mu": noise}}
    if zd is not None:
        kwargs["posterior"] = {**kwargs["posterior"], "zd": zd}
    if step_size is not None:
        kwargs["sample_stats"] = {step_stat: step_size}
    return az.from_dict(**kwargs)


def _healthy_zd(n_chains: int = 3, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n_chains, N_DRAWS, N_SIDES))


def _freeze(zd: np.ndarray, chain: int, sides: slice = slice(None)) -> np.ndarray:
    """Shrink a chain's draws to a slow random walk of sd ~0.08."""
    out = zd.copy()
    out[chain, :, sides] = 0.08 * out[chain, :, sides]
    return out


_ALL = np.ones(N_SIDES, dtype=bool)


def test_healthy_chains_are_not_frozen() -> None:
    health = chain_health(_idata(_healthy_zd()), _ALL)

    assert health["frozen_chains"] == []
    assert all(0.8 < sd < 1.2 for sd in health["chain_zd_sd"])


def test_a_chain_with_collapsed_zd_is_frozen() -> None:
    health = chain_health(_idata(_freeze(_healthy_zd(), 2)), _ALL)

    assert health["frozen_chains"] == [2]
    assert health["chain_zd_sd"][2] < 0.15


def test_only_marginal_sides_are_judged() -> None:
    mask = np.array([True, True, True, False, False, False])
    zd = _freeze(_healthy_zd(), 1, slice(3, None))

    assert chain_health(_idata(zd), mask)["frozen_chains"] == []
    assert chain_health(_idata(zd), ~mask)["frozen_chains"] == [1]


def test_no_marginal_sides_gives_no_verdict() -> None:
    for mask in (np.zeros(N_SIDES, dtype=bool), None):
        health = chain_health(_idata(_freeze(_healthy_zd(), 0)), mask)
        assert health["chain_zd_sd"] is None
        assert health["frozen_chains"] == []


def test_a_fit_without_zd_gives_no_verdict() -> None:
    health = chain_health(_idata(None), _ALL)

    assert health["chain_zd_sd"] is None
    assert health["frozen_chains"] == []
    assert health["chain_step_size"] is None
    assert health["small_step_chains"] == []


def test_the_min_zd_sd_argument_moves_the_cutoff() -> None:
    zd = _healthy_zd()

    assert chain_health(_idata(zd), _ALL, min_zd_sd=1.5)["frozen_chains"] == [0, 1, 2]


def test_step_size_is_read_per_chain() -> None:
    steps = np.array([[0.5] * N_DRAWS, [0.4] * N_DRAWS, [0.45] * N_DRAWS])
    health = chain_health(_idata(_healthy_zd(), step_size=steps), _ALL)

    assert health["chain_step_size"] == [0.5, 0.4, 0.45]
    assert health["small_step_chains"] == []


def test_step_size_bar_is_the_fallback() -> None:
    steps = np.array([[0.5] * N_DRAWS, [0.5] * N_DRAWS, [0.05] * N_DRAWS])
    health = chain_health(_idata(_healthy_zd(), step_size=steps, step_stat="step_size_bar"), _ALL)

    assert health["chain_step_size"] == [0.5, 0.5, 0.05]
    assert health["small_step_chains"] == [2]


def test_a_collapsed_step_size_is_flagged_without_being_a_freeze() -> None:
    steps = np.array([[0.5] * N_DRAWS, [0.5] * N_DRAWS, [0.05] * N_DRAWS])
    health = chain_health(_idata(_healthy_zd(), step_size=steps), _ALL)

    assert health["small_step_chains"] == [2]
    assert health["frozen_chains"] == []


def test_the_rollup_reads_the_mask_from_constant_data() -> None:
    from pipeline.reconcilers.summarise import _convergence

    idata = _idata(_freeze(_healthy_zd(), 2))
    # As pymc's nutpie path stores it: no chain or draw dims.
    marginal = xr.Dataset({"data_side_marginal": (("side",), np.ones(N_SIDES))})
    idata.add_groups(constant_data=marginal)

    worst, _, _ = _convergence(idata)

    assert worst["frozen_chains"] == [2]


def test_a_single_chain_works() -> None:
    zd = _healthy_zd(n_chains=1)
    steps = np.full((1, N_DRAWS), 0.3)
    health = chain_health(_idata(zd, step_size=steps), _ALL)

    assert health["frozen_chains"] == []
    assert health["small_step_chains"] == []
    assert len(health["chain_zd_sd"]) == 1


def test_the_gate_reports_frozen_chains() -> None:
    worst = {"max_rhat": 1.0, "min_ess_bulk": 5000, "divergences": 0}
    worst |= {"frozen_chains": [2], "chain_zd_sd": [1.0, 0.99, 0.078]}

    failed = _global_failures(worst, {})

    assert failed == ["frozen chains [2] (median marginal zd sd 0.078 < 0.5)"]


def test_the_gate_threshold_comes_from_write_min_chain_zd_sd() -> None:
    worst = {"max_rhat": 1.0, "min_ess_bulk": 5000, "divergences": 0}
    worst |= {"frozen_chains": [1], "chain_zd_sd": [1.0, 0.3]}

    assert _global_failures(worst, {"write_min_chain_zd_sd": 0.25}) == []
    assert _global_failures(worst, {"write_min_chain_zd_sd": 0.4}) == [
        "frozen chains [1] (median marginal zd sd 0.3 < 0.4)"
    ]


def test_a_payload_without_the_fields_adds_no_failure() -> None:
    worst = {"max_rhat": 1.0, "min_ess_bulk": 5000, "divergences": 0}

    assert _global_failures(worst, {}) == []
