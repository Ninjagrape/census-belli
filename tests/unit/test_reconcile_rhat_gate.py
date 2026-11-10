"""The rhat gate skips the raw battle offsets but keeps every published variable."""

from __future__ import annotations

import arviz as az
import numpy as np
import pytest

from pipeline.reconcilers.summarise import RHAT_GATE_EXCLUDED, _convergence

N_CHAINS = 4
N_DRAWS = 500


def _idata(bad: str | None) -> az.InferenceData:
    """A fit of healthy variables, with chain means shifted on ``bad`` if given."""
    rng = np.random.default_rng(3)
    names = ("mu", "b", "zb", "b_raw")
    posterior = {n: rng.normal(size=(N_CHAINS, N_DRAWS, 3)) for n in names}
    if bad is not None:
        shift = np.arange(N_CHAINS, dtype=float)[:, None, None] * 2.0
        posterior[bad] = posterior[bad] + shift
    return az.from_dict(posterior=posterior)


def test_the_excluded_set_names_the_raw_battle_offsets() -> None:
    assert frozenset({"zb", "b_raw"}) == RHAT_GATE_EXCLUDED


@pytest.mark.parametrize("raw", ["zb", "b_raw"])
def test_a_bad_raw_offset_does_not_drive_max_rhat(raw: str) -> None:
    worst, which, _ = _convergence(_idata(raw))
    assert worst["max_rhat"] is not None and worst["max_rhat"] < 1.1
    assert worst["max_rhat_raw_battle"] > 1.1
    assert which["rhat"] in {"mu", "b"}


def test_a_bad_b_still_drives_max_rhat() -> None:
    worst, which, _ = _convergence(_idata("b"))
    assert worst["max_rhat"] > 1.1
    assert which["rhat"] == "b"


def test_raw_battle_rhat_is_none_when_absent() -> None:
    idata = az.from_dict(
        posterior={"mu": np.random.default_rng(0).normal(size=(N_CHAINS, N_DRAWS, 3))}
    )
    worst, _, _ = _convergence(idata)
    assert worst["max_rhat_raw_battle"] is None
