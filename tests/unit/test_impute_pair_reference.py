"""The impute kept-pair gate compares with reconcile's within-battle slope.

A calibrated reconcile fit shrinks reported force ratios by their reliability, so
a fixed ``>= 0.8`` floor on impute's slope asks for more than reconcile gives.
Impute is instead held to reconcile's own ``wb`` within a tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.imputers.gates import kept_pair_ratio_slope, run_gates
from pipeline.imputers.priors import ImputeParams


def _battles(n: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    return np.repeat(np.arange(n), 2), rng.uniform(6.0, 13.0, size=2 * n)


def _rounds(report: np.ndarray, slope: float) -> np.ndarray:
    centre = float(np.mean(report))
    return (centre + slope * (report - centre))[None, :].repeat(4, axis=0)


@pytest.mark.parametrize(
    ("slope", "passes"),
    [(0.60, True), (0.51, True), (0.69, True), (0.45, False), (0.75, False), (0.97, False)],
)
def test_the_pair_slope_must_sit_within_the_tolerance_of_the_reconcile_slope(
    slope: float, passes: bool
) -> None:
    battle_ids, report = _battles(30)

    gate = kept_pair_ratio_slope(
        _rounds(report, slope),
        battle_ids,
        report,
        np.ones(60, dtype=bool),
        minimum=0.8,
        min_n=10,
        reference=0.6,
        tolerance=0.1,
    )

    assert gate.passed is passes
    assert gate.severity == "error"
    assert "reference=0.600" in gate.actual


def test_without_a_reference_the_gate_keeps_its_floor() -> None:
    battle_ids, report = _battles(30)

    low = kept_pair_ratio_slope(
        _rounds(report, 0.6), battle_ids, report, np.ones(60, dtype=bool), minimum=0.8, min_n=10
    )
    high = kept_pair_ratio_slope(
        _rounds(report, 0.9), battle_ids, report, np.ones(60, dtype=bool), minimum=0.8, min_n=10
    )

    assert not low.passed
    assert high.passed


def test_run_gates_threads_the_reference_into_the_pair_gate() -> None:
    battle_ids, report = _battles(12)
    values = _rounds(report, 0.6)
    common = {
        "is_imputed": np.zeros(24, dtype=bool),
        "battle_ids": battle_ids,
        "era_of_side": np.array(["modern"] * 24),
        "scale": "neutral",
        "n_rounds": 4,
        "ratio_limit": 4.0,
        "value_min": 50.0,
        "value_max": 1e7,
        "ks_alpha": 0.01,
        "ks_min_n": 5,
        "report": report,
        "is_observed": np.ones(24, dtype=bool),
        "slope_min_n": 10,
    }

    gates = run_gates(values, reference_pair_slope=0.6, **common)  # type: ignore[arg-type]
    pair = next(g for g in gates if g.name == "kept_pair_ratio_slope")

    assert pair.passed


def test_the_tolerance_comes_from_the_spec_with_a_default_of_one_tenth() -> None:
    assert ImputeParams.from_params({}).pair_slope_tolerance == pytest.approx(0.1)
    assert ImputeParams.from_params({"pair_slope_tolerance": 0.05}).pair_slope_tolerance == (
        pytest.approx(0.05)
    )
