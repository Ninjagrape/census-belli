"""The impute recovery gates: kept-pair log-ratio slope and kept-side slope.

Rounds are built from known numbers, so each slope is exact.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.imputers.gates import kept_pair_ratio_slope, kept_side_slope, run_gates
from pipeline.imputers.records import ImputeCounts


def _battles(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two sides per battle, reported log values spread widely."""
    rng = np.random.default_rng(seed)
    battle_ids = np.repeat(np.arange(n), 2)
    report = rng.uniform(6.0, 13.0, size=2 * n)
    return battle_ids, report


def _rounds(report: np.ndarray, slope: float, n_rounds: int = 5) -> np.ndarray:
    """Imputed values that keep ``slope`` of each side's distance from the centre."""
    centre = float(np.mean(report))
    noise = np.random.default_rng(1).normal(0.0, 0.01, size=(n_rounds, report.size))
    return centre + slope * (report - centre) + noise


def test_the_pair_slope_recovers_a_constructed_compression() -> None:
    battle_ids, report = _battles(30)
    values = _rounds(report, 0.43)

    gate = kept_pair_ratio_slope(
        values, battle_ids, report, np.ones(60, dtype=bool), minimum=0.8, min_n=10
    )

    assert not gate.passed
    assert gate.severity == "error"
    assert "slope=0.43" in gate.actual
    assert "battles=30" in gate.actual


def test_the_pair_slope_passes_when_ratios_are_kept() -> None:
    battle_ids, report = _battles(30)

    gate = kept_pair_ratio_slope(
        _rounds(report, 0.97), battle_ids, report, np.ones(60, dtype=bool), minimum=0.8, min_n=10
    )

    assert gate.passed
    assert "slope=0.9" in gate.actual


def test_only_battles_with_two_observed_sides_enter_the_pair_slope() -> None:
    battle_ids, report = _battles(30)
    observed = np.ones(60, dtype=bool)
    observed[0] = False  # battle 0 now has one observed side
    values = _rounds(report, 0.9)
    values[:, :2] = 0.0  # garbage that would wreck the slope if battle 0 were used

    gate = kept_pair_ratio_slope(values, battle_ids, report, observed, minimum=0.8, min_n=10)

    assert gate.passed
    assert "battles=29" in gate.actual


def test_the_pair_slope_skips_below_the_minimum_with_its_reason() -> None:
    battle_ids, report = _battles(4)

    gate = kept_pair_ratio_slope(
        _rounds(report, 0.1), battle_ids, report, np.ones(8, dtype=bool), minimum=0.8, min_n=10
    )

    assert gate.passed
    assert gate.actual.startswith("skipped: 4 pairs, need 10")


def test_the_side_slope_is_the_ols_slope_of_imputed_on_reported() -> None:
    _, report = _battles(30)
    observed = np.ones(60, dtype=bool)

    low = kept_side_slope(_rounds(report, 0.6), report, observed, minimum=0.9, min_n=10)
    high = kept_side_slope(_rounds(report, 0.98), report, observed, minimum=0.9, min_n=10)

    assert not low.passed
    assert "slope=0.60" in low.actual
    assert high.passed


def test_the_side_slope_ignores_sides_without_a_report() -> None:
    _, report = _battles(30)
    observed = np.ones(60, dtype=bool)
    observed[:10] = False
    report = report.copy()
    report[:10] = np.nan
    values = _rounds(np.nan_to_num(report, nan=9.0), 0.95)

    gate = kept_side_slope(values, report, observed, minimum=0.9, min_n=10)

    assert gate.passed
    assert "sides=50" in gate.actual


def test_the_side_slope_skips_on_few_sides() -> None:
    report = np.array([8.0, 9.0, 10.0])

    gate = kept_side_slope(
        _rounds(report, 0.1), report, np.ones(3, dtype=bool), minimum=0.9, min_n=10
    )

    assert gate.passed
    assert "skipped" in gate.actual


def test_run_gates_adds_the_slope_gates_only_when_given_reports() -> None:
    battle_ids, report = _battles(12)
    values = _rounds(report, 1.0)
    common = {
        "is_imputed": np.zeros(24, dtype=bool),
        "battle_ids": battle_ids,
        "era_of_side": np.array(["modern"] * 24),
        "scale": "neutral",
        "n_rounds": 5,
        "ratio_limit": 4.0,
        "value_min": 50.0,
        "value_max": 1e7,
        "ks_alpha": 0.01,
        "ks_min_n": 5,
    }

    without = run_gates(values, **common)  # type: ignore[arg-type]
    with_slopes = run_gates(
        values,
        report=report,
        is_observed=np.ones(24, dtype=bool),
        slope_min_n=10,
        **common,  # type: ignore[arg-type]
    )

    assert [g.name for g in without] == [
        "imputation_completeness",
        "imputed_ratio_plausible",
        "imputed_values_plausible",
        "distributional_check",
    ]
    assert [g.name for g in with_slopes][-2:] == ["kept_pair_ratio_slope", "kept_side_slope"]
    assert all(g.passed for g in with_slopes)


def test_the_counts_carry_the_cause_and_retired_row_fields() -> None:
    counts = ImputeCounts(
        sides_missing_bound_only=3, sides_missing_no_reports=2, sides_missing_other=1
    )
    counts.log_rows_retired = 4

    as_dict = counts.as_dict()

    assert (
        as_dict["sides_missing_bound_only"],
        as_dict["sides_missing_no_reports"],
        as_dict["sides_missing_other"],
        as_dict["log_rows_retired"],
    ) == (3, 2, 1, 4)
    assert "sides_weak_as_missing" not in as_dict


def test_the_removed_min_lik_precision_param_is_ignored_with_a_warning() -> None:
    from pipeline.imputers.priors import ImputeParams

    params = ImputeParams.from_params({"min_lik_precision": 0.5, "min_pair_slope": 0.7})

    assert params.min_pair_slope == pytest.approx(0.7)
    assert not hasattr(params, "min_lik_precision")
