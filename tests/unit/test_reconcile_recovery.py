"""The recovery diagnostics: slopes, group ratios, skips and the variance check.

Every number is built from known values (a slope of 0.6 is constructed, not
fitted), so the expected statistic is exact.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pipeline.reconcilers.design import Design, build_design
from pipeline.reconcilers.records import Report, SourceKey
from pipeline.reconcilers.recovery import (
    RecoverySettings,
    battle_pairs,
    between_battle_ppc,
    group_ratio,
    recovery_diagnostics,
    side_report_logs,
    slope_through_origin,
    within_battle_slope,
)
from pipeline.reconcilers.summarise import singleton_slope, spread_ratio


def _report(report_id: int, side_id: int, battle_id: int, value: float, year: int) -> Report:
    return Report(
        report_id=report_id,
        side_id=side_id,
        battle_id=battle_id,
        source_id=report_id,
        source_key=SourceKey("wikipedia_infobox", f"https://example/{report_id}"),
        source_type="wikipedia_infobox",
        quantity="troops",
        reported_value=value,
        scope="engaged",
        year_astronomical=year,
        claim_regime="unlabelled",
        lineage_id=report_id,
    )


def _battles(n: int, *, ratio_scale: float = 1.0, year: int = 1850) -> list[Report]:
    """``n`` two-sided battles with reported log ratios spread over [-1.5, 1.5]."""
    rng = np.random.default_rng(11)
    reports: list[Report] = []
    for b in range(n):
        base = rng.uniform(8.5, 11.5)
        gap = rng.uniform(-1.5, 1.5) * ratio_scale
        reports.append(_report(2 * b, 2 * b, b, math.exp(base + gap / 2), year))
        reports.append(_report(2 * b + 1, 2 * b + 1, b, math.exp(base - gap / 2), year))
    return reports


def _design(reports: list[Report]) -> Design:
    return build_design(reports, quantity="troops", ancient_cutoff_year=500)


def _estimates(design: Design, slope: float) -> dict[int, float]:
    """Log estimates that are ``slope`` times the (centred) log report, per side."""
    logs = side_report_logs(design)
    centre = float(np.mean(list(logs.values())))
    return {s: centre + slope * (x - centre) for s, x in logs.items()}


def _maps(reports: list[Report]) -> tuple[dict[int, int], dict[int, int | None]]:
    return (
        {r.side_id: r.battle_id for r in reports},
        {r.side_id: r.year_astronomical for r in reports},
    )


def test_slope_through_origin_is_exact_and_needs_three_points() -> None:
    assert slope_through_origin([1.0, 2.0, -3.0], [0.5, 1.0, -1.5]) == pytest.approx(0.5)
    assert slope_through_origin([1.0, 2.0], [0.5, 1.0]) is None
    assert slope_through_origin([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]) is None


@pytest.mark.parametrize("slope", [0.6, 0.85, 1.0])
def test_the_within_battle_slope_recovers_the_constructed_compression(slope: float) -> None:
    reports = _battles(20)
    design = _design(reports)
    battle_of_side, _ = _maps(reports)
    pairs = battle_pairs(design, battle_of_side)

    result, n = within_battle_slope(pairs, side_report_logs(design), _estimates(design, slope))

    assert n == 20
    assert result == pytest.approx(slope)


def test_battle_pairs_need_exactly_two_point_reported_sides() -> None:
    reports = _battles(3)
    reports.append(_report(100, 100, 0, 5_000.0, 1850))  # a third side in battle 0
    reports.append(_report(101, 101, 99, 5_000.0, 1850))  # a battle with one side
    design = _design(reports)
    battle_of_side, _ = _maps(reports)

    pairs = battle_pairs(design, battle_of_side)

    assert [p.battle_id for p in pairs] == [1, 2]


def test_the_group_ratio_is_the_geometric_mean_of_estimate_over_report() -> None:
    reports = [_report(i, i, i, 100_000.0, 1850) for i in range(5)]
    design = _design(reports)
    logs = side_report_logs(design)
    estimates = {s: x + math.log(0.7) for s, x in logs.items()}

    ratio, n = group_ratio(logs, estimates, list(logs))

    assert n == 5
    assert ratio == pytest.approx(0.7)
    assert group_ratio(logs, estimates, []) == (None, 0)


def _diagnostics(reports: list[Report], slope: float, **overrides):  # type: ignore[no-untyped-def]
    design = _design(reports)
    estimates = _estimates(design, slope)
    battle_of_side, year_of_side = _maps(reports)
    settings = overrides.pop("settings", RecoverySettings())
    return recovery_diagnostics(
        design,
        estimates,
        battle_of_side,
        year_of_side,
        settings=settings,
        ancient_cutoff_year=500,
        singleton_slope=singleton_slope(design, estimates),
        spread_ratio=spread_ratio(design, estimates),
        ppc=overrides.pop("ppc", None),
    )


def test_a_compressed_fit_reports_the_slopes_the_gates_read() -> None:
    block = _diagnostics(_battles(30), 0.6)

    assert block["singleton_slope"] == pytest.approx(0.6)
    assert block["within_battle_slope"] == pytest.approx(0.6)
    assert block["spread_ratio"] == pytest.approx(0.6, abs=0.02)
    assert block["gate_singleton_slope"] == pytest.approx(0.6)
    assert block["gate_within_battle_slope"] == pytest.approx(0.6)
    assert block["n_two_sided_battles"] == 30


def test_the_post_1800_band_flags_shrunk_large_modern_armies() -> None:
    # Every side reports > 50k in 1850; the fit halves the log distance to the centre.
    reports = [_report(i, i, i // 2, math.exp(11.5 + 0.4 * (i % 3 - 1)), 1850) for i in range(30)]
    design = _design(reports)
    logs = side_report_logs(design)
    shrunk = {s: x + math.log(0.7) for s, x in logs.items()}
    battle_of_side, year_of_side = _maps(reports)

    block = recovery_diagnostics(
        design,
        shrunk,
        battle_of_side,
        year_of_side,
        settings=RecoverySettings(),
        ancient_cutoff_year=500,
        singleton_slope=None,
        spread_ratio=None,
        ppc=None,
    )
    faithful = recovery_diagnostics(
        design,
        {s: x + math.log(1.02) for s, x in logs.items()},
        battle_of_side,
        year_of_side,
        settings=RecoverySettings(),
        ancient_cutoff_year=500,
        singleton_slope=None,
        spread_ratio=None,
        ppc=None,
    )

    assert block["post1800_large_ratio"] == pytest.approx(0.7)
    assert block["gate_post1800_band_violation"] == 1
    assert faithful["gate_post1800_band_violation"] == 0


def test_ancient_and_small_groups_are_reported_separately() -> None:
    reports = [
        _report(0, 0, 0, 200_000.0, -300),
        _report(1, 1, 0, 100_000.0, -300),
        _report(2, 2, 1, 5_000.0, 1850),
        _report(3, 3, 1, 4_000.0, 1850),
    ]
    design = _design(reports)
    logs = side_report_logs(design)
    estimates = {s: x + (math.log(0.9) if logs[s] > 10 else math.log(2.0)) for s, x in logs.items()}
    battle_of_side, year_of_side = _maps(reports)

    block = recovery_diagnostics(
        design,
        estimates,
        battle_of_side,
        year_of_side,
        settings=RecoverySettings(),
        ancient_cutoff_year=500,
        singleton_slope=None,
        spread_ratio=None,
        ppc=None,
    )

    assert block["ancient_large_ratio"] == pytest.approx(0.9)
    assert block["n_ancient_large"] == 2
    assert block["small_ratio"] == pytest.approx(2.0)
    assert block["n_small"] == 2


def test_gates_skip_with_a_reason_on_a_small_fixture() -> None:
    block = _diagnostics(_battles(3), 0.3)

    for gate in ("gate_singleton_slope", "gate_spread_ratio", "gate_within_battle_slope"):
        assert block[gate] is None
    assert block["gate_post1800_band_violation"] is None
    assert block["gate_ppc_tail_p"] is None
    assert set(block["gates_skipped"]) == {
        "singleton_slope",
        "spread_ratio",
        "within_battle_slope",
        "post1800_ratio",
        "post1800_relative_level",
        "all_sides_ratio",
        "mixed_ppc",
        "s_d_median",
        "noise_ratio",
        "wb_reliability_gap",
    }
    # The statistic itself is still recorded.
    assert block["within_battle_slope"] == pytest.approx(0.3)


def test_the_minimum_sizes_come_from_the_settings() -> None:
    settings = RecoverySettings.from_params(
        {"recovery_min_sides": 3, "recovery_min_battles": 3, "post1800_ratio_band_hi": 1.3}
    )

    block = _diagnostics(_battles(4), 0.5, settings=settings)

    assert block["gate_within_battle_slope"] == pytest.approx(0.5)
    assert settings.ratio_band == (0.85, 1.3)


def _ppc_inputs(observed_scale: float, noise_sd: float, n_battles: int = 40):  # type: ignore[no-untyped-def]
    """Reports drawn at ``observed_scale`` spread, against a model with ``noise_sd`` noise."""
    reports = _battles(n_battles, ratio_scale=observed_scale)
    design = _design(reports)
    battle_of_side, _ = _maps(reports)
    pairs = battle_pairs(design, battle_of_side)
    rows = np.arange(design.n_obs)
    n_draws = 400
    # The model's location is the report's own side centre, so the replicated
    # ratios have only the noise, and the check compares that to what was seen.
    eta = np.tile(design.log_y, (n_draws, 1))
    var = np.full_like(eta, noise_sd**2)
    return pairs, design.log_y, eta, var, rows


def test_the_variance_check_is_central_when_the_fit_reproduces_the_spread() -> None:
    pairs, y, eta, var, rows = _ppc_inputs(observed_scale=1.0, noise_sd=0.05)

    out = between_battle_ppc(pairs, y, eta, var, rows, None, seed=1)

    assert out["extremity"] is not None
    assert out["extremity"] > 0.025
    assert out["observed"] == pytest.approx(out["replicated_median"], rel=0.1)


def test_the_variance_check_is_extreme_when_the_noise_implies_too_much_spread() -> None:
    # Reports barely differ between sides; a noise sd of 1 implies large ratios.
    pairs, y, eta, var, rows = _ppc_inputs(observed_scale=0.05, noise_sd=1.0)

    out = between_battle_ppc(pairs, y, eta, var, rows, np.full(400, 5.0), seed=1)

    assert out["p_value"] is not None
    assert out["p_value"] > 0.99
    assert out["extremity"] is not None
    assert out["extremity"] < 0.025
    assert out["observed"] < out["replicated_median"]


def test_the_variance_check_uses_its_own_seed() -> None:
    pairs, y, eta, var, rows = _ppc_inputs(observed_scale=1.0, noise_sd=0.5)

    a = between_battle_ppc(pairs, y, eta, var, rows, None, seed=7)
    b = between_battle_ppc(pairs, y, eta, var, rows, None, seed=7)

    assert a == b
