"""P1 recovery checks: the mixed-predictive PPC and the new gate values.

The mixed predictive draws new side deviations (battle effect and side effect) per
posterior draw, keeps the fitted offsets and draws new observation noise, then
compares the between-battle second moment of the reported log ratio with the
observed one. Every input here is constructed, so the verdict is known.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pipeline.reconcilers.design import build_design
from pipeline.reconcilers.records import Report, SourceKey
from pipeline.reconcilers.recovery import (
    BattlePair,
    RecoverySettings,
    battle_pairs,
    mixed_predictive_ppc,
    recovery_diagnostics,
    side_report_logs,
)

N_BATTLES = 60
N_DRAWS = 300
S_B = 0.8
S_D = 0.4
NOISE = 0.22


def _pairs(n: int) -> list[BattlePair]:
    return [
        BattlePair(b, 2 * b, 2 * b + 1, np.array([2 * b]), np.array([2 * b + 1])) for b in range(n)
    ]


def _simulate(rng: np.random.Generator, *, s_d: float, noise: float) -> np.ndarray:
    zb = rng.standard_t(4, size=N_BATTLES)
    zd = rng.standard_normal(2 * N_BATTLES)
    battle = np.repeat(np.arange(N_BATTLES), 2)
    return 10.0 + S_B * zb[battle] + s_d * zd + noise * rng.standard_normal(2 * N_BATTLES)


def _ppc(y: np.ndarray, *, seed: int = 5) -> dict[str, float]:
    n_rows = 2 * N_BATTLES
    return mixed_predictive_ppc(
        _pairs(N_BATTLES),
        y,
        np.arange(n_rows),
        np.full((N_DRAWS, n_rows), 10.0),
        np.full((N_DRAWS, n_rows), NOISE**2),
        np.arange(n_rows),
        np.repeat(np.arange(N_BATTLES), 2),
        np.full(N_DRAWS, S_B),
        np.full(N_DRAWS, S_D),
        None,
        battle_nu=4.0,
        seed=seed,
    )


def test_the_mixed_ppc_is_not_extreme_when_the_data_come_from_the_model() -> None:
    y = _simulate(np.random.default_rng(21), s_d=S_D, noise=NOISE)

    result = _ppc(y)

    assert result["tail_p"] > 0.05
    assert result["n_battles"] == N_BATTLES


def test_the_mixed_ppc_flags_reports_that_spread_far_more_than_the_model_allows() -> None:
    y = _simulate(np.random.default_rng(22), s_d=3 * S_D, noise=NOISE)

    result = _ppc(y)

    assert result["tail_p"] < 0.01
    assert result["observed"] > result["replicated_median"]


def test_the_mixed_ppc_is_two_sided_so_it_also_flags_far_too_little_spread() -> None:
    # Both sides of every battle reported almost identically: the model's
    # predicted spread is far larger, so the observed statistic sits in the lower tail.
    y = _simulate(np.random.default_rng(23), s_d=0.0, noise=0.01)
    y = y.reshape(N_BATTLES, 2).mean(axis=1).repeat(2) + 0.005

    result = _ppc(y)

    assert result["tail_p"] < 0.01
    assert result["observed"] < result["replicated_median"]


def test_the_battle_effect_cancels_in_the_ratio_but_the_side_effect_does_not() -> None:
    y = _simulate(np.random.default_rng(24), s_d=S_D, noise=NOISE)
    n_rows = 2 * N_BATTLES
    kwargs: dict = {
        "pairs": _pairs(N_BATTLES),
        "y_obs": y,
        "rows": np.arange(n_rows),
        "fixed": np.full((N_DRAWS, n_rows), 10.0),
        "var": np.full((N_DRAWS, n_rows), NOISE**2),
        "side_of_row": np.arange(n_rows),
        "battle_of_side": np.repeat(np.arange(N_BATTLES), 2),
        "nu": None,
        "battle_nu": 4.0,
        "seed": 5,
    }

    big_battle = mixed_predictive_ppc(
        **kwargs, s_b=np.full(N_DRAWS, 5.0), s_d=np.full(N_DRAWS, S_D)
    )
    no_battle = mixed_predictive_ppc(**kwargs, s_b=np.zeros(N_DRAWS), s_d=np.full(N_DRAWS, S_D))
    no_side = mixed_predictive_ppc(**kwargs, s_b=np.zeros(N_DRAWS), s_d=np.zeros(N_DRAWS))

    assert big_battle["replicated_median"] == pytest.approx(
        no_battle["replicated_median"], rel=0.25
    )
    assert no_side["replicated_median"] < 0.5 * no_battle["replicated_median"]


def test_the_mixed_ppc_uses_its_own_seed() -> None:
    y = _simulate(np.random.default_rng(21), s_d=S_D, noise=NOISE)

    assert _ppc(y, seed=1) == _ppc(y, seed=1)
    assert _ppc(y, seed=1)["replicated_median"] != _ppc(y, seed=2)["replicated_median"]


# ─── gate values ─────────────────────────────────────────────────────────────


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


def _two_group_design() -> tuple[object, list[Report]]:
    """12 large post-1800 sides and 12 small ones, one battle per pair of sides."""
    rng = np.random.default_rng(31)
    reports: list[Report] = []
    for b in range(12):
        big = math.exp(rng.uniform(11.2, 12.5))
        small = math.exp(rng.uniform(6.9, 8.5))
        reports.append(_report(2 * b, 2 * b, b, big, 1850))
        reports.append(_report(2 * b + 1, 2 * b + 1, b, small, 1850))
    return build_design(reports, quantity="troops", ancient_cutoff_year=500), reports


def _diagnostics(
    gap_large: float,
    gap_small: float,
    side_prior: dict[str, float | None] | None = None,
    ppc: dict[str, float] | None = None,
) -> dict:
    design, reports = _two_group_design()
    logs = side_report_logs(design)  # type: ignore[arg-type]
    large = {s for s, x in logs.items() if math.exp(x) > 50_000}
    estimates = {s: x + (gap_large if s in large else gap_small) for s, x in logs.items()}
    return recovery_diagnostics(
        design,  # type: ignore[arg-type]
        estimates,
        {r.side_id: r.battle_id for r in reports},
        {r.side_id: r.year_astronomical for r in reports},
        settings=RecoverySettings(),
        ancient_cutoff_year=500,
        singleton_slope=0.95,
        spread_ratio=0.95,
        ppc=ppc,
        side_prior=side_prior,
    )


@pytest.mark.parametrize(
    ("relative", "violation"),
    [(1.14, 0), (1.16, 1), (0.86, 0), (0.84, 1), (1.0, 0)],
)
def test_the_modern_to_all_sides_ratio_band_is_0_85_to_1_15(
    relative: float, violation: int
) -> None:
    # Large gap g_l and small gap g_s: all-sides ratio exp((g_l + g_s) / 2), modern ratio exp(g_l),
    # so relative = exp((g_l - g_s) / 2).
    half = math.log(relative)
    rec = _diagnostics(gap_large=half, gap_small=-half)

    assert rec["relative_level"] == pytest.approx(relative, rel=1e-6)
    assert rec["gate_relative_level_violation"] == violation


@pytest.mark.parametrize(
    ("all_ratio", "violation"),
    [(1.09, 0), (1.11, 1), (0.91, 0), (0.89, 1), (1.0, 0)],
)
def test_the_all_sides_ratio_band_is_0_9_to_1_1(all_ratio: float, violation: int) -> None:
    g = math.log(all_ratio)
    rec = _diagnostics(gap_large=g, gap_small=g)

    assert rec["all_ratio"] == pytest.approx(all_ratio, rel=1e-6)
    assert rec["gate_all_ratio_violation"] == violation


def test_the_side_prior_statistics_reach_the_gate_values() -> None:
    rec = _diagnostics(
        0.0,
        0.0,
        side_prior={
            "s_d_median": 0.31,
            "noise_ratio_modern": 1.2,
            "noise_ratio_ancient": 1.4,
            "reliability": 0.9,
        },
        ppc={"tail_p": 0.2, "observed": 0.5, "replicated_median": 0.5},
    )

    assert rec["gate_s_d_median"] == pytest.approx(0.31)
    assert rec["gate_noise_ratio"] == pytest.approx(1.4)
    assert rec["gate_ppc_tail_p"] == pytest.approx(0.2)
    # Estimates equal reports here, so wb is 1 and the gap is |1 - reliability|.
    assert rec["within_battle_slope"] == pytest.approx(1.0)
    assert rec["gate_wb_reliability_gap"] == pytest.approx(0.1)


def test_gate_values_are_none_when_the_inputs_are_missing() -> None:
    rec = _diagnostics(0.0, 0.0)

    assert rec["gate_s_d_median"] is None
    assert rec["gate_noise_ratio"] is None
    assert rec["gate_ppc_tail_p"] is None
    assert rec["gate_wb_reliability_gap"] is None


def test_the_battle_pairs_helper_still_pairs_two_sided_battles() -> None:
    design, reports = _two_group_design()

    pairs = battle_pairs(design, {r.side_id: r.battle_id for r in reports})  # type: ignore[arg-type]

    assert len(pairs) == 12
