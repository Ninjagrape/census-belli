"""Synthetic corpus for the impute recovery test.

Nothing here is a test. The corpus is generated from the same linear-Gaussian
structure the model assumes (``pipeline/imputers/model.py``), with known
coefficients, so a recovery test can check the fit against truth. Outcome is
both a covariate of the imputation model and the downstream regressor: that
is what makes the attenuation check meaningful, because multiple imputation
preserves the joint law of the outcome and the imputed values only when the
outcome is in the imputation model (Rubin's compatibility condition).

All randomness goes through an explicit ``numpy.random.Generator``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pipeline.imputers.records import BattleLikelihood, SideRow

ERA_YEARS: tuple[int | None, ...] = (-200, 1000, 1650, 1805, 1870, 1944, None)
TRUE_A_ERA: tuple[float, ...] = (9.0, 8.7, 9.8, 10.4, 10.6, 11.2, 10.0)
TRUE_BETA_OUT: dict[str, float] = {"victory": 0.5, "defeat": -0.5, "indecisive": 0.0}
TRUE_BETA_TYPE: dict[str, float] = {"field": 0.4, "naval": -0.5, "siege_defensive": 0.1}
TRUE_TAU_B = 0.6
TRUE_SIGMA = 0.5
TRUE_TAU_P = 0.4
TRUE_TAU_C = 0.4
N_POLITIES = 8
N_COMMANDERS = 40
MEASUREMENT_SD_RANGE = (0.4, 1.0)


@dataclass(frozen=True)
class SimCorpus:
    """A simulated corpus with its truth.

    Attributes:
        sides: Every side, with covariates.
        likelihoods: Likelihoods for the sides left observed.
        true_y: True log troops per side id.
        measured_y: The noisy likelihood mean for observed sides.
        is_missing: Side ids with no likelihood.
        outcome_sign_a: For each two-sided battle id, +1 if side A won, -1 lost.
        two_sided: ``battle_id -> (side_a_id, side_b_id)``.
        polity_effect: True polity effect by polity key (``'Q3'``).
        commander_effect: True commander effect by commander key.
    """

    sides: list[SideRow]
    likelihoods: list[BattleLikelihood]
    true_y: dict[int, float]
    measured_y: dict[int, float]
    is_missing: set[int]
    outcome_sign_a: dict[int, int]
    two_sided: dict[int, tuple[int, int]]
    polity_effect: dict[str, float]
    commander_effect: dict[str, float]


def simulate_corpus(
    seed: int,
    *,
    n_battles: int = 300,
    p_missing: float = 0.3,
) -> SimCorpus:
    """Simulate battles, sides, latent troop counts and reconcile-style likelihoods.

    Args:
        seed: Generator seed.
        n_battles: Battles to simulate; about 85% two-sided, the rest one- or
            three-sided.
        p_missing: Probability each side has no likelihood (independent, so
            some two-sided battles lose both sides).

    Returns:
        The corpus.
    """
    rng = np.random.default_rng(seed)
    polity_effect = TRUE_TAU_P * rng.standard_normal(N_POLITIES)
    commander_effect = TRUE_TAU_C * rng.standard_normal(N_COMMANDERS)
    types = list(TRUE_BETA_TYPE)

    sides: list[SideRow] = []
    true_y: dict[int, float] = {}
    measured: dict[int, float] = {}
    sd_of: dict[int, float] = {}
    outcome_sign_a: dict[int, int] = {}
    two_sided: dict[int, tuple[int, int]] = {}
    side_id = 0

    for battle_id in range(1, n_battles + 1):
        era = int(rng.integers(len(ERA_YEARS)))
        battle_type = types[int(rng.integers(len(types)))]
        u_b = TRUE_TAU_B * rng.standard_normal()
        draw = rng.random()
        n_sides = 1 if draw < 0.08 else 3 if draw < 0.15 else 2
        if n_sides == 2:
            sign = 1 if rng.random() < 0.5 else -1
            outcomes = ["victory", "defeat"] if sign == 1 else ["defeat", "victory"]
            outcome_sign_a[battle_id] = sign
        elif n_sides == 3:
            outcomes = ["victory", "defeat", "indecisive"]
        else:
            outcomes = [["victory", "defeat", "indecisive"][int(rng.integers(3))]]

        ids = []
        for outcome in outcomes:
            side_id += 1
            polity = int(rng.integers(N_POLITIES))
            commander = int(rng.integers(N_COMMANDERS))
            y = (
                TRUE_A_ERA[era]
                + TRUE_BETA_OUT[outcome]
                + TRUE_BETA_TYPE[battle_type]
                + polity_effect[polity]
                + commander_effect[commander]
                + u_b
                + TRUE_SIGMA * rng.standard_normal()
            )
            sd = float(rng.uniform(*MEASUREMENT_SD_RANGE))
            true_y[side_id] = float(y)
            measured[side_id] = float(y + sd * rng.standard_normal())
            sd_of[side_id] = sd
            sides.append(
                SideRow(
                    side_id=side_id,
                    battle_id=battle_id,
                    year=ERA_YEARS[era],
                    outcome=outcome,
                    battle_type=battle_type,
                    is_defender=None,
                    polity_key=f"Q{polity}",
                    commander_key=f"general_{commander}",
                )
            )
            ids.append(side_id)
        if n_sides == 2:
            two_sided[battle_id] = (ids[0], ids[1])

    missing = {s.side_id for s in sides if rng.random() < p_missing}
    by_battle: dict[int, list[int]] = {}
    for s in sides:
        if s.side_id not in missing:
            by_battle.setdefault(s.battle_id, []).append(s.side_id)
    likelihoods = [
        BattleLikelihood(
            battle_id=bid,
            side_ids=tuple(ids),
            m_lik=np.array([measured[i] for i in ids]),
            precision=np.diag([1.0 / sd_of[i] ** 2 for i in ids]),
            method="joint",
        )
        for bid, ids in sorted(by_battle.items())
    ]
    return SimCorpus(
        sides,
        likelihoods,
        true_y,
        measured,
        missing,
        outcome_sign_a,
        two_sided,
        {f"Q{i}": float(v) for i, v in enumerate(polity_effect)},
        {f"general_{i}": float(v) for i, v in enumerate(commander_effect)},
    )
