"""Simulated tournament for the Bradley-Terry recovery test.

Nothing here is a test. The simulator produces raw :class:`SideInput` rows,
not arrays, so the recovery test exercises outcome assembly, orientation and
side classification in :func:`pipeline.modelers.design.build_design` as well as
the PyMC model. Outcomes are drawn from the same ordinal logit the model
assumes; a recovery test can only show that the model inverts its own
generative story, which is exactly the property a wiring or sign bug breaks.

Two eras of generals who never meet, so their relative level is identified by
the prior alone. Some sides are co-commanded (weights 0.7 / 0.3) and a few have
an unrecorded commander.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pipeline.modelers.ordinal import category_probs
from pipeline.modelers.records import CommanderInput, SideInput

__all__ = [
    "TRUE_BETA_DEFENDER",
    "TRUE_BETA_FORCE",
    "TRUE_C1",
    "TRUE_C2",
    "TRUE_MU_UNKNOWN",
    "TRUE_SIGMA_SKILL",
    "SimTournament",
    "simulate_tournament",
]

TRUE_SIGMA_SKILL = 1.3
TRUE_BETA_FORCE = 0.5
TRUE_BETA_DEFENDER = 0.4
TRUE_C1 = 0.3
TRUE_C2 = 1.2
TRUE_MU_UNKNOWN = -0.4
LEVELS = ("decisive_defeat", "defeat", "indecisive", "victory", "decisive_victory")


@dataclass(frozen=True)
class SimTournament:
    """A simulated corpus and its truth.

    Attributes:
        sides: Raw side rows in shuffled order.
        log_troops: Log troop count per ``side_id``.
        true_skill: Skill per ``general_id``.
        era_of: Era (0 or 1) per ``general_id``.
    """

    sides: list[SideInput]
    log_troops: dict[int, float]
    true_skill: dict[int, float]
    era_of: dict[int, int]


def simulate_tournament(
    seed: int,
    *,
    generals_per_era: int = 30,
    battles_per_era: int = 200,
    p_unknown: float = 0.04,
    p_pair: float = 0.12,
    beta_defender: float = 0.0,
) -> SimTournament:
    """Simulate two disconnected eras of battles.

    Args:
        seed: Generator seed.
        generals_per_era: Generals in each era.
        battles_per_era: Battles in each era.
        p_unknown: Probability a side has no recorded commander.
        p_pair: Probability a side has two co-commanders.
        beta_defender: True defender coefficient. Non-zero also flags every
            battle (one side defends, chosen by a coin flip drawn from a
            generator of its own so the default tournament is unchanged);
            zero leaves every flag unknown.

    Returns:
        The tournament.
    """
    rng = np.random.default_rng(seed)
    flag_rng = np.random.default_rng(seed + 7919)
    n = 2 * generals_per_era
    skill = rng.normal(0.0, TRUE_SIGMA_SKILL, n)
    true_skill = {g + 1: float(skill[g]) for g in range(n)}
    era_of = {g + 1: g // generals_per_era for g in range(n)}

    sides: list[SideInput] = []
    log_troops: dict[int, float] = {}
    side_id = 0
    battle_id = 0
    for era in (0, 1):
        pool = np.arange(era * generals_per_era, (era + 1) * generals_per_era) + 1
        for _ in range(battles_per_era):
            battle_id += 1
            picks = rng.choice(pool, 4, replace=False)
            sides_gen: list[tuple[int, ...]] = []
            used = 0
            unknown_taken = False
            for _side in range(2):
                roll = rng.random()
                if roll < p_unknown and not unknown_taken:
                    sides_gen.append(())
                    unknown_taken = True
                    continue
                if roll < p_unknown + p_pair:
                    sides_gen.append((int(picks[used]), int(picks[used + 1])))
                    used += 2
                else:
                    sides_gen.append((int(picks[used]),))
                    used += 1
            scores: list[float] = []
            weights_of: list[tuple[float, ...]] = []
            for gens in sides_gen:
                if not gens:
                    scores.append(TRUE_MU_UNKNOWN + TRUE_SIGMA_SKILL * float(rng.normal()))
                    weights_of.append(())
                    continue
                w = (1.0,) if len(gens) == 1 else (0.7, 0.3)
                weights_of.append(w)
                scores.append(sum(wi * true_skill[g] for wi, g in zip(w, gens, strict=True)))
            x = rng.normal(0.0, 0.8, 2)
            ids = (side_id, side_id + 1)
            side_id += 2
            # Orientation must carry no information: randomise which side gets
            # the lower id by swapping the pair's labels.
            order = (0, 1) if rng.random() < 0.5 else (1, 0)
            defends = (0, 0)
            if beta_defender:
                defends = (1, 0) if flag_rng.random() < 0.5 else (0, 1)
            eta = (
                scores[order[0]]
                - scores[order[1]]
                + TRUE_BETA_FORCE * (x[order[0]] - x[order[1]])
                + beta_defender * (defends[order[0]] - defends[order[1]])
            )
            k = int(rng.choice(5, p=category_probs(eta, TRUE_C1, TRUE_C2)))
            report_on_a = rng.random() < 0.7
            for slot, which in enumerate(order):
                gens = sides_gen[which]
                own = k if slot == 0 else 4 - k
                text = LEVELS[own] if (slot == 0) == report_on_a else None
                log_troops[ids[slot]] = float(x[which])
                sides.append(
                    SideInput(
                        side_id=ids[slot],
                        battle_id=battle_id,
                        year=-300 + battle_id if era == 0 else 1800 + battle_id,
                        outcome=text,
                        is_defender=bool(defends[which]) if beta_defender else None,
                        commanders=tuple(
                            CommanderInput(g, w, "rule_roles")
                            for g, w in zip(gens, weights_of[which], strict=True)
                        ),
                    )
                )
    order_rng = np.random.default_rng(seed + 1)
    shuffled = [sides[i] for i in order_rng.permutation(len(sides))]
    return SimTournament(shuffled, log_troops, true_skill, era_of)
