"""Aligned arrays for the imputation model.

Every side in scope gets a row in the all-sides arrays. The observed sides
(those with a likelihood) are additionally laid out flat, in battle order, so
the model can compute one linear-predictor vector over them and then slice it
per battle group.

Random-effect blocks (polity, commander) are sampled only for levels with at
least one observed side. A level seen only on missing sides has no data behind
it, so its effect is a draw from the prior; those draws are made afterwards,
once per level per round, in :mod:`pipeline.imputers.conditional`. Sampling
them would add dimensions the sampler has nothing to learn about.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from pipeline.imputers.priors import ImputePriors, era_of_year
from pipeline.imputers.records import UNKNOWN, BattleLikelihood, SideRow

__all__ = [
    "BattleBlock",
    "CategoricalBlock",
    "ImputeDesign",
    "ObservedGroup",
    "RandomBlock",
    "build_design",
]


@dataclass(frozen=True)
class CategoricalBlock:
    """A ZeroSumNormal block over a small fixed vocabulary.

    Attributes:
        name: Block name, e.g. ``'outcome'``.
        levels: The vocabulary, sorted, over every in-scope side.
        index: Per side, the level's position.
        sd: ZeroSumNormal sd.
    """

    name: str
    levels: tuple[str, ...]
    index: np.ndarray
    sd: float

    @property
    def is_active(self) -> bool:
        """Whether the block can carry any signal (two or more levels)."""
        return len(self.levels) >= 2


@dataclass(frozen=True)
class RandomBlock:
    """A non-centred random-effect block over a large vocabulary.

    Attributes:
        name: Block name, e.g. ``'polity'``.
        levels: Full vocabulary over every in-scope side.
        sampled: Levels with at least one observed side; the model dimension.
        index: Per side, position in ``sampled`` or -1 when the side's level
            is not sampled.
        full_index: Per side, position in ``levels``.
    """

    name: str
    levels: tuple[str, ...]
    sampled: tuple[str, ...]
    index: np.ndarray
    full_index: np.ndarray


@dataclass(frozen=True)
class BattleBlock:
    """One battle's sides and likelihood, for the conditional draws.

    Attributes:
        battle_id: The battle.
        rows: Row indices of all its in-scope sides in the all-sides arrays.
        obs_local: Positions within ``rows`` of the sides with a likelihood.
        m_lik: Likelihood mean for those sides.
        precision: Likelihood precision for those sides.
    """

    battle_id: int
    rows: np.ndarray
    obs_local: np.ndarray
    m_lik: np.ndarray
    precision: np.ndarray


@dataclass(frozen=True)
class ObservedGroup:
    """Battles with the same number ``k`` of observed sides, batched.

    Attributes:
        k: Observed sides per battle.
        positions: ``(n, k)`` indices into the flat observed arrays.
        y: ``(n, k)`` likelihood means.
        cov: ``(n, k, k)`` measurement covariance (inverse likelihood precision).
    """

    k: int
    positions: np.ndarray
    y: np.ndarray
    cov: np.ndarray


@dataclass(frozen=True)
class ImputeDesign:
    """Everything the model and the conditional draws read.

    Attributes:
        side_ids: Side id per row.
        battle_ids: Battle id per row.
        eras: Era names in model order.
        era_index: Per row, the era's position.
        era_mu: Era prior means.
        era_sd: Era prior sds.
        outcome: Outcome block.
        battle_type: Battle-type block.
        defender: Defender block.
        polity: Polity random effects.
        commander: Commander random effects.
        observed_rows: Row index of each flat observed side.
        battles: Per-battle blocks, every battle with at least one side.
        groups: Observed sides batched by count.
        mnar: Per row, whether the side's troop row is classified mnar.
        hyper: Hyperparameter scales.
    """

    side_ids: np.ndarray
    battle_ids: np.ndarray
    eras: tuple[str, ...]
    era_index: np.ndarray
    era_mu: np.ndarray
    era_sd: np.ndarray
    outcome: CategoricalBlock
    battle_type: CategoricalBlock
    defender: CategoricalBlock
    polity: RandomBlock
    commander: RandomBlock
    observed_rows: np.ndarray
    battles: tuple[BattleBlock, ...]
    groups: tuple[ObservedGroup, ...]
    mnar: np.ndarray
    hyper: dict[str, float]

    @property
    def n_sides(self) -> int:
        """Number of in-scope sides."""
        return int(len(self.side_ids))

    @property
    def n_observed(self) -> int:
        """Number of sides with a likelihood."""
        return int(len(self.observed_rows))

    @property
    def is_observed(self) -> np.ndarray:
        """Boolean per row: whether the side has a likelihood."""
        mask = np.zeros(self.n_sides, dtype=bool)
        mask[self.observed_rows] = True
        return mask


def _categorical(name: str, values: Sequence[str], sd: float) -> CategoricalBlock:
    """Index a per-side categorical column against its own sorted vocabulary.

    Args:
        name: Block name.
        values: One level string per side.
        sd: ZeroSumNormal sd.

    Returns:
        The block.
    """
    levels = tuple(sorted(set(values)))
    lookup = {level: i for i, level in enumerate(levels)}
    return CategoricalBlock(name, levels, np.array([lookup[v] for v in values], dtype=int), sd)


def _random(name: str, values: Sequence[str], observed_mask: np.ndarray) -> RandomBlock:
    """Index a per-side keyed column, sampling only levels with observed sides.

    Args:
        name: Block name.
        values: One key per side.
        observed_mask: Per side, whether it has a likelihood.

    Returns:
        The block.
    """
    levels = tuple(sorted(set(values)))
    full_lookup = {level: i for i, level in enumerate(levels)}
    sampled = tuple(sorted({v for v, seen in zip(values, observed_mask, strict=True) if seen}))
    sampled_lookup = {level: i for i, level in enumerate(sampled)}
    return RandomBlock(
        name=name,
        levels=levels,
        sampled=sampled,
        index=np.array([sampled_lookup.get(v, -1) for v in values], dtype=int),
        full_index=np.array([full_lookup[v] for v in values], dtype=int),
    )


def _defender_level(value: bool | None) -> str:
    """Spell a nullable boolean as a categorical level."""
    if value is None:
        return UNKNOWN
    return "true" if value else "false"


def build_design(
    sides: Sequence[SideRow],
    likelihoods: Sequence[BattleLikelihood],
    priors: ImputePriors,
) -> ImputeDesign:
    """Align the sides, covariates and likelihoods into model arrays.

    Args:
        sides: Every in-scope side, in a stable order.
        likelihoods: One entry per battle that has observed sides.
        priors: Era bins and hyperparameter scales.

    Returns:
        The design.

    Raises:
        ValueError: If a likelihood names a side that is not in ``sides``.
    """
    row_of_side = {s.side_id: i for i, s in enumerate(sides)}
    observed_mask = np.zeros(len(sides), dtype=bool)
    for lik in likelihoods:
        for side_id in lik.side_ids:
            if side_id not in row_of_side:
                raise ValueError(f"likelihood names side {side_id}, which is not in scope")
            observed_mask[row_of_side[side_id]] = True

    eras = priors.era_names
    era_lookup = {name: i for i, name in enumerate(eras)}
    era_names_per_side = [era_of_year(s.year, priors.era_bins) for s in sides]
    hyper = priors.hyper

    rows_by_battle: dict[int, list[int]] = defaultdict(list)
    for i, side in enumerate(sides):
        rows_by_battle[side.battle_id].append(i)

    battles: list[BattleBlock] = []
    flat_rows: list[int] = []
    flat_pos: dict[int, int] = {}
    lik_by_battle = {lik.battle_id: lik for lik in likelihoods}
    for battle_id in sorted(rows_by_battle):
        rows = np.array(rows_by_battle[battle_id], dtype=int)
        found = lik_by_battle.get(battle_id)
        if found is None:
            battles.append(
                BattleBlock(battle_id, rows, np.empty(0, int), np.empty(0), np.empty((0, 0)))
            )
            continue
        lik = found
        local = {int(side_id): j for j, side_id in enumerate(sides[r].side_id for r in rows)}
        obs_local = np.array([local[sid] for sid in lik.side_ids], dtype=int)
        for j in obs_local:
            flat_pos[int(rows[j])] = len(flat_rows)
            flat_rows.append(int(rows[j]))
        battles.append(BattleBlock(battle_id, rows, obs_local, lik.m_lik, lik.precision))

    grouped: dict[int, list[BattleBlock]] = defaultdict(list)
    for block in battles:
        if len(block.obs_local):
            grouped[len(block.obs_local)].append(block)
    groups = []
    for k in sorted(grouped):
        members = grouped[k]
        groups.append(
            ObservedGroup(
                k=k,
                positions=np.array(
                    [[flat_pos[int(b.rows[j])] for j in b.obs_local] for b in members], dtype=int
                ),
                y=np.array([b.m_lik for b in members], dtype=float),
                cov=np.array([np.linalg.inv(b.precision) for b in members], dtype=float),
            )
        )

    return ImputeDesign(
        side_ids=np.array([s.side_id for s in sides], dtype=np.int64),
        battle_ids=np.array([s.battle_id for s in sides], dtype=np.int64),
        eras=eras,
        era_index=np.array([era_lookup[n] for n in era_names_per_side], dtype=int),
        era_mu=np.array([priors.era_priors[n].log_mu for n in eras], dtype=float),
        era_sd=np.array([priors.era_priors[n].log_sd for n in eras], dtype=float),
        outcome=_categorical(
            "outcome", [s.outcome or UNKNOWN for s in sides], hyper.get("zsn_outcome", 0.5)
        ),
        battle_type=_categorical(
            "type", [s.battle_type or UNKNOWN for s in sides], hyper.get("zsn_type", 0.5)
        ),
        defender=_categorical(
            "defender",
            [_defender_level(s.is_defender) for s in sides],
            hyper.get("zsn_defender", 0.3),
        ),
        polity=_random("polity", [s.polity_key for s in sides], observed_mask),
        commander=_random("commander", [s.commander_key for s in sides], observed_mask),
        observed_rows=np.array(flat_rows, dtype=int),
        battles=tuple(battles),
        groups=tuple(groups),
        mnar=np.array([s.mnar for s in sides], dtype=bool),
        hyper=dict(hyper),
    )
