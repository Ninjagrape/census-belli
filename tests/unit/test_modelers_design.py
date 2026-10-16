"""Design construction: outcome rules, classification, exclusions, orientation."""

from __future__ import annotations

import math
import random
from dataclasses import replace

import numpy as np
import pytest

from pipeline.modelers.design import build_design, outcome_interval, score_of_interval, x_difference
from pipeline.modelers.ordinal import linear_predictor, log_likelihood
from pipeline.modelers.records import (
    KIND_NAMED,
    KIND_UNATTRIBUTED,
    KIND_UNKNOWN,
    DesignParams,
    SideInput,
)
from tests.unit.modelers_helpers import battle, side


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (4, None, (4, 4)),
        (None, 0, (4, 4)),
        (None, 3, (1, 1)),
        (3, 1, (3, 3)),
        (3, 0, (3, 4)),
        (4, 0, (4, 4)),
        (2, 2, (2, 2)),
        (1, 3, (1, 1)),
        (0, 3, (0, 1)),
        (3, 3, "conflict"),
        (1, 1, "conflict"),
        (2, 1, "conflict"),
        (2, 3, "conflict"),
        (None, None, "no_outcome"),
    ],
)
def test_outcome_interval(a: int | None, b: int | None, expected: object) -> None:
    assert outcome_interval(a, b) == expected


def test_score_of_interval() -> None:
    assert score_of_interval(3, 4) == 1.0
    assert score_of_interval(2, 2) == 0.5
    assert score_of_interval(0, 1) == 0.0


def test_text_mapping_and_pyrrhic_is_a_win() -> None:
    sides = battle(1, (1,), (2,), a_outcome="pyrrhic_victory", b_outcome=None)
    design = build_design(sides)
    assert (design.k_lo[0], design.k_hi[0]) == (3, 3)
    sides = battle(2, (1,), (2,), a_outcome=None, b_outcome="decisive_victory")
    assert (build_design(sides).k_lo[0]) == 0


def test_conflict_counted_not_included() -> None:
    sides = battle(1, (1,), (2,), a_outcome="victory", b_outcome="victory")
    design = build_design(sides)
    assert design.n_battles == 0
    assert design.counts.battles_outcome_conflict == 1


def test_no_outcome_counted() -> None:
    design = build_design(battle(1, (1,), (2,), a_outcome=None, b_outcome=None))
    assert design.counts.battles_no_outcome == 1


def test_grade_disagreement_becomes_an_interval() -> None:
    design = build_design(battle(1, (1,), (2,), a_outcome="victory", b_outcome="decisive_defeat"))
    assert (design.k_lo[0], design.k_hi[0]) == (3, 4)
    assert design.y_a[0] == 1.0
    assert design.counts.battles_grade_disagree == 1


def test_unrecognised_outcome_is_missing_and_counted() -> None:
    design = build_design(battle(1, (1,), (2,), a_outcome="draw?", b_outcome="defeat"))
    assert design.counts.outcomes_unrecognised == 1
    assert (design.k_lo[0], design.k_hi[0]) == (3, 3)


def test_multi_side_battle_excluded_with_per_general_tally() -> None:
    sides = [side(i, 7, (i + 10,)) for i in range(3)] + battle(1, (10,), (20,))
    design = build_design(sides)
    assert design.counts.battles_multi_side == 1
    assert design.n_battles == 1
    index = list(design.general_ids).index(10)
    assert design.n_excluded_multi_side[index] == 1
    assert design.n_uncredited_battles[index] == 1


def test_orientation_is_lower_side_id() -> None:
    sides = [
        side(9, 1, (2,), outcome="defeat"),
        side(4, 1, (1,), outcome="victory"),
    ]
    design = build_design(sides)
    assert list(design.side_ids) == [4, 9]
    assert design.k_lo[0] == 3


def test_weights_renormalised_with_deviation_recorded() -> None:
    sides = [
        side(0, 1, (1, 2), weights=(0.3, 0.3), method="rule_roles"),
        side(1, 1, (3,), outcome="defeat"),
    ]
    design = build_design(sides)
    assert design.counts.max_weight_deviation == pytest.approx(0.4)
    assert design.cmd_weight[:2].tolist() == pytest.approx([0.5, 0.5])


def test_unknown_and_unattributed_classification() -> None:
    sides = (
        battle(1, (), (1,))  # A unknown
        + battle(2, (5, 6), (2,), method="default_split")  # A unattributed
        + battle(3, (5, 6), (3,), method="rule_roles")  # all named
    )
    design = build_design(sides)
    kinds = design.side_kind.tolist()
    assert kinds[:2] == [KIND_UNKNOWN, KIND_NAMED]
    assert kinds[2:4] == [KIND_UNATTRIBUTED, KIND_NAMED]
    assert kinds[4:] == [KIND_NAMED, KIND_NAMED]
    assert design.counts.sides_unknown == 1
    assert design.counts.sides_unattributed == 1
    assert design.counts.sides_multi_commander_all == 2
    assert design.counts.sides_default_split_all == 1
    # Generals 5 and 6 are credited in battle 3 and uncredited in battle 2.
    idx5 = list(design.general_ids).index(5)
    assert design.n_included_battles[idx5] == 1
    assert design.n_uncredited_battles[idx5] == 1


def test_single_commander_default_split_is_named() -> None:
    design = build_design(battle(1, (1,), (2,), method="default_split"))
    assert design.side_kind.tolist() == [KIND_NAMED, KIND_NAMED]


def test_no_creditable_side_excluded() -> None:
    design = build_design(battle(1, (), (5, 6), method="default_split"))
    assert design.n_battles == 0
    assert design.counts.battles_no_creditable_side == 1


def test_general_on_both_sides_excluded() -> None:
    design = build_design(battle(1, (1, 2), (2, 3)))
    assert design.counts.battles_general_both_sides == 1
    assert design.n_battles == 0


def test_defender_diff_fills_the_complement_of_a_lone_stated_flag() -> None:
    known = battle(1, (1,), (2,))
    known = [replace(known[0], is_defender=True), replace(known[1], is_defender=False)]
    half = battle(2, (1,), (3,))
    half[0] = replace(half[0], is_defender=True)
    design = build_design(known + half, DesignParams(min_defender_battles=1))
    assert design.defender_diff.tolist() == [1.0, 1.0]
    assert design.defender_known.tolist() == [True, True]
    assert design.counts.defender_pairs_both_stated == 1
    assert design.counts.defender_pairs_filled == 1
    assert design.defender_term_used


def test_defender_diff_is_zero_when_neither_flag_is_stated() -> None:
    design = build_design(battle(1, (1,), (2,)), DesignParams(min_defender_battles=1))
    assert design.defender_diff.tolist() == [0.0]
    assert not design.defender_term_used


def test_defender_term_unused_when_all_null() -> None:
    assert not build_design(battle(1, (1,), (2,))).defender_term_used


def test_x_difference_and_missing_side_raises() -> None:
    design = build_design(battle(1, (1,), (2,)))
    assert x_difference(design, {2: 5.0, 3: 3.5})[0] == pytest.approx(1.5)
    with pytest.raises(ValueError, match="missing"):
        x_difference(design, {2: 5.0})


def _random_raw_battles(seed: int) -> list[tuple[SideInput, SideInput]]:
    rng = random.Random(seed)
    levels = ["decisive_defeat", "defeat", "indecisive", "victory", "decisive_victory"]
    pairs = []
    for b in range(1, 80):
        a_gen = tuple(rng.sample(range(1, 15), rng.choice([1, 1, 2])))
        b_gen = tuple(rng.sample(range(15, 30), rng.choice([1, 1, 2])))
        k = rng.randrange(5)
        style = rng.random()
        a_out: str | None = levels[k]
        b_out: str | None = levels[4 - k]
        if style < 0.25:
            b_out = None
        elif style < 0.5:
            a_out = None
        method = "rule_roles" if len(a_gen) > 1 else "rule_single"
        pairs.append(
            (
                side(0, b, a_gen, outcome=a_out, method=method),
                side(0, b, () if rng.random() < 0.1 else b_gen, outcome=b_out, method=method),
            )
        )
    return pairs


def _loglik(
    sides: list[SideInput],
    skill: dict[int, float],
    x_by_tag: dict[tuple[int, int], float],
    tag_of_side: dict[int, tuple[int, int]],
    latent_by_tag: dict[tuple[int, int], float],
) -> float:
    design = build_design(sides)
    x_by_id = {sid: x_by_tag[tag] for sid, tag in tag_of_side.items()}
    scores = np.zeros(design.n_sides)
    for i, sid in enumerate(design.side_ids):
        if design.side_kind[i] != KIND_NAMED:
            scores[i] = latent_by_tag[tag_of_side[int(sid)]]
    for s, g, w in zip(design.cmd_side, design.cmd_general, design.cmd_weight, strict=True):
        scores[s] += w * skill[int(design.general_ids[g])]
    eta = linear_predictor(
        design.side_a,
        design.side_b,
        scores,
        x_difference(design, x_by_id),
        design.defender_diff,
        0.6,
        0.0,
    )
    return log_likelihood(eta, design.k_lo, design.k_hi, 0.7, 2.1)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_loglik_is_invariant_to_side_order_and_labelling(seed: int) -> None:
    """R5: outcome assembly from raw rows must not depend on which side is A."""
    rng = random.Random(seed)
    pairs = _random_raw_battles(seed)
    skill = {g: rng.gauss(0, 1) for g in range(1, 30)}
    x_by_tag = {(i, j): rng.gauss(0, 1) for i in range(len(pairs)) for j in (0, 1)}
    latent = {tag: rng.gauss(0, 1) for tag in x_by_tag}

    def assemble(
        flip: set[int], order_seed: int
    ) -> tuple[list[SideInput], dict[int, tuple[int, int]]]:
        rows: list[SideInput] = []
        tags: dict[int, tuple[int, int]] = {}
        for i, pair in enumerate(pairs):
            ids = (2 * i, 2 * i + 1) if i not in flip else (2 * i + 1, 2 * i)
            for j in (0, 1):
                rows.append(replace(pair[j], side_id=ids[j]))
                tags[ids[j]] = (i, j)
        random.Random(order_seed).shuffle(rows)
        return rows, tags

    base_rows, base_tags = assemble(set(), 0)
    flip = {i for i in range(len(pairs)) if rng.random() < 0.5}
    alt_rows, alt_tags = assemble(flip, 99)
    base = _loglik(base_rows, skill, x_by_tag, base_tags, latent)
    alt = _loglik(alt_rows, skill, x_by_tag, alt_tags, latent)
    assert math.isfinite(base)
    assert alt == pytest.approx(base, rel=1e-12)


def test_commanders_on_a_zero_weight_side_are_counted_uncredited() -> None:
    from pipeline.modelers.records import CommanderInput, SideInput

    zero = SideInput(
        side_id=0,
        battle_id=1,
        year=1800,
        outcome="victory",
        is_defender=None,
        commanders=(CommanderInput(9, 0.0, "rule_single"),),
    )
    named = side(1, 1, (2,), outcome="defeat")

    design = build_design([zero, named])

    assert design.counts.sides_unknown == 1
    assert design.general_ids.tolist() == [2]
    # General 9 is on the unknown side only: tallied through the tally dict,
    # which has no array slot because 9 is never credited anywhere.
    other = build_design([zero, named, *battle(2, (9,), (2,))])
    index = other.general_ids.tolist().index(9)
    assert other.n_uncredited_battles[index] == 1
    assert other.n_included_battles[index] == 1
