"""Design terms: campaign exclusion, defender threshold, force groups, two-camp merge."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from pipeline.modelers.design import (
    build_design,
    camp_measurements,
    side_log_troops,
    x_difference,
)
from pipeline.modelers.records import KIND_NAMED, KIND_UNATTRIBUTED, KIND_UNKNOWN, DesignParams
from tests.unit.modelers_helpers import battle, side


def _flagged(battle_id: int, a: tuple[int, ...], b: tuple[int, ...], **kwargs: object) -> list:
    """A battle where side A defends and side B attacks."""
    sides = battle(battle_id, a, b, **kwargs)
    return [replace(sides[0], is_defender=True), replace(sides[1], is_defender=False)]


def test_campaign_battle_is_excluded_and_counted_separately() -> None:
    sides = battle(1, (1,), (2,), battle_type="campaign") + battle(2, (3,), (4,))
    design = build_design(sides)
    assert design.n_battles == 1
    assert design.counts.battles_campaign == 1
    assert design.counts.battles_multi_side == 0
    assert design.counts.battles_excluded == 1


def test_campaign_with_three_sides_counts_as_campaign_not_multi_side() -> None:
    sides = [side(i, 1, (i,), outcome="victory", battle_type="campaign") for i in (1, 2, 3)]
    counts = build_design(sides).counts
    assert (counts.battles_campaign, counts.battles_multi_side) == (1, 0)


def test_defender_term_absent_below_threshold_and_present_at_it() -> None:
    sides = [s for i in range(1, 10) for s in _flagged(i, (i,), (100 + i,))]
    params = DesignParams(min_defender_battles=10)
    below = build_design(sides, params)
    assert not below.defender_term_used
    assert below.counts.defender_battles_known == 9
    assert not below.defender_diff.any()  # inert: the model builds no coefficient

    at = build_design([*sides, *_flagged(10, (10,), (110,))], params)
    assert at.defender_term_used
    assert at.defender_diff.tolist() == [1.0] * 10


def test_unstated_or_equal_flags_do_not_count_toward_threshold() -> None:
    both_def = [replace(s, is_defender=True) for s in battle(1, (1,), (2,))]
    both_att = [replace(s, is_defender=False) for s in battle(2, (3,), (4,))]
    neither = battle(3, (5,), (6,))
    design = build_design(both_def + both_att + neither, DesignParams(min_defender_battles=1))
    assert design.counts.defender_battles_known == 0
    assert not design.defender_term_used
    assert not design.defender_diff.any()


def test_filled_and_stated_pairs_both_count_toward_threshold() -> None:
    stated = _flagged(1, (1,), (2,))
    lone = battle(2, (3,), (4,))
    lone[1] = replace(lone[1], is_defender=False)
    design = build_design(stated + lone, DesignParams(min_defender_battles=2))
    assert design.counts.defender_pairs_both_stated == 1
    assert design.counts.defender_pairs_filled == 1
    assert design.defender_term_used
    # B is the stated attacker, so A is filled in as the defender.
    assert design.defender_diff.tolist() == [1.0, 1.0]


def test_term_overrides_keep_a_subset_on_the_main_designs_decisions() -> None:
    sides = [s for i in range(1, 4) for s in _flagged(i, (i,), (100 + i,), battle_type="naval")]
    forced_on = build_design(
        sides, DesignParams(defender_term=True, water_air_group=True, min_defender_battles=99)
    )
    assert forced_on.defender_term_used
    assert forced_on.n_force_groups == 2
    forced_off = build_design(
        sides, DesignParams(defender_term=False, water_air_group=False, min_defender_battles=1)
    )
    assert not forced_off.defender_term_used
    assert forced_off.n_force_groups == 1
    assert not forced_off.defender_diff.any()


def test_swapping_sides_flips_defender_and_force_differences() -> None:
    # Same battle with the side ids exchanged: the former A becomes B.
    original = [
        replace(side(2, 1, (1,), outcome="victory"), is_defender=True),
        replace(side(3, 1, (2,), outcome="defeat"), is_defender=False),
    ]
    swapped = [replace(original[0], side_id=3), replace(original[1], side_id=2)]
    params = DesignParams(min_defender_battles=1)
    d1, d2 = build_design(original, params), build_design(swapped, params)
    assert d1.defender_diff[0] == 1.0
    assert d2.defender_diff[0] == -d1.defender_diff[0]
    # Troops travel with the physical side, so they are keyed by the new ids.
    x1 = x_difference(d1, {2: 6.0, 3: 5.0})[0]
    x2 = x_difference(d2, {3: 6.0, 2: 5.0})[0]
    assert (x1, x2) == pytest.approx((1.0, -1.0))
    # The outcome flips with the orientation too: the same facts, mirrored.
    assert (d1.y_a[0], d2.y_a[0]) == (1.0, 0.0)
    assert d1.force_group[0] == d2.force_group[0]


def test_water_air_force_group_needs_enough_battles() -> None:
    naval = [s for i in range(1, 4) for s in battle(i, (i,), (50 + i,), battle_type="naval")]
    land = [s for i in range(10, 13) for s in battle(i, (i,), (50 + i,), battle_type="field")]
    params = DesignParams(min_battles_per_force_group=3)
    two = build_design(naval + land, params)
    assert two.n_force_groups == 2
    assert two.force_group.tolist() == [1, 1, 1, 0, 0, 0]
    one = build_design(naval[:4] + land, params)  # two naval battles only
    assert one.n_force_groups == 1
    assert not one.force_group.any()
    assert one.counts.battles_water_air == 2


def test_aerial_joins_the_water_air_group_and_unknown_type_is_land() -> None:
    sides = (
        battle(1, (1,), (2,), battle_type="aerial")
        + battle(2, (3,), (4,), battle_type="naval")
        + battle(3, (5,), (6,), battle_type=None)
    )
    design = build_design(sides, DesignParams(min_battles_per_force_group=2))
    assert design.force_group.tolist() == [1, 1, 0]


def _three_sided(troops: tuple[float | None, ...] = (None, None, None)) -> list:
    # Sides 1 and 2 won together against side 3.
    return [
        side(1, 1, (11, 12), outcome="victory", weights=(0.5, 0.5), log_troops=troops[0]),
        side(2, 1, (21,), outcome="decisive_victory", log_troops=troops[1]),
        side(3, 1, (31,), outcome="defeat", log_troops=troops[2]),
    ]


def test_two_camp_merge_with_equal_shares_when_troops_missing() -> None:
    design = build_design(_three_sided())
    assert design.counts.battles_merged_camps == 1
    assert design.counts.battles_multi_side == 0
    assert design.n_battles == 1
    assert design.side_members == ((1, 2), (3,))
    assert design.side_ids.tolist() == [1, 3]
    weights = {
        int(design.general_ids[g]): float(w)
        for sd, g, w in zip(design.cmd_side, design.cmd_general, design.cmd_weight, strict=True)
        if sd == 0
    }
    assert {int(g) for g in design.general_ids} == {11, 12, 21, 31}
    assert weights == pytest.approx({11: 0.25, 12: 0.25, 21: 0.5})
    # Grades 3 (victory) and 4 (decisive) from A's side, 4 (decisive) mirrored from B's defeat=1.
    assert (design.k_lo[0], design.k_hi[0]) == (3, 4)


def test_two_camp_merge_weights_follow_troop_shares() -> None:
    logs = (math.log(3000), math.log(1000), None)
    design = build_design(_three_sided(logs))
    weights = {
        int(design.general_ids[g]): float(w)
        for s, g, w in zip(design.cmd_side, design.cmd_general, design.cmd_weight, strict=True)
        if s == 0
    }
    # Member 1 holds 3/4 of the camp, split between its two generals.
    assert weights == pytest.approx({11: 0.375, 12: 0.375, 21: 0.25})
    assert sum(weights.values()) == pytest.approx(1.0)
    assert design.side_unknown_share.sum() == 0.0


def test_merged_camp_troops_are_log_sum_exp_of_members() -> None:
    design = build_design(_three_sided())
    log_troops = {1: math.log(300.0), 2: math.log(100.0), 3: math.log(200.0)}
    x = x_difference(design, log_troops)
    assert x[0] == pytest.approx(math.log(400.0) - math.log(200.0))
    assert side_log_troops(design, log_troops).tolist() == pytest.approx(
        [math.log(400.0), math.log(200.0)]
    )
    with pytest.raises(ValueError, match="missing"):
        x_difference(design, {1: 1.0, 3: 1.0})


def test_camp_measurements_need_every_member_measured() -> None:
    design = build_design(_three_sided())
    full = {1: (math.log(300.0), 0.2), 2: (math.log(100.0), 0.4), 3: (math.log(200.0), 0.1)}
    mean, sd = camp_measurements(design, full)
    assert mean[0] == pytest.approx(math.log(400.0))
    assert sd[0] == pytest.approx(math.sqrt((0.75 * 0.2) ** 2 + (0.25 * 0.4) ** 2))
    assert (mean[1], sd[1]) == pytest.approx((math.log(200.0), 0.1))
    partial_mean, partial_sd = camp_measurements(design, {1: full[1], 3: full[3]})
    assert math.isnan(partial_mean[0]) and math.isnan(partial_sd[0])
    assert partial_mean[1] == pytest.approx(math.log(200.0))


@pytest.mark.parametrize(
    "outcomes",
    [
        ("victory", "indecisive", "defeat"),  # a side with no clear result
        ("victory", None, "defeat"),  # missing outcome
        ("victory", "victory", "victory"),  # no losing camp
        ("defeat", "defeat", "decisive_defeat"),  # no winning camp
    ],
)
def test_non_clean_split_stays_excluded(outcomes: tuple[str | None, ...]) -> None:
    sides = [side(i + 1, 1, (i + 1,), outcome=o) for i, o in enumerate(outcomes)]
    design = build_design(sides)
    assert design.n_battles == 0
    assert design.counts.battles_multi_side == 1
    assert design.counts.battles_merged_camps == 0


def test_camp_kind_unknown_and_unattributed_when_no_member_is_named() -> None:
    named = side(3, 1, (31,), outcome="defeat")
    unattributed = [
        side(1, 1, (11, 12), outcome="victory", method="default_split"),
        side(2, 1, (21, 22), outcome="victory", method="default_split"),
    ]
    design = build_design([*unattributed, named])
    assert design.side_kind.tolist() == [KIND_UNATTRIBUTED, KIND_NAMED]

    mixed = [
        side(1, 1, (11, 12), outcome="victory", method="default_split"),
        side(2, 1, (), outcome="victory"),
    ]
    design = build_design([*mixed, named])
    assert design.side_kind.tolist() == [KIND_UNKNOWN, KIND_NAMED]


def test_camp_keeps_the_unnamed_members_share_as_a_latent_component() -> None:
    sides = [
        side(1, 1, (11,), outcome="victory", log_troops=math.log(500.0)),
        side(2, 1, (), outcome="victory", log_troops=math.log(500.0)),
        side(3, 1, (31,), outcome="defeat"),
    ]
    design = build_design(sides)
    assert design.side_kind.tolist() == [KIND_NAMED, KIND_NAMED]
    assert design.cmd_weight[design.cmd_side == 0].tolist() == pytest.approx([0.5])
    assert design.side_unknown_share.tolist() == pytest.approx([0.5, 0.0])
    assert design.side_unattributed_share.tolist() == [0.0, 0.0]
    assert design.latent_sides.tolist() == [0]


def test_ten_percent_named_and_ninety_percent_unknown_credits_a_tenth() -> None:
    sides = [
        side(1, 1, (11,), outcome="victory", log_troops=math.log(100.0)),
        side(2, 1, (), outcome="victory", log_troops=math.log(900.0)),
        side(3, 1, (31,), outcome="defeat"),
    ]
    design = build_design(sides)
    camp_a = design.cmd_side == 0
    assert design.cmd_weight[camp_a].tolist() == pytest.approx([0.1])
    assert design.side_unknown_share[0] == pytest.approx(0.9)
    idx = int(np.flatnonzero(design.general_ids == 11)[0])
    assert design.effective_battles[idx] == pytest.approx(0.1)
    counts = design.counts
    assert (counts.camps_merged, counts.camps_partial_credit) == (1, 1)
    assert counts.camp_credited_share_min == pytest.approx(0.1)
    assert counts.camp_credited_share_mean == pytest.approx(0.1)


def test_mixed_unnamed_members_split_across_the_two_latent_kinds() -> None:
    sides = [
        side(1, 1, (11,), outcome="victory", log_troops=math.log(200.0)),
        side(2, 1, (), outcome="victory", log_troops=math.log(100.0)),
        side(3, 1, (41, 42), outcome="victory", method="default_split", log_troops=math.log(100.0)),
        side(4, 1, (31,), outcome="defeat"),
    ]
    design = build_design(sides)
    assert design.cmd_weight[design.cmd_side == 0].tolist() == pytest.approx([0.5])
    assert design.side_unknown_share[0] == pytest.approx(0.25)
    assert design.side_unattributed_share[0] == pytest.approx(0.25)


def test_zero_weight_commanders_of_an_ordinary_named_side_are_not_uncredited() -> None:
    sides = [
        side(2, 1, (1, 2), outcome="victory", weights=(1.0, 0.0)),
        side(3, 1, (3,), outcome="defeat"),
    ]
    design = build_design(sides)
    assert design.n_uncredited_battles.tolist() == [0, 0]


def test_campaign_battle_does_not_add_to_uncredited_tally() -> None:
    sides = battle(1, (1,), (2,), battle_type="campaign") + battle(2, (1,), (3,))
    design = build_design(sides)
    idx = int(np.flatnonzero(design.general_ids == 1)[0])
    assert design.n_uncredited_battles[idx] == 0
    assert design.counts.battles_campaign == 1


def test_missing_camp_troops_are_filled_from_the_rounds_else_left_unset() -> None:
    from pipeline.modelers.load import ImputedRound, fill_camp_log_troops

    sides = _three_sided()
    rounds = [
        ImputedRound(0, 1, "neutral", {1: math.log(300.0), 2: math.log(100.0), 3: 1.0}),
        ImputedRound(1, 1, "neutral", {1: math.log(300.0), 2: math.log(100.0), 3: 1.0}),
    ]
    filled = fill_camp_log_troops(sides, rounds)
    assert [s.log_troops for s in filled][:2] == pytest.approx([math.log(300.0), math.log(100.0)])
    design = build_design(filled)
    weights = design.cmd_weight[design.cmd_side == 0].tolist()
    assert sorted(weights) == pytest.approx([0.25, 0.375, 0.375])
    # Nothing to fill from: equal shares.
    assert sum(s.log_troops is None for s in fill_camp_log_troops(sides, [])) == 3


def test_camp_defender_flag_rule() -> None:
    sides = _three_sided()
    sides[1] = replace(sides[1], is_defender=True)
    sides[2] = replace(sides[2], is_defender=False)
    design = build_design(sides, DesignParams(min_defender_battles=1))
    assert design.defender_diff[0] == 1.0

    # One member unknown and none defending: the camp is unknown, so its
    # stated opponent fills it in as the defender.
    unknown_member = [replace(s, is_defender=False) for s in _three_sided()]
    unknown_member[0] = replace(unknown_member[0], is_defender=None)
    filled = build_design(unknown_member, DesignParams(min_defender_battles=1))
    assert filled.counts.defender_pairs_filled == 1
    assert filled.defender_diff[0] == 1.0
    both_unknown = [replace(s, is_defender=None) for s in _three_sided()]
    assert build_design(both_unknown).counts.defender_battles_known == 0
