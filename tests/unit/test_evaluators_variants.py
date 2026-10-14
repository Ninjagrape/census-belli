"""Sensitivity variants: parsing, side and battle transforms, design transforms."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pipeline.evaluators.variants import (
    VariantSpec,
    apply_battle_filter,
    apply_side_transform,
    collapse_three_level,
    designs_equivalent,
    observed_only_x_diff,
    parse_variants,
    prior_overrides,
)
from pipeline.modelers.design import build_design
from pipeline.modelers.priors import ModelPriors
from pipeline.modelers.records import (
    DEFAULT_SPLIT,
    KIND_NAMED,
    KIND_UNATTRIBUTED,
    CommanderInput,
    SideInput,
)
from tests.unit.modelers_helpers import battle, side


def _spec(name: str, kind: str, **options: object) -> VariantSpec:
    return parse_variants([{"name": name, "kind": kind, "description": "d", **options}])[0]


def _with_roles(s: SideInput, roles: tuple[str | None, ...]) -> SideInput:
    cmds = tuple(replace(c, command_role=r) for c, r in zip(s.commanders, roles, strict=True))
    return replace(s, commanders=cmds)


def test_parse_fills_defaults() -> None:
    specs = parse_variants(
        [
            {"name": "field_commanders_only", "kind": "side_transform", "description": "x"},
            {"name": "drop_low_quality", "kind": "battle_filter", "description": "x"},
            {"name": "weak_hyperpriors", "kind": "priors", "description": "x"},
            {"name": "replacement_50th", "kind": "post_hoc", "description": "x"},
            {"name": "realised_war", "kind": "post_hoc", "description": "x"},
        ]
    )
    assert specs[0].options["exclude_roles"] == ["sovereign", "nominal"]
    assert specs[1].options["min_quality"] == 0.3
    assert specs[2].options == {"skill_prior_sd": 5.0, "beta_force_sd": 2.0}
    assert specs[3].options["replacement_percentile"] == 50
    assert specs[4].options["rank_by"] == "realised"


@pytest.mark.parametrize(
    "entry",
    [
        {"kind": "post_hoc"},
        {"name": "x", "kind": "nonsense"},
        {"name": "equal_attribution", "kind": "battle_filter"},
        {"name": "equal_attribution", "kind": "side_transform", "bogus": 1},
        {"name": "mystery", "kind": "side_transform"},
        {"name": "mystery", "kind": "post_hoc", "a": 1},
        {"name": "drop_low_quality", "kind": "battle_filter", "min_quality": 2.0},
        {"name": "exclude_extreme_ratio", "kind": "battle_filter", "max_abs_log_ratio": 0},
        {"name": "weak_hyperpriors", "kind": "priors", "not_a_field": 1.0},
        {"name": "realised_war", "kind": "post_hoc", "rank_by": "other"},
    ],
)
def test_parse_rejects(entry: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        parse_variants([entry])


def test_parse_rejects_duplicates() -> None:
    entry = {"name": "equal_attribution", "kind": "side_transform"}
    with pytest.raises(ValueError, match="duplicate"):
        parse_variants([entry, entry])


def test_parse_allows_new_name_with_sufficient_options() -> None:
    spec = _spec("only_generals", "side_transform", exclude_roles=["subordinate"])
    assert spec.options["exclude_roles"] == ["subordinate"]
    assert _spec("tight", "battle_filter", min_quality=0.9).options["min_quality"] == 0.9


def test_equal_attribution_credits_default_split_side() -> None:
    sides = [
        side(2, 1, (10, 11), method=DEFAULT_SPLIT, weights=(0.9, 0.1)),
        side(3, 1, (20,), outcome="defeat"),
    ]
    base = build_design(sides)
    assert base.side_kind[0] == KIND_UNATTRIBUTED
    new, counts = apply_side_transform(_spec("equal_attribution", "side_transform"), sides)
    design = build_design(new)
    assert design.side_kind[0] == KIND_NAMED
    assert design.cmd_weight[design.cmd_side == 0].tolist() == [0.5, 0.5]
    assert {c.attribution_method for c in new[0].commanders} == {"variant_equal"}
    assert counts["sides_changed"] >= 1
    assert sides[0].commanders[0].attribution_weight == 0.9


def test_equal_attribution_dedupes_repeated_general() -> None:
    s = SideInput(
        1,
        1,
        1800,
        "victory",
        None,
        (
            CommanderInput(5, 0.7, "llm"),
            CommanderInput(5, 0.3, "llm"),
            CommanderInput(6, 0.1, "llm"),
        ),
    )
    (new,), _ = apply_side_transform(_spec("equal_attribution", "side_transform"), [s])
    assert [(c.general_id, c.attribution_weight) for c in new.commanders] == [(5, 0.5), (6, 0.5)]


def test_field_commanders_only_drops_roles_and_never_empties() -> None:
    kept_side = _with_roles(side(0, 1, (1, 2)), ("sovereign", "field_commander"))
    all_out = _with_roles(side(1, 1, (3, 4), outcome="defeat"), ("nominal", "sovereign"))
    spec = _spec("field_commanders_only", "side_transform")
    new, counts = apply_side_transform(spec, [kept_side, all_out])
    assert [c.general_id for c in new[0].commanders] == [2]
    assert new[0].commanders[0].attribution_weight == 0.5
    assert new[1] == all_out
    assert counts == {
        "sides_changed": 1,
        "commanders_dropped": 1,
        "sides_kept_unchanged": 1,
        "sides_reweighted": 0,
    }


def test_field_commanders_only_reweights_equally_when_kept_weights_are_zero() -> None:
    zero = _with_roles(side(0, 1, (1, 2, 3)), ("sovereign", "field_commander", "field_commander"))
    zero = replace(
        zero,
        commanders=tuple(replace(c, attribution_weight=0.0) for c in zero.commanders),
    )
    positive = _with_roles(side(1, 1, (4, 5), outcome="defeat"), ("sovereign", "field_commander"))
    spec = _spec("field_commanders_only", "side_transform")

    (new_zero, new_positive), counts = apply_side_transform(spec, [zero, positive])

    assert [(c.general_id, c.attribution_weight) for c in new_zero.commanders] == [
        (2, 0.5),
        (3, 0.5),
    ]
    assert new_positive.commanders[0].general_id == 5
    assert counts["sides_reweighted"] == 1


def test_field_commanders_only_keeps_a_commander_without_a_role() -> None:
    plain = side(0, 1, (1, 2))  # command_role None on both
    (new,), counts = apply_side_transform(_spec("field_commanders_only", "side_transform"), [plain])
    assert new == plain
    assert counts["commanders_dropped"] == 0


def test_field_commanders_only_noop_without_roles() -> None:
    sides = battle(1, (1,), (2,))
    new, counts = apply_side_transform(_spec("field_commanders_only", "side_transform"), sides)
    assert new == sides
    assert counts["sides_changed"] == 0
    assert designs_equivalent(build_design(sides), build_design(new))


def test_side_transform_rejects_wrong_kind() -> None:
    with pytest.raises(ValueError):
        apply_side_transform(_spec("drop_low_quality", "battle_filter"), [])


def test_drop_low_quality_keeps_unscored() -> None:
    sides = battle(1, (1,), (2,)) + battle(2, (3,), (4,)) + battle(3, (5,), (6,))
    spec = _spec("drop_low_quality", "battle_filter")
    kept, counts = apply_battle_filter(spec, sides, battle_quality={1: 0.1, 2: 0.3})
    assert {s.battle_id for s in kept} == {2, 3}
    assert counts == {"battles_dropped": 1, "sides_dropped": 2}


def test_drop_low_quality_noop_when_all_equal() -> None:
    sides = battle(1, (1,), (2,))
    kept, counts = apply_battle_filter(
        _spec("drop_low_quality", "battle_filter"), sides, battle_quality={1: 0.5}
    )
    assert kept == sides
    assert counts["battles_dropped"] == 0


def test_exclude_extreme_ratio() -> None:
    sides = battle(1, (1,), (2,)) + battle(2, (3,), (4,))
    spec = _spec("exclude_extreme_ratio", "battle_filter")
    kept, counts = apply_battle_filter(spec, sides, measured_x_diff={1: -2.5, 2: 2.0})
    assert {s.battle_id for s in kept} == {2}
    assert counts["battles_dropped"] == 1


def test_battle_filter_requires_its_input() -> None:
    with pytest.raises(ValueError, match="battle_quality"):
        apply_battle_filter(_spec("drop_low_quality", "battle_filter"), [])
    with pytest.raises(ValueError, match="measured_x_diff"):
        apply_battle_filter(_spec("exclude_extreme_ratio", "battle_filter"), [])


def test_three_level_on_exact_and_interval_outcomes() -> None:
    sides = (
        battle(1, (1,), (2,), a_outcome="decisive_victory", b_outcome=None)
        + battle(2, (3,), (4,), a_outcome="defeat", b_outcome=None)
        + battle(3, (5,), (6,), a_outcome="indecisive", b_outcome=None)
        + battle(4, (7,), (8,), a_outcome="victory", b_outcome="decisive_defeat")
        + battle(5, (9,), (10,), a_outcome="decisive_defeat", b_outcome="victory")
    )
    design = build_design(sides)
    out = collapse_three_level(design)
    assert design.k_lo.tolist() == [4, 1, 2, 3, 0]
    assert out.k_lo.tolist() == [3, 0, 2, 3, 0]
    assert out.k_hi.tolist() == [4, 1, 2, 4, 1]
    assert np.array_equal(out.y_a, design.y_a)
    assert not designs_equivalent(design, out)


def test_observed_only_needs_both_sides_measured() -> None:
    design = build_design(battle(1, (1,), (2,)) + battle(2, (3,), (4,)) + battle(3, (5,), (6,)))
    obs_mean = np.array([5.0, 3.0, 4.0, np.nan, 6.0, 1.0])
    obs_sd = np.array([0.2, 0.2, 0.2, 0.2, np.nan, 0.2])
    assert observed_only_x_diff(design, obs_mean, obs_sd) == {1: 2.0}


def test_prior_overrides() -> None:
    base = ModelPriors()
    out = prior_overrides(base, {"skill_prior_sd": 5.0, "beta_force_sd": 2})
    assert (out.skill_prior_sd, out.beta_force_sd) == (5.0, 2.0)
    assert base.skill_prior_sd == 1.0
    assert out.beta_def_sd == base.beta_def_sd
    with pytest.raises(ValueError):
        prior_overrides(base, {"nope": 1.0})


def test_designs_equivalent_identical_and_changed() -> None:
    sides = battle(1, (1, 2), (3,)) + battle(2, (4,), (5,))
    a = build_design(sides)
    assert designs_equivalent(a, build_design(list(reversed(sides))))
    assert not designs_equivalent(a, build_design(sides[:2]))
    reweighted = [
        replace(
            sides[0],
            commanders=tuple(
                replace(c, attribution_weight=w)
                for c, w in zip(sides[0].commanders, (0.8, 0.2), strict=True)
            ),
        ),
        *sides[1:],
    ]
    assert not designs_equivalent(a, build_design(reweighted))
