"""Comparison graph components, bridges and career spans."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.modelers.components import career_spans, comparison_edges, components, find_bridges
from pipeline.modelers.design import build_design
from tests.unit.modelers_helpers import battle


def test_two_components_and_latent_sides_are_not_nodes() -> None:
    sides = (
        battle(1, (1,), (2,), year=-300)
        + battle(2, (2,), (3,), year=-290)
        + battle(3, (10,), (11,), year=1800)
        + battle(4, (), (12,), year=1805)  # unknown opponent: 12 is isolated
    )
    design = build_design(sides)
    result = components(design)
    ids = list(design.general_ids)
    by_general = {ids[i]: int(c) for i, c in enumerate(result.component_id)}
    assert by_general[1] == by_general[2] == by_general[3] == 0
    assert by_general[10] == by_general[11] == 1
    assert by_general[12] == 2
    assert result.n_components == 3
    assert result.sizes.tolist() == [3, 2, 1]
    assert result.largest_share == 0.5


def test_cogeneral_sides_join_all_opponents_only() -> None:
    design = build_design(battle(1, (1, 2), (3,), method="rule_roles"))
    assert comparison_edges(design) == [(0, 2), (1, 2)]


def test_bridges_hand_built_graph() -> None:
    # Triangle 0-1-2, bridge 2-3, square 3-4-5-6, pendant 6-7, separate edge 8-9.
    edges = [(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (3, 6), (6, 7), (8, 9)]
    assert find_bridges(10, edges) == [(2, 3), (6, 7), (8, 9)]


def test_bridges_agree_with_networkx() -> None:
    nx = pytest.importorskip("networkx")
    rng = np.random.default_rng(3)
    for _ in range(20):
        edges = sorted(
            {(int(min(u, v)), int(max(u, v))) for u, v in rng.integers(0, 12, (14, 2)) if u != v}
        )
        expected = sorted((min(u, v), max(u, v)) for u, v in nx.bridges(nx.Graph(edges)))
        assert find_bridges(12, edges) == expected


def test_career_span_ignores_undated_battles() -> None:
    sides = (
        battle(1, (1,), (2,), year=1800)
        + battle(2, (1,), (3,), year=1870)
        + battle(3, (1,), (4,), year=None)
    )
    design = build_design(sides)
    spans = career_spans(design)
    i = list(design.general_ids).index(1)
    assert spans[i].tolist() == [1800.0, 1870.0, 70.0]
    j = list(design.general_ids).index(4)
    assert np.isnan(spans[j]).all()
