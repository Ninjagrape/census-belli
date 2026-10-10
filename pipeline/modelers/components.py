"""The general-general comparison graph: components, bridges, career spans.

Nodes are the credited generals. An edge joins two generals who commanded
opposing named sides of one included battle. Co-commanders on the same side
are not joined: they are never compared with each other, only their weighted
sum is. Unknown and unattributed sides are not nodes and add no edges
(``mu_unknown`` and ``mu_unattributed`` are a soft link between components
that this graph deliberately does not count).

Bridges use an iterative Tarjan low-link search written here rather than
networkx, which is not a runtime dependency; the unit tests cross-check it
against networkx, a dev dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pipeline.modelers.records import KIND_NAMED, ModelDesign

__all__ = [
    "ComponentResult",
    "career_spans",
    "comparison_edges",
    "components",
    "find_bridges",
]


@dataclass(frozen=True)
class ComponentResult:
    """Connected components of the comparison graph.

    Attributes:
        component_id: Component per general index, numbered from 0 by
            descending size (ties by lowest general index).
        sizes: Number of generals per component, same numbering.
        largest_share: Fraction of generals in the largest component (0 when
            there are no generals).
    """

    component_id: np.ndarray
    sizes: np.ndarray
    largest_share: float

    @property
    def n_components(self) -> int:
        """Number of components."""
        return int(self.sizes.shape[0])


def comparison_edges(design: ModelDesign) -> list[tuple[int, int]]:
    """Distinct opposing-general pairs, as general-index pairs ``(u, v)`` with ``u < v``.

    Args:
        design: The design.

    Returns:
        Sorted list of unique edges.
    """
    members: dict[int, list[int]] = {}
    for side, general in zip(design.cmd_side, design.cmd_general, strict=True):
        members.setdefault(int(side), []).append(int(general))
    edges: set[tuple[int, int]] = set()
    for a, b in zip(design.side_a, design.side_b, strict=True):
        if design.side_kind[a] != KIND_NAMED or design.side_kind[b] != KIND_NAMED:
            continue
        for u in members.get(int(a), []):
            for v in members.get(int(b), []):
                edges.add((min(u, v), max(u, v)))
    return sorted(edges)


def _adjacency(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    adjacency: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        adjacency[u].append(v)
        adjacency[v].append(u)
    return adjacency


def components(design: ModelDesign) -> ComponentResult:
    """Label the connected components of the comparison graph.

    Args:
        design: The design.

    Returns:
        Per-general component ids and sizes.
    """
    n = design.n_generals
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for u, v in comparison_edges(design):
        root_u, root_v = find(u), find(v)
        if root_u != root_v:
            parent[max(root_u, root_v)] = min(root_u, root_v)

    roots = np.array([find(i) for i in range(n)], dtype=np.int64)
    unique_roots, first_index, counts = np.unique(roots, return_index=True, return_counts=True)
    order = np.lexsort((first_index, -counts))
    relabel = {int(unique_roots[i]): rank for rank, i in enumerate(order)}
    component_id = np.array([relabel[int(r)] for r in roots], dtype=np.int64)
    sizes = counts[order].astype(np.int64)
    share = float(sizes[0] / n) if n else 0.0
    return ComponentResult(component_id=component_id, sizes=sizes, largest_share=share)


def find_bridges(n_nodes: int, edges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Edges whose removal disconnects their component (iterative Tarjan).

    Args:
        n_nodes: Number of nodes, labelled ``0..n_nodes-1``.
        edges: Undirected edges without duplicates or self loops.

    Returns:
        Bridges as sorted ``(u, v)`` pairs with ``u < v``.
    """
    adjacency = _adjacency(n_nodes, edges)
    discovery = [-1] * n_nodes
    low = [0] * n_nodes
    bridges: list[tuple[int, int]] = []
    clock = 0
    for root in range(n_nodes):
        if discovery[root] != -1:
            continue
        discovery[root] = low[root] = clock
        clock += 1
        # Each frame: (node, parent, next neighbour position).
        stack: list[tuple[int, int, int]] = [(root, -1, 0)]
        while stack:
            node, parent, pos = stack.pop()
            if pos < len(adjacency[node]):
                stack.append((node, parent, pos + 1))
                nxt = adjacency[node][pos]
                if nxt == parent:
                    continue
                if discovery[nxt] == -1:
                    discovery[nxt] = low[nxt] = clock
                    clock += 1
                    stack.append((nxt, node, 0))
                else:
                    low[node] = min(low[node], discovery[nxt])
            elif parent != -1:
                low[parent] = min(low[parent], low[node])
                if low[node] > discovery[parent]:
                    bridges.append((min(node, parent), max(node, parent)))
    return sorted(bridges)


def career_spans(design: ModelDesign) -> np.ndarray:
    """First year, last year and span of each general's credited battles.

    Args:
        design: The design.

    Returns:
        Array of shape ``(n_generals, 3)`` with columns ``first``, ``last`` and
        ``span``; NaN where a general has no dated battle.
    """
    out = np.full((design.n_generals, 3), np.nan)
    battle_of_side = np.repeat(np.arange(design.n_battles), 2)
    for general in range(design.n_generals):
        sides = design.cmd_side[design.cmd_general == general]
        years = design.year[battle_of_side[sides]]
        years = years[~np.isnan(years)]
        if years.size:
            out[general] = (years.min(), years.max(), years.max() - years.min())
    return out
