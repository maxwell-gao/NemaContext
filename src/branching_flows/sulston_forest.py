"""Build coalescent forests from the known Sulston lineage tree.

Instead of BranchingFlows' default random coalescent (which merges random
adjacent pairs), this uses the ACTUAL parent-child relationships from
WormGUIDES to construct biologically correct trees.

The key function ``sulston_sample_forest`` is a drop-in replacement for
``sample_forest`` in ``flow.py``, producing the same ``(roots, coal_times)``
output but with real developmental structure.
"""

from __future__ import annotations

from typing import Any, Callable

from .merging import canonical_anchor_merge
from .trees import FlowNode
from .wormguides_parser import DivisionEvent


def build_lineage_lookup(
    division_events: list[DivisionEvent],
) -> dict[str, tuple[str, str, int]]:
    """Build parent -> (child1, child2, t_division) lookup from events."""
    lookup = {}
    for ev in division_events:
        lookup[ev.parent] = (ev.child1, ev.child2, ev.t_division)
    return lookup


def sulston_sample_forest(
    flow: Any,
    elements: list[Any],
    cell_names: list[str],
    division_events: list[DivisionEvent],
    death_set: set[str],
    total_timepoints: int = 360,
    merger: Callable = canonical_anchor_merge,
) -> tuple[list[FlowNode], list[float]]:
    """Build a coalescent forest using the known Sulston lineage.

    Instead of randomly merging adjacent leaves, this reconstructs the
    actual developmental tree:

    1. Create leaf nodes from the current cell states (elements).
    2. Look up each cell's parent in the division events.
    3. Recursively merge siblings into their parent node.
    4. Assign real division times (normalized to [0, 1]).

    Args:
        flow: The CoalescentFlow (used for time distribution, but we
            override with real division times).
        elements: Per-element anchor states (tensors or tuples).
        cell_names: Lineage name for each element (same order as elements).
        division_events: List of DivisionEvent from WormGUIDES parser.
        death_set: Set of cell names that undergo programmed death.
        total_timepoints: Total timepoints for normalizing times to [0, 1].
        merger: Anchor merge function for internal nodes.

    Returns:
        ``(roots, coal_times)`` -- same format as ``sample_forest``.
    """
    lineage = build_lineage_lookup(division_events)

    # Invert: child -> parent_name
    child_to_parent: dict[str, str] = {}
    for ev in division_events:
        child_to_parent[ev.child1] = ev.parent
        child_to_parent[ev.child2] = ev.parent

    # Create leaf nodes at t=1 (normalized)
    name_to_node: dict[str, FlowNode] = {}
    for i, (elem, name) in enumerate(zip(elements, cell_names)):
        is_dead = name in death_set
        node = FlowNode(
            time=1.0,
            data=elem,
            weight=1,
            group=0,
            branchable=True,
            del_flag=is_dead,
            id=i + 1,
            flowable=True,
        )
        name_to_node[name] = node

    coal_times: list[float] = []

    # Iteratively merge siblings bottom-up
    changed = True
    while changed:
        changed = False
        # Find pairs of siblings both present in name_to_node
        to_merge: list[tuple[str, str, str, int]] = []

        for parent_name, (c1, c2, t_div) in lineage.items():
            if c1 in name_to_node and c2 in name_to_node:
                if parent_name not in name_to_node:
                    to_merge.append((parent_name, c1, c2, t_div))

        for parent_name, c1, c2, t_div in to_merge:
            node1 = name_to_node[c1]
            node2 = name_to_node[c2]

            # Normalize division time to [0, 1]
            t_norm = t_div / total_timepoints
            t_norm = max(0.0, min(t_norm, 0.999))

            merged_data = merger(
                node1.data,
                node2.data,
                node1.weight,
                node2.weight,
            )

            parent_node = FlowNode(
                time=t_norm,
                data=merged_data,
                weight=node1.weight + node2.weight,
                group=0,
                branchable=True,
                del_flag=False,
                id=0,
                flowable=True,
            )
            parent_node.children = [node1, node2]
            node1.parent = parent_node
            node2.parent = parent_node

            name_to_node[parent_name] = parent_node
            del name_to_node[c1]
            del name_to_node[c2]

            coal_times.append(t_norm)
            changed = True

    # Remaining nodes without parents are roots
    roots = list(name_to_node.values())

    return roots, coal_times
