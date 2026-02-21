"""Coalescence selection policies for forest construction.

Port of BranchingFlows.jl/src/merging.jl (policy classes).

Policies decide *which* pair of nodes to coalesce at each step of backward-time
forest construction.  Sequential policies restrict merges to adjacent pairs in
the node list and respect group boundaries.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any

from .trees import FlowNode


class CoalescencePolicy(ABC):
    """Abstract base for coalescence selection policies."""

    def init(self, nodes: list[FlowNode]) -> None:  # noqa: B027
        """Optional stateful initialization hook (no-op by default)."""

    def update(
        self, nodes: list[FlowNode], i: int, j: int, new_index: int,
    ) -> None:  # noqa: B027
        """Optional hook after a merge (no-op by default)."""

    def reorder_forest(self, nodes: list[FlowNode]) -> list[FlowNode]:
        """Optional reordering of roots/children after all merges."""
        return nodes

    @abstractmethod
    def max_coalescences(self, nodes: list[FlowNode]) -> int:
        """Upper bound on feasible merge events from *nodes*."""

    @abstractmethod
    def select_coalescence(
        self,
        nodes: list[FlowNode],
        group_mins: Any = None,
    ) -> tuple[int, int] | None:
        """Pick a pair ``(i, j)`` to merge, or ``None`` if none eligible."""


# ---------------------------------------------------------------------------
# Sequential policies (merge adjacent pairs only)
# ---------------------------------------------------------------------------

class SequentialUniform(CoalescencePolicy):
    """Uniformly choose one adjacent branchable pair within the same group.

    Supports three forms of *group_mins*:

    * ``None`` -- no per-group minimum enforced.
    * ``dict[int, int]`` -- maps ``group_id -> minimum_count``; a merge is
      blocked if it would reduce a group below its minimum.
    * ``int`` -- per-contiguous-block minimum (each run of adjacent branchable
      nodes within one group must stay above this size).
    """

    def max_coalescences(self, nodes: list[FlowNode]) -> int:
        count = 0
        for i in range(len(nodes) - 1):
            if (
                nodes[i].branchable
                and nodes[i + 1].branchable
                and nodes[i].group == nodes[i + 1].group
            ):
                count += 1
        return count

    # -- dispatch on group_mins type --

    def select_coalescence(
        self,
        nodes: list[FlowNode],
        group_mins: dict[int, int] | int | None = None,
    ) -> tuple[int, int] | None:
        if group_mins is None:
            return self._select_none(nodes)
        if isinstance(group_mins, int):
            return self._select_block_min(nodes, group_mins)
        return self._select_dict_min(nodes, group_mins)

    # -- internals --

    @staticmethod
    def _select_none(nodes: list[FlowNode]) -> tuple[int, int] | None:
        eligible: list[int] = []
        for i in range(len(nodes) - 1):
            if (
                nodes[i].branchable
                and nodes[i + 1].branchable
                and nodes[i].group == nodes[i + 1].group
            ):
                eligible.append(i)
        if not eligible:
            return None
        i = random.choice(eligible)
        return (i, i + 1)

    @staticmethod
    def _select_dict_min(
        nodes: list[FlowNode], group_mins: dict[int, int],
    ) -> tuple[int, int] | None:
        group_sizes: dict[int, int] = {}
        for n in nodes:
            if n.branchable:
                group_sizes[n.group] = group_sizes.get(n.group, 0) + 1

        eligible: list[int] = []
        for i in range(len(nodes) - 1):
            if (
                nodes[i].branchable
                and nodes[i + 1].branchable
                and nodes[i].group == nodes[i + 1].group
                and group_sizes.get(nodes[i].group, 0) > group_mins.get(nodes[i].group, 1)
            ):
                eligible.append(i)
        if not eligible:
            return None
        i = random.choice(eligible)
        return (i, i + 1)

    @staticmethod
    def _select_block_min(
        nodes: list[FlowNode], block_min: int,
    ) -> tuple[int, int] | None:
        # First pass: identify contiguous branchable blocks and their sizes.
        block_ids: list[int] = [0] * len(nodes)
        block_sizes: dict[int, int] = {}
        cur_block = 0
        for i in range(len(nodes)):
            block_ids[i] = cur_block
            if i < len(nodes) - 1:
                if not (
                    nodes[i].branchable
                    and nodes[i + 1].branchable
                    and nodes[i].group == nodes[i + 1].group
                ):
                    cur_block += 1

        for i in range(len(nodes) - 1):
            if (
                nodes[i].branchable
                and nodes[i + 1].branchable
                and nodes[i].group == nodes[i + 1].group
            ):
                bid = block_ids[i]
                block_sizes[bid] = block_sizes.get(bid, 0) + 1

        eligible: list[int] = []
        for i in range(len(nodes) - 1):
            if (
                nodes[i].branchable
                and nodes[i + 1].branchable
                and nodes[i].group == nodes[i + 1].group
            ):
                bid = block_ids[i]
                if block_sizes.get(bid, 0) > (block_min - 1):
                    eligible.append(i)
        if not eligible:
            return None
        i = random.choice(eligible)
        return (i, i + 1)


class BalancedSequential(CoalescencePolicy):
    """Sequential policy that prefers coalescing smaller adjacent clusters.

    Sampling weight for pair ``(i, i+1)`` is ``(w_i + w_{i+1})^{-alpha}``.
    ``alpha=0`` recovers uniform sequential; larger alpha favors smaller clusters.
    """

    def __init__(self, alpha: float = 1.0):
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        self.alpha = alpha

    def max_coalescences(self, nodes: list[FlowNode]) -> int:
        count = 0
        for i in range(len(nodes) - 1):
            if (
                nodes[i].branchable
                and nodes[i + 1].branchable
                and nodes[i].group == nodes[i + 1].group
            ):
                count += 1
        return count

    def select_coalescence(
        self,
        nodes: list[FlowNode],
        group_mins: Any = None,
    ) -> tuple[int, int] | None:
        pairs: list[int] = []
        weights: list[float] = []
        for i in range(len(nodes) - 1):
            if (
                nodes[i].branchable
                and nodes[i + 1].branchable
                and nodes[i].group == nodes[i + 1].group
            ):
                pairs.append(i)
                weights.append(
                    (nodes[i].weight + nodes[i + 1].weight) ** (-self.alpha)
                )
        if not pairs:
            return None
        chosen = random.choices(pairs, weights=weights, k=1)[0]
        return (chosen, chosen + 1)
