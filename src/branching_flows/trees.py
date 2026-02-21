"""Binary tree data structure for coalescent forest construction.

Port of BranchingFlows.jl/src/trees.jl
"""

from __future__ import annotations

from typing import Any


class FlowNode:
    """A node in a coalescent/branching binary tree.

    Each node carries a state (anchor data), a weight (descendant count),
    group membership, and flags controlling branching/deletion/flow behavior.

    During backward-time forest construction, leaves represent data elements
    at t=1 and internal nodes are created by merging adjacent pairs.
    During forward-time generation, roots represent initial elements at t=0
    and splits create new children.
    """

    __slots__ = (
        "parent", "children", "time", "data", "weight",
        "group", "branchable", "del_flag", "id", "flowable",
    )

    def __init__(
        self,
        time: float,
        data: Any,
        weight: int = 1,
        group: int = 0,
        branchable: bool = True,
        del_flag: bool = False,
        id: int = 1,
        flowable: bool = True,
    ):
        self.parent: FlowNode | None = None
        self.children: list[FlowNode] = []
        self.time = time
        self.data = data
        self.weight = weight
        self.group = group
        self.branchable = branchable
        self.del_flag = del_flag
        self.id = id
        self.flowable = flowable

    def __repr__(self) -> str:
        return (
            f"FlowNode(time={self.time:.4f}, weight={self.weight}, "
            f"group={self.group}, branchable={self.branchable}, "
            f"del={self.del_flag}, id={self.id}, flowable={self.flowable})"
        )


def add_child(parent: FlowNode, child: FlowNode) -> None:
    """Attach *child* to *parent*. Raises if child already has a parent."""
    if child.parent is not None:
        raise ValueError("Child already has a parent.")
    parent.children.append(child)
    child.parent = parent


def merge_nodes(
    n1: FlowNode,
    n2: FlowNode,
    time: float,
    data: Any,
    weight: int,
    group: int,
    branchable: bool = True,
    del_flag: bool = False,
    id: int = 0,
    flowable: bool = True,
) -> FlowNode:
    """Create a new parent node by merging two root nodes.

    Both *n1* and *n2* must be parentless (roots). The returned node
    has *n1* and *n2* as its two children.  Merged internal nodes
    conventionally have ``id=0``, ``del_flag=False``, and ``flowable=True``.
    """
    if n1.parent is not None or n2.parent is not None:
        raise ValueError("Cannot merge nodes that already have parents.")
    parent = FlowNode(
        time=time, data=data, weight=weight, group=group,
        branchable=branchable, del_flag=del_flag, id=id, flowable=flowable,
    )
    add_child(parent, n1)
    add_child(parent, n2)
    return parent
