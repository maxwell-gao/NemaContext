"""Anchor merging strategies for coalescent tree construction.

Port of BranchingFlows.jl/src/states.jl (anchor merge functions).

When two child nodes are coalesced into a parent, the parent needs an anchor
state.  These functions define how to compute that anchor from the children's
anchors and descendant weights.
"""

from __future__ import annotations

import random
from typing import Any

import torch


def canonical_anchor_merge(
    s1: Any, s2: Any, w1: int, w2: int,
) -> Any:
    """Merge two anchor states into a parent anchor.

    Dispatches by type:

    * **tuple** of states: applies element-wise with the same weights.
    * **Tensor** (continuous): weighted Euclidean average
      ``(w1*s1 + w2*s2) / (w1+w2)``.
    * **int** (discrete): returns the value directly -- caller is responsible
      for passing the mask/dummy token ``K`` as both ``s1`` and ``s2`` for
      internal nodes, or overriding this for custom behavior.
    """
    if isinstance(s1, tuple):
        return tuple(
            canonical_anchor_merge(a, b, w1, w2) for a, b in zip(s1, s2)
        )

    if isinstance(s1, torch.Tensor):
        return (s1 * w1 + s2 * w2) / (w1 + w2)

    if isinstance(s1, int):
        return s1

    raise TypeError(f"Unsupported anchor type: {type(s1)}")


def select_anchor_merge(
    s1: Any, s2: Any, w1: int, w2: int,
) -> Any:
    """Stochastic merge that copies one child instead of interpolating.

    Chooses *s1* with probability ``w1 / (w1 + w2)``; otherwise *s2*.
    Then delegates to :func:`canonical_anchor_merge` with weights ``(1, 0)``
    or ``(0, 1)`` to preserve type-specific behavior (e.g. discrete masking).
    """
    if random.random() < w1 / (w1 + w2):
        return canonical_anchor_merge(s1, s2, 1, 0)
    return canonical_anchor_merge(s2, s1, 1, 0)
