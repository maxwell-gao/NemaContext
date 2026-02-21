"""CoalescentFlow: the core branching/coalescent flow algorithm.

Port of BranchingFlows.jl/src/coalescent_flow.jl

This module implements:
- Forest sampling (backward-time coalescent construction)
- Conditional bridge sampling along forest branches
- Batched bridge construction for training

The main entry point is :func:`branching_bridge`, which takes a batch of
data samples (X1) and returns training targets (Xt, X1anchor, splits, dels).
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any, Callable

import numpy as np
import torch
from scipy.stats import binom, expon

from .merging import canonical_anchor_merge
from .policies import CoalescencePolicy, SequentialUniform
from .processes import BaseProcess, bridge_multi
from .states import (
    BranchingState,
    BridgeOutput,
    SampleState,
    group_fixedcount_del_insertions,
)
from .trees import FlowNode, merge_nodes


class CoalescentFlow:
    """Branching/coalescent flow wrapping base processes with splits and deletions.

    Args:
        processes: Base process or tuple of processes for each modality.
        branch_time_dist: scipy distribution on [0,1] controlling split timing
            (e.g. ``scipy.stats.beta(1, 2)``).
        split_transform: Maps model logits to positive split intensities.
        coalescence_policy: Policy for choosing which elements coalesce.
        deletion_time_dist: scipy distribution on [0,1] for deletion hazard.
    """

    def __init__(
        self,
        processes: BaseProcess | tuple[BaseProcess, ...],
        branch_time_dist: Any,
        split_transform: Callable | None = None,
        coalescence_policy: CoalescencePolicy | None = None,
        deletion_time_dist: Any | None = None,
    ):
        self.processes = processes
        self.branch_time_dist = branch_time_dist
        self.split_transform = split_transform or _default_split_transform
        self.coalescence_policy = coalescence_policy or SequentialUniform()
        if deletion_time_dist is None:
            from scipy.stats import uniform
            deletion_time_dist = uniform(0, 1)
        self.deletion_time_dist = deletion_time_dist


def _default_split_transform(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(torch.clamp(x, -100.0, 11.0))


# ---------------------------------------------------------------------------
# Split time sampling
# ---------------------------------------------------------------------------

def next_split_time(H: Any, W: int, t0: float) -> float:
    """Sample the absolute time of the next split.

    Uses the counting-flow interarrival time formula (paper Appendix A):
    draw ``E ~ Exp(1)``, set ``S* = S(t0) * exp(-E / m)`` where ``m = W - 1``,
    return ``H.ppf(1 - S*)``.

    Args:
        H: Hazard distribution (scipy) on [0,1] with ``.cdf`` and ``.ppf``.
        W: Descendant count (must be >= 2, so m = W-1 >= 1).
        t0: Current time in [0, 1).
    """
    assert 0.0 <= t0 < 1.0
    assert W >= 2
    m = W - 1
    S0 = 1.0 - H.cdf(t0)
    assert S0 > 0
    E = expon.rvs()
    S_star = S0 * math.exp(-E / m)
    p = 1.0 - S_star
    t = float(H.ppf(p))
    return max(t0, min(t, 1.0))


def sample_split_times(
    flow: CoalescentFlow,
    node: FlowNode,
    t0: float,
    collection: list[float] | None = None,
) -> None:
    """Recursively assign split times to internal nodes of *node*'s subtree."""
    if node.weight > 1:
        t_split = next_split_time(flow.branch_time_dist, node.weight, t0)
        node.time = t_split
        if collection is not None:
            collection.append(t_split)
        for child in node.children:
            sample_split_times(flow, child, t_split, collection)


# ---------------------------------------------------------------------------
# Forest sampling (backward-time coalescent)
# ---------------------------------------------------------------------------

def sample_forest(
    flow: CoalescentFlow,
    elements: list[Any],
    *,
    groupings: list[int] | None = None,
    branchable: list[bool] | None = None,
    flowable: list[bool] | None = None,
    deleted: list[bool] | None = None,
    ids: list[int] | None = None,
    coalescence_factor: float = 1.0,
    merger: Callable = canonical_anchor_merge,
    group_mins: dict[int, int] | int | None = None,
) -> tuple[list[FlowNode], list[float]]:
    """Sample a coalescent forest over *elements*.

    Starting from leaves at t=1, iteratively coalesce adjacent pairs
    backward in time.  Returns ``(roots, coal_times)`` where *roots* is
    the list of forest root nodes and *coal_times* collects the sampled
    absolute split times across the forest.

    Args:
        flow: The CoalescentFlow specifying hazard distributions and policy.
        elements: Per-element anchor states (tensors or tuples thereof).
        groupings: Per-element group IDs (default all 0).
        branchable: Per-element branchable flag (default all True).
        flowable: Per-element flowable flag (default all True).
        deleted: Per-element deletion flag (default all False).
        ids: Per-element IDs (default 1..N).
        coalescence_factor: Binomial p for number of merges (1.0 = full).
        merger: Anchor merge function.
        group_mins: Per-group minimum sizes passed to the policy.
    """
    n = len(elements)
    if groupings is None:
        groupings = [0] * n
    if branchable is None:
        branchable = [True] * n
    if flowable is None:
        flowable = [True] * n
    if deleted is None:
        deleted = [False] * n
    if ids is None:
        ids = list(range(1, n + 1))

    policy = flow.coalescence_policy

    nodes: list[FlowNode] = [
        FlowNode(
            time=1.0,
            data=elements[i],
            weight=1,
            group=groupings[i],
            branchable=branchable[i],
            del_flag=deleted[i],
            id=ids[i],
            flowable=flowable[i],
        )
        for i in range(n)
    ]

    policy.init(nodes)
    max_merges = policy.max_coalescences(nodes)

    # Sample how many merges to perform
    if isinstance(coalescence_factor, (int, float)):
        cf = float(coalescence_factor)
    else:
        cf = float(coalescence_factor)
    sampled_merges = int(binom.rvs(max_merges, cf)) if max_merges > 0 else 0

    for _ in range(sampled_merges):
        pair = policy.select_coalescence(nodes, group_mins)
        if pair is None:
            break
        i, j = pair
        if i > j:
            i, j = j, i

        left, right = nodes[i], nodes[j]
        assert left.group == right.group
        assert left.branchable and right.branchable

        merged_data = merger(
            left.data, right.data, left.weight, right.weight,
        )
        merged = merge_nodes(
            left, right,
            time=0.0,
            data=merged_data,
            weight=left.weight + right.weight,
            group=left.group,
            branchable=True,
            del_flag=False,
            id=0,
            flowable=True,
        )
        nodes[i] = merged
        del nodes[j]
        policy.update(nodes, i, j, i)

    nodes = policy.reorder_forest(nodes)

    # Recursively sample split times
    coal_times: list[float] = []
    for root in nodes:
        sample_split_times(flow, root, 0.0, coal_times)

    return nodes, coal_times


# ---------------------------------------------------------------------------
# Conditional bridge sampling along a tree
# ---------------------------------------------------------------------------

def tree_bridge(
    flow: CoalescentFlow,
    node: FlowNode,
    x_source: Any,
    target_t: float,
    current_t: float,
    collection: list[dict[str, Any]],
) -> None:
    """Recursively traverse *node*'s tree, sampling the conditional bridge at *target_t*.

    For each branch that crosses *target_t*, appends a segment dict to
    *collection* with keys: ``Xt``, ``t``, ``X1anchor``, ``descendants``,
    ``del``, ``branchable``, ``flowable``, ``group``, ``last_coalescence``, ``id``.

    Deletion handling uses the exact survival ratio ``S(target_t) / S(current_t)``
    from the deletion hazard distribution.
    """
    if not node.flowable:
        collection.append({
            "Xt": node.data,
            "t": target_t,
            "X1anchor": node.data,
            "descendants": node.weight,
            "del": node.del_flag,
            "branchable": False,
            "flowable": False,
            "group": node.group,
            "last_coalescence": current_t,
            "id": node.id,
        })
        return

    if node.time > target_t:
        # This branch crosses target_t -- sample bridge state here
        # Handle deletion survival
        if node.del_flag:
            S_cur = max(1.0 - flow.deletion_time_dist.cdf(current_t), 0.0)
            S_tgt = max(1.0 - flow.deletion_time_dist.cdf(target_t), 0.0)
            surv_ratio = (S_tgt / S_cur) if S_cur > 0 else 0.0
            if random.random() < (1.0 - surv_ratio):
                # Deleted before reaching target_t
                return

        Xt = bridge_multi(
            flow.processes, x_source, node.data, current_t, target_t,
        )
        collection.append({
            "Xt": Xt,
            "t": target_t,
            "X1anchor": node.data,
            "descendants": node.weight,
            "del": node.del_flag,
            "branchable": node.branchable,
            "flowable": True,
            "group": node.group,
            "last_coalescence": current_t,
            "id": node.id,
        })
    else:
        # Bridge to this node's split time, then recurse into children
        next_x = bridge_multi(
            flow.processes, x_source, node.data, current_t, node.time,
        )
        for child in node.children:
            tree_bridge(flow, child, next_x, target_t, node.time, collection)


# ---------------------------------------------------------------------------
# Forest bridge (single sample)
# ---------------------------------------------------------------------------

def forest_bridge(
    flow: CoalescentFlow,
    x0_sampler: Callable[[FlowNode], Any],
    x1_elements: list[Any],
    t: float,
    groupings: list[int],
    branchmask: list[bool],
    flowmask: list[bool],
    del_flags: list[bool],
    ids: list[int],
    *,
    use_branching_time_prob: float = 0.0,
    maxlen: float = float("inf"),
    coalescence_factor: float = 1.0,
    merger: Callable = canonical_anchor_merge,
    group_mins: dict[int, int] | int | None = None,
) -> list[dict[str, Any]]:
    """Run conditional bridge for one sample at time *t*.

    Samples a forest from *x1_elements*, then runs :func:`tree_bridge`
    for each root.  Returns a flat list of segment dicts.
    """
    forest, coal_times = sample_forest(
        flow,
        x1_elements,
        groupings=groupings,
        branchable=branchmask,
        flowable=flowmask,
        deleted=del_flags,
        ids=ids,
        coalescence_factor=coalescence_factor,
        merger=merger,
        group_mins=group_mins,
    )

    # Optionally override t with a sampled coalescence time
    if coal_times and random.random() < use_branching_time_prob:
        t = random.choice(coal_times)

    # Check length constraint (resample if needed)
    num_at_t = len(forest) + sum(1 for ct in coal_times if ct <= t)
    if num_at_t > maxlen:
        return forest_bridge(
            flow, x0_sampler, x1_elements, t,
            groupings, branchmask, flowmask, del_flags, ids,
            use_branching_time_prob=use_branching_time_prob,
            maxlen=maxlen,
            coalescence_factor=coalescence_factor,
            merger=merger,
            group_mins=group_mins,
        )

    collection: list[dict[str, Any]] = []
    for root in forest:
        x0 = x0_sampler(root)
        tree_bridge(flow, root, x0, t, 0.0, collection)

    return collection


# ---------------------------------------------------------------------------
# Group-mins resolution helpers
# ---------------------------------------------------------------------------

def _resolve_group_mins(
    length_mins: Any, groupings: list[int],
) -> dict[int, int] | None:
    """Convert various group_mins argument forms to a dict or None."""
    if length_mins is None:
        return None
    if isinstance(length_mins, int):
        unique_groups = set(groupings)
        return {g: length_mins for g in unique_groups}
    if isinstance(length_mins, dict):
        return length_mins
    return None


# ---------------------------------------------------------------------------
# Batched bridge (main training entry point)
# ---------------------------------------------------------------------------

def branching_bridge(
    flow: CoalescentFlow,
    x0_sampler: Callable[[FlowNode], Any],
    x1_list: list[SampleState],
    times: torch.Tensor | Any,
    *,
    use_branching_time_prob: float = 0.0,
    maxlen: float = float("inf"),
    coalescence_factor: float = 1.0,
    merger: Callable = canonical_anchor_merge,
    length_mins: Any = None,
    deletion_pad: float = 0.0,
    x1_modifier: Callable[[SampleState], SampleState] | None = None,
) -> BridgeOutput:
    """Vectorized conditional bridging over a batch.

    For each ``(X1, t)`` pair, samples an independent forest, runs conditional
    bridges, and collates the results into padded batched tensors.

    Args:
        flow: The CoalescentFlow.
        x0_sampler: ``fn(root_node) -> initial_state`` (tensor or tuple).
        x1_list: List of :class:`SampleState`, one per batch item.
        times: ``(batch,)`` tensor of times, or a scipy distribution to
            sample from.
        use_branching_time_prob: Probability of overriding *t* with a
            random split time from the forest.
        maxlen: Maximum number of elements at time *t* (resamples if exceeded).
        coalescence_factor: Binomial parameter for number of merges.
        merger: Anchor merge function.
        length_mins: Per-group minimum sizes (see :func:`sample_forest`).
        deletion_pad: If > 0, pads X1 with to-be-deleted duplicates so
            each group has ``deletion_pad * max(|x0_group|, |x1_group|)``
            elements in expectation.
        x1_modifier: Optional transform applied to each X1 after deletion
            padding.

    Returns:
        :class:`BridgeOutput` with all training targets.
    """
    batch_size = len(x1_list)

    # Resolve times
    if isinstance(times, torch.Tensor):
        t_values = times.tolist()
    elif hasattr(times, "rvs"):
        t_values = times.rvs(batch_size).tolist()
    else:
        t_values = list(times)

    # Resolve per-sample group_mins
    resolved_mins_list: list[Any] = []
    for i in range(batch_size):
        resolved_mins_list.append(
            _resolve_group_mins(length_mins, x1_list[i].groupings)
        )

    # Deletion padding
    if deletion_pad > 0:
        padded_x1_list: list[SampleState] = []
        for i, x1 in enumerate(x1_list):
            x1_lengths = Counter(x1.groupings)
            pad_counts: dict[int, int] = {}
            for grp, cnt in x1_lengths.items():
                mins = resolved_mins_list[i] or {}
                min_g = mins.get(grp, 1) if isinstance(mins, dict) else 1
                total_expected = deletion_pad * max(cnt, min_g)
                n_pad = max(0, int(np.random.poisson(total_expected - cnt)))
                pad_counts[grp] = n_pad

            x1_padded = group_fixedcount_del_insertions(x1, pad_counts)
            if x1_modifier is not None:
                x1_padded = x1_modifier(x1_padded)
            padded_x1_list.append(x1_padded)
        x1_list = padded_x1_list

    # Run per-sample forest bridges
    batch_bridges: list[list[dict[str, Any]]] = []
    for i, x1 in enumerate(x1_list):
        segments = forest_bridge(
            flow,
            x0_sampler,
            x1.elements,
            t_values[i],
            x1.groupings,
            x1.branchmask,
            x1.flowmask,
            x1.del_flags,
            x1.ids,
            use_branching_time_prob=use_branching_time_prob,
            maxlen=maxlen,
            coalescence_factor=coalescence_factor,
            merger=merger,
            group_mins=resolved_mins_list[i],
        )
        batch_bridges.append(segments)

    # Collate into batched tensors
    return _collate_bridges(flow, batch_bridges, batch_size)


# ---------------------------------------------------------------------------
# Collation: per-sample segment lists -> batched BridgeOutput
# ---------------------------------------------------------------------------

def _collate_bridges(
    flow: CoalescentFlow,
    batch_bridges: list[list[dict[str, Any]]],
    batch_size: int,
) -> BridgeOutput:
    """Collate per-sample segment lists into padded batched tensors."""
    max_len = max(len(segs) for segs in batch_bridges)
    if max_len == 0:
        raise ValueError("All samples produced empty bridges.")

    # Determine modality structure from first non-empty segment
    example_xt = batch_bridges[0][0]["Xt"]
    is_multimodal = isinstance(example_xt, tuple)
    n_modalities = len(example_xt) if is_multimodal else 1

    # Pre-allocate masks and scalar targets
    used_times = torch.zeros(batch_size)
    flowmask = torch.zeros(max_len, batch_size, dtype=torch.bool)
    branchmask = torch.zeros(max_len, batch_size, dtype=torch.bool)
    descendants = torch.zeros(max_len, batch_size, dtype=torch.long)
    padmask = torch.zeros(max_len, batch_size, dtype=torch.bool)
    del_flags = torch.zeros(max_len, batch_size, dtype=torch.bool)
    prev_coal = torch.zeros(max_len, batch_size)
    groups = torch.zeros(max_len, batch_size, dtype=torch.long)
    ids = torch.zeros(max_len, batch_size, dtype=torch.long)

    # Collect per-element states for later stacking
    xt_elements: list[list[Any]] = [[] for _ in range(batch_size)]
    anchor_elements: list[list[Any]] = [[] for _ in range(batch_size)]

    for b, segments in enumerate(batch_bridges):
        if segments:
            used_times[b] = segments[0]["t"]
        for i, seg in enumerate(segments):
            flowmask[i, b] = seg["flowable"]
            branchmask[i, b] = seg["branchable"]
            descendants[i, b] = seg["descendants"]
            padmask[i, b] = True
            del_flags[i, b] = seg["del"]
            prev_coal[i, b] = seg["last_coalescence"]
            groups[i, b] = seg["group"]
            ids[i, b] = seg["id"]
            xt_elements[b].append(seg["Xt"])
            anchor_elements[b].append(seg["X1anchor"])

    # Stack states into tensors per modality
    xt_batched = _stack_elements(xt_elements, max_len, is_multimodal, n_modalities)
    anchor_batched = _stack_elements(anchor_elements, max_len, is_multimodal, n_modalities)

    # Transpose masks from (length, batch) to (batch, length)
    flowmask = flowmask.T.contiguous()
    branchmask = branchmask.T.contiguous()
    descendants = descendants.T.contiguous()
    padmask_t = padmask.T.contiguous()
    del_flags = del_flags.T.contiguous()
    prev_coal = prev_coal.T.contiguous()
    groups = groups.T.contiguous()
    ids = ids.T.contiguous()

    splits_target = (descendants - 1).clamp(min=0).float()

    Xt = BranchingState(
        states=xt_batched,
        groupings=groups,
        del_flags=del_flags,
        ids=ids,
        branchmask=branchmask,
        flowmask=flowmask,
        padmask=padmask_t,
    )

    return BridgeOutput(
        t=used_times,
        Xt=Xt,
        X1anchor=anchor_batched,
        del_flags=del_flags,
        descendants=descendants,
        splits_target=splits_target,
        prev_coalescence=prev_coal,
    )


def _stack_elements(
    elements_per_batch: list[list[Any]],
    max_len: int,
    is_multimodal: bool,
    n_modalities: int,
) -> tuple[torch.Tensor, ...]:
    """Stack per-element states into ``(batch, length, ...)`` tensors."""
    batch_size = len(elements_per_batch)

    if not is_multimodal:
        # Single modality
        example = _find_example_element(elements_per_batch)
        if isinstance(example, torch.Tensor):
            feat_shape = example.shape
            out = torch.zeros(batch_size, max_len, *feat_shape)
            for b, elems in enumerate(elements_per_batch):
                for i, e in enumerate(elems):
                    out[b, i] = e
            return (out,)
        else:
            # Scalar (int for discrete)
            out = torch.zeros(batch_size, max_len, dtype=torch.long)
            for b, elems in enumerate(elements_per_batch):
                for i, e in enumerate(elems):
                    out[b, i] = int(e) if not isinstance(e, torch.Tensor) else e.item()
            return (out,)

    # Multimodal: tuple of states
    results: list[torch.Tensor] = []
    for m in range(n_modalities):
        example = _find_example_modality(elements_per_batch, m)
        if isinstance(example, torch.Tensor) and example.dim() > 0:
            feat_shape = example.shape
            out = torch.zeros(batch_size, max_len, *feat_shape)
            for b, elems in enumerate(elements_per_batch):
                for i, e in enumerate(elems):
                    out[b, i] = e[m]
            results.append(out)
        else:
            out = torch.zeros(batch_size, max_len, dtype=torch.long)
            for b, elems in enumerate(elements_per_batch):
                for i, e in enumerate(elems):
                    val = e[m]
                    out[b, i] = int(val) if not isinstance(val, torch.Tensor) else val.item()
            results.append(out)

    return tuple(results)


def _find_example_element(elements_per_batch: list[list[Any]]) -> Any:
    for elems in elements_per_batch:
        if elems:
            return elems[0]
    raise ValueError("No elements found in batch.")


def _find_example_modality(
    elements_per_batch: list[list[Any]], m: int,
) -> Any:
    for elems in elements_per_batch:
        if elems:
            return elems[0][m]
    raise ValueError("No elements found in batch.")
