"""State containers and deletion-augmentation utilities.

Port of BranchingState from BranchingFlows.jl/src/coalescent_flow.jl and
the deletion insertion functions (uniform_del_insertions, etc.).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import torch


# ---------------------------------------------------------------------------
# Batched state containers
# ---------------------------------------------------------------------------

@dataclass
class BranchingState:
    """Batched state for a variable-length branching process.

    All mask/index tensors have shape ``(batch, length)``.
    State tensors are a tuple -- each is ``(batch, length, features)`` for
    continuous modalities or ``(batch, length)`` for discrete.

    Attributes:
        states: Tuple of per-modality state tensors.
        groupings: Per-element group IDs; elements only coalesce within groups.
        del_flags: Marks elements destined for deletion at t=1.
        ids: Element IDs (merged internal nodes use 0).
        branchmask: Where True, splits/deletions are permitted.
        flowmask: Where True, the base process evolves the state.
        padmask: Where True, the position is valid (not padding).
    """

    states: tuple[torch.Tensor, ...]
    groupings: torch.Tensor
    del_flags: torch.Tensor
    ids: torch.Tensor
    branchmask: torch.Tensor
    flowmask: torch.Tensor
    padmask: torch.Tensor

    def to(self, device: torch.device | str) -> BranchingState:
        return BranchingState(
            states=tuple(s.to(device) for s in self.states),
            groupings=self.groupings.to(device),
            del_flags=self.del_flags.to(device),
            ids=self.ids.to(device),
            branchmask=self.branchmask.to(device),
            flowmask=self.flowmask.to(device),
            padmask=self.padmask.to(device),
        )


@dataclass
class BridgeOutput:
    """Return type of :func:`branching_bridge`.

    Contains the batched bridge states at time *t* together with the
    training targets (anchor states, split counts, deletion flags).
    """

    t: torch.Tensor                    # (batch,)
    Xt: BranchingState                 # states at time t
    X1anchor: tuple[torch.Tensor, ...] # anchor targets per modality
    del_flags: torch.Tensor            # (batch, length) bool
    descendants: torch.Tensor          # (batch, length) int
    splits_target: torch.Tensor        # (batch, length) float
    prev_coalescence: torch.Tensor     # (batch, length) float

    def to(self, device: torch.device | str) -> BridgeOutput:
        return BridgeOutput(
            t=self.t.to(device),
            Xt=self.Xt.to(device),
            X1anchor=tuple(a.to(device) for a in self.X1anchor),
            del_flags=self.del_flags.to(device),
            descendants=self.descendants.to(device),
            splits_target=self.splits_target.to(device),
            prev_coalescence=self.prev_coalescence.to(device),
        )


# ---------------------------------------------------------------------------
# Per-sample (unbatched) state used during forest construction
# ---------------------------------------------------------------------------

@dataclass
class SampleState:
    """Unbatched per-sample state before batching.

    Holds parallel lists for each element in a single data sample.
    This is the Python equivalent of the Julia BranchingState used
    before the batched ``branching_bridge`` collation.
    """

    elements: list[Any]          # per-element state (tensor or tuple)
    groupings: list[int]
    del_flags: list[bool]
    ids: list[int]
    branchmask: list[bool]
    flowmask: list[bool]

    @property
    def length(self) -> int:
        return len(self.elements)


# ---------------------------------------------------------------------------
# Deletion augmentation
# ---------------------------------------------------------------------------

def uniform_del_insertions(
    sample: SampleState,
    del_p: float,
) -> SampleState:
    """Duplicate elements with independent probability *del_p*.

    Only elements where both ``flowmask`` and ``branchmask`` are True are
    eligible.  Each duplication inserts a copy either before or after the
    original (chosen uniformly); exactly one of the pair is marked for deletion.

    Returns a new :class:`SampleState` with expanded length.
    """
    n = sample.length
    new_elements: list[Any] = []
    new_groups: list[int] = []
    new_del: list[bool] = []
    new_ids: list[int] = []
    new_bmask: list[bool] = []
    new_fmask: list[bool] = []

    for i in range(n):
        eligible = sample.flowmask[i] and sample.branchmask[i]
        do_dup = eligible and (random.random() < del_p)

        if do_dup:
            # Decide which copy is deleted
            del_first = random.random() < 0.5
            elem = sample.elements[i]
            grp = sample.groupings[i]
            eid = sample.ids[i]
            bm = sample.branchmask[i]
            fm = sample.flowmask[i]

            # Insert original + duplicate (order chosen uniformly)
            for j in range(2):
                new_elements.append(_clone_element(elem))
                new_groups.append(grp)
                new_ids.append(eid)
                new_bmask.append(bm)
                new_fmask.append(fm)
                if j == 0:
                    new_del.append(del_first)
                else:
                    new_del.append(not del_first)
        else:
            new_elements.append(sample.elements[i])
            new_groups.append(sample.groupings[i])
            new_del.append(sample.del_flags[i])
            new_ids.append(sample.ids[i])
            new_bmask.append(sample.branchmask[i])
            new_fmask.append(sample.flowmask[i])

    return SampleState(
        elements=new_elements,
        groupings=new_groups,
        del_flags=new_del,
        ids=new_ids,
        branchmask=new_bmask,
        flowmask=new_fmask,
    )


def fixedcount_del_insertions(
    sample: SampleState,
    num_events: int,
) -> SampleState:
    """Insert exactly *num_events* duplication-deletion events.

    Targets are sampled uniformly with replacement from eligible positions.
    """
    if num_events <= 0:
        return sample

    n = sample.length
    eligible = [
        i for i in range(n) if sample.flowmask[i] and sample.branchmask[i]
    ]
    if not eligible:
        return sample

    # Track insertions per position
    before_flags: list[list[bool]] = [[] for _ in range(n)]
    after_flags: list[list[bool]] = [[] for _ in range(n)]
    orig_del = [False] * n

    for _ in range(num_events):
        i = random.choice(eligible)
        insert_before = random.random() < 0.5
        del_original = random.random() < 0.5 and not orig_del[i]

        if insert_before:
            if del_original:
                orig_del[i] = True
                before_flags[i].append(False)
            else:
                before_flags[i].append(True)
        else:
            if del_original:
                orig_del[i] = True
                after_flags[i].append(False)
            else:
                after_flags[i].append(True)

    return _build_augmented_sample(
        sample, before_flags, after_flags, orig_del,
    )


def group_fixedcount_del_insertions(
    sample: SampleState,
    group_num_events: dict[int, int],
) -> SampleState:
    """Insert a fixed number of duplication events per group."""
    if not any(v > 0 for v in group_num_events.values()):
        return sample

    n = sample.length
    eligible = [
        i for i in range(n) if sample.flowmask[i] and sample.branchmask[i]
    ]
    if not eligible:
        return sample

    before_flags: list[list[bool]] = [[] for _ in range(n)]
    after_flags: list[list[bool]] = [[] for _ in range(n)]
    orig_del = [False] * n

    for grp, count in group_num_events.items():
        if count <= 0:
            continue
        eligible_g = [i for i in eligible if sample.groupings[i] == grp]
        if not eligible_g:
            continue

        for _ in range(count):
            i = random.choice(eligible_g)
            insert_before = random.random() < 0.5
            del_original = random.random() < 0.5 and not orig_del[i]

            if insert_before:
                if del_original:
                    orig_del[i] = True
                    before_flags[i].append(False)
                else:
                    before_flags[i].append(True)
            else:
                if del_original:
                    orig_del[i] = True
                    after_flags[i].append(False)
                else:
                    after_flags[i].append(True)

    return _build_augmented_sample(
        sample, before_flags, after_flags, orig_del,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clone_element(elem: Any) -> Any:
    if isinstance(elem, torch.Tensor):
        return elem.clone()
    if isinstance(elem, tuple):
        return tuple(
            e.clone() if isinstance(e, torch.Tensor) else e for e in elem
        )
    return elem


def _build_augmented_sample(
    sample: SampleState,
    before_flags: list[list[bool]],
    after_flags: list[list[bool]],
    orig_del: list[bool],
) -> SampleState:
    """Assemble augmented sample from per-position insertion flags."""
    new_elements: list[Any] = []
    new_groups: list[int] = []
    new_del: list[bool] = []
    new_ids: list[int] = []
    new_bmask: list[bool] = []
    new_fmask: list[bool] = []

    for i in range(sample.length):
        elem = sample.elements[i]
        grp = sample.groupings[i]
        eid = sample.ids[i]
        bm = sample.branchmask[i]
        fm = sample.flowmask[i]

        # Duplicates before
        for flag in before_flags[i]:
            new_elements.append(_clone_element(elem))
            new_groups.append(grp)
            new_del.append(flag)
            new_ids.append(eid)
            new_bmask.append(bm)
            new_fmask.append(fm)

        # Original
        new_elements.append(elem)
        new_groups.append(grp)
        new_del.append(orig_del[i])
        new_ids.append(eid)
        new_bmask.append(bm)
        new_fmask.append(fm)

        # Duplicates after
        for flag in after_flags[i]:
            new_elements.append(_clone_element(elem))
            new_groups.append(grp)
            new_del.append(flag)
            new_ids.append(eid)
            new_bmask.append(bm)
            new_fmask.append(fm)

    return SampleState(
        elements=new_elements,
        groupings=new_groups,
        del_flags=new_del,
        ids=new_ids,
        branchmask=new_bmask,
        flowmask=new_fmask,
    )
