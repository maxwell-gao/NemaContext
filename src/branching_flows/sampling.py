"""Forward-time generation: step function and generation loop.

Port of the Flowfusion.step(CoalescentFlow, ...) and gen() from
BranchingFlows.jl/src/coalescent_flow.jl (lines 717-781) and
Flowfusion's generation loop.

During inference, the model predicts per-element:
- endpoint states (x1_pred)
- split intensities (split_logits)
- deletion probabilities (del_logits)

The step function advances the base process, then samples splits and
deletions to produce a new BranchingState with potentially different length.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from .flow import CoalescentFlow
from .states import BranchingState


def step(
    flow: CoalescentFlow,
    state: BranchingState,
    model_output: tuple[Any, torch.Tensor, torch.Tensor],
    s1: float,
    s2: float,
) -> BranchingState:
    """Advance the branching process from *s1* to *s2* for a single-batch state.

    Args:
        flow: The CoalescentFlow.
        state: Current BranchingState (batch=1).
        model_output: Tuple of ``(x1_preds, split_logits, del_logits)``.
            - *x1_preds*: predicted endpoint states (tuple of tensors or
              single tensor), matching the structure of ``state.states``.
            - *split_logits*: ``(1, length)`` logits for split intensities.
            - *del_logits*: ``(1, length)`` logits for deletion probability.
        s1: Current time.
        s2: Target time.

    Returns:
        New BranchingState with updated states, possibly different length
        due to splits and deletions.
    """
    x1_preds, split_logits, del_logits = model_output
    dt = s2 - s1

    # 1. Advance base process
    if isinstance(flow.processes, (list, tuple)):
        new_states = tuple(
            p.step(state.states[k], x1_preds[k], s1, s2)
            for k, p in enumerate(flow.processes)
        )
    else:
        new_states = (flow.processes.step(state.states[0], x1_preds, s1, s2),)

    # Apply flowmask: keep old state where flowmask is False
    masked_states: list[torch.Tensor] = []
    for k in range(len(new_states)):
        old = state.states[k]
        new = new_states[k]
        fm = state.flowmask  # (1, L)
        if new.dim() > fm.dim():
            mask = fm.unsqueeze(-1).expand_as(new)
        else:
            mask = fm
        masked_states.append(torch.where(mask, new, old))
    new_states = tuple(masked_states)

    L = state.padmask.shape[1]
    bmask = state.branchmask[0]  # (L,)

    # 2. Sample splits
    H = flow.branch_time_dist
    # pdf of Truncated(H, s1, 1) at s1
    try:
        H_pdf_s1 = H.pdf(s1)
        H_cdf_s1 = H.cdf(s1)
        trunc_pdf = H_pdf_s1 / max(1.0 - H_cdf_s1, 1e-12)
    except Exception:
        trunc_pdf = 1.0

    split_rates = flow.split_transform(split_logits[0]) * dt * trunc_pdf  # (L,)
    split_counts = torch.poisson(split_rates.clamp(min=0.0))
    split_counts = (split_counts * bmask.float()).long()  # mask by branchmask

    # 3. Sample deletions
    S1 = max(1.0 - flow.deletion_time_dist.cdf(s1), 0.0)
    f1 = flow.deletion_time_dist.pdf(s1)
    h1 = (f1 / S1) if S1 > 0 else 0.0
    base_p = 1.0 - math.exp(-h1 * dt)
    rho = torch.sigmoid(del_logits[0])  # (L,)
    del_probs = rho * base_p
    dels = (torch.rand_like(del_probs) < del_probs) & bmask

    # Suppress splits where deletions occur
    split_counts[dels] = 0

    # 4. Suppress splits where discrete state changed
    if isinstance(flow.processes, (list, tuple)):
        for k, p in enumerate(flow.processes):
            from .processes import DiscreteInterpolatingFlow

            if isinstance(p, DiscreteInterpolatingFlow):
                old_tokens = state.states[k][0]  # (L,) or (L, ...)
                new_tokens = new_states[k][0]
                changed = old_tokens != new_tokens
                if changed.dim() > 1:
                    changed = changed.any(dim=-1)
                split_counts[changed] = 0

    # 5. Build new state with insertions and deletions
    current_L = L
    new_L = current_L + split_counts.sum().item() - dels.sum().item()

    if new_L <= 0:
        new_L = max(new_L, 1)

    # Allocate new tensors
    out_states: list[torch.Tensor] = []
    for k in range(len(new_states)):
        s = new_states[k]
        if s.dim() == 3:
            out_states.append(
                torch.zeros(1, new_L, s.shape[2], dtype=s.dtype, device=s.device)
            )
        else:
            out_states.append(torch.zeros(1, new_L, dtype=s.dtype, device=s.device))

    out_groups = torch.zeros(1, new_L, dtype=torch.long, device=state.groupings.device)
    out_fmask = torch.zeros(1, new_L, dtype=torch.bool, device=state.flowmask.device)
    out_bmask = torch.zeros(1, new_L, dtype=torch.bool, device=state.branchmask.device)
    out_padmask = torch.ones(1, new_L, dtype=torch.bool, device=state.padmask.device)
    out_del = torch.zeros(1, new_L, dtype=torch.bool, device=state.del_flags.device)
    out_ids = torch.zeros(1, new_L, dtype=torch.long, device=state.ids.device)

    idx = 0
    for i in range(current_L):
        if dels[i]:
            continue

        # Copy original element
        for k in range(len(new_states)):
            if new_states[k].dim() == 3:
                out_states[k][0, idx] = new_states[k][0, i]
            else:
                out_states[k][0, idx] = new_states[k][0, i]
        out_groups[0, idx] = state.groupings[0, i]
        out_fmask[0, idx] = state.flowmask[0, i]
        out_bmask[0, idx] = state.branchmask[0, i]
        out_ids[0, idx] = state.ids[0, i]
        idx += 1

        # Insert duplicates for splits
        n_splits = split_counts[i].item()
        for _ in range(n_splits):
            if idx >= new_L:
                break
            for k in range(len(new_states)):
                if new_states[k].dim() == 3:
                    out_states[k][0, idx] = new_states[k][0, i]
                else:
                    out_states[k][0, idx] = new_states[k][0, i]
            out_groups[0, idx] = state.groupings[0, i]
            out_fmask[0, idx] = state.flowmask[0, i]
            out_bmask[0, idx] = state.branchmask[0, i]
            out_ids[0, idx] = 0
            idx += 1

    return BranchingState(
        states=tuple(out_states),
        groupings=out_groups,
        del_flags=out_del,
        ids=out_ids,
        branchmask=out_bmask,
        flowmask=out_fmask,
        padmask=out_padmask,
    )


# ---------------------------------------------------------------------------
# Generation loop
# ---------------------------------------------------------------------------


@dataclass
class Tracker:
    """Records generation trajectory snapshots."""

    xt: list[tuple[BranchingState, float]] = field(default_factory=list)

    def record(self, state: BranchingState, t: float) -> None:
        self.xt.append((state, t))


def generate(
    flow: CoalescentFlow,
    x0_state: BranchingState,
    model_fn: Callable[[float, BranchingState], tuple[Any, torch.Tensor, torch.Tensor]],
    timesteps: torch.Tensor,
    tracker: Tracker | None = None,
) -> BranchingState:
    """Generate a sample by Euler integration from t=0 to t=1.

    Args:
        flow: The CoalescentFlow.
        x0_state: Initial BranchingState (batch=1).
        model_fn: ``fn(t, state) -> (x1_preds, split_logits, del_logits)``.
        timesteps: 1-D tensor of time values, e.g. ``torch.linspace(0, 1, 1000)``.
        tracker: Optional Tracker for recording snapshots.

    Returns:
        Final BranchingState at the last timestep.
    """
    state = x0_state
    times = timesteps.tolist()

    for k in range(len(times) - 1):
        s1 = times[k]
        s2 = times[k + 1]

        if tracker is not None:
            tracker.record(state, s1)

        model_output = model_fn(s1, state)
        state = step(flow, state, model_output, s1, s2)

    if tracker is not None:
        tracker.record(state, times[-1])

    return state
