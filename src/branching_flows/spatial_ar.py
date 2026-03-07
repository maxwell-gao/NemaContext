"""Spatial-first autoregressive baseline for real WormGUIDES trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .autoregressive_model import TransformerBlockAutoregressive
from .dynamic_cell_manager import DynamicCellManager, EventDecision
from .states import BranchingState


@dataclass
class SpatialStepOutput:
    """Single-step output for the spatial baseline."""

    continuous_delta: torch.Tensor
    split_logits: torch.Tensor
    del_logits: torch.Tensor
    events: EventDecision | None = None


class SpatialAutoregressiveModel(nn.Module):
    """Spatial-only whole-embryo rollout baseline."""

    def __init__(
        self,
        continuous_dim: int = 6,
        discrete_K: int = 7,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        head_dim: int = 32,
        max_seq_len: int = 512,
        dt: float = 1.0,
        deterministic_topk_events: bool = False,
    ):
        super().__init__()
        self.continuous_dim = continuous_dim
        self.discrete_K = discrete_K
        self.d_model = d_model
        self.dt = dt

        self.continuous_proj = nn.Sequential(
            nn.Linear(continuous_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.discrete_embed = nn.Embedding(discrete_K, d_model)

        self.blocks = nn.ModuleList(
            [
                TransformerBlockAutoregressive(d_model, n_heads, head_dim)
                for _ in range(n_layers)
            ]
        )

        self.delta_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, continuous_dim),
        )
        self.split_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.del_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        self.cell_manager = DynamicCellManager(
            split_threshold=0.5,
            del_threshold=0.5,
            max_cells=max_seq_len,
            use_gumbel=True,
            deterministic_topk=deterministic_topk_events,
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode_state(self, state: BranchingState) -> torch.Tensor:
        cont = state.states[0]
        disc = state.states[1]
        h = self.continuous_proj(cont) + self.discrete_embed(
            disc.clamp(0, self.discrete_K - 1)
        )
        return h

    def forward_step(self, state: BranchingState) -> SpatialStepOutput:
        B, L = state.states[0].shape[:2]
        if L == 0:
            device = state.states[0].device
            dtype = state.states[0].dtype
            empty_cont = torch.zeros(
                B, 0, self.continuous_dim, device=device, dtype=dtype
            )
            empty_event = torch.zeros(B, 0, 1, device=device, dtype=dtype)
            return SpatialStepOutput(
                continuous_delta=empty_cont,
                split_logits=empty_event,
                del_logits=empty_event,
            )

        h = self.encode_state(state)
        mask = state.padmask
        for block in self.blocks:
            h = block(h, mask)

        continuous_delta = self.delta_head(h) * self.dt
        split_logits = self.split_head(h)
        del_logits = self.del_head(h)

        pad_mask = state.padmask.unsqueeze(-1)
        continuous_delta = continuous_delta * pad_mask
        split_logits = split_logits * pad_mask
        del_logits = del_logits * pad_mask

        return SpatialStepOutput(
            continuous_delta=continuous_delta,
            split_logits=split_logits,
            del_logits=del_logits,
        )

    def step(
        self,
        state: BranchingState,
        deterministic: bool = False,
        apply_events: bool = True,
    ) -> tuple[BranchingState, EventDecision | None]:
        with torch.no_grad():
            output = self.forward_step(state)
            new_cont = state.states[0] + output.continuous_delta
            new_state = BranchingState(
                states=(new_cont, state.states[1].clone()),
                groupings=state.groupings,
                del_flags=state.del_flags,
                ids=state.ids,
                padmask=state.padmask,
                flowmask=state.flowmask,
                branchmask=state.branchmask,
            )
            if apply_events:
                events = self.cell_manager.sample_events(
                    output.split_logits,
                    output.del_logits,
                    deterministic=deterministic,
                    valid_mask=state.padmask,
                )
                new_state = self.cell_manager.apply_events(new_state, events)
                return new_state, events
            return new_state, None
