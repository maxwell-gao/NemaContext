"""Autoregressive developmental model.

True causal simulation: x_{t+dt} = x_t + model(x_t) * dt

Migrates from conditional flow matching to forward dynamics.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from .crossmodal_model import CrossModalFusion
from .dynamic_cell_manager import DynamicCellManager, EventDecision
from .states import BranchingState


@dataclass
class StepOutput:
    """Output of a single autoregressive step."""

    gene_delta: torch.Tensor  # [B, L, gene_dim] change in gene expression
    spatial_vel: torch.Tensor  # [B, L, 3] velocity in space
    discrete_logits: torch.Tensor  # [B, L, K] discrete state prediction
    split_logits: torch.Tensor  # [B, L, 1] logit for cell division
    del_logits: torch.Tensor  # [B, L, 1] logit for cell deletion
    noise_pred: torch.Tensor | None = None  # [B, L, gene_dim+spatial_dim]
    events: EventDecision | None = None  # Event decisions if using dynamic cells


class AutoregressiveNemaModel(nn.Module):
    """Autoregressive model for developmental simulation.

    Migrates components from CrossModalNemaModel but changes:
    - Output: state changes (delta) rather than absolute targets
    - Usage: step-by-step evolution rather than conditional sampling
    - Training: next-step prediction rather than flow matching

    Args:
        gene_dim: Dimension of gene expression (2000)
        spatial_dim: Dimension of spatial coordinates (3)
        discrete_K: Number of discrete states
        d_model: Hidden dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        cross_modal_every: Add cross-modal fusion every N layers
        dt: Time step for Euler integration
    """

    def __init__(
        self,
        gene_dim: int = 2000,
        spatial_dim: int = 3,
        discrete_K: int = 7,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        head_dim: int = 32,
        rff_dim: int = 256,
        max_seq_len: int = 2048,
        cross_modal_every: int = 2,
        dt: float = 0.02,
        deterministic_topk_events: bool = False,
    ):
        super().__init__()
        self.gene_dim = gene_dim
        self.spatial_dim = spatial_dim
        self.discrete_K = discrete_K
        self.d_model = d_model
        self.dt = dt
        self.cross_modal_every = cross_modal_every

        half_dim = d_model // 2

        # Projections (MIGRATED from CrossModalNemaModel)
        self.gene_proj = nn.Linear(gene_dim, half_dim)
        self.spatial_proj = nn.Linear(spatial_dim, half_dim)
        self.discrete_embed = nn.Embedding(discrete_K, half_dim)

        # Combine continuous modalities
        self.fusion_proj = nn.Linear(d_model, d_model)
        # Diffusion-style conditioning on noise level sigma.
        self.sigma_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.sigma_film = nn.Linear(d_model, 2 * d_model)

        # Transformer blocks (MIGRATED from CrossModalNemaModel)
        self.blocks = nn.ModuleList(
            [
                TransformerBlockAutoregressive(d_model, n_heads, head_dim)
                for _ in range(n_layers)
            ]
        )

        # Cross-modal fusion layers (MIGRATED)
        self.cross_modal_layers = nn.ModuleList(
            [
                CrossModalFusion(d_model, n_heads)
                if i % cross_modal_every == 0
                else None
                for i in range(n_layers)
            ]
        )

        # Output heads (ADAPTED: output changes, not absolute values)
        self.gene_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, gene_dim),
            nn.Tanh(),  # Constrain gene changes
        )

        self.spatial_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, spatial_dim),
            # No activation - velocity can be positive or negative
        )

        self.discrete_head = nn.Linear(d_model, discrete_K)
        # Predict diffusion noise (epsilon-style) for denoising objective.
        self.noise_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, gene_dim + spatial_dim),
        )

        # Event prediction heads (MIGRATED but repurposed)
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

        # Dynamic cell manager for division/deletion
        self.cell_manager = DynamicCellManager(
            split_threshold=0.5,
            del_threshold=0.5,
            max_cells=max_seq_len,
            use_gumbel=True,
            deterministic_topk=deterministic_topk_events,
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize with small weights for stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode_state(self, state: BranchingState) -> torch.Tensor:
        """Encode current state to latent representation.

        Args:
            state: BranchingState with continuous and discrete states

        Returns:
            [B, L, d_model] latent representation
        """
        cont = state.states[0]  # [B, L, gene_dim + spatial_dim]
        disc = state.states[1] if len(state.states) > 1 else None  # [B, L]

        B, L = cont.shape[:2]

        # Split continuous into gene and spatial
        genes = cont[..., : self.gene_dim]
        spatial = cont[..., self.gene_dim : self.gene_dim + self.spatial_dim]

        # Project
        g_emb = self.gene_proj(genes)  # [B, L, d_model//2]
        s_emb = self.spatial_proj(spatial)  # [B, L, d_model//2]

        # Combine
        combined = torch.cat([g_emb, s_emb], dim=-1)  # [B, L, d_model]

        # Add discrete embedding if available
        if disc is not None:
            d_emb = self.discrete_embed(disc.clamp(0, self.discrete_K - 1))
            # Project discrete embedding to full dimension
            combined = combined + torch.cat([d_emb, d_emb], dim=-1)

        return self.fusion_proj(combined)

    def forward_step(
        self,
        state: BranchingState,
        sigma: torch.Tensor | float | None = None,
    ) -> StepOutput:
        """Single autoregressive step: predict changes from current state.

        This is the core method that replaces conditional flow matching.
        Instead of predicting "where to go" given a target,
        it predicts "how to change" based on current state.

        Args:
            state: Current BranchingState

        Returns:
            StepOutput with predicted changes and event probabilities
        """
        B, L = state.states[0].shape[:2]
        if L == 0:
            device = state.states[0].device
            dtype = state.states[0].dtype
            empty_gene = torch.zeros(B, 0, self.gene_dim, device=device, dtype=dtype)
            empty_spatial = torch.zeros(
                B, 0, self.spatial_dim, device=device, dtype=dtype
            )
            empty_disc = torch.zeros(B, 0, self.discrete_K, device=device, dtype=dtype)
            empty_event = torch.zeros(B, 0, 1, device=device, dtype=dtype)
            empty_noise = torch.zeros(
                B, 0, self.gene_dim + self.spatial_dim, device=device, dtype=dtype
            )
            return StepOutput(
                gene_delta=empty_gene,
                spatial_vel=empty_spatial,
                discrete_logits=empty_disc,
                split_logits=empty_event,
                del_logits=empty_event,
                noise_pred=empty_noise,
            )

        # Encode current state
        h = self.encode_state(state)  # [B, L, d_model]
        # Apply sigma conditioning as FiLM over token states.
        if sigma is None:
            sigma_tensor = torch.zeros(B, device=h.device, dtype=h.dtype)
        elif not torch.is_tensor(sigma):
            sigma_tensor = torch.full(
                (B,), float(sigma), device=h.device, dtype=h.dtype
            )
        else:
            sigma_tensor = sigma.to(device=h.device, dtype=h.dtype).flatten()
            if sigma_tensor.numel() == 1 and B > 1:
                sigma_tensor = sigma_tensor.expand(B)
            if sigma_tensor.numel() != B:
                raise ValueError(
                    f"sigma batch size mismatch: expected {B}, got {sigma_tensor.numel()}"
                )

        sigma_cond = self.sigma_embed(sigma_tensor.unsqueeze(-1))  # [B, d_model]
        gamma_beta = self.sigma_film(sigma_cond)  # [B, 2*d_model]
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        h = h * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

        # Create causal mask (can only attend to previous cells)
        # For developmental simulation, we use permutation-invariant attention
        # but with optional lineage bias
        mask = state.padmask  # [B, L] - CrossModalFusion expects 2D mask

        # Transformer blocks with optional cross-modal fusion
        h_block = h

        for i, (block, cm_layer) in enumerate(
            zip(self.blocks, self.cross_modal_layers)
        ):
            # Apply transformer block first
            h_block = block(h_block, mask)

            # Apply cross-modal fusion if scheduled
            # CrossModalFusion returns (gene_out, spatial_out) - average them for simplicity
            if cm_layer is not None:
                gene_out, spatial_out = cm_layer(h_block, h_block, mask)
                h_block = (gene_out + spatial_out) / 2  # Combine both modalities

        h_out = h_block

        # Predict changes (not absolute values!)
        gene_delta = self.gene_head(h_out) * self.dt  # Constrained changes
        spatial_vel = self.spatial_head(h_out) * self.dt  # Velocity
        discrete_logits = self.discrete_head(h_out)

        # Event predictions
        split_logits = self.split_head(h_out)
        del_logits = self.del_head(h_out)
        noise_pred = self.noise_head(h_out)

        # Apply padding mask
        pad_mask = state.padmask.unsqueeze(-1)
        gene_delta = gene_delta * pad_mask
        spatial_vel = spatial_vel * pad_mask
        discrete_logits = discrete_logits * pad_mask
        split_logits = split_logits * pad_mask
        del_logits = del_logits * pad_mask
        noise_pred = noise_pred * pad_mask

        return StepOutput(
            gene_delta=gene_delta,
            spatial_vel=spatial_vel,
            discrete_logits=discrete_logits,
            split_logits=split_logits,
            del_logits=del_logits,
            noise_pred=noise_pred,
        )

    def step(
        self,
        state: BranchingState,
        deterministic: bool = False,
        apply_events: bool = True,
    ) -> tuple[BranchingState, EventDecision | None]:
        """Take one Euler step with optional cell division/deletion.

        Args:
            state: Current state
            deterministic: If True, use threshold for events; else sample
            apply_events: If True, apply cell division and deletion

        Returns:
            (new_state, event_decisions)
        """
        with torch.no_grad():
            output = self.forward_step(state)

            # Get current continuous state
            cont = state.states[0]  # [B, L, gene_dim + spatial_dim]

            # Apply state changes (movement/gene expression)
            new_genes = cont[..., : self.gene_dim] + output.gene_delta
            new_spatial = (
                cont[..., self.gene_dim : self.gene_dim + self.spatial_dim]
                + output.spatial_vel
            )

            new_cont = torch.cat([new_genes, new_spatial], dim=-1)

            # Update discrete state
            if state.states[1] is not None:
                # Founder identity is lineage metadata, not a latent state the
                # rollout should freely rewrite at each Euler step.
                new_disc = state.states[1].clone()
            else:
                new_disc = None

            # Create intermediate state
            intermediate_state = BranchingState(
                states=(new_cont, new_disc),
                groupings=state.groupings,
                del_flags=state.del_flags,
                ids=state.ids,
                padmask=state.padmask,
                flowmask=state.flowmask,
                branchmask=state.branchmask,
            )

            # Apply cell events (division/deletion) if enabled
            if apply_events:
                events = self.cell_manager.sample_events(
                    output.split_logits,
                    output.del_logits,
                    deterministic=deterministic,
                    valid_mask=state.padmask,
                )
                new_state = self.cell_manager.apply_events(intermediate_state, events)
                return new_state, events
            else:
                return intermediate_state, None

    def simulate(
        self,
        initial_state: BranchingState,
        n_steps: int,
        deterministic: bool = True,
    ) -> list[BranchingState]:
        """Simulate development for n_steps.

        Args:
            initial_state: Starting state
            n_steps: Number of Euler steps
            deterministic: If True, use hard thresholds for events

        Returns:
            List of states [initial, step1, step2, ..., step_n]
        """
        trajectory = [initial_state]
        state = initial_state

        for _ in range(n_steps):
            state, _ = self.step(state, deterministic=deterministic, apply_events=True)
            trajectory.append(state)

        return trajectory


class TransformerBlockAutoregressive(nn.Module):
    """Transformer block without time conditioning (simpler than flow version)."""

    def __init__(self, d_model: int, n_heads: int, head_dim: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim

        # Self-attention
        self.qkv = nn.Linear(d_model, 3 * n_heads * head_dim)
        self.attn_out = nn.Linear(n_heads * head_dim, d_model)

        # Feedforward
        self.ff_gate = nn.Linear(d_model, d_model * 4)
        self.ff_up = nn.Linear(d_model, d_model * 4)
        self.ff_out = nn.Linear(d_model * 4, d_model)

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, L, d_model]
            mask: [B, L] padding mask

        Returns:
            [B, L, d_model]
        """
        B, L, D = x.shape

        # Self-attention with residual
        h = self.norm1(x)

        qkv = self.qkv(h).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        # Scaled dot-product attention
        q = q.transpose(1, 2)  # [B, H, L, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

        # Apply mask [B, L] -> [B, 1, 1, L] for broadcasting
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask_expanded == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, 0.0)  # Handle all-masked positions

        out = torch.matmul(attn, v)  # [B, H, L, head_dim]
        out = out.transpose(1, 2).reshape(B, L, -1)
        out = self.attn_out(out)

        x = x + out

        # Feedforward with residual (SwiGLU)
        h = self.norm2(x)
        gate = F.silu(self.ff_gate(h))
        up = self.ff_up(h)
        x = x + self.ff_out(gate * up)

        return x


def autoregressive_loss(
    output: StepOutput,
    next_state: BranchingState,
    current_state: BranchingState,
    gamma: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """Compute autoregressive training loss.

    Args:
        output: Model predictions from current state
        next_state: True next state
        current_state: Current state (for computing deltas)
        gamma: Weight for event prediction losses

    Returns:
        (total_loss, loss_dict)
    """
    # Extract true deltas
    cont_current = current_state.states[0]
    cont_next = next_state.states[0]

    true_gene_delta = cont_next[..., :2000] - cont_current[..., :2000]
    true_spatial_vel = cont_next[..., 2000:2003] - cont_current[..., 2000:2003]

    # State prediction losses
    gene_loss = F.mse_loss(output.gene_delta, true_gene_delta)
    spatial_loss = F.mse_loss(output.spatial_vel, true_spatial_vel)

    # Discrete state loss (if available)
    if next_state.states[1] is not None:
        disc_loss = F.cross_entropy(
            output.discrete_logits.reshape(-1, output.discrete_logits.size(-1)),
            next_state.states[1].reshape(-1),
            ignore_index=-100,
        )
    else:
        disc_loss = torch.tensor(0.0, device=gene_loss.device)

    # Total state loss
    state_loss = gene_loss + spatial_loss + 0.1 * disc_loss

    # Event losses (need target events from data)
    # For now, assume we don't have ground truth events
    # In full version, these would come from tracking cell divisions
    split_loss = torch.tensor(0.0, device=gene_loss.device)
    del_loss = torch.tensor(0.0, device=gene_loss.device)

    total_loss = state_loss + gamma * (split_loss + del_loss)

    loss_dict = {
        "total": total_loss.item(),
        "gene": gene_loss.item(),
        "spatial": spatial_loss.item(),
        "discrete": disc_loss.item(),
        "split": split_loss.item(),
        "delete": del_loss.item(),
    }

    return total_loss, loss_dict
