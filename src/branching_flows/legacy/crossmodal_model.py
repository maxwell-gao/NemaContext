"""Cross-modal attention model for trimodal integration.

Separates gene expression and spatial coordinates into parallel streams
with cross-attention for explicit modality fusion.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..fusion import CrossModalFusion
from .nema_model import (
    AdaLNTransformerBlock,
    RandomFourierFeatures,
    RotaryPositionalEncoding,
)
from ..states import BranchingState


class CrossModalNemaModel(nn.Module):
    """NemaFlowModel with explicit cross-modal attention.

    Architecture:
    1. Separate projections for genes and spatial
    2. Parallel Transformer streams
    3. Periodic cross-modal fusion layers
    4. Combined output heads
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
        cross_modal_every: int = 2,  # Add cross-attention every N layers
    ):
        super().__init__()
        self.d_model = d_model
        self.gene_dim = gene_dim
        self.spatial_dim = spatial_dim
        self.discrete_K = discrete_K
        self.cross_modal_every = cross_modal_every

        half_dim = d_model // 2

        # Separate input projections
        self.gene_proj = nn.Linear(gene_dim, half_dim)
        self.spatial_proj = nn.Linear(spatial_dim, half_dim)
        self.discrete_embed = nn.Embedding(discrete_K, half_dim)

        # Time conditioning
        self.rff = RandomFourierFeatures(1, rff_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(rff_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # RoPE
        self.rope = RotaryPositionalEncoding(head_dim, max_seq_len)

        # Parallel Transformer blocks
        self.gene_blocks = nn.ModuleList()
        self.spatial_blocks = nn.ModuleList()
        self.cross_modal_layers = nn.ModuleList()

        for i in range(n_layers):
            self.gene_blocks.append(
                AdaLNTransformerBlock(half_dim, n_heads // 2, head_dim)
            )
            self.spatial_blocks.append(
                AdaLNTransformerBlock(half_dim, n_heads // 2, head_dim)
            )

            # Add cross-modal fusion every N layers
            if (i + 1) % cross_modal_every == 0:
                self.cross_modal_layers.append(CrossModalFusion(half_dim, n_heads // 2))
            else:
                self.cross_modal_layers.append(None)

        # Fusion and output
        self.fusion_proj = nn.Linear(d_model, d_model)

        # Output heads
        self.gene_head = nn.Linear(d_model, gene_dim)
        self.spatial_head = nn.Linear(d_model, spatial_dim)
        self.discrete_head = nn.Linear(d_model, discrete_K)
        self.split_head = nn.Linear(d_model, 1)
        self.del_head = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        for block in self.gene_blocks:
            nn.init.zeros_(block.attn_out.weight)
            nn.init.zeros_(block.ff_out.weight)
        for block in self.spatial_blocks:
            nn.init.zeros_(block.attn_out.weight)
            nn.init.zeros_(block.ff_out.weight)

    def forward(
        self,
        t: torch.Tensor,
        state: BranchingState,
        lineage_bias: torch.Tensor | None = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Forward pass with cross-modal attention.

        Args:
            t: Flow time, shape (B,)
            state: BranchingState with continuous and discrete states
            lineage_bias: Optional lineage-based attention bias

        Returns:
            ((gene_pred, spatial_pred), discrete_logits, split_logits, del_logits)
        """
        cont = state.states[0]  # (B, L, D_cont)
        disc = state.states[1].long()  # (B, L)
        padmask = state.padmask  # (B, L)

        B, L, _ = cont.shape

        # Split continuous into genes and spatial
        genes = cont[..., : self.gene_dim]  # (B, L, gene_dim)
        spatial = cont[..., self.gene_dim :]  # (B, L, spatial_dim)

        # Separate projections
        g = self.gene_proj(genes) + self.discrete_embed(disc)  # (B, L, half_dim)
        s = self.spatial_proj(spatial) + self.discrete_embed(disc)  # (B, L, half_dim)

        # Time conditioning
        t_feat = self.rff(t.unsqueeze(-1))  # (B, rff_dim)
        t_cond = self.time_proj(t_feat)  # (B, d_model)
        t_cond_g = t_cond[:, : self.d_model // 2]
        t_cond_s = t_cond[:, self.d_model // 2 :]

        # RoPE
        rope_cos, rope_sin = self.rope(L, device=cont.device)

        # Attention mask
        attn_mask = padmask.unsqueeze(1).unsqueeze(2)

        # Process through parallel streams with cross-modal fusion
        for i, (g_block, s_block, cm_layer) in enumerate(
            zip(self.gene_blocks, self.spatial_blocks, self.cross_modal_layers)
        ):
            # Self-attention in each modality
            g = g_block(g, t_cond_g, rope_cos, rope_sin, attn_mask, lineage_bias)
            s = s_block(s, t_cond_s, rope_cos, rope_sin, attn_mask, lineage_bias)

            # Cross-modal fusion (if applicable)
            if cm_layer is not None:
                g, s = cm_layer(g, s, padmask)

        # Concatenate and fuse
        fused = torch.cat([g, s], dim=-1)  # (B, L, d_model)
        fused = self.fusion_proj(fused)

        # Output heads
        gene_pred = self.gene_head(fused)
        spatial_pred = self.spatial_head(fused)
        discrete_logits = self.discrete_head(fused)
        split_logits = self.split_head(fused).squeeze(-1)
        del_logits = self.del_head(fused).squeeze(-1)

        # Combine predictions
        cont_pred = torch.cat([gene_pred, spatial_pred], dim=-1)

        return (cont_pred, discrete_logits), split_logits, del_logits
