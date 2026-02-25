"""Cross-modal attention model for trimodal integration.

Separates gene expression and spatial coordinates into parallel streams
with cross-attention for explicit modality fusion.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nema_model import (
    AdaLNTransformerBlock,
    RandomFourierFeatures,
    RotaryPositionalEncoding,
)
from .states import BranchingState


class CrossModalFusion(nn.Module):
    """Cross-modal attention between gene and spatial features.

    Allows information to flow between modalities:
    - Genes attend to spatial context (where am I?)
    - Spatial attends to gene expression (what is around me?)
    """

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Cross-attention: genes -> spatial
        self.gene_to_spatial_q = nn.Linear(d_model, d_model)
        self.gene_to_spatial_kv = nn.Linear(d_model, 2 * d_model)

        # Cross-attention: spatial -> genes
        self.spatial_to_gene_q = nn.Linear(d_model, d_model)
        self.spatial_to_gene_kv = nn.Linear(d_model, 2 * d_model)

        # Output projections
        self.gene_out = nn.Linear(d_model, d_model)
        self.spatial_out = nn.Linear(d_model, d_model)

        self.scale = self.head_dim**-0.5

    def forward(
        self,
        gene_features: torch.Tensor,
        spatial_features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cross-modal fusion.

        Args:
            gene_features: [B, L, D] gene embeddings
            spatial_features: [B, L, D] spatial embeddings
            mask: Optional attention mask

        Returns:
            (updated_gene, updated_spatial) both [B, L, D]
        """
        B, L, _ = gene_features.shape

        # Gene features attend to spatial context
        q_g = self.gene_to_spatial_q(gene_features).view(
            B, L, self.n_heads, self.head_dim
        )
        kv_s = self.gene_to_spatial_kv(spatial_features).view(
            B, L, self.n_heads, 2 * self.head_dim
        )
        k_s, v_s = kv_s.chunk(2, dim=-1)

        # Spatial features attend to gene context
        q_s = self.spatial_to_gene_q(spatial_features).view(
            B, L, self.n_heads, self.head_dim
        )
        kv_g = self.spatial_to_gene_kv(gene_features).view(
            B, L, self.n_heads, 2 * self.head_dim
        )
        k_g, v_g = kv_g.chunk(2, dim=-1)

        # Cross-attention: genes query spatial
        # einsum 'blhd,bmhd->blhm' produces [B, L_q, H, L_k]
        attn_gs = torch.einsum("blhd,bmhd->blhm", q_g, k_s) * self.scale
        if mask is not None:
            # mask is [B, L_k], expand to [B, 1, H, L_k] for broadcasting
            # Shape: [B, L_k] -> [B, 1, L_k] -> [B, 1, 1, L_k] -> [B, 1, H, L_k]
            mask_expanded = (
                mask.unsqueeze(1).unsqueeze(1).expand(-1, -1, self.n_heads, -1)
            )
            attn_gs = attn_gs.masked_fill(~mask_expanded, float("-inf"))
        attn_gs = F.softmax(attn_gs, dim=-1)  # softmax over L_k dimension
        # For einsum, we need [B, L_q, H, L_k] @ [B, L_k, H, D] -> [B, L_q, H, D]
        # But v_s is [B, L, H, D], so we transpose: [B, H, L_k, D]
        gene_update = torch.einsum("blhm,bmhd->blhd", attn_gs, v_s).reshape(
            B, L, self.d_model
        )

        # Cross-attention: spatial query genes
        attn_sg = torch.einsum("blhd,bmhd->blhm", q_s, k_g) * self.scale
        if mask is not None:
            attn_sg = attn_sg.masked_fill(~mask_expanded, float("-inf"))
        attn_sg = F.softmax(attn_sg, dim=-1)
        spatial_update = torch.einsum("blhm,bmhd->blhd", attn_sg, v_g).reshape(
            B, L, self.d_model
        )

        # Output projections with residual
        gene_out = gene_features + self.gene_out(gene_update)
        spatial_out = spatial_features + self.spatial_out(spatial_update)

        return gene_out, spatial_out


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
