"""Cross-modal fusion utilities used by the active autoregressive model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalFusion(nn.Module):
    """Cross-modal attention between gene and spatial features."""

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.gene_to_spatial_q = nn.Linear(d_model, d_model)
        self.gene_to_spatial_kv = nn.Linear(d_model, 2 * d_model)

        self.spatial_to_gene_q = nn.Linear(d_model, d_model)
        self.spatial_to_gene_kv = nn.Linear(d_model, 2 * d_model)

        self.gene_out = nn.Linear(d_model, d_model)
        self.spatial_out = nn.Linear(d_model, d_model)

        self.scale = self.head_dim**-0.5

    def forward(
        self,
        gene_features: torch.Tensor,
        spatial_features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse gene and spatial token features with cross-attention."""
        B, L, _ = gene_features.shape

        q_g = self.gene_to_spatial_q(gene_features).view(
            B, L, self.n_heads, self.head_dim
        )
        kv_s = self.gene_to_spatial_kv(spatial_features).view(
            B, L, self.n_heads, 2 * self.head_dim
        )
        k_s, v_s = kv_s.chunk(2, dim=-1)

        q_s = self.spatial_to_gene_q(spatial_features).view(
            B, L, self.n_heads, self.head_dim
        )
        kv_g = self.spatial_to_gene_kv(gene_features).view(
            B, L, self.n_heads, 2 * self.head_dim
        )
        k_g, v_g = kv_g.chunk(2, dim=-1)

        attn_gs = torch.einsum("blhd,bmhd->blhm", q_g, k_s) * self.scale
        if mask is not None:
            mask_expanded = (
                mask.unsqueeze(1).unsqueeze(1).expand(-1, -1, self.n_heads, -1)
            )
            attn_gs = attn_gs.masked_fill(~mask_expanded, float("-inf"))
        attn_gs = F.softmax(attn_gs, dim=-1)
        gene_update = torch.einsum("blhm,bmhd->blhd", attn_gs, v_s).reshape(
            B, L, self.d_model
        )

        attn_sg = torch.einsum("blhd,bmhd->blhm", q_s, k_g) * self.scale
        if mask is not None:
            attn_sg = attn_sg.masked_fill(~mask_expanded, float("-inf"))
        attn_sg = F.softmax(attn_sg, dim=-1)
        spatial_update = torch.einsum("blhm,bmhd->blhd", attn_sg, v_g).reshape(
            B, L, self.d_model
        )

        gene_out = gene_features + self.gene_out(gene_update)
        spatial_out = spatial_features + self.spatial_out(spatial_update)
        return gene_out, spatial_out
