"""Shared outputs and local-code codec for gene-context models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class GeneContextOutput:
    gene_delta: torch.Tensor
    split_logits: torch.Tensor
    del_logits: torch.Tensor


@dataclass
class PatchSetOutput:
    pred_future_genes: torch.Tensor
    pred_patch_size: torch.Tensor
    pred_mean_gene: torch.Tensor
    patch_latent: torch.Tensor


@dataclass
class MultiPatchSetOutput:
    pred_future_genes: torch.Tensor
    pred_patch_size: torch.Tensor
    pred_mean_gene: torch.Tensor
    patch_latent: torch.Tensor
    state_latent: torch.Tensor
    patch_attention_logits: torch.Tensor
    patch_attention_weights: torch.Tensor


@dataclass
class EmbryoStateOutput:
    embryo_latent: torch.Tensor
    local_latents: torch.Tensor
    future_founder_composition: torch.Tensor
    future_celltype_composition: torch.Tensor
    future_lineage_depth_stats: torch.Tensor
    future_spatial_extent: torch.Tensor
    future_split_fraction: torch.Tensor


@dataclass
class EmbryoMaskedOutput:
    embryo_latent: torch.Tensor
    visible_embryo_latent: torch.Tensor
    local_latents: torch.Tensor
    future_local_latents: torch.Tensor | None
    pred_masked_view_latents: torch.Tensor
    pred_masked_view_genes: torch.Tensor
    pred_masked_future_view_latents: torch.Tensor | None
    pred_masked_future_view_genes: torch.Tensor | None
    masked_view_mask: torch.Tensor
    masked_future_view_mask: torch.Tensor | None


@dataclass
class EmbryoFutureSetOutput:
    context_embryo_latent: torch.Tensor
    future_local_latents: torch.Tensor
    pred_future_set_latents: torch.Tensor
    pred_future_set_genes: torch.Tensor
    pred_future_split_logits: torch.Tensor
    pred_future_count_logits: torch.Tensor
    pred_future_local_codes: torch.Tensor
    target_future_set_latents: torch.Tensor
    target_future_set_genes: torch.Tensor
    target_future_split_fraction: torch.Tensor
    target_future_count_ratio: torch.Tensor
    target_future_local_codes: torch.Tensor
    masked_view_mask: torch.Tensor
    masked_future_view_mask: torch.Tensor
    current_local_token_gate: torch.Tensor | None


@dataclass
class LocalCellDecodeOutput:
    pred_cell_genes: torch.Tensor
    pred_cell_positions: torch.Tensor
    pred_cell_valid_logits: torch.Tensor
    pred_cell_spatial_logits: torch.Tensor
    pred_cell_count: torch.Tensor
    pred_mean_gene: torch.Tensor
    pred_patch_latent: torch.Tensor


@dataclass
class LocalCellCodeOutput:
    patch_latent: torch.Tensor
    local_code_tokens: torch.Tensor
    pred_cell_genes: torch.Tensor
    pred_cell_positions: torch.Tensor
    pred_cell_valid_logits: torch.Tensor
    pred_cell_spatial_logits: torch.Tensor
    pred_cell_count: torch.Tensor
    pred_mean_gene: torch.Tensor
    pred_patch_latent: torch.Tensor


class LocalCellCodeCodec(nn.Module):
    """Continuous local-code bottleneck for structured cell-state pred-x modeling."""

    def __init__(
        self,
        gene_dim: int,
        context_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        code_tokens: int = 8,
    ):
        super().__init__()
        if code_tokens < 1:
            raise ValueError("code_tokens must be >= 1")
        self.gene_dim = gene_dim
        self.context_size = context_size
        self.code_tokens = code_tokens
        attn_heads = max(1, min(n_heads, 4))
        self.code_queries = nn.Parameter(torch.randn(code_tokens, d_model) * 0.02)
        self.code_from_patch = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.code_query_norm = nn.LayerNorm(d_model)
        self.code_memory_norm = nn.LayerNorm(d_model)
        self.code_attn = nn.MultiheadAttention(d_model, attn_heads, batch_first=True)
        self.code_ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.cell_queries = nn.Parameter(torch.randn(context_size, d_model) * 0.02)
        self.cell_from_code = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.cell_query_norm = nn.LayerNorm(d_model)
        self.cell_memory_norm = nn.LayerNorm(d_model)
        self.cell_attn = nn.MultiheadAttention(d_model, attn_heads, batch_first=True)
        self.cell_ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.gene_decode_ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.spatial_decode_ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.cell_gene_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, gene_dim),
        )
        self.cell_position_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 3),
        )
        self.cell_valid_head = nn.Linear(d_model, 1)
        self.cell_spatial_head = nn.Linear(d_model, 1)
        self.mean_gene_head = nn.Linear(d_model, gene_dim)
        self.patch_latent_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.count_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self._init_heads()

    def _init_heads(self):
        if self.cell_valid_head.bias is not None:
            nn.init.constant_(self.cell_valid_head.bias, 2.0)
        if self.cell_spatial_head.bias is not None:
            nn.init.constant_(self.cell_spatial_head.bias, 1.0)
        count_out = self.count_head[-1]
        if isinstance(count_out, nn.Linear) and count_out.bias is not None:
            nn.init.constant_(count_out.bias, 2.0)

    def encode_from_patch(
        self,
        patch_latent: torch.Tensor,
        token_states: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        code_queries = self.code_queries.unsqueeze(0) + self.code_from_patch(patch_latent).unsqueeze(1)
        code_updates, _ = self.code_attn(
            query=self.code_query_norm(code_queries),
            key=self.code_memory_norm(token_states),
            value=token_states,
            key_padding_mask=~valid_mask,
            need_weights=False,
        )
        local_code_tokens = code_queries + code_updates
        local_code_tokens = local_code_tokens + self.code_ff(local_code_tokens)
        return local_code_tokens

    def decode(self, local_code_tokens: torch.Tensor) -> LocalCellDecodeOutput:
        leading_shape = local_code_tokens.shape[:-2]
        flat_code_tokens = local_code_tokens.reshape(-1, local_code_tokens.shape[-2], local_code_tokens.shape[-1])
        pooled_code = flat_code_tokens.mean(dim=1)
        cell_queries = self.cell_queries.unsqueeze(0) + self.cell_from_code(pooled_code).unsqueeze(1)
        cell_updates, _ = self.cell_attn(
            query=self.cell_query_norm(cell_queries),
            key=self.cell_memory_norm(flat_code_tokens),
            value=flat_code_tokens,
            need_weights=False,
        )
        decoded_cells = cell_queries + cell_updates
        decoded_cells = decoded_cells + self.cell_ff(decoded_cells)
        gene_cells = decoded_cells + self.gene_decode_ff(decoded_cells)
        spatial_cells = decoded_cells + self.spatial_decode_ff(decoded_cells)
        pred_cell_genes = self.cell_gene_head(gene_cells).reshape(
            *leading_shape, self.context_size, self.gene_dim
        )
        pred_cell_positions = self.cell_position_head(spatial_cells).reshape(*leading_shape, self.context_size, 3)
        pred_cell_valid_logits = self.cell_valid_head(spatial_cells).squeeze(-1).reshape(
            *leading_shape, self.context_size
        )
        pred_cell_spatial_logits = self.cell_spatial_head(spatial_cells).squeeze(-1).reshape(
            *leading_shape, self.context_size
        )
        pooled_spatial = spatial_cells.mean(dim=1)
        pred_cell_count = self.count_head(pooled_spatial).squeeze(-1).reshape(*leading_shape)
        pred_mean_gene = self.mean_gene_head(pooled_code).reshape(*leading_shape, self.gene_dim)
        pred_patch_latent = self.patch_latent_head(pooled_code).reshape(*leading_shape, flat_code_tokens.shape[-1])
        return LocalCellDecodeOutput(
            pred_cell_genes=pred_cell_genes,
            pred_cell_positions=pred_cell_positions,
            pred_cell_valid_logits=pred_cell_valid_logits,
            pred_cell_spatial_logits=pred_cell_spatial_logits,
            pred_cell_count=pred_cell_count,
            pred_mean_gene=pred_mean_gene,
            pred_patch_latent=pred_patch_latent,
        )
