"""Multi-cell gene-context baseline model."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .autoregressive_model import TransformerBlockAutoregressive


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
class EmbryoOneStepOutput:
    current_embryo_latent: torch.Tensor
    target_future_embryo_latent: torch.Tensor
    pred_future_embryo_latent: torch.Tensor
    target_future_delta: torch.Tensor | None
    pred_future_delta: torch.Tensor | None
    current_prediction_space: torch.Tensor | None
    target_prediction_space: torch.Tensor | None
    pred_prediction_space: torch.Tensor | None
    future_founder_composition: torch.Tensor
    future_celltype_composition: torch.Tensor
    future_lineage_depth_stats: torch.Tensor
    future_spatial_extent: torch.Tensor
    future_split_fraction: torch.Tensor


@dataclass
class EmbryoJEPAOutput:
    context_embryo_latent: torch.Tensor
    target_masked_future_latent: torch.Tensor
    pred_masked_future_latent: torch.Tensor
    masked_view_mask: torch.Tensor
    masked_future_view_mask: torch.Tensor


@dataclass
class EmbryoFutureSetOutput:
    context_embryo_latent: torch.Tensor
    future_local_latents: torch.Tensor
    pred_future_set_latents: torch.Tensor
    pred_future_set_genes: torch.Tensor
    pred_future_cell_tokens: torch.Tensor | None
    target_future_set_latents: torch.Tensor
    target_future_set_genes: torch.Tensor
    target_future_cell_tokens: torch.Tensor | None
    masked_view_mask: torch.Tensor
    masked_future_view_mask: torch.Tensor
    current_local_token_gate: torch.Tensor | None


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


class GeneContextModel(nn.Module):
    """Time-conditioned multi-cell transformer over gene states."""

    def __init__(
        self,
        gene_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        head_dim: int = 32,
        use_pairwise_spatial_bias: bool = False,
    ):
        super().__init__()
        self.gene_dim = gene_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.use_pairwise_spatial_bias = use_pairwise_spatial_bias
        self.gene_proj = nn.Sequential(
            nn.Linear(gene_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.time_proj = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.context_role_emb = nn.Embedding(4, d_model)
        self.relative_spatial_proj = nn.Sequential(
            nn.Linear(5, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        if use_pairwise_spatial_bias:
            self.pairwise_spatial_bias = nn.Sequential(
                nn.Linear(5, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, n_heads),
            )
        else:
            self.pairwise_spatial_bias = None
        self.blocks = nn.ModuleList(
            [TransformerBlockAutoregressive(d_model, n_heads, head_dim) for _ in range(n_layers)]
        )
        self.gene_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, gene_dim),
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
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _build_pairwise_spatial_bias(
        self,
        relative_position: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.pairwise_spatial_bias is None:
            return None

        coords = relative_position[..., :3]
        has_spatial = relative_position[..., 4]
        pairwise_delta = coords.unsqueeze(2) - coords.unsqueeze(1)
        pairwise_radius = torch.linalg.norm(pairwise_delta, dim=-1, keepdim=True)
        pairwise_valid = (
            has_spatial.unsqueeze(2) * has_spatial.unsqueeze(1) * valid_mask.unsqueeze(2).float() * valid_mask.unsqueeze(1).float()
        ).unsqueeze(-1)
        pairwise_features = torch.cat(
            [pairwise_delta, pairwise_radius, pairwise_valid],
            dim=-1,
        )
        pairwise_features = pairwise_features * pairwise_valid
        bias = self.pairwise_spatial_bias(pairwise_features)
        bias = bias.permute(0, 3, 1, 2)
        return bias

    def forward(
        self,
        genes: torch.Tensor,
        time: torch.Tensor,
        future_time: torch.Tensor,
        token_times: torch.Tensor,
        valid_mask: torch.Tensor,
        context_role: torch.Tensor | None = None,
        relative_position: torch.Tensor | None = None,
    ) -> GeneContextOutput:
        x = self.encode_tokens(
            genes=genes,
            time=time,
            future_time=future_time,
            token_times=token_times,
            valid_mask=valid_mask,
            context_role=context_role,
            relative_position=relative_position,
        )

        gene_delta = self.gene_head(x)
        split_logits = self.split_head(x).squeeze(-1)
        del_logits = self.del_head(x).squeeze(-1)

        gene_delta = gene_delta * valid_mask.unsqueeze(-1)
        split_logits = split_logits * valid_mask
        del_logits = del_logits * valid_mask
        return GeneContextOutput(
            gene_delta=gene_delta,
            split_logits=split_logits,
            del_logits=del_logits,
        )

    def encode_tokens(
        self,
        genes: torch.Tensor,
        time: torch.Tensor,
        future_time: torch.Tensor,
        token_times: torch.Tensor,
        valid_mask: torch.Tensor,
        context_role: torch.Tensor | None = None,
        relative_position: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.gene_proj(genes)
        global_time = torch.stack([time, future_time - time], dim=-1)
        time_emb = self.time_proj(global_time).unsqueeze(1)
        x = x + time_emb
        x = x + token_times.unsqueeze(-1)
        if context_role is not None:
            x = x + self.context_role_emb(context_role.clamp(min=0, max=3))
        if relative_position is not None:
            x = x + self.relative_spatial_proj(relative_position)
            pairwise_bias = self._build_pairwise_spatial_bias(relative_position, valid_mask)
        else:
            pairwise_bias = None

        for block in self.blocks:
            x = block(x, valid_mask, attn_bias=pairwise_bias)

        return x

    @staticmethod
    def pool_patch(
        token_states: torch.Tensor,
        valid_mask: torch.Tensor,
        anchor_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask_f = valid_mask.float()
        pooled = (token_states * mask_f.unsqueeze(-1)).sum(dim=1) / mask_f.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        if anchor_mask is None:
            anchor_state = pooled
        else:
            anchor_f = anchor_mask.float()
            anchor_state = (token_states * anchor_f.unsqueeze(-1)).sum(dim=1) / anchor_f.sum(
                dim=1, keepdim=True
            ).clamp_min(1.0)
        return torch.cat([anchor_state, pooled], dim=-1)


class SingleCellGeneTimeModel(nn.Module):
    """Single-cell gene+time baseline without multi-cell context."""

    def __init__(
        self,
        gene_dim: int,
        d_model: int = 256,
        n_layers: int = 3,
    ):
        super().__init__()
        self.gene_dim = gene_dim
        self.d_model = d_model
        self.relative_spatial_proj = nn.Sequential(
            nn.Linear(5, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.gene_proj = nn.Sequential(
            nn.Linear(gene_dim + 3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        mlp_layers: list[nn.Module] = []
        for _ in range(n_layers):
            mlp_layers.extend(
                [
                    nn.Linear(d_model, d_model),
                    nn.LayerNorm(d_model),
                    nn.GELU(),
                ]
            )
        self.backbone = nn.Sequential(*mlp_layers)
        self.gene_head = nn.Linear(d_model, gene_dim)
        self.split_head = nn.Linear(d_model, 1)
        self.del_head = nn.Linear(d_model, 1)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        genes: torch.Tensor,
        time: torch.Tensor,
        future_time: torch.Tensor,
        token_times: torch.Tensor,
        valid_mask: torch.Tensor,
        context_role: torch.Tensor | None = None,
        relative_position: torch.Tensor | None = None,
    ) -> GeneContextOutput:
        x = self.encode_tokens(
            genes=genes,
            time=time,
            future_time=future_time,
            token_times=token_times,
            valid_mask=valid_mask,
            context_role=context_role,
            relative_position=relative_position,
        )

        gene_delta = self.gene_head(x) * valid_mask.unsqueeze(-1)
        split_logits = self.split_head(x).squeeze(-1) * valid_mask
        del_logits = self.del_head(x).squeeze(-1) * valid_mask
        return GeneContextOutput(
            gene_delta=gene_delta,
            split_logits=split_logits,
            del_logits=del_logits,
        )

    def encode_tokens(
        self,
        genes: torch.Tensor,
        time: torch.Tensor,
        future_time: torch.Tensor,
        token_times: torch.Tensor,
        valid_mask: torch.Tensor,
        context_role: torch.Tensor | None = None,
        relative_position: torch.Tensor | None = None,
    ) -> torch.Tensor:
        global_time = time.unsqueeze(1).expand_as(token_times)
        delta_time = (future_time - time).unsqueeze(1).expand_as(token_times)
        features = torch.cat(
            [
                genes,
                global_time.unsqueeze(-1),
                delta_time.unsqueeze(-1),
                token_times.unsqueeze(-1),
            ],
            dim=-1,
        )
        x = self.gene_proj(features)
        if relative_position is not None:
            x = x + self.relative_spatial_proj(relative_position)
        x = self.backbone(x)
        return x

    @staticmethod
    def pool_patch(
        token_states: torch.Tensor,
        valid_mask: torch.Tensor,
        anchor_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask_f = valid_mask.float()
        pooled = (token_states * mask_f.unsqueeze(-1)).sum(dim=1) / mask_f.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        if anchor_mask is None:
            anchor_state = pooled
        else:
            anchor_f = anchor_mask.float()
            anchor_state = (token_states * anchor_f.unsqueeze(-1)).sum(dim=1) / anchor_f.sum(
                dim=1, keepdim=True
            ).clamp_min(1.0)
        return torch.cat([anchor_state, pooled], dim=-1)


class MultiCellPatchSetModel(nn.Module):
    """Patch-to-patch set predictor built on the multi-cell encoder."""

    def __init__(
        self,
        gene_dim: int,
        context_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        head_dim: int = 32,
        use_pairwise_spatial_bias: bool = True,
    ):
        super().__init__()
        self.context_size = context_size
        self.encoder = GeneContextModel(
            gene_dim=gene_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            head_dim=head_dim,
            use_pairwise_spatial_bias=use_pairwise_spatial_bias,
        )
        self.patch_proj = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.future_token_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, context_size * gene_dim),
        )
        self.patch_size_head = nn.Linear(d_model, 1)
        self.mean_gene_head = nn.Linear(d_model, gene_dim)

    def encode_patch(
        self,
        genes: torch.Tensor,
        time: torch.Tensor,
        future_time: torch.Tensor,
        token_times: torch.Tensor,
        valid_mask: torch.Tensor,
        anchor_mask: torch.Tensor,
        context_role: torch.Tensor | None = None,
        relative_position: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_states = self.encoder.encode_tokens(
            genes=genes,
            time=time,
            future_time=future_time,
            token_times=token_times,
            valid_mask=valid_mask,
            context_role=context_role,
            relative_position=relative_position,
        )
        pooled = self.encoder.pool_patch(token_states, valid_mask, anchor_mask)
        patch_latent = self.patch_proj(pooled)
        return patch_latent, token_states

    def forward(
        self,
        genes: torch.Tensor,
        time: torch.Tensor,
        future_time: torch.Tensor,
        token_times: torch.Tensor,
        valid_mask: torch.Tensor,
        anchor_mask: torch.Tensor,
        context_role: torch.Tensor | None = None,
        relative_position: torch.Tensor | None = None,
    ) -> PatchSetOutput:
        patch_latent, _ = self.encode_patch(
            genes=genes,
            time=time,
            future_time=future_time,
            token_times=token_times,
            valid_mask=valid_mask,
            anchor_mask=anchor_mask,
            context_role=context_role,
            relative_position=relative_position,
        )
        pred_future_genes = self.future_token_head(patch_latent).view(
            genes.shape[0], self.context_size, self.encoder.gene_dim
        )
        pred_patch_size = self.patch_size_head(patch_latent).squeeze(-1)
        pred_mean_gene = self.mean_gene_head(patch_latent)
        return PatchSetOutput(
            pred_future_genes=pred_future_genes,
            pred_patch_size=pred_patch_size,
            pred_mean_gene=pred_mean_gene,
            patch_latent=patch_latent,
        )


class SingleCellPatchSetModel(nn.Module):
    """Patch-to-patch set predictor built on the single-cell encoder."""

    def __init__(
        self,
        gene_dim: int,
        context_size: int,
        d_model: int = 256,
        n_layers: int = 3,
    ):
        super().__init__()
        self.context_size = context_size
        self.encoder = SingleCellGeneTimeModel(
            gene_dim=gene_dim,
            d_model=d_model,
            n_layers=n_layers,
        )
        self.patch_proj = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.future_token_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, context_size * gene_dim),
        )
        self.patch_size_head = nn.Linear(d_model, 1)
        self.mean_gene_head = nn.Linear(d_model, gene_dim)

    def encode_patch(
        self,
        genes: torch.Tensor,
        time: torch.Tensor,
        future_time: torch.Tensor,
        token_times: torch.Tensor,
        valid_mask: torch.Tensor,
        anchor_mask: torch.Tensor,
        context_role: torch.Tensor | None = None,
        relative_position: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_states = self.encoder.encode_tokens(
            genes=genes,
            time=time,
            future_time=future_time,
            token_times=token_times,
            valid_mask=valid_mask,
            context_role=context_role,
            relative_position=relative_position,
        )
        pooled = self.encoder.pool_patch(token_states, valid_mask, anchor_mask)
        patch_latent = self.patch_proj(pooled)
        return patch_latent, token_states

    def forward(
        self,
        genes: torch.Tensor,
        time: torch.Tensor,
        future_time: torch.Tensor,
        token_times: torch.Tensor,
        valid_mask: torch.Tensor,
        anchor_mask: torch.Tensor,
        context_role: torch.Tensor | None = None,
        relative_position: torch.Tensor | None = None,
    ) -> PatchSetOutput:
        patch_latent, _ = self.encode_patch(
            genes=genes,
            time=time,
            future_time=future_time,
            token_times=token_times,
            valid_mask=valid_mask,
            anchor_mask=anchor_mask,
            context_role=context_role,
            relative_position=relative_position,
        )
        pred_future_genes = self.future_token_head(patch_latent).view(
            genes.shape[0], self.context_size, self.encoder.gene_dim
        )
        pred_patch_size = self.patch_size_head(patch_latent).squeeze(-1)
        pred_mean_gene = self.mean_gene_head(patch_latent)
        return PatchSetOutput(
            pred_future_genes=pred_future_genes,
            pred_patch_size=pred_patch_size,
            pred_mean_gene=pred_mean_gene,
            patch_latent=patch_latent,
        )


class LocalCellCodeModel(nn.Module):
    """Local patch autoencoder with a compact cell-code bottleneck.

    The target is a structured local cell state, not just genes: predicted cell
    tokens decode gene expression, relative positions, and local cardinality.
    """

    def __init__(
        self,
        gene_dim: int,
        context_size: int,
        model_type: str = "multi_cell",
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        head_dim: int = 32,
        use_pairwise_spatial_bias: bool = True,
        code_tokens: int = 8,
    ):
        super().__init__()
        if code_tokens < 1:
            raise ValueError("code_tokens must be >= 1")
        if model_type == "single_cell":
            self.patch_model = SingleCellPatchSetModel(
                gene_dim=gene_dim,
                context_size=context_size,
                d_model=d_model,
                n_layers=n_layers,
            )
        else:
            self.patch_model = MultiCellPatchSetModel(
                gene_dim=gene_dim,
                context_size=context_size,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                head_dim=head_dim,
                use_pairwise_spatial_bias=use_pairwise_spatial_bias,
            )
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

    def encode_patch(
        self,
        genes: torch.Tensor,
        time: torch.Tensor,
        token_times: torch.Tensor,
        valid_mask: torch.Tensor,
        anchor_mask: torch.Tensor,
        context_role: torch.Tensor | None = None,
        relative_position: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.patch_model.encode_patch(
            genes=genes,
            time=time,
            future_time=time,
            token_times=token_times,
            valid_mask=valid_mask,
            anchor_mask=anchor_mask,
            context_role=context_role,
            relative_position=relative_position,
        )

    def forward(
        self,
        genes: torch.Tensor,
        time: torch.Tensor,
        token_times: torch.Tensor,
        valid_mask: torch.Tensor,
        anchor_mask: torch.Tensor,
        context_role: torch.Tensor | None = None,
        relative_position: torch.Tensor | None = None,
    ) -> LocalCellCodeOutput:
        patch_latent, token_states = self.encode_patch(
            genes=genes,
            time=time,
            token_times=token_times,
            valid_mask=valid_mask,
            anchor_mask=anchor_mask,
            context_role=context_role,
            relative_position=relative_position,
        )

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

        pooled_code = local_code_tokens.mean(dim=1)
        cell_queries = self.cell_queries.unsqueeze(0) + self.cell_from_code(pooled_code).unsqueeze(1)
        cell_updates, _ = self.cell_attn(
            query=self.cell_query_norm(cell_queries),
            key=self.cell_memory_norm(local_code_tokens),
            value=local_code_tokens,
            need_weights=False,
        )
        decoded_cells = cell_queries + cell_updates
        decoded_cells = decoded_cells + self.cell_ff(decoded_cells)

        return LocalCellCodeOutput(
            patch_latent=patch_latent,
            local_code_tokens=local_code_tokens,
            pred_cell_genes=self.cell_gene_head(decoded_cells),
            pred_cell_positions=self.cell_position_head(decoded_cells),
            pred_cell_valid_logits=self.cell_valid_head(decoded_cells).squeeze(-1),
            pred_cell_spatial_logits=self.cell_spatial_head(decoded_cells).squeeze(-1),
            pred_cell_count=self.count_head(pooled_code).squeeze(-1),
            pred_mean_gene=self.mean_gene_head(pooled_code),
            pred_patch_latent=self.patch_latent_head(pooled_code),
        )


class MultiPatchSetModel(nn.Module):
    """Patch-count extrapolation model over multiple patches per state."""

    def __init__(
        self,
        gene_dim: int,
        context_size: int,
        model_type: str = "multi_cell",
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        head_dim: int = 32,
        use_pairwise_spatial_bias: bool = True,
    ):
        super().__init__()
        if model_type == "single_cell":
            self.patch_model = SingleCellPatchSetModel(
                gene_dim=gene_dim,
                context_size=context_size,
                d_model=d_model,
                n_layers=n_layers,
            )
        else:
            self.patch_model = MultiCellPatchSetModel(
                gene_dim=gene_dim,
                context_size=context_size,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                head_dim=head_dim,
                use_pairwise_spatial_bias=use_pairwise_spatial_bias,
            )
        self.model_type = model_type
        self.patch_blocks = nn.ModuleList(
            [TransformerBlockAutoregressive(d_model, n_heads=max(1, min(n_heads, 4)), head_dim=head_dim) for _ in range(2)]
        )
        self.patch_score = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.state_proj = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.patch_film = nn.Linear(d_model, 2 * d_model)

    def encode_state(
        self,
        genes: torch.Tensor,
        time: torch.Tensor,
        future_time: torch.Tensor,
        token_times: torch.Tensor,
        valid_mask: torch.Tensor,
        anchor_mask: torch.Tensor,
        context_role: torch.Tensor | None = None,
        relative_position: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, n_patches, patch_len, gene_dim = genes.shape
        flat_kwargs = {
            "genes": genes.view(batch_size * n_patches, patch_len, gene_dim),
            "time": time.view(batch_size * n_patches),
            "future_time": future_time.view(batch_size * n_patches),
            "token_times": token_times.view(batch_size * n_patches, patch_len),
            "valid_mask": valid_mask.view(batch_size * n_patches, patch_len),
            "anchor_mask": anchor_mask.view(batch_size * n_patches, patch_len),
            "context_role": None if context_role is None else context_role.view(batch_size * n_patches, patch_len),
            "relative_position": None if relative_position is None else relative_position.view(batch_size * n_patches, patch_len, relative_position.shape[-1]),
        }
        patch_latent, _ = self.patch_model.encode_patch(**flat_kwargs)
        patch_latent = patch_latent.view(batch_size, n_patches, -1)
        patch_valid = valid_mask.any(dim=-1)
        x = patch_latent
        for block in self.patch_blocks:
            x = block(x, patch_valid)
        patch_attention_logits = self.patch_score(x).squeeze(-1)
        patch_attention_logits = patch_attention_logits.masked_fill(~patch_valid, float("-inf"))
        patch_attention_weights = torch.softmax(patch_attention_logits, dim=1)
        patch_attention_weights = patch_attention_weights.masked_fill(~patch_valid, 0.0)
        patch_attention_weights = patch_attention_weights / patch_attention_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        pooled = (x * patch_attention_weights.unsqueeze(-1)).sum(dim=1)
        anchor_patch = x[:, 0]
        state_latent = self.state_proj(torch.cat([anchor_patch, pooled], dim=-1))
        return state_latent, x, patch_attention_logits, patch_attention_weights

    def forward(
        self,
        genes: torch.Tensor,
        time: torch.Tensor,
        future_time: torch.Tensor,
        token_times: torch.Tensor,
        valid_mask: torch.Tensor,
        anchor_mask: torch.Tensor,
        context_role: torch.Tensor | None = None,
        relative_position: torch.Tensor | None = None,
    ) -> MultiPatchSetOutput:
        state_latent, patch_latent, patch_attention_logits, patch_attention_weights = self.encode_state(
            genes=genes,
            time=time,
            future_time=future_time,
            token_times=token_times,
            valid_mask=valid_mask,
            anchor_mask=anchor_mask,
            context_role=context_role,
            relative_position=relative_position,
        )
        scale, shift = self.patch_film(state_latent).chunk(2, dim=-1)
        attention_gain = 1.0 + patch_attention_weights.unsqueeze(-1)
        conditioned = (patch_latent * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)) * attention_gain
        flat = conditioned.view(conditioned.shape[0] * conditioned.shape[1], conditioned.shape[2])
        pred_future_genes = self.patch_model.future_token_head(flat).view(
            genes.shape[0],
            genes.shape[1],
            self.patch_model.context_size,
            self.patch_model.encoder.gene_dim,
        )
        pred_patch_size = self.patch_model.patch_size_head(flat).view(genes.shape[0], genes.shape[1])
        pred_mean_gene = self.patch_model.mean_gene_head(flat).view(
            genes.shape[0],
            genes.shape[1],
            self.patch_model.encoder.gene_dim,
        )
        return MultiPatchSetOutput(
            pred_future_genes=pred_future_genes,
            pred_patch_size=pred_patch_size,
            pred_mean_gene=pred_mean_gene,
            patch_latent=conditioned,
            state_latent=state_latent,
            patch_attention_logits=patch_attention_logits,
            patch_attention_weights=patch_attention_weights,
        )


class EmbryoStateModel(nn.Module):
    """Embryo-scale state encoder from multiple local observation views.

    Patches are views, not ontology. The model encodes each view with the shared
    local patch encoder and pools them into one embryo-level latent.
    """

    def __init__(
        self,
        gene_dim: int,
        context_size: int,
        celltype_dim: int,
        model_type: str = "multi_cell",
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        head_dim: int = 32,
        use_pairwise_spatial_bias: bool = True,
    ):
        super().__init__()
        if model_type == "single_cell":
            self.local_model = SingleCellPatchSetModel(
                gene_dim=gene_dim,
                context_size=context_size,
                d_model=d_model,
                n_layers=n_layers,
            )
        else:
            self.local_model = MultiCellPatchSetModel(
                gene_dim=gene_dim,
                context_size=context_size,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                head_dim=head_dim,
                use_pairwise_spatial_bias=use_pairwise_spatial_bias,
            )
        self.state_proj = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.future_founder_head = nn.Linear(d_model, 8)
        self.future_celltype_head = nn.Linear(d_model, celltype_dim)
        self.future_depth_head = nn.Linear(d_model, 3)
        self.future_spatial_head = nn.Linear(d_model, 4)
        self.future_split_head = nn.Linear(d_model, 1)

    def encode_embryo(
        self,
        genes: torch.Tensor,
        time: torch.Tensor,
        token_times: torch.Tensor,
        valid_mask: torch.Tensor,
        anchor_mask: torch.Tensor,
        context_role: torch.Tensor | None = None,
        relative_position: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, n_views, patch_len, gene_dim = genes.shape
        flat_kwargs = {
            "genes": genes.view(batch_size * n_views, patch_len, gene_dim),
            "time": time.view(batch_size * n_views),
            "future_time": time.view(batch_size * n_views),
            "token_times": token_times.view(batch_size * n_views, patch_len),
            "valid_mask": valid_mask.view(batch_size * n_views, patch_len),
            "anchor_mask": anchor_mask.view(batch_size * n_views, patch_len),
            "context_role": None if context_role is None else context_role.view(batch_size * n_views, patch_len),
            "relative_position": None if relative_position is None else relative_position.view(
                batch_size * n_views, patch_len, relative_position.shape[-1]
            ),
        }
        local_latents, _ = self.local_model.encode_patch(**flat_kwargs)
        local_latents = local_latents.view(batch_size, n_views, -1)
        valid_views = valid_mask.any(dim=-1)
        pooled = (local_latents * valid_views.unsqueeze(-1).float()).sum(dim=1) / valid_views.float().sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        anchor_view = local_latents[:, 0]
        embryo_latent = self.state_proj(torch.cat([anchor_view, pooled], dim=-1))
        return embryo_latent, local_latents

    def forward(
        self,
        genes: torch.Tensor,
        time: torch.Tensor,
        token_times: torch.Tensor,
        valid_mask: torch.Tensor,
        anchor_mask: torch.Tensor,
        context_role: torch.Tensor | None = None,
        relative_position: torch.Tensor | None = None,
    ) -> EmbryoStateOutput:
        embryo_latent, local_latents = self.encode_embryo(
            genes=genes,
            time=time,
            token_times=token_times,
            valid_mask=valid_mask,
            anchor_mask=anchor_mask,
            context_role=context_role,
            relative_position=relative_position,
        )
        return EmbryoStateOutput(
            embryo_latent=embryo_latent,
            local_latents=local_latents,
            future_founder_composition=self.future_founder_head(embryo_latent),
            future_celltype_composition=self.future_celltype_head(embryo_latent),
            future_lineage_depth_stats=self.future_depth_head(embryo_latent),
            future_spatial_extent=self.future_spatial_head(embryo_latent),
            future_split_fraction=self.future_split_head(embryo_latent),
        )


class EmbryoMaskedViewModel(nn.Module):
    """Embryo-level masked multi-view model.

    Local views are observations of the same embryo state. A subset of views is
    hidden, and the visible views must reconstruct the masked views' latent and
    mean gene content.
    """

    def __init__(
        self,
        gene_dim: int,
        context_size: int,
        model_type: str = "multi_cell",
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        head_dim: int = 32,
        use_pairwise_spatial_bias: bool = True,
    ):
        super().__init__()
        if model_type == "single_cell":
            self.local_model = SingleCellPatchSetModel(
                gene_dim=gene_dim,
                context_size=context_size,
                d_model=d_model,
                n_layers=n_layers,
            )
        else:
            self.local_model = MultiCellPatchSetModel(
                gene_dim=gene_dim,
                context_size=context_size,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                head_dim=head_dim,
                use_pairwise_spatial_bias=use_pairwise_spatial_bias,
            )
        self.state_proj = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.masked_view_latent_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.masked_view_gene_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, gene_dim),
        )
        self.masked_future_view_latent_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.masked_future_view_gene_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, gene_dim),
        )

    def encode_local_views(
        self,
        genes: torch.Tensor,
        time: torch.Tensor,
        token_times: torch.Tensor,
        valid_mask: torch.Tensor,
        anchor_mask: torch.Tensor,
        context_role: torch.Tensor | None = None,
        relative_position: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, n_views, patch_len, gene_dim = genes.shape
        flat_kwargs = {
            "genes": genes.view(batch_size * n_views, patch_len, gene_dim),
            "time": time.view(batch_size * n_views),
            "future_time": time.view(batch_size * n_views),
            "token_times": token_times.view(batch_size * n_views, patch_len),
            "valid_mask": valid_mask.view(batch_size * n_views, patch_len),
            "anchor_mask": anchor_mask.view(batch_size * n_views, patch_len),
            "context_role": None if context_role is None else context_role.view(batch_size * n_views, patch_len),
            "relative_position": None if relative_position is None else relative_position.view(
                batch_size * n_views, patch_len, relative_position.shape[-1]
            ),
        }
        local_latents, _ = self.local_model.encode_patch(**flat_kwargs)
        return local_latents.view(batch_size, n_views, -1)

    def pool_visible(
        self,
        local_latents: torch.Tensor,
        visible_mask: torch.Tensor,
    ) -> torch.Tensor:
        pooled = (local_latents * visible_mask.unsqueeze(-1).float()).sum(dim=1) / visible_mask.float().sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        first_visible_idx = visible_mask.float().argmax(dim=1)
        first_visible = local_latents[torch.arange(local_latents.shape[0], device=local_latents.device), first_visible_idx]
        return self.state_proj(torch.cat([first_visible, pooled], dim=-1))

    def encode_embryo_latent(
        self,
        genes: torch.Tensor,
        time: torch.Tensor,
        token_times: torch.Tensor,
        valid_mask: torch.Tensor,
        anchor_mask: torch.Tensor,
        context_role: torch.Tensor | None = None,
        relative_position: torch.Tensor | None = None,
        visible_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local_latents = self.encode_local_views(
            genes=genes,
            time=time,
            token_times=token_times,
            valid_mask=valid_mask,
            anchor_mask=anchor_mask,
            context_role=context_role,
            relative_position=relative_position,
        )
        if visible_mask is None:
            visible_mask = torch.ones(
                local_latents.shape[:2],
                dtype=torch.bool,
                device=local_latents.device,
            )
        embryo_latent = self.pool_visible(local_latents, visible_mask)
        return embryo_latent, local_latents

    def forward(
        self,
        genes: torch.Tensor,
        time: torch.Tensor,
        token_times: torch.Tensor,
        valid_mask: torch.Tensor,
        anchor_mask: torch.Tensor,
        masked_view_mask: torch.Tensor,
        future_genes: torch.Tensor | None = None,
        future_time: torch.Tensor | None = None,
        future_token_times: torch.Tensor | None = None,
        future_valid_mask: torch.Tensor | None = None,
        future_anchor_mask: torch.Tensor | None = None,
        masked_future_view_mask: torch.Tensor | None = None,
        context_role: torch.Tensor | None = None,
        relative_position: torch.Tensor | None = None,
        future_context_role: torch.Tensor | None = None,
        future_relative_position: torch.Tensor | None = None,
    ) -> EmbryoMaskedOutput:
        local_latents = self.encode_local_views(
            genes=genes,
            time=time,
            token_times=token_times,
            valid_mask=valid_mask,
            anchor_mask=anchor_mask,
            context_role=context_role,
            relative_position=relative_position,
        )
        visible_mask = ~masked_view_mask
        visible_embryo_latent = self.pool_visible(local_latents, visible_mask)
        full_embryo_latent = self.pool_visible(
            local_latents,
            torch.ones_like(masked_view_mask, dtype=torch.bool),
        )
        pred_masked_view_latents = self.masked_view_latent_head(visible_embryo_latent)
        pred_masked_view_genes = self.masked_view_gene_head(visible_embryo_latent)
        future_local_latents = None
        pred_masked_future_view_latents = None
        pred_masked_future_view_genes = None
        if (
            future_genes is not None
            and future_time is not None
            and future_token_times is not None
            and future_valid_mask is not None
            and future_anchor_mask is not None
            and masked_future_view_mask is not None
        ):
            future_local_latents = self.encode_local_views(
                genes=future_genes,
                time=future_time,
                token_times=future_token_times,
                valid_mask=future_valid_mask,
                anchor_mask=future_anchor_mask,
                context_role=future_context_role,
                relative_position=future_relative_position,
            )
            pred_masked_future_view_latents = self.masked_future_view_latent_head(visible_embryo_latent)
            pred_masked_future_view_genes = self.masked_future_view_gene_head(visible_embryo_latent)
        return EmbryoMaskedOutput(
            embryo_latent=full_embryo_latent,
            visible_embryo_latent=visible_embryo_latent,
            local_latents=local_latents,
            future_local_latents=future_local_latents,
            pred_masked_view_latents=pred_masked_view_latents,
            pred_masked_view_genes=pred_masked_view_genes,
            pred_masked_future_view_latents=pred_masked_future_view_latents,
            pred_masked_future_view_genes=pred_masked_future_view_genes,
            masked_view_mask=masked_view_mask,
            masked_future_view_mask=masked_future_view_mask,
        )


class EmbryoOneStepLatentModel(nn.Module):
    """Predict future embryo latent and future probes from current embryo state."""

    def __init__(
        self,
        backbone: EmbryoMaskedViewModel,
        celltype_dim: int,
        d_model: int = 256,
        predict_delta: bool = False,
        prediction_space_dim: int | None = None,
        target_ema_decay: float = 0.99,
    ):
        super().__init__()
        self.backbone = backbone
        self.predict_delta = predict_delta
        self.prediction_space_dim = prediction_space_dim
        self.target_ema_decay = target_ema_decay
        self.use_prediction_space = prediction_space_dim is not None
        if self.use_prediction_space:
            self.online_projector = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Linear(d_model, prediction_space_dim),
            )
            self.target_projector = copy.deepcopy(self.online_projector)
            for param in self.target_projector.parameters():
                param.requires_grad_(False)
            self.predictor = nn.Sequential(
                nn.Linear(prediction_space_dim, prediction_space_dim),
                nn.LayerNorm(prediction_space_dim),
                nn.GELU(),
                nn.Linear(prediction_space_dim, prediction_space_dim),
            )
            self.transition_decoder = nn.Sequential(
                nn.Linear(prediction_space_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.target_projector.eval()
        else:
            self.predictor = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
        self.future_founder_head = nn.Linear(d_model, 8)
        self.future_celltype_head = nn.Linear(d_model, celltype_dim)
        self.future_depth_head = nn.Linear(d_model, 3)
        self.future_spatial_head = nn.Linear(d_model, 4)
        self.future_split_head = nn.Linear(d_model, 1)

    @torch.no_grad()
    def update_target_encoder(self):
        if not self.use_prediction_space:
            return
        for target_param, online_param in zip(
            self.target_projector.parameters(),
            self.online_projector.parameters(),
            strict=True,
        ):
            target_param.data.mul_(self.target_ema_decay).add_(
                online_param.data, alpha=1.0 - self.target_ema_decay
            )

    def forward(
        self,
        genes: torch.Tensor,
        time: torch.Tensor,
        token_times: torch.Tensor,
        valid_mask: torch.Tensor,
        anchor_mask: torch.Tensor,
        future_genes: torch.Tensor,
        future_time: torch.Tensor,
        future_token_times: torch.Tensor,
        future_valid_mask: torch.Tensor,
        future_anchor_mask: torch.Tensor,
        context_role: torch.Tensor | None = None,
        relative_position: torch.Tensor | None = None,
        future_context_role: torch.Tensor | None = None,
        future_relative_position: torch.Tensor | None = None,
    ) -> EmbryoOneStepOutput:
        current_embryo_latent, _ = self.backbone.encode_embryo_latent(
            genes=genes,
            time=time,
            token_times=token_times,
            valid_mask=valid_mask,
            anchor_mask=anchor_mask,
            context_role=context_role,
            relative_position=relative_position,
        )
        target_future_embryo_latent, _ = self.backbone.encode_embryo_latent(
            genes=future_genes,
            time=future_time,
            token_times=future_token_times,
            valid_mask=future_valid_mask,
            anchor_mask=future_anchor_mask,
            context_role=future_context_role,
            relative_position=future_relative_position,
        )
        target_future_delta = target_future_embryo_latent - current_embryo_latent
        current_prediction_space = None
        target_prediction_space = None
        pred_prediction_space = None
        if self.use_prediction_space:
            current_prediction_space = self.online_projector(current_embryo_latent)
            with torch.no_grad():
                target_prediction_space = self.target_projector(target_future_embryo_latent)
            pred_prediction_space = self.predictor(current_prediction_space)
            pred_future_delta = None
            pred_future_embryo_latent = self.transition_decoder(pred_prediction_space)
        else:
            pred_future_delta = self.predictor(current_embryo_latent)
            if self.predict_delta:
                pred_future_embryo_latent = current_embryo_latent + pred_future_delta
            else:
                pred_future_embryo_latent = pred_future_delta
        return EmbryoOneStepOutput(
            current_embryo_latent=current_embryo_latent,
            target_future_embryo_latent=target_future_embryo_latent,
            pred_future_embryo_latent=pred_future_embryo_latent,
            target_future_delta=target_future_delta,
            pred_future_delta=pred_future_delta,
            current_prediction_space=current_prediction_space,
            target_prediction_space=target_prediction_space,
            pred_prediction_space=pred_prediction_space,
            future_founder_composition=self.future_founder_head(pred_future_embryo_latent),
            future_celltype_composition=self.future_celltype_head(pred_future_embryo_latent),
            future_lineage_depth_stats=self.future_depth_head(pred_future_embryo_latent),
            future_spatial_extent=self.future_spatial_head(pred_future_embryo_latent),
            future_split_fraction=self.future_split_head(pred_future_embryo_latent),
        )


class EmbryoJEPAModel(nn.Module):
    """Minimal JEPA for embryo views using a frozen/EMA target encoder."""

    def __init__(
        self,
        backbone: EmbryoMaskedViewModel,
        d_model: int = 256,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.online_backbone = backbone
        self.target_backbone = copy.deepcopy(backbone)
        self.ema_decay = ema_decay
        for param in self.target_backbone.parameters():
            param.requires_grad_(False)
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.target_backbone.eval()

    @staticmethod
    def aggregate_masked_latents(
        local_latents: torch.Tensor,
        masked_view_mask: torch.Tensor,
    ) -> torch.Tensor:
        targets = []
        for i in range(local_latents.shape[0]):
            masked_idx = torch.nonzero(masked_view_mask[i], as_tuple=False).squeeze(-1)
            targets.append(local_latents[i, masked_idx].mean(dim=0))
        return torch.stack(targets, dim=0)

    @torch.no_grad()
    def update_target_encoder(self):
        for target_param, online_param in zip(
            self.target_backbone.parameters(),
            self.online_backbone.parameters(),
            strict=True,
        ):
            target_param.data.mul_(self.ema_decay).add_(online_param.data, alpha=1.0 - self.ema_decay)

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_backbone.eval()
        return self

    def forward(
        self,
        genes: torch.Tensor,
        time: torch.Tensor,
        token_times: torch.Tensor,
        valid_mask: torch.Tensor,
        anchor_mask: torch.Tensor,
        masked_view_mask: torch.Tensor,
        future_genes: torch.Tensor,
        future_time: torch.Tensor,
        future_token_times: torch.Tensor,
        future_valid_mask: torch.Tensor,
        future_anchor_mask: torch.Tensor,
        masked_future_view_mask: torch.Tensor,
        context_role: torch.Tensor | None = None,
        relative_position: torch.Tensor | None = None,
        future_context_role: torch.Tensor | None = None,
        future_relative_position: torch.Tensor | None = None,
    ) -> EmbryoJEPAOutput:
        context_embryo_latent, _ = self.online_backbone.encode_embryo_latent(
            genes=genes,
            time=time,
            token_times=token_times,
            valid_mask=valid_mask,
            anchor_mask=anchor_mask,
            context_role=context_role,
            relative_position=relative_position,
            visible_mask=~masked_view_mask,
        )
        pred_masked_future_latent = self.predictor(context_embryo_latent)
        with torch.no_grad():
            future_local_latents = self.target_backbone.encode_local_views(
                genes=future_genes,
                time=future_time,
                token_times=future_token_times,
                valid_mask=future_valid_mask,
                anchor_mask=future_anchor_mask,
                context_role=future_context_role,
                relative_position=future_relative_position,
            )
            target_masked_future_latent = self.aggregate_masked_latents(
                future_local_latents,
                masked_future_view_mask,
            )
        return EmbryoJEPAOutput(
            context_embryo_latent=context_embryo_latent,
            target_masked_future_latent=target_masked_future_latent,
            pred_masked_future_latent=pred_masked_future_latent,
            masked_view_mask=masked_view_mask,
            masked_future_view_mask=masked_future_view_mask,
        )


class EmbryoFutureSetModel(nn.Module):
    """MAE-style masked future local-view set prediction from current and visible future parts."""

    def __init__(
        self,
        backbone: EmbryoMaskedViewModel,
        future_slots: int,
        d_model: int = 256,
        gene_dim: int | None = None,
        n_heads: int = 4,
        decoder_layers: int = 3,
        head_dim: int = 32,
        use_current_local_tokens: bool = False,
        learn_current_token_gate: bool = True,
        current_token_gate_init: float = 0.5,
        current_conditioning_mode: str = "flat_tokens",
        predict_future_cell_tokens: bool = False,
        cell_tokens_per_view: int | None = None,
    ):
        super().__init__()
        if future_slots < 1:
            raise ValueError("future_slots must be >= 1")
        if current_conditioning_mode not in {"flat_tokens", "cross_attention_memory"}:
            raise ValueError("current_conditioning_mode must be 'flat_tokens' or 'cross_attention_memory'")
        self.backbone = backbone
        self.future_slots = future_slots
        self.use_current_local_tokens = use_current_local_tokens
        self.learn_current_token_gate = learn_current_token_gate
        self.current_conditioning_mode = current_conditioning_mode
        self.predict_future_cell_tokens = predict_future_cell_tokens
        self.cell_tokens_per_view = (
            int(cell_tokens_per_view)
            if cell_tokens_per_view is not None
            else int(backbone.local_model.context_size)
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.slot_queries = nn.Parameter(torch.randn(future_slots, d_model) * 0.02)
        n_token_types = 4 if use_current_local_tokens else 3
        self.token_type = nn.Embedding(n_token_types, d_model)
        if use_current_local_tokens:
            gate_init = min(max(float(current_token_gate_init), 1e-4), 1.0 - 1e-4)
            self.current_view_proj = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            if current_conditioning_mode == "cross_attention_memory":
                self.current_memory_attn = nn.MultiheadAttention(
                    d_model,
                    num_heads=max(1, min(n_heads, 4)),
                    batch_first=True,
                )
                self.current_memory_query_norm = nn.LayerNorm(d_model)
                self.current_memory_key_norm = nn.LayerNorm(d_model)
                self.current_memory_ff = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.LayerNorm(d_model),
                    nn.GELU(),
                    nn.Linear(d_model, d_model),
                )
            if learn_current_token_gate:
                self.current_token_gate_logit = nn.Parameter(
                    torch.tensor(math.log(gate_init / (1.0 - gate_init)), dtype=torch.float32)
                )
                self.register_buffer("current_token_gate_value", torch.tensor(gate_init, dtype=torch.float32))
            else:
                self.current_token_gate_logit = None
                self.register_buffer("current_token_gate_value", torch.tensor(gate_init, dtype=torch.float32))
        self.visible_future_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.decoder_blocks = nn.ModuleList(
            [
                TransformerBlockAutoregressive(
                    d_model=d_model,
                    n_heads=max(1, min(n_heads, 4)),
                    head_dim=head_dim,
                )
                for _ in range(decoder_layers)
            ]
        )
        self.decoder_norm = nn.LayerNorm(d_model)
        self.slot_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.slot_gene_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, gene_dim or backbone.local_model.gene_dim),
        )
        if predict_future_cell_tokens:
            self.slot_cell_gene_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Linear(d_model, self.cell_tokens_per_view * (gene_dim or backbone.local_model.gene_dim)),
            )
        else:
            self.slot_cell_gene_head = None

    @staticmethod
    def gather_masked_future_targets(
        future_local_latents: torch.Tensor,
        future_genes: torch.Tensor,
        future_valid_mask: torch.Tensor,
        masked_future_view_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_latents = []
        target_genes = []
        for i in range(future_local_latents.shape[0]):
            masked_idx = torch.nonzero(masked_future_view_mask[i], as_tuple=False).squeeze(-1)
            if masked_idx.numel() == 0:
                raise ValueError("Each sample must mask at least one future view")
            target_latents.append(future_local_latents[i, masked_idx])
            masked_valid = future_valid_mask[i, masked_idx]
            masked_genes = future_genes[i, masked_idx]
            mean_genes = (
                (masked_genes * masked_valid.unsqueeze(-1).float()).sum(dim=1)
                / masked_valid.sum(dim=1, keepdim=True).clamp_min(1.0)
            )
            target_genes.append(mean_genes)
        return torch.stack(target_latents, dim=0), torch.stack(target_genes, dim=0)

    @staticmethod
    def gather_masked_future_cell_targets(
        future_genes: torch.Tensor,
        future_valid_mask: torch.Tensor,
        masked_future_view_mask: torch.Tensor,
    ) -> torch.Tensor:
        target_cells = []
        for i in range(future_genes.shape[0]):
            masked_idx = torch.nonzero(masked_future_view_mask[i], as_tuple=False).squeeze(-1)
            if masked_idx.numel() == 0:
                raise ValueError("Each sample must mask at least one future view")
            masked_genes = future_genes[i, masked_idx]
            masked_valid = future_valid_mask[i, masked_idx]
            masked_genes = masked_genes * masked_valid.unsqueeze(-1).float()
            target_cells.append(masked_genes.reshape(-1, masked_genes.shape[-1]))
        return torch.stack(target_cells, dim=0)

    def get_current_local_token_gate(self) -> torch.Tensor | None:
        if not self.use_current_local_tokens:
            return None
        if self.learn_current_token_gate and self.current_token_gate_logit is not None:
            return torch.sigmoid(self.current_token_gate_logit)
        return self.current_token_gate_value

    def forward(
        self,
        genes: torch.Tensor,
        time: torch.Tensor,
        token_times: torch.Tensor,
        valid_mask: torch.Tensor,
        anchor_mask: torch.Tensor,
        future_genes: torch.Tensor,
        future_time: torch.Tensor,
        future_token_times: torch.Tensor,
        future_valid_mask: torch.Tensor,
        future_anchor_mask: torch.Tensor,
        masked_future_view_mask: torch.Tensor,
        masked_view_mask: torch.Tensor | None = None,
        context_role: torch.Tensor | None = None,
        relative_position: torch.Tensor | None = None,
        future_context_role: torch.Tensor | None = None,
        future_relative_position: torch.Tensor | None = None,
    ) -> EmbryoFutureSetOutput:
        if masked_view_mask is None:
            masked_view_mask = torch.zeros(
                genes.shape[:2],
                dtype=torch.bool,
                device=genes.device,
            )
        visible_mask = ~masked_view_mask
        context_embryo_latent, current_local_latents = self.backbone.encode_embryo_latent(
            genes=genes,
            time=time,
            token_times=token_times,
            valid_mask=valid_mask,
            anchor_mask=anchor_mask,
            context_role=context_role,
            relative_position=relative_position,
            visible_mask=visible_mask,
        )
        future_local_latents = self.backbone.encode_local_views(
            genes=future_genes,
            time=future_time,
            token_times=future_token_times,
            valid_mask=future_valid_mask,
            anchor_mask=future_anchor_mask,
            context_role=future_context_role,
            relative_position=future_relative_position,
        )
        target_future_set_latents, target_future_set_genes = self.gather_masked_future_targets(
            future_local_latents,
            future_genes,
            future_valid_mask,
            masked_future_view_mask,
        )
        visible_future_mask = ~masked_future_view_mask
        current_token = context_embryo_latent.unsqueeze(1) + self.token_type.weight[0].view(1, 1, -1)
        current_local_tokens = None
        current_local_token_gate = None
        if self.use_current_local_tokens:
            current_local_tokens = self.current_view_proj(current_local_latents)
            current_local_tokens = current_local_tokens + self.token_type.weight[1].view(1, 1, -1)
            current_local_token_gate = self.get_current_local_token_gate()
            current_local_tokens = current_local_tokens * visible_mask.unsqueeze(-1).float()
        visible_future_tokens = self.visible_future_proj(future_local_latents)
        future_type_idx = 2 if self.use_current_local_tokens else 1
        visible_future_tokens = visible_future_tokens + self.token_type.weight[future_type_idx].view(1, 1, -1)
        visible_future_tokens = visible_future_tokens * visible_future_mask.unsqueeze(-1).float()
        masked_slot_tokens = self.mask_token + self.slot_queries.unsqueeze(0)
        slot_type_idx = 3 if self.use_current_local_tokens else 2
        masked_slot_tokens = masked_slot_tokens + self.token_type.weight[slot_type_idx].view(1, 1, -1)
        token_parts = [current_token]
        mask_parts = [torch.ones(genes.shape[0], 1, dtype=torch.bool, device=genes.device)]
        if current_local_tokens is not None and self.current_conditioning_mode == "flat_tokens":
            current_local_tokens = current_local_tokens * current_local_token_gate.view(1, 1, 1)
            token_parts.append(current_local_tokens)
            mask_parts.append(visible_mask)
        token_parts.append(visible_future_tokens)
        mask_parts.append(visible_future_mask)
        token_parts.append(masked_slot_tokens.expand(genes.shape[0], -1, -1))
        mask_parts.append(torch.ones(genes.shape[0], self.future_slots, dtype=torch.bool, device=genes.device))
        decoder_tokens = torch.cat(token_parts, dim=1)
        decoder_mask = torch.cat(mask_parts, dim=1)
        for block in self.decoder_blocks:
            decoder_tokens = block(decoder_tokens, decoder_mask)
        decoder_tokens = self.decoder_norm(decoder_tokens)
        pred_slot_tokens = decoder_tokens[:, -self.future_slots :]
        if current_local_tokens is not None and self.current_conditioning_mode == "cross_attention_memory":
            current_memory = self.current_memory_key_norm(current_local_tokens)
            slot_queries = self.current_memory_query_norm(pred_slot_tokens)
            memory_out, _ = self.current_memory_attn(
                query=slot_queries,
                key=current_memory,
                value=current_memory,
                key_padding_mask=~visible_mask,
                need_weights=False,
            )
            pred_slot_tokens = pred_slot_tokens + current_local_token_gate.view(1, 1, 1) * memory_out
            pred_slot_tokens = pred_slot_tokens + self.current_memory_ff(pred_slot_tokens)
        pred_future_set_latents = self.slot_predictor(pred_slot_tokens)
        pred_future_set_genes = self.slot_gene_head(pred_future_set_latents)
        pred_future_cell_tokens = None
        target_future_cell_tokens = None
        if self.predict_future_cell_tokens and self.slot_cell_gene_head is not None:
            pred_future_cell_tokens = self.slot_cell_gene_head(pred_future_set_latents).view(
                genes.shape[0],
                self.future_slots * self.cell_tokens_per_view,
                pred_future_set_genes.shape[-1],
            )
            target_future_cell_tokens = self.gather_masked_future_cell_targets(
                future_genes,
                future_valid_mask,
                masked_future_view_mask,
            )
        return EmbryoFutureSetOutput(
            context_embryo_latent=context_embryo_latent,
            future_local_latents=future_local_latents,
            pred_future_set_latents=pred_future_set_latents,
            pred_future_set_genes=pred_future_set_genes,
            pred_future_cell_tokens=pred_future_cell_tokens,
            target_future_set_latents=target_future_set_latents,
            target_future_set_genes=target_future_set_genes,
            target_future_cell_tokens=target_future_cell_tokens,
            masked_view_mask=masked_view_mask,
            masked_future_view_mask=masked_future_view_mask,
            current_local_token_gate=current_local_token_gate,
        )
