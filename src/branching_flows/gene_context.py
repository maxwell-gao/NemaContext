"""Multi-cell gene-context baseline model."""

from __future__ import annotations

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
