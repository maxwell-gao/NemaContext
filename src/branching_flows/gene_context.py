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
