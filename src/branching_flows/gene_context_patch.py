"""Patch/local gene-context models."""

from __future__ import annotations

import torch
import torch.nn as nn

from .autoregressive_model import TransformerBlockAutoregressive
from .gene_context_shared import (
    GeneContextOutput,
    JiTGenePatchOutput,
    LocalCellCodeCodec,
    LocalCellCodeOutput,
    MultiPatchSetOutput,
    PatchSetOutput,
)


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
            has_spatial.unsqueeze(2)
            * has_spatial.unsqueeze(1)
            * valid_mask.unsqueeze(2).float()
            * valid_mask.unsqueeze(1).float()
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


class JiTGenePatchModel(nn.Module):
    """Direct gene-space future patch predictor with ViT-like query tokens."""

    def __init__(
        self,
        gene_dim: int,
        context_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        head_dim: int = 32,
    ):
        super().__init__()
        self.gene_dim = gene_dim
        self.context_size = context_size
        self.d_model = d_model
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
        self.token_time_proj = nn.Sequential(
            nn.Linear(1, d_model),
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
        self.current_type = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.future_type = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.future_queries = nn.Parameter(torch.randn(context_size, d_model) * 0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlockAutoregressive(d_model, n_heads, head_dim) for _ in range(n_layers)]
        )
        self.gene_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, gene_dim),
        )
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
    ) -> JiTGenePatchOutput:
        batch_size = genes.shape[0]
        current_time = torch.stack([time, future_time - time], dim=-1)
        current_x = self.gene_proj(genes)
        current_x = current_x + self.time_proj(current_time).unsqueeze(1)
        current_x = current_x + self.token_time_proj(token_times.unsqueeze(-1))
        current_x = current_x + self.current_type
        if context_role is not None:
            current_x = current_x + self.context_role_emb(context_role.clamp(min=0, max=3))
        if relative_position is not None:
            current_x = current_x + self.relative_spatial_proj(relative_position)

        future_x = self.future_queries.unsqueeze(0).expand(batch_size, -1, -1)
        future_x = future_x + self.future_type
        future_x = future_x + self.time_proj(current_time).unsqueeze(1)

        x = torch.cat([current_x, future_x], dim=1)
        future_mask = torch.ones(
            batch_size,
            self.context_size,
            dtype=valid_mask.dtype,
            device=valid_mask.device,
        )
        mask = torch.cat([valid_mask, future_mask], dim=1)

        for block in self.blocks:
            x = block(x, mask)

        future_states = x[:, -self.context_size :]
        pred_future_genes = self.gene_head(future_states)
        pred_mean_gene = pred_future_genes.mean(dim=1)
        return JiTGenePatchOutput(
            pred_future_genes=pred_future_genes,
            pred_future_token_states=future_states,
            pred_mean_gene=pred_mean_gene,
        )


class LocalCellCodeModel(nn.Module):
    """Local patch autoencoder with a compact cell-code bottleneck."""

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
        self.codec = LocalCellCodeCodec(
            gene_dim=gene_dim,
            context_size=context_size,
            d_model=d_model,
            n_heads=n_heads,
            code_tokens=code_tokens,
        )

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

    def encode_local_code_from_patch(
        self,
        patch_latent: torch.Tensor,
        token_states: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.codec.encode_from_patch(
            patch_latent=patch_latent,
            token_states=token_states,
            valid_mask=valid_mask,
        )

    def encode_local_code(
        self,
        genes: torch.Tensor,
        time: torch.Tensor,
        token_times: torch.Tensor,
        valid_mask: torch.Tensor,
        anchor_mask: torch.Tensor,
        context_role: torch.Tensor | None = None,
        relative_position: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        patch_latent, token_states = self.encode_patch(
            genes=genes,
            time=time,
            token_times=token_times,
            valid_mask=valid_mask,
            anchor_mask=anchor_mask,
            context_role=context_role,
            relative_position=relative_position,
        )
        local_code_tokens = self.encode_local_code_from_patch(
            patch_latent=patch_latent,
            token_states=token_states,
            valid_mask=valid_mask,
        )
        return patch_latent, local_code_tokens

    def decode_local_code(self, local_code_tokens: torch.Tensor):
        return self.codec.decode(local_code_tokens)

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
        patch_latent, local_code_tokens = self.encode_local_code(
            genes=genes,
            time=time,
            token_times=token_times,
            valid_mask=valid_mask,
            anchor_mask=anchor_mask,
            context_role=context_role,
            relative_position=relative_position,
        )
        decoded = self.decode_local_code(local_code_tokens)
        return LocalCellCodeOutput(
            patch_latent=patch_latent,
            local_code_tokens=local_code_tokens,
            pred_cell_genes=decoded.pred_cell_genes,
            pred_cell_positions=decoded.pred_cell_positions,
            pred_cell_valid_logits=decoded.pred_cell_valid_logits,
            pred_cell_spatial_logits=decoded.pred_cell_spatial_logits,
            pred_cell_count=decoded.pred_cell_count,
            pred_mean_gene=decoded.pred_mean_gene,
            pred_patch_latent=decoded.pred_patch_latent,
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
            [
                TransformerBlockAutoregressive(
                    d_model, n_heads=max(1, min(n_heads, 4)), head_dim=head_dim
                )
                for _ in range(2)
            ]
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
            "context_role": None
            if context_role is None
            else context_role.view(batch_size * n_patches, patch_len),
            "relative_position": None
            if relative_position is None
            else relative_position.view(batch_size * n_patches, patch_len, relative_position.shape[-1]),
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
        patch_attention_weights = patch_attention_weights / patch_attention_weights.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-8)
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
