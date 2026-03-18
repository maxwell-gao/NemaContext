"""Embryo-scale gene-context models."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .autoregressive_model import TransformerBlockAutoregressive
from .gene_context_patch import MultiCellPatchSetModel, SingleCellPatchSetModel
from .gene_context_shared import (
    EmbryoFutureSetOutput,
    EmbryoMaskedOutput,
    EmbryoStateOutput,
    LocalCellCodeCodec,
    LocalCellDecodeOutput,
)


class EmbryoStateModel(nn.Module):
    """Embryo-scale state encoder from multiple local observation views."""

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
            "context_role": None
            if context_role is None
            else context_role.view(batch_size * n_views, patch_len),
            "relative_position": None
            if relative_position is None
            else relative_position.view(batch_size * n_views, patch_len, relative_position.shape[-1]),
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
    """Embryo-level masked multi-view model."""

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

    def encode_local_views_with_tokens(
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
            "context_role": None
            if context_role is None
            else context_role.view(batch_size * n_views, patch_len),
            "relative_position": None
            if relative_position is None
            else relative_position.view(batch_size * n_views, patch_len, relative_position.shape[-1]),
        }
        local_latents, token_states = self.local_model.encode_patch(**flat_kwargs)
        local_latents = local_latents.view(batch_size, n_views, -1)
        token_states = token_states.view(batch_size, n_views, patch_len, -1)
        return local_latents, token_states

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
        local_latents, _ = self.encode_local_views_with_tokens(
            genes=genes,
            time=time,
            token_times=token_times,
            valid_mask=valid_mask,
            anchor_mask=anchor_mask,
            context_role=context_role,
            relative_position=relative_position,
        )
        return local_latents

    def pool_visible(
        self,
        local_latents: torch.Tensor,
        visible_mask: torch.Tensor,
    ) -> torch.Tensor:
        pooled = (local_latents * visible_mask.unsqueeze(-1).float()).sum(dim=1) / visible_mask.float().sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        first_visible_idx = visible_mask.float().argmax(dim=1)
        first_visible = local_latents[
            torch.arange(local_latents.shape[0], device=local_latents.device), first_visible_idx
        ]
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
        code_tokens: int = 8,
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
        self.code_tokens = int(code_tokens)
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
        self.local_code_codec = LocalCellCodeCodec(
            gene_dim=(gene_dim or backbone.local_model.gene_dim),
            context_size=int(backbone.local_model.context_size),
            d_model=d_model,
            n_heads=n_heads,
            code_tokens=self.code_tokens,
        )
        self.slot_code_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, self.code_tokens * d_model),
        )

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
    def gather_masked_future_view_tensor(
        tensor: torch.Tensor,
        masked_future_view_mask: torch.Tensor,
    ) -> torch.Tensor:
        gathered = []
        for i in range(tensor.shape[0]):
            masked_idx = torch.nonzero(masked_future_view_mask[i], as_tuple=False).squeeze(-1)
            if masked_idx.numel() == 0:
                raise ValueError("Each sample must mask at least one future view")
            gathered.append(tensor[i, masked_idx])
        return torch.stack(gathered, dim=0)

    def get_current_local_token_gate(self) -> torch.Tensor | None:
        if not self.use_current_local_tokens:
            return None
        if self.learn_current_token_gate and self.current_token_gate_logit is not None:
            return torch.sigmoid(self.current_token_gate_logit)
        return self.current_token_gate_value

    def encode_masked_future_local_codes(
        self,
        future_local_latents: torch.Tensor,
        future_token_states: torch.Tensor,
        future_valid_mask: torch.Tensor,
        masked_future_view_mask: torch.Tensor,
    ) -> torch.Tensor:
        masked_future_latents = self.gather_masked_future_view_tensor(
            future_local_latents,
            masked_future_view_mask,
        )
        masked_future_token_states = self.gather_masked_future_view_tensor(
            future_token_states,
            masked_future_view_mask,
        )
        masked_future_valid_mask = self.gather_masked_future_view_tensor(
            future_valid_mask, masked_future_view_mask
        )
        batch_size, n_slots, patch_len, _ = masked_future_token_states.shape
        local_codes = self.local_code_codec.encode_from_patch(
            patch_latent=masked_future_latents.reshape(batch_size * n_slots, -1),
            token_states=masked_future_token_states.reshape(batch_size * n_slots, patch_len, -1),
            valid_mask=masked_future_valid_mask.reshape(batch_size * n_slots, patch_len),
        )
        return local_codes.view(batch_size, n_slots, self.code_tokens, -1)

    def decode_future_local_codes(self, local_codes: torch.Tensor) -> LocalCellDecodeOutput:
        return self.local_code_codec.decode(local_codes)

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
        future_local_latents, future_local_token_states = self.backbone.encode_local_views_with_tokens(
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
        visible_future_tokens = visible_future_tokens + self.token_type.weight[future_type_idx].view(
            1, 1, -1
        )
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
        pred_future_local_codes = self.slot_code_head(pred_slot_tokens).view(
            genes.shape[0],
            self.future_slots,
            self.code_tokens,
            pred_slot_tokens.shape[-1],
        )
        with torch.no_grad():
            target_future_local_codes = self.encode_masked_future_local_codes(
                future_local_latents=future_local_latents,
                future_token_states=future_local_token_states,
                future_valid_mask=future_valid_mask,
                masked_future_view_mask=masked_future_view_mask,
            )
        return EmbryoFutureSetOutput(
            context_embryo_latent=context_embryo_latent,
            future_local_latents=future_local_latents,
            pred_future_set_latents=pred_future_set_latents,
            pred_future_set_genes=pred_future_set_genes,
            pred_future_local_codes=pred_future_local_codes,
            target_future_set_latents=target_future_set_latents,
            target_future_set_genes=target_future_set_genes,
            target_future_local_codes=target_future_local_codes,
            masked_view_mask=masked_view_mask,
            masked_future_view_mask=masked_future_view_mask,
            current_local_token_gate=current_local_token_gate,
        )
