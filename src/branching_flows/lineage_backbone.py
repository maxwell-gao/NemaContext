"""Active lineage-first whole-embryo temporal backbone."""

from __future__ import annotations

import torch
import torch.nn as nn

from .autoregressive_model import TransformerBlockAutoregressive
from .gene_context_shared import GenePatchVideoOutput


class CrossAttentionRefinementBlock(nn.Module):
    """Future-query refinement against history memory."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(
        self,
        future_tokens: torch.Tensor,
        history_tokens: torch.Tensor,
        history_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.norm1(future_tokens)
        self_out, _ = self.self_attn(q, q, q, need_weights=False)
        future_tokens = future_tokens + self_out

        q = self.norm2(future_tokens)
        key_padding_mask = None if history_mask is None else ~history_mask.bool()
        cross_out, _ = self.cross_attn(
            q,
            history_tokens,
            history_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        future_tokens = future_tokens + cross_out
        future_tokens = future_tokens + self.mlp(self.norm3(future_tokens))
        return future_tokens


class LineageTokenEmbedder(nn.Module):
    """Embed gene/time/lineage token features into backbone states."""

    def __init__(
        self,
        gene_dim: int,
        context_size: int,
        history_frames: int,
        lineage_binary_dim: int,
        founder_vocab_size: int,
        d_model: int,
    ):
        super().__init__()
        self.context_size = context_size
        self.history_frames = history_frames
        self.founder_vocab_size = founder_vocab_size
        self.d_model = d_model
        self.gene_proj = nn.Sequential(
            nn.Linear(gene_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.global_time_proj = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.token_time_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.lineage_binary_proj = nn.Sequential(
            nn.Linear(lineage_binary_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.founder_emb = nn.Embedding(founder_vocab_size, d_model)
        self.lineage_depth_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.lineage_valid_emb = nn.Embedding(2, d_model)
        self.frame_index_emb = nn.Embedding(history_frames + 1, d_model)
        self.token_rank_emb = nn.Embedding(context_size, d_model)
        self.history_type = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.future_type = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.future_token_queries = nn.Parameter(torch.randn(context_size, d_model) * 0.02)

    def forward(
        self,
        genes: torch.Tensor,
        time: torch.Tensor,
        future_time: torch.Tensor,
        token_times: torch.Tensor,
        lineage_binary: torch.Tensor,
        founder_ids: torch.Tensor,
        lineage_depth: torch.Tensor,
        lineage_valid: torch.Tensor,
        token_rank: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, n_frames, token_count, _ = genes.shape
        if token_rank is None:
            rank_ids = torch.arange(token_count, device=genes.device).view(1, 1, token_count).expand(batch_size, n_frames, -1)
        else:
            rank_ids = token_rank.clamp(min=0, max=self.context_size - 1)

        global_time = torch.stack([time, future_time - time], dim=-1)
        global_time_emb = self.global_time_proj(global_time).view(batch_size, 1, 1, self.d_model)
        frame_ids = torch.arange(n_frames, device=genes.device)
        frame_emb = self.frame_index_emb(frame_ids).view(1, n_frames, 1, self.d_model)

        x = self.gene_proj(genes)
        x = x + global_time_emb
        x = x + self.token_time_proj(token_times.unsqueeze(-1))
        x = x + self.lineage_binary_proj(lineage_binary)
        x = x + self.founder_emb(founder_ids.clamp(min=0, max=self.founder_vocab_size - 1))
        x = x + self.lineage_depth_proj(lineage_depth.unsqueeze(-1))
        x = x + self.lineage_valid_emb(lineage_valid.long())
        x = x + self.token_rank_emb(rank_ids)
        x = x + frame_emb
        x = x + self.history_type.view(1, 1, 1, self.d_model)
        return x, rank_ids, global_time

    def build_future_queries(
        self,
        future_seed: torch.Tensor,
        query_rank: torch.Tensor,
        global_time: torch.Tensor,
        n_frames: int,
    ) -> torch.Tensor:
        future_tokens = future_seed + self.future_token_queries[query_rank]
        future_tokens = future_tokens + self.token_rank_emb(query_rank)
        future_tokens = future_tokens + self.frame_index_emb(torch.tensor([n_frames], device=future_seed.device)).view(1, 1, self.d_model)
        future_tokens = future_tokens + self.global_time_proj(global_time).unsqueeze(1)
        future_tokens = future_tokens + self.future_type
        return future_tokens


class FrameTokenEncoder(nn.Module):
    """Per-frame token encoder."""

    def __init__(self, d_model: int, n_heads: int, n_layers: int, head_dim: int):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlockAutoregressive(d_model, n_heads, head_dim) for _ in range(n_layers)]
        )
        self.d_model = d_model

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        batch_size, n_frames, token_count, _ = x.shape
        flat_x = x.view(batch_size * n_frames, token_count, self.d_model)
        flat_mask = valid_mask.view(batch_size * n_frames, token_count)
        for block in self.blocks:
            flat_x = block(flat_x, flat_mask)
        return flat_x.view(batch_size, n_frames, token_count, self.d_model)


class TokenTemporalBackbone(nn.Module):
    """Token-rank temporal mixer over frame-encoded tokens."""

    def __init__(self, d_model: int, n_heads: int, n_layers: int, head_dim: int):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlockAutoregressive(d_model, n_heads, head_dim) for _ in range(n_layers)]
        )
        self.d_model = d_model

    def forward(self, encoded_frames: torch.Tensor, valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, n_frames, token_count, _ = encoded_frames.shape
        token_temporal = encoded_frames.permute(0, 2, 1, 3).contiguous()
        token_temporal = token_temporal.view(batch_size * token_count, n_frames, self.d_model)
        temporal_mask = valid_mask.permute(0, 2, 1).contiguous().view(batch_size * token_count, n_frames)
        for block in self.blocks:
            token_temporal = block(token_temporal, temporal_mask)
        token_temporal = token_temporal.view(batch_size, token_count, n_frames, self.d_model).permute(0, 2, 1, 3).contiguous()

        last_valid_index = valid_mask.long().sum(dim=1).clamp_min(1) - 1
        gather_index = last_valid_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, self.d_model)
        future_seed = token_temporal.permute(0, 2, 1, 3).gather(2, gather_index).squeeze(2)
        return token_temporal, future_seed, last_valid_index


class FutureRefinementHead(nn.Module):
    """Refine future token seeds against history memory."""

    def __init__(self, d_model: int, n_heads: int, n_layers: int):
        super().__init__()
        self.blocks = nn.ModuleList([CrossAttentionRefinementBlock(d_model, n_heads) for _ in range(n_layers)])

    def forward(
        self,
        future_tokens: torch.Tensor,
        history_tokens: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = future_tokens
        for block in self.blocks:
            x = block(x, history_tokens, history_mask)
        return x


class LineageWholeEmbryoModel(nn.Module):
    """Lineage-first whole-embryo gene dynamics model."""

    def __init__(
        self,
        gene_dim: int,
        context_size: int,
        history_frames: int = 1,
        lineage_binary_dim: int = 20,
        founder_vocab_size: int = 15,
        d_model: int = 256,
        n_heads: int = 8,
        n_spatial_layers: int = 2,
        n_temporal_layers: int = 4,
        n_decoder_layers: int = 2,
        head_dim: int = 32,
    ):
        super().__init__()
        if history_frames < 1:
            raise ValueError("history_frames must be >= 1")
        self.gene_dim = gene_dim
        self.context_size = context_size
        self.history_frames = history_frames
        self.lineage_binary_dim = lineage_binary_dim
        self.founder_vocab_size = founder_vocab_size
        self.d_model = d_model
        self.embedder = LineageTokenEmbedder(
            gene_dim=gene_dim,
            context_size=context_size,
            history_frames=history_frames,
            lineage_binary_dim=lineage_binary_dim,
            founder_vocab_size=founder_vocab_size,
            d_model=d_model,
        )
        self.frame_encoder = FrameTokenEncoder(d_model=d_model, n_heads=n_heads, n_layers=n_spatial_layers, head_dim=head_dim)
        self.temporal_backbone = TokenTemporalBackbone(d_model=d_model, n_heads=n_heads, n_layers=n_temporal_layers, head_dim=head_dim)
        self.future_head = FutureRefinementHead(d_model=d_model, n_heads=n_heads, n_layers=n_decoder_layers)
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
        lineage_binary: torch.Tensor,
        founder_ids: torch.Tensor,
        lineage_depth: torch.Tensor,
        lineage_valid: torch.Tensor,
        token_rank: torch.Tensor | None = None,
    ) -> GenePatchVideoOutput:
        batch_size, n_frames, token_count, _ = genes.shape
        if n_frames != self.history_frames:
            raise ValueError(f"Expected {self.history_frames} history frames, got {n_frames}")
        if token_count > self.context_size:
            raise ValueError(f"Token count {token_count} exceeds context_size={self.context_size}")

        x, rank_ids, global_time = self.embedder(
            genes=genes,
            time=time,
            future_time=future_time,
            token_times=token_times,
            lineage_binary=lineage_binary,
            founder_ids=founder_ids,
            lineage_depth=lineage_depth,
            lineage_valid=lineage_valid,
            token_rank=token_rank,
        )
        encoded_frames = self.frame_encoder(x, valid_mask)
        token_temporal, future_seed, _last_valid_index = self.temporal_backbone(encoded_frames, valid_mask)
        query_rank = rank_ids[:, -1]
        future_tokens = self.embedder.build_future_queries(future_seed, query_rank, global_time, n_frames)
        history_tokens = token_temporal.view(batch_size, n_frames * token_count, self.d_model)
        history_mask = valid_mask.view(batch_size, n_frames * token_count)
        pred_future_token_states = self.future_head(future_tokens, history_tokens, history_mask)
        pred_history_genes = self.gene_head(token_temporal)
        pred_future_genes = self.gene_head(pred_future_token_states)
        pred_mean_gene = pred_future_genes.mean(dim=1)
        pred_future_frame_latent = future_seed.mean(dim=1)
        return GenePatchVideoOutput(
            pred_future_genes=pred_future_genes,
            pred_future_token_states=pred_future_token_states,
            pred_future_frame_latent=pred_future_frame_latent,
            pred_mean_gene=pred_mean_gene,
            pred_history_genes=pred_history_genes,
        )


__all__ = [
    "CrossAttentionRefinementBlock",
    "LineageTokenEmbedder",
    "FrameTokenEncoder",
    "TokenTemporalBackbone",
    "FutureRefinementHead",
    "LineageWholeEmbryoModel",
]
