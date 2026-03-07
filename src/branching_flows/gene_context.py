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


class GeneContextModel(nn.Module):
    """Time-conditioned multi-cell transformer over gene states."""

    def __init__(
        self,
        gene_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        head_dim: int = 32,
    ):
        super().__init__()
        self.gene_dim = gene_dim
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

    def forward(
        self,
        genes: torch.Tensor,
        time: torch.Tensor,
        future_time: torch.Tensor,
        token_times: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> GeneContextOutput:
        x = self.gene_proj(genes)
        global_time = torch.stack([time, future_time - time], dim=-1)
        time_emb = self.time_proj(global_time).unsqueeze(1)
        x = x + time_emb
        x = x + token_times.unsqueeze(-1)

        for block in self.blocks:
            x = block(x, valid_mask)

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


class SingleCellGeneTimeModel(nn.Module):
    """Single-cell gene+time baseline without multi-cell context."""

    def __init__(
        self,
        gene_dim: int,
        d_model: int = 256,
        n_layers: int = 3,
    ):
        super().__init__()
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
    ) -> GeneContextOutput:
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
        x = self.backbone(x)

        gene_delta = self.gene_head(x) * valid_mask.unsqueeze(-1)
        split_logits = self.split_head(x).squeeze(-1) * valid_mask
        del_logits = self.del_head(x).squeeze(-1) * valid_mask
        return GeneContextOutput(
            gene_delta=gene_delta,
            split_logits=split_logits,
            del_logits=del_logits,
        )
