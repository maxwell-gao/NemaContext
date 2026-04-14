"""Minimal autoregressive transformer block retained for the worm mainline."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlockAutoregressive(nn.Module):
    """Transformer block used by the active lineage-first whole-embryo model."""

    def __init__(self, d_model: int, n_heads: int, head_dim: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.qkv = nn.Linear(d_model, 3 * n_heads * head_dim)
        self.attn_out = nn.Linear(n_heads * head_dim, d_model)
        self.ff_gate = nn.Linear(d_model, d_model * 4)
        self.ff_up = nn.Linear(d_model, d_model * 4)
        self.ff_out = nn.Linear(d_model * 4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, length, _ = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(batch_size, length, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if attn_bias is not None:
            scores = scores + attn_bias
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, 0.0)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, length, -1)
        x = x + self.attn_out(out)
        h = self.norm2(x)
        gate = F.silu(self.ff_gate(h))
        up = self.ff_up(h)
        return x + self.ff_out(gate * up)


__all__ = ["TransformerBlockAutoregressive"]
