"""Transformer model for NemaContext BranchingFlows training.

Architecture follows the BranchingFlows demo (Toy model) pattern:
- Learned projection from raw cell-token (HVG expression + spatial) to d_model
- Discrete embedding for founder identity
- RoPE positional encoding
- adaLN-Zero time conditioning via Random Fourier Features
- Full self-attention over ALL cells (organism as context)
- Four output heads: continuous endpoint, discrete endpoint, splits, deletions
"""

from __future__ import annotations


import torch
import torch.nn as nn
import torch.nn.functional as F

from .states import BranchingState


class NemaFlowModel(nn.Module):
    """Transformer for predicting BranchingFlows training targets.

    Args:
        continuous_dim: Dimensionality of continuous state (n_hvg + 3).
        discrete_K: Number of discrete categories (founders + mask).
        d_model: Transformer hidden dimension.
        n_heads: Number of attention heads.
        n_layers: Number of Transformer blocks.
        head_dim: Dimension per attention head.
        rff_dim: Random Fourier Feature dimension for time embedding.
        max_seq_len: Maximum sequence length for RoPE.
    """

    def __init__(
        self,
        continuous_dim: int,
        discrete_K: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        head_dim: int = 32,
        rff_dim: int = 256,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.d_model = d_model
        self.continuous_dim = continuous_dim
        self.discrete_K = discrete_K

        # Input projections
        self.continuous_proj = nn.Linear(continuous_dim, d_model)
        self.discrete_embed = nn.Embedding(discrete_K, d_model)

        # Time conditioning: RFF -> projection -> adaLN parameters
        self.rff = RandomFourierFeatures(1, rff_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(rff_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # RoPE
        self.rope = RotaryPositionalEncoding(head_dim, max_seq_len)

        # Transformer blocks with adaLN-Zero
        self.blocks = nn.ModuleList(
            [AdaLNTransformerBlock(d_model, n_heads, head_dim) for _ in range(n_layers)]
        )

        # Output heads
        self.continuous_head = nn.Linear(d_model, continuous_dim)
        self.discrete_head = nn.Linear(d_model, discrete_K)
        self.split_head = nn.Linear(d_model, 1)
        self.del_head = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        for block in self.blocks:
            nn.init.zeros_(block.attn_out.weight)
            nn.init.zeros_(block.ff_out.weight)

    def forward(
        self,
        t: torch.Tensor,
        state: BranchingState,
        lineage_bias: torch.Tensor | None = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            t: Flow time, shape ``(B,)``.
            state: BranchingState with ``states[0]`` continuous ``(B, L, D)``
                and ``states[1]`` discrete ``(B, L)``.
            lineage_bias: Optional lineage-based attention bias of shape ``(B, L, L)``.
                Computed from pairwise lineage distances. Closer cells in the
                lineage tree receive higher attention bias.

        Returns:
            ``((x1_cont, x1_disc_logits), split_logits, del_logits)`` where:
            - ``x1_cont``: ``(B, L, continuous_dim)`` predicted endpoint.
            - ``x1_disc_logits``: ``(B, L, K)`` predicted discrete logits.
            - ``split_logits``: ``(B, L)`` per-element split intensity.
            - ``del_logits``: ``(B, L)`` per-element deletion logit.
        """
        cont = state.states[0]  # (B, L, D)
        disc = state.states[1].long()  # (B, L)
        padmask = state.padmask  # (B, L)

        B, L, _ = cont.shape

        # Encode inputs
        x = self.continuous_proj(cont) + self.discrete_embed(disc)

        # Time conditioning
        t_feat = self.rff(t.unsqueeze(-1))  # (B, rff_dim)
        t_cond = self.time_proj(t_feat)  # (B, d_model)

        # RoPE frequencies
        rope_cos, rope_sin = self.rope(L, device=x.device)

        # Attention mask from padmask: (B, 1, 1, L) for broadcasting
        attn_mask = padmask.unsqueeze(1).unsqueeze(2)  # True = attend

        # Transformer blocks
        for block in self.blocks:
            x = block(x, t_cond, rope_cos, rope_sin, attn_mask, lineage_bias)

        # Output heads
        x1_cont = self.continuous_head(x)
        x1_disc = self.discrete_head(x)
        split_logits = self.split_head(x).squeeze(-1)
        del_logits = self.del_head(x).squeeze(-1)

        return (x1_cont, x1_disc), split_logits, del_logits


# ---------------------------------------------------------------------------
# Transformer building blocks
# ---------------------------------------------------------------------------


class AdaLNTransformerBlock(nn.Module):
    """Transformer block with adaptive layer normalization (adaLN-Zero).

    Time conditioning is injected by predicting per-sample scale and shift
    for layer norm, plus a gating factor for the residual.
    """

    def __init__(self, d_model: int, n_heads: int, head_dim: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        inner_dim = n_heads * head_dim

        # Attention
        self.qkv = nn.Linear(d_model, 3 * inner_dim, bias=False)
        self.attn_out = nn.Linear(inner_dim, d_model, bias=False)

        # Feed-forward (SwiGLU)
        ff_dim = int(d_model * 8 / 3)
        ff_dim = ((ff_dim + 63) // 64) * 64  # round to 64
        self.ff_gate = nn.Linear(d_model, ff_dim, bias=False)
        self.ff_up = nn.Linear(d_model, ff_dim, bias=False)
        self.ff_out = nn.Linear(ff_dim, d_model, bias=False)

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)

        # adaLN-Zero: predict (scale1, shift1, gate1, scale2, shift2, gate2)
        self.ada_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model),
        )
        nn.init.zeros_(self.ada_proj[1].weight)
        nn.init.zeros_(self.ada_proj[1].bias)

    def forward(
        self,
        x: torch.Tensor,
        t_cond: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        attn_mask: torch.Tensor,
        lineage_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, L, D = x.shape

        # adaLN parameters from time conditioning
        ada = self.ada_proj(t_cond)  # (B, 6*D)
        s1, sh1, g1, s2, sh2, g2 = ada.chunk(6, dim=-1)  # each (B, D)

        # -- Attention branch --
        h = self.norm1(x)
        h = h * (1 + s1.unsqueeze(1)) + sh1.unsqueeze(1)

        qkv = self.qkv(h).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, L, H, Hd)
        q, k = q.transpose(1, 2), k.transpose(1, 2)  # (B, H, L, Hd)
        v = v.transpose(1, 2)

        # Apply RoPE
        q = _apply_rope(q, rope_cos, rope_sin)
        k = _apply_rope(k, rope_cos, rope_sin)

        # Scaled dot-product attention with optional lineage bias
        scale = self.head_dim**-0.5
        attn = (q @ k.transpose(-2, -1)) * scale

        # Apply lineage bias if provided (broadcast across heads)
        if lineage_bias is not None:
            # lineage_bias: [B, L, L] -> [B, 1, L, L] for broadcasting
            attn = attn + lineage_bias.unsqueeze(1)

        attn = attn.masked_fill(~attn_mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = attn.masked_fill(~attn_mask, 0.0)

        out = (attn @ v).transpose(1, 2).reshape(B, L, -1)
        out = self.attn_out(out)
        x = x + g1.unsqueeze(1) * out

        # -- Feed-forward branch --
        h = self.norm2(x)
        h = h * (1 + s2.unsqueeze(1)) + sh2.unsqueeze(1)

        ff = F.silu(self.ff_gate(h)) * self.ff_up(h)
        ff = self.ff_out(ff)
        x = x + g2.unsqueeze(1) * ff

        return x


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


class RotaryPositionalEncoding(nn.Module):
    """Rotary Positional Encoding (RoPE)."""

    def __init__(self, dim: int, max_len: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_len = max_len

    def forward(
        self,
        seq_len: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device))  # (L, dim/2)
        cos = freqs.cos()  # (L, dim/2)
        sin = freqs.sin()  # (L, dim/2)
        return cos, sin


def _apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE to the last dimension of *x* ``(B, H, L, Hd)``."""
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    cos = cos[: x.shape[2], :d2].unsqueeze(0).unsqueeze(0)
    sin = sin[: x.shape[2], :d2].unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# ---------------------------------------------------------------------------
# Random Fourier Features
# ---------------------------------------------------------------------------


class RandomFourierFeatures(nn.Module):
    """Random Fourier Features for time embedding."""

    def __init__(self, in_dim: int, out_dim: int, scale: float = 1.0):
        super().__init__()
        self.register_buffer(
            "W",
            torch.randn(in_dim, out_dim // 2) * scale,
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x @ self.W  # (..., out_dim//2)
        return torch.cat([proj.cos(), proj.sin()], dim=-1)
