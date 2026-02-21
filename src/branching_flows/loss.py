"""Loss functions for Branching Flows training.

Port of BranchingFlows.jl/src/loss.jl

Three loss components:

1. **Split loss** -- Bregman Poisson divergence between predicted and target
   remaining-split counts.
2. **Deletion loss** -- Binary cross-entropy (from logits) between predicted
   deletion probability and ground-truth deletion flag.
3. **Base process loss** -- application-dependent (e.g. MSE for continuous,
   cross-entropy for discrete).  Provided by the user; not in this module.

All losses support per-element masking via a ``padmask`` and optional
time-dependent scaling via :func:`loss_scale`.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Primitive losses
# ---------------------------------------------------------------------------

def bregman_poisson_loss(
    pred: torch.Tensor, target: torch.Tensor,
) -> torch.Tensor:
    """Shifted Bregman Poisson divergence (per-element, unreduced).

    ``sbpl(mu, c) = mu - c * log(mu) - (c - c * log(c))``

    where *mu* = ``pred`` (positive) and *c* = ``target`` (non-negative).
    The constant term ``c - c*log(c)`` makes the minimum 0.
    """
    eps = 1e-8
    pred = pred.clamp(min=eps)
    # xlogy(c, mu) = c * log(mu), with xlogy(0, *) = 0
    return pred - torch.xlogy(target, pred) - (target - torch.xlogy(target, target))


def logit_bce_loss(
    logits: torch.Tensor, target: torch.Tensor,
) -> torch.Tensor:
    """Numerically stable binary cross-entropy from logits (unreduced).

    ``lbce(z, y) = (1 - y) * z - log_sigmoid(z)``
    """
    return (1.0 - target) * logits - F.logsigmoid(logits)


# ---------------------------------------------------------------------------
# High-level losses
# ---------------------------------------------------------------------------

def split_loss(
    split_transform: Callable[[torch.Tensor], torch.Tensor],
    pred_logits: torch.Tensor,
    targets: torch.Tensor,
    padmask: torch.Tensor,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked Bregman Poisson loss for split-count prediction.

    Args:
        split_transform: Maps raw logits to positive intensities
            (e.g. ``lambda x: exp(clamp(x, -100, 11))``).
        pred_logits: Model output for splits, shape ``(batch, length)``.
        targets: Ground-truth remaining splits ``(batch, length)``.
        padmask: Bool mask ``(batch, length)``, True for valid positions.
        scale: Optional per-element weight ``(batch, length)`` or broadcastable.

    Returns:
        Scalar loss (masked mean).
    """
    pred = split_transform(pred_logits)
    elem_loss = bregman_poisson_loss(pred, targets)
    return _scaled_masked_mean(elem_loss, padmask, scale)


def deletion_loss(
    pred_logits: torch.Tensor,
    targets: torch.Tensor,
    padmask: torch.Tensor,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked logit BCE loss for deletion prediction.

    Args:
        pred_logits: Model output for deletion, shape ``(batch, length)``.
        targets: Ground-truth deletion flags ``(batch, length)`` float.
        padmask: Bool mask ``(batch, length)``.
        scale: Optional per-element weight.

    Returns:
        Scalar loss (masked mean).
    """
    elem_loss = logit_bce_loss(pred_logits, targets.float())
    return _scaled_masked_mean(elem_loss, padmask, scale)


# ---------------------------------------------------------------------------
# Loss scaling
# ---------------------------------------------------------------------------

def loss_scale(
    t: torch.Tensor,
    power: float = 1.0,
    min_val: float = 0.2,
) -> torch.Tensor:
    """Time-dependent per-element loss weight.

    Returns ``max(1 / (1 - t)^power, 1 / (1 - min_val)^power)``-normalized
    weights.  In practice this upweights samples near t=1 where predictions
    are more informative.

    Args:
        t: Time values, shape ``(batch,)`` or broadcastable.
        power: Exponent for ``1/(1-t)``.
        min_val: Clamp ``t`` below this to avoid extreme weights near t=0.
    """
    t_clamped = t.clamp(min=min_val)
    return (1.0 - t_clamped).pow(-power)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scaled_masked_mean(
    loss: torch.Tensor,
    mask: torch.Tensor,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean of *loss* over positions where *mask* is True, with optional *scale*."""
    mask_f = mask.float()
    if scale is not None:
        loss = loss * scale
    total = (loss * mask_f).sum()
    count = mask_f.sum().clamp(min=1.0)
    return total / count
