"""Loss functions for BROT (Branching Regularized Optimal Transport).

Combines BranchingFlows per-element losses (port of loss.jl) with
RUOT-inspired distributional losses absorbed from DeepRUOTv2.

Per-element losses (from BranchingFlows):

1. Split loss -- Bregman Poisson divergence for remaining-split counts.
2. Deletion loss -- BCE for deletion probability.
3. Base process loss -- application-dependent (MSE, CE, etc.).

Distributional losses (from RUOT / DeepRUOTv2):

4. Sinkhorn distributional loss -- match generated cell POPULATION to real.
5. Mass matching loss -- ensure predicted cell count matches expected growth.
6. Energy regularization -- penalize kinetic energy for smooth trajectories.

Per-element losses supervise each cell individually.
Distributional losses enforce that the organism as a whole looks right.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Primitive losses (BranchingFlows)
# ---------------------------------------------------------------------------

def bregman_poisson_loss(
    pred: torch.Tensor, target: torch.Tensor,
) -> torch.Tensor:
    """Shifted Bregman Poisson divergence (per-element, unreduced).

    ``sbpl(mu, c) = mu - c * log(mu) - (c - c * log(c))``

    where *mu* = ``pred`` (positive) and *c* = ``target`` (non-negative).
    The constant term makes the minimum 0.
    """
    eps = 1e-8
    pred = pred.clamp(min=eps)
    return pred - torch.xlogy(target, pred) - (target - torch.xlogy(target, target))


def logit_bce_loss(
    logits: torch.Tensor, target: torch.Tensor,
) -> torch.Tensor:
    """Numerically stable binary cross-entropy from logits (unreduced).

    ``lbce(z, y) = (1 - y) * z - log_sigmoid(z)``
    """
    return (1.0 - target) * logits - F.logsigmoid(logits)


# ---------------------------------------------------------------------------
# High-level per-element losses (BranchingFlows)
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
        split_transform: Maps raw logits to positive intensities.
        pred_logits: ``(batch, length)`` model output for splits.
        targets: ``(batch, length)`` ground-truth remaining splits.
        padmask: ``(batch, length)`` bool, True for valid positions.
        scale: Optional per-element weight.

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
        pred_logits: ``(batch, length)`` model output for deletion.
        targets: ``(batch, length)`` ground-truth deletion flags.
        padmask: ``(batch, length)`` bool.
        scale: Optional per-element weight.

    Returns:
        Scalar loss (masked mean).
    """
    elem_loss = logit_bce_loss(pred_logits, targets.float())
    return _scaled_masked_mean(elem_loss, padmask, scale)


# ---------------------------------------------------------------------------
# BROT distributional losses (from RUOT / DeepRUOTv2)
# ---------------------------------------------------------------------------

def sinkhorn_distributional_loss(
    x1_pred: torch.Tensor,
    x1_anchor: torch.Tensor,
    padmask: torch.Tensor,
    blur: float = 0.1,
    p: int = 2,
) -> torch.Tensor:
    """Sinkhorn distance between predicted and real endpoint distributions.

    Unlike per-element MSE (which requires knowing which prediction matches
    which target), this measures distributional similarity -- the generated
    embryo should LOOK like the real embryo regardless of element ordering.

    Computes the Sinkhorn divergence per batch item and averages.

    Args:
        x1_pred: ``(batch, length, features)`` predicted continuous endpoints.
        x1_anchor: ``(batch, length, features)`` ground-truth anchors.
        padmask: ``(batch, length)`` bool mask for valid positions.
        blur: Sinkhorn blur parameter (entropic regularization).
        p: Exponent for the ground cost (2 = squared Euclidean).

    Returns:
        Scalar loss (mean Sinkhorn divergence over batch).
    """
    from geomloss import SamplesLoss

    loss_fn = SamplesLoss("sinkhorn", p=p, blur=blur, backend="tensorized")

    B = x1_pred.shape[0]
    total = torch.tensor(0.0, device=x1_pred.device)

    for b in range(B):
        mask_b = padmask[b]
        pred_b = x1_pred[b][mask_b]
        anchor_b = x1_anchor[b][mask_b]

        if pred_b.shape[0] < 2:
            continue

        total = total + loss_fn(pred_b, anchor_b)

    return total / max(B, 1)


def mass_matching_loss(
    split_logits: torch.Tensor,
    del_logits: torch.Tensor,
    expected_ratio: float,
    padmask: torch.Tensor,
    split_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Penalize if predicted total mass doesn't match expected cell count.

    Each element contributes ``(1 + split_intensity) * survival_prob`` to
    the predicted future mass. The total should match
    ``expected_ratio * current_count``.

    Args:
        split_logits: ``(batch, length)`` raw split intensity logits.
        del_logits: ``(batch, length)`` raw deletion logits.
        expected_ratio: Expected ratio of final to current cell count.
        padmask: ``(batch, length)`` bool mask.
        split_transform: Maps logits to positive intensities.

    Returns:
        Scalar loss.
    """
    if split_transform is None:
        split_transform = lambda x: torch.exp(torch.clamp(x, -100.0, 11.0))

    mask_f = padmask.float()
    current_count = mask_f.sum(dim=1)  # (B,)

    splits = split_transform(split_logits)
    survival = 1.0 - torch.sigmoid(del_logits)

    per_element_mass = (1.0 + splits) * survival * mask_f
    predicted_mass = per_element_mass.sum(dim=1)

    expected_mass = expected_ratio * current_count

    return ((predicted_mass - expected_mass) ** 2).mean()


def energy_regularization(
    x1_pred: torch.Tensor,
    xt_states: torch.Tensor,
    t: torch.Tensor,
    padmask: torch.Tensor,
) -> torch.Tensor:
    """Penalize kinetic energy of the predicted velocity field.

    From the Benamou-Brenier formulation of dynamical OT, the kinetic
    energy ``||v||^2`` where ``v = (x1_pred - xt) / (1 - t)`` should be
    minimized for optimal transport paths.

    Args:
        x1_pred: ``(batch, length, features)`` predicted endpoints.
        xt_states: ``(batch, length, features)`` current states at time t.
        t: ``(batch,)`` flow times.
        padmask: ``(batch, length)`` bool mask.

    Returns:
        Scalar loss (mean kinetic energy over valid positions).
    """
    denom = (1.0 - t).clamp(min=1e-4)
    denom = denom.unsqueeze(1).unsqueeze(2)

    velocity = (x1_pred - xt_states) / denom
    energy = (velocity ** 2).sum(dim=-1)

    return _scaled_masked_mean(energy, padmask)


# ---------------------------------------------------------------------------
# Loss scaling
# ---------------------------------------------------------------------------

def loss_scale(
    t: torch.Tensor,
    power: float = 1.0,
    min_val: float = 0.2,
) -> torch.Tensor:
    """Time-dependent per-element loss weight.

    Upweights samples near t=1 where predictions are more informative.

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
