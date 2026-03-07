"""Masked loss functions for trimodal training.

Handles missing modalities through learned masking rather than imputation,
following the Bitter Lesson principle.
"""

from __future__ import annotations

import torch


def masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """MSE loss with masking for missing values.

    Args:
        pred: Predicted values [B, D]
        target: Target values [B, D]
        mask: Binary mask [B, D] (1=available, 0=missing)

    Returns:
        Masked mean squared error
    """
    squared_error = (pred - target) ** 2
    masked_error = squared_error * mask

    # Normalize by number of available elements
    n_available = mask.sum().clamp(min=1.0)
    return masked_error.sum() / n_available


def masked_sinkhorn_divergence(
    pred: torch.Tensor,
    target: torch.Tensor,
    modality_mask: torch.Tensor,
    gene_dim: int,
    spatial_dim: int = 3,
    blur: float = 0.05,
) -> torch.Tensor:
    """Sinkhorn divergence with per-modality masking.

    Computes OT distance separately for each modality and combines
    based on availability.

    Args:
        pred: Predicted continuous state [B, L, D]
        target: Target continuous state [B, L, D]
        modality_mask: Modality availability [B, 3] (transcriptome, spatial, lineage)
        gene_dim: Dimension of gene expression
        spatial_dim: Dimension of spatial coordinates
        blur: Sinkhorn regularization

    Returns:
        Weighted Sinkhorn divergence
    """
    from ..emergent_loss import sinkhorn_divergence

    B, L, D = pred.shape

    # Split by modality
    pred_genes = pred[..., :gene_dim]
    target_genes = target[..., :gene_dim]
    pred_spatial = pred[..., gene_dim : gene_dim + spatial_dim]
    target_spatial = target[..., gene_dim : gene_dim + spatial_dim]

    loss = torch.tensor(0.0, device=pred.device)
    weights = torch.tensor(0.0, device=pred.device)

    # Gene expression loss (if transcriptome available)
    if modality_mask[..., 0].any():
        gene_mask = modality_mask[..., 0:1]  # [B, 1]
        gene_loss = sinkhorn_divergence(
            pred_genes * gene_mask.unsqueeze(-1),
            target_genes * gene_mask.unsqueeze(-1),
            blur=blur,
        )
        loss = loss + gene_loss * gene_mask.float().mean()
        weights = weights + gene_mask.float().mean()

    # Spatial loss (if spatial available)
    if modality_mask[..., 1].any():
        spatial_mask = modality_mask[..., 1:2]
        spatial_loss = sinkhorn_divergence(
            pred_spatial * spatial_mask.unsqueeze(-1),
            target_spatial * spatial_mask.unsqueeze(-1),
            blur=blur,
        )
        loss = loss + spatial_loss * spatial_mask.float().mean()
        weights = weights + spatial_mask.float().mean()

    return (
        loss / weights.clamp(min=1e-6)
        if weights > 0
        else torch.tensor(0.0, device=pred.device)
    )


def trimodal_context_loss(
    pred_continuous: torch.Tensor,
    target_continuous: torch.Tensor,
    pred_count: torch.Tensor,
    target_count: torch.Tensor,
    modality_mask: torch.Tensor,
    gene_dim: int,
    spatial_dim: int = 3,
    lambda_sinkhorn: float = 1.0,
    lambda_count: float = 0.1,
    lambda_diversity: float = 0.01,
    sinkhorn_blur: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined trimodal loss with masking.

    Args:
        pred_continuous: Predicted continuous state [B, L, D]
        target_continuous: Target continuous state [B, L, D]
        pred_count: Predicted cell count [B]
        target_count: Target cell count [B]
        modality_mask: Modality availability [B, 3]
        gene_dim: Dimension of gene expression
        spatial_dim: Dimension of spatial coordinates
        lambda_sinkhorn: Weight for Sinkhorn divergence
        lambda_count: Weight for count loss
        lambda_diversity: Weight for diversity loss
        sinkhorn_blur: Sinkhorn regularization

    Returns:
        (total_loss, loss_dict)
    """
    from ..emergent_loss import diversity_loss, cell_count_loss

    # Sinkhorn divergence (modality-aware)
    sink_loss = masked_sinkhorn_divergence(
        pred_continuous,
        target_continuous,
        modality_mask,
        gene_dim=gene_dim,
        spatial_dim=spatial_dim,
        blur=sinkhorn_blur,
    )

    # Cell count loss (always applies)
    count_loss = cell_count_loss(pred_count, target_count)

    # Diversity loss (always applies - prevents mode collapse)
    div_loss = diversity_loss(pred_continuous)

    # Total loss
    total = (
        lambda_sinkhorn * sink_loss
        + lambda_count * count_loss
        + lambda_diversity * div_loss
    )

    loss_dict = {
        "total": total.item(),
        "sinkhorn": sink_loss.item(),
        "count": count_loss.item(),
        "diversity": div_loss.item(),
    }

    return total, loss_dict


def curriculum_trimodal_loss(
    pred_continuous: torch.Tensor,
    target_continuous: torch.Tensor,
    pred_count: torch.Tensor,
    target_count: torch.Tensor,
    modality_mask: torch.Tensor,
    gene_dim: int,
    training_phase: int,
    spatial_dim: int = 3,
    lambda_sinkhorn: float = 1.0,
    lambda_count: float = 0.1,
    lambda_diversity: float = 0.01,
    sinkhorn_blur: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Curriculum-based trimodal loss.

    Training phases:
        1: Spatial-only focus
        2: Transcriptome-only focus
        3: Joint training with all modalities

    Args:
        pred_continuous: Predicted continuous state
        target_continuous: Target continuous state
        pred_count: Predicted cell count
        target_count: Target cell count
        modality_mask: Modality availability
        gene_dim: Dimension of gene expression
        training_phase: Current curriculum phase (1, 2, or 3)
        spatial_dim: Dimension of spatial coordinates
        lambda_sinkhorn: Weight for Sinkhorn divergence
        lambda_count: Weight for count loss
        lambda_diversity: Weight for diversity loss
        sinkhorn_blur: Sinkhorn regularization

    Returns:
        (total_loss, loss_dict)
    """
    # Modify mask based on curriculum phase
    modified_mask = modality_mask.clone()

    if training_phase == 1:
        # Phase 1: Only spatial matters
        modified_mask[..., 0] = 0  # Ignore transcriptome
        modified_mask[..., 2] = 0  # Ignore lineage (for loss)
    elif training_phase == 2:
        # Phase 2: Only transcriptome matters
        modified_mask[..., 1] = 0  # Ignore spatial
        modified_mask[..., 2] = 0  # Ignore lineage (for loss)
    # Phase 3: Use all available modalities

    return trimodal_context_loss(
        pred_continuous,
        target_continuous,
        pred_count,
        target_count,
        modified_mask,
        gene_dim=gene_dim,
        spatial_dim=spatial_dim,
        lambda_sinkhorn=lambda_sinkhorn,
        lambda_count=lambda_count,
        lambda_diversity=lambda_diversity,
        sinkhorn_blur=sinkhorn_blur,
    )


def weak_anchor_loss_masked(
    pred: torch.Tensor,
    target: torch.Tensor,
    act: torch.Tensor,
    modality_mask: torch.Tensor,
    gene_dim: int,
    spatial_weight: float = 1.0,
    gene_weight: float = 1.0,
) -> torch.Tensor:
    """Weak anchor loss with per-modality weighting.

    Args:
        pred: Predicted state [B, L, D]
        target: Target state [B, L, D]
        act: Active mask [B, L]
        modality_mask: Modality availability [B, 3]
        gene_dim: Dimension of gene expression
        spatial_weight: Weight for spatial component
        gene_weight: Weight for gene component

    Returns:
        Weighted MSE loss
    """
    B, L, D = pred.shape

    # Expand masks
    act_expanded = act.unsqueeze(-1).float()  # [B, L, 1]

    # Split by modality
    pred_genes = pred[..., :gene_dim]
    target_genes = target[..., :gene_dim]
    pred_spatial = pred[..., gene_dim : gene_dim + 3]
    target_spatial = target[..., gene_dim : gene_dim + 3]

    loss = 0.0

    # Gene loss
    if modality_mask[..., 0].any():
        gene_mask = modality_mask[..., 0:1].unsqueeze(1)  # [B, 1, 1]
        gene_diff = (pred_genes - target_genes) ** 2
        gene_loss = (
            gene_diff * act_expanded * gene_mask
        ).sum() / act_expanded.sum().clamp(min=1.0)
        loss += gene_weight * gene_loss

    # Spatial loss
    if modality_mask[..., 1].any():
        spatial_mask = modality_mask[..., 1:2].unsqueeze(1)
        spatial_diff = (pred_spatial - target_spatial) ** 2
        spatial_loss = (
            spatial_diff * act_expanded * spatial_mask
        ).sum() / act_expanded.sum().clamp(min=1.0)
        loss += spatial_weight * spatial_loss

    return loss
