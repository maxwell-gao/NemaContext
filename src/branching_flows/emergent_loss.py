"""Distribution-level losses for Emergent Context training.

This module implements losses that operate at the population/distribution level
rather than per-element supervision. The key principle: no X1_anchor, no
splits_target, no del_flags as training targets. The model learns to match
the data distribution through optimal transport.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    from geomloss import SamplesLoss

    HAS_GEOMLOSS = True
except ImportError:
    HAS_GEOMLOSS = False
    SamplesLoss = None


def sinkhorn_divergence(
    x_pred: torch.Tensor,
    x_real: torch.Tensor,
    blur: float = 0.05,
    scaling: float = 0.9,
    p: int = 2,
) -> torch.Tensor:
    """Compute Sinkhorn divergence between predicted and real distributions.

    This is the primary loss for emergent context training. It measures the
    distributional distance between predicted and real endpoint cell states
    without requiring per-element correspondence.

    Args:
        x_pred: Predicted endpoints, shape [B, N, D] where N is predicted cells
        x_real: Real endpoints, shape [B, M, D] where M is real cells (N != M ok)
        blur: Blur parameter for Sinkhorn (epsilon regularization)
        scaling: Multiscale scaling parameter for geomloss
        p: Power for the distance metric (2 = L2)

    Returns:
        Scalar Sinkhorn divergence loss
    """
    B = x_pred.shape[0]
    device = x_pred.device

    # Handle variable number of cells per batch
    total_loss = torch.tensor(0.0, device=device)

    for b in range(B):
        pred_b = x_pred[b]  # [N, D]
        real_b = x_real[b]  # [M, D]

        # Remove padding (all-zero rows)
        pred_mask = pred_b.abs().sum(dim=-1) > 1e-8
        real_mask = real_b.abs().sum(dim=-1) > 1e-8
        pred_valid = pred_b[pred_mask]
        real_valid = real_b[real_mask]

        if len(pred_valid) == 0 or len(real_valid) == 0:
            continue

        if HAS_GEOMLOSS and SamplesLoss is not None:
            # Use geomloss for Sinkhorn divergence
            # Note: geomloss expects [batch, n_points, dim]
            sinkhorn = SamplesLoss(
                loss="sinkhorn",
                p=p,
                blur=blur,
                scaling=scaling,
                backend="tensorized",
            )
            loss_b = sinkhorn(pred_valid.unsqueeze(0), real_valid.unsqueeze(0))
        else:
            # Fallback: Use maximum mean discrepancy (MMD) with RBF kernel
            loss_b = _mmd_rbf(pred_valid, real_valid)

        total_loss = total_loss + loss_b

    return total_loss / B if B > 0 else total_loss


def _mmd_rbf(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Maximum Mean Discrepancy with RBF kernel (fallback when geomloss unavailable).

    Args:
        x: First sample [N, D]
        y: Second sample [M, D]
        sigma: RBF kernel bandwidth

    Returns:
        MMD^2 value
    """

    def rbf_kernel(a, b, sigma):
        # Compute RBF kernel matrix
        dist_sq = torch.cdist(a, b, p=2) ** 2
        return torch.exp(-dist_sq / (2 * sigma**2))

    xx = rbf_kernel(x, x, sigma)
    yy = rbf_kernel(y, y, sigma)
    xy = rbf_kernel(x, y, sigma)

    # MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
    mmd = xx.mean() + yy.mean() - 2 * xy.mean()
    return mmd.clamp(min=0.0)


def cell_count_loss(
    predicted_count: torch.Tensor,
    target_count: torch.Tensor,
    loss_type: str = "l2",
) -> torch.Tensor:
    """Soft constraint on final cell count.

    Args:
        predicted_count: Predicted number of cells, shape [B]
        target_count: Target number of cells, shape [B]
        loss_type: Type of loss ("l1", "l2", "huber")

    Returns:
        Scalar count matching loss
    """
    diff = predicted_count.float() - target_count.float()

    if loss_type == "l1":
        return diff.abs().mean()
    elif loss_type == "l2":
        return diff.pow(2).mean()
    elif loss_type == "huber":
        delta = 1.0
        return torch.where(
            diff.abs() <= delta, 0.5 * diff.pow(2), delta * (diff.abs() - 0.5 * delta)
        ).mean()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


def temporal_smoothness(
    trajectory: list[torch.Tensor],
    order: int = 1,
) -> torch.Tensor:
    """Encourage smooth developmental trajectories.

    Penalizes high-frequency changes in cell states over time.

    Args:
        trajectory: List of cell states at different time points
        order: Order of smoothness (1 = velocity, 2 = acceleration)

    Returns:
        Scalar smoothness loss
    """
    if len(trajectory) < order + 1:
        return torch.tensor(0.0, device=trajectory[0].device)

    total_variation = torch.tensor(0.0, device=trajectory[0].device)

    if order == 1:
        # Penalize large velocities
        for t in range(len(trajectory) - 1):
            diff = trajectory[t + 1] - trajectory[t]
            total_variation = total_variation + diff.pow(2).mean()

    elif order == 2:
        # Penalize large accelerations
        for t in range(len(trajectory) - 2):
            acc = trajectory[t + 2] - 2 * trajectory[t + 1] + trajectory[t]
            total_variation = total_variation + acc.pow(2).mean()

    return total_variation / max(len(trajectory) - order, 1)


def physics_constraints(
    positions: torch.Tensor,
    volumes: torch.Tensor | None = None,
    parent_child_pairs: list[tuple[int, int, int]] | None = None,
    min_cell_distance: float = 0.5,
    volume_conservation_weight: float = 1.0,
    non_overlap_weight: float = 1.0,
) -> torch.Tensor:
    """Penalize physically impossible cell configurations.

    Args:
        positions: Cell spatial positions [B, N, 3]
        volumes: Cell volumes [B, N] (optional)
        parent_child_pairs: List of (parent, child1, child2) indices
        min_cell_distance: Minimum allowed distance between cells
        volume_conservation_weight: Weight for volume conservation term
        non_overlap_weight: Weight for non-overlap term

    Returns:
        Scalar physics constraint loss
    """
    device = positions.device
    loss = torch.tensor(0.0, device=device)

    # Volume conservation: daughter volumes ≈ parent volume
    if volumes is not None and parent_child_pairs is not None:
        for parent, child1, child2 in parent_child_pairs:
            if (
                parent < volumes.shape[1]
                and child1 < volumes.shape[1]
                and child2 < volumes.shape[1]
            ):
                vol_diff = volumes[:, parent] - volumes[:, child1] - volumes[:, child2]
                loss = loss + vol_diff.pow(2).mean() * volume_conservation_weight

    # Non-overlap: cells shouldn't occupy same space
    # Compute pairwise distances
    distances = torch.cdist(positions, positions)  # [B, N, N]

    # Create mask to exclude diagonal (self-distances)
    mask = ~torch.eye(positions.shape[1], dtype=torch.bool, device=device).unsqueeze(0)

    # Penalize distances below threshold
    overlap = F.relu(min_cell_distance - distances)
    overlap = overlap * mask.float()
    loss = loss + overlap.sum() * non_overlap_weight

    return loss


def diversity_loss(
    x: torch.Tensor,
    method: str = "pairwise",
    temperature: float = 0.1,
) -> torch.Tensor:
    """Encourage diversity among predicted cells to prevent mode collapse.

    Mode collapse is a risk in distribution-only training where the model
    predicts the same state for all cells. This loss encourages diversity.

    Args:
        x: Predicted cell states [B, N, D]
        method: Diversity method ("pairwise", "determinantal", "batch_disc")
        temperature: Temperature for similarity computation

    Returns:
        Scalar diversity loss (negative = more diversity)
    """
    B, N, D = x.shape

    if method == "pairwise":
        # Maximize pairwise distances
        # Compute pairwise similarities
        x_norm = F.normalize(x, dim=-1)  # [B, N, D]
        sim = torch.bmm(x_norm, x_norm.transpose(1, 2))  # [B, N, N]

        # Remove diagonal
        mask = ~torch.eye(N, dtype=torch.bool, device=x.device).unsqueeze(0)
        sim_offdiag = sim[mask.expand(B, -1, -1)].view(B, N, N - 1)

        # Minimize similarity (maximize diversity)
        # Loss is negative of similarity to encourage minimization
        diversity = -sim_offdiag.mean()

    elif method == "determinantal":
        # Determinantal point process diversity
        # Higher determinant = more diverse
        x_norm = F.normalize(x, dim=-1)
        gram = torch.bmm(x_norm, x_norm.transpose(1, 2))  # [B, N, N]

        # Add small identity for numerical stability
        gram = gram + 1e-5 * torch.eye(N, device=x.device).unsqueeze(0)

        # Loss is negative log determinant
        try:
            det = torch.linalg.det(gram)
            diversity = -torch.log(det + 1e-8).mean()
        except RuntimeError:
            # Fallback if det computation fails
            diversity = torch.tensor(0.0, device=x.device)

    else:
        raise ValueError(f"Unknown diversity method: {method}")

    return diversity


def emergent_context_loss(
    x_pred: torch.Tensor,
    x_real: torch.Tensor,
    lineage_bias: torch.Tensor | None,
    predicted_count: torch.Tensor,
    real_count: torch.Tensor,
    positions: torch.Tensor | None = None,
    lambda_sinkhorn: float = 1.0,
    lambda_count: float = 0.1,
    lambda_diversity: float = 0.01,
    lambda_physics: float = 0.001,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined loss for emergent context training.

    No per-element supervision—only population-level constraints.

    Args:
        x_pred: Predicted endpoints [B, N, D]
        x_real: Real endpoints [B, M, D]
        lineage_bias: Attention bias used (for logging/debugging)
        predicted_count: Predicted cell count [B]
        real_count: Real cell count [B]
        positions: Spatial positions for physics constraints [B, N, 3]
        lambda_sinkhorn: Weight for Sinkhorn divergence
        lambda_count: Weight for count matching
        lambda_diversity: Weight for diversity loss
        lambda_physics: Weight for physics constraints

    Returns:
        Tuple of (total_loss, loss_dict) where loss_dict contains
        individual loss components for logging.
    """
    # 1. Distribution matching (Sinkhorn)
    loss_sinkhorn = sinkhorn_divergence(x_pred, x_real)

    # 2. Cell count matching
    loss_count = cell_count_loss(predicted_count, real_count)

    # 3. Diversity to prevent mode collapse
    loss_diversity = diversity_loss(x_pred)

    # 4. Physics constraints
    loss_physics = torch.tensor(0.0, device=x_pred.device)
    if positions is not None:
        loss_physics = physics_constraints(positions)

    # Combined loss
    total = (
        lambda_sinkhorn * loss_sinkhorn
        + lambda_count * loss_count
        + lambda_diversity * loss_diversity
        + lambda_physics * loss_physics
    )

    loss_dict = {
        "total": total.item(),
        "sinkhorn": loss_sinkhorn.item(),
        "count": loss_count.item(),
        "diversity": loss_diversity.item(),
        "physics": loss_physics.item(),
    }

    return total, loss_dict


def weak_anchor_loss(
    x_pred: torch.Tensor,
    x_anchor: torch.Tensor,
    mask: torch.Tensor,
    weight: float = 0.1,
) -> torch.Tensor:
    """Optional weak anchor loss for curriculum learning.

    Can be used early in training to stabilize, then gradually removed.

    Args:
        x_pred: Predicted endpoints [B, N, D]
        x_anchor: Anchor targets [B, N, D]
        mask: Valid positions mask [B, N]
        weight: Weight for this loss (can be annealed to 0)

    Returns:
        Weighted MSE loss
    """
    if weight == 0:
        return torch.tensor(0.0, device=x_pred.device)

    diff = (x_pred - x_anchor).pow(2).sum(dim=-1)  # [B, N]
    mask_f = mask.float()
    mse = (diff * mask_f).sum() / mask_f.sum().clamp(min=1.0)

    return weight * mse
