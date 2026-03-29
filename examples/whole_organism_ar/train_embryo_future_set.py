#!/usr/bin/env python3
"""Train embryo-level masked future local-view set prediction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from examples.whole_organism_ar.train_embryo_state import stack_view_tensor  # noqa: E402
from examples.whole_organism_ar.train_gene_context import (  # noqa: E402
    EVENT_SUBSET_THRESHOLDS,
    resolve_event_filters,
)
from src.branching_flows.emergent_loss import sinkhorn_divergence  # noqa: E402
from src.branching_flows.gene_context import (  # noqa: E402
    EmbryoFutureSetModel,
    EmbryoMaskedViewModel,
    FrozenLinearTokenReadout,
    PooledLatentCanonicalizer,
)
from src.data.gene_context_dataset import EmbryoViewDataset, collate_embryo_view  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Train embryo masked future local-view set prediction.")
    p.add_argument("--model_type", choices=["multi_cell", "single_cell"], default="multi_cell")
    p.add_argument("--backbone_checkpoint", default=None)
    p.add_argument("--freeze_backbone", action="store_true", default=True)
    p.add_argument("--no_freeze_backbone", dest="freeze_backbone", action="store_false")
    p.add_argument("--h5ad_path", default="dataset/processed/nema_extended_large2025.h5ad")
    p.add_argument("--n_hvg", type=int, default=256)
    p.add_argument("--context_size", type=int, default=256)
    p.add_argument("--global_context_size", type=int, default=32)
    p.add_argument("--dt_minutes", type=float, default=40.0)
    p.add_argument("--time_window_minutes", type=float, default=10.0)
    p.add_argument("--samples_per_pair", type=int, default=16)
    p.add_argument("--val_samples_per_pair", type=int, default=None)
    p.add_argument("--views_per_embryo", type=int, default=8)
    p.add_argument("--future_views_per_embryo", type=int, default=8)
    p.add_argument("--top_cell_types", type=int, default=8)
    p.add_argument("--context_mask_ratio", type=float, default=0.0)
    p.add_argument("--future_mask_ratio", type=float, default=0.25)
    p.add_argument("--min_cells_per_window", type=int, default=32)
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument(
        "--sampling_strategy",
        choices=["random_window", "spatial_neighbors", "spatial_anchor"],
        default="spatial_anchor",
    )
    p.add_argument("--min_spatial_cells_per_window", type=int, default=8)
    p.add_argument("--spatial_neighbor_pool_size", type=int, default=None)
    p.add_argument("--event_subset", choices=sorted(EVENT_SUBSET_THRESHOLDS), default="none")
    p.add_argument("--min_event_positive", type=int, default=0)
    p.add_argument("--min_anchor_event_positive", type=int, default=0)
    p.add_argument("--min_split_positive", type=int, default=0)
    p.add_argument("--min_del_positive", type=int, default=0)
    p.add_argument("--min_anchor_split_positive", type=int, default=0)
    p.add_argument("--min_anchor_del_positive", type=int, default=0)
    p.add_argument("--val_event_subset", choices=sorted(EVENT_SUBSET_THRESHOLDS), default="none")
    p.add_argument("--val_min_event_positive", type=int, default=None)
    p.add_argument("--val_min_anchor_event_positive", type=int, default=None)
    p.add_argument("--val_min_split_positive", type=int, default=None)
    p.add_argument("--val_min_del_positive", type=int, default=None)
    p.add_argument("--val_min_anchor_split_positive", type=int, default=None)
    p.add_argument("--val_min_anchor_del_positive", type=int, default=None)
    p.add_argument("--delete_target_mode", choices=["weak", "strict"], default="strict")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--head_dim", type=int, default=32)
    p.add_argument("--pairwise_spatial_bias", action="store_true")
    p.add_argument("--decoder_layers", type=int, default=3)
    p.add_argument("--use_current_local_tokens", action="store_true", default=True)
    p.add_argument("--no_use_current_local_tokens", dest="use_current_local_tokens", action="store_false")
    p.add_argument(
        "--current_conditioning_mode",
        choices=["flat_tokens", "cross_attention_memory"],
        default="flat_tokens",
    )
    p.add_argument("--learn_current_token_gate", action="store_true", default=True)
    p.add_argument("--no_learn_current_token_gate", dest="learn_current_token_gate", action="store_false")
    p.add_argument("--current_token_gate_init", type=float, default=0.5)
    p.add_argument("--latent_set_weight", type=float, default=0.0)
    p.add_argument("--gene_set_weight", type=float, default=0.0)
    p.add_argument("--mean_latent_weight", type=float, default=0.0)
    p.add_argument("--pooled_canonicalizer_path", default=None)
    p.add_argument("--pooled_align_weight", type=float, default=0.0)
    p.add_argument("--pooled_var_weight", type=float, default=0.0)
    p.add_argument("--pooled_cov_weight", type=float, default=0.0)
    p.add_argument("--token_readout_anchor_path", default=None)
    p.add_argument("--token_readout_anchor_weight", type=float, default=0.0)
    p.add_argument("--code_tokens", type=int, default=8)
    p.add_argument("--predict_dense_future_tokens", action="store_true", default=True)
    p.add_argument("--no_predict_dense_future_tokens", dest="predict_dense_future_tokens", action="store_false")
    p.add_argument("--strict_token_jepa", action="store_true", default=True)
    p.add_argument("--no_strict_token_jepa", dest="strict_token_jepa", action="store_false")
    p.add_argument("--local_code_weight", type=float, default=1.0)
    p.add_argument("--token_var_weight", type=float, default=0.01)
    p.add_argument("--token_cov_weight", type=float, default=0.01)
    p.add_argument("--decoded_gene_weight", type=float, default=0.0)
    p.add_argument("--decoded_spatial_weight", type=float, default=0.0)
    p.add_argument("--decoded_mean_gene_weight", type=float, default=0.0)
    p.add_argument("--decoded_count_weight", type=float, default=0.0)
    p.add_argument("--explicit_count_weight", type=float, default=0.0)
    p.add_argument("--explicit_split_weight", type=float, default=0.0)
    p.add_argument("--survival_weight", type=float, default=0.0)
    p.add_argument("--split_count_weight", type=float, default=0.0)
    p.add_argument("--sinkhorn_blur", type=float, default=0.05)
    p.add_argument("--gene_sinkhorn_blur", type=float, default=0.1)
    p.add_argument("--decoded_gene_sinkhorn_blur", type=float, default=0.1)
    p.add_argument("--decoded_spatial_sinkhorn_blur", type=float, default=0.1)
    p.add_argument("--decoder_position_weight", type=float, default=0.25)
    p.add_argument("--decoder_spatial_flag_weight", type=float, default=0.5)
    p.add_argument("--decoder_valid_gate_threshold", type=float, default=0.5)
    p.add_argument("--checkpoint_dir", default="checkpoints_embryo_future_set")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_mask(
    batch_size: int,
    n_views: int,
    mask_ratio: float,
    device: torch.device,
    allow_empty: bool = False,
) -> torch.Tensor:
    if n_views < 1:
        raise ValueError("n_views must be >= 1")
    if allow_empty and mask_ratio <= 0.0:
        return torch.zeros(batch_size, n_views, dtype=torch.bool, device=device)
    min_mask = 0 if allow_empty else 1
    max_mask = max(0, n_views - 1) if allow_empty else max(1, n_views - 1)
    n_mask = int(round(mask_ratio * n_views))
    n_mask = max(min_mask, min(max_mask, n_mask))
    if not allow_empty and n_mask == 0:
        n_mask = 1
    mask = torch.zeros(batch_size, n_views, dtype=torch.bool, device=device)
    for i in range(batch_size):
        perm = torch.randperm(n_views, device=device)
        if n_mask > 0:
            mask[i, perm[:n_mask]] = True
    return mask


def load_backbone(args) -> tuple[EmbryoMaskedViewModel, dict | None]:
    if args.backbone_checkpoint is None:
        backbone = EmbryoMaskedViewModel(
            gene_dim=args.n_hvg,
            context_size=args.context_size,
            model_type=args.model_type,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            head_dim=args.head_dim,
            use_pairwise_spatial_bias=args.pairwise_spatial_bias,
        )
        return backbone, None

    ckpt = torch.load(args.backbone_checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    backbone = EmbryoMaskedViewModel(
        gene_dim=ckpt["gene_dim"],
        context_size=cfg["context_size"],
        model_type=cfg["model_type"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        head_dim=cfg["head_dim"],
        use_pairwise_spatial_bias=cfg["pairwise_spatial_bias"],
    )
    state_dict = ckpt["model_state_dict"]
    if any(key.startswith("online_backbone.") for key in state_dict):
        state_dict = {
            key.removeprefix("online_backbone."): value
            for key, value in state_dict.items()
            if key.startswith("online_backbone.")
        }
    backbone.load_state_dict(state_dict)
    return backbone, ckpt


def load_pooled_canonicalizer(path: str | None, d_model: int) -> PooledLatentCanonicalizer:
    canonicalizer = PooledLatentCanonicalizer(d_model)
    if path is None:
        return canonicalizer
    payload = torch.load(path, map_location="cpu")
    state_dict = payload.get("canonicalizer_state_dict", payload.get("state_dict", payload))
    canonicalizer.load_state_dict(state_dict)
    return canonicalizer


def load_token_readout_anchor(path: str | None, d_model: int) -> FrozenLinearTokenReadout | None:
    if path is None:
        return None
    payload = torch.load(path, map_location="cpu")
    state_dict = payload.get("token_readout_state_dict", payload.get("state_dict", payload))
    readout = FrozenLinearTokenReadout(d_model)
    readout.load_state_dict(state_dict)
    return readout


def pooled_variance_covariance_losses(
    pooled_latent: torch.Tensor,
    target_std: float = 1.0,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pooled_latent.shape[0] < 2:
        zero = pooled_latent.new_zeros(())
        return zero, zero
    centered = pooled_latent - pooled_latent.mean(dim=0, keepdim=True)
    std = torch.sqrt(centered.var(dim=0, unbiased=False) + eps)
    var_loss = torch.relu(target_std - std).mean()
    cov = centered.T @ centered / max(1, pooled_latent.shape[0] - 1)
    off_diag = cov - torch.diag_embed(torch.diagonal(cov))
    cov_loss = off_diag.pow(2).mean()
    return var_loss, cov_loss


def token_variance_covariance_losses(
    token_states: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    target_std: float = 1.0,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat = token_states.reshape(-1, token_states.shape[-1])
    if valid_mask is not None:
        flat = flat[valid_mask.reshape(-1)]
    if flat.shape[0] < 2:
        zero = token_states.new_zeros(())
        return zero, zero
    centered = flat - flat.mean(dim=0, keepdim=True)
    std = torch.sqrt(centered.var(dim=0, unbiased=False) + eps)
    var_loss = torch.relu(target_std - std).mean()
    cov = centered.T @ centered / max(1, flat.shape[0] - 1)
    off_diag = cov - torch.diag_embed(torch.diagonal(cov))
    cov_loss = off_diag.pow(2).mean()
    return var_loss, cov_loss


def build_target_structured_state(
    genes: torch.Tensor,
    relative_position: torch.Tensor,
    valid_mask: torch.Tensor,
    position_weight: float,
    spatial_flag_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    norm_genes = F.layer_norm(genes, (genes.shape[-1],))
    target_spatial_valid = valid_mask & (relative_position[..., 4] > 0.5)
    target_positions = relative_position[..., :3] * target_spatial_valid.unsqueeze(-1).float()
    target_gene_state = norm_genes * valid_mask.unsqueeze(-1).float()
    target_spatial_state = torch.cat(
        [
            position_weight * target_positions,
            spatial_flag_weight * target_spatial_valid.unsqueeze(-1).float(),
        ],
        dim=-1,
    )
    target_spatial_state = target_spatial_state * valid_mask.unsqueeze(-1).float()
    return target_gene_state, target_spatial_state, target_spatial_valid, target_positions


def build_pred_structured_state(
    pred_genes: torch.Tensor,
    pred_positions: torch.Tensor,
    pred_valid_logits: torch.Tensor,
    pred_spatial_logits: torch.Tensor,
    position_weight: float,
    spatial_flag_weight: float,
    valid_gate_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_valid_prob = torch.sigmoid(pred_valid_logits)
    if valid_gate_threshold > 0.0:
        pred_valid = F.relu(pred_valid_prob - valid_gate_threshold) / max(1e-6, 1.0 - valid_gate_threshold)
    else:
        pred_valid = pred_valid_prob
    pred_valid = pred_valid.unsqueeze(-1)
    pred_spatial = torch.sigmoid(pred_spatial_logits).unsqueeze(-1)
    norm_pred_genes = F.layer_norm(pred_genes, (pred_genes.shape[-1],))
    pred_gene_state = norm_pred_genes * pred_valid
    pred_spatial_state = torch.cat(
        [
            position_weight * (pred_positions * pred_spatial),
            spatial_flag_weight * pred_spatial,
        ],
        dim=-1,
    )
    pred_spatial_state = pred_spatial_state * pred_valid
    return pred_gene_state, pred_spatial_state, pred_valid_prob, pred_spatial.squeeze(-1)


def compute_metrics(
    model: EmbryoFutureSetModel,
    batch: dict[str, torch.Tensor],
    context_mask_ratio: float,
    future_mask_ratio: float,
    latent_set_weight: float,
    gene_set_weight: float,
    mean_latent_weight: float,
    pooled_align_weight: float,
    pooled_var_weight: float,
    pooled_cov_weight: float,
    token_readout_anchor_weight: float,
    local_code_weight: float,
    token_var_weight: float,
    token_cov_weight: float,
    decoded_gene_weight: float,
    decoded_spatial_weight: float,
    decoded_mean_gene_weight: float,
    decoded_count_weight: float,
    explicit_count_weight: float,
    explicit_split_weight: float,
    survival_weight: float,
    split_count_weight: float,
    sinkhorn_blur: float,
    gene_sinkhorn_blur: float,
    decoded_gene_sinkhorn_blur: float,
    decoded_spatial_sinkhorn_blur: float,
    decoder_position_weight: float,
    decoder_spatial_flag_weight: float,
    decoder_valid_gate_threshold: float,
):
    n_views = int(batch["views_per_embryo"][0].item())
    genes = stack_view_tensor(batch, "genes", n_views)
    context_role = stack_view_tensor(batch, "context_role", n_views)
    relative_position = stack_view_tensor(batch, "relative_position", n_views)
    token_times = stack_view_tensor(batch, "token_times", n_views)
    valid_mask = stack_view_tensor(batch, "valid_mask", n_views)
    anchor_mask = stack_view_tensor(batch, "anchor_mask", n_views)
    time = stack_view_tensor(batch, "time", n_views)

    n_future_views = int(batch["future_views_per_embryo"][0].item())
    future_genes = stack_view_tensor(batch, "genes", n_future_views, prefix_template="future_view_{i}_")
    future_context_role = stack_view_tensor(
        batch,
        "context_role",
        n_future_views,
        prefix_template="future_view_{i}_",
    )
    future_relative_position = stack_view_tensor(
        batch,
        "relative_position",
        n_future_views,
        prefix_template="future_view_{i}_",
    )
    future_split_fraction = stack_view_tensor(
        batch,
        "split_fraction",
        n_future_views,
        prefix_template="future_view_{i}_",
    )
    future_token_times = stack_view_tensor(batch, "token_times", n_future_views, prefix_template="future_view_{i}_")
    future_valid_mask = stack_view_tensor(batch, "valid_mask", n_future_views, prefix_template="future_view_{i}_")
    future_anchor_mask = stack_view_tensor(batch, "anchor_mask", n_future_views, prefix_template="future_view_{i}_")
    future_time = stack_view_tensor(batch, "time", n_future_views, prefix_template="future_view_{i}_")

    masked_view_mask = build_mask(
        genes.shape[0],
        n_views,
        context_mask_ratio,
        genes.device,
        allow_empty=True,
    )
    masked_future_view_mask = build_mask(
        genes.shape[0],
        n_future_views,
        future_mask_ratio,
        genes.device,
        allow_empty=False,
    )
    future_slots = int(masked_future_view_mask[0].sum().item())
    if future_slots != model.future_slots:
        raise ValueError(
            f"Mask ratio produced {future_slots} future slots, but model expects {model.future_slots}"
        )

    out = model(
        genes=genes,
        time=time,
        token_times=token_times,
        valid_mask=valid_mask,
        anchor_mask=anchor_mask,
        future_genes=future_genes,
        future_time=future_time,
        future_token_times=future_token_times,
        future_valid_mask=future_valid_mask,
        future_anchor_mask=future_anchor_mask,
        masked_future_view_mask=masked_future_view_mask,
        future_split_fraction=future_split_fraction,
        masked_view_mask=masked_view_mask,
        context_role=context_role,
        relative_position=relative_position,
        future_context_role=future_context_role,
        future_relative_position=future_relative_position,
    )

    latent_set_loss = sinkhorn_divergence(
        out.pred_future_set_latents,
        out.target_future_set_latents.detach(),
        blur=sinkhorn_blur,
    )
    gene_set_loss = sinkhorn_divergence(
        out.pred_future_set_genes,
        out.target_future_set_genes.detach(),
        blur=gene_sinkhorn_blur,
    )
    mean_latent_loss = (
        1.0
        - F.cosine_similarity(
            out.pred_future_set_pooled_latent,
            out.target_future_set_pooled_latent.detach(),
            dim=-1,
        )
    ).mean()
    if model.predict_dense_future_tokens:
        masked_future_valid_mask = model.gather_masked_future_view_tensor(
            future_valid_mask,
            masked_future_view_mask,
        )
        token_mask = masked_future_valid_mask.unsqueeze(-1).float()
        local_code_loss = F.mse_loss(
            out.pred_future_local_codes * token_mask,
            out.target_future_local_codes.detach(),
        )
    else:
        local_code_loss = F.mse_loss(
            out.pred_future_local_codes,
            out.target_future_local_codes.detach(),
        )
        masked_future_valid_mask = None
    token_var_loss, token_cov_loss = token_variance_covariance_losses(
        out.pred_future_local_codes,
        masked_future_valid_mask,
    )
    pooled_align_loss = F.mse_loss(
        out.pred_future_set_pooled_latent,
        out.target_future_set_pooled_latent.detach(),
    )
    readout_anchor_loss = pooled_align_loss
    pooled_var_loss, pooled_cov_loss = pooled_variance_covariance_losses(out.pred_future_set_pooled_latent)
    masked_future_genes = model.gather_masked_future_view_tensor(future_genes, masked_future_view_mask)
    if masked_future_valid_mask is None:
        masked_future_valid_mask = model.gather_masked_future_view_tensor(
            future_valid_mask,
            masked_future_view_mask,
        )
    masked_future_relative_position = model.gather_masked_future_view_tensor(
        future_relative_position,
        masked_future_view_mask,
    )
    if (
        decoded_gene_weight > 0.0
        or decoded_spatial_weight > 0.0
        or decoded_mean_gene_weight > 0.0
        or decoded_count_weight > 0.0
    ):
        decoded = model.decode_future_local_codes(out.pred_future_local_codes)
        target_gene_state, target_spatial_state, _, _ = build_target_structured_state(
            genes=masked_future_genes,
            relative_position=masked_future_relative_position,
            valid_mask=masked_future_valid_mask,
            position_weight=decoder_position_weight,
            spatial_flag_weight=decoder_spatial_flag_weight,
        )
        pred_gene_state, pred_spatial_state, _, _ = build_pred_structured_state(
            pred_genes=decoded.pred_cell_genes,
            pred_positions=decoded.pred_cell_positions,
            pred_valid_logits=decoded.pred_cell_valid_logits,
            pred_spatial_logits=decoded.pred_cell_spatial_logits,
            position_weight=decoder_position_weight,
            spatial_flag_weight=decoder_spatial_flag_weight,
            valid_gate_threshold=decoder_valid_gate_threshold,
        )
        decoded_gene_loss = sinkhorn_divergence(
            pred_gene_state.view(pred_gene_state.shape[0], -1, pred_gene_state.shape[-1]),
            target_gene_state.detach().view(
                target_gene_state.shape[0],
                -1,
                target_gene_state.shape[-1],
            ),
            blur=decoded_gene_sinkhorn_blur,
        )
        decoded_spatial_loss = sinkhorn_divergence(
            pred_spatial_state.view(pred_spatial_state.shape[0], -1, pred_spatial_state.shape[-1]),
            target_spatial_state.detach().view(
                target_spatial_state.shape[0],
                -1,
                target_spatial_state.shape[-1],
            ),
            blur=decoded_spatial_sinkhorn_blur,
        )
        target_mean_gene = (
            (masked_future_genes * masked_future_valid_mask.unsqueeze(-1).float()).sum(dim=2)
            / masked_future_valid_mask.sum(dim=2, keepdim=True).clamp_min(1.0)
        )
        target_count_ratio = masked_future_valid_mask.float().sum(dim=2) / float(masked_future_valid_mask.shape[-1])
        decoded_mean_gene_loss = F.mse_loss(decoded.pred_mean_gene, target_mean_gene)
        decoded_count_loss = F.mse_loss(torch.sigmoid(decoded.pred_cell_count), target_count_ratio)
    else:
        zero = local_code_loss.new_zeros(())
        decoded_gene_loss = zero
        decoded_spatial_loss = zero
        decoded_mean_gene_loss = zero
        decoded_count_loss = zero
    explicit_count_loss = F.mse_loss(out.pred_future_mass, out.target_future_mass)
    explicit_split_loss = F.mse_loss(
        torch.sigmoid(out.pred_future_split_logits),
        out.target_future_split_fraction,
    )
    survival_loss = F.binary_cross_entropy_with_logits(
        out.pred_future_survival_logits,
        out.target_future_survival,
    )
    split_count_loss = F.mse_loss(out.pred_future_split_count, out.target_future_split_count)
    total = (
        latent_set_weight * latent_set_loss
        + gene_set_weight * gene_set_loss
        + mean_latent_weight * mean_latent_loss
        + pooled_align_weight * pooled_align_loss
        + pooled_var_weight * pooled_var_loss
        + pooled_cov_weight * pooled_cov_loss
        + token_readout_anchor_weight * readout_anchor_loss
        + local_code_weight * local_code_loss
        + token_var_weight * token_var_loss
        + token_cov_weight * token_cov_loss
        + decoded_gene_weight * decoded_gene_loss
        + decoded_spatial_weight * decoded_spatial_loss
        + decoded_mean_gene_weight * decoded_mean_gene_loss
        + decoded_count_weight * decoded_count_loss
        + explicit_count_weight * explicit_count_loss
        + explicit_split_weight * explicit_split_loss
        + survival_weight * survival_loss
        + split_count_weight * split_count_loss
    )
    return total, {
        "total": total.item(),
        "latent_set": latent_set_loss.item(),
        "gene_set": gene_set_loss.item(),
        "mean_latent": mean_latent_loss.item(),
        "pooled_align": pooled_align_loss.item(),
        "readout_anchor": readout_anchor_loss.item(),
        "pooled_var": pooled_var_loss.item(),
        "pooled_cov": pooled_cov_loss.item(),
        "local_code": local_code_loss.item(),
        "token_var": token_var_loss.item(),
        "token_cov": token_cov_loss.item(),
        "decoded_gene": decoded_gene_loss.item(),
        "decoded_spatial": decoded_spatial_loss.item(),
        "decoded_mean_gene": decoded_mean_gene_loss.item(),
        "decoded_count": decoded_count_loss.item(),
        "explicit_count": explicit_count_loss.item(),
        "explicit_split": explicit_split_loss.item(),
        "survival": survival_loss.item(),
        "split_count": split_count_loss.item(),
    }


def run_epoch(
    model: EmbryoFutureSetModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    context_mask_ratio: float,
    future_mask_ratio: float,
    latent_set_weight: float,
    gene_set_weight: float,
    mean_latent_weight: float,
    pooled_align_weight: float,
    pooled_var_weight: float,
    pooled_cov_weight: float,
    token_readout_anchor_weight: float,
    local_code_weight: float,
    token_var_weight: float,
    token_cov_weight: float,
    decoded_gene_weight: float,
    decoded_spatial_weight: float,
    decoded_mean_gene_weight: float,
    decoded_count_weight: float,
    explicit_count_weight: float,
    explicit_split_weight: float,
    survival_weight: float,
    split_count_weight: float,
    sinkhorn_blur: float,
    gene_sinkhorn_blur: float,
    decoded_gene_sinkhorn_blur: float,
    decoded_spatial_sinkhorn_blur: float,
    decoder_position_weight: float,
    decoder_spatial_flag_weight: float,
    decoder_valid_gate_threshold: float,
):
    train = optimizer is not None
    model.train(train)
    totals: dict[str, float] = {}
    n_batches = 0
    for batch in loader:
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        loss, metrics = compute_metrics(
            model,
            batch,
            context_mask_ratio,
            future_mask_ratio,
            latent_set_weight,
            gene_set_weight,
            mean_latent_weight,
            pooled_align_weight,
            pooled_var_weight,
            pooled_cov_weight,
            token_readout_anchor_weight,
            local_code_weight,
            token_var_weight,
            token_cov_weight,
            decoded_gene_weight,
            decoded_spatial_weight,
            decoded_mean_gene_weight,
            decoded_count_weight,
            explicit_count_weight,
            explicit_split_weight,
            survival_weight,
            split_count_weight,
            sinkhorn_blur,
            gene_sinkhorn_blur,
            decoded_gene_sinkhorn_blur,
            decoded_spatial_sinkhorn_blur,
            decoder_position_weight,
            decoder_spatial_flag_weight,
            decoder_valid_gate_threshold,
        )
        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        n_batches += 1
    return {key: value / max(1, n_batches) for key, value in totals.items()}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if (
        args.pooled_canonicalizer_path is None
        and (args.pooled_align_weight > 0.0 or args.pooled_var_weight > 0.0 or args.pooled_cov_weight > 0.0)
    ):
        raise ValueError("pooled canonicalizer path is required when pooled latent alignment regularization is enabled")
    if args.token_readout_anchor_weight > 0.0 and args.token_readout_anchor_path is None:
        raise ValueError("token readout anchor path is required when token readout anchor loss is enabled")
    train_filters = resolve_event_filters(args)
    val_filters = resolve_event_filters(args, prefix="val_")

    train_ds = EmbryoViewDataset(
        h5ad_path=args.h5ad_path,
        n_hvg=args.n_hvg,
        context_size=args.context_size,
        global_context_size=args.global_context_size,
        dt_minutes=args.dt_minutes,
        time_window_minutes=args.time_window_minutes,
        samples_per_pair=args.samples_per_pair,
        min_cells_per_window=args.min_cells_per_window,
        split="train",
        val_fraction=args.val_fraction,
        random_seed=args.seed,
        sampling_strategy=args.sampling_strategy,
        min_spatial_cells_per_window=args.min_spatial_cells_per_window,
        spatial_neighbor_pool_size=args.spatial_neighbor_pool_size,
        delete_target_mode=args.delete_target_mode,
        views_per_embryo=args.views_per_embryo,
        future_views_per_embryo=args.future_views_per_embryo,
        top_cell_types=args.top_cell_types,
        **train_filters,
    )
    val_ds = EmbryoViewDataset(
        h5ad_path=args.h5ad_path,
        n_hvg=args.n_hvg,
        context_size=args.context_size,
        global_context_size=args.global_context_size,
        dt_minutes=args.dt_minutes,
        time_window_minutes=args.time_window_minutes,
        samples_per_pair=args.val_samples_per_pair or max(1, args.samples_per_pair // 2),
        min_cells_per_window=args.min_cells_per_window,
        split="val",
        val_fraction=args.val_fraction,
        random_seed=args.seed,
        sampling_strategy=args.sampling_strategy,
        min_spatial_cells_per_window=args.min_spatial_cells_per_window,
        spatial_neighbor_pool_size=args.spatial_neighbor_pool_size,
        delete_target_mode=args.delete_target_mode,
        views_per_embryo=args.views_per_embryo,
        future_views_per_embryo=args.future_views_per_embryo,
        top_cell_types=args.top_cell_types,
        **val_filters,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_embryo_view,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_embryo_view,
    )

    backbone, ckpt = load_backbone(args)
    outer_gene_dim = args.n_hvg if ckpt is None else int(ckpt["gene_dim"])
    future_slots = max(
        1,
        min(
            train_ds.future_views_per_embryo - 1,
            int(round(args.future_mask_ratio * train_ds.future_views_per_embryo)),
        ),
    )
    pooled_canonicalizer = load_pooled_canonicalizer(
        args.pooled_canonicalizer_path,
        args.d_model if ckpt is None else int(ckpt["config"]["d_model"]),
    )
    token_readout_anchor = load_token_readout_anchor(
        args.token_readout_anchor_path,
        args.d_model if ckpt is None else int(ckpt["config"]["d_model"]),
    )
    model = EmbryoFutureSetModel(
        backbone=backbone,
        future_slots=future_slots,
        d_model=args.d_model if ckpt is None else int(ckpt["config"]["d_model"]),
        gene_dim=outer_gene_dim,
        n_heads=args.n_heads if ckpt is None else int(ckpt["config"]["n_heads"]),
        decoder_layers=args.decoder_layers,
        head_dim=args.head_dim if ckpt is None else int(ckpt["config"]["head_dim"]),
        use_current_local_tokens=args.use_current_local_tokens,
        learn_current_token_gate=args.learn_current_token_gate,
        current_token_gate_init=args.current_token_gate_init,
        current_conditioning_mode=args.current_conditioning_mode,
        code_tokens=args.code_tokens,
        predict_dense_future_tokens=args.predict_dense_future_tokens,
        strict_token_jepa=args.strict_token_jepa,
        pooled_latent_canonicalizer=pooled_canonicalizer,
        token_readout_anchor=token_readout_anchor,
    ).to(args.device)
    if args.freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad_(False)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            args.device,
            args.context_mask_ratio,
            args.future_mask_ratio,
            args.latent_set_weight,
            args.gene_set_weight,
            args.mean_latent_weight,
            args.pooled_align_weight,
            args.pooled_var_weight,
            args.pooled_cov_weight,
            args.token_readout_anchor_weight,
            args.local_code_weight,
            args.token_var_weight,
            args.token_cov_weight,
            args.decoded_gene_weight,
            args.decoded_spatial_weight,
            args.decoded_mean_gene_weight,
            args.decoded_count_weight,
            args.explicit_count_weight,
            args.explicit_split_weight,
            args.survival_weight,
            args.split_count_weight,
            args.sinkhorn_blur,
            args.gene_sinkhorn_blur,
            args.decoded_gene_sinkhorn_blur,
            args.decoded_spatial_sinkhorn_blur,
            args.decoder_position_weight,
            args.decoder_spatial_flag_weight,
            args.decoder_valid_gate_threshold,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            None,
            args.device,
            args.context_mask_ratio,
            args.future_mask_ratio,
            args.latent_set_weight,
            args.gene_set_weight,
            args.mean_latent_weight,
            args.pooled_align_weight,
            args.pooled_var_weight,
            args.pooled_cov_weight,
            args.token_readout_anchor_weight,
            args.local_code_weight,
            args.token_var_weight,
            args.token_cov_weight,
            args.decoded_gene_weight,
            args.decoded_spatial_weight,
            args.decoded_mean_gene_weight,
            args.decoded_count_weight,
            args.explicit_count_weight,
            args.explicit_split_weight,
            args.survival_weight,
            args.split_count_weight,
            args.sinkhorn_blur,
            args.gene_sinkhorn_blur,
            args.decoded_gene_sinkhorn_blur,
            args.decoded_spatial_sinkhorn_blur,
            args.decoder_position_weight,
            args.decoder_spatial_flag_weight,
            args.decoder_valid_gate_threshold,
        )
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"epoch={epoch} train_total={train_metrics['total']:.4f} "
            f"val_total={val_metrics['total']:.4f} val_latent_set={val_metrics['latent_set']:.4f} "
            f"val_gene_set={val_metrics['gene_set']:.4f} val_mean_latent={val_metrics['mean_latent']:.4f} "
            f"val_pooled_align={val_metrics['pooled_align']:.4f} "
            f"val_readout_anchor={val_metrics['readout_anchor']:.4f} "
            f"val_pooled_var={val_metrics['pooled_var']:.4f} "
            f"val_pooled_cov={val_metrics['pooled_cov']:.4f} "
            f"val_local_code={val_metrics['local_code']:.4f} "
            f"val_token_var={val_metrics['token_var']:.4f} "
            f"val_token_cov={val_metrics['token_cov']:.4f} "
            f"val_decoded_gene={val_metrics['decoded_gene']:.4f} "
            f"val_decoded_spatial={val_metrics['decoded_spatial']:.4f} "
            f"val_split={val_metrics['explicit_split']:.4f} "
            f"val_survival={val_metrics['survival']:.4f}"
        )
        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": vars(args),
                    "gene_dim": args.n_hvg if ckpt is None else ckpt["gene_dim"],
                    "backbone_checkpoint": args.backbone_checkpoint,
                    "future_slots": future_slots,
                },
                checkpoint_dir / "best.pt",
            )

    with open(checkpoint_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
