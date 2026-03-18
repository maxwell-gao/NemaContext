#!/usr/bin/env python3
"""Train a local structured cell-code component on single patch views."""

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

from examples.whole_organism_ar.train_gene_context import (  # noqa: E402
    EVENT_SUBSET_THRESHOLDS,
    resolve_event_filters,
)
from src.branching_flows.emergent_loss import sinkhorn_divergence  # noqa: E402
from src.branching_flows.gene_context import LocalCellCodeModel  # noqa: E402
from src.data.gene_context_dataset import PatchSetDataset, collate_patch_set  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Train a local structured cell-code model.")
    p.add_argument("--model_type", choices=["multi_cell", "single_cell"], default="multi_cell")
    p.add_argument("--h5ad_path", default="dataset/processed/nema_extended_large2025.h5ad")
    p.add_argument("--n_hvg", type=int, default=256)
    p.add_argument("--context_size", type=int, default=256)
    p.add_argument("--global_context_size", type=int, default=32)
    p.add_argument("--dt_minutes", type=float, default=40.0)
    p.add_argument("--time_window_minutes", type=float, default=10.0)
    p.add_argument("--samples_per_pair", type=int, default=16)
    p.add_argument("--val_samples_per_pair", type=int, default=None)
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
    p.add_argument("--code_tokens", type=int, default=8)
    p.add_argument("--cell_set_weight", type=float, default=0.05)
    p.add_argument("--mean_weight", type=float, default=0.2)
    p.add_argument("--latent_weight", type=float, default=0.2)
    p.add_argument("--count_weight", type=float, default=0.1)
    p.add_argument("--valid_weight", type=float, default=0.1)
    p.add_argument("--spatial_valid_weight", type=float, default=0.1)
    p.add_argument("--cell_sinkhorn_blur", type=float, default=0.1)
    p.add_argument("--position_weight", type=float, default=0.25)
    p.add_argument("--spatial_flag_weight", type=float, default=0.5)
    p.add_argument("--valid_gate_threshold", type=float, default=0.5)
    p.add_argument("--checkpoint_dir", default="checkpoints_local_cell_code")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_target_mean_gene(genes: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    denom = valid_mask.sum(dim=1, keepdim=True).clamp_min(1).float()
    return (genes * valid_mask.unsqueeze(-1).float()).sum(dim=1) / denom


def build_target_structured_state(
    genes: torch.Tensor,
    relative_position: torch.Tensor,
    valid_mask: torch.Tensor,
    position_weight: float,
    spatial_flag_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    norm_genes = F.layer_norm(genes, (genes.shape[-1],))
    target_spatial_valid = valid_mask & (relative_position[..., 4] > 0.5)
    target_positions = relative_position[..., :3] * target_spatial_valid.unsqueeze(-1).float()
    target_state = torch.cat(
        [
            norm_genes,
            position_weight * target_positions,
            spatial_flag_weight * target_spatial_valid.unsqueeze(-1).float(),
        ],
        dim=-1,
    )
    target_state = target_state * valid_mask.unsqueeze(-1).float()
    return target_state, target_spatial_valid


def build_pred_structured_state(
    pred_genes: torch.Tensor,
    pred_positions: torch.Tensor,
    pred_valid_logits: torch.Tensor,
    pred_spatial_logits: torch.Tensor,
    position_weight: float,
    spatial_flag_weight: float,
    valid_gate_threshold: float,
) -> torch.Tensor:
    pred_valid_prob = torch.sigmoid(pred_valid_logits)
    if valid_gate_threshold > 0.0:
        pred_valid = F.relu(pred_valid_prob - valid_gate_threshold) / max(1e-6, 1.0 - valid_gate_threshold)
    else:
        pred_valid = pred_valid_prob
    pred_valid = pred_valid.unsqueeze(-1)
    pred_spatial = torch.sigmoid(pred_spatial_logits).unsqueeze(-1)
    norm_pred_genes = F.layer_norm(pred_genes, (pred_genes.shape[-1],))
    pred_state = torch.cat(
        [
            norm_pred_genes,
            position_weight * (pred_positions * pred_spatial),
            spatial_flag_weight * pred_spatial,
        ],
        dim=-1,
    )
    return pred_state * pred_valid


def compute_metrics(
    model: LocalCellCodeModel,
    batch: dict[str, torch.Tensor],
    cell_set_weight: float,
    mean_weight: float,
    latent_weight: float,
    count_weight: float,
    valid_weight: float,
    spatial_valid_weight: float,
    cell_sinkhorn_blur: float,
    position_weight: float,
    spatial_flag_weight: float,
    valid_gate_threshold: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    genes = batch["current_genes"]
    time = batch["current_time"]
    token_times = batch["current_token_times"]
    valid_mask = batch["current_valid_mask"]
    anchor_mask = batch["current_anchor_mask"]
    context_role = batch.get("current_context_role")
    relative_position = batch.get("current_relative_position")

    out = model(
        genes=genes,
        time=time,
        token_times=token_times,
        valid_mask=valid_mask,
        anchor_mask=anchor_mask,
        context_role=context_role,
        relative_position=relative_position,
    )

    target_structured_state, target_spatial_valid = build_target_structured_state(
        genes=genes,
        relative_position=relative_position,
        valid_mask=valid_mask,
        position_weight=position_weight,
        spatial_flag_weight=spatial_flag_weight,
    )
    pred_structured_state = build_pred_structured_state(
        pred_genes=out.pred_cell_genes,
        pred_positions=out.pred_cell_positions,
        pred_valid_logits=out.pred_cell_valid_logits,
        pred_spatial_logits=out.pred_cell_spatial_logits,
        position_weight=position_weight,
        spatial_flag_weight=spatial_flag_weight,
        valid_gate_threshold=valid_gate_threshold,
    )
    target_mean_gene = build_target_mean_gene(genes, valid_mask)
    target_count = valid_mask.sum(dim=1).float()
    target_count_ratio = target_count / float(valid_mask.shape[1])
    pred_count_ratio = torch.sigmoid(out.pred_cell_count)

    cell_set_loss = sinkhorn_divergence(pred_structured_state, target_structured_state, blur=cell_sinkhorn_blur)
    mean_loss = F.mse_loss(out.pred_mean_gene, target_mean_gene)
    latent_loss = (1.0 - F.cosine_similarity(out.pred_patch_latent, out.patch_latent.detach(), dim=-1)).mean()
    count_loss = F.mse_loss(pred_count_ratio, target_count_ratio)
    valid_loss = F.binary_cross_entropy_with_logits(out.pred_cell_valid_logits, valid_mask.float())
    spatial_valid_loss = F.binary_cross_entropy_with_logits(
        out.pred_cell_spatial_logits,
        target_spatial_valid.float(),
    )

    total = (
        cell_set_weight * cell_set_loss
        + mean_weight * mean_loss
        + latent_weight * latent_loss
        + count_weight * count_loss
        + valid_weight * valid_loss
        + spatial_valid_weight * spatial_valid_loss
    )
    metrics = {
        "total": float(total.item()),
        "cell_set": float(cell_set_loss.item()),
        "mean_gene": float(mean_loss.item()),
        "latent": float(latent_loss.item()),
        "count": float(count_loss.item()),
        "valid": float(valid_loss.item()),
        "spatial_valid": float(spatial_valid_loss.item()),
        "count_rmse": float(
            (torch.sqrt(count_loss.clamp_min(0.0)) * valid_mask.shape[1]).item()
        ),
        "pred_count_mean": float((pred_count_ratio.mean() * valid_mask.shape[1]).item()),
        "target_count_mean": float(target_count.mean().item()),
    }
    return total, metrics


def run_epoch(model, loader, optimizer, device, args):
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    n_batches = 0

    for batch in loader:
        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        total, metrics = compute_metrics(
            model,
            batch,
            cell_set_weight=args.cell_set_weight,
            mean_weight=args.mean_weight,
            latent_weight=args.latent_weight,
            count_weight=args.count_weight,
            valid_weight=args.valid_weight,
            spatial_valid_weight=args.spatial_valid_weight,
            cell_sinkhorn_blur=args.cell_sinkhorn_blur,
            position_weight=args.position_weight,
            spatial_flag_weight=args.spatial_flag_weight,
            valid_gate_threshold=args.valid_gate_threshold,
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        n_batches += 1

    return {key: value / max(1, n_batches) for key, value in totals.items()}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    train_filters = resolve_event_filters(args)
    val_filters = resolve_event_filters(args, prefix="val_")

    train_ds = PatchSetDataset(
        h5ad_path=args.h5ad_path,
        n_hvg=args.n_hvg,
        context_size=args.context_size,
        global_context_size=args.global_context_size,
        dt_minutes=args.dt_minutes,
        time_window_minutes=args.time_window_minutes,
        samples_per_pair=args.samples_per_pair,
        min_cells_per_window=args.min_cells_per_window,
        sampling_strategy=args.sampling_strategy,
        min_spatial_cells_per_window=args.min_spatial_cells_per_window,
        spatial_neighbor_pool_size=args.spatial_neighbor_pool_size,
        delete_target_mode=args.delete_target_mode,
        **train_filters,
        split="train",
        val_fraction=args.val_fraction,
        random_seed=args.seed,
    )
    val_ds = PatchSetDataset(
        h5ad_path=args.h5ad_path,
        n_hvg=args.n_hvg,
        context_size=args.context_size,
        global_context_size=args.global_context_size,
        dt_minutes=args.dt_minutes,
        time_window_minutes=args.time_window_minutes,
        samples_per_pair=args.val_samples_per_pair or max(1, args.samples_per_pair // 2),
        min_cells_per_window=args.min_cells_per_window,
        sampling_strategy=args.sampling_strategy,
        min_spatial_cells_per_window=args.min_spatial_cells_per_window,
        spatial_neighbor_pool_size=args.spatial_neighbor_pool_size,
        delete_target_mode=args.delete_target_mode,
        **val_filters,
        split="val",
        val_fraction=args.val_fraction,
        random_seed=args.seed + 1000,
    )
    if not train_ds.time_pairs:
        raise ValueError("Training dataset is empty after filtering.")
    if not val_ds.time_pairs:
        raise ValueError("Validation dataset is empty after filtering.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_patch_set)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_patch_set)

    model = LocalCellCodeModel(
        gene_dim=train_ds.gene_dim,
        context_size=args.context_size,
        model_type=args.model_type,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        head_dim=args.head_dim,
        use_pairwise_spatial_bias=args.pairwise_spatial_bias,
        code_tokens=args.code_tokens,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, args.device, args)
        val_metrics = run_epoch(model, val_loader, None, args.device, args)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"epoch={epoch} train_total={train_metrics['total']:.4f} "
            f"val_total={val_metrics['total']:.4f} "
            f"val_cell_set={val_metrics['cell_set']:.4f} "
            f"val_count_rmse={val_metrics['count_rmse']:.4f}"
        )
        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": vars(args),
                    "gene_dim": train_ds.gene_dim,
                    "best_val": best_val,
                    "model_type": args.model_type,
                },
                checkpoint_dir / "best.pt",
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": vars(args),
            "gene_dim": train_ds.gene_dim,
            "best_val": best_val,
            "model_type": args.model_type,
        },
        checkpoint_dir / "final.pt",
    )
    with open(checkpoint_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
