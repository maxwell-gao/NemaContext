#!/usr/bin/env python3
"""Train embryo-scale one-step developmental probe predictors from local views."""

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
from src.branching_flows.gene_context import EmbryoStateModel  # noqa: E402
from src.data.gene_context_dataset import (  # noqa: E402
    EmbryoViewDataset,
    collate_embryo_view,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train embryo-state one-step developmental probe predictors.")
    p.add_argument("--model_type", choices=["multi_cell", "single_cell"], default="multi_cell")
    p.add_argument("--h5ad_path", default="dataset/processed/nema_extended_large2025.h5ad")
    p.add_argument("--n_hvg", type=int, default=256)
    p.add_argument("--context_size", type=int, default=256)
    p.add_argument("--global_context_size", type=int, default=32)
    p.add_argument("--dt_minutes", type=float, default=40.0)
    p.add_argument("--time_window_minutes", type=float, default=10.0)
    p.add_argument("--samples_per_pair", type=int, default=16)
    p.add_argument("--val_samples_per_pair", type=int, default=None)
    p.add_argument("--views_per_embryo", type=int, default=8)
    p.add_argument("--top_cell_types", type=int, default=8)
    p.add_argument("--min_cells_per_window", type=int, default=32)
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--sampling_strategy", choices=["random_window", "spatial_neighbors", "spatial_anchor"], default="spatial_anchor")
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
    p.add_argument("--checkpoint_dir", default="checkpoints_embryo_state")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def stack_view_tensor(batch: dict[str, torch.Tensor], base: str, n_views: int) -> torch.Tensor:
    return torch.stack([batch[f"view_{i}_{base}"] for i in range(n_views)], dim=1)


def r2_score_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - y_true.mean(dim=0, keepdim=True)) ** 2)
    if float(ss_tot.item()) <= 1e-8:
        return 0.0
    return float((1.0 - ss_res / ss_tot).item())


def compute_metrics(model: EmbryoStateModel, batch: dict[str, torch.Tensor]):
    n_views = int(batch["views_per_embryo"][0].item())
    genes = stack_view_tensor(batch, "genes", n_views)
    context_role = stack_view_tensor(batch, "context_role", n_views)
    relative_position = stack_view_tensor(batch, "relative_position", n_views)
    token_times = stack_view_tensor(batch, "token_times", n_views)
    valid_mask = stack_view_tensor(batch, "valid_mask", n_views)
    anchor_mask = stack_view_tensor(batch, "anchor_mask", n_views)
    time = stack_view_tensor(batch, "time", n_views)

    out = model(
        genes=genes,
        time=time,
        token_times=token_times,
        valid_mask=valid_mask,
        anchor_mask=anchor_mask,
        context_role=context_role,
        relative_position=relative_position,
    )

    founder_loss = F.mse_loss(out.future_founder_composition, batch["future_founder_composition"])
    celltype_loss = F.mse_loss(out.future_celltype_composition, batch["future_celltype_composition"])
    depth_loss = F.mse_loss(out.future_lineage_depth_stats, batch["future_lineage_depth_stats"])
    spatial_loss = F.mse_loss(out.future_spatial_extent, batch["future_spatial_extent"])
    split_loss = F.mse_loss(out.future_split_fraction, batch["future_split_fraction"])
    total = founder_loss + celltype_loss + depth_loss + spatial_loss + split_loss

    metrics = {
        "total": total.item(),
        "founder": founder_loss.item(),
        "celltype": celltype_loss.item(),
        "depth": depth_loss.item(),
        "spatial": spatial_loss.item(),
        "split": split_loss.item(),
        "founder_r2": r2_score_torch(batch["future_founder_composition"], out.future_founder_composition),
        "celltype_r2": r2_score_torch(batch["future_celltype_composition"], out.future_celltype_composition),
        "depth_r2": r2_score_torch(batch["future_lineage_depth_stats"], out.future_lineage_depth_stats),
        "spatial_r2": r2_score_torch(batch["future_spatial_extent"], out.future_spatial_extent),
        "split_r2": r2_score_torch(batch["future_split_fraction"], out.future_split_fraction),
    }
    return total, metrics


def run_epoch(model, loader, optimizer, device: str):
    train = optimizer is not None
    model.train(train)
    totals: dict[str, float] = {}
    n_batches = 0
    for batch in loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        loss, metrics = compute_metrics(model, batch)
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
        random_seed=args.seed + 1000,
        sampling_strategy=args.sampling_strategy,
        min_spatial_cells_per_window=args.min_spatial_cells_per_window,
        spatial_neighbor_pool_size=args.spatial_neighbor_pool_size,
        delete_target_mode=args.delete_target_mode,
        views_per_embryo=args.views_per_embryo,
        top_cell_types=args.top_cell_types,
        **val_filters,
    )
    if not train_ds.time_pairs or not val_ds.time_pairs:
        raise ValueError("Embryo-view dataset is empty after filtering.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_embryo_view)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_embryo_view)

    model = EmbryoStateModel(
        gene_dim=train_ds.gene_dim,
        context_size=args.context_size,
        celltype_dim=len(train_ds._top_cell_type_vocab),
        model_type=args.model_type,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        head_dim=args.head_dim,
        use_pairwise_spatial_bias=args.pairwise_spatial_bias,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, args.device)
        val_metrics = run_epoch(model, val_loader, None, args.device)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"epoch={epoch} train_total={train_metrics['total']:.4f} "
            f"val_total={val_metrics['total']:.4f} val_founder_r2={val_metrics['founder_r2']:.4f} "
            f"val_celltype_r2={val_metrics['celltype_r2']:.4f} val_depth_r2={val_metrics['depth_r2']:.4f} "
            f"val_spatial_r2={val_metrics['spatial_r2']:.4f} val_split_r2={val_metrics['split_r2']:.4f}"
        )
        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": vars(args),
                    "gene_dim": train_ds.gene_dim,
                    "celltype_dim": len(train_ds._top_cell_type_vocab),
                    "top_cell_types": list(train_ds._top_cell_type_vocab),
                    "best_val": best_val,
                    "best_val_metrics": val_metrics,
                },
                checkpoint_dir / "best.pt",
            )

    with open(checkpoint_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
