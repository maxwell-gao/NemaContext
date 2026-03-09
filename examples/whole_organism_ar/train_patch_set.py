#!/usr/bin/env python3
"""Train a patch-to-patch set prediction baseline."""

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
from src.branching_flows.gene_context import (  # noqa: E402
    MultiCellPatchSetModel,
    SingleCellPatchSetModel,
)
from src.data.gene_context_dataset import PatchSetDataset, collate_patch_set  # noqa: E402


def compute_patch_set_metrics(model, batch, latent_weight: float, size_weight: float, mean_weight: float, spatial_input_mode: str):
    relative_key = (
        batch.get("current_relative_position")
        if spatial_input_mode == "relative_position"
        else None
    )
    output = model(
        genes=batch["current_genes"],
        time=batch["current_time"],
        future_time=batch["future_time"],
        token_times=batch["current_token_times"],
        valid_mask=batch["current_valid_mask"],
        anchor_mask=batch["current_anchor_mask"],
        context_role=batch.get("current_context_role"),
        relative_position=relative_key,
    )

    future_relative = (
        batch.get("future_relative_position")
        if spatial_input_mode == "relative_position"
        else None
    )
    with torch.no_grad():
        target_latent, _ = model.encode_patch(
            genes=batch["future_genes"],
            time=batch["future_time"],
            future_time=batch["future_time"],
            token_times=batch["future_token_times"],
            valid_mask=batch["future_valid_mask"],
            anchor_mask=batch["future_anchor_mask"],
            context_role=batch.get("future_context_role"),
            relative_position=future_relative,
        )

    ot_loss = sinkhorn_divergence(
        output.pred_future_genes,
        batch["future_genes"],
        blur=0.1,
        p=2,
    )
    size_loss = F.mse_loss(output.pred_patch_size, batch["future_patch_size"])
    mean_loss = F.mse_loss(output.pred_mean_gene, batch["future_mean_gene"])
    latent_loss = (1.0 - F.cosine_similarity(output.patch_latent, target_latent.detach(), dim=-1)).mean()
    total = ot_loss + latent_weight * latent_loss + size_weight * size_loss + mean_weight * mean_loss

    return total, {
        "total": total.item(),
        "ot": ot_loss.item(),
        "latent": latent_loss.item(),
        "size": size_loss.item(),
        "mean_gene": mean_loss.item(),
        "future_patch_size": batch["future_patch_size"].mean().item(),
    }


def run_epoch(model, loader, optimizer, device, latent_weight: float, size_weight: float, mean_weight: float, spatial_input_mode: str):
    train = optimizer is not None
    model.train(train)
    totals: dict[str, float] = {}
    n_batches = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss, metrics = compute_patch_set_metrics(
            model,
            batch,
            latent_weight,
            size_weight,
            mean_weight,
            spatial_input_mode,
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


def parse_args():
    p = argparse.ArgumentParser(description="Train the patch-to-patch set prediction baseline.")
    p.add_argument("--model_type", choices=["multi_cell", "single_cell"], default="multi_cell")
    p.add_argument("--h5ad_path", default="dataset/processed/nema_extended_large2025.h5ad")
    p.add_argument("--n_hvg", type=int, default=256)
    p.add_argument("--context_size", type=int, default=64)
    p.add_argument("--global_context_size", type=int, default=None)
    p.add_argument("--dt_minutes", type=float, default=20.0)
    p.add_argument("--time_window_minutes", type=float, default=10.0)
    p.add_argument("--samples_per_pair", type=int, default=4)
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
    p.add_argument(
        "--event_subset",
        choices=sorted(EVENT_SUBSET_THRESHOLDS),
        default="none",
    )
    p.add_argument("--min_event_positive", type=int, default=0)
    p.add_argument("--min_anchor_event_positive", type=int, default=0)
    p.add_argument("--min_split_positive", type=int, default=0)
    p.add_argument("--min_del_positive", type=int, default=0)
    p.add_argument("--min_anchor_split_positive", type=int, default=0)
    p.add_argument("--min_anchor_del_positive", type=int, default=0)
    p.add_argument(
        "--delete_target_mode",
        choices=["weak", "strict"],
        default="strict",
    )
    p.add_argument(
        "--val_event_subset",
        choices=sorted(EVENT_SUBSET_THRESHOLDS),
        default="none",
    )
    p.add_argument("--val_min_event_positive", type=int, default=None)
    p.add_argument("--val_min_anchor_event_positive", type=int, default=None)
    p.add_argument("--val_min_split_positive", type=int, default=None)
    p.add_argument("--val_min_del_positive", type=int, default=None)
    p.add_argument("--val_min_anchor_split_positive", type=int, default=None)
    p.add_argument("--val_min_anchor_del_positive", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--head_dim", type=int, default=32)
    p.add_argument("--pairwise_spatial_bias", action="store_true")
    p.add_argument("--latent_weight", type=float, default=0.2)
    p.add_argument("--size_weight", type=float, default=0.2)
    p.add_argument("--mean_weight", type=float, default=0.5)
    p.add_argument(
        "--spatial_input_mode",
        choices=["none", "relative_position"],
        default="relative_position",
    )
    p.add_argument("--checkpoint_dir", default="checkpoints_patch_set")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


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

    if args.model_type == "single_cell":
        model = SingleCellPatchSetModel(
            gene_dim=train_ds.gene_dim,
            context_size=args.context_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
        ).to(args.device)
    else:
        model = MultiCellPatchSetModel(
            gene_dim=train_ds.gene_dim,
            context_size=args.context_size,
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
        train_metrics = run_epoch(
            model, train_loader, optimizer, args.device, args.latent_weight, args.size_weight, args.mean_weight, args.spatial_input_mode
        )
        val_metrics = run_epoch(
            model, val_loader, None, args.device, args.latent_weight, args.size_weight, args.mean_weight, args.spatial_input_mode
        )
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"epoch={epoch} train_total={train_metrics['total']:.4f} "
            f"val_total={val_metrics['total']:.4f} val_ot={val_metrics['ot']:.4f}"
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
