#!/usr/bin/env python3
"""Train a multi-cell gene-context baseline on real transcriptome data."""

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

from src.branching_flows.gene_context import GeneContextModel  # noqa: E402
from src.data.gene_context_dataset import (  # noqa: E402
    GeneContextDataset,
    collate_gene_context,
)


def compute_metrics(output, batch, split_weight: float, del_weight: float):
    supervision_mask = batch.get("anchor_mask", batch["valid_mask"]) & batch["valid_mask"]
    match_mask = batch["match_mask"] & supervision_mask
    if match_mask.any():
        current = batch["genes"][match_mask]
        target = batch["target_genes"][match_mask]
        pred = current + output.gene_delta[match_mask]
        gene_loss = F.mse_loss(pred, target)
    else:
        gene_loss = torch.tensor(0.0, device=batch["genes"].device)

    split_mask = supervision_mask
    split_targets = batch["split_target"][split_mask]
    del_targets = batch["del_target"][split_mask]
    split_logits = output.split_logits[split_mask]
    del_logits = output.del_logits[split_mask]

    split_pos = float(split_targets.sum().item())
    split_neg = max(1.0, float(split_targets.numel() - split_pos))
    split_pos_weight = torch.tensor(
        [split_neg / max(1.0, split_pos)],
        device=split_logits.device,
    )
    split_loss = F.binary_cross_entropy_with_logits(
        split_logits,
        split_targets,
        pos_weight=split_pos_weight,
    )
    del_loss = F.binary_cross_entropy_with_logits(del_logits, del_targets)

    total = gene_loss + split_weight * split_loss + del_weight * del_loss
    return total, {
        "total": total.item(),
        "gene": gene_loss.item(),
        "split": split_loss.item(),
        "del": del_loss.item(),
        "match_rate": (
            (match_mask.float().sum() / supervision_mask.float().sum()).item()
            if supervision_mask.any()
            else 0.0
        ),
        "split_rate": split_targets.mean().item() if split_targets.numel() else 0.0,
        "del_rate": del_targets.mean().item() if del_targets.numel() else 0.0,
    }


def run_epoch(model, loader, optimizer, device, split_weight: float, del_weight: float):
    train = optimizer is not None
    model.train(train)
    totals: dict[str, float] = {}
    n_batches = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(
            genes=batch["genes"],
            time=batch["time"],
            future_time=batch["future_time"],
            token_times=batch["token_times"],
            valid_mask=batch["valid_mask"],
            context_role=batch.get("context_role"),
            anchor_distance_bucket=batch.get("anchor_distance_bucket"),
        )
        loss, metrics = compute_metrics(output, batch, split_weight, del_weight)
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
    p = argparse.ArgumentParser(description="Train the active multi-cell gene-context baseline.")
    p.add_argument(
        "--h5ad_path",
        default="dataset/processed/nema_extended_large2025.h5ad",
    )
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
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--head_dim", type=int, default=32)
    p.add_argument("--split_weight", type=float, default=1.0)
    p.add_argument("--del_weight", type=float, default=1.0)
    p.add_argument("--checkpoint_dir", default="checkpoints_gene_context")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    train_ds = GeneContextDataset(
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
        split="train",
        val_fraction=args.val_fraction,
        random_seed=args.seed,
    )
    val_ds = GeneContextDataset(
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
        split="val",
        val_fraction=args.val_fraction,
        random_seed=args.seed + 1000,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_gene_context,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_gene_context,
    )

    model = GeneContextModel(
        gene_dim=train_ds.gene_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        head_dim=args.head_dim,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

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
            args.split_weight,
            args.del_weight,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            None,
            args.device,
            args.split_weight,
            args.del_weight,
        )
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
        )
        print(
            f"epoch={epoch} "
            f"train_total={train_metrics['total']:.4f} "
            f"val_total={val_metrics['total']:.4f} "
            f"val_gene={val_metrics['gene']:.4f}"
        )
        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": vars(args),
                    "gene_dim": train_ds.gene_dim,
                    "best_val": best_val,
                },
                checkpoint_dir / "best.pt",
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": vars(args),
            "gene_dim": train_ds.gene_dim,
            "best_val": best_val,
        },
        checkpoint_dir / "final.pt",
    )
    with open(checkpoint_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
