#!/usr/bin/env python3
"""Train a single-cell gene+time baseline on the active gene-context task."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from examples.whole_organism_ar.train_gene_context import run_epoch  # noqa: E402
from src.branching_flows.gene_context import SingleCellGeneTimeModel  # noqa: E402
from src.data.gene_context_dataset import (  # noqa: E402
    GeneContextDataset,
    collate_gene_context,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Train the single-cell gene+time control baseline."
    )
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
        default="random_window",
    )
    p.add_argument("--min_spatial_cells_per_window", type=int, default=8)
    p.add_argument("--spatial_neighbor_pool_size", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--split_weight", type=float, default=1.0)
    p.add_argument("--del_weight", type=float, default=1.0)
    p.add_argument("--checkpoint_dir", default="checkpoints_gene_single_cell")
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

    model = SingleCellGeneTimeModel(
        gene_dim=train_ds.gene_dim,
        d_model=args.d_model,
        n_layers=args.n_layers,
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
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
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
                    "model_type": "single_cell",
                },
                checkpoint_dir / "best.pt",
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": vars(args),
            "gene_dim": train_ds.gene_dim,
            "best_val": best_val,
            "model_type": "single_cell",
        },
        checkpoint_dir / "final.pt",
    )
    with open(checkpoint_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
