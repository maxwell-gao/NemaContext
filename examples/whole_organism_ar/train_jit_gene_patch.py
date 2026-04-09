#!/usr/bin/env python3
"""Train the mainline gene-patch predictor on local spatial cell groups."""

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

from src.branching_flows.emergent_loss import sinkhorn_divergence  # noqa: E402
from src.branching_flows.gene_context import JiTGenePatchModel  # noqa: E402
from src.data.gene_context_dataset import (  # noqa: E402
    PatchSetDataset,
    TemporalPatchSetDataset,
    collate_history_patch_set,
    collate_patch_set,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5ad_path", default="dataset/processed/nema_extended_large2025.h5ad")
    p.add_argument("--n_hvg", type=int, default=256)
    p.add_argument("--context_size", type=int, default=32)
    p.add_argument("--global_context_size", type=int, default=0)
    p.add_argument("--patch_composition", choices=["local_only", "local_global"], default="local_only")
    p.add_argument("--history_patches", type=int, default=1)
    p.add_argument("--dt_minutes", type=float, default=40.0)
    p.add_argument("--time_window_minutes", type=float, default=10.0)
    p.add_argument("--samples_per_pair", type=int, default=16)
    p.add_argument("--min_cells_per_window", type=int, default=32)
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--sampling_strategy", choices=["random_window", "spatial_neighbors", "spatial_anchor"], default="spatial_anchor")
    p.add_argument("--min_spatial_cells_per_window", type=int, default=8)
    p.add_argument("--spatial_neighbor_pool_size", type=int, default=None)
    p.add_argument("--use_context_role", action="store_true", default=True)
    p.add_argument("--no_use_context_role", dest="use_context_role", action="store_false")
    p.add_argument("--use_relative_position", action="store_true", default=True)
    p.add_argument("--no_use_relative_position", dest="use_relative_position", action="store_false")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_layers", type=int, default=6)
    p.add_argument("--head_dim", type=int, default=32)
    p.add_argument("--gene_set_weight", type=float, default=1.0)
    p.add_argument("--mean_gene_weight", type=float, default=0.2)
    p.add_argument("--gene_sinkhorn_blur", type=float, default=0.1)
    p.add_argument("--checkpoint_dir", default="checkpoints/jit_gene_patch")
    p.add_argument("--experiment_name", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def resolve_checkpoint_dir(args) -> Path:
    checkpoint_dir = Path(args.checkpoint_dir)
    if args.experiment_name:
        checkpoint_dir = checkpoint_dir / args.experiment_name
    return checkpoint_dir


def build_dataset(args, split: str):
    common = dict(
        h5ad_path=args.h5ad_path,
        n_hvg=args.n_hvg,
        context_size=args.context_size,
        global_context_size=args.global_context_size,
        dt_minutes=args.dt_minutes,
        time_window_minutes=args.time_window_minutes,
        samples_per_pair=args.samples_per_pair,
        min_cells_per_window=args.min_cells_per_window,
        split=split,
        val_fraction=args.val_fraction,
        random_seed=args.seed,
        sampling_strategy=args.sampling_strategy,
        min_spatial_cells_per_window=args.min_spatial_cells_per_window,
        spatial_neighbor_pool_size=args.spatial_neighbor_pool_size,
        delete_target_mode="strict",
        patch_composition=args.patch_composition,
    )
    if args.history_patches > 1:
        return TemporalPatchSetDataset(**common, history_patches=args.history_patches)
    return PatchSetDataset(**common)


def _stack_history_tensor(batch: dict[str, torch.Tensor], field: str, n_patches: int) -> torch.Tensor:
    return torch.stack([batch[f"history_patch_{i}_{field}"] for i in range(n_patches)], dim=1)


def prepare_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    if "history_patches" not in batch:
        return batch

    n_patches = int(batch["history_patches"][0].item())
    current_genes = _stack_history_tensor(batch, "genes", n_patches)
    current_context_role = _stack_history_tensor(batch, "context_role", n_patches)
    current_relative_position = _stack_history_tensor(batch, "relative_position", n_patches)
    current_token_times = _stack_history_tensor(batch, "token_times", n_patches)
    current_valid_mask = _stack_history_tensor(batch, "valid_mask", n_patches)
    current_time = batch[f"history_patch_{n_patches - 1}_time"]

    batch["current_genes"] = current_genes.flatten(1, 2)
    batch["current_context_role"] = current_context_role.flatten(1, 2)
    batch["current_relative_position"] = current_relative_position.flatten(1, 2)
    batch["current_token_times"] = current_token_times.flatten(1, 2)
    batch["current_valid_mask"] = current_valid_mask.flatten(1, 2)
    batch["current_time"] = current_time
    return batch


def build_model_inputs(batch: dict[str, torch.Tensor], args) -> dict[str, torch.Tensor | None]:
    return {
        "genes": batch["current_genes"],
        "time": batch["current_time"],
        "future_time": batch["future_time"],
        "token_times": batch["current_token_times"],
        "valid_mask": batch["current_valid_mask"],
        "context_role": batch["current_context_role"] if args.use_context_role else None,
        "relative_position": batch["current_relative_position"] if args.use_relative_position else None,
    }


def compute_current_mean_gene(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    valid = batch["current_valid_mask"].unsqueeze(-1).float()
    denom = valid.sum(dim=1).clamp_min(1.0)
    return (batch["current_genes"] * valid).sum(dim=1) / denom


def compute_loss(model: JiTGenePatchModel, batch: dict[str, torch.Tensor], args) -> tuple[torch.Tensor, dict[str, float]]:
    out = model(**build_model_inputs(batch, args))
    gene_set = sinkhorn_divergence(out.pred_future_genes, batch["future_genes"], blur=args.gene_sinkhorn_blur)
    mean_gene = F.mse_loss(out.pred_mean_gene, batch["future_mean_gene"])
    persistence_mean_gene = F.mse_loss(compute_current_mean_gene(batch), batch["future_mean_gene"])
    total = args.gene_set_weight * gene_set + args.mean_gene_weight * mean_gene
    return total, {
        "total": float(total.item()),
        "gene_set": float(gene_set.item()),
        "mean_gene": float(mean_gene.item()),
        "persistence_mean_gene": float(persistence_mean_gene.item()),
    }


def run_epoch(model, loader, args, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = {"total": 0.0, "gene_set": 0.0, "mean_gene": 0.0, "persistence_mean_gene": 0.0}
    steps = 0
    for batch in loader:
        batch = prepare_batch(batch, args.device)
        loss, metrics = compute_loss(model, batch, args)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        for key in totals:
            totals[key] += metrics[key]
        steps += 1
    return {key: value / max(1, steps) for key, value in totals.items()}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    train_ds = build_dataset(args, split="train")
    val_ds = build_dataset(args, split="val")
    val_split = "val"
    if len(val_ds) == 0:
        val_ds = build_dataset(args, split="all")
        val_split = "all"
    collate_fn = collate_history_patch_set if args.history_patches > 1 else collate_patch_set
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = JiTGenePatchModel(
        gene_dim=train_ds.gene_dim,
        context_size=args.context_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        head_dim=args.head_dim,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    checkpoint_dir = resolve_checkpoint_dir(args)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_val = float("inf")
    best_path = checkpoint_dir / "best.pt"

    print(
        json.dumps(
            {
                "task": "gene_patch_mainline",
                "patch_composition": args.patch_composition,
                "use_relative_position": args.use_relative_position,
                "use_context_role": args.use_context_role,
                "history_patches": args.history_patches,
                "checkpoint_dir": str(checkpoint_dir),
            },
            indent=2,
        ),
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, args, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, args, optimizer=None)
        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)
        with open(checkpoint_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": vars(args),
                    "gene_dim": train_ds.gene_dim,
                    "val_split_used": val_split,
                    "best_val_total": best_val,
                },
                best_path,
            )
        print(
            f"epoch={epoch} train_total={train_metrics['total']:.4f} "
            f"val_total={val_metrics['total']:.4f} "
            f"val_gene_set={val_metrics['gene_set']:.4f} "
            f"val_mean_gene={val_metrics['mean_gene']:.4f} "
            f"val_persist_mean={val_metrics['persistence_mean_gene']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
