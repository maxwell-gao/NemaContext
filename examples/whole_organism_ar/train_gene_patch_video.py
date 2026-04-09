#!/usr/bin/env python3
"""Train an end-to-end next-frame GenePatchVideoModel."""

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
from src.branching_flows.gene_context import GenePatchVideoModel  # noqa: E402
from src.data.gene_context_dataset import (  # noqa: E402
    TemporalPatchSetDataset,
    collate_history_patch_set,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5ad_path", default="dataset/processed/nema_extended_large2025.h5ad")
    p.add_argument("--n_hvg", type=int, default=256)
    p.add_argument("--context_size", type=int, default=24)
    p.add_argument("--history_patches", type=int, default=4)
    p.add_argument("--dt_minutes", type=float, default=40.0)
    p.add_argument("--time_window_minutes", type=float, default=10.0)
    p.add_argument("--samples_per_pair", type=int, default=16)
    p.add_argument("--min_cells_per_window", type=int, default=32)
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--sampling_strategy", choices=["spatial_anchor"], default="spatial_anchor")
    p.add_argument("--min_spatial_cells_per_window", type=int, default=8)
    p.add_argument("--spatial_neighbor_pool_size", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_spatial_layers", type=int, default=2)
    p.add_argument("--n_temporal_layers", type=int, default=4)
    p.add_argument("--n_decoder_layers", type=int, default=2)
    p.add_argument("--head_dim", type=int, default=32)
    p.add_argument("--gene_set_weight", type=float, default=1.0)
    p.add_argument("--mean_gene_weight", type=float, default=0.2)
    p.add_argument("--mask_ratio", type=float, default=0.3)
    p.add_argument("--masked_gene_weight", type=float, default=0.2)
    p.add_argument("--gene_sinkhorn_blur", type=float, default=0.1)
    p.add_argument("--checkpoint_dir", default="checkpoints/gene_patch_video")
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
    return TemporalPatchSetDataset(
        h5ad_path=args.h5ad_path,
        n_hvg=args.n_hvg,
        context_size=args.context_size,
        global_context_size=0,
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
        history_patches=args.history_patches,
        patch_composition="local_only",
    )


def _stack_history_tensor(batch: dict[str, torch.Tensor], field: str, n_patches: int) -> torch.Tensor:
    tensors = [batch[f"history_patch_{i}_{field}"] for i in range(n_patches)]
    max_len = max(t.shape[1] for t in tensors)
    padded = []
    for tensor in tensors:
        pad_len = max_len - tensor.shape[1]
        if pad_len == 0:
            padded.append(tensor)
            continue
        if tensor.dim() == 3:
            pad = tensor.new_zeros(tensor.shape[0], pad_len, tensor.shape[2])
        else:
            pad = tensor.new_zeros(tensor.shape[0], pad_len)
        padded.append(torch.cat([tensor, pad], dim=1))
    return torch.stack(padded, dim=1)


def prepare_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    n_patches = int(batch["history_patches"][0].item())
    batch["history_genes"] = _stack_history_tensor(batch, "genes", n_patches)
    batch["history_context_role"] = _stack_history_tensor(batch, "context_role", n_patches)
    batch["history_relative_position"] = _stack_history_tensor(batch, "relative_position", n_patches)
    batch["history_token_times"] = _stack_history_tensor(batch, "token_times", n_patches)
    batch["history_valid_mask"] = _stack_history_tensor(batch, "valid_mask", n_patches)
    batch["history_time"] = batch[f"history_patch_{n_patches - 1}_time"]
    return batch


def compute_current_mean_gene(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    current_genes = batch[f"history_patch_{int(batch['history_patches'][0].item()) - 1}_genes"]
    current_valid = batch[f"history_patch_{int(batch['history_patches'][0].item()) - 1}_valid_mask"].unsqueeze(-1).float()
    denom = current_valid.sum(dim=1).clamp_min(1.0)
    return (current_genes * current_valid).sum(dim=1) / denom


def sample_history_mask(valid_mask: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    if mask_ratio <= 0.0:
        return torch.zeros_like(valid_mask, dtype=torch.bool)
    if mask_ratio >= 1.0:
        return valid_mask.bool()
    rand = torch.rand_like(valid_mask.float())
    return (rand < mask_ratio) & valid_mask.bool()


def apply_history_mask(history_genes: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
    masked_genes = history_genes.clone()
    masked_genes[history_mask] = 0.0
    return masked_genes


def masked_gene_recon_loss(pred_history_genes: torch.Tensor, history_genes: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
    if not history_mask.any():
        return pred_history_genes.new_zeros(())
    return F.mse_loss(pred_history_genes[history_mask], history_genes[history_mask])


def compute_loss(model: GenePatchVideoModel, batch: dict[str, torch.Tensor], args, apply_mask: bool):
    history_mask = sample_history_mask(batch["history_valid_mask"], args.mask_ratio) if apply_mask else torch.zeros_like(batch["history_valid_mask"], dtype=torch.bool)
    input_history_genes = apply_history_mask(batch["history_genes"], history_mask) if apply_mask else batch["history_genes"]
    out = model(
        genes=input_history_genes,
        time=batch["history_time"],
        future_time=batch["future_time"],
        token_times=batch["history_token_times"],
        valid_mask=batch["history_valid_mask"],
        context_role=None,
        relative_position=None,
    )
    target_future = batch["future_genes"][:, : out.pred_future_genes.shape[1]]
    target_mean_gene = batch["future_mean_gene"]
    gene_set = sinkhorn_divergence(out.pred_future_genes, target_future, blur=args.gene_sinkhorn_blur)
    mean_gene = F.mse_loss(out.pred_mean_gene, target_mean_gene)
    masked_gene = masked_gene_recon_loss(out.pred_history_genes, batch["history_genes"], history_mask)
    persistence_mean_gene = F.mse_loss(compute_current_mean_gene(batch), target_mean_gene)
    total = args.gene_set_weight * gene_set + args.mean_gene_weight * mean_gene + args.masked_gene_weight * masked_gene
    return total, {
        "total": float(total.item()),
        "gene_set": float(gene_set.item()),
        "mean_gene": float(mean_gene.item()),
        "masked_gene": float(masked_gene.item()),
        "persistence_mean_gene": float(persistence_mean_gene.item()),
    }


def run_epoch(model, loader, args, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = {"total": 0.0, "gene_set": 0.0, "mean_gene": 0.0, "masked_gene": 0.0, "persistence_mean_gene": 0.0}
    steps = 0
    for batch in loader:
        batch = prepare_batch(batch, args.device)
        loss, metrics = compute_loss(model, batch, args, apply_mask=training)
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
    if len(val_ds) == 0:
        val_ds = build_dataset(args, split="all")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_history_patch_set)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_history_patch_set)
    model = GenePatchVideoModel(
        gene_dim=train_ds.gene_dim,
        context_size=args.context_size,
        history_frames=args.history_patches,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_spatial_layers=args.n_spatial_layers,
        n_temporal_layers=args.n_temporal_layers,
        n_decoder_layers=args.n_decoder_layers,
        head_dim=args.head_dim,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    checkpoint_dir = resolve_checkpoint_dir(args)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_val = float("inf")
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
            torch.save({"model_state_dict": model.state_dict(), "config": vars(args), "gene_dim": train_ds.gene_dim, "best_val_total": best_val}, checkpoint_dir / "best.pt")
        print(f"epoch={epoch} train_total={train_metrics['total']:.4f} val_total={val_metrics['total']:.4f} val_gene_set={val_metrics['gene_set']:.4f} val_mean_gene={val_metrics['mean_gene']:.4f} train_masked_gene={train_metrics['masked_gene']:.4f} val_masked_gene={val_metrics['masked_gene']:.4f} val_persist_mean={val_metrics['persistence_mean_gene']:.4f}", flush=True)


if __name__ == "__main__":
    main()
