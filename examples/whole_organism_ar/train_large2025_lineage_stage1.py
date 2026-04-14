#!/usr/bin/env python3
"""Stage 1 lineage-first whole-embryo dynamics training on raw Large2025."""

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
from src.branching_flows.gene_context import LineageWholeEmbryoModel  # noqa: E402
from src.data.gene_context_dataset import (  # noqa: E402
    Large2025WholeEmbryoDataset,
    collate_large2025_whole_embryo,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", default="dataset/raw")
    p.add_argument("--n_hvg", type=int, default=256)
    p.add_argument("--token_budget", type=int, default=256)
    p.add_argument("--history_frames", type=int, default=1)
    p.add_argument("--dt_minutes", type=float, default=40.0)
    p.add_argument("--time_bin_minutes", type=float, default=40.0)
    p.add_argument("--min_cells_per_snapshot", type=int, default=64)
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--species_filter", default="C.elegans")
    p.add_argument("--min_umi", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_spatial_layers", type=int, default=2)
    p.add_argument("--n_temporal_layers", type=int, default=4)
    p.add_argument("--n_decoder_layers", type=int, default=2)
    p.add_argument("--head_dim", type=int, default=32)
    p.add_argument("--mask_ratio", type=float, default=0.0)
    p.add_argument("--masked_gene_weight", type=float, default=0.0)
    p.add_argument("--gene_set_weight", type=float, default=1.0)
    p.add_argument("--mean_gene_weight", type=float, default=0.2)
    p.add_argument("--gene_sinkhorn_blur", type=float, default=0.1)
    p.add_argument("--checkpoint_dir", default="checkpoints/large2025_lineage_stage1")
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
    return Large2025WholeEmbryoDataset(
        data_dir=args.data_dir,
        n_hvg=args.n_hvg,
        token_budget=args.token_budget,
        history_frames=args.history_frames,
        dt_minutes=args.dt_minutes,
        time_bin_minutes=args.time_bin_minutes,
        min_cells_per_snapshot=args.min_cells_per_snapshot,
        split=split,
        val_fraction=args.val_fraction,
        random_seed=args.seed,
        species_filter=args.species_filter,
        min_umi=args.min_umi,
    )


def _stack_history_tensor(batch: dict[str, torch.Tensor], field: str, n_frames: int) -> torch.Tensor:
    return torch.stack([batch[f"history_frame_{i}_{field}"] for i in range(n_frames)], dim=1)


def prepare_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    n_frames = int(batch["history_frames"][0].item())
    batch["history_genes"] = _stack_history_tensor(batch, "genes", n_frames)
    batch["history_token_times"] = _stack_history_tensor(batch, "token_times", n_frames)
    batch["history_valid_mask"] = _stack_history_tensor(batch, "valid_mask", n_frames)
    batch["history_lineage_binary"] = _stack_history_tensor(batch, "lineage_binary", n_frames)
    batch["history_founder_ids"] = _stack_history_tensor(batch, "founder_ids", n_frames)
    batch["history_lineage_depth"] = _stack_history_tensor(batch, "lineage_depth", n_frames)
    batch["history_lineage_valid"] = _stack_history_tensor(batch, "lineage_valid", n_frames)
    batch["history_token_rank"] = _stack_history_tensor(batch, "token_rank", n_frames)
    batch["history_time"] = batch[f"history_frame_{n_frames - 1}_time"]
    return batch


def sample_history_mask(valid_mask: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    if mask_ratio <= 0.0:
        return torch.zeros_like(valid_mask, dtype=torch.bool)
    if mask_ratio >= 1.0:
        return valid_mask.bool()
    return (torch.rand_like(valid_mask.float()) < mask_ratio) & valid_mask.bool()


def apply_history_mask(history_genes: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
    masked_genes = history_genes.clone()
    masked_genes[history_mask] = 0.0
    return masked_genes


def masked_gene_recon_loss(pred_history_genes: torch.Tensor, history_genes: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
    if not history_mask.any():
        return pred_history_genes.new_zeros(())
    return F.mse_loss(pred_history_genes[history_mask], history_genes[history_mask])


def compute_current_mean_gene(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    current_genes = batch["history_genes"][:, -1]
    current_valid = batch["history_valid_mask"][:, -1].unsqueeze(-1).float()
    denom = current_valid.sum(dim=1).clamp_min(1.0)
    return (current_genes * current_valid).sum(dim=1) / denom


def compute_loss(model: LineageWholeEmbryoModel, batch: dict[str, torch.Tensor], args, apply_mask: bool):
    use_mask = apply_mask and args.masked_gene_weight > 0.0 and args.mask_ratio > 0.0
    history_mask = sample_history_mask(batch["history_valid_mask"], args.mask_ratio) if use_mask else torch.zeros_like(batch["history_valid_mask"], dtype=torch.bool)
    input_history_genes = apply_history_mask(batch["history_genes"], history_mask) if use_mask else batch["history_genes"]
    out = model(
        genes=input_history_genes,
        time=batch["history_time"],
        future_time=batch["future_time"],
        token_times=batch["history_token_times"],
        valid_mask=batch["history_valid_mask"],
        lineage_binary=batch["history_lineage_binary"],
        founder_ids=batch["history_founder_ids"],
        lineage_depth=batch["history_lineage_depth"],
        lineage_valid=batch["history_lineage_valid"],
        token_rank=batch["history_token_rank"],
    )
    target_future = batch["future_genes"]
    target_mean_gene = batch["future_mean_gene"]
    gene_set = sinkhorn_divergence(out.pred_future_genes, target_future, blur=args.gene_sinkhorn_blur)
    mean_gene = F.mse_loss(out.pred_mean_gene, target_mean_gene)
    masked_gene = masked_gene_recon_loss(out.pred_history_genes, batch["history_genes"], history_mask)
    persistence_mean_gene = F.mse_loss(compute_current_mean_gene(batch), target_mean_gene)
    total = (
        args.masked_gene_weight * masked_gene
        + args.gene_set_weight * gene_set
        + args.mean_gene_weight * mean_gene
    )
    return total, {
        "total": float(total.item()),
        "masked_gene": float(masked_gene.item()),
        "gene_set": float(gene_set.item()),
        "mean_gene": float(mean_gene.item()),
        "persistence_mean_gene": float(persistence_mean_gene.item()),
    }


def run_epoch(model, loader, args, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = {
        "total": 0.0,
        "masked_gene": 0.0,
        "gene_set": 0.0,
        "mean_gene": 0.0,
        "persistence_mean_gene": 0.0,
    }
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
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_large2025_whole_embryo)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_large2025_whole_embryo)
    model = LineageWholeEmbryoModel(
        gene_dim=train_ds.gene_dim,
        context_size=args.token_budget,
        history_frames=args.history_frames,
        lineage_binary_dim=train_ds.lineage_binary.shape[1],
        founder_vocab_size=len(train_ds.FOUNDER_VOCAB),
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
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": vars(args),
                    "gene_dim": train_ds.gene_dim,
                    "best_val_total": best_val,
                },
                checkpoint_dir / "best.pt",
            )
        print(
            f"epoch={epoch} train_total={train_metrics['total']:.4f} "
            f"val_total={val_metrics['total']:.4f} "
            f"train_masked_gene={train_metrics['masked_gene']:.4f} "
            f"val_masked_gene={val_metrics['masked_gene']:.4f} "
            f"val_gene_set={val_metrics['gene_set']:.4f} "
            f"val_mean_gene={val_metrics['mean_gene']:.4f} "
            f"val_persist_mean={val_metrics['persistence_mean_gene']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
