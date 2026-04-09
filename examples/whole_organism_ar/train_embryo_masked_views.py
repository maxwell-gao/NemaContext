#!/usr/bin/env python3
"""Train embryo-level masked multi-view state representations."""

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
from src.branching_flows.gene_context import EmbryoMaskedViewModel  # noqa: E402
from src.data.gene_context_dataset import EmbryoViewDataset, collate_embryo_view  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Train embryo-level masked multi-view model.")
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
    p.add_argument("--future_views_per_embryo", type=int, default=None)
    p.add_argument("--top_cell_types", type=int, default=8)
    p.add_argument("--mask_ratio", type=float, default=0.25)
    p.add_argument("--future_mask_ratio", type=float, default=0.25)
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
    p.add_argument("--latent_weight", type=float, default=1.0)
    p.add_argument("--gene_weight", type=float, default=1.0)
    p.add_argument("--future_latent_weight", type=float, default=1.0)
    p.add_argument("--future_gene_weight", type=float, default=1.0)
    p.add_argument("--checkpoint_dir", default="checkpoints/embryo_masked_views")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_mask(batch_size: int, n_views: int, mask_ratio: float, device: str) -> torch.Tensor:
    n_mask = max(1, min(n_views - 1, int(round(mask_ratio * n_views))))
    mask = torch.zeros(batch_size, n_views, dtype=torch.bool, device=device)
    for i in range(batch_size):
        perm = torch.randperm(n_views, device=device)
        mask[i, perm[:n_mask]] = True
    return mask


def compute_metrics(
    model: EmbryoMaskedViewModel,
    batch: dict[str, torch.Tensor],
    mask_ratio: float,
    future_mask_ratio: float,
    latent_weight: float,
    gene_weight: float,
    future_latent_weight: float,
    future_gene_weight: float,
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
    future_context_role = stack_view_tensor(batch, "context_role", n_future_views, prefix_template="future_view_{i}_")
    future_relative_position = stack_view_tensor(batch, "relative_position", n_future_views, prefix_template="future_view_{i}_")
    future_token_times = stack_view_tensor(batch, "token_times", n_future_views, prefix_template="future_view_{i}_")
    future_valid_mask = stack_view_tensor(batch, "valid_mask", n_future_views, prefix_template="future_view_{i}_")
    future_anchor_mask = stack_view_tensor(batch, "anchor_mask", n_future_views, prefix_template="future_view_{i}_")
    future_time = stack_view_tensor(batch, "time", n_future_views, prefix_template="future_view_{i}_")
    masked_view_mask = build_mask(genes.shape[0], n_views, mask_ratio, genes.device)
    masked_future_view_mask = build_mask(genes.shape[0], n_future_views, future_mask_ratio, genes.device)

    out = model(
        genes=genes,
        time=time,
        token_times=token_times,
        valid_mask=valid_mask,
        anchor_mask=anchor_mask,
        masked_view_mask=masked_view_mask,
        future_genes=future_genes,
        future_time=future_time,
        future_token_times=future_token_times,
        future_valid_mask=future_valid_mask,
        future_anchor_mask=future_anchor_mask,
        masked_future_view_mask=masked_future_view_mask,
        context_role=context_role,
        relative_position=relative_position,
        future_context_role=future_context_role,
        future_relative_position=future_relative_position,
    )

    target_latents = []
    target_genes = []
    for i in range(genes.shape[0]):
        masked_idx = torch.nonzero(masked_view_mask[i], as_tuple=False).squeeze(-1)
        target_latents.append(out.local_latents[i, masked_idx].mean(dim=0))
        valid_views = valid_mask[i, masked_idx]
        masked_genes = genes[i, masked_idx]
        target_genes.append(
            (masked_genes * valid_views.unsqueeze(-1).float()).sum(dim=(0, 1))
            / valid_views.float().sum().clamp_min(1.0)
        )
    target_latents = torch.stack(target_latents, dim=0)
    target_genes = torch.stack(target_genes, dim=0)
    future_target_latents = []
    future_target_genes = []
    for i in range(genes.shape[0]):
        masked_idx = torch.nonzero(masked_future_view_mask[i], as_tuple=False).squeeze(-1)
        future_target_latents.append(out.future_local_latents[i, masked_idx].mean(dim=0))
        valid_views = future_valid_mask[i, masked_idx]
        masked_genes = future_genes[i, masked_idx]
        future_target_genes.append(
            (masked_genes * valid_views.unsqueeze(-1).float()).sum(dim=(0, 1))
            / valid_views.float().sum().clamp_min(1.0)
        )
    future_target_latents = torch.stack(future_target_latents, dim=0)
    future_target_genes = torch.stack(future_target_genes, dim=0)

    latent_loss = (1.0 - F.cosine_similarity(out.pred_masked_view_latents, target_latents.detach(), dim=-1)).mean()
    gene_loss = F.mse_loss(out.pred_masked_view_genes, target_genes)
    future_latent_loss = (
        1.0 - F.cosine_similarity(out.pred_masked_future_view_latents, future_target_latents.detach(), dim=-1)
    ).mean()
    future_gene_loss = F.mse_loss(out.pred_masked_future_view_genes, future_target_genes)
    total = (
        latent_weight * latent_loss
        + gene_weight * gene_loss
        + future_latent_weight * future_latent_loss
        + future_gene_weight * future_gene_loss
    )
    return total, {
        "total": total.item(),
        "masked_latent": latent_loss.item(),
        "masked_gene": gene_loss.item(),
        "masked_future_latent": future_latent_loss.item(),
        "masked_future_gene": future_gene_loss.item(),
    }


def run_epoch(
    model,
    loader,
    optimizer,
    device: str,
    mask_ratio: float,
    future_mask_ratio: float,
    latent_weight: float,
    gene_weight: float,
    future_latent_weight: float,
    future_gene_weight: float,
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
            mask_ratio,
            future_mask_ratio,
            latent_weight,
            gene_weight,
            future_latent_weight,
            future_gene_weight,
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
        random_seed=args.seed + 1000,
        sampling_strategy=args.sampling_strategy,
        min_spatial_cells_per_window=args.min_spatial_cells_per_window,
        spatial_neighbor_pool_size=args.spatial_neighbor_pool_size,
        delete_target_mode=args.delete_target_mode,
        views_per_embryo=args.views_per_embryo,
        future_views_per_embryo=args.future_views_per_embryo,
        top_cell_types=args.top_cell_types,
        **val_filters,
    )
    if not train_ds.time_pairs or not val_ds.time_pairs:
        raise ValueError("Embryo masked-view dataset is empty after filtering.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_embryo_view)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_embryo_view)

    model = EmbryoMaskedViewModel(
        gene_dim=train_ds.gene_dim,
        context_size=args.context_size,
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
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            args.device,
            args.mask_ratio,
            args.future_mask_ratio,
            args.latent_weight,
            args.gene_weight,
            args.future_latent_weight,
            args.future_gene_weight,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            None,
            args.device,
            args.mask_ratio,
            args.future_mask_ratio,
            args.latent_weight,
            args.gene_weight,
            args.future_latent_weight,
            args.future_gene_weight,
        )
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"epoch={epoch} train_total={train_metrics['total']:.4f} "
            f"val_total={val_metrics['total']:.4f} val_masked_latent={val_metrics['masked_latent']:.4f} "
            f"val_masked_gene={val_metrics['masked_gene']:.4f} "
            f"val_masked_future_latent={val_metrics['masked_future_latent']:.4f} "
            f"val_masked_future_gene={val_metrics['masked_future_gene']:.4f}"
        )
        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": vars(args),
                    "gene_dim": train_ds.gene_dim,
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
