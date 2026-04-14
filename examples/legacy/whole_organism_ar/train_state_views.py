#!/usr/bin/env python3
"""Train shared-encoder multi-view state representations from local patch views."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
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
from src.data.gene_context_dataset import (  # noqa: E402
    MultiViewPatchStateDataset,
    collate_multi_view_patch_state,
)


class StateViewModel(nn.Module):
    def __init__(
        self,
        gene_dim: int,
        context_size: int,
        model_type: str = "multi_cell",
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        head_dim: int = 32,
        use_pairwise_spatial_bias: bool = True,
    ):
        super().__init__()
        if model_type == "single_cell":
            self.encoder = SingleCellPatchSetModel(
                gene_dim=gene_dim,
                context_size=context_size,
                d_model=d_model,
                n_layers=n_layers,
            )
        else:
            self.encoder = MultiCellPatchSetModel(
                gene_dim=gene_dim,
                context_size=context_size,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                head_dim=head_dim,
                use_pairwise_spatial_bias=use_pairwise_spatial_bias,
            )
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.future_decoder = self.encoder.future_token_head

    def encode_view(self, batch: dict[str, torch.Tensor], prefix: str):
        return self.encoder.encode_patch(
            genes=batch[f"{prefix}_genes"],
            time=batch[f"{prefix}_time"],
            future_time=batch[f"{prefix}_time"],
            token_times=batch[f"{prefix}_token_times"],
            valid_mask=batch[f"{prefix}_valid_mask"],
            anchor_mask=batch[f"{prefix}_anchor_mask"],
            context_role=batch.get(f"{prefix}_context_role"),
            relative_position=batch.get(f"{prefix}_relative_position"),
        )


def compute_metrics(model: StateViewModel, batch: dict[str, torch.Tensor], ot_weight: float):
    z_a, _ = model.encode_view(batch, "current_view_0")
    z_b, _ = model.encode_view(batch, "current_view_1")
    z_future, _ = model.encode_view(batch, "future_view_0")

    view_loss = (1.0 - F.cosine_similarity(z_a, z_b, dim=-1)).mean()
    pred_future = model.predictor(z_a)
    future_loss = (1.0 - F.cosine_similarity(pred_future, z_future.detach(), dim=-1)).mean()

    pred_future_genes = model.future_decoder(pred_future).view(
        batch["current_view_0_genes"].shape[0],
        model.encoder.context_size,
        model.encoder.encoder.gene_dim,
    )
    ot_loss = sinkhorn_divergence(
        pred_future_genes,
        batch["future_view_0_genes"],
        blur=0.1,
        p=2,
    )
    total = view_loss + future_loss + ot_weight * ot_loss
    return total, {
        "total": total.item(),
        "view": view_loss.item(),
        "future": future_loss.item(),
        "ot": ot_loss.item(),
    }


def run_epoch(model, loader, optimizer, device: str, ot_weight: float):
    train = optimizer is not None
    model.train(train)
    totals: dict[str, float] = {}
    n_batches = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss, metrics = compute_metrics(model, batch, ot_weight)
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
    p = argparse.ArgumentParser(description="Train multi-view state representations from local patch views.")
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
    p.add_argument("--sampling_strategy", choices=["random_window", "spatial_neighbors", "spatial_anchor"], default="spatial_anchor")
    p.add_argument("--min_spatial_cells_per_window", type=int, default=8)
    p.add_argument("--spatial_neighbor_pool_size", type=int, default=None)
    p.add_argument("--views_per_state", type=int, default=2)
    p.add_argument("--future_views_per_state", type=int, default=1)
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
    p.add_argument("--ot_weight", type=float, default=0.1)
    p.add_argument("--checkpoint_dir", default="checkpoints/state_views")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    train_filters = resolve_event_filters(args)
    val_filters = resolve_event_filters(args, prefix="val_")

    train_ds = MultiViewPatchStateDataset(
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
        views_per_state=args.views_per_state,
        future_views_per_state=args.future_views_per_state,
        split="train",
        val_fraction=args.val_fraction,
        random_seed=args.seed,
        **train_filters,
    )
    val_ds = MultiViewPatchStateDataset(
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
        views_per_state=args.views_per_state,
        future_views_per_state=args.future_views_per_state,
        split="val",
        val_fraction=args.val_fraction,
        random_seed=args.seed + 1000,
        **val_filters,
    )
    if not train_ds.time_pairs or not val_ds.time_pairs:
        raise ValueError("State-view dataset is empty after filtering.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_multi_view_patch_state)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_multi_view_patch_state)

    model = StateViewModel(
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
        train_metrics = run_epoch(model, train_loader, optimizer, args.device, args.ot_weight)
        val_metrics = run_epoch(model, val_loader, None, args.device, args.ot_weight)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"epoch={epoch} train_total={train_metrics['total']:.4f} "
            f"val_total={val_metrics['total']:.4f} val_view={val_metrics['view']:.4f} val_future={val_metrics['future']:.4f}"
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

    with open(checkpoint_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
