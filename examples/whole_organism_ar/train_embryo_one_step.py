#!/usr/bin/env python3
"""Train embryo one-step latent prediction on top of a masked-future backbone."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
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
from src.branching_flows.gene_context import (  # noqa: E402
    EmbryoMaskedViewModel,
    EmbryoOneStepLatentModel,
)
from src.data.gene_context_dataset import EmbryoViewDataset, collate_embryo_view  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Train embryo one-step latent predictor.")
    p.add_argument("--backbone_checkpoint", required=True)
    p.add_argument("--h5ad_path", default="dataset/processed/nema_extended_large2025.h5ad")
    p.add_argument("--n_hvg", type=int, default=256)
    p.add_argument("--context_size", type=int, default=256)
    p.add_argument("--global_context_size", type=int, default=32)
    p.add_argument("--dt_minutes", type=float, default=40.0)
    p.add_argument("--time_window_minutes", type=float, default=10.0)
    p.add_argument("--samples_per_pair", type=int, default=16)
    p.add_argument("--val_samples_per_pair", type=int, default=None)
    p.add_argument("--views_per_embryo", type=int, default=8)
    p.add_argument("--future_views_per_embryo", type=int, default=8)
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
    p.add_argument("--predict_delta", action="store_true", default=True)
    p.add_argument("--no_predict_delta", dest="predict_delta", action="store_false")
    p.add_argument("--latent_weight", type=float, default=10.0)
    p.add_argument("--probe_weight", type=float, default=0.0)
    p.add_argument("--semantic_probe_weight", type=float, default=1.0)
    p.add_argument("--freeze_backbone", action="store_true", default=True)
    p.add_argument("--no_freeze_backbone", dest="freeze_backbone", action="store_false")
    p.add_argument("--checkpoint_dir", default="checkpoints_embryo_one_step")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def r2_score_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - y_true.mean(dim=0, keepdim=True)) ** 2)
    if float(ss_tot.item()) <= 1e-8:
        return 0.0
    return float((1.0 - ss_res / ss_tot).item())


def compute_probe_stats(dataset: EmbryoViewDataset) -> dict[str, torch.Tensor]:
    founder = []
    celltype = []
    depth = []
    spatial = []
    split = []
    for i in range(len(dataset)):
        item = dataset[i]
        founder.append(item["future_founder_composition"])
        celltype.append(item["future_celltype_composition"])
        depth.append(item["future_lineage_depth_stats"])
        spatial.append(item["future_spatial_extent"])
        split.append(item["future_split_fraction"])

    def _stack(xs):
        x = torch.stack(xs, dim=0).float()
        mean = x.mean(dim=0)
        std = x.std(dim=0, unbiased=False).clamp_min(1e-6)
        return mean, std

    founder_mean, founder_std = _stack(founder)
    celltype_mean, celltype_std = _stack(celltype)
    depth_mean, depth_std = _stack(depth)
    spatial_mean, spatial_std = _stack(spatial)
    split_mean, split_std = _stack(split)
    return {
        "founder_mean": founder_mean,
        "founder_std": founder_std,
        "celltype_mean": celltype_mean,
        "celltype_std": celltype_std,
        "depth_mean": depth_mean,
        "depth_std": depth_std,
        "spatial_mean": spatial_mean,
        "spatial_std": spatial_std,
        "split_mean": split_mean,
        "split_std": split_std,
    }


def standardize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


def fit_linear_probe_bank(
    backbone: EmbryoMaskedViewModel,
    dataset: EmbryoViewDataset,
    batch_size: int,
    device: str,
) -> dict[str, dict[str, torch.Tensor]]:
    """Fit linear probes on true future embryo latents from the frozen backbone."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_embryo_view)
    latents = []
    targets = {k: [] for k in ("founder", "celltype", "depth", "spatial", "split")}

    was_training = backbone.training
    backbone.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            n_future_views = int(batch["future_views_per_embryo"][0].item())
            future_genes = stack_view_tensor(batch, "genes", n_future_views, prefix_template="future_view_{i}_")
            future_context_role = stack_view_tensor(batch, "context_role", n_future_views, prefix_template="future_view_{i}_")
            future_relative_position = stack_view_tensor(
                batch,
                "relative_position",
                n_future_views,
                prefix_template="future_view_{i}_",
            )
            future_token_times = stack_view_tensor(batch, "token_times", n_future_views, prefix_template="future_view_{i}_")
            future_valid_mask = stack_view_tensor(batch, "valid_mask", n_future_views, prefix_template="future_view_{i}_")
            future_anchor_mask = stack_view_tensor(batch, "anchor_mask", n_future_views, prefix_template="future_view_{i}_")
            future_time = stack_view_tensor(batch, "time", n_future_views, prefix_template="future_view_{i}_")

            future_latent, _ = backbone.encode_embryo_latent(
                genes=future_genes,
                time=future_time,
                token_times=future_token_times,
                valid_mask=future_valid_mask,
                anchor_mask=future_anchor_mask,
                context_role=future_context_role,
                relative_position=future_relative_position,
            )
            latents.append(future_latent.cpu().numpy().astype(np.float32))
            targets["founder"].append(batch["future_founder_composition"].cpu().numpy().astype(np.float32))
            targets["celltype"].append(batch["future_celltype_composition"].cpu().numpy().astype(np.float32))
            targets["depth"].append(batch["future_lineage_depth_stats"].cpu().numpy().astype(np.float32))
            targets["spatial"].append(batch["future_spatial_extent"].cpu().numpy().astype(np.float32))
            targets["split"].append(batch["future_split_fraction"].cpu().numpy().astype(np.float32))
    if was_training:
        backbone.train()

    x = np.concatenate(latents, axis=0)
    design = np.concatenate([x, np.ones((len(x), 1), dtype=x.dtype)], axis=1)
    probe_bank = {}
    for key, value in targets.items():
        y = np.concatenate(value, axis=0)
        weight, *_ = np.linalg.lstsq(design, y, rcond=None)
        w = torch.from_numpy(weight[:-1].T.copy()).float()
        b = torch.from_numpy(weight[-1].copy()).float()
        probe_bank[key] = {"weight": w, "bias": b}
    return probe_bank


def apply_probe_bank(
    latent: torch.Tensor,
    probe_bank: dict[str, dict[str, torch.Tensor]],
    key: str,
) -> torch.Tensor:
    probe = probe_bank[key]
    weight = probe["weight"].to(latent.device)
    bias = probe["bias"].to(latent.device)
    return F.linear(latent, weight, bias)


def compute_metrics(
    model: EmbryoOneStepLatentModel,
    batch: dict[str, torch.Tensor],
    latent_weight: float,
    probe_weight: float,
    semantic_probe_weight: float,
    probe_stats: dict[str, torch.Tensor],
    probe_bank: dict[str, dict[str, torch.Tensor]] | None,
):
    n_views = int(batch["views_per_embryo"][0].item())
    n_future_views = int(batch["future_views_per_embryo"][0].item())
    genes = stack_view_tensor(batch, "genes", n_views)
    context_role = stack_view_tensor(batch, "context_role", n_views)
    relative_position = stack_view_tensor(batch, "relative_position", n_views)
    token_times = stack_view_tensor(batch, "token_times", n_views)
    valid_mask = stack_view_tensor(batch, "valid_mask", n_views)
    anchor_mask = stack_view_tensor(batch, "anchor_mask", n_views)
    time = stack_view_tensor(batch, "time", n_views)
    future_genes = stack_view_tensor(batch, "genes", n_future_views, prefix_template="future_view_{i}_")
    future_context_role = stack_view_tensor(batch, "context_role", n_future_views, prefix_template="future_view_{i}_")
    future_relative_position = stack_view_tensor(batch, "relative_position", n_future_views, prefix_template="future_view_{i}_")
    future_token_times = stack_view_tensor(batch, "token_times", n_future_views, prefix_template="future_view_{i}_")
    future_valid_mask = stack_view_tensor(batch, "valid_mask", n_future_views, prefix_template="future_view_{i}_")
    future_anchor_mask = stack_view_tensor(batch, "anchor_mask", n_future_views, prefix_template="future_view_{i}_")
    future_time = stack_view_tensor(batch, "time", n_future_views, prefix_template="future_view_{i}_")

    out = model(
        genes=genes,
        time=time,
        token_times=token_times,
        valid_mask=valid_mask,
        anchor_mask=anchor_mask,
        future_genes=future_genes,
        future_time=future_time,
        future_token_times=future_token_times,
        future_valid_mask=future_valid_mask,
        future_anchor_mask=future_anchor_mask,
        context_role=context_role,
        relative_position=relative_position,
        future_context_role=future_context_role,
        future_relative_position=future_relative_position,
    )
    if out.pred_future_delta is not None and out.target_future_delta is not None and model.predict_delta:
        latent_loss = (
            1.0 - F.cosine_similarity(out.pred_future_delta, out.target_future_delta.detach(), dim=-1)
        ).mean()
    else:
        latent_loss = (
            1.0 - F.cosine_similarity(out.pred_future_embryo_latent, out.target_future_embryo_latent.detach(), dim=-1)
        ).mean()
    founder_target = standardize(
        batch["future_founder_composition"],
        probe_stats["founder_mean"].to(batch["future_founder_composition"].device),
        probe_stats["founder_std"].to(batch["future_founder_composition"].device),
    )
    celltype_target = standardize(
        batch["future_celltype_composition"],
        probe_stats["celltype_mean"].to(batch["future_celltype_composition"].device),
        probe_stats["celltype_std"].to(batch["future_celltype_composition"].device),
    )
    depth_target = standardize(
        batch["future_lineage_depth_stats"],
        probe_stats["depth_mean"].to(batch["future_lineage_depth_stats"].device),
        probe_stats["depth_std"].to(batch["future_lineage_depth_stats"].device),
    )
    spatial_target = standardize(
        batch["future_spatial_extent"],
        probe_stats["spatial_mean"].to(batch["future_spatial_extent"].device),
        probe_stats["spatial_std"].to(batch["future_spatial_extent"].device),
    )
    split_target = standardize(
        batch["future_split_fraction"],
        probe_stats["split_mean"].to(batch["future_split_fraction"].device),
        probe_stats["split_std"].to(batch["future_split_fraction"].device),
    )
    founder_loss = F.mse_loss(out.future_founder_composition, founder_target)
    celltype_loss = F.mse_loss(out.future_celltype_composition, celltype_target)
    depth_loss = F.mse_loss(out.future_lineage_depth_stats, depth_target)
    spatial_loss = F.mse_loss(out.future_spatial_extent, spatial_target)
    split_loss = F.mse_loss(out.future_split_fraction, split_target)
    probe_loss = founder_loss + celltype_loss + depth_loss + spatial_loss + split_loss
    semantic_founder_loss = torch.tensor(0.0, device=latent_loss.device)
    semantic_celltype_loss = torch.tensor(0.0, device=latent_loss.device)
    semantic_depth_loss = torch.tensor(0.0, device=latent_loss.device)
    semantic_spatial_loss = torch.tensor(0.0, device=latent_loss.device)
    semantic_split_loss = torch.tensor(0.0, device=latent_loss.device)
    semantic_probe_loss = torch.tensor(0.0, device=latent_loss.device)
    if probe_bank is not None:
        semantic_founder_loss = F.mse_loss(
            apply_probe_bank(out.pred_future_embryo_latent, probe_bank, "founder"),
            founder_target,
        )
        semantic_celltype_loss = F.mse_loss(
            apply_probe_bank(out.pred_future_embryo_latent, probe_bank, "celltype"),
            celltype_target,
        )
        semantic_depth_loss = F.mse_loss(
            apply_probe_bank(out.pred_future_embryo_latent, probe_bank, "depth"),
            depth_target,
        )
        semantic_spatial_loss = F.mse_loss(
            apply_probe_bank(out.pred_future_embryo_latent, probe_bank, "spatial"),
            spatial_target,
        )
        semantic_split_loss = F.mse_loss(
            apply_probe_bank(out.pred_future_embryo_latent, probe_bank, "split"),
            split_target,
        )
        semantic_probe_loss = (
            semantic_founder_loss
            + semantic_celltype_loss
            + semantic_depth_loss
            + 0.5 * semantic_spatial_loss
            + semantic_split_loss
        )
    total = (
        latent_weight * latent_loss
        + probe_weight * probe_loss
        + semantic_probe_weight * semantic_probe_loss
    )
    metrics = {
        "total": total.item(),
        "latent": latent_loss.item(),
        "founder": founder_loss.item(),
        "celltype": celltype_loss.item(),
        "depth": depth_loss.item(),
        "spatial": spatial_loss.item(),
        "split": split_loss.item(),
        "semantic_founder": semantic_founder_loss.item(),
        "semantic_celltype": semantic_celltype_loss.item(),
        "semantic_depth": semantic_depth_loss.item(),
        "semantic_spatial": semantic_spatial_loss.item(),
        "semantic_split": semantic_split_loss.item(),
        "founder_r2": r2_score_torch(batch["future_founder_composition"], out.future_founder_composition),
        "celltype_r2": r2_score_torch(batch["future_celltype_composition"], out.future_celltype_composition),
        "depth_r2": r2_score_torch(batch["future_lineage_depth_stats"], out.future_lineage_depth_stats),
        "spatial_r2": r2_score_torch(batch["future_spatial_extent"], out.future_spatial_extent),
        "split_r2": r2_score_torch(batch["future_split_fraction"], out.future_split_fraction),
    }
    return total, metrics


def run_epoch(
    model,
    loader,
    optimizer,
    device: str,
    latent_weight: float,
    probe_weight: float,
    semantic_probe_weight: float,
    probe_stats: dict[str, torch.Tensor],
    probe_bank: dict[str, dict[str, torch.Tensor]] | None,
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
            latent_weight,
            probe_weight,
            semantic_probe_weight,
            probe_stats,
            probe_bank,
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
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_embryo_view)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_embryo_view)
    probe_stats = compute_probe_stats(train_ds)

    ckpt = torch.load(args.backbone_checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    backbone = EmbryoMaskedViewModel(
        gene_dim=ckpt["gene_dim"],
        context_size=cfg["context_size"],
        model_type=cfg["model_type"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        head_dim=cfg["head_dim"],
        use_pairwise_spatial_bias=cfg["pairwise_spatial_bias"],
    )
    backbone.load_state_dict(ckpt["model_state_dict"])
    model = EmbryoOneStepLatentModel(
        backbone=backbone,
        celltype_dim=len(train_ds._top_cell_type_vocab),
        d_model=cfg["d_model"],
        predict_delta=args.predict_delta,
    ).to(args.device)
    if args.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
        model.backbone.eval()

    probe_bank = fit_linear_probe_bank(
        backbone=model.backbone,
        dataset=train_ds,
        batch_size=args.batch_size,
        device=args.device,
    )

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
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
            args.latent_weight,
            args.probe_weight,
            args.semantic_probe_weight,
            probe_stats,
            probe_bank,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            None,
            args.device,
            args.latent_weight,
            args.probe_weight,
            args.semantic_probe_weight,
            probe_stats,
            probe_bank,
        )
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"epoch={epoch} train_total={train_metrics['total']:.4f} "
            f"val_total={val_metrics['total']:.4f} val_latent={val_metrics['latent']:.4f} "
            f"val_sem_founder={val_metrics['semantic_founder']:.4f} "
            f"val_sem_celltype={val_metrics['semantic_celltype']:.4f} "
            f"val_founder_r2={val_metrics['founder_r2']:.4f} val_celltype_r2={val_metrics['celltype_r2']:.4f} "
            f"val_depth_r2={val_metrics['depth_r2']:.4f} val_spatial_r2={val_metrics['spatial_r2']:.4f} "
            f"val_split_r2={val_metrics['split_r2']:.4f}"
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
                    "probe_bank": {
                        key: {"weight": value["weight"].cpu(), "bias": value["bias"].cpu()}
                        for key, value in probe_bank.items()
                    },
                    "backbone_checkpoint": args.backbone_checkpoint,
                },
                checkpoint_dir / "best.pt",
            )
    with open(checkpoint_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
