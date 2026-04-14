#!/usr/bin/env python3
"""Simple local-region gene-expression baseline for embryo time prediction."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.gene_context_dataset_patch import PatchSetDataset  # noqa: E402


@dataclass(frozen=True)
class RegionBatch:
    x: np.ndarray
    y: np.ndarray
    pair_id: np.ndarray
    current_count: np.ndarray
    future_count: np.ndarray


class MeanGeneMLP(nn.Module):
    def __init__(self, gene_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(gene_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, gene_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5ad_path", default="dataset/processed/nema_extended_large2025.h5ad")
    p.add_argument("--n_hvg", type=int, default=256)
    p.add_argument("--region_sizes", type=int, nargs="+", default=[8, 16, 32])
    p.add_argument("--dt_minutes", type=float, default=40.0)
    p.add_argument("--time_window_minutes", type=float, default=10.0)
    p.add_argument("--samples_per_pair", type=int, default=16)
    p.add_argument("--min_cells_per_window", type=int, default=32)
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--sampling_strategy", choices=["spatial_anchor"], default="spatial_anchor")
    p.add_argument("--min_spatial_cells_per_window", type=int, default=8)
    p.add_argument("--spatial_neighbor_pool_size", type=int, default=None)
    p.add_argument("--ridge_alpha", type=float, default=1.0)
    p.add_argument("--mlp_hidden_dim", type=int, default=256)
    p.add_argument("--mlp_epochs", type=int, default=200)
    p.add_argument("--mlp_lr", type=float, default=1e-3)
    p.add_argument("--mlp_weight_decay", type=float, default=1e-4)
    p.add_argument("--mlp_batch_size", type=int, default=128)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_json", required=True)
    return p.parse_args()


def build_dataset(args, split: str) -> PatchSetDataset:
    return PatchSetDataset(
        h5ad_path=args.h5ad_path,
        n_hvg=args.n_hvg,
        context_size=max(args.region_sizes),
        global_context_size=0,
        dt_minutes=args.dt_minutes,
        time_window_minutes=args.time_window_minutes,
        samples_per_pair=args.samples_per_pair,
        min_cells_per_window=args.min_cells_per_window,
        split=split,
        val_fraction=args.val_fraction,
        random_seed=args.seed,
        sampling_strategy=args.sampling_strategy,
        min_spatial_cells_per_window=max(args.min_spatial_cells_per_window, max(args.region_sizes)),
        spatial_neighbor_pool_size=args.spatial_neighbor_pool_size or max(args.region_sizes),
        delete_target_mode="strict",
        patch_composition="local_only",
    )


def select_region(indices: np.ndarray, coords: np.ndarray, anchor_coord: np.ndarray, take: int) -> np.ndarray:
    distances = np.linalg.norm(coords[indices] - anchor_coord, axis=1)
    order = np.argsort(distances)
    return indices[order[:take]]


def collect_region_mean_pairs(dataset: PatchSetDataset, region_size: int, max_samples: int | None) -> RegionBatch:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    pair_ids: list[int] = []
    current_counts: list[int] = []
    future_counts: list[int] = []

    total = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    for idx in range(total):
        pair_local_idx = idx % len(dataset.time_pairs)
        pair = dataset.time_pairs[pair_local_idx]
        rng = np.random.default_rng(dataset.random_seed + idx)
        current_spatial = pair.current_indices[dataset.valid_spatial[pair.current_indices]]
        future_spatial = pair.future_indices[dataset.valid_spatial[pair.future_indices]]
        if len(current_spatial) == 0 or len(future_spatial) == 0:
            continue

        anchor = int(rng.choice(current_spatial))
        anchor_coord = dataset.spatial_coords[anchor]

        current_take = min(region_size, len(current_spatial))
        future_take = min(region_size, len(future_spatial))
        current_region = select_region(current_spatial, dataset.spatial_coords, anchor_coord, current_take)
        future_region = select_region(future_spatial, dataset.spatial_coords, anchor_coord, future_take)

        xs.append(dataset.genes[current_region].mean(axis=0).astype(np.float32))
        ys.append(dataset.genes[future_region].mean(axis=0).astype(np.float32))
        pair_ids.append(pair_local_idx)
        current_counts.append(int(current_take))
        future_counts.append(int(future_take))

    if not xs:
        raise RuntimeError("No valid spatial region pairs were collected.")

    return RegionBatch(
        x=np.stack(xs).astype(np.float32),
        y=np.stack(ys).astype(np.float32),
        pair_id=np.asarray(pair_ids, dtype=np.int64),
        current_count=np.asarray(current_counts, dtype=np.int64),
        future_count=np.asarray(future_counts, dtype=np.int64),
    )


def fit_ridge(train_x: np.ndarray, train_y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    x_mean = train_x.mean(axis=0, keepdims=True)
    y_mean = train_y.mean(axis=0, keepdims=True)
    x_centered = train_x - x_mean
    y_centered = train_y - y_mean
    gram = x_centered.T @ x_centered
    reg = alpha * np.eye(gram.shape[0], dtype=np.float32)
    weight = np.linalg.solve(gram + reg, x_centered.T @ y_centered)
    bias = (y_mean - x_mean @ weight).squeeze(0)
    return weight.astype(np.float32), bias.astype(np.float32)


def apply_ridge(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return x @ weight + bias


def r2_per_dim(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    mean = np.mean(y_true, axis=0, keepdims=True)
    ss_tot = np.sum((y_true - mean) ** 2, axis=0)
    out = np.zeros_like(ss_res, dtype=np.float64)
    valid = ss_tot > 1e-8
    out[valid] = 1.0 - ss_res[valid] / ss_tot[valid]
    return out


def summarize_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    per_gene_r2 = r2_per_dim(y_true, y_pred)
    mse = float(np.mean((y_true - y_pred) ** 2))
    return {
        "mse": mse,
        "r2_mean": float(np.mean(per_gene_r2)),
        "r2_median": float(np.median(per_gene_r2)),
        "r2_p25": float(np.percentile(per_gene_r2, 25)),
        "r2_p75": float(np.percentile(per_gene_r2, 75)),
    }


def summarize_metric_list(items: list[dict[str, float]]) -> dict[str, float]:
    keys = items[0].keys()
    out: dict[str, float] = {}
    for key in keys:
        values = np.asarray([item[key] for item in items], dtype=np.float64)
        out[f"{key}_mean"] = float(values.mean())
        out[f"{key}_std"] = float(values.std())
    return out


def fit_mlp(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    args,
) -> tuple[MeanGeneMLP, dict[str, float]]:
    device = torch.device(args.device)
    model = MeanGeneMLP(train_x.shape[1], args.mlp_hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.mlp_lr, weight_decay=args.mlp_weight_decay)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=min(args.mlp_batch_size, len(train_x)),
        shuffle=True,
    )
    val_x_t = torch.from_numpy(val_x).to(device)
    val_y_t = torch.from_numpy(val_y).to(device)
    best_state = None
    best_val = float("inf")
    best_epoch = -1

    for epoch in range(args.mlp_epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(val_x_t)
            val_loss = float(loss_fn(val_pred, val_y_t).item())
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    return model, {"best_val_mse": best_val, "best_epoch": best_epoch}


def predict_mlp(model: MeanGeneMLP, x: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(x).to(device))
    return pred.cpu().numpy().astype(np.float32)


def build_pair_folds(pair_ids: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    unique_pairs = np.unique(pair_ids)
    n_folds = max(2, min(n_folds, len(unique_pairs)))
    perm = np.random.default_rng(seed).permutation(unique_pairs)
    raw_folds = [fold for fold in np.array_split(perm, n_folds) if len(fold) > 0]
    return [np.isin(pair_ids, fold_pairs) for fold_pairs in raw_folds]


def main():
    args = parse_args()
    metrics = {
        "task": "local_region_mean_gene_next_step",
        "region_definition": "fixed_local_only_spatial_anchor_nearest_neighbors",
        "n_hvg": args.n_hvg,
        "region_sizes": list(args.region_sizes),
        "dt_minutes": args.dt_minutes,
        "time_window_minutes": args.time_window_minutes,
        "samples_per_pair": args.samples_per_pair,
        "n_folds": args.n_folds,
        "results_by_region_size": {},
    }

    all_ds = build_dataset(args, split="all")
    for region_size in args.region_sizes:
        batch = collect_region_mean_pairs(all_ds, region_size, args.max_samples)
        fold_masks = build_pair_folds(batch.pair_id, args.n_folds, args.seed)
        persistence_scores = []
        ridge_scores = []
        mlp_scores = []
        mlp_train_stats = []
        fold_sizes = []
        for test_mask in fold_masks:
            train_mask = ~test_mask
            train_x = batch.x[train_mask]
            train_y = batch.y[train_mask]
            test_x = batch.x[test_mask]
            test_y = batch.y[test_mask]
            if len(train_x) == 0 or len(test_x) == 0:
                continue
            fold_sizes.append(
                {
                    "train_samples": float(len(train_x)),
                    "test_samples": float(len(test_x)),
                    "train_pairs": float(len(np.unique(batch.pair_id[train_mask]))),
                    "test_pairs": float(len(np.unique(batch.pair_id[test_mask]))),
                }
            )

            persistence_pred = test_x.copy()
            persistence_scores.append(summarize_predictions(test_y, persistence_pred))

            ridge_weight, ridge_bias = fit_ridge(train_x, train_y, args.ridge_alpha)
            ridge_pred = apply_ridge(test_x, ridge_weight, ridge_bias)
            ridge_scores.append(summarize_predictions(test_y, ridge_pred))

            mlp_model, mlp_train_metrics = fit_mlp(train_x, train_y, test_x, test_y, args)
            mlp_pred = predict_mlp(mlp_model, test_x, args.device)
            mlp_scores.append(summarize_predictions(test_y, mlp_pred))
            mlp_train_stats.append(mlp_train_metrics)

        if not persistence_scores:
            raise RuntimeError(f"No valid folds produced for region_size={region_size}")
        region_metrics = {
            "n_samples": int(len(batch.x)),
            "n_unique_pairs": int(len(np.unique(batch.pair_id))),
            "current_count_mean": float(batch.current_count.mean()),
            "future_count_mean": float(batch.future_count.mean()),
            "fold_sizes": summarize_metric_list(fold_sizes),
            "persistence": summarize_metric_list(persistence_scores),
            "ridge": summarize_metric_list(ridge_scores),
            "mlp": summarize_metric_list(mlp_scores),
            "mlp_train": summarize_metric_list(mlp_train_stats),
        }
        metrics["results_by_region_size"][str(region_size)] = region_metrics

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
