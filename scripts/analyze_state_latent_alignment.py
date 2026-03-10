#!/usr/bin/env python3
"""Analyze biological alignment of learned multi-view state latents."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from examples.whole_organism_ar.train_gene_context import EVENT_SUBSET_THRESHOLDS, resolve_event_filters  # noqa: E402
from examples.whole_organism_ar.train_masked_state_views import MaskedStateViewModel  # noqa: E402
from examples.whole_organism_ar.train_state_views import StateViewModel  # noqa: E402
from src.data.gene_context_dataset import MultiViewPatchStateDataset, collate_multi_view_patch_state  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Analyze latent biological alignment for state-view models.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=["train", "val", "all"], default="val")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--samples_per_pair_override", type=int, default=None)
    p.add_argument("--event_subset_override", choices=["none", *sorted(EVENT_SUBSET_THRESHOLDS)], default=None)
    p.add_argument("--output_json", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--output_prefix", required=True)
    return p.parse_args()


def founder_from_lineage(lineage: str) -> str:
    for founder in ("AB", "MS", "E", "C", "D", "P4", "Z"):
        if lineage.startswith(founder):
            return founder
    return "UNK"


def dominant_label(values: list[str]) -> str:
    if not values:
        return "UNK"
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]


def pca_2d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = x - x.mean(axis=0, keepdims=True)
    u, s, _vh = np.linalg.svd(centered, full_matrices=False)
    scores = u[:, :2] * s[:2]
    explained = (s**2) / max(np.sum(s**2), 1e-8)
    return scores, explained[:2]


def pairwise_sq_dists(x: np.ndarray) -> np.ndarray:
    x2 = np.sum(x * x, axis=1, keepdims=True)
    d = x2 + x2.T - 2.0 * (x @ x.T)
    return np.maximum(d, 0.0)


def knn_purity(x: np.ndarray, labels: list[str], k: int = 5) -> float:
    if len(x) <= 1:
        return 0.0
    d = pairwise_sq_dists(x)
    np.fill_diagonal(d, np.inf)
    labels_arr = np.asarray(labels)
    k = min(k, len(x) - 1)
    nn = np.argpartition(d, kth=k - 1, axis=1)[:, :k]
    purity = []
    for i in range(len(x)):
        purity.append(float(np.mean(labels_arr[nn[i]] == labels_arr[i])))
    return float(np.mean(purity))


def linear_probe_r2(x: np.ndarray, y: np.ndarray) -> float:
    n = len(x)
    if n < 4:
        return 0.0
    split = max(1, int(round(0.8 * n)))
    train_x = np.concatenate([x[:split], np.ones((split, 1), dtype=x.dtype)], axis=1)
    test_x = np.concatenate([x[split:], np.ones((n - split, 1), dtype=x.dtype)], axis=1)
    train_y = y[:split]
    test_y = y[split:]
    if len(test_y) == 0:
        return 0.0
    w, *_ = np.linalg.lstsq(train_x, train_y, rcond=None)
    pred = test_x @ w
    ss_res = float(np.sum((test_y - pred) ** 2))
    ss_tot = float(np.sum((test_y - np.mean(test_y)) ** 2))
    if ss_tot <= 1e-8:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(x, y) / denom)


def cosine_loss(x: np.ndarray, y: np.ndarray) -> float:
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 1e-8:
        return 1.0
    return float(1.0 - np.dot(x, y) / denom)


def region_labels(coords: np.ndarray, medians: np.ndarray) -> list[str]:
    labels = []
    for coord in coords:
        bits = ["H" if coord[i] >= medians[i] else "L" for i in range(3)]
        labels.append("".join(bits))
    return labels


def scatter_plot(scores: np.ndarray, values, title: str, output_path: Path, categorical: bool):
    plt.figure(figsize=(6, 5))
    if categorical:
        cats = sorted(set(values))
        cmap = plt.get_cmap("tab10")
        for idx, cat in enumerate(cats):
            mask = np.asarray(values) == cat
            plt.scatter(scores[mask, 0], scores[mask, 1], s=14, alpha=0.8, label=str(cat), color=cmap(idx % 10))
        plt.legend(fontsize=8, frameon=False)
    else:
        sc = plt.scatter(scores[:, 0], scores[:, 1], c=values, s=14, alpha=0.85, cmap="viridis")
        plt.colorbar(sc, fraction=0.046, pad=0.04)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main():
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config = ckpt["config"]

    class _Args:
        pass

    ns = _Args()
    for key, value in config.items():
        setattr(ns, key, value)
    if args.event_subset_override is not None:
        ns.event_subset = args.event_subset_override
        ns.val_event_subset = args.event_subset_override
        for key in (
            "min_event_positive",
            "min_anchor_event_positive",
            "min_split_positive",
            "min_del_positive",
            "min_anchor_split_positive",
            "min_anchor_del_positive",
            "val_min_event_positive",
            "val_min_anchor_event_positive",
            "val_min_split_positive",
            "val_min_del_positive",
            "val_min_anchor_split_positive",
            "val_min_anchor_del_positive",
        ):
            setattr(ns, key, 0)
    filters = resolve_event_filters(ns, prefix="val_" if args.split == "val" else "")

    dataset = MultiViewPatchStateDataset(
        h5ad_path=config["h5ad_path"],
        n_hvg=config["n_hvg"],
        context_size=config["context_size"],
        global_context_size=config["global_context_size"],
        dt_minutes=config["dt_minutes"],
        time_window_minutes=config["time_window_minutes"],
        samples_per_pair=args.samples_per_pair_override or config["val_samples_per_pair"] or max(1, config["samples_per_pair"] // 2),
        min_cells_per_window=config["min_cells_per_window"],
        sampling_strategy=config["sampling_strategy"],
        min_spatial_cells_per_window=config["min_spatial_cells_per_window"],
        spatial_neighbor_pool_size=config["spatial_neighbor_pool_size"],
        delete_target_mode=config["delete_target_mode"],
        views_per_state=config["views_per_state"],
        future_views_per_state=config["future_views_per_state"],
        split=args.split,
        val_fraction=config["val_fraction"],
        random_seed=config["seed"] + 1000,
        **filters,
    )
    if len(dataset) == 0:
        raise ValueError("State-view dataset is empty.")

    model_cls = MaskedStateViewModel if "masked_view_predictor.0.weight" in ckpt["model_state_dict"] else StateViewModel
    model = model_cls(
        gene_dim=ckpt["gene_dim"],
        context_size=config["context_size"],
        model_type=config["model_type"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        head_dim=config["head_dim"],
        use_pairwise_spatial_bias=config["pairwise_spatial_bias"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(args.device)
    model.eval()

    valid_spatial = dataset.spatial_coords[dataset.valid_spatial]
    medians = np.median(valid_spatial, axis=0)

    rows = []
    batch_items: list[dict[str, torch.Tensor]] = []
    metadata_batch = []

    def flush_batch():
        nonlocal batch_items, metadata_batch, rows
        if not batch_items:
            return
        batch = collate_multi_view_patch_state(batch_items)
        batch = {k: v.to(args.device) for k, v in batch.items()}
        with torch.no_grad():
            z_a, _ = model.encode_view(batch, "current_view_0")
            z_b, _ = model.encode_view(batch, "current_view_1")
            z_f, _ = model.encode_view(batch, "future_view_0")
            z_state = 0.5 * (z_a + z_b)
            if hasattr(model, "predictor"):
                pred_future = model.predictor(z_a)
            else:
                pred_future = model.future_predictor(z_a)
        z_state = z_state.cpu().numpy()
        z_future = z_f.cpu().numpy()
        pred_future = pred_future.cpu().numpy()
        for meta, state_latent, future_latent, pred_latent in zip(metadata_batch, z_state, z_future, pred_future, strict=True):
            row = dict(meta)
            row["latent"] = state_latent
            row["future_latent"] = future_latent
            row["pred_future_latent"] = pred_latent
            row["future_latent_cosine"] = cosine_loss(
                pred_latent.astype(np.float64),
                future_latent.astype(np.float64),
            )
            rows.append(row)
        batch_items = []
        metadata_batch = []

    for idx in range(len(dataset)):
        item = dataset[idx]
        current_indices = item["current_view_0_indices"].numpy()
        future_indices = item["future_view_0_indices"].numpy()
        current_lineages = [str(x) for x in dataset.lineages[current_indices]]
        future_lineages = [str(x) for x in dataset.lineages[future_indices]]
        founder = dominant_label([founder_from_lineage(x) for x in current_lineages])
        anchor_coord = dataset.spatial_coords[int(current_indices[0])]
        region = region_labels(anchor_coord[None, :], medians)[0]
        future_split = float(dataset._compute_split_fraction(np.asarray(future_lineages, dtype=object)))
        metadata_batch.append(
            {
                "sample_idx": idx,
                "current_time": float(item["current_view_0_time"].item()),
                "future_time": float(item["future_view_0_time"].item()),
                "anchor_x": float(anchor_coord[0]),
                "anchor_y": float(anchor_coord[1]),
                "anchor_z": float(anchor_coord[2]),
                "dominant_founder": founder,
                "spatial_region": region,
                "future_split_fraction": future_split,
                "current_patch_cells": int(len(current_indices)),
                "future_patch_cells": int(len(future_indices)),
            }
        )
        batch_items.append(item)
        if len(batch_items) >= args.batch_size:
            flush_batch()
    flush_batch()

    latents = np.stack([row["latent"] for row in rows], axis=0)
    scores, explained = pca_2d(latents)
    current_time = np.asarray([row["current_time"] for row in rows], dtype=np.float64)
    future_split_fraction = np.asarray([row["future_split_fraction"] for row in rows], dtype=np.float64)
    founders = [row["dominant_founder"] for row in rows]
    regions = [row["spatial_region"] for row in rows]
    future_latent_cosine = np.asarray([row["future_latent_cosine"] for row in rows], dtype=np.float64)

    for row, score in zip(rows, scores, strict=True):
        row["pc1"] = float(score[0])
        row["pc2"] = float(score[1])

    metrics = {
        "n_samples": len(rows),
        "checkpoint": args.checkpoint,
        "split": args.split,
        "time_alignment": {
            "pc1_corr": pearson_corr(scores[:, 0], current_time),
            "pc2_corr": pearson_corr(scores[:, 1], current_time),
            "latent_to_time_r2": linear_probe_r2(latents, current_time),
        },
        "lineage_alignment": {
            "founder_knn_purity_k5": knn_purity(latents, founders, k=5),
            "n_founders": len(set(founders)),
        },
        "spatial_alignment": {
            "region_knn_purity_k5": knn_purity(latents, regions, k=5),
            "n_regions": len(set(regions)),
        },
        "future_split_alignment": {
            "pc1_corr": pearson_corr(scores[:, 0], future_split_fraction),
            "pc2_corr": pearson_corr(scores[:, 1], future_split_fraction),
            "latent_to_future_split_r2": linear_probe_r2(latents, future_split_fraction),
        },
        "future_prediction": {
            "mean_future_latent_cosine_loss": float(np.mean(future_latent_cosine)),
            "future_latent_cosine_std": float(np.std(future_latent_cosine)),
        },
        "pca_explained_variance_ratio": [float(x) for x in explained],
    }

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_prefix = Path(args.output_prefix)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    serializable_rows = []
    for row in rows:
        serializable_rows.append({k: v for k, v in row.items() if k not in {"latent", "future_latent", "pred_future_latent"}})

    with open(output_json, "w") as f:
        json.dump({"metrics": metrics, "samples": serializable_rows[:50]}, f, indent=2)

    fieldnames = [
        "sample_idx",
        "current_time",
        "future_time",
        "anchor_x",
        "anchor_y",
        "anchor_z",
        "dominant_founder",
        "spatial_region",
        "future_split_fraction",
        "current_patch_cells",
        "future_patch_cells",
        "future_latent_cosine",
        "pc1",
        "pc2",
    ]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in serializable_rows:
            writer.writerow({k: row[k] for k in fieldnames})

    scatter_plot(scores, current_time, "State Latent PCA colored by time", output_prefix.with_name(output_prefix.name + "_time.png"), categorical=False)
    scatter_plot(scores, founders, "State Latent PCA colored by founder", output_prefix.with_name(output_prefix.name + "_founder.png"), categorical=True)
    scatter_plot(scores, regions, "State Latent PCA colored by spatial region", output_prefix.with_name(output_prefix.name + "_region.png"), categorical=True)
    scatter_plot(scores, future_split_fraction, "State Latent PCA colored by future split fraction", output_prefix.with_name(output_prefix.name + "_future_split.png"), categorical=False)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
