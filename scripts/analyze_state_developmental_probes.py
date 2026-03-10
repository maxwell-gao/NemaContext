#!/usr/bin/env python3
"""Probe whether state latents predict future developmental features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from examples.whole_organism_ar.train_gene_context import EVENT_SUBSET_THRESHOLDS, resolve_event_filters  # noqa: E402
from examples.whole_organism_ar.train_masked_state_views import MaskedStateViewModel  # noqa: E402
from examples.whole_organism_ar.train_state_views import StateViewModel  # noqa: E402
from src.data.gene_context_dataset import MultiViewPatchStateDataset, collate_multi_view_patch_state  # noqa: E402


FOUNDERS = ("AB", "MS", "E", "C", "D", "P4", "Z", "UNK")


def parse_args():
    p = argparse.ArgumentParser(description="Probe future developmental features from state latents.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=["train", "val", "all"], default="all")
    p.add_argument("--samples_per_pair_override", type=int, default=8)
    p.add_argument("--event_subset_override", choices=["none", *sorted(EVENT_SUBSET_THRESHOLDS)], default="none")
    p.add_argument("--top_cell_types", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_json", required=True)
    return p.parse_args()


def founder_from_lineage(lineage: str) -> str:
    for founder in FOUNDERS[:-1]:
        if lineage.startswith(founder):
            return founder
    return "UNK"


def composition_vector(values: list[str], vocab: list[str]) -> np.ndarray:
    out = np.zeros(len(vocab), dtype=np.float32)
    if not values:
        return out
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    total = float(len(values))
    for i, token in enumerate(vocab):
        out[i] = counts.get(token, 0) / total
    return out


def dominant_label(values: list[str]) -> str:
    if not values:
        return "UNK"
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]


def fit_linear_multioutput(train_x: np.ndarray, train_y: np.ndarray) -> np.ndarray:
    design = np.concatenate([train_x, np.ones((len(train_x), 1), dtype=train_x.dtype)], axis=1)
    w, *_ = np.linalg.lstsq(design, train_y, rcond=None)
    return w


def predict_linear(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    design = np.concatenate([x, np.ones((len(x), 1), dtype=x.dtype)], axis=1)
    return design @ w


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true, axis=0, keepdims=True)) ** 2))
    if ss_tot <= 1e-8:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def per_feature_r2(y_true: np.ndarray, y_pred: np.ndarray) -> list[float]:
    scores = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
        scores.append(0.0 if ss_tot <= 1e-8 else float(1.0 - ss_res / ss_tot))
    return scores


def nearest_centroid_accuracy(train_x: np.ndarray, train_labels: list[str], test_x: np.ndarray, test_labels: list[str]) -> float:
    centroids: dict[str, np.ndarray] = {}
    for label in sorted(set(train_labels)):
        mask = np.asarray(train_labels) == label
        centroids[label] = train_x[mask].mean(axis=0)
    correct = 0
    for x, label in zip(test_x, test_labels, strict=True):
        pred = min(centroids, key=lambda key: float(np.sum((x - centroids[key]) ** 2)))
        correct += int(pred == label)
    return correct / max(1, len(test_labels))


def main():
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config = ckpt["config"]

    class _Args:
        pass

    ns = _Args()
    for key, value in config.items():
        setattr(ns, key, value)
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
        samples_per_pair=args.samples_per_pair_override,
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

    future_cell_types_all: list[str] = []
    samples = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        future_indices = item["future_view_0_indices"].numpy()
        future_cell_types = [str(x) for x in dataset.adata_obs_cell_type[future_indices]]
        future_cell_types_all.extend(future_cell_types)
        samples.append((item, future_indices, future_cell_types))

    cell_type_counts: dict[str, int] = {}
    for ct in future_cell_types_all:
        if ct == "nan":
            continue
        cell_type_counts[ct] = cell_type_counts.get(ct, 0) + 1
    top_cell_types = [ct for ct, _n in sorted(cell_type_counts.items(), key=lambda kv: (-kv[1], kv[0]))[: args.top_cell_types]]

    latents = []
    founder_comp = []
    celltype_comp = []
    depth_stats = []
    spatial_extent = []
    dom_founder = []
    dom_cell_type = []

    batch_items: list[dict[str, torch.Tensor]] = []
    metas = []

    def flush():
        nonlocal batch_items, metas
        if not batch_items:
            return
        batch = collate_multi_view_patch_state(batch_items)
        batch = {k: v.to(args.device) for k, v in batch.items()}
        with torch.no_grad():
            z_a, _ = model.encode_view(batch, "current_view_0")
            z_b, _ = model.encode_view(batch, "current_view_1")
            z = 0.5 * (z_a + z_b)
        z = z.cpu().numpy()
        for latent, meta in zip(z, metas, strict=True):
            latents.append(latent)
            founder_comp.append(meta["founder_comp"])
            celltype_comp.append(meta["celltype_comp"])
            depth_stats.append(meta["depth_stats"])
            spatial_extent.append(meta["spatial_extent"])
            dom_founder.append(meta["dom_founder"])
            dom_cell_type.append(meta["dom_cell_type"])
        batch_items = []
        metas = []

    for item, future_indices, future_cell_types in samples:
        future_lineages = [str(x) for x in dataset.lineages[future_indices]]
        future_founders = [founder_from_lineage(x) for x in future_lineages]
        founder_comp_vec = composition_vector(future_founders, list(FOUNDERS))
        celltype_comp_vec = composition_vector(future_cell_types, top_cell_types)

        depths = np.asarray(dataset.lineage_depths[future_indices], dtype=np.float32)
        valid_depths = depths[depths >= 0]
        if len(valid_depths) == 0:
            depth_stats_vec = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        else:
            depth_stats_vec = np.array(
                [float(valid_depths.mean()), float(valid_depths.std()), float(len(valid_depths) / max(1, len(depths)))],
                dtype=np.float32,
            )

        valid_future_spatial = dataset.valid_spatial[future_indices]
        if np.any(valid_future_spatial):
            coords = dataset.spatial_coords[future_indices[valid_future_spatial]]
            centered = coords - coords.mean(axis=0, keepdims=True)
            spatial_extent_vec = np.concatenate(
                [coords.std(axis=0), [float(np.linalg.norm(centered, axis=1).mean())]],
                axis=0,
            ).astype(np.float32)
        else:
            spatial_extent_vec = np.zeros(4, dtype=np.float32)

        metas.append(
            {
                "founder_comp": founder_comp_vec,
                "celltype_comp": celltype_comp_vec,
                "depth_stats": depth_stats_vec,
                "spatial_extent": spatial_extent_vec,
                "dom_founder": dominant_label(future_founders),
                "dom_cell_type": dominant_label(future_cell_types),
            }
        )
        batch_items.append(item)
        if len(batch_items) >= 16:
            flush()
    flush()

    x = np.asarray(latents, dtype=np.float32)
    founder_comp = np.asarray(founder_comp, dtype=np.float32)
    celltype_comp = np.asarray(celltype_comp, dtype=np.float32)
    depth_stats = np.asarray(depth_stats, dtype=np.float32)
    spatial_extent = np.asarray(spatial_extent, dtype=np.float32)
    n = len(x)
    split = max(1, int(round(0.8 * n)))

    train_x, test_x = x[:split], x[split:]
    metrics = {
        "n_samples": n,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "top_cell_types": top_cell_types,
    }

    def add_probe(name: str, target: np.ndarray):
        train_y, test_y = target[:split], target[split:]
        w = fit_linear_multioutput(train_x, train_y)
        pred = predict_linear(test_x, w)
        metrics[name] = {
            "r2": r2_score(test_y, pred),
            "per_feature_r2": per_feature_r2(test_y, pred),
        }

    add_probe("future_founder_composition", founder_comp)
    add_probe("future_celltype_composition", celltype_comp)
    add_probe("future_lineage_depth_stats", depth_stats)
    add_probe("future_spatial_extent", spatial_extent)

    metrics["future_dominant_founder"] = {
        "nearest_centroid_accuracy": nearest_centroid_accuracy(
            train_x, dom_founder[:split], test_x, dom_founder[split:]
        )
    }
    metrics["future_dominant_cell_type"] = {
        "nearest_centroid_accuracy": nearest_centroid_accuracy(
            train_x, dom_cell_type[:split], test_x, dom_cell_type[split:]
        )
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
