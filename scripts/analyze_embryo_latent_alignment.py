#!/usr/bin/env python3
"""Analyze biological alignment of embryo-level latents."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from examples.whole_organism_ar.train_embryo_state import stack_view_tensor  # noqa: E402
from examples.whole_organism_ar.train_gene_context import (  # noqa: E402
    EVENT_SUBSET_THRESHOLDS,
    resolve_event_filters,
)
from src.branching_flows.gene_context import EmbryoMaskedViewModel, EmbryoStateModel  # noqa: E402
from src.data.gene_context_dataset import EmbryoViewDataset, collate_embryo_view  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Analyze embryo-level latent biological alignment.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=["train", "val", "all"], default="all")
    p.add_argument("--samples_per_pair_override", type=int, default=16)
    p.add_argument("--event_subset_override", choices=["none", *sorted(EVENT_SUBSET_THRESHOLDS)], default="none")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--top_cell_types", type=int, default=8)
    p.add_argument("--output_json", required=True)
    return p.parse_args()


def fit_linear(train_x: np.ndarray, train_y: np.ndarray) -> np.ndarray:
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

    dataset = EmbryoViewDataset(
        h5ad_path=config["h5ad_path"],
        n_hvg=config["n_hvg"],
        context_size=config["context_size"],
        global_context_size=config["global_context_size"],
        dt_minutes=config["dt_minutes"],
        time_window_minutes=config["time_window_minutes"],
        samples_per_pair=args.samples_per_pair_override,
        min_cells_per_window=config["min_cells_per_window"],
        split=args.split,
        val_fraction=config["val_fraction"],
        random_seed=config["seed"] + 1000,
        sampling_strategy=config["sampling_strategy"],
        min_spatial_cells_per_window=config["min_spatial_cells_per_window"],
        spatial_neighbor_pool_size=config["spatial_neighbor_pool_size"],
        delete_target_mode=config["delete_target_mode"],
        views_per_embryo=config["views_per_embryo"],
        top_cell_types=max(
            args.top_cell_types,
            len(ckpt.get("top_cell_types", [])),
        ),
        **filters,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_embryo_view,
    )

    if "masked_view_latent_head.0.weight" in ckpt["model_state_dict"]:
        model = EmbryoMaskedViewModel(
            gene_dim=ckpt["gene_dim"],
            context_size=config["context_size"],
            model_type=config["model_type"],
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            n_layers=config["n_layers"],
            head_dim=config["head_dim"],
            use_pairwise_spatial_bias=config["pairwise_spatial_bias"],
        )
        is_masked = True
    else:
        model = EmbryoStateModel(
            gene_dim=ckpt["gene_dim"],
            context_size=config["context_size"],
            celltype_dim=ckpt["celltype_dim"],
            model_type=config["model_type"],
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            n_layers=config["n_layers"],
            head_dim=config["head_dim"],
            use_pairwise_spatial_bias=config["pairwise_spatial_bias"],
        )
        is_masked = False
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(args.device)
    model.eval()

    latents = []
    current_time = []
    future_founder = []
    future_celltype = []
    future_depth = []
    future_spatial = []
    future_split = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            n_views = int(batch["views_per_embryo"][0].item())
            genes = stack_view_tensor(batch, "genes", n_views)
            context_role = stack_view_tensor(batch, "context_role", n_views)
            relative_position = stack_view_tensor(batch, "relative_position", n_views)
            token_times = stack_view_tensor(batch, "token_times", n_views)
            valid_mask = stack_view_tensor(batch, "valid_mask", n_views)
            anchor_mask = stack_view_tensor(batch, "anchor_mask", n_views)
            time = stack_view_tensor(batch, "time", n_views)

            if is_masked:
                masked_view_mask = torch.zeros(genes.shape[0], n_views, dtype=torch.bool, device=args.device)
                masked_view_mask[:, -1] = True
                out = model(
                    genes=genes,
                    time=time,
                    token_times=token_times,
                    valid_mask=valid_mask,
                    anchor_mask=anchor_mask,
                    masked_view_mask=masked_view_mask,
                    context_role=context_role,
                    relative_position=relative_position,
                )
                z = out.visible_embryo_latent
            else:
                out = model(
                    genes=genes,
                    time=time,
                    token_times=token_times,
                    valid_mask=valid_mask,
                    anchor_mask=anchor_mask,
                    context_role=context_role,
                    relative_position=relative_position,
                )
                z = out.embryo_latent
            latents.append(z.cpu().numpy())
            current_time.append(batch["current_center_min"].cpu().numpy())
            future_founder.append(batch["future_founder_composition"].cpu().numpy())
            future_celltype.append(batch["future_celltype_composition"].cpu().numpy())
            future_depth.append(batch["future_lineage_depth_stats"].cpu().numpy())
            future_spatial.append(batch["future_spatial_extent"].cpu().numpy())
            future_split.append(batch["future_split_fraction"].cpu().numpy())

    x = np.concatenate(latents, axis=0).astype(np.float32)
    time_y = np.concatenate(current_time, axis=0).astype(np.float32)[:, None]
    founder_y = np.concatenate(future_founder, axis=0).astype(np.float32)
    celltype_y = np.concatenate(future_celltype, axis=0).astype(np.float32)
    depth_y = np.concatenate(future_depth, axis=0).astype(np.float32)
    spatial_y = np.concatenate(future_spatial, axis=0).astype(np.float32)
    split_y = np.concatenate(future_split, axis=0).astype(np.float32)

    n = len(x)
    split_idx = max(1, int(round(0.8 * n)))
    train_x, test_x = x[:split_idx], x[split_idx:]

    metrics = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "n_samples": n,
        "is_masked": is_masked,
        "top_cell_types": list(dataset._top_cell_type_vocab),
    }

    def add_probe(name: str, target: np.ndarray):
        w = fit_linear(train_x, target[:split_idx])
        pred = predict_linear(test_x, w)
        metrics[name] = {"r2": r2_score(target[split_idx:], pred)}

    add_probe("latent_to_time", time_y)
    add_probe("future_founder_composition", founder_y)
    add_probe("future_celltype_composition", celltype_y)
    add_probe("future_lineage_depth_stats", depth_y)
    add_probe("future_spatial_extent", spatial_y)
    add_probe("future_split_fraction", split_y)

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(x)
    metrics["pca_explained_variance_ratio"] = pca.explained_variance_ratio_.tolist()
    metrics["pc1_time_corr"] = float(np.corrcoef(pcs[:, 0], time_y[:, 0])[0, 1])
    metrics["pc2_time_corr"] = float(np.corrcoef(pcs[:, 1], time_y[:, 0])[0, 1])

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
