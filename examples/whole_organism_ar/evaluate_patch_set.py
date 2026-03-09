#!/usr/bin/env python3
"""Evaluate the patch-to-patch set prediction baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from examples.whole_organism_ar.train_patch_set import compute_patch_set_metrics  # noqa: E402
from src.branching_flows.gene_context import (  # noqa: E402
    MultiCellPatchSetModel,
    SingleCellPatchSetModel,
)
from src.data.gene_context_dataset import PatchSetDataset, collate_patch_set  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate the patch-to-patch set baseline.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--h5ad_path", default="dataset/processed/nema_extended_large2025.h5ad")
    p.add_argument("--split", choices=["train", "val", "all"], default="val")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default="result/gene_context/evaluation_patch_set.json")
    p.add_argument("--context_ablation", choices=["full", "anchor_only"], default="full")
    return p.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    cfg = checkpoint["config"]

    dataset = PatchSetDataset(
        h5ad_path=args.h5ad_path,
        n_hvg=cfg["n_hvg"],
        context_size=cfg["context_size"],
        global_context_size=cfg.get("global_context_size"),
        dt_minutes=cfg["dt_minutes"],
        time_window_minutes=cfg["time_window_minutes"],
        samples_per_pair=cfg.get("val_samples_per_pair") or max(1, cfg["samples_per_pair"] // 2),
        min_cells_per_window=cfg["min_cells_per_window"],
        sampling_strategy=cfg.get("sampling_strategy", "spatial_anchor"),
        min_spatial_cells_per_window=cfg.get("min_spatial_cells_per_window", 8),
        spatial_neighbor_pool_size=cfg.get("spatial_neighbor_pool_size"),
        delete_target_mode=cfg.get("delete_target_mode", "strict"),
        min_event_positive=cfg.get("min_event_positive", 0),
        min_anchor_event_positive=cfg.get("min_anchor_event_positive", 0),
        min_split_positive=cfg.get("min_split_positive", 0),
        min_del_positive=cfg.get("min_del_positive", 0),
        min_anchor_split_positive=cfg.get("min_anchor_split_positive", 0),
        min_anchor_del_positive=cfg.get("min_anchor_del_positive", 0),
        split=args.split,
        val_fraction=cfg.get("val_fraction", 0.2),
        random_seed=cfg["seed"] + 2000,
    )
    if not dataset.time_pairs:
        raise ValueError("Evaluation dataset is empty after filtering.")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_patch_set)

    if checkpoint.get("model_type", "multi_cell") == "single_cell":
        model = SingleCellPatchSetModel(
            gene_dim=checkpoint["gene_dim"],
            context_size=cfg["context_size"],
            d_model=cfg["d_model"],
            n_layers=cfg["n_layers"],
        ).to(args.device)
    else:
        model = MultiCellPatchSetModel(
            gene_dim=checkpoint["gene_dim"],
            context_size=cfg["context_size"],
            d_model=cfg["d_model"],
            n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"],
            head_dim=cfg["head_dim"],
            use_pairwise_spatial_bias=cfg.get("pairwise_spatial_bias", False),
        ).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    totals: dict[str, float] = {}
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            if args.context_ablation == "anchor_only":
                batch["current_valid_mask"] = batch["current_anchor_mask"] & batch["current_valid_mask"]
            _, metrics = compute_patch_set_metrics(
                model,
                batch,
                cfg["latent_weight"],
                cfg["size_weight"],
                cfg["mean_weight"],
                cfg.get("spatial_input_mode", "relative_position"),
            )
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value
            n_batches += 1

    results = {key: value / max(1, n_batches) for key, value in totals.items()}
    results["checkpoint"] = args.checkpoint
    results["context_ablation"] = args.context_ablation
    results["model_type"] = checkpoint.get("model_type", "multi_cell")
    results["spatial_input_mode"] = cfg.get("spatial_input_mode", "relative_position")
    results["pairwise_spatial_bias"] = cfg.get("pairwise_spatial_bias", False)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
