#!/usr/bin/env python3
"""Evaluate embryo-state probe predictors on the full split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from examples.whole_organism_ar.train_gene_context import (  # noqa: E402
    EVENT_SUBSET_THRESHOLDS,
    resolve_event_filters,
)
from examples.whole_organism_ar.train_embryo_state import stack_view_tensor  # noqa: E402
from src.branching_flows.gene_context import EmbryoStateModel  # noqa: E402
from src.data.gene_context_dataset import EmbryoViewDataset, collate_embryo_view  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate embryo-state predictor on full split.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=["train", "val", "all"], default="val")
    p.add_argument("--samples_per_pair_override", type=int, default=8)
    p.add_argument("--event_subset_override", choices=["none", *sorted(EVENT_SUBSET_THRESHOLDS)], default="none")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_json", required=True)
    return p.parse_args()


def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - y_true.mean(dim=0, keepdim=True)) ** 2)
    if float(ss_tot.item()) <= 1e-8:
        return 0.0
    return float((1.0 - ss_res / ss_tot).item())


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
        top_cell_types=ckpt.get("celltype_dim", len(ckpt.get("top_cell_types", []))),
        **filters,
    )

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
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(args.device)
    model.eval()

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_embryo_view)
    preds = {
        "founder": [],
        "celltype": [],
        "depth": [],
        "spatial": [],
        "split": [],
    }
    trues = {
        "founder": [],
        "celltype": [],
        "depth": [],
        "spatial": [],
        "split": [],
    }

    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v.to(args.device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            n_views = int(batch["views_per_embryo"][0].item())
            genes = stack_view_tensor(batch, "genes", n_views)
            context_role = stack_view_tensor(batch, "context_role", n_views)
            relative_position = stack_view_tensor(batch, "relative_position", n_views)
            token_times = stack_view_tensor(batch, "token_times", n_views)
            valid_mask = stack_view_tensor(batch, "valid_mask", n_views)
            anchor_mask = stack_view_tensor(batch, "anchor_mask", n_views)
            time = stack_view_tensor(batch, "time", n_views)

            out = model(
                genes=genes,
                time=time,
                token_times=token_times,
                valid_mask=valid_mask,
                anchor_mask=anchor_mask,
                context_role=context_role,
                relative_position=relative_position,
            )
            preds["founder"].append(out.future_founder_composition.cpu())
            preds["celltype"].append(out.future_celltype_composition.cpu())
            preds["depth"].append(out.future_lineage_depth_stats.cpu())
            preds["spatial"].append(out.future_spatial_extent.cpu())
            preds["split"].append(out.future_split_fraction.cpu())
            trues["founder"].append(batch["future_founder_composition"].cpu())
            trues["celltype"].append(batch["future_celltype_composition"].cpu())
            trues["depth"].append(batch["future_lineage_depth_stats"].cpu())
            trues["spatial"].append(batch["future_spatial_extent"].cpu())
            trues["split"].append(batch["future_split_fraction"].cpu())

    results = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "n_samples": sum(t.shape[0] for t in trues["founder"]),
        "top_cell_types": ckpt.get("top_cell_types", []),
    }
    for key in preds:
        y_pred = torch.cat(preds[key], dim=0)
        y_true = torch.cat(trues[key], dim=0)
        results[f"{key}_r2"] = r2_score(y_true, y_pred)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
