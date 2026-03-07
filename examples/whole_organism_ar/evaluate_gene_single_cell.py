#!/usr/bin/env python3
"""Evaluate the single-cell gene+time control baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from examples.whole_organism_ar.train_gene_context import compute_metrics  # noqa: E402
from src.branching_flows.gene_context import SingleCellGeneTimeModel  # noqa: E402
from src.data.gene_context_dataset import (  # noqa: E402
    GeneContextDataset,
    collate_gene_context,
)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate the single-cell gene+time control baseline.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--h5ad_path",
        default="dataset/processed/nema_extended_large2025.h5ad",
    )
    p.add_argument("--split", choices=["train", "val", "all"], default="val")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default="result/gene_context/evaluation_single_cell.json")
    return p.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    cfg = checkpoint["config"]

    dataset = GeneContextDataset(
        h5ad_path=args.h5ad_path,
        n_hvg=cfg["n_hvg"],
        context_size=cfg["context_size"],
        dt_minutes=cfg["dt_minutes"],
        time_window_minutes=cfg["time_window_minutes"],
        samples_per_pair=max(1, cfg["samples_per_pair"] // 2),
        min_cells_per_window=cfg["min_cells_per_window"],
        split=args.split,
        random_seed=cfg["seed"] + 2000,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_gene_context,
    )

    model = SingleCellGeneTimeModel(
        gene_dim=checkpoint["gene_dim"],
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
    ).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    totals: dict[str, float] = {}
    n_batches = 0
    split_tp = split_fp = split_fn = 0.0
    del_tp = del_fp = del_fn = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            output = model(
                genes=batch["genes"],
                time=batch["time"],
                future_time=batch["future_time"],
                token_times=batch["token_times"],
                valid_mask=batch["valid_mask"],
            )
            _, metrics = compute_metrics(output, batch, cfg["split_weight"], cfg["del_weight"])
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value
            n_batches += 1

            valid = batch["valid_mask"]
            split_pred = torch.sigmoid(output.split_logits[valid]) >= 0.5
            split_true = batch["split_target"][valid] >= 0.5
            del_pred = torch.sigmoid(output.del_logits[valid]) >= 0.5
            del_true = batch["del_target"][valid] >= 0.5

            split_tp += float((split_pred & split_true).sum().item())
            split_fp += float((split_pred & ~split_true).sum().item())
            split_fn += float((~split_pred & split_true).sum().item())
            del_tp += float((del_pred & del_true).sum().item())
            del_fp += float((del_pred & ~del_true).sum().item())
            del_fn += float((~del_pred & del_true).sum().item())

    results = {key: value / max(1, n_batches) for key, value in totals.items()}
    results["split_precision"] = split_tp / max(1.0, split_tp + split_fp)
    results["split_recall"] = split_tp / max(1.0, split_tp + split_fn)
    results["del_precision"] = del_tp / max(1.0, del_tp + del_fp)
    results["del_recall"] = del_tp / max(1.0, del_tp + del_fn)
    results["split"] = args.split
    results["checkpoint"] = args.checkpoint

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
