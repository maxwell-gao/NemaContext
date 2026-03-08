#!/usr/bin/env python3
"""Audit event supervision coverage for the gene-context dataset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.data.gene_context_dataset import GeneContextDataset


def _to_builtin(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def summarize_records(records: list[dict[str, float | int]]) -> dict[str, float | int]:
    if not records:
        return {"num_pairs": 0}

    current_counts = np.array([r["current_count"] for r in records], dtype=np.float64)
    future_counts = np.array([r["future_count"] for r in records], dtype=np.float64)
    split_counts = np.array([r["split_positive_count"] for r in records], dtype=np.float64)
    del_counts = np.array([r["del_positive_count"] for r in records], dtype=np.float64)
    anchor_split_counts = np.array(
        [r["anchor_split_positive_count"] for r in records], dtype=np.float64
    )
    anchor_del_counts = np.array(
        [r["anchor_del_positive_count"] for r in records], dtype=np.float64
    )
    match_rates = np.array([r["match_rate"] for r in records], dtype=np.float64)

    summary = {
        "num_pairs": len(records),
        "mean_current_count": float(current_counts.mean()),
        "mean_future_count": float(future_counts.mean()),
        "total_split_positive": int(split_counts.sum()),
        "total_del_positive": int(del_counts.sum()),
        "pairs_with_split_positive": int((split_counts > 0).sum()),
        "pairs_with_del_positive": int((del_counts > 0).sum()),
        "pairs_with_any_event_positive": int(((split_counts + del_counts) > 0).sum()),
        "mean_match_rate": float(match_rates.mean()),
        "mean_split_positive_rate": float(
            np.mean([r["split_positive_rate"] for r in records])
        ),
        "mean_del_positive_rate": float(np.mean([r["del_positive_rate"] for r in records])),
        "mean_anchor_split_positive_rate": float(
            np.mean([r["anchor_split_positive_rate"] for r in records])
        ),
        "mean_anchor_del_positive_rate": float(
            np.mean([r["anchor_del_positive_rate"] for r in records])
        ),
        "total_anchor_split_positive": int(anchor_split_counts.sum()),
        "total_anchor_del_positive": int(anchor_del_counts.sum()),
    }
    return summary


def top_records(
    records: list[dict[str, float | int]],
    key: str,
    limit: int,
) -> list[dict[str, float | int]]:
    sorted_records = sorted(records, key=lambda r: (r[key], r["current_center_min"]), reverse=True)
    return [{k: _to_builtin(v) for k, v in rec.items()} for rec in sorted_records[:limit]]


def parse_args():
    p = argparse.ArgumentParser(description="Audit split/delete supervision coverage.")
    p.add_argument(
        "--h5ad_path",
        default="dataset/processed/nema_extended_large2025.h5ad",
    )
    p.add_argument("--n_hvg", type=int, default=128)
    p.add_argument("--context_size", type=int, default=64)
    p.add_argument("--global_context_size", type=int, default=16)
    p.add_argument("--dt_minutes", type=float, default=20.0)
    p.add_argument("--time_window_minutes", type=float, default=10.0)
    p.add_argument("--samples_per_pair", type=int, default=1)
    p.add_argument("--min_cells_per_window", type=int, default=32)
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument(
        "--sampling_strategy",
        choices=["random_window", "spatial_neighbors", "spatial_anchor"],
        default="spatial_anchor",
    )
    p.add_argument("--min_spatial_cells_per_window", type=int, default=8)
    p.add_argument("--spatial_neighbor_pool_size", type=int, default=None)
    p.add_argument("--min_event_positive", type=int, default=0)
    p.add_argument("--min_anchor_event_positive", type=int, default=0)
    p.add_argument("--min_split_positive", type=int, default=0)
    p.add_argument("--min_del_positive", type=int, default=0)
    p.add_argument("--min_anchor_split_positive", type=int, default=0)
    p.add_argument("--min_anchor_del_positive", type=int, default=0)
    p.add_argument(
        "--delete_target_mode",
        choices=["weak", "strict"],
        default="weak",
    )
    p.add_argument("--split", choices=["train", "val", "all"], default="all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--output_json", default="result/gene_context/audit_summary.json")
    p.add_argument("--output_csv", default="result/gene_context/audit_pairs.csv")
    return p.parse_args()


def main():
    args = parse_args()
    dataset = GeneContextDataset(
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
        min_event_positive=args.min_event_positive,
        min_anchor_event_positive=args.min_anchor_event_positive,
        min_split_positive=args.min_split_positive,
        min_del_positive=args.min_del_positive,
        min_anchor_split_positive=args.min_anchor_split_positive,
        min_anchor_del_positive=args.min_anchor_del_positive,
        delete_target_mode=args.delete_target_mode,
        split=args.split,
        val_fraction=args.val_fraction,
        random_seed=args.seed,
    )

    records = [dataset.summarize_time_pair(i) for i in range(len(dataset.time_pairs))]

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(records[0].keys()) if records else []
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({k: _to_builtin(v) for k, v in record.items()})

    payload = {
        "config": vars(args),
        "summary": summarize_records(records),
        "top_split_pairs": top_records(records, "split_positive_count", args.top_k),
        "top_del_pairs": top_records(records, "del_positive_count", args.top_k),
        "top_anchor_event_pairs": top_records(records, "anchor_event_positive_count", args.top_k),
    }
    with output_json.open("w") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload["summary"], indent=2))
    print(f"wrote {output_json}")
    print(f"wrote {output_csv}")


if __name__ == "__main__":
    main()
