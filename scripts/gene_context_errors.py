#!/usr/bin/env python3
"""Compare per-sample errors between gene-context checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from examples.whole_organism_ar.train_gene_context import compute_metrics
from src.branching_flows.gene_context import GeneContextModel, SingleCellGeneTimeModel
from src.data.gene_context_dataset import GeneContextDataset, collate_gene_context


def parse_args():
    p = argparse.ArgumentParser(description="Compare per-sample errors across checkpoints.")
    p.add_argument("--left_checkpoint", required=True)
    p.add_argument("--right_checkpoint", required=True)
    p.add_argument(
        "--h5ad_path",
        default="dataset/processed/nema_extended_large2025.h5ad",
    )
    p.add_argument("--split", choices=["train", "val", "all"], default="val")
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--left_context_ablation",
        choices=["full", "anchor_only"],
        default="full",
    )
    p.add_argument(
        "--right_context_ablation",
        choices=["full", "anchor_only"],
        default="full",
    )
    p.add_argument("--output_json", default="result/gene_context/error_compare.json")
    p.add_argument("--output_csv", default="result/gene_context/error_compare.csv")
    return p.parse_args()


def load_model(checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint["config"]
    model_type = checkpoint.get("model_type", "multi_cell")

    if model_type == "single_cell":
        model = SingleCellGeneTimeModel(
            gene_dim=checkpoint["gene_dim"],
            d_model=cfg["d_model"],
            n_layers=cfg["n_layers"],
        ).to(device)
    else:
        model = GeneContextModel(
            gene_dim=checkpoint["gene_dim"],
            d_model=cfg["d_model"],
            n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"],
            head_dim=cfg["head_dim"],
            use_pairwise_spatial_bias=cfg.get("pairwise_spatial_bias", False),
        ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return checkpoint, model, model_type


def build_dataset(cfg: dict, h5ad_path: str, split: str):
    return GeneContextDataset(
        h5ad_path=h5ad_path,
        n_hvg=cfg["n_hvg"],
        context_size=cfg["context_size"],
        global_context_size=cfg.get("global_context_size"),
        dt_minutes=cfg["dt_minutes"],
        time_window_minutes=cfg["time_window_minutes"],
        samples_per_pair=cfg.get("val_samples_per_pair") or max(1, cfg["samples_per_pair"] // 2),
        min_cells_per_window=cfg["min_cells_per_window"],
        sampling_strategy=cfg.get("sampling_strategy", "random_window"),
        min_spatial_cells_per_window=cfg.get("min_spatial_cells_per_window", 8),
        spatial_neighbor_pool_size=cfg.get("spatial_neighbor_pool_size"),
        delete_target_mode=cfg.get("delete_target_mode", "weak"),
        supervision_mode=cfg.get("supervision_mode", "anchor_only"),
        local_group_size=cfg.get("local_group_size"),
        min_event_positive=cfg.get("min_event_positive", 0),
        min_anchor_event_positive=cfg.get("min_anchor_event_positive", 0),
        min_split_positive=cfg.get("min_split_positive", 0),
        min_del_positive=cfg.get("min_del_positive", 0),
        min_anchor_split_positive=cfg.get("min_anchor_split_positive", 0),
        min_anchor_del_positive=cfg.get("min_anchor_del_positive", 0),
        split=split,
        val_fraction=cfg.get("val_fraction", 0.2),
        random_seed=cfg["seed"] + 2000,
    )


def run_model(
    model,
    model_type: str,
    batch: dict[str, torch.Tensor],
    cfg: dict,
    context_ablation: str,
):
    model_valid_mask = batch["valid_mask"]
    if context_ablation == "anchor_only":
        model_valid_mask = (
            batch.get("anchor_mask", batch["valid_mask"]) & batch["valid_mask"]
        )
    if model_type == "single_cell":
        output = model(
            genes=batch["genes"],
            time=batch["time"],
            future_time=batch["future_time"],
            token_times=batch["token_times"],
            valid_mask=model_valid_mask,
            relative_position=(
                batch.get("relative_position")
                if cfg.get("spatial_input_mode", "relative_position") == "relative_position"
                else None
            ),
        )
    else:
        output = model(
            genes=batch["genes"],
            time=batch["time"],
            future_time=batch["future_time"],
            token_times=batch["token_times"],
            valid_mask=model_valid_mask,
            context_role=batch.get("context_role"),
            relative_position=(
                batch.get("relative_position")
                if cfg.get("spatial_input_mode", "relative_position") == "relative_position"
                else None
            ),
        )
    _, metrics = compute_metrics(output, batch, cfg["split_weight"], cfg["del_weight"])
    return metrics


def main():
    args = parse_args()
    left_ckpt, left_model, left_type = load_model(args.left_checkpoint, args.device)
    right_ckpt, right_model, right_type = load_model(args.right_checkpoint, args.device)

    left_ds = build_dataset(left_ckpt["config"], args.h5ad_path, args.split)
    right_ds = build_dataset(right_ckpt["config"], args.h5ad_path, args.split)

    if len(left_ds) != len(right_ds):
        raise ValueError("Datasets differ in length; checkpoints are not directly comparable.")

    records: list[dict[str, float | int | str]] = []

    with torch.no_grad():
        for idx in range(len(left_ds)):
            left_item = left_ds[idx]
            right_item = right_ds[idx]

            if not torch.allclose(left_item["time"], right_item["time"]):
                raise ValueError("Datasets are misaligned in time.")

            left_batch = {
                k: v.to(args.device)
                for k, v in collate_gene_context([left_item]).items()
            }
            right_batch = {
                k: v.to(args.device)
                for k, v in collate_gene_context([right_item]).items()
            }

            left_metrics = run_model(
                left_model,
                left_type,
                left_batch,
                left_ckpt["config"],
                args.left_context_ablation,
            )
            right_metrics = run_model(
                right_model,
                right_type,
                right_batch,
                right_ckpt["config"],
                args.right_context_ablation,
            )
            pair_summary = left_ds.summarize_time_pair(idx % len(left_ds.time_pairs))

            record = {
                "sample_idx": idx,
                "current_center_min": pair_summary["current_center_min"],
                "future_center_min": pair_summary["future_center_min"],
                "anchor_event_positive_count": pair_summary["anchor_event_positive_count"],
                "split_positive_count": pair_summary["split_positive_count"],
                "del_positive_count": pair_summary["del_positive_count"],
                "left_name": Path(args.left_checkpoint).stem,
                "right_name": Path(args.right_checkpoint).stem,
                "left_context_ablation": args.left_context_ablation,
                "right_context_ablation": args.right_context_ablation,
                "left_total": left_metrics["total"],
                "left_gene": left_metrics["gene"],
                "left_split": left_metrics["split"],
                "left_del": left_metrics["del"],
                "right_total": right_metrics["total"],
                "right_gene": right_metrics["gene"],
                "right_split": right_metrics["split"],
                "right_del": right_metrics["del"],
                "delta_total": left_metrics["total"] - right_metrics["total"],
                "delta_gene": left_metrics["gene"] - right_metrics["gene"],
                "delta_split": left_metrics["split"] - right_metrics["split"],
                "delta_del": left_metrics["del"] - right_metrics["del"],
            }
            records.append(record)

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    summary = {
        "num_samples": len(records),
        "mean_left_total": sum(r["left_total"] for r in records) / len(records),
        "mean_right_total": sum(r["right_total"] for r in records) / len(records),
        "mean_delta_total": sum(r["delta_total"] for r in records) / len(records),
        "mean_delta_gene": sum(r["delta_gene"] for r in records) / len(records),
        "mean_delta_split": sum(r["delta_split"] for r in records) / len(records),
        "mean_delta_del": sum(r["delta_del"] for r in records) / len(records),
        "worst_for_left": sorted(records, key=lambda r: r["delta_total"], reverse=True)[:5],
        "worst_for_right": sorted(records, key=lambda r: r["delta_total"])[:5],
    }

    with output_json.open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"wrote {output_json}")
    print(f"wrote {output_csv}")


if __name__ == "__main__":
    main()
