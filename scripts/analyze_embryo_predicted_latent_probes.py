#!/usr/bin/env python3
"""Compare probe readability of true vs predicted future embryo latents."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

project_root = Path(__file__).parent.parent
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
    p = argparse.ArgumentParser(description="Probe true vs predicted future embryo latents.")
    p.add_argument("--checkpoint", required=True, help="Embryo one-step checkpoint.")
    p.add_argument("--split", choices=["val", "all"], default="val")
    p.add_argument("--samples_per_pair_override", type=int, default=16)
    p.add_argument("--event_subset_override", choices=["none", *sorted(EVENT_SUBSET_THRESHOLDS)], default="none")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_folds", type=int, default=5, help="Only used for split=all cross-validation.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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


def build_dataset(
    config: dict,
    split: str,
    samples_per_pair: int,
    event_subset_override: str,
    seed_offset: int,
) -> EmbryoViewDataset:
    class _Args:
        pass

    ns = _Args()
    for key, value in config.items():
        setattr(ns, key, value)
    ns.event_subset = event_subset_override
    ns.val_event_subset = event_subset_override
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
    filters = resolve_event_filters(ns, prefix="val_" if split == "val" else "")

    return EmbryoViewDataset(
        h5ad_path=config["h5ad_path"],
        n_hvg=config["n_hvg"],
        context_size=config["context_size"],
        global_context_size=config["global_context_size"],
        dt_minutes=config["dt_minutes"],
        time_window_minutes=config["time_window_minutes"],
        samples_per_pair=samples_per_pair,
        min_cells_per_window=config["min_cells_per_window"],
        split=split,
        val_fraction=config["val_fraction"],
        random_seed=config["seed"] + seed_offset,
        sampling_strategy=config["sampling_strategy"],
        min_spatial_cells_per_window=config["min_spatial_cells_per_window"],
        spatial_neighbor_pool_size=config["spatial_neighbor_pool_size"],
        delete_target_mode=config["delete_target_mode"],
        views_per_embryo=config["views_per_embryo"],
        future_views_per_embryo=config["future_views_per_embryo"],
        top_cell_types=config.get("top_cell_types", 8),
        **filters,
    )


def load_model(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = ckpt["config"]
    backbone_ckpt = torch.load(ckpt["backbone_checkpoint"], map_location="cpu")
    backbone_cfg = backbone_ckpt["config"]
    backbone = EmbryoMaskedViewModel(
        gene_dim=backbone_ckpt["gene_dim"],
        context_size=backbone_cfg["context_size"],
        model_type=backbone_cfg["model_type"],
        d_model=backbone_cfg["d_model"],
        n_heads=backbone_cfg["n_heads"],
        n_layers=backbone_cfg["n_layers"],
        head_dim=backbone_cfg["head_dim"],
        use_pairwise_spatial_bias=backbone_cfg["pairwise_spatial_bias"],
    )
    backbone.load_state_dict(backbone_ckpt["model_state_dict"])
    model = EmbryoOneStepLatentModel(
        backbone=backbone,
        celltype_dim=ckpt["celltype_dim"],
        d_model=backbone_cfg["d_model"],
        predict_delta=config.get("predict_delta", False),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return ckpt, config, model


def collect_latents_and_targets(
    model: EmbryoOneStepLatentModel,
    dataset: EmbryoViewDataset,
    batch_size: int,
    device: str,
):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_embryo_view)
    true_latents = []
    pred_latents = []
    targets = {k: [] for k in ("founder", "celltype", "depth", "spatial", "split")}

    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
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
            true_latents.append(out.target_future_embryo_latent.cpu().numpy())
            pred_latents.append(out.pred_future_embryo_latent.cpu().numpy())
            targets["founder"].append(batch["future_founder_composition"].cpu().numpy())
            targets["celltype"].append(batch["future_celltype_composition"].cpu().numpy())
            targets["depth"].append(batch["future_lineage_depth_stats"].cpu().numpy())
            targets["spatial"].append(batch["future_spatial_extent"].cpu().numpy())
            targets["split"].append(batch["future_split_fraction"].cpu().numpy())

    return (
        np.concatenate(true_latents, axis=0).astype(np.float32),
        np.concatenate(pred_latents, axis=0).astype(np.float32),
        {k: np.concatenate(v, axis=0).astype(np.float32) for k, v in targets.items()},
    )


def evaluate_probe_set(probes: dict[str, np.ndarray], x: np.ndarray, targets: dict[str, np.ndarray]) -> dict[str, float]:
    metrics = {}
    for key, w in probes.items():
        pred = predict_linear(x, w)
        metrics[f"{key}_r2"] = r2_score(targets[key], pred)
    return metrics


def build_cv_folds(n_samples: int, n_folds: int, seed: int) -> list[np.ndarray]:
    n_folds = max(2, min(n_folds, n_samples))
    perm = np.random.default_rng(seed).permutation(n_samples)
    return [fold for fold in np.array_split(perm, n_folds) if len(fold) > 0]


def main():
    args = parse_args()
    ckpt, config, model = load_model(args.checkpoint, args.device)
    eval_ds = build_dataset(
        config=config,
        split=args.split,
        samples_per_pair=args.samples_per_pair_override,
        event_subset_override=args.event_subset_override,
        seed_offset=1000,
    )
    eval_true_latent, eval_pred_latent, eval_targets = collect_latents_and_targets(
        model=model,
        dataset=eval_ds,
        batch_size=args.batch_size,
        device=args.device,
    )

    metrics = {
        "checkpoint": args.checkpoint,
        "backbone_checkpoint": ckpt["backbone_checkpoint"],
        "split": args.split,
        "n_eval_samples": int(len(eval_true_latent)),
        "targets": {},
    }
    for key, target in eval_targets.items():
        metrics["targets"][key] = {
            "target_variance": float(np.var(target)),
        }

    if args.split == "all":
        folds = build_cv_folds(len(eval_true_latent), args.n_folds, config["seed"])
        true_scores = {key: [] for key in eval_targets}
        pred_scores = {key: [] for key in eval_targets}
        for test_idx in folds:
            train_mask = np.ones(len(eval_true_latent), dtype=bool)
            train_mask[test_idx] = False
            train_idx = np.nonzero(train_mask)[0]
            fold_probes = {
                key: fit_linear(eval_true_latent[train_idx], target[train_idx]) for key, target in eval_targets.items()
            }
            fold_true = evaluate_probe_set(
                fold_probes,
                eval_true_latent[test_idx],
                {key: target[test_idx] for key, target in eval_targets.items()},
            )
            fold_pred = evaluate_probe_set(
                fold_probes,
                eval_pred_latent[test_idx],
                {key: target[test_idx] for key, target in eval_targets.items()},
            )
            for key in eval_targets:
                true_scores[key].append(fold_true[f"{key}_r2"])
                pred_scores[key].append(fold_pred[f"{key}_r2"])
        metrics["cv_n_folds"] = len(folds)
        for key in eval_targets:
            true_r2 = float(np.mean(true_scores[key]))
            pred_r2 = float(np.mean(pred_scores[key]))
            metrics["targets"][key] |= {
                "true_future_latent_r2": true_r2,
                "predicted_future_latent_r2": pred_r2,
                "gap_true_minus_pred": true_r2 - pred_r2,
                "true_future_latent_r2_std": float(np.std(true_scores[key])),
                "predicted_future_latent_r2_std": float(np.std(pred_scores[key])),
            }
    else:
        train_ds = build_dataset(
            config=config,
            split="train",
            samples_per_pair=config["samples_per_pair"],
            event_subset_override="none",
            seed_offset=0,
        )
        train_true_latent, _train_pred_latent, train_targets = collect_latents_and_targets(
            model=model,
            dataset=train_ds,
            batch_size=args.batch_size,
            device=args.device,
        )
        metrics["n_train_samples"] = int(len(train_true_latent))
        probes = {key: fit_linear(train_true_latent, target) for key, target in train_targets.items()}
        true_metrics = evaluate_probe_set(probes, eval_true_latent, eval_targets)
        pred_metrics = evaluate_probe_set(probes, eval_pred_latent, eval_targets)
        for key in probes:
            true_r2 = true_metrics[f"{key}_r2"]
            pred_r2 = pred_metrics[f"{key}_r2"]
            metrics["targets"][key] |= {
                "true_future_latent_r2": true_r2,
                "predicted_future_latent_r2": pred_r2,
                "gap_true_minus_pred": true_r2 - pred_r2,
            }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
