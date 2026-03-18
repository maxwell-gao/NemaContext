#!/usr/bin/env python3
"""Compare probe readability of true vs predicted embryo future-set latents."""

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

from examples.whole_organism_ar.train_embryo_future_set import (  # noqa: E402
    build_mask,
    load_backbone,
)
from examples.whole_organism_ar.train_embryo_state import stack_view_tensor  # noqa: E402
from examples.whole_organism_ar.train_gene_context import (  # noqa: E402
    EVENT_SUBSET_THRESHOLDS,
    resolve_event_filters,
)
from src.branching_flows.emergent_loss import sinkhorn_divergence  # noqa: E402
from src.branching_flows.gene_context import EmbryoFutureSetModel  # noqa: E402
from src.data.gene_context_dataset import EmbryoViewDataset, collate_embryo_view  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Probe true vs predicted future-set latents.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=["val", "all"], default="all")
    p.add_argument("--samples_per_pair_override", type=int, default=16)
    p.add_argument("--event_subset_override", choices=["none", *sorted(EVENT_SUBSET_THRESHOLDS)], default="none")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_folds", type=int, default=5)
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

    class _Args:
        pass

    args = _Args()
    for key, value in config.items():
        setattr(args, key, value)
    args.backbone_checkpoint = ckpt["backbone_checkpoint"]
    backbone, backbone_ckpt = load_backbone(args)
    model = EmbryoFutureSetModel(
        backbone=backbone,
        future_slots=int(ckpt["future_slots"]),
        d_model=int(config["d_model"] if backbone_ckpt is None else backbone_ckpt["config"]["d_model"]),
        gene_dim=int(ckpt["gene_dim"]),
        n_heads=int(config.get("n_heads", backbone_ckpt["config"]["n_heads"] if backbone_ckpt is not None else 4)),
        decoder_layers=int(config.get("decoder_layers", 3)),
        head_dim=int(config.get("head_dim", backbone_ckpt["config"]["head_dim"] if backbone_ckpt is not None else 32)),
        use_current_local_tokens=bool(config.get("use_current_local_tokens", False)),
        learn_current_token_gate=bool(config.get("learn_current_token_gate", True)),
        current_token_gate_init=float(config.get("current_token_gate_init", 0.5)),
        current_conditioning_mode=str(config.get("current_conditioning_mode", "flat_tokens")),
        predict_future_cell_tokens=bool(config.get("predict_future_cell_tokens", False)),
        cell_tokens_per_view=(
            None if config.get("cell_tokens_per_view", None) is None else int(config.get("cell_tokens_per_view"))
        ),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)
    model.eval()
    return ckpt, config, model


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


def collect_latents_and_targets(
    model: EmbryoFutureSetModel,
    dataset: EmbryoViewDataset,
    config: dict,
    batch_size: int,
    device: str,
):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_embryo_view)
    true_latents = []
    pred_latents = []
    targets = {k: [] for k in ("founder", "celltype", "depth", "spatial", "split")}
    losses = {"latent_set": [], "gene_set": [], "mean_latent": []}

    for batch_idx, batch in enumerate(loader):
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
        future_relative_position = stack_view_tensor(
            batch,
            "relative_position",
            n_future_views,
            prefix_template="future_view_{i}_",
        )
        future_token_times = stack_view_tensor(batch, "token_times", n_future_views, prefix_template="future_view_{i}_")
        future_valid_mask = stack_view_tensor(batch, "valid_mask", n_future_views, prefix_template="future_view_{i}_")
        future_anchor_mask = stack_view_tensor(batch, "anchor_mask", n_future_views, prefix_template="future_view_{i}_")
        future_time = stack_view_tensor(batch, "time", n_future_views, prefix_template="future_view_{i}_")

        torch.manual_seed(int(config["seed"]) + batch_idx)
        masked_view_mask = build_mask(
            genes.shape[0],
            n_views,
            float(config["context_mask_ratio"]),
            genes.device,
            allow_empty=True,
        )
        masked_future_view_mask = build_mask(
            genes.shape[0],
            n_future_views,
            float(config["future_mask_ratio"]),
            genes.device,
            allow_empty=False,
        )

        with torch.no_grad():
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
                masked_future_view_mask=masked_future_view_mask,
                masked_view_mask=masked_view_mask,
                context_role=context_role,
                relative_position=relative_position,
                future_context_role=future_context_role,
                future_relative_position=future_relative_position,
            )

        true_latents.append(out.target_future_set_latents.mean(dim=1).cpu().numpy())
        pred_latents.append(out.pred_future_set_latents.mean(dim=1).cpu().numpy())
        targets["founder"].append(batch["future_founder_composition"].cpu().numpy())
        targets["celltype"].append(batch["future_celltype_composition"].cpu().numpy())
        targets["depth"].append(batch["future_lineage_depth_stats"].cpu().numpy())
        targets["spatial"].append(batch["future_spatial_extent"].cpu().numpy())
        targets["split"].append(batch["future_split_fraction"].cpu().numpy())
        losses["latent_set"].append(
            float(
                sinkhorn_divergence(
                    out.pred_future_set_latents,
                    out.target_future_set_latents,
                    blur=float(config["sinkhorn_blur"]),
                ).item()
            )
        )
        losses["gene_set"].append(
            float(
                sinkhorn_divergence(
                    out.pred_future_set_genes,
                    out.target_future_set_genes,
                    blur=float(config["gene_sinkhorn_blur"]),
                ).item()
            )
        )
        losses["mean_latent"].append(
            float(
                (
                    1.0
                    - torch.nn.functional.cosine_similarity(
                        out.pred_future_set_latents.mean(dim=1),
                        out.target_future_set_latents.mean(dim=1),
                        dim=-1,
                    )
                )
                .mean()
                .item()
            )
        )

    return (
        np.concatenate(true_latents, axis=0).astype(np.float32),
        np.concatenate(pred_latents, axis=0).astype(np.float32),
        {k: np.concatenate(v, axis=0).astype(np.float32) for k, v in targets.items()},
        {k: float(np.mean(v)) for k, v in losses.items()},
    )


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
    eval_true_latent, eval_pred_latent, eval_targets, eval_losses = collect_latents_and_targets(
        model=model,
        dataset=eval_ds,
        config=config,
        batch_size=args.batch_size,
        device=args.device,
    )

    metrics = {
        "checkpoint": args.checkpoint,
        "backbone_checkpoint": ckpt["backbone_checkpoint"],
        "split": args.split,
        "n_eval_samples": int(len(eval_true_latent)),
        "future_slots": int(ckpt["future_slots"]),
        "eval_losses": eval_losses,
        "targets": {},
    }
    for key, target in eval_targets.items():
        metrics["targets"][key] = {"target_variance": float(np.var(target))}

    if args.split == "all":
        folds = build_cv_folds(len(eval_true_latent), args.n_folds, int(config["seed"]))
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
                {key: value[test_idx] for key, value in eval_targets.items()},
            )
            fold_pred = evaluate_probe_set(
                fold_probes,
                eval_pred_latent[test_idx],
                {key: value[test_idx] for key, value in eval_targets.items()},
            )
            for key in eval_targets:
                true_scores[key].append(fold_true[f"{key}_r2"])
                pred_scores[key].append(fold_pred[f"{key}_r2"])
        metrics["mode"] = "cv"
        for key in eval_targets:
            metrics["targets"][key]["true_r2_mean"] = float(np.mean(true_scores[key]))
            metrics["targets"][key]["true_r2_std"] = float(np.std(true_scores[key]))
            metrics["targets"][key]["pred_r2_mean"] = float(np.mean(pred_scores[key]))
            metrics["targets"][key]["pred_r2_std"] = float(np.std(pred_scores[key]))
            metrics["targets"][key]["gap_true_minus_pred"] = (
                metrics["targets"][key]["true_r2_mean"] - metrics["targets"][key]["pred_r2_mean"]
            )
    else:
        split_idx = max(1, int(round(0.8 * len(eval_true_latent))))
        probes = {
            key: fit_linear(eval_true_latent[:split_idx], target[:split_idx]) for key, target in eval_targets.items()
        }
        true_scores = evaluate_probe_set(
            probes,
            eval_true_latent[split_idx:],
            {key: value[split_idx:] for key, value in eval_targets.items()},
        )
        pred_scores = evaluate_probe_set(
            probes,
            eval_pred_latent[split_idx:],
            {key: value[split_idx:] for key, value in eval_targets.items()},
        )
        metrics["mode"] = "holdout"
        for key in eval_targets:
            metrics["targets"][key]["true_r2"] = float(true_scores[f"{key}_r2"])
            metrics["targets"][key]["pred_r2"] = float(pred_scores[f"{key}_r2"])
            metrics["targets"][key]["gap_true_minus_pred"] = (
                metrics["targets"][key]["true_r2"] - metrics["targets"][key]["pred_r2"]
            )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
