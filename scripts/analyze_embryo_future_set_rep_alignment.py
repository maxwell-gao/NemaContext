#!/usr/bin/env python3
"""Analyze true/pred representation alignment for embryo future-set pooled latents."""

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
    load_token_readout_anchor,
)
from examples.whole_organism_ar.train_embryo_state import stack_view_tensor  # noqa: E402
from examples.whole_organism_ar.train_gene_context import (  # noqa: E402
    EVENT_SUBSET_THRESHOLDS,
    resolve_event_filters,
)
from src.branching_flows.gene_context import EmbryoFutureSetModel  # noqa: E402
from src.data.gene_context_dataset import EmbryoViewDataset, collate_embryo_view  # noqa: E402

FOCUS_TARGETS = ("spatial", "split")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=["val", "all"], default="all")
    p.add_argument("--samples_per_pair_override", type=int, default=16)
    p.add_argument(
        "--event_subset_override",
        choices=["none", *sorted(EVENT_SUBSET_THRESHOLDS)],
        default="none",
    )
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--n_bins", type=int, default=3)
    p.add_argument("--ridge_lambda", type=float, default=1e-3)
    p.add_argument("--knn_k", type=int, default=5)
    p.add_argument("--max_batches", type=int, default=None)
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


def build_cv_folds(n_samples: int, n_folds: int, seed: int) -> list[np.ndarray]:
    n_folds = max(2, min(n_folds, n_samples))
    perm = np.random.default_rng(seed).permutation(n_samples)
    return [fold for fold in np.array_split(perm, n_folds) if len(fold) > 0]


def standardize_train_test(
    train_x: np.ndarray,
    test_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (train_x - mean) / std, (test_x - mean) / std, mean, std


def scalarize_target(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return y
    if y.shape[1] == 1:
        return y[:, 0]
    return np.linalg.norm(y, axis=1)


def fit_ridge_map(x: np.ndarray, y: np.ndarray, ridge_lambda: float) -> np.ndarray:
    design = np.concatenate([x, np.ones((len(x), 1), dtype=x.dtype)], axis=1)
    reg = ridge_lambda * np.eye(design.shape[1], dtype=x.dtype)
    reg[-1, -1] = 0.0
    lhs = design.T @ design + reg
    rhs = design.T @ y
    return np.linalg.solve(lhs, rhs)


def apply_affine_map(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    design = np.concatenate([x, np.ones((len(x), 1), dtype=x.dtype)], axis=1)
    return design @ w


def fit_procrustes_map(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_mean = x.mean(axis=0, keepdims=True)
    y_mean = y.mean(axis=0, keepdims=True)
    xc = x - x_mean
    yc = y - y_mean
    cov = xc.T @ yc
    u, _, vt = np.linalg.svd(cov, full_matrices=False)
    rot = u @ vt
    return rot.astype(x.dtype, copy=False), (y_mean - x_mean @ rot).astype(x.dtype, copy=False)


def apply_procrustes_map(x: np.ndarray, rot: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return x @ rot + bias


def knn_jaccard(x: np.ndarray, y: np.ndarray, k: int) -> float:
    k = max(1, min(k, len(x) - 1))
    if k < 1:
        return 0.0
    dx = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)
    dy = np.sum((y[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    np.fill_diagonal(dx, np.inf)
    np.fill_diagonal(dy, np.inf)
    x_nn = np.argpartition(dx, kth=k - 1, axis=1)[:, :k]
    y_nn = np.argpartition(dy, kth=k - 1, axis=1)[:, :k]
    overlaps = []
    for a, b in zip(x_nn, y_nn, strict=True):
        sa = set(a.tolist())
        sb = set(b.tolist())
        overlaps.append(len(sa & sb) / len(sa | sb))
    return float(np.mean(overlaps))


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
        code_tokens=int(config.get("code_tokens", 8)),
        predict_dense_future_tokens=bool(config.get("predict_dense_future_tokens", False)),
        strict_token_jepa=bool(config.get("strict_token_jepa", False)),
        token_readout_anchor=load_token_readout_anchor(
            config.get("token_readout_anchor_path"),
            int(config["d_model"] if backbone_ckpt is None else backbone_ckpt["config"]["d_model"]),
        ),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)
    model.eval()
    return ckpt, config, model


def collect_latents_and_targets(
    model: EmbryoFutureSetModel,
    dataset: EmbryoViewDataset,
    config: dict,
    batch_size: int,
    device: str,
    max_batches: int | None,
):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_embryo_view)
    true_latents = []
    pred_latents = []
    targets = {k: [] for k in ("founder", "celltype", "depth", "spatial", "split")}
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
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
        future_split_fraction = stack_view_tensor(
            batch,
            "split_fraction",
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
                future_split_fraction=future_split_fraction,
                masked_view_mask=masked_view_mask,
                context_role=context_role,
                relative_position=relative_position,
                future_context_role=future_context_role,
                future_relative_position=future_relative_position,
            )

        true_latents.append(out.target_future_set_pooled_latent.cpu().numpy())
        pred_latents.append(out.pred_future_set_pooled_latent.cpu().numpy())
        targets["founder"].append(batch["future_founder_composition"].cpu().numpy())
        targets["celltype"].append(batch["future_celltype_composition"].cpu().numpy())
        targets["depth"].append(batch["future_lineage_depth_stats"].cpu().numpy())
        targets["spatial"].append(batch["future_spatial_extent"].cpu().numpy())
        targets["split"].append(batch["future_split_fraction"].cpu().numpy())

    true_latents = np.concatenate(true_latents, axis=0).astype(np.float32)
    pred_latents = np.concatenate(pred_latents, axis=0).astype(np.float32)
    targets = {k: np.concatenate(v, axis=0).astype(np.float32) for k, v in targets.items()}
    return true_latents, pred_latents, targets


def fit_bin_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    q = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges = np.quantile(values, q) if len(values) > 0 else np.array([])
    return np.unique(edges)


def assign_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(values, edges, right=False)


def analyze_alignment(
    true_latents: np.ndarray,
    pred_latents: np.ndarray,
    targets: dict[str, np.ndarray],
    n_folds: int,
    n_bins: int,
    ridge_lambda: float,
    knn_k: int,
    seed: int,
) -> dict:
    n_samples = len(true_latents)
    folds = build_cv_folds(n_samples, n_folds, seed)
    target_scalars = {
        "spatial": scalarize_target(targets["spatial"]),
        "split": scalarize_target(targets["split"]),
    }
    fold_reports = []
    target_reports = {
        key: {
            "true_self_r2": [],
            "direct_transfer_r2": [],
            "ridge_transfer_r2": [],
            "procrustes_transfer_r2": [],
            "conditional_ridge_transfer_r2": [],
            "ridge_latent_r2": [],
            "procrustes_latent_r2": [],
            "knn_jaccard_raw": [],
            "knn_jaccard_ridge": [],
            "knn_jaccard_procrustes": [],
        }
        for key in FOCUS_TARGETS
    }

    for fold_id, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(np.arange(n_samples), test_idx)
        if train_idx.size == 0 or test_idx.size == 0:
            continue

        true_train_std, true_test_std, true_mean, true_std = standardize_train_test(
            true_latents[train_idx], true_latents[test_idx]
        )
        pred_train_std, pred_test_std, pred_mean, pred_std = standardize_train_test(
            pred_latents[train_idx], pred_latents[test_idx]
        )

        ridge_map = fit_ridge_map(pred_train_std, true_train_std, ridge_lambda=ridge_lambda)
        pred_test_ridge = apply_affine_map(pred_test_std, ridge_map)

        procrustes_rot, procrustes_bias = fit_procrustes_map(pred_train_std, true_train_std)
        pred_test_proc = apply_procrustes_map(pred_test_std, procrustes_rot, procrustes_bias)

        fold_result = {}
        for target_name in FOCUS_TARGETS:
            y_train = targets[target_name][train_idx]
            y_test = targets[target_name][test_idx]
            probe = fit_linear(true_train_std, y_train)
            true_self = r2_score(y_test, predict_linear(true_test_std, probe))
            direct_transfer = r2_score(y_test, predict_linear(pred_test_std, probe))
            ridge_transfer = r2_score(y_test, predict_linear(pred_test_ridge, probe))
            proc_transfer = r2_score(y_test, predict_linear(pred_test_proc, probe))

            ridge_latent_r2 = r2_score(true_test_std, pred_test_ridge)
            proc_latent_r2 = r2_score(true_test_std, pred_test_proc)

            edges = fit_bin_edges(target_scalars[target_name][train_idx], n_bins)
            bins_train = assign_bins(target_scalars[target_name][train_idx], edges)
            bins_test = assign_bins(target_scalars[target_name][test_idx], edges)
            pred_test_conditional = np.zeros_like(pred_test_std)
            for bin_id in np.unique(bins_train):
                train_mask = bins_train == bin_id
                test_mask = bins_test == bin_id
                if train_mask.sum() < 2:
                    pred_test_conditional[test_mask] = pred_test_ridge[test_mask]
                    continue
                local_map = fit_ridge_map(
                    pred_train_std[train_mask],
                    true_train_std[train_mask],
                    ridge_lambda=ridge_lambda,
                )
                if test_mask.any():
                    pred_test_conditional[test_mask] = apply_affine_map(pred_test_std[test_mask], local_map)
            missing_mask = np.all(pred_test_conditional == 0.0, axis=1)
            if missing_mask.any():
                pred_test_conditional[missing_mask] = pred_test_ridge[missing_mask]
            conditional_transfer = r2_score(y_test, predict_linear(pred_test_conditional, probe))

            raw_knn = knn_jaccard(true_test_std, pred_test_std, knn_k)
            ridge_knn = knn_jaccard(true_test_std, pred_test_ridge, knn_k)
            proc_knn = knn_jaccard(true_test_std, pred_test_proc, knn_k)

            target_reports[target_name]["true_self_r2"].append(true_self)
            target_reports[target_name]["direct_transfer_r2"].append(direct_transfer)
            target_reports[target_name]["ridge_transfer_r2"].append(ridge_transfer)
            target_reports[target_name]["procrustes_transfer_r2"].append(proc_transfer)
            target_reports[target_name]["conditional_ridge_transfer_r2"].append(conditional_transfer)
            target_reports[target_name]["ridge_latent_r2"].append(ridge_latent_r2)
            target_reports[target_name]["procrustes_latent_r2"].append(proc_latent_r2)
            target_reports[target_name]["knn_jaccard_raw"].append(raw_knn)
            target_reports[target_name]["knn_jaccard_ridge"].append(ridge_knn)
            target_reports[target_name]["knn_jaccard_procrustes"].append(proc_knn)

            fold_result[target_name] = {
                "true_self_r2": true_self,
                "direct_transfer_r2": direct_transfer,
                "ridge_transfer_r2": ridge_transfer,
                "procrustes_transfer_r2": proc_transfer,
                "conditional_ridge_transfer_r2": conditional_transfer,
                "ridge_latent_r2": ridge_latent_r2,
                "procrustes_latent_r2": proc_latent_r2,
                "knn_jaccard_raw": raw_knn,
                "knn_jaccard_ridge": ridge_knn,
                "knn_jaccard_procrustes": proc_knn,
            }
        fold_reports.append({"fold": fold_id, "metrics": fold_result})

    summary = {}
    diagnosis = {}
    for target_name, values in target_reports.items():
        summary[target_name] = {key: float(np.mean(v)) for key, v in values.items()}
        direct_gap = summary[target_name]["true_self_r2"] - summary[target_name]["direct_transfer_r2"]
        ridge_gain = summary[target_name]["ridge_transfer_r2"] - summary[target_name]["direct_transfer_r2"]
        conditional_gain = (
            summary[target_name]["conditional_ridge_transfer_r2"] - summary[target_name]["ridge_transfer_r2"]
        )
        proc_gain = summary[target_name]["procrustes_transfer_r2"] - summary[target_name]["direct_transfer_r2"]
        if ridge_gain > max(0.10, conditional_gain + 0.05):
            primary = "global_linear_misalignment"
        elif conditional_gain > 0.05:
            primary = "conditional_misalignment"
        elif proc_gain > 0.10:
            primary = "orthogonal_misalignment"
        else:
            primary = "nonlinear_or_mixed"
        diagnosis[target_name] = {
            "primary_alignment_issue": primary,
            "direct_transfer_gap": direct_gap,
            "ridge_gain": ridge_gain,
            "procrustes_gain": proc_gain,
            "conditional_ridge_gain": conditional_gain,
        }

    return {
        "summary": summary,
        "diagnosis": diagnosis,
        "fold_metrics": fold_reports,
    }


def main():
    args = parse_args()
    _, config, model = load_model(args.checkpoint, args.device)
    dataset = build_dataset(
        config=config,
        split=args.split,
        samples_per_pair=args.samples_per_pair_override,
        event_subset_override=args.event_subset_override,
        seed_offset=1000,
    )
    true_latents, pred_latents, targets = collect_latents_and_targets(
        model=model,
        dataset=dataset,
        config=config,
        batch_size=args.batch_size,
        device=args.device,
        max_batches=args.max_batches,
    )
    payload = {
        "checkpoint": args.checkpoint,
        "config": {
            "split": args.split,
            "samples_per_pair_override": args.samples_per_pair_override,
            "event_subset_override": args.event_subset_override,
            "batch_size": args.batch_size,
            "n_folds": args.n_folds,
            "n_bins": args.n_bins,
            "ridge_lambda": args.ridge_lambda,
            "knn_k": args.knn_k,
            "max_batches": args.max_batches,
            "model_seed": int(config["seed"]),
            "n_samples": int(len(true_latents)),
        },
    }
    payload.update(
        analyze_alignment(
            true_latents=true_latents,
            pred_latents=pred_latents,
            targets=targets,
            n_folds=args.n_folds,
            n_bins=args.n_bins,
            ridge_lambda=args.ridge_lambda,
            knn_k=args.knn_k,
            seed=int(config["seed"]),
        )
    )
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["diagnosis"], indent=2))


if __name__ == "__main__":
    main()
