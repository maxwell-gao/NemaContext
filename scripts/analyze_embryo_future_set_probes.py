#!/usr/bin/env python3
"""Probe token/set-level gene readability of embryo future-set states."""

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
from src.branching_flows.emergent_loss import sinkhorn_divergence  # noqa: E402
from src.branching_flows.gene_context import EmbryoFutureSetModel  # noqa: E402
from src.data.gene_context_dataset import EmbryoViewDataset, collate_embryo_view  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
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


def mean_dim_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.ndim == 1:
        return r2_score(y_true, y_pred)
    scores = [r2_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    return float(np.mean(scores))


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


def build_cv_folds(n_samples: int, n_folds: int, seed: int) -> list[np.ndarray]:
    n_folds = max(2, min(n_folds, n_samples))
    perm = np.random.default_rng(seed).permutation(n_samples)
    return [fold for fold in np.array_split(perm, n_folds) if len(fold) > 0]


def flatten_valid_tokens(
    token_states: np.ndarray,
    target_genes: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    flat_x = token_states.reshape(-1, token_states.shape[-1])
    flat_y = target_genes.reshape(-1, target_genes.shape[-1])
    flat_valid = valid_mask.reshape(-1).astype(bool)
    return flat_x[flat_valid], flat_y[flat_valid]


def collect_token_states_and_targets(
    model: EmbryoFutureSetModel,
    dataset: EmbryoViewDataset,
    config: dict,
    batch_size: int,
    device: str,
):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_embryo_view)
    true_tokens = []
    pred_tokens = []
    target_genes = []
    decoded_true_genes = []
    decoded_pred_genes = []
    valid_masks = []
    losses = {"latent_set": [], "gene_set": []}

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
            batch, "relative_position", n_future_views, prefix_template="future_view_{i}_"
        )
        future_split_fraction = stack_view_tensor(
            batch, "split_fraction", n_future_views, prefix_template="future_view_{i}_"
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
            decoded_true = model.decode_future_local_codes(out.target_future_local_codes)
            decoded = model.decode_future_local_codes(out.pred_future_local_codes)

        masked_future_valid_mask = model.gather_masked_future_view_tensor(future_valid_mask, masked_future_view_mask)
        masked_future_genes = model.gather_masked_future_view_tensor(future_genes, masked_future_view_mask)

        true_tokens.append(out.target_future_local_codes.cpu().numpy())
        pred_tokens.append(out.pred_future_local_codes.cpu().numpy())
        target_genes.append(masked_future_genes.cpu().numpy())
        decoded_true_genes.append(decoded_true.pred_cell_genes.cpu().numpy())
        decoded_pred_genes.append(decoded.pred_cell_genes.cpu().numpy())
        valid_masks.append(masked_future_valid_mask.cpu().numpy())
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
    return (
        np.concatenate(true_tokens, axis=0).astype(np.float32),
        np.concatenate(pred_tokens, axis=0).astype(np.float32),
        np.concatenate(target_genes, axis=0).astype(np.float32),
        np.concatenate(decoded_true_genes, axis=0).astype(np.float32),
        np.concatenate(decoded_pred_genes, axis=0).astype(np.float32),
        np.concatenate(valid_masks, axis=0).astype(bool),
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
    (
        true_tokens,
        pred_tokens,
        target_genes,
        decoded_true_genes,
        decoded_pred_genes,
        valid_masks,
        eval_losses,
    ) = collect_token_states_and_targets(
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
        "n_eval_samples": int(true_tokens.shape[0]),
        "future_slots": int(ckpt["future_slots"]),
        "n_eval_valid_tokens": int(valid_masks.sum()),
        "eval_losses": eval_losses,
        "gene_probe": {},
        "decoded_true_gene": {},
        "decoded_gene": {},
    }

    if args.split != "all":
        raise ValueError("token/set gene evaluation currently requires split=all for sample-held-out CV")

    if true_tokens.shape[2] == valid_masks.shape[2]:
        folds = build_cv_folds(true_tokens.shape[0], args.n_folds, int(config["seed"]))
        true_scores = []
        pred_scores = []
        for test_idx in folds:
            train_mask = np.ones(true_tokens.shape[0], dtype=bool)
            train_mask[test_idx] = False
            train_idx = np.nonzero(train_mask)[0]
            train_x, train_y = flatten_valid_tokens(
                true_tokens[train_idx],
                target_genes[train_idx],
                valid_masks[train_idx],
            )
            test_true_x, test_y = flatten_valid_tokens(
                true_tokens[test_idx],
                target_genes[test_idx],
                valid_masks[test_idx],
            )
            test_pred_x, _ = flatten_valid_tokens(
                pred_tokens[test_idx],
                target_genes[test_idx],
                valid_masks[test_idx],
            )
            probe = fit_linear(train_x, train_y)
            true_scores.append(mean_dim_r2(test_y, predict_linear(test_true_x, probe)))
            pred_scores.append(mean_dim_r2(test_y, predict_linear(test_pred_x, probe)))
        metrics["gene_probe"]["mode"] = "token_state_cv"
        metrics["gene_probe"]["target_variance"] = float(np.var(target_genes[valid_masks]))
        metrics["gene_probe"]["true_r2_mean"] = float(np.mean(true_scores))
        metrics["gene_probe"]["true_r2_std"] = float(np.std(true_scores))
        metrics["gene_probe"]["pred_r2_mean"] = float(np.mean(pred_scores))
        metrics["gene_probe"]["pred_r2_std"] = float(np.std(pred_scores))
        metrics["gene_probe"]["gap_true_minus_pred"] = (
            metrics["gene_probe"]["true_r2_mean"] - metrics["gene_probe"]["pred_r2_mean"]
        )
    else:
        metrics["gene_probe"]["mode"] = "unavailable"
        metrics["gene_probe"]["reason"] = "future token states use a compressed code bottleneck; evaluate decoded gene sets instead"
        metrics["gene_probe"]["token_count"] = int(true_tokens.shape[2])
        metrics["gene_probe"]["target_count"] = int(valid_masks.shape[2])

    decoded_true_y_pred, decoded_true_y_true = flatten_valid_tokens(
        decoded_true_genes,
        target_genes,
        valid_masks,
    )
    decoded_y_pred, decoded_y_true = flatten_valid_tokens(
        decoded_pred_genes,
        target_genes,
        valid_masks,
    )
    decoded_true_mse = float(np.mean((decoded_true_y_pred - decoded_true_y_true) ** 2))
    decoded_true_r2 = mean_dim_r2(decoded_true_y_true, decoded_true_y_pred)
    decoded_mse = float(np.mean((decoded_y_pred - decoded_y_true) ** 2))
    decoded_r2 = mean_dim_r2(decoded_y_true, decoded_y_pred)

    metrics["mode"] = "cv"
    metrics["decoded_true_gene"]["token_mse"] = decoded_true_mse
    metrics["decoded_true_gene"]["token_r2_mean"] = decoded_true_r2
    metrics["decoded_gene"]["token_mse"] = decoded_mse
    metrics["decoded_gene"]["token_r2_mean"] = decoded_r2

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
