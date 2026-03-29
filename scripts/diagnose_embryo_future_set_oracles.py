#!/usr/bin/env python3
"""Diagnose whether embryo future-set failures come from readout, code, or object contract."""

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

try:  # pragma: no cover - scipy may or may not be available
    from scipy.optimize import linear_sum_assignment as _linear_sum_assignment
except Exception:  # pragma: no cover
    _linear_sum_assignment = None


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
) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (train_x - mean) / std, (test_x - mean) / std


def evaluate_probe_set(
    features: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    n_folds: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    n_samples = next(iter(features.values())).shape[0]
    folds = build_cv_folds(n_samples, n_folds, seed)
    metrics: dict[str, dict[str, float]] = {}
    for name, x in features.items():
        rep_metrics: dict[str, float] = {}
        for target_name, y in targets.items():
            fold_scores = []
            for fold in folds:
                train_idx = np.setdiff1d(np.arange(n_samples), fold)
                if train_idx.size == 0 or fold.size == 0:
                    continue
                train_x, test_x = standardize_train_test(x[train_idx], x[fold])
                w = fit_linear(train_x, y[train_idx])
                pred = predict_linear(test_x, w)
                fold_scores.append(r2_score(y[fold], pred))
            rep_metrics[target_name] = float(np.mean(fold_scores)) if fold_scores else 0.0
        metrics[name] = rep_metrics
    return metrics


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


def masked_gather(model: EmbryoFutureSetModel, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return model.gather_masked_future_view_tensor(tensor, mask)


def safe_weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return (values * weights.unsqueeze(-1)).sum(dim=-2) / denom


def safe_weighted_std(values: torch.Tensor, weights: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    var = (weights.unsqueeze(-1) * (values - mean.unsqueeze(-2)).pow(2)).sum(dim=-2) / denom
    return var.clamp_min(0.0).sqrt()


def summarize_slot_features(slot_features: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if weights is None:
        weights = torch.ones(
            slot_features.shape[:2],
            dtype=slot_features.dtype,
            device=slot_features.device,
        )
    weighted_mean = safe_weighted_mean(slot_features, weights)
    mean = slot_features.mean(dim=1)
    std = slot_features.std(dim=1, unbiased=False)
    weight_stats = torch.stack(
        [
            weights.mean(dim=1),
            weights.std(dim=1, unbiased=False),
            weights.sum(dim=1),
        ],
        dim=-1,
    )
    return torch.cat([weighted_mean, mean, std, weight_stats], dim=-1)


def summarize_codes(codes: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    slot_code_mean = codes.mean(dim=2)
    slot_code_std = codes.std(dim=2, unbiased=False)
    slot_features = torch.cat([slot_code_mean, slot_code_std], dim=-1)
    return summarize_slot_features(slot_features, weights)


def summarize_raw_structured_state(
    genes: torch.Tensor,
    relative_position: torch.Tensor,
    valid_mask: torch.Tensor,
    split_fraction: torch.Tensor,
    mass: torch.Tensor,
    survival: torch.Tensor,
    split_count: torch.Tensor,
) -> torch.Tensor:
    valid = valid_mask.float()
    gene_mean = safe_weighted_mean(genes, valid)
    spatial_valid = valid * (relative_position[..., 4] > 0.5).float()
    pos = relative_position[..., :3]
    pos_mean = safe_weighted_mean(pos, spatial_valid)
    pos_std = safe_weighted_std(pos, spatial_valid, pos_mean)
    spatial_ratio = spatial_valid.mean(dim=2, keepdim=True)
    count_ratio = valid.mean(dim=2, keepdim=True)
    slot_features = torch.cat(
        [
            gene_mean,
            pos_mean,
            pos_std,
            count_ratio,
            spatial_ratio,
            split_fraction.unsqueeze(-1),
            mass.unsqueeze(-1),
            survival.unsqueeze(-1),
            split_count.unsqueeze(-1),
        ],
        dim=-1,
    )
    return summarize_slot_features(slot_features, mass * survival)


def summarize_decoded_structured_state(
    decoded,
    split_fraction: torch.Tensor,
    mass: torch.Tensor,
    survival: torch.Tensor,
    split_count: torch.Tensor,
) -> torch.Tensor:
    valid_prob = torch.sigmoid(decoded.pred_cell_valid_logits)
    spatial_prob = torch.sigmoid(decoded.pred_cell_spatial_logits)
    spatial_weights = valid_prob * spatial_prob
    pos_mean = safe_weighted_mean(decoded.pred_cell_positions, spatial_weights)
    pos_std = safe_weighted_std(decoded.pred_cell_positions, spatial_weights, pos_mean)
    count_ratio = torch.sigmoid(decoded.pred_cell_count).unsqueeze(-1)
    spatial_ratio = spatial_prob.mean(dim=2, keepdim=True)
    slot_features = torch.cat(
        [
            decoded.pred_mean_gene,
            pos_mean,
            pos_std,
            count_ratio,
            spatial_ratio,
            split_fraction.unsqueeze(-1),
            mass.unsqueeze(-1),
            survival.unsqueeze(-1),
            split_count.unsqueeze(-1),
        ],
        dim=-1,
    )
    return summarize_slot_features(slot_features, mass * survival)


def linear_sum_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if _linear_sum_assignment is not None:
        return _linear_sum_assignment(cost)
    n_rows, n_cols = cost.shape
    used_cols: set[int] = set()
    row_idx = []
    col_idx = []
    for row in range(n_rows):
        best_col = None
        best_cost = None
        for col in range(n_cols):
            if col in used_cols:
                continue
            val = float(cost[row, col])
            if best_cost is None or val < best_cost:
                best_cost = val
                best_col = col
        if best_col is None:
            best_col = 0
        used_cols.add(best_col)
        row_idx.append(row)
        col_idx.append(best_col)
    return np.asarray(row_idx, dtype=np.int64), np.asarray(col_idx, dtype=np.int64)


def align_predicted_slots_to_targets(
    pred_slots: torch.Tensor,
    target_slots: torch.Tensor,
) -> torch.Tensor:
    perms = []
    for pred, target in zip(pred_slots, target_slots, strict=True):
        cost = torch.cdist(pred, target).detach().cpu().numpy()
        row_idx, col_idx = linear_sum_assignment(cost)
        perm = np.empty(len(col_idx), dtype=np.int64)
        perm[col_idx] = row_idx
        perms.append(torch.from_numpy(perm))
    perm_tensor = torch.stack(perms, dim=0).to(pred_slots.device)
    gather_index = perm_tensor.unsqueeze(-1).expand(-1, -1, pred_slots.shape[-1])
    return torch.gather(pred_slots, 1, gather_index)


def align_tensor_with_perm(tensor: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    view_shape = [perm.shape[0], perm.shape[1]] + [1] * (tensor.dim() - 2)
    gather_index = perm.view(*view_shape).expand(-1, -1, *tensor.shape[2:])
    return torch.gather(tensor, 1, gather_index)


def build_sample_features(
    out,
    decoded_true,
    decoded_pred,
    masked_future_genes: torch.Tensor,
    masked_future_relative_position: torch.Tensor,
    masked_future_valid_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    target_weights = out.target_future_mass * out.target_future_survival
    pred_survival = torch.sigmoid(out.pred_future_survival_logits)
    pred_split_fraction = torch.sigmoid(out.pred_future_split_logits)
    pred_weights = out.pred_future_mass * pred_survival

    true_raw_state = summarize_raw_structured_state(
        genes=masked_future_genes,
        relative_position=masked_future_relative_position,
        valid_mask=masked_future_valid_mask,
        split_fraction=out.target_future_split_fraction,
        mass=out.target_future_mass,
        survival=out.target_future_survival,
        split_count=out.target_future_split_count,
    )
    true_code = summarize_codes(out.target_future_local_codes, target_weights)
    pred_code = summarize_codes(out.pred_future_local_codes, pred_weights)
    true_decoded = summarize_decoded_structured_state(
        decoded_true,
        split_fraction=out.target_future_split_fraction,
        mass=out.target_future_mass,
        survival=out.target_future_survival,
        split_count=out.target_future_split_count,
    )
    pred_decoded = summarize_decoded_structured_state(
        decoded_pred,
        split_fraction=pred_split_fraction,
        mass=out.pred_future_mass,
        survival=pred_survival,
        split_count=out.pred_future_split_count,
    )
    pred_slot_latent = summarize_slot_features(out.pred_future_set_latents, pred_weights)
    true_slot_latent = summarize_slot_features(out.target_future_set_latents, target_weights)
    pred_pooled_latent = out.pred_future_set_pooled_latent
    true_pooled_latent = out.target_future_set_pooled_latent

    perm_list = []
    for pred, target in zip(out.pred_future_set_latents, out.target_future_set_latents, strict=True):
        cost = torch.cdist(pred, target).detach().cpu().numpy()
        row_idx, col_idx = linear_sum_assignment(cost)
        perm = np.empty(len(col_idx), dtype=np.int64)
        perm[col_idx] = row_idx
        perm_list.append(torch.from_numpy(perm))
    perm = torch.stack(perm_list, dim=0).to(out.pred_future_set_latents.device)
    aligned_pred_latents = align_tensor_with_perm(out.pred_future_set_latents, perm)
    aligned_pred_codes = align_tensor_with_perm(out.pred_future_local_codes, perm)
    aligned_decoded_pred = model_decode_like(decoded_pred, perm)

    pred_slot_latent_oracle = summarize_slot_features(aligned_pred_latents, target_weights)
    pred_code_oracle = summarize_codes(aligned_pred_codes, target_weights)
    pred_decoded_oracle = summarize_decoded_structured_state(
        aligned_decoded_pred,
        split_fraction=out.target_future_split_fraction,
        mass=out.target_future_mass,
        survival=out.target_future_survival,
        split_count=out.target_future_split_count,
    )

    support_stats = torch.stack(
        [
            out.target_future_mass.sum(dim=1),
            out.pred_future_mass.sum(dim=1),
            out.target_future_split_count.sum(dim=1),
            out.pred_future_split_count.sum(dim=1),
            out.target_future_split_fraction.mean(dim=1),
            pred_split_fraction.mean(dim=1),
            out.target_future_survival.mean(dim=1),
            pred_survival.mean(dim=1),
        ],
        dim=-1,
    )

    return {
        "true_raw_state": true_raw_state,
        "true_code": true_code,
        "pred_code": pred_code,
        "true_decoded": true_decoded,
        "pred_decoded": pred_decoded,
        "true_slot_latent": true_slot_latent,
        "pred_slot_latent": pred_slot_latent,
        "true_pooled_latent": true_pooled_latent,
        "pred_pooled_latent": pred_pooled_latent,
        "pred_slot_latent_oracle_support": pred_slot_latent_oracle,
        "pred_code_oracle_support": pred_code_oracle,
        "pred_decoded_oracle_support": pred_decoded_oracle,
        "support_stats": support_stats,
    }


def model_decode_like(decoded, perm: torch.Tensor):
    class _Decoded:
        pass

    out = _Decoded()
    out.pred_cell_genes = align_tensor_with_perm(decoded.pred_cell_genes, perm)
    out.pred_cell_positions = align_tensor_with_perm(decoded.pred_cell_positions, perm)
    out.pred_cell_valid_logits = align_tensor_with_perm(decoded.pred_cell_valid_logits, perm)
    out.pred_cell_spatial_logits = align_tensor_with_perm(decoded.pred_cell_spatial_logits, perm)
    out.pred_cell_count = align_tensor_with_perm(decoded.pred_cell_count, perm)
    out.pred_mean_gene = align_tensor_with_perm(decoded.pred_mean_gene, perm)
    out.pred_patch_latent = align_tensor_with_perm(decoded.pred_patch_latent, perm)
    return out


def collect_representations(
    model: EmbryoFutureSetModel,
    dataset: EmbryoViewDataset,
    config: dict,
    batch_size: int,
    device: str,
    max_batches: int | None,
):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_embryo_view)
    features = {
        key: []
        for key in (
            "true_raw_state",
            "true_code",
            "pred_code",
            "true_decoded",
            "pred_decoded",
            "true_slot_latent",
            "pred_slot_latent",
            "true_pooled_latent",
            "pred_pooled_latent",
            "pred_slot_latent_oracle_support",
            "pred_code_oracle_support",
            "pred_decoded_oracle_support",
            "support_stats",
        )
    }
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
        future_context_role = stack_view_tensor(
            batch,
            "context_role",
            n_future_views,
            prefix_template="future_view_{i}_",
        )
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
            decoded_true = model.decode_future_local_codes(out.target_future_local_codes)
            decoded_pred = model.decode_future_local_codes(out.pred_future_local_codes)

        masked_future_genes = masked_gather(model, future_genes, masked_future_view_mask)
        masked_future_relative_position = masked_gather(model, future_relative_position, masked_future_view_mask)
        masked_future_valid_mask = masked_gather(model, future_valid_mask, masked_future_view_mask)
        batch_features = build_sample_features(
            out=out,
            decoded_true=decoded_true,
            decoded_pred=decoded_pred,
            masked_future_genes=masked_future_genes,
            masked_future_relative_position=masked_future_relative_position,
            masked_future_valid_mask=masked_future_valid_mask,
        )
        for key, value in batch_features.items():
            features[key].append(value.cpu().numpy())
        targets["founder"].append(batch["future_founder_composition"].cpu().numpy())
        targets["celltype"].append(batch["future_celltype_composition"].cpu().numpy())
        targets["depth"].append(batch["future_lineage_depth_stats"].cpu().numpy())
        targets["spatial"].append(batch["future_spatial_extent"].cpu().numpy())
        targets["split"].append(batch["future_split_fraction"].cpu().numpy())

    features = {key: np.concatenate(value, axis=0) for key, value in features.items()}
    targets = {key: np.concatenate(value, axis=0) for key, value in targets.items()}
    return features, targets


def summarize_support_mismatch(features: dict[str, np.ndarray]) -> dict[str, float]:
    stats = features["support_stats"]
    return {
        "mass_total_mae": float(np.mean(np.abs(stats[:, 0] - stats[:, 1]))),
        "split_count_total_mae": float(np.mean(np.abs(stats[:, 2] - stats[:, 3]))),
        "split_fraction_mean_mae": float(np.mean(np.abs(stats[:, 4] - stats[:, 5]))),
        "survival_mean_mae": float(np.mean(np.abs(stats[:, 6] - stats[:, 7]))),
    }


def diagnose(metrics: dict[str, dict[str, float]]) -> dict[str, dict[str, float | str]]:
    diagnosis: dict[str, dict[str, float | str]] = {}
    for target in FOCUS_TARGETS:
        baseline = metrics["pred_pooled_latent"][target]
        true_raw = metrics["true_raw_state"][target]
        true_code = max(metrics["true_code"][target], metrics["true_decoded"][target])
        readout_best = max(
            metrics["pred_slot_latent"][target],
            metrics["pred_code"][target],
            metrics["pred_decoded"][target],
        )
        object_best = max(
            metrics["pred_slot_latent_oracle_support"][target],
            metrics["pred_code_oracle_support"][target],
            metrics["pred_decoded_oracle_support"][target],
        )
        code_gap = true_raw - true_code
        readout_gain = readout_best - baseline
        object_gain = object_best - readout_best
        if code_gap > max(0.10, readout_gain + 0.05, object_gain + 0.05):
            primary = "code_bottleneck"
        elif object_gain > max(0.10, readout_gain + 0.05):
            primary = "object_contract"
        elif readout_gain > 0.10:
            primary = "readout"
        else:
            primary = "mixed_or_other"
        diagnosis[target] = {
            "primary_failure": primary,
            "baseline_pred_pooled_latent_r2": baseline,
            "true_raw_state_r2": true_raw,
            "best_true_code_r2": true_code,
            "best_pred_readout_oracle_r2": readout_best,
            "best_pred_object_oracle_r2": object_best,
            "code_gap": code_gap,
            "readout_gain": readout_gain,
            "object_gain": object_gain,
        }
    return diagnosis


def main():
    args = parse_args()
    ckpt, config, model = load_model(args.checkpoint, args.device)
    dataset = build_dataset(
        config=config,
        split=args.split,
        samples_per_pair=args.samples_per_pair_override,
        event_subset_override=args.event_subset_override,
        seed_offset=1000,
    )
    features, targets = collect_representations(
        model=model,
        dataset=dataset,
        config=config,
        batch_size=args.batch_size,
        device=args.device,
        max_batches=args.max_batches,
    )
    metrics = evaluate_probe_set(features, targets, n_folds=args.n_folds, seed=int(config["seed"]))
    payload = {
        "checkpoint": args.checkpoint,
        "config": {
            "split": args.split,
            "samples_per_pair_override": args.samples_per_pair_override,
            "event_subset_override": args.event_subset_override,
            "batch_size": args.batch_size,
            "n_folds": args.n_folds,
            "max_batches": args.max_batches,
            "model_seed": int(config["seed"]),
            "n_samples": int(next(iter(features.values())).shape[0]),
        },
        "representation_probe_r2": metrics,
        "support_mismatch": summarize_support_mismatch(features),
        "diagnosis": diagnose(metrics),
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["diagnosis"], indent=2))


if __name__ == "__main__":
    main()
