#!/usr/bin/env python3
"""Fit a fixed canonical basis for embryo future-set pooled latents."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from examples.whole_organism_ar.train_embryo_future_set import build_mask, load_backbone  # noqa: E402
from examples.whole_organism_ar.train_embryo_state import stack_view_tensor  # noqa: E402
from examples.whole_organism_ar.train_gene_context import (  # noqa: E402
    EVENT_SUBSET_THRESHOLDS,
    resolve_event_filters,
)
from src.branching_flows.gene_context import EmbryoFutureSetModel, PooledLatentCanonicalizer  # noqa: E402
from src.data.gene_context_dataset import EmbryoViewDataset, collate_embryo_view  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="Existing future-set or compatible checkpoint for config reuse.")
    p.add_argument("--split", choices=["train", "all"], default="train")
    p.add_argument("--samples_per_pair_override", type=int, default=None)
    p.add_argument("--event_subset_override", choices=["none", *sorted(EVENT_SUBSET_THRESHOLDS)], default="none")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--mode", choices=["diag_standardize", "pca_whiten"], default="pca_whiten")
    p.add_argument("--eps", type=float, default=1e-4)
    p.add_argument("--max_batches", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_path", required=True)
    return p.parse_args()


def build_dataset(config: dict, split: str, samples_per_pair: int, event_subset_override: str) -> EmbryoViewDataset:
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
    filters = resolve_event_filters(ns, prefix="val_" if split != "train" else "")
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
        random_seed=config["seed"],
        sampling_strategy=config["sampling_strategy"],
        min_spatial_cells_per_window=config["min_spatial_cells_per_window"],
        spatial_neighbor_pool_size=config["spatial_neighbor_pool_size"],
        delete_target_mode=config["delete_target_mode"],
        views_per_embryo=config["views_per_embryo"],
        future_views_per_embryo=config["future_views_per_embryo"],
        top_cell_types=config.get("top_cell_types", 8),
        **filters,
    )


def fit_transform(pooled_latents: torch.Tensor, mode: str, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    mean = pooled_latents.mean(dim=0)
    centered = pooled_latents - mean
    dim = pooled_latents.shape[-1]
    if mode == "diag_standardize":
        std = centered.std(dim=0, unbiased=False).clamp_min(eps)
        transform = torch.diag(1.0 / std)
        return mean, transform
    cov = centered.T @ centered / max(1, pooled_latents.shape[0] - 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    inv_sqrt = torch.rsqrt(eigvals.clamp_min(0.0) + eps)
    transform = torch.diag(inv_sqrt) @ eigvecs.T
    if transform.shape != (dim, dim):
        raise RuntimeError("unexpected pooled transform shape")
    return mean, transform


def summarize_canonical_latents(canonical_latents: torch.Tensor) -> dict[str, float]:
    centered = canonical_latents - canonical_latents.mean(dim=0, keepdim=True)
    std = centered.std(dim=0, unbiased=False)
    if canonical_latents.shape[0] < 2:
        cov = torch.zeros(
            canonical_latents.shape[-1],
            canonical_latents.shape[-1],
            dtype=canonical_latents.dtype,
        )
    else:
        cov = centered.T @ centered / max(1, canonical_latents.shape[0] - 1)
    off_diag = cov - torch.diag_embed(torch.diagonal(cov))
    return {
        "canonical_std_mean": float(std.mean().item()),
        "canonical_std_min": float(std.min().item()),
        "canonical_cov_offdiag_mse": float(off_diag.pow(2).mean().item()),
    }


def main():
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config = ckpt["config"]

    class _BackboneArgs:
        pass

    backbone_args = _BackboneArgs()
    for key, value in config.items():
        setattr(backbone_args, key, value)
    backbone_args.backbone_checkpoint = ckpt["backbone_checkpoint"]
    backbone, _ = load_backbone(backbone_args)
    backbone.to(args.device)
    backbone.eval()

    samples_per_pair = args.samples_per_pair_override or int(config["samples_per_pair"])
    dataset = build_dataset(config, args.split, samples_per_pair, args.event_subset_override)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_embryo_view)

    pooled_latents = []
    split_count_scale = float(backbone.local_model.context_size)
    for batch_idx, batch in enumerate(loader):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break
        batch = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        n_future_views = int(batch["future_views_per_embryo"][0].item())
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
        masked_future_view_mask = build_mask(
            future_genes.shape[0],
            n_future_views,
            float(config["future_mask_ratio"]),
            future_genes.device,
            allow_empty=False,
        )
        with torch.no_grad():
            future_local_latents, _ = backbone.encode_local_views_with_tokens(
                genes=future_genes,
                time=future_time,
                token_times=future_token_times,
                valid_mask=future_valid_mask,
                anchor_mask=future_anchor_mask,
                context_role=future_context_role,
                relative_position=future_relative_position,
            )
        (
            target_future_set_latents,
            _target_future_set_genes,
            target_future_mass,
            _target_future_split_fraction,
            target_future_survival,
            _target_future_split_count,
        ) = EmbryoFutureSetModel.gather_masked_future_targets(
            future_local_latents,
            future_genes,
            future_split_fraction,
            future_valid_mask,
            masked_future_view_mask,
            split_count_scale=split_count_scale,
        )
        target_future_weights = target_future_mass * target_future_survival
        pooled_latents.append(
            EmbryoFutureSetModel.weighted_pool(target_future_set_latents, target_future_weights).cpu()
        )

    pooled_latents = torch.cat(pooled_latents, dim=0)
    mean, transform = fit_transform(pooled_latents, args.mode, args.eps)
    canonicalizer = PooledLatentCanonicalizer(
        dim=pooled_latents.shape[-1],
        mean=mean,
        transform=transform,
        mode=args.mode,
    )
    canonical_latents = canonicalizer(pooled_latents)
    summary = summarize_canonical_latents(canonical_latents)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "canonicalizer_state_dict": canonicalizer.state_dict(),
            "dim": pooled_latents.shape[-1],
            "mode": args.mode,
            "eps": args.eps,
            "source_checkpoint": args.checkpoint,
            "n_samples": int(pooled_latents.shape[0]),
            "summary": summary,
        },
        output_path,
    )
    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "mode": args.mode,
                "n_samples": int(pooled_latents.shape[0]),
                **summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
