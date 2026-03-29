#!/usr/bin/env python3
"""Fit a frozen linear readout from true future token states to true pooled future latents."""

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
from src.branching_flows.gene_context import EmbryoFutureSetModel, FrozenLinearTokenReadout  # noqa: E402
from src.data.gene_context_dataset import EmbryoViewDataset, collate_embryo_view  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Fit frozen linear token readout anchor for embryo future-set world models.")
    p.add_argument("--backbone_checkpoint", required=True)
    p.add_argument("--h5ad_path", default="dataset/processed/nema_extended_large2025.h5ad")
    p.add_argument("--n_hvg", type=int, default=256)
    p.add_argument("--context_size", type=int, default=256)
    p.add_argument("--global_context_size", type=int, default=32)
    p.add_argument("--dt_minutes", type=float, default=40.0)
    p.add_argument("--time_window_minutes", type=float, default=10.0)
    p.add_argument("--samples_per_pair", type=int, default=16)
    p.add_argument("--views_per_embryo", type=int, default=8)
    p.add_argument("--future_views_per_embryo", type=int, default=8)
    p.add_argument("--top_cell_types", type=int, default=8)
    p.add_argument("--future_mask_ratio", type=float, default=0.25)
    p.add_argument("--min_cells_per_window", type=int, default=32)
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument(
        "--sampling_strategy",
        choices=["random_window", "spatial_neighbors", "spatial_anchor"],
        default="spatial_anchor",
    )
    p.add_argument("--min_spatial_cells_per_window", type=int, default=8)
    p.add_argument("--spatial_neighbor_pool_size", type=int, default=None)
    p.add_argument("--delete_target_mode", choices=["weak", "strict"], default="strict")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--ridge_lambda", type=float, default=1e-3)
    p.add_argument("--max_batches", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_path", required=True)
    return p.parse_args()


def gather_masked_future_view_tensor(tensor: torch.Tensor, masked_future_view_mask: torch.Tensor) -> torch.Tensor:
    gathered = []
    for i in range(tensor.shape[0]):
        masked_idx = torch.nonzero(masked_future_view_mask[i], as_tuple=False).squeeze(-1)
        if masked_idx.numel() == 0:
            raise ValueError("Each sample must mask at least one future view")
        gathered.append(tensor[i, masked_idx])
    return torch.stack(gathered, dim=0)


def fit_affine_readout(x: torch.Tensor, y: torch.Tensor, ridge_lambda: float) -> tuple[torch.Tensor, torch.Tensor]:
    ones = torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
    design = torch.cat([x, ones], dim=1)
    gram = design.T @ design
    reg = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device) * ridge_lambda
    reg[-1, -1] = 0.0
    rhs = design.T @ y
    solution = torch.linalg.solve(gram + reg, rhs)
    weight = solution[:-1].T.contiguous()
    bias = solution[-1].contiguous()
    return weight, bias


def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    ss_res = float(torch.sum((y_true - y_pred) ** 2).item())
    ss_tot = float(torch.sum((y_true - y_true.mean(dim=0, keepdim=True)) ** 2).item())
    if ss_tot <= 1e-8:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    class _Args:
        pass

    backbone_args = _Args()
    backbone_args.backbone_checkpoint = args.backbone_checkpoint
    backbone, ckpt = load_backbone(backbone_args)
    backbone.to(args.device)
    backbone.eval()

    dataset = EmbryoViewDataset(
        h5ad_path=args.h5ad_path,
        n_hvg=args.n_hvg,
        context_size=args.context_size,
        global_context_size=args.global_context_size,
        dt_minutes=args.dt_minutes,
        time_window_minutes=args.time_window_minutes,
        samples_per_pair=args.samples_per_pair,
        min_cells_per_window=args.min_cells_per_window,
        split="train",
        val_fraction=args.val_fraction,
        random_seed=args.seed,
        sampling_strategy=args.sampling_strategy,
        min_spatial_cells_per_window=args.min_spatial_cells_per_window,
        spatial_neighbor_pool_size=args.spatial_neighbor_pool_size,
        delete_target_mode=args.delete_target_mode,
        views_per_embryo=args.views_per_embryo,
        future_views_per_embryo=args.future_views_per_embryo,
        top_cell_types=args.top_cell_types,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_embryo_view)

    token_pooled = []
    latent_targets = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            batch = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            n_future_views = int(batch["future_views_per_embryo"][0].item())
            future_genes = stack_view_tensor(batch, "genes", n_future_views, prefix_template="future_view_{i}_")
            future_context_role = stack_view_tensor(
                batch, "context_role", n_future_views, prefix_template="future_view_{i}_"
            )
            future_relative_position = stack_view_tensor(
                batch, "relative_position", n_future_views, prefix_template="future_view_{i}_"
            )
            future_split_fraction = stack_view_tensor(
                batch, "split_fraction", n_future_views, prefix_template="future_view_{i}_"
            )
            future_token_times = stack_view_tensor(
                batch, "token_times", n_future_views, prefix_template="future_view_{i}_"
            )
            future_valid_mask = stack_view_tensor(
                batch, "valid_mask", n_future_views, prefix_template="future_view_{i}_"
            )
            future_anchor_mask = stack_view_tensor(
                batch, "anchor_mask", n_future_views, prefix_template="future_view_{i}_"
            )
            future_time = stack_view_tensor(batch, "time", n_future_views, prefix_template="future_view_{i}_")

            torch.manual_seed(args.seed + batch_idx)
            masked_future_view_mask = build_mask(
                future_genes.shape[0],
                n_future_views,
                args.future_mask_ratio,
                future_genes.device,
                allow_empty=False,
            )

            future_local_latents, future_local_token_states = backbone.encode_local_views_with_tokens(
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
                split_count_scale=float(args.context_size),
            )
            masked_future_valid_mask = gather_masked_future_view_tensor(future_valid_mask, masked_future_view_mask)
            masked_future_token_states = gather_masked_future_view_tensor(
                future_local_token_states,
                masked_future_view_mask,
            )
            pooled_tokens = FrozenLinearTokenReadout.pool_token_states(
                masked_future_token_states,
                masked_future_valid_mask,
            )
            target_future_weights = target_future_mass * target_future_survival
            target_pooled_latent = EmbryoFutureSetModel.weighted_pool(
                target_future_set_latents,
                target_future_weights,
            )
            token_pooled.append(pooled_tokens.cpu())
            latent_targets.append(target_pooled_latent.cpu())

    x = torch.cat(token_pooled, dim=0)
    y = torch.cat(latent_targets, dim=0)
    weight, bias = fit_affine_readout(x, y, ridge_lambda=args.ridge_lambda)
    pred = torch.nn.functional.linear(x, weight, bias)
    mse = float(torch.mean((pred - y) ** 2).item())
    r2 = r2_score(y, pred)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "token_readout_state_dict": {
                "weight": weight.cpu(),
                "bias": bias.cpu(),
            },
            "config": vars(args),
            "fit_metrics": {
                "mse": mse,
                "r2": r2,
                "n_samples": int(x.shape[0]),
            },
        },
        output_path,
    )
    print(json.dumps({"mse": mse, "r2": r2, "n_samples": int(x.shape[0])}, indent=2))


if __name__ == "__main__":
    main()
