#!/usr/bin/env python3
"""Diagnose future embryo latent stability under resampling and view permutation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from examples.whole_organism_ar.train_gene_context import (  # noqa: E402
    EVENT_SUBSET_THRESHOLDS,
    resolve_event_filters,
)
from src.branching_flows.gene_context import EmbryoMaskedViewModel  # noqa: E402
from src.data.gene_context_dataset import EmbryoViewDataset  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Analyze future embryo latent target stability.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=["train", "val", "all"], default="all")
    p.add_argument("--event_subset_override", choices=["none", *sorted(EVENT_SUBSET_THRESHOLDS)], default="none")
    p.add_argument("--samples_per_pair_override", type=int, default=16)
    p.add_argument("--n_resamples", type=int, default=8)
    p.add_argument("--n_permutations", type=int, default=8)
    p.add_argument("--max_pairs", type=int, default=40)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_json", required=True)
    return p.parse_args()


def cosine_matrix(x: torch.Tensor) -> torch.Tensor:
    x = torch.nn.functional.normalize(x, dim=-1)
    return x @ x.T


def mean_upper_triangular(x: torch.Tensor) -> float:
    n = x.shape[0]
    if n < 2:
        return 1.0
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=x.device), diagonal=1)
    return float(x[mask].mean().item())


def build_dataset(args, config: dict, random_seed: int) -> EmbryoViewDataset:
    class _Args:
        pass

    ns = _Args()
    for key, value in config.items():
        setattr(ns, key, value)
    ns.event_subset = args.event_subset_override
    ns.val_event_subset = args.event_subset_override
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
    filters = resolve_event_filters(ns, prefix="val_" if args.split == "val" else "")

    top_cell_type_cfg = config.get("top_cell_types", 8)
    if isinstance(top_cell_type_cfg, int):
        top_cell_types = top_cell_type_cfg
    else:
        top_cell_types = max(8, len(top_cell_type_cfg))

    return EmbryoViewDataset(
        h5ad_path=config["h5ad_path"],
        n_hvg=config["n_hvg"],
        context_size=config["context_size"],
        global_context_size=config["global_context_size"],
        dt_minutes=config["dt_minutes"],
        time_window_minutes=config["time_window_minutes"],
        samples_per_pair=args.samples_per_pair_override,
        min_cells_per_window=config["min_cells_per_window"],
        split=args.split,
        val_fraction=config["val_fraction"],
        random_seed=random_seed,
        sampling_strategy=config["sampling_strategy"],
        min_spatial_cells_per_window=config["min_spatial_cells_per_window"],
        spatial_neighbor_pool_size=config["spatial_neighbor_pool_size"],
        delete_target_mode=config["delete_target_mode"],
        views_per_embryo=config["views_per_embryo"],
        future_views_per_embryo=config["future_views_per_embryo"],
        top_cell_types=top_cell_types,
        **filters,
    )


def stack_future_views(item: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    n_future_views = int(item["future_views_per_embryo"].item())
    genes = []
    context_role = []
    relative_position = []
    token_times = []
    valid_mask = []
    anchor_mask = []
    time = []
    for i in range(n_future_views):
        prefix = f"future_view_{i}"
        genes.append(item[f"{prefix}_genes"])
        context_role.append(item[f"{prefix}_context_role"])
        relative_position.append(item[f"{prefix}_relative_position"])
        token_times.append(item[f"{prefix}_token_times"])
        valid_mask.append(item[f"{prefix}_valid_mask"])
        anchor_mask.append(item[f"{prefix}_anchor_mask"])
        time.append(item[f"{prefix}_time"])
    return (
        torch.stack(genes, dim=0),
        torch.stack(context_role, dim=0),
        torch.stack(relative_position, dim=0),
        torch.stack(token_times, dim=0),
        torch.stack(valid_mask, dim=0),
        torch.stack(anchor_mask, dim=0),
        torch.stack(time, dim=0),
    )


def encode_future_latent(
    model: EmbryoMaskedViewModel,
    genes: torch.Tensor,
    context_role: torch.Tensor,
    relative_position: torch.Tensor,
    token_times: torch.Tensor,
    valid_mask: torch.Tensor,
    anchor_mask: torch.Tensor,
    time: torch.Tensor,
) -> torch.Tensor:
    latent, _ = model.encode_embryo_latent(
        genes=genes.unsqueeze(0),
        time=time.unsqueeze(0),
        token_times=token_times.unsqueeze(0),
        valid_mask=valid_mask.unsqueeze(0),
        anchor_mask=anchor_mask.unsqueeze(0),
        context_role=context_role.unsqueeze(0),
        relative_position=relative_position.unsqueeze(0),
    )
    return latent.squeeze(0)


def main():
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config = ckpt["config"]

    dataset = build_dataset(args, config, random_seed=config["seed"] + 1000)
    model = EmbryoMaskedViewModel(
        gene_dim=ckpt["gene_dim"],
        context_size=config["context_size"],
        model_type=config["model_type"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        head_dim=config["head_dim"],
        use_pairwise_spatial_bias=config["pairwise_spatial_bias"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(args.device)
    model.eval()

    n_pairs = min(len(dataset.time_pairs), args.max_pairs)
    pair_latent_means = []
    pair_results = []
    order_sensitivities = []
    resample_cosines = []
    resample_radii = []

    with torch.no_grad():
        for pair_idx in range(n_pairs):
            sample_latents = []
            first_sample_views = None
            for rep in range(args.n_resamples):
                item_idx = pair_idx + rep * len(dataset.time_pairs)
                item = dataset[item_idx]
                views = stack_future_views(item)
                if first_sample_views is None:
                    first_sample_views = views
                latent = encode_future_latent(model, *(x.to(args.device) for x in views))
                sample_latents.append(latent)

            latents = torch.stack(sample_latents, dim=0)
            pair_mean = latents.mean(dim=0)
            pair_latent_means.append(pair_mean)
            cos = cosine_matrix(latents)
            within_cos = mean_upper_triangular(cos)
            within_radius = float(torch.linalg.norm(latents - pair_mean.unsqueeze(0), dim=-1).mean().item())
            resample_cosines.append(within_cos)
            resample_radii.append(within_radius)

            # Order sensitivity on one fixed sampled future view set.
            base_views = first_sample_views
            base_latent = encode_future_latent(model, *(x.to(args.device) for x in base_views))
            perm_cosines = []
            n_views = base_views[0].shape[0]
            for _ in range(args.n_permutations):
                perm = torch.randperm(n_views)
                permuted = tuple(x[perm] for x in base_views)
                perm_latent = encode_future_latent(model, *(x.to(args.device) for x in permuted))
                perm_cos = torch.nn.functional.cosine_similarity(
                    base_latent.unsqueeze(0), perm_latent.unsqueeze(0), dim=-1
                )
                perm_cosines.append(float(perm_cos.item()))
            order_sensitivity = 1.0 - float(np.mean(perm_cosines))
            order_sensitivities.append(order_sensitivity)

            pair_results.append(
                {
                    "pair_idx": int(pair_idx),
                    "current_center_min": float(dataset.time_pairs[pair_idx].current_center),
                    "future_center_min": float(dataset.time_pairs[pair_idx].future_center),
                    "within_pair_resample_cosine_mean": within_cos,
                    "within_pair_resample_radius_mean": within_radius,
                    "order_sensitivity_1_minus_cosine": order_sensitivity,
                }
            )

    pair_latent_means = torch.stack(pair_latent_means, dim=0)
    between_pair_cosine = mean_upper_triangular(cosine_matrix(pair_latent_means))

    output = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "n_pairs_evaluated": n_pairs,
        "n_resamples": args.n_resamples,
        "n_permutations": args.n_permutations,
        "future_latent_stability": {
            "mean_within_pair_resample_cosine": float(np.mean(resample_cosines)),
            "std_within_pair_resample_cosine": float(np.std(resample_cosines)),
            "mean_within_pair_resample_radius": float(np.mean(resample_radii)),
            "std_within_pair_resample_radius": float(np.std(resample_radii)),
            "mean_between_pair_centroid_cosine": between_pair_cosine,
            "resample_cosine_minus_between_pair_cosine": float(np.mean(resample_cosines) - between_pair_cosine),
        },
        "order_sensitivity": {
            "mean_1_minus_cosine": float(np.mean(order_sensitivities)),
            "std_1_minus_cosine": float(np.std(order_sensitivities)),
        },
        "per_pair": pair_results,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
