#!/usr/bin/env python3
"""Evaluate the spatial-only engineering rollout baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from examples.whole_organism_ar.train_spatial_rollout import SpatialTrajectoryDataset  # noqa: E402
from src.branching_flows.spatial_ar import SpatialAutoregressiveModel  # noqa: E402
from src.branching_flows.states import BranchingState  # noqa: E402


def count_alive_cells(state: BranchingState) -> int:
    return int(state.padmask[0].sum().item())


@torch.no_grad()
def rollout_from_state(
    model: SpatialAutoregressiveModel,
    initial_state: BranchingState,
    n_steps: int,
    device: str,
) -> list[BranchingState]:
    state = initial_state.to(device)
    trajectory = [state]
    for _ in range(n_steps):
        state, _ = model.step(state, deterministic=True, apply_events=True)
        trajectory.append(state)
    return trajectory


@torch.no_grad()
def evaluate_counts(
    model: SpatialAutoregressiveModel,
    dataset: SpatialTrajectoryDataset,
    device: str,
    max_steps: int,
) -> dict[str, Any]:
    n_eval = min(max_steps, len(dataset.trajectory) - 1)
    initial = dataset[0]["current"].to(device)
    pred_traj = rollout_from_state(model, initial, n_eval, device)
    pred_counts = [count_alive_cells(state) for state in pred_traj]
    true_counts = [int(dataset.trajectory[i]["n_cells"]) for i in range(n_eval + 1)]
    abs_errors = [abs(pred - true) for pred, true in zip(pred_counts, true_counts)]
    return {
        "pred_counts": pred_counts,
        "true_counts": true_counts,
        "mae": float(sum(abs_errors) / len(abs_errors)),
    }


@torch.no_grad()
def evaluate_divisions(
    model: SpatialAutoregressiveModel,
    dataset: SpatialTrajectoryDataset,
    device: str,
    max_steps: int,
    split_threshold: float,
) -> dict[str, Any]:
    n_eval = min(max_steps, len(dataset))
    pred_counts = []
    true_counts = []
    abs_errors = []
    for idx in range(n_eval):
        sample = dataset[idx]
        current = sample["current"].to(device)
        target_split = sample["target_split"]
        output = model.forward_step(current)
        split_probs = torch.sigmoid(output.split_logits[0, :, 0])
        valid = current.padmask[0]
        pred = int((split_probs[valid] > split_threshold).sum().item())
        true = int(target_split[valid.cpu()].sum().item())
        pred_counts.append(pred)
        true_counts.append(true)
        abs_errors.append(abs(pred - true))
    return {
        "pred_division_counts": pred_counts,
        "true_division_counts": true_counts,
        "count_mae": float(sum(abs_errors) / len(abs_errors)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the spatial-only engineering rollout baseline."
    )
    parser.add_argument("--trajectory_file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--include_velocity", action="store_true")
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--split_threshold", type=float, default=0.5)
    parser.add_argument("--del_threshold", type=float, default=0.7)
    parser.add_argument("--deterministic_topk_events", action="store_true")
    parser.add_argument(
        "--output",
        default="result/autoregressive_results/evaluation_spatial_rollout.json",
    )
    args = parser.parse_args()

    dataset = SpatialTrajectoryDataset(
        args.trajectory_file,
        include_velocity=args.include_velocity,
    )
    continuous_dim = 6 if args.include_velocity else 3
    model = SpatialAutoregressiveModel(
        continuous_dim=continuous_dim,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_len=args.max_seq_len,
        deterministic_topk_events=args.deterministic_topk_events,
    ).to(args.device)
    model.cell_manager.split_threshold = args.split_threshold
    model.cell_manager.del_threshold = args.del_threshold

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    report = {
        "config": vars(args),
        "cell_count_trajectory": evaluate_counts(
            model,
            dataset,
            args.device,
            args.max_steps,
        ),
        "division_timing": evaluate_divisions(
            model,
            dataset,
            args.device,
            args.max_steps,
            args.split_threshold,
        ),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 70)
    print("Spatial rollout evaluation complete")
    print(f"Output: {out_path}")
    print("- Cell count MAE:", report["cell_count_trajectory"]["mae"])
    print("- Division count MAE:", report["division_timing"]["count_mae"])
    print("=" * 70)


if __name__ == "__main__":
    main()
