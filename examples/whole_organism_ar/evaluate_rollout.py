#!/usr/bin/env python3
"""Unified biological evaluation for whole-organism AR model.

Metrics:
1) Cell count trajectory error
2) Division timing/count error
3) Context perturbation sensitivity across rollout time
4) Predictive uncertainty from diffusion-style noisy inputs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from examples.whole_organism_ar.train_autoregressive_full import (  # noqa: E402
    EmbryoTrajectoryDataset,
    add_diffusion_noise,
    sample_log_uniform_sigma,
)
from src.branching_flows.autoregressive_model import AutoregressiveNemaModel  # noqa: E402
from src.branching_flows.states import BranchingState  # noqa: E402


def count_alive_cells(state: BranchingState) -> int:
    return int(state.padmask[0].sum().item())


def build_subset_state(
    state: BranchingState,
    keep_mask: torch.Tensor,
    device: str,
) -> BranchingState:
    """Build a new BranchingState from selected cells."""
    n_kept = int(keep_mask.sum().item())
    if n_kept == 0:
        valid_indices = torch.where(state.padmask[0])[0]
        if valid_indices.numel() > 0:
            keep_mask = torch.zeros_like(keep_mask)
            keep_mask[valid_indices[0]] = True
            n_kept = 1
            new_cont = state.states[0][:, keep_mask, :]
            new_disc = (
                state.states[1][:, keep_mask] if state.states[1] is not None else None
            )
        else:
            # Fully empty state: synthesize a single placeholder cell.
            d = state.states[0].shape[-1]
            new_cont = torch.zeros(1, 1, d, device=device, dtype=state.states[0].dtype)
            if state.states[1] is not None:
                new_disc = torch.zeros(1, 1, device=device, dtype=state.states[1].dtype)
            else:
                new_disc = None
            n_kept = 1
    else:
        new_cont = state.states[0][:, keep_mask, :]
        new_disc = state.states[1][:, keep_mask] if state.states[1] is not None else None

    if new_disc is None:
        new_states = (new_cont,)
    else:
        new_states = (new_cont, new_disc)

    return BranchingState(
        states=new_states,
        groupings=torch.zeros(1, n_kept, dtype=torch.long, device=device),
        del_flags=torch.zeros(1, n_kept, dtype=torch.bool, device=device),
        ids=torch.arange(1, n_kept + 1, dtype=torch.long, device=device).unsqueeze(0),
        padmask=torch.ones(1, n_kept, dtype=torch.bool, device=device),
        flowmask=torch.ones(1, n_kept, dtype=torch.bool, device=device),
        branchmask=torch.ones(1, n_kept, dtype=torch.bool, device=device),
    )


@torch.no_grad()
def rollout_from_state(
    model: AutoregressiveNemaModel,
    initial_state: BranchingState,
    n_steps: int,
    device: str,
) -> list[BranchingState]:
    model.eval()
    traj = [initial_state.to(device)]
    state = traj[0]
    for _ in range(n_steps):
        state, _ = model.step(state, deterministic=True, apply_events=True)
        traj.append(state)
    return traj


@torch.no_grad()
def evaluate_cell_count_trajectory(
    model: AutoregressiveNemaModel,
    dataset: EmbryoTrajectoryDataset,
    device: str,
    max_steps: int,
) -> dict[str, Any]:
    """Compare predicted vs true cell counts in rollout."""
    if len(dataset) == 0:
        return {"error": "empty dataset"}

    n_eval = min(max_steps, len(dataset.trajectory) - 1)
    initial = dataset[0]["current"].to(device)
    pred_traj = rollout_from_state(model, initial, n_eval, device)

    pred_counts = [count_alive_cells(s) for s in pred_traj]
    true_counts = [int(dataset.trajectory[i]["n_cells"]) for i in range(n_eval + 1)]

    abs_errors = [abs(p - t) for p, t in zip(pred_counts, true_counts)]
    sq_errors = [(p - t) ** 2 for p, t in zip(pred_counts, true_counts)]

    mae = float(sum(abs_errors) / len(abs_errors))
    rmse = float((sum(sq_errors) / len(sq_errors)) ** 0.5)

    return {
        "n_steps": n_eval,
        "pred_counts": pred_counts,
        "true_counts": true_counts,
        "mae": mae,
        "rmse": rmse,
    }


@torch.no_grad()
def evaluate_division_timing(
    model: AutoregressiveNemaModel,
    dataset: EmbryoTrajectoryDataset,
    device: str,
    max_steps: int,
    split_threshold: float,
) -> dict[str, Any]:
    """Evaluate one-step division count prediction error over time."""
    if len(dataset) == 0:
        return {"error": "empty dataset"}

    n_eval = min(max_steps, len(dataset))
    pred_counts = []
    true_counts = []
    abs_errors = []

    for i in range(n_eval):
        sample = dataset[i]
        current = sample["current"].to(device)
        target_split = sample["target_split"].to(device)

        output = model.forward_step(current, sigma=0.0)
        split_probs = torch.sigmoid(output.split_logits[0, :, 0])
        valid = current.padmask[0]
        pred = int((split_probs[valid] > split_threshold).sum().item())
        true = int((target_split[valid.cpu()] > 0.5).sum().item())

        pred_counts.append(pred)
        true_counts.append(true)
        abs_errors.append(abs(pred - true))

    return {
        "n_steps": n_eval,
        "pred_division_counts": pred_counts,
        "true_division_counts": true_counts,
        "count_mae": float(sum(abs_errors) / len(abs_errors)) if abs_errors else 0.0,
    }


@torch.no_grad()
def evaluate_context_perturbation(
    model: AutoregressiveNemaModel,
    dataset: EmbryoTrajectoryDataset,
    device: str,
    max_steps: int,
    perturb_times: list[int],
    removal_fraction: float = 0.2,
) -> dict[str, Any]:
    """Time x removal perturbation grid and rollout sensitivity score."""
    if len(dataset) == 0:
        return {"error": "empty dataset"}

    initial = dataset[0]["current"].to(device)
    baseline = rollout_from_state(model, initial, max_steps, device)
    baseline_total = max(1, count_alive_cells(baseline[-1]))

    grid: dict[str, dict[str, Any]] = {}
    scores = []

    for t_perturb in perturb_times:
        state = initial
        for t in range(max_steps):
            if t == t_perturb:
                valid_indices = torch.where(state.padmask[0])[0]
                if valid_indices.numel() > 1:
                    n_remove = max(1, int(valid_indices.numel() * removal_fraction))
                    remove_indices = valid_indices[:n_remove]
                    keep_mask = state.padmask[0].clone()
                    keep_mask[remove_indices] = False
                    state = build_subset_state(state, keep_mask, device)
            state, _ = model.step(state, deterministic=True, apply_events=True)

        perturbed_total = count_alive_cells(state)
        score = float(abs(perturbed_total - baseline_total) / baseline_total)
        scores.append(score)
        grid[str(t_perturb)] = {
            "baseline_final_cell_count": baseline_total,
            "perturbed_final_cell_count": perturbed_total,
            "relative_cell_count_change": score,
        }

    return {
        "max_steps": max_steps,
        "perturb_times": perturb_times,
        "removal_fraction": removal_fraction,
        "mean_relative_cell_count_change": float(sum(scores) / len(scores))
        if scores
        else 0.0,
        "grid": grid,
    }


@torch.no_grad()
def evaluate_predictive_uncertainty(
    model: AutoregressiveNemaModel,
    dataset: EmbryoTrajectoryDataset,
    device: str,
    n_states: int,
    n_samples_per_state: int,
    sigma_min: float,
    sigma_max: float,
) -> dict[str, Any]:
    """Estimate uncertainty from noisy inputs and sigma-conditioned predictions."""
    if len(dataset) == 0:
        return {"error": "empty dataset"}

    n_eval_states = min(n_states, len(dataset))
    per_state = []

    for i in range(n_eval_states):
        current = dataset[i]["current"].to(device)
        preds_gene = []
        preds_spatial = []

        for _ in range(n_samples_per_state):
            sigma = sample_log_uniform_sigma(sigma_min, sigma_max, device)
            noisy_state, _ = add_diffusion_noise(current, sigma)
            output = model.forward_step(noisy_state, sigma=sigma)
            valid = current.padmask[0]
            preds_gene.append(output.gene_delta[0, valid, :])
            preds_spatial.append(output.spatial_vel[0, valid, :])

        gene_stack = torch.stack(preds_gene, dim=0)
        spatial_stack = torch.stack(preds_spatial, dim=0)

        gene_std = gene_stack.std(dim=0).mean().item()
        spatial_std = spatial_stack.std(dim=0).mean().item()
        per_state.append(
            {
                "state_index": i,
                "gene_delta_std": float(gene_std),
                "spatial_vel_std": float(spatial_std),
            }
        )

    return {
        "n_states": n_eval_states,
        "n_samples_per_state": n_samples_per_state,
        "mean_gene_delta_std": float(
            sum(x["gene_delta_std"] for x in per_state) / len(per_state)
        )
        if per_state
        else 0.0,
        "mean_spatial_vel_std": float(
            sum(x["spatial_vel_std"] for x in per_state) / len(per_state)
        )
        if per_state
        else 0.0,
        "per_state": per_state,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate whole-organism AR model.")
    parser.add_argument("--trajectory_file", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gene_dim", type=int, default=2000)
    parser.add_argument("--spatial_dim", type=int, default=3)
    parser.add_argument("--discrete_k", type=int, default=7)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--cross_modal_every", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument(
        "--deterministic_topk_events",
        action="store_true",
        help="Use top-k event selection at deterministic inference time for ablations.",
    )
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--split_threshold", type=float, default=0.5)
    parser.add_argument("--perturb_times", type=str, default="2,4,6")
    parser.add_argument("--perturb_fraction", type=float, default=0.2)
    parser.add_argument("--uncertainty_states", type=int, default=5)
    parser.add_argument("--uncertainty_samples", type=int, default=8)
    parser.add_argument("--sigma_min", type=float, default=0.01)
    parser.add_argument("--sigma_max", type=float, default=0.2)
    parser.add_argument("--strict_load", action="store_true")
    parser.add_argument(
        "--output",
        type=str,
        default="result/autoregressive_results/evaluation_rollout.json",
    )
    args = parser.parse_args()

    dataset = EmbryoTrajectoryDataset(args.trajectory_file)
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Please check trajectory file.")

    model = AutoregressiveNemaModel(
        gene_dim=args.gene_dim,
        spatial_dim=args.spatial_dim,
        discrete_K=args.discrete_k,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        cross_modal_every=args.cross_modal_every,
        max_seq_len=args.max_seq_len,
        dt=args.dt,
        deterministic_topk_events=args.deterministic_topk_events,
    ).to(args.device)

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    load_result = model.load_state_dict(state_dict, strict=args.strict_load)
    model.eval()

    perturb_times = [int(x.strip()) for x in args.perturb_times.split(",") if x.strip()]

    report = {
        "config": {
            "trajectory_file": args.trajectory_file,
            "checkpoint": args.checkpoint,
            "max_steps": args.max_steps,
            "split_threshold": args.split_threshold,
            "perturb_times": perturb_times,
            "uncertainty_states": args.uncertainty_states,
            "uncertainty_samples": args.uncertainty_samples,
            "sigma_min": args.sigma_min,
            "sigma_max": args.sigma_max,
            "strict_load": args.strict_load,
            "model": {
                "gene_dim": args.gene_dim,
                "spatial_dim": args.spatial_dim,
                "discrete_k": args.discrete_k,
                "d_model": args.d_model,
                "n_layers": args.n_layers,
                "n_heads": args.n_heads,
                "cross_modal_every": args.cross_modal_every,
                "max_seq_len": args.max_seq_len,
                "dt": args.dt,
            },
        },
        "checkpoint_load": {
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
        },
    }

    report["cell_count_trajectory"] = evaluate_cell_count_trajectory(
        model=model,
        dataset=dataset,
        device=args.device,
        max_steps=args.max_steps,
    )
    report["division_timing"] = evaluate_division_timing(
        model=model,
        dataset=dataset,
        device=args.device,
        max_steps=args.max_steps,
        split_threshold=args.split_threshold,
    )
    report["context_perturbation"] = evaluate_context_perturbation(
        model=model,
        dataset=dataset,
        device=args.device,
        max_steps=args.max_steps,
        perturb_times=perturb_times,
        removal_fraction=args.perturb_fraction,
    )
    report["predictive_uncertainty"] = evaluate_predictive_uncertainty(
        model=model,
        dataset=dataset,
        device=args.device,
        n_states=args.uncertainty_states,
        n_samples_per_state=args.uncertainty_samples,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 70)
    print("Evaluation complete")
    print(f"Output: {out_path}")
    print("- Cell count MAE:", report["cell_count_trajectory"].get("mae"))
    print("- Division count MAE:", report["division_timing"].get("count_mae"))
    print(
        "- Mean context sensitivity:",
        report["context_perturbation"].get("mean_relative_cell_count_change"),
    )
    print(
        "- Mean uncertainty (gene/spatial):",
        report["predictive_uncertainty"].get("mean_gene_delta_std"),
        "/",
        report["predictive_uncertainty"].get("mean_spatial_vel_std"),
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
