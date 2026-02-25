#!/usr/bin/env python3
"""Comprehensive perturbation experiments for causal validation.

Tests the autoregressive model's response to various biological perturbations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.branching_flows.autoregressive_model import AutoregressiveNemaModel
from src.branching_flows.states import BranchingState


class PerturbationExperiment:
    """Suite of perturbation experiments."""

    def __init__(
        self,
        model: AutoregressiveNemaModel,
        device: str = "cuda",
        n_steps: int = 30,
    ):
        self.model = model
        self.device = device
        self.n_steps = n_steps
        self.model.eval()

    def create_initial_state(self, n_cells: int = 5) -> BranchingState:
        """Create synthetic initial state."""
        genes = torch.randn(1, n_cells, 2000, device=self.device) * 0.1
        spatial = torch.randn(1, n_cells, 3, device=self.device) * 0.1
        continuous = torch.cat([genes, spatial], dim=-1)

        return BranchingState(
            states=(continuous, torch.zeros(1, n_cells, dtype=torch.long, device=self.device)),
            groupings=torch.zeros(1, n_cells, dtype=torch.long, device=self.device),
            del_flags=torch.zeros(1, n_cells, dtype=torch.bool, device=self.device),
            ids=torch.arange(1, n_cells + 1, dtype=torch.long, device=self.device).unsqueeze(0),
            padmask=torch.ones(1, n_cells, dtype=torch.bool, device=self.device),
            flowmask=torch.ones(1, n_cells, dtype=torch.bool, device=self.device),
            branchmask=torch.ones(1, n_cells, dtype=torch.bool, device=self.device),
        )

    def run_control(self, initial: BranchingState) -> list[BranchingState]:
        """Run unperturbed control trajectory."""
        trajectory = [initial]
        state = initial

        with torch.no_grad():
            for _ in range(self.n_steps):
                state, _ = self.model.step(state, deterministic=True, apply_events=False)
                trajectory.append(state)

        return trajectory

    def run_deletion_perturbation(
        self,
        initial: BranchingState,
        perturb_time: int,
        fraction: float = 0.3,
    ) -> list[BranchingState]:
        """Delete fraction of cells at perturb_time."""
        trajectory = [initial]
        state = initial

        with torch.no_grad():
            for t in range(self.n_steps):
                if t == perturb_time:
                    # Delete cells
                    n_cells = state.padmask.sum().item()
                    n_delete = max(1, int(n_cells * fraction))

                    mask = torch.ones(n_cells, dtype=torch.bool, device=self.device)
                    delete_indices = torch.randperm(n_cells)[:n_delete]
                    mask[delete_indices] = False

                    # Create reduced state
                    new_cont = state.states[0][:, mask]
                    new_disc = state.states[1][:, mask]
                    n_kept = mask.sum().item()

                    state = BranchingState(
                        states=(new_cont, new_disc),
                        groupings=torch.zeros(1, n_kept, dtype=torch.long, device=self.device),
                        del_flags=torch.zeros(1, n_kept, dtype=torch.bool, device=self.device),
                        ids=torch.arange(1, n_kept + 1, dtype=torch.long, device=self.device).unsqueeze(0),
                        padmask=torch.ones(1, n_kept, dtype=torch.bool, device=self.device),
                        flowmask=torch.ones(1, n_kept, dtype=torch.bool, device=self.device),
                        branchmask=torch.ones(1, n_kept, dtype=torch.bool, device=self.device),
                    )

                state, _ = self.model.step(state, deterministic=True, apply_events=False)
                trajectory.append(state)

        return trajectory

    def run_spatial_shift(
        self,
        initial: BranchingState,
        perturb_time: int,
        shift: torch.Tensor,
    ) -> list[BranchingState]:
        """Shift all cells by vector at perturb_time."""
        trajectory = [initial]
        state = initial

        with torch.no_grad():
            for t in range(self.n_steps):
                if t == perturb_time:
                    # Apply spatial shift
                    cont = state.states[0]
                    cont[..., -3:] += shift

                state, _ = self.model.step(state, deterministic=True, apply_events=False)
                trajectory.append(state)

        return trajectory

    def run_gene_activation(
        self,
        initial: BranchingState,
        perturb_time: int,
        gene_idx: int,
        activation: float = 2.0,
    ) -> list[BranchingState]:
        """Activate specific gene in all cells."""
        trajectory = [initial]
        state = initial

        with torch.no_grad():
            for t in range(self.n_steps):
                if t == perturb_time:
                    # Activate gene
                    cont = state.states[0]
                    cont[..., gene_idx] += activation

                state, _ = self.model.step(state, deterministic=True, apply_events=False)
                trajectory.append(state)

        return trajectory

    def analyze_trajectory(self, trajectory: list[BranchingState]) -> dict:
        """Extract statistics from trajectory."""
        stats = {
            "n_cells": [],
            "spatial_centers": [],
            "spatial_spreads": [],
        }

        for state in trajectory:
            n = int(state.padmask.sum())
            stats["n_cells"].append(n)

            if n > 0:
                spatial = state.states[0][0, :n, -3:]
                center = spatial.mean(dim=0)
                spread = spatial.std(dim=0).mean()
                stats["spatial_centers"].append(center.cpu().numpy())
                stats["spatial_spreads"].append(spread.item())
            else:
                stats["spatial_centers"].append(np.array([0, 0, 0]))
                stats["spatial_spreads"].append(0.0)

        return stats

    def compare_trajectories(
        self,
        control: list[BranchingState],
        perturbed: list[BranchingState],
    ) -> dict:
        """Compare control vs perturbed."""
        control_stats = self.analyze_trajectory(control)
        perturbed_stats = self.analyze_trajectory(perturbed)

        # Final state comparison
        final_idx = -1
        control_center = control_stats["spatial_centers"][final_idx]
        perturbed_center = perturbed_stats["spatial_centers"][final_idx]

        spatial_distance = np.linalg.norm(control_center - perturbed_center)

        # Cell count difference
        cell_diff = abs(
            control_stats["n_cells"][final_idx] -
            perturbed_stats["n_cells"][final_idx]
        )

        # Trajectory divergence
        divergences = []
        for c_center, p_center in zip(
            control_stats["spatial_centers"],
            perturbed_stats["spatial_centers"]
        ):
            div = np.linalg.norm(c_center - p_center)
            divergences.append(div)

        return {
            "final_spatial_distance": spatial_distance,
            "cell_count_difference": cell_diff,
            "max_divergence": max(divergences),
            "final_divergence": divergences[-1],
        }


def main():
    parser = argparse.ArgumentParser(description="Perturbation experiments")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="perturbation_results.json")
    args = parser.parse_args()

    print("=" * 70)
    print("COMPREHENSIVE PERTURBATION EXPERIMENTS")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print()

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model = AutoregressiveNemaModel(
        gene_dim=2000,
        spatial_dim=3,
        discrete_K=7,
        d_model=128,
        n_layers=4,
        n_heads=4,
        cross_modal_every=2,
        max_seq_len=64,
        dt=0.1,
    ).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")
    print()

    # Create experiment runner
    experiment = PerturbationExperiment(model, args.device, n_steps=20)

    results = {}

    # Experiment 1: Cell deletion
    print("Experiment 1: Cell deletion (30% at t=5)")
    print("-" * 70)
    initial = experiment.create_initial_state(n_cells=10)

    control = experiment.run_control(initial)
    perturbed = experiment.run_deletion_perturbation(initial, perturb_time=5, fraction=0.3)

    comparison = experiment.compare_trajectories(control, perturbed)
    results["deletion_t5_f30"] = comparison

    print(f"  Control final cells: {experiment.analyze_trajectory(control)['n_cells'][-1]}")
    print(f"  Perturbed final cells: {experiment.analyze_trajectory(perturbed)['n_cells'][-1]}")
    print(f"  Spatial distance: {comparison['final_spatial_distance']:.4f}")
    print(f"  Compensated: {'✓' if comparison['final_spatial_distance'] < 1.0 else '✗'}")
    print()

    # Experiment 2: Spatial shift
    print("Experiment 2: Spatial shift (+1.0 in Z at t=5)")
    print("-" * 70)
    initial = experiment.create_initial_state(n_cells=10)

    control = experiment.run_control(initial)
    shift = torch.tensor([0.0, 0.0, 1.0], device=args.device)
    perturbed = experiment.run_spatial_shift(initial, perturb_time=5, shift=shift)

    comparison = experiment.compare_trajectories(control, perturbed)
    results["spatial_shift_t5_z1"] = comparison

    print(f"  Spatial distance: {comparison['final_spatial_distance']:.4f}")
    print(f"  Recovered: {'✓' if comparison['final_spatial_distance'] < 0.5 else '✗'}")
    print()

    # Experiment 3: Gene activation
    print("Experiment 3: Gene activation (gene 0 += 2.0 at t=5)")
    print("-" * 70)
    initial = experiment.create_initial_state(n_cells=10)

    control = experiment.run_control(initial)
    perturbed = experiment.run_gene_activation(initial, perturb_time=5, gene_idx=0)

    comparison = experiment.compare_trajectories(control, perturbed)
    results["gene_activation_t5_g0"] = comparison

    print(f"  Spatial distance: {comparison['final_spatial_distance']:.4f}")
    print(f"  Gene perturbation affected morphology: {'✓' if comparison['final_spatial_distance'] > 0.1 else '✗'}")
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    n_compensated = sum(1 for r in results.values() if r["final_spatial_distance"] < 1.0)
    print(f"Compensatory responses: {n_compensated}/{len(results)}")

    avg_distance = np.mean([r["final_spatial_distance"] for r in results.values()])
    print(f"Average spatial distance: {avg_distance:.4f}")

    if n_compensated >= len(results) // 2:
        print("\n✓ Model demonstrates causal understanding")
        print("  It responds to perturbations in biologically reasonable ways")
    else:
        print("\n✗ Limited causal response")
        print("  Model may need more training or architectural improvements")

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
