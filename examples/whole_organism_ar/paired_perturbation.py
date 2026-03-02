#!/usr/bin/env python3
"""
Paired perturbation experiment with identical initial conditions.

Runs two simulations from the same starting state:
- Control: normal development
- Perturbed: cells deleted at t=0.5

This allows fair comparison of compensatory behavior.
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

from src.branching_flows.crossmodal_model import CrossModalNemaModel
from autoregressive_simulation import (
    AutoregressiveSimulator,
    create_initial_state,
    BranchingState,
)


def main():
    parser = argparse.ArgumentParser(description="Paired perturbation experiment")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints_trimodal_crossmodal/best.pt"
    )
    parser.add_argument("--initial_cells", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--perturbation_time", type=float, default=0.5)
    parser.add_argument("--deletion_fraction", type=float, default=0.25)
    parser.add_argument("--output", type=str, default="paired_results")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print("=" * 70)
    print("PAIRED PERTURBATION EXPERIMENT")
    print("=" * 70)
    print(f"Initial cells: {args.initial_cells}")
    print(
        f"Will delete {args.deletion_fraction * 100:.0f}% at t={args.perturbation_time}"
    )
    print()

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model = CrossModalNemaModel(
        gene_dim=2000,
        spatial_dim=3,
        discrete_K=7,
        d_model=256,
        n_layers=6,
        n_heads=8,
        cross_modal_every=2,
    ).to(args.device)
    model.load_state_dict(
        checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    )

    simulator = AutoregressiveSimulator(model, args.device, args.dt)

    # Create shared initial state
    initial_state = create_initial_state(args.initial_cells, 2000, args.device)
    print(
        f"Initial spatial center: {initial_state.states[0][0, :, -3:].mean(dim=0).cpu().numpy()}"
    )
    print()

    # Run control
    print("Running CONTROL simulation...")
    control_results = simulator.simulate(initial_state, 0.0, 1.0)
    print(f"  Final: {control_results['trajectory'][-1]['n_cells']} cells")

    # Define perturbation
    def deletion_perturb(state: BranchingState) -> BranchingState:
        n_cells = state.padmask.sum().item()
        n_delete = max(1, int(n_cells * args.deletion_fraction))

        mask = torch.ones(n_cells, dtype=torch.bool, device=state.states[0].device)
        mask[torch.randperm(n_cells)[:n_delete]] = False

        n_kept = mask.sum()
        new_cont = state.states[0][:, mask]
        new_disc = state.states[1][:, mask] if state.states[1] is not None else None

        print(
            f"  [t={args.perturbation_time}] Deleted {n_delete} cells, {n_kept} remain"
        )

        return BranchingState(
            states=(new_cont, new_disc),
            groupings=torch.zeros(
                1, n_kept, dtype=torch.long, device=state.states[0].device
            ),
            del_flags=torch.zeros(
                1, n_kept, dtype=torch.bool, device=state.states[0].device
            ),
            ids=torch.arange(
                1, n_kept + 1, dtype=torch.long, device=state.states[0].device
            ).unsqueeze(0),
            padmask=torch.ones(
                1, n_kept, dtype=torch.bool, device=state.states[0].device
            ),
            flowmask=torch.ones(
                1, n_kept, dtype=torch.bool, device=state.states[0].device
            ),
            branchmask=torch.ones(
                1, n_kept, dtype=torch.bool, device=state.states[0].device
            ),
        )

    # Run perturbed
    print("\nRunning PERTURBED simulation...")
    perturbed_results = simulator.simulate(
        initial_state,
        0.0,
        1.0,
        perturbation=deletion_perturb,
        perturbation_time=args.perturbation_time,
    )
    print(f"  Final: {perturbed_results['trajectory'][-1]['n_cells']} cells")

    # Compare
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    ctrl_traj = control_results["trajectory"]
    pert_traj = perturbed_results["trajectory"]

    print(
        f"{'Time':<8} {'Control (cells)':<20} {'Perturbed (cells)':<20} {'Distance':<15}"
    )
    print("-" * 70)

    for i in range(0, len(ctrl_traj), max(1, len(ctrl_traj) // 10)):
        ctrl, pert = ctrl_traj[i], pert_traj[i]
        dist = np.linalg.norm(
            np.array(ctrl["spatial_mean"]) - np.array(pert["spatial_mean"])
        )
        print(
            f"{ctrl['t']:<8.2f} {ctrl['n_cells']:<20} {pert['n_cells']:<20} {dist:<15.4f}"
        )

    # Final analysis
    ctrl_final = ctrl_traj[-1]
    pert_final = pert_traj[-1]
    final_dist = np.linalg.norm(
        np.array(ctrl_final["spatial_mean"]) - np.array(pert_final["spatial_mean"])
    )

    print("\n" + "=" * 70)
    print("FINAL ANALYSIS")
    print("=" * 70)
    print(
        f"Control:   {ctrl_final['n_cells']} cells at ({ctrl_final['spatial_mean'][0]:.2f}, {ctrl_final['spatial_mean'][1]:.2f}, {ctrl_final['spatial_mean'][2]:.2f})"
    )
    print(
        f"Perturbed: {pert_final['n_cells']} cells at ({pert_final['spatial_mean'][0]:.2f}, {pert_final['spatial_mean'][1]:.2f}, {pert_final['spatial_mean'][2]:.2f})"
    )
    print(f"\nSpatial distance: {final_dist:.4f}")

    cell_diff = abs(ctrl_final["n_cells"] - pert_final["n_cells"])
    if cell_diff > 0 and final_dist < 0.5:
        print(
            f"\n✓ COMPENSATORY BEHAVIOR: Cell count changed by {cell_diff}, but spatial pattern conserved"
        )
    elif cell_diff > 0:
        print(
            f"\n~ Partial compensation: Cell count changed and spatial pattern shifted"
        )
    else:
        print(f"\n✗ No difference between control and perturbed")

    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "paired_comparison.json", "w") as f:
        json.dump(
            {
                "control": control_results["trajectory"],
                "perturbed": perturbed_results["trajectory"],
                "final_distance": float(final_dist),
                "cell_count_difference": int(cell_diff),
                "compensatory_behavior": bool(cell_diff > 0 and final_dist < 0.5),
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to {output_dir}/paired_comparison.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
