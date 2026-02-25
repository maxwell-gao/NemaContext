#!/usr/bin/env python3
"""
Compare control vs perturbed developmental trajectories.

This demonstrates whether the model shows compensatory behavior
in autoregressive simulation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.branching_flows.crossmodal_model import CrossModalNemaModel
from src.branching_flows.trimodal_dataset import TrimodalDataset
from autoregressive_simulation import (
    AutoregressiveSimulator,
    create_initial_state,
)


def compare_trajectories(control_file: str, perturbed_file: str):
    """Compare two trajectory files and analyze differences."""

    with open(control_file) as f:
        control = json.load(f)
    with open(perturbed_file) as f:
        perturbed = json.load(f)

    print("=" * 70)
    print("TRAJECTORY COMPARISON")
    print("=" * 70)
    print(f"Control: {control['mode']} (initial: {control['initial_cells']} cells)")
    print(f"Perturbed: {perturbed['mode']} (initial: {perturbed['initial_cells']} cells)")
    print()

    # Extract trajectories
    ctrl_traj = control["trajectory"]
    pert_traj = perturbed["trajectory"]

    print("Development comparison:")
    print(f"{'Time':<8} {'Control':<20} {'Perturbed':<20} {'Difference':<20}")
    print("-" * 70)

    max_len = max(len(ctrl_traj), len(pert_traj))

    for i in range(max_len):
        ctrl = ctrl_traj[i] if i < len(ctrl_traj) else None
        pert = pert_traj[i] if i < len(pert_traj) else None

        if ctrl is None or pert is None:
            break

        t = ctrl["t"]
        ctrl_pos = np.array(ctrl["spatial_mean"])
        pert_pos = np.array(pert["spatial_mean"])

        dist = np.linalg.norm(ctrl_pos - pert_pos)

        print(f"{t:<8.2f} "
              f"({ctrl['n_cells']:>3} cells) {ctrl_pos[2]:>6.2f}  "
              f"({pert['n_cells']:>3} cells) {pert_pos[2]:>6.2f}  "
              f"dist={dist:.3f}")

    # Final comparison
    print()
    print("=" * 70)
    print("FINAL STATE ANALYSIS")
    print("=" * 70)

    ctrl_final = ctrl_traj[-1]
    pert_final = pert_traj[-1]

    print(f"Control final:   {ctrl_final['n_cells']} cells")
    print(f"Perturbed final: {pert_final['n_cells']} cells")

    # Check if perturbation had lasting effect
    ctrl_pos = np.array(ctrl_final["spatial_mean"])
    pert_pos = np.array(pert_final["spatial_mean"])
    final_dist = np.linalg.norm(ctrl_pos - pert_pos)

    print()
    print(f"Final spatial center distance: {final_dist:.4f}")

    if final_dist > 0.5:
        print("✓ Perturbation had SIGNIFICANT lasting effect on development")
    elif final_dist > 0.1:
        print("~ Perturbation had MODERATE effect")
    else:
        print("✗ Perturbation had MINIMAL effect (trajectories converged)")

    # Check compensatory behavior
    # If cell count changed but final position is similar -> compensation
    cell_count_diff = abs(ctrl_final['n_cells'] - pert_final['n_cells'])

    print()
    if cell_count_diff > 0 and final_dist < 1.0:
        print("✓ Possible compensatory behavior detected:")
        print(f"  Cell count changed by {cell_count_diff}, but spatial pattern adapted")
    elif cell_count_diff > 0:
        print("  Cell count changed but spatial pattern also shifted")
    else:
        print("  No cell count difference")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Compare control vs perturbed trajectories"
    )
    parser.add_argument(
        "--control",
        type=str,
        default="autoregressive_results/control_t0.5_dt0.02.json",
        help="Control trajectory file",
    )
    parser.add_argument(
        "--perturbed",
        type=str,
        default="autoregressive_results/lineage_deletion_t0.5_dt0.02.json",
        help="Perturbed trajectory file",
    )
    args = parser.parse_args()

    compare_trajectories(args.control, args.perturbed)


if __name__ == "__main__":
    main()
