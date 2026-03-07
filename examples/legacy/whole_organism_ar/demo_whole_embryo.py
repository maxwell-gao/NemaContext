#!/usr/bin/env python3
"""Demonstration of whole-embryo context architecture.

This script demonstrates the key biological principle:
All cells from all lineages coexist in shared embryonic coordinates,
enabling cross-lineage attention and influence.
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

from src.branching_flows.autoregressive_model import AutoregressiveNemaModel  # noqa: E402
from examples.whole_organism_ar.train_autoregressive_full import EmbryoTrajectoryDataset  # noqa: E402


def analyze_trajectory_structure(trajectory_file: str):
    """Analyze and display the whole-embryo trajectory structure."""
    print("=" * 70)
    print("WHOLE-EMBRYO TRAJECTORY ANALYSIS")
    print("=" * 70)

    with open(trajectory_file) as f:
        trajectory = json.load(f)

    print(f"\nTotal time points: {len(trajectory)}")

    # Analyze founder composition over time
    print("\nFounder composition over time:")
    print("-" * 50)

    for state in trajectory[::5]:  # Sample every 5 time points
        time = state["time"]
        n_cells = state["n_cells"]
        founders = state.get("founders", [])

        # Count cells per founder
        founder_counts = {}
        for f in founders:
            founder_counts[f] = founder_counts.get(f, 0) + 1

        founder_str = ", ".join(f"{f}:{n}" for f, n in sorted(founder_counts.items()))
        print(f"  t={time:5.0f}min: {n_cells:4d} cells [{founder_str}]")

    # Verify cross-lineage coexistence
    print("\nCross-lineage coexistence verification:")
    print("-" * 50)

    multi_founder_points = 0
    for state in trajectory:
        founders = set(state.get("founders", []))
        if len(founders) > 1:
            multi_founder_points += 1

    print(
        f"  Time points with multiple founders: {multi_founder_points}/{len(trajectory)}"
    )

    if multi_founder_points > 0:
        print("  ✓ Cross-lineage interaction is possible")
    else:
        print("  ✗ No cross-lineage interaction possible")

    # Spatial extent analysis
    print("\nSpatial extent over time:")
    print("-" * 50)

    for state in trajectory[::10]:
        positions = np.array(state["positions"])
        if len(positions) > 0:
            extent = positions.max(axis=0) - positions.min(axis=0)
            print(
                f"  t={state['time']:5.0f}min: "
                f"AP={extent[0]:.3f}, DV={extent[1]:.3f}, LR={extent[2]:.3f}"
            )

    print("=" * 70)


def demonstrate_cross_lineage_attention(
    model: AutoregressiveNemaModel,
    dataset: EmbryoTrajectoryDataset,
    device: str,
):
    """Demonstrate that attention spans across lineages."""
    print("\n" + "=" * 70)
    print("CROSS-LINEAGE ATTENTION DEMONSTRATION")
    print("=" * 70)

    # Find a state with multiple founders
    multi_founder_state = None
    for idx in range(len(dataset)):
        sample = dataset[idx]
        current = sample["current"]
        founders = current.states[1][0][current.padmask[0]]
        if len(torch.unique(founders)) > 1:
            multi_founder_state = current
            break

    if multi_founder_state is None:
        print("No multi-founder states found!")
        return

    founders = multi_founder_state.states[1][0][multi_founder_state.padmask[0]]
    founder_map_inv = {0: "P0", 1: "AB", 2: "MS", 3: "E", 4: "C", 5: "D", 6: "P4"}

    print(f"\nSelected state with {founders.shape[0]} cells:")
    for fid in torch.unique(founders):
        count = (founders == fid).sum().item()
        name = founder_map_inv.get(fid.item(), "UNKNOWN")
        print(f"  {name}: {count} cells")

    # Compute attention pattern
    model.eval()
    with torch.no_grad():
        h = model.encode_state(multi_founder_state.to(device))

        # Get attention scores from first layer
        B, L, D = h.shape
        qkv = model.blocks[0].qkv(h).reshape(B, L, 3, model.blocks[0].n_heads, -1)
        q, k, _ = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
        attn = torch.softmax(scores, dim=-1).mean(dim=1)  # Average over heads

    # Analyze cross-lineage attention
    print("\nAttention statistics:")
    print("-" * 50)

    valid_founders = founders

    cross_lineage_attn = []
    same_lineage_attn = []

    for i in range(valid_founders.shape[0]):
        my_founder = valid_founders[i]
        for j in range(valid_founders.shape[0]):
            if i != j:
                if valid_founders[j] == my_founder:
                    same_lineage_attn.append(attn[0, i, j].item())
                else:
                    cross_lineage_attn.append(attn[0, i, j].item())

    if cross_lineage_attn:
        print(f"  Cross-lineage attention:  {np.mean(cross_lineage_attn):.4f} (mean)")
    if same_lineage_attn:
        print(f"  Same-lineage attention:   {np.mean(same_lineage_attn):.4f} (mean)")

    if cross_lineage_attn and same_lineage_attn:
        ratio = np.mean(cross_lineage_attn) / np.mean(same_lineage_attn)
        print(f"  Cross/same ratio:         {ratio:.2f}")

    print("=" * 70)


def visualize_embryo_state(state_dict: dict, output_file: str | None = None):
    """Create a simple text visualization of embryo state."""
    print("\n" + "=" * 70)
    print(f"EMBRYO STATE AT t={state_dict['time']:.0f}min")
    print("=" * 70)

    positions = np.array(state_dict["positions"])
    founders = state_dict.get("founders", [])
    names = state_dict.get("cell_names", [])

    print(f"\nTotal cells: {len(names)}")

    # Group by founder and show spatial extent
    founder_groups = {}
    for i, (name, founder, pos) in enumerate(zip(names, founders, positions)):
        if founder not in founder_groups:
            founder_groups[founder] = []
        founder_groups[founder].append((name, pos))

    print("\nLineage positions (anterior-posterior axis):")
    print("-" * 50)

    for founder in sorted(founder_groups.keys()):
        cells = founder_groups[founder]
        x_positions = [pos[0] for _, pos in cells]  # AP axis
        mean_x = np.mean(x_positions)
        std_x = np.std(x_positions)

        # Create simple bar
        bar_len = 30
        mean_pos = int(mean_x * bar_len)
        left = int((mean_x - std_x) * bar_len)
        right = int((mean_x + std_x) * bar_len)

        bar = [" "] * bar_len
        for i in range(max(0, left), min(bar_len, right + 1)):
            bar[i] = "-"
        if 0 <= mean_pos < bar_len:
            bar[mean_pos] = "*"

        print(f"  {founder:3s}: [{''.join(bar)}] μ={mean_x:.2f}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Demonstrate whole-embryo architecture"
    )
    parser.add_argument(
        "--trajectory_file",
        type=str,
        default="dataset/processed/embryo_trajectory.json",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # Analyze trajectory structure
    analyze_trajectory_structure(args.trajectory_file)

    # Load dataset for further analysis
    print("\nLoading dataset...")
    dataset = EmbryoTrajectoryDataset(args.trajectory_file)

    # Visualize a few states
    for idx in [0, len(dataset) // 2, len(dataset) - 1]:
        if idx < len(dataset.trajectory):
            visualize_embryo_state(dataset.trajectory[idx])

    # If checkpoint provided, demonstrate attention
    if args.checkpoint:
        print("\nLoading model for attention analysis...")
        model = AutoregressiveNemaModel(
            gene_dim=2000,
            spatial_dim=3,
            discrete_K=7,
            d_model=256,
            n_layers=6,
            n_heads=8,
            cross_modal_every=2,
            max_seq_len=128,
            dt=0.1,
        ).to(args.device)

        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(checkpoint["model_state_dict"])

        demonstrate_cross_lineage_attention(model, dataset, args.device)
    else:
        print("\n(No checkpoint provided - skipping attention demonstration)")
        print("Run training first or provide --checkpoint")


if __name__ == "__main__":
    main()
