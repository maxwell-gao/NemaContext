#!/usr/bin/env python3
"""Visualize whole-embryo developmental trajectories.

Generates plots showing:
1. Cell count growth over time (all lineages combined)
2. Founder composition over time
3. Spatial distribution in 3D
4. Cross-lineage proximity analysis
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def export_for_blender(trajectory: list[dict], output_dir: str):
    """Export trajectory as CSV files for Blender visualization."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    founder_colors = {
        "P0": (1.0, 1.0, 1.0),  # White
        "AB": (1.0, 0.2, 0.2),  # Red
        "MS": (0.2, 1.0, 0.2),  # Green
        "E": (0.2, 0.2, 1.0),  # Blue
        "C": (1.0, 1.0, 0.2),  # Yellow
        "D": (1.0, 0.2, 1.0),  # Magenta
        "P4": (0.2, 1.0, 1.0),  # Cyan
    }

    for i, state in enumerate(trajectory):
        time = state["time"]
        positions = state["positions"]
        founders = state.get("founders", [])

        rows = []
        for pos, founder in zip(positions, founders):
            color = founder_colors.get(founder, (0.5, 0.5, 0.5))
            rows.append(
                f"{pos[0]},{pos[1]},{pos[2]},{color[0]},{color[1]},{color[2]},{founder}"
            )

        filename = output_path / f"frame_{i:04d}_t{int(time):04d}.csv"
        with open(filename, "w") as f:
            f.write("x,y,z,r,g,b,founder\n")
            f.write("\n".join(rows))

    print(f"Exported {len(trajectory)} frames to {output_dir}")


def analyze_lineage_interactions(trajectory: list[dict]):
    """Analyze spatial proximity between different lineages."""
    print("\n" + "=" * 70)
    print("CROSS-LINEAGE SPATIAL ANALYSIS")
    print("=" * 70)

    founders = ["AB", "MS", "E", "C", "D", "P4"]

    # Sample a few time points for analysis
    for state in trajectory[5::10]:  # Start after some divisions
        if state["n_cells"] < 10:
            continue

        time = state["time"]
        positions = np.array(state["positions"])
        cell_founders = state.get("founders", [])

        print(f"\nt = {time:.0f} min ({state['n_cells']} cells):")

        # Compute pairwise distances
        for i, f1 in enumerate(founders):
            for f2 in founders[i + 1 :]:
                # Get positions for each founder
                mask1 = [cf == f1 for cf in cell_founders]
                mask2 = [cf == f2 for cf in cell_founders]

                if not any(mask1) or not any(mask2):
                    continue

                pos1 = positions[mask1]
                pos2 = positions[mask2]

                # Compute min distance between any pair
                distances = np.sqrt(
                    ((pos1[:, None, :] - pos2[None, :, :]) ** 2).sum(axis=-1)
                )
                min_dist = distances.min()

                status = "✓ touching" if min_dist < 0.1 else "  separated"
                print(f"  {f1}-{f2}: min_dist={min_dist:.3f} {status}")


def generate_statistics(trajectory: list[dict]) -> dict:
    """Generate comprehensive statistics about the trajectory."""
    stats = {
        "total_time_points": len(trajectory),
        "time_range": (trajectory[0]["time"], trajectory[-1]["time"]),
        "cell_count_range": (trajectory[0]["n_cells"], trajectory[-1]["n_cells"]),
        "founders_present": set(),
        "per_timepoint": [],
    }

    for state in trajectory:
        founders = set(state.get("founders", []))
        stats["founders_present"].update(founders)

        positions = np.array(state["positions"])
        spatial_extent = (
            positions.max(axis=0) - positions.min(axis=0)
            if len(positions) > 0
            else [0, 0, 0]
        )

        stats["per_timepoint"].append(
            {
                "time": state["time"],
                "n_cells": state["n_cells"],
                "n_founders": len(founders),
                "spatial_extent_ap": float(spatial_extent[0]),
                "spatial_extent_dv": float(spatial_extent[1]),
                "spatial_extent_lr": float(spatial_extent[2]),
            }
        )

    stats["founders_present"] = sorted(stats["founders_present"])
    return stats


def main():
    parser = argparse.ArgumentParser(description="Visualize embryo trajectories")
    parser.add_argument(
        "--trajectory_file",
        type=str,
        default="dataset/processed/embryo_trajectory.json",
    )
    parser.add_argument("--export_blender", type=str, default=None)
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--stats_output", type=str, default=None)
    args = parser.parse_args()

    # Load trajectory
    with open(args.trajectory_file) as f:
        trajectory = json.load(f)

    print(f"Loaded trajectory: {len(trajectory)} time points")

    # Generate statistics
    stats = generate_statistics(trajectory)

    print("\n" + "=" * 70)
    print("TRAJECTORY STATISTICS")
    print("=" * 70)
    print(
        f"Time range: {stats['time_range'][0]:.0f} - {stats['time_range'][1]:.0f} min"
    )
    print(
        f"Cell count: {stats['cell_count_range'][0]} -> {stats['cell_count_range'][1]}"
    )
    print(f"Founders present: {', '.join(stats['founders_present'])}")

    # Show growth curve
    print("\nGrowth curve:")
    print("-" * 40)
    for tp in stats["per_timepoint"][::5]:
        bar_len = int(np.log10(tp["n_cells"] + 1) * 8)
        bar = "█" * bar_len
        print(
            f"  t={tp['time']:5.0f}min: {tp['n_cells']:4d} cells {bar} "
            f"({tp['n_founders']} lineages)"
        )

    # Cross-lineage analysis
    if args.analyze:
        analyze_lineage_interactions(trajectory)

    # Export for Blender
    if args.export_blender:
        export_for_blender(trajectory, args.export_blender)

    # Save statistics
    if args.stats_output:
        with open(args.stats_output, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nStatistics saved to: {args.stats_output}")

    print("=" * 70)


if __name__ == "__main__":
    main()
