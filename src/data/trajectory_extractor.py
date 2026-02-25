"""Extract developmental trajectories from Sulston lineage tree.

Converts the static lineage tree into time-series trajectories
for autoregressive model training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


class SulstonTrajectoryExtractor:
    """Extract trajectories from C. elegans Sulston lineage tree.

    The Sulston tree records:
    - Cell names (lineage nomenclature: AB, ABa, ABal, etc.)
    - Division times (in minutes post-fertilization)
    - Parent-child relationships

    This extractor converts the tree into sequences of cell states
    suitable for autoregressive training.
    """

    def __init__(self, lineage_file: str | Path):
        """Initialize with lineage data.

        Args:
            lineage_file: Path to WormBase lineage JSON
        """
        self.lineage_file = Path(lineage_file)
        with open(self.lineage_file) as f:
            self.data = json.load(f)

        # Build lookup tables
        self._build_indices()

    def _build_indices(self):
        """Build efficient lookup structures."""
        self.cell_by_name = {}
        self.cell_by_time = {}

        for cell in self.data.get("cells", []):
            name = cell["name"]
            self.cell_by_name[name] = cell

            # Index by time ranges
            start = cell.get("start_time", 0)
            end = cell.get("end_time", start)

            for t in range(int(start), int(end) + 1, 10):  # 10-min bins
                if t not in self.cell_by_time:
                    self.cell_by_time[t] = []
                self.cell_by_time[t].append(cell)

    def extract_trajectory(
        self,
        founder: str = "AB",
        time_resolution: int = 10,  # minutes
        add_spatial: bool = True,
    ) -> list[dict[str, Any]]:
        """Extract trajectory for a founder lineage.

        Args:
            founder: Founder cell (AB, MS, E, C, D, P4)
            time_resolution: Time step in minutes
            add_spatial: Whether to add synthetic spatial positions

        Returns:
            List of time points, each with cell states
        """
        trajectory = []

        # Get time range for this founder
        if founder not in self.cell_by_name:
            return trajectory

        founder_cell = self.cell_by_name[founder]
        start_time = founder_cell.get("start_time", 0)
        end_time = max(
            c.get("end_time", 0)
            for c in self.cell_by_name.values()
            if c["name"].startswith(founder)
        )

        # Generate trajectory
        for t in range(int(start_time), int(end_time) + 1, time_resolution):
            cells_at_t = self._get_cells_at_time(founder, t)

            if not cells_at_t:
                continue

            # Create state representation
            state = {
                "time": t,
                "n_cells": len(cells_at_t),
                "cell_names": [c["name"] for c in cells_at_t],
                "cell_ids": list(range(1, len(cells_at_t) + 1)),
            }

            # Add synthetic gene expression based on lineage
            if add_spatial:
                state["positions"] = self._synthesize_positions(cells_at_t, t)
                state["genes"] = self._synthesize_genes(cells_at_t)

            # Record division events
            state["divisions"] = self._detect_divisions(cells_at_t, t, time_resolution)
            state["deaths"] = self._detect_deaths(cells_at_t, t, time_resolution)

            trajectory.append(state)

        return trajectory

    def _get_cells_at_time(self, founder: str, time: int) -> list[dict]:
        """Get all cells of a lineage alive at given time."""
        cells = []

        for name, cell in self.cell_by_name.items():
            if not name.startswith(founder):
                continue

            start = cell.get("start_time", 0)
            end = cell.get("end_time", start)

            if start <= time <= end:
                cells.append(cell)

        return cells

    def _synthesize_positions(
        self, cells: list[dict], time: int
    ) -> np.ndarray:
        """Synthesize spatial positions based on lineage.

        In real embryo:
        - AB lineage: anterior (neurons, epidermis)
        - MS lineage: ventral (pharynx, muscle)
        - E lineage: posterior (intestine)
        - C/D lineages: posterior (muscle, germline)
        """
        positions = []

        for cell in cells:
            name = cell["name"]

            # Founder-specific base positions
            if name.startswith("AB"):
                base = np.array([0.5, 0.5, 0.3])  # Anterior
            elif name.startswith("MS"):
                base = np.array([0.5, 0.3, 0.5])  # Ventral
            elif name.startswith("E"):
                base = np.array([0.5, 0.5, 0.8])  # Posterior
            else:
                base = np.array([0.5, 0.7, 0.7])  # Posterior/dorsal

            # Add lineage-specific offset
            depth = len(name) - 1  # Division depth
            noise = np.random.randn(3) * 0.05 * depth

            # Time-based migration
            migration = np.array([0, 0, time * 0.001])

            positions.append(base + noise + migration)

        return np.array(positions)

    def _synthesize_genes(self, cells: list[dict]) -> np.ndarray:
        """Synthesize gene expression based on cell type."""
        n_genes = 2000
        expressions = []

        for cell in cells:
            name = cell["name"]

            # Base expression
            expr = np.random.randn(n_genes) * 0.1

            # Founder-specific markers
            if name.startswith("AB"):
                # Neuronal markers
                expr[0:50] += 2.0
            elif name.startswith("MS"):
                # Muscle markers
                expr[50:100] += 2.0
            elif name.startswith("E"):
                # Gut markers
                expr[100:150] += 2.0

            # Lineage depth effect
            depth = len(name) - 1
            expr += np.random.randn(n_genes) * 0.01 * depth

            expressions.append(expr)

        return np.array(expressions)

    def _detect_divisions(
        self, cells: list[dict], time: int, dt: int
    ) -> list[int]:
        """Detect which cells divide in this time window."""
        divisions = []

        for i, cell in enumerate(cells):
            end_time = cell.get("end_time", None)
            if end_time and time <= end_time < time + dt:
                # This cell ends in this window = division
                divisions.append(i)

        return divisions

    def _detect_deaths(
        self, cells: list[dict], time: int, dt: int
    ) -> list[int]:
        """Detect programmed cell deaths in this window."""
        deaths = []

        death_set = {"C1", "C2", "C3"}  # Known deaths in C. elegans

        for i, cell in enumerate(cells):
            if cell["name"] in death_set:
                end_time = cell.get("end_time", None)
                if end_time and time <= end_time < time + dt:
                    deaths.append(i)

        return deaths

    def extract_all_trajectories(
        self,
        founders: list[str] | None = None,
        **kwargs,
    ) -> dict[str, list]:
        """Extract trajectories for all founder lineages."""
        if founders is None:
            founders = ["AB", "MS", "E", "C", "D", "P4"]

        trajectories = {}
        for founder in founders:
            print(f"Extracting {founder} lineage...")
            traj = self.extract_trajectory(founder, **kwargs)
            if traj:
                trajectories[founder] = traj
                print(f"  {len(traj)} time points, up to {max(s['n_cells'] for s in traj)} cells")

        return trajectories


def save_trajectories(
    trajectories: dict[str, list],
    output_file: str | Path,
):
    """Save trajectories to disk."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    serializable = {}
    for founder, traj in trajectories.items():
        serializable[founder] = []
        for state in traj:
            state_copy = state.copy()
            if "positions" in state_copy:
                state_copy["positions"] = state_copy["positions"].tolist()
            if "genes" in state_copy:
                state_copy["genes"] = state_copy["genes"].tolist()
            serializable[founder].append(state_copy)

    with open(output_file, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"\nTrajectories saved to: {output_file}")


def main():
    """Extract trajectories from Sulston data."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lineage_file",
        type=str,
        default="dataset/raw/wormbase/lineage.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset/processed/sulston_trajectories.json",
    )
    parser.add_argument("--time_resolution", type=int, default=10)
    args = parser.parse_args()

    extractor = SulstonTrajectoryExtractor(args.lineage_file)
    trajectories = extractor.extract_all_trajectories(
        time_resolution=args.time_resolution,
    )

    save_trajectories(trajectories, args.output)


if __name__ == "__main__":
    main()
