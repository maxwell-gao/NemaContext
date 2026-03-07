"""Extract whole-embryo developmental trajectories from Sulston lineage tree.

Converts the static lineage tree into unified time-series trajectories
where all cells coexist in global embryonic coordinates.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np  # noqa: E402

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.branching_flows.wormguides_parser import parse_wormguides  # noqa: E402


class WholeEmbryoTrajectoryExtractor:
    """Extract unified whole-embryo trajectories from C. elegans Sulston lineage.

    Key principle: Cells from all lineages (AB, MS, E, C, D, etc.) coexist
    in a shared global coordinate system at each time point. Founder identity
    is preserved as a discrete feature, not a trajectory separator.
    """

    def __init__(self, lineage_file: str | Path):
        """Initialize with lineage data.

        Args:
            lineage_file: Path to WormBase lineage JSON (parent-children format)
        """
        self.lineage_file = Path(lineage_file)
        with open(self.lineage_file) as f:
            self.tree = json.load(f)

        # Load timing data if available
        timing_file = self.lineage_file.parent / "cell_timing.json"
        if timing_file.exists():
            with open(timing_file) as f:
                self.timing = json.load(f)
        else:
            self.timing = {}

        # Founder mapping for discrete features
        self.founder_map = {
            "P0": 0,
            "AB": 1,
            "MS": 2,
            "E": 3,
            "C": 4,
            "D": 5,
            "P4": 6,
        }
        self.founders = list(self.founder_map.keys())

    def get_founder(self, cell_name: str) -> str:
        """Get founder lineage for a cell."""
        for founder in self.founders:
            if cell_name.startswith(founder):
                return founder
        return "P0" if cell_name == "P0" else "UNKNOWN"

    def get_cell_birth_time(self, cell_name: str) -> float:
        """Get birth time for a cell from timing data or estimate from depth."""
        if cell_name in self.timing:
            return self.timing[cell_name].get("birth_time_min", 0.0)

        # Estimate from division depth
        if cell_name == "P0":
            return 0.0

        depth = len(cell_name) - len(self.get_founder(cell_name))
        base_time = 20.0  # First division
        time_per_division = 12.0
        return base_time + depth * time_per_division

    def get_cell_division_time(self, cell_name: str) -> float | None:
        """Get division time for a cell, or None if terminal."""
        if cell_name in self.timing:
            return self.timing[cell_name].get("division_time_min")

        # Estimate: alive for ~12 minutes per division level
        return self.get_cell_birth_time(cell_name) + 12.0

    def is_cell_alive_at_time(self, cell_name: str, time: float) -> bool:
        """Check if a cell exists at given time."""
        birth = self.get_cell_birth_time(cell_name)
        division = self.get_cell_division_time(cell_name)

        if time < birth:
            return False
        # Cell is alive from birth (inclusive) to division (exclusive)
        if division is not None and time >= division:
            return False
        return True

    def get_all_cells_at_time(self, time: float) -> list[str]:
        """Get all cells alive at given time across ALL lineages."""
        cells = []
        for cell_name in self.tree.keys():
            # Only include cells that have timing data (embryonic cells)
            if cell_name not in self.timing:
                continue
            if self.is_cell_alive_at_time(cell_name, time):
                cells.append(cell_name)
        return sorted(cells)

    def _synthesize_global_positions(self, cells: list[str], time: float) -> np.ndarray:
        """Synthesize spatial positions in GLOBAL embryo coordinates.

        Coordinate system:
        - X: Anterior-Posterior axis, 0=anterior, 1=posterior
        - Y: Dorsal-Ventral axis, 0=dorsal, 1=ventral
        - Z: Left-Right axis, 0=left, 1=right
        """
        positions = []

        for cell in cells:
            founder = self.get_founder(cell)
            depth = len(cell) - len(founder) if founder != "P0" else 0

            # Founder-specific base positions in GLOBAL coordinates
            # These positions reflect the actual anatomical locations in embryo
            if founder == "P0":
                base = np.array([0.5, 0.5, 0.5])  # Center at start
            elif founder == "AB":
                # AB lineage: anterior, spreads dorsal-ventral with division
                base_x = 0.15 + depth * 0.02  # Anterior region
                base_y = 0.5 + (hash(cell) % 100 / 100 - 0.5) * 0.3 * depth * 0.1
                base_z = 0.5 + (hash(cell[::-1]) % 100 / 100 - 0.5) * 0.2
                base = np.array([base_x, base_y, base_z])
            elif founder == "MS":
                # MS lineage: ventral-anterior to ventral-posterior
                base_x = 0.35 + depth * 0.03
                base_y = 0.3  # Ventral
                base_z = 0.5
                base = np.array([base_x, base_y, base_z])
            elif founder == "E":
                # E lineage: posterior-ventral (gut)
                base_x = 0.7 + depth * 0.02
                base_y = 0.4
                base_z = 0.5
                base = np.array([base_x, base_y, base_z])
            elif founder == "C":
                # C lineage: posterior-dorsal (hypodermis)
                base_x = 0.75 + depth * 0.02
                base_y = 0.7  # Dorsal
                base_z = 0.5
                base = np.array([base_x, base_y, base_z])
            elif founder == "D":
                # D lineage: posterior (muscle)
                base_x = 0.8 + depth * 0.015
                base_y = 0.6
                base_z = 0.5
                base = np.array([base_x, base_y, base_z])
            elif founder == "P4":
                # P4: posterior pole (germline)
                base_x = 0.9
                base_y = 0.5
                base_z = 0.5
                base = np.array([base_x, base_y, base_z])
            else:
                base = np.array([0.5, 0.5, 0.5])

            # Add small deterministic noise based on cell name
            name_hash = sum(ord(c) for c in cell)
            np.random.seed(name_hash)
            noise = np.random.randn(3) * 0.02 * max(1, depth)
            np.random.seed()

            positions.append(base + noise)

        return np.array(positions)

    def _synthesize_genes(self, cells: list[str]) -> np.ndarray:
        """Synthesize gene expression based on cell type."""
        n_genes = 2000
        expressions = []

        for cell in cells:
            founder = self.get_founder(cell)
            name_hash = sum(ord(c) for c in cell)
            np.random.seed(name_hash)

            # Base expression
            expr = np.random.randn(n_genes) * 0.1

            # Founder-specific marker genes
            if founder == "AB":
                expr[0:50] += 1.5  # Neuronal markers
                expr[100:120] -= 0.5
            elif founder == "MS":
                expr[50:100] += 1.5  # Muscle markers
                expr[0:20] -= 0.5
            elif founder == "E":
                expr[100:150] += 1.5  # Gut markers
            elif founder in ["C", "D"]:
                expr[150:200] += 1.5  # Hypodermis/muscle markers
            elif founder == "P4":
                expr[200:220] += 1.5  # Germline markers

            # Depth effect (differentiation)
            depth = len(cell)
            expr += np.random.randn(n_genes) * 0.01 * depth

            expressions.append(expr)
            np.random.seed()

        return np.array(expressions)

    def _detect_divisions(self, cells: list[str], time: float) -> list[int]:
        """Detect which current cells are about to divide."""
        divisions = []
        for i, cell in enumerate(cells):
            division_time = self.get_cell_division_time(cell)
            if division_time is not None and abs(time - division_time) < 5.0:
                divisions.append(i)
        return divisions

    def _zero_genes(self, n_cells: int, n_genes: int = 2000) -> np.ndarray:
        """Represent missing gene modality explicitly with zeros."""
        return np.zeros((n_cells, n_genes), dtype=np.float32)

    @staticmethod
    def _time_grid(max_time: float, time_resolution: float) -> np.ndarray:
        return np.arange(0, max_time + time_resolution, time_resolution)

    def extract_wormguides_trajectory(
        self,
        nuclei_dir: str | Path,
        deaths_csv: str | Path | None = None,
        max_time: float = 400.0,
        time_resolution: float = 10.0,
    ) -> list[dict[str, Any]]:
        """Build whole-embryo trajectory from real WormGUIDES nuclei tracking.

        Uses observed cell names and 3D coordinates directly. Gene expression is
        left as an all-zero placeholder to mark the modality as unavailable
        rather than inventing synthetic transcriptomic structure.
        """
        wg = parse_wormguides(nuclei_dir, deaths_csv)
        trajectory = []

        all_coords = np.array(
            [
                [x, y, z]
                for traj in wg.cell_trajectories.values()
                for _, x, y, z in traj
            ],
            dtype=np.float32,
        )
        coord_min = all_coords.min(axis=0)
        coord_range = np.clip(
            all_coords.max(axis=0) - coord_min, a_min=1e-6, a_max=None
        )

        sampled_minutes = np.arange(0, max_time + time_resolution, time_resolution)
        sampled_tps = sorted(
            {
                max(
                    1,
                    int(round((minute - wg.START_TIME_MIN) / wg.TIME_RESOLUTION_MIN))
                    + 1,
                )
                for minute in sampled_minutes
                if minute >= wg.START_TIME_MIN
            }
        )

        division_map: dict[int, dict[str, tuple[str, str]]] = {}
        for event in wg.division_events:
            division_map.setdefault(event.t_division, {})[event.parent] = (
                event.child1,
                event.child2,
            )

        for tp in sampled_tps:
            if tp > wg.n_timepoints:
                break

            cells = sorted(wg.cells_at(tp), key=lambda cell: cell.name)
            if not cells:
                continue

            names = [cell.name for cell in cells]
            founders = [self.get_founder(name) for name in names]
            founder_ids = [self.founder_map.get(founder, 0) for founder in founders]
            positions = [
                (
                    (np.array([cell.x, cell.y, cell.z], dtype=np.float32) - coord_min)
                    / coord_range
                ).tolist()
                for cell in cells
            ]

            divisions = []
            current_divisions = division_map.get(tp, {})
            next_names = {cell.name for cell in wg.cells_at(tp + 1)}
            for idx, name in enumerate(names):
                children = current_divisions.get(name)
                if children is not None and all(
                    child in next_names for child in children
                ):
                    divisions.append(idx)

            deaths = []
            for idx, name in enumerate(names):
                if name in wg.death_set and all(
                    child.name != name for child in wg.cells_at(tp + 1)
                ):
                    deaths.append(idx)

            trajectory.append(
                {
                    "time": float(wg.timepoint_to_minutes(tp)),
                    "timepoint": int(tp),
                    "source": "wormguides",
                    "n_cells": len(names),
                    "cell_names": names,
                    "founders": founders,
                    "founder_ids": founder_ids,
                    "positions": positions,
                    "genes": self._zero_genes(len(names)).tolist(),
                    "divisions": divisions,
                    "deaths": deaths,
                }
            )

        return trajectory

    def extract_embryo_trajectory(
        self,
        max_time: float = 400.0,
        time_resolution: float = 10.0,
        source: str = "wormguides",
        nuclei_dir: str | Path = "dataset/raw/wormguides/nuclei_files",
        deaths_csv: str | Path | None = "dataset/raw/wormguides/CellDeaths.csv",
    ) -> list[dict[str, Any]]:
        """Extract complete whole-embryo trajectory.

        Returns a single global timeline where all cells from all lineages
        coexist at each time point in shared embryonic coordinates.

        Args:
            max_time: Maximum developmental time in minutes (~400 for hatching)
            time_resolution: Time step in minutes

        Returns:
            List of embryo states, one per time point
        """
        nuclei_path = Path(nuclei_dir)
        if source != "wormguides":
            raise ValueError(f"Unsupported source: {source}")
        if not nuclei_path.exists():
            raise FileNotFoundError(f"WormGUIDES nuclei directory not found: {nuclei_path}")

        return self.extract_wormguides_trajectory(
            nuclei_dir=nuclei_path,
            deaths_csv=deaths_csv,
            max_time=max_time,
            time_resolution=time_resolution,
        )


def save_trajectory(
    trajectory: list[dict],
    output_file: str | Path,
):
    """Save single whole-embryo trajectory to disk."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(trajectory, f, indent=2)

    print(f"\nTrajectory saved to: {output_file}")
    print(f"  {len(trajectory)} time points")
    if trajectory:
        n_cells = [s["n_cells"] for s in trajectory]
        print(f"  Cell count: {min(n_cells)} -> {max(n_cells)}")


def main():
    """Extract trajectories from Sulston data."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lineage_file",
        type=str,
        default="dataset/raw/wormbase/lineage_tree.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset/processed/embryo_trajectory.json",
    )
    parser.add_argument("--max_time", type=float, default=400.0)
    parser.add_argument("--time_resolution", type=float, default=10.0)
    parser.add_argument(
        "--source",
        type=str,
        default="wormguides",
        choices=["wormguides"],
        help="Trajectory source for the active whole-embryo path.",
    )
    parser.add_argument(
        "--nuclei_dir",
        type=str,
        default="dataset/raw/wormguides/nuclei_files",
    )
    parser.add_argument(
        "--deaths_csv",
        type=str,
        default="dataset/raw/wormguides/CellDeaths.csv",
    )
    args = parser.parse_args()

    extractor = WholeEmbryoTrajectoryExtractor(args.lineage_file)
    print("Extracting whole-embryo trajectory...")
    trajectory = extractor.extract_embryo_trajectory(
        max_time=args.max_time,
        time_resolution=args.time_resolution,
        source=args.source,
        nuclei_dir=args.nuclei_dir,
        deaths_csv=args.deaths_csv,
    )
    save_trajectory(trajectory, args.output)


if __name__ == "__main__":
    main()
