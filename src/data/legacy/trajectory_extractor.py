"""Legacy synthetic and per-founder trajectory extraction utilities.

These paths are archived for reference only. Active whole-organism training
should use `src.data.trajectory_extractor`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.trajectory_extractor import WholeEmbryoTrajectoryExtractor, save_trajectory


class LegacyTrajectoryExtractor(WholeEmbryoTrajectoryExtractor):
    """Archived extractor for synthetic and per-founder trajectories."""

    def extract_synthetic_embryo_trajectory(
        self,
        max_time: float = 400.0,
        time_resolution: float = 10.0,
    ) -> list[dict]:
        trajectory = []

        for t in self._time_grid(max_time, time_resolution):
            cells_at_t = self.get_all_cells_at_time(t)
            if not cells_at_t:
                continue

            positions = self._synthesize_global_positions(cells_at_t, t)
            founders = [self.get_founder(c) for c in cells_at_t]
            founder_ids = [self.founder_map.get(f, 0) for f in founders]

            trajectory.append(
                {
                    "time": float(t),
                    "source": "synthetic",
                    "n_cells": len(cells_at_t),
                    "cell_names": cells_at_t,
                    "founders": founders,
                    "founder_ids": founder_ids,
                    "positions": positions.tolist(),
                    "genes": self._synthesize_genes(cells_at_t).tolist(),
                    "divisions": self._detect_divisions(cells_at_t, t),
                }
            )

        return trajectory

    def extract_founder_trajectory(
        self,
        founder: str = "AB",
        max_depth: int = 8,
        time_resolution: int = 10,
    ) -> list[dict]:
        trajectory = []
        max_time = 20.0 + max_depth * 12.0

        for t in self._time_grid(max_time, time_resolution):
            cells_at_t = []
            for depth in range(max_depth + 1):
                depth_start = 20.0 + depth * 12.0
                depth_end = depth_start + 12.0
                if depth_start <= t < depth_end:
                    for name in self.tree.keys():
                        if name.startswith(founder) and len(name) == len(founder) + depth:
                            cells_at_t.append(name)

            if not cells_at_t:
                continue

            cells_at_t = sorted(cells_at_t)
            state = {
                "time": float(t),
                "n_cells": len(cells_at_t),
                "cell_names": cells_at_t,
                "founders": [founder] * len(cells_at_t),
                "founder_ids": [self.founder_map.get(founder, 0)] * len(cells_at_t),
                "division_depth": int((t - 20.0) / 12.0) if t >= 20 else 0,
                "positions": self._synthesize_global_positions(cells_at_t, t).tolist(),
                "genes": self._synthesize_genes(cells_at_t).tolist(),
            }

            divisions = []
            for idx, cell in enumerate(cells_at_t):
                cell_depth = len(cell) - len(founder)
                cell_start = 20.0 + cell_depth * 12.0 if cell_depth > 0 else 0
                if abs(t - cell_start) < 5.0:
                    divisions.append(idx)
            state["divisions"] = divisions
            trajectory.append(state)

        return trajectory

    def extract_all_founder_trajectories(
        self,
        founders: list[str] | None = None,
        **kwargs,
    ) -> dict[str, list]:
        if founders is None:
            founders = ["AB", "MS", "E", "C", "D"]

        trajectories = {}
        for founder in founders:
            print(f"Extracting {founder} lineage...")
            traj = self.extract_founder_trajectory(founder, **kwargs)
            if traj:
                trajectories[founder] = traj
        return trajectories

    @staticmethod
    def save_trajectories(trajectories: dict[str, list], output_file: str | Path):
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(__import__("json").dumps(trajectories, indent=2))
        print(f"\nTrajectories saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lineage_file",
        type=str,
        default="dataset/raw/wormbase/lineage_tree.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset/processed/embryo_trajectory_legacy.json",
    )
    parser.add_argument("--max_time", type=float, default=400.0)
    parser.add_argument("--time_resolution", type=float, default=10.0)
    parser.add_argument(
        "--mode",
        choices=["synthetic_whole_embryo", "per_founder"],
        default="synthetic_whole_embryo",
    )
    args = parser.parse_args()

    extractor = LegacyTrajectoryExtractor(args.lineage_file)
    if args.mode == "per_founder":
        trajectories = extractor.extract_all_founder_trajectories(
            max_depth=8,
            time_resolution=int(args.time_resolution),
        )
        extractor.save_trajectories(trajectories, args.output)
        return

    trajectory = extractor.extract_synthetic_embryo_trajectory(
        max_time=args.max_time,
        time_resolution=args.time_resolution,
    )
    save_trajectory(trajectory, args.output)


if __name__ == "__main__":
    main()
