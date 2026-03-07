"""
Spatial data parser for WormGUIDES nuclei 4D positions.

This module provides utilities to parse and work with 4D cell position data
from WormGUIDES nuclei files, which contain cell positions at each timepoint
during C. elegans embryonic development.

The data can be used for:
- Spatial embedding of cells
- Multi-modal integration (transcriptome-spatial-lineage)
- Developmental trajectory analysis
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

# Column indices in nuclei files (0-based)
COL_ID = 0
COL_FLAG = 1
COL_X = 5
COL_Y = 6
COL_Z = 7
COL_DIAMETER = 8
COL_CELL_NAME = 9

# Time resolution: 1 timepoint = 1 minute (approximately)
TIME_RESOLUTION_MIN = 1.0
START_TIME_MIN = 20  # Development starts ~20 min post-fertilization


@dataclass
class CellPosition:
    """A single cell's position at a specific timepoint."""

    cell_id: int
    cell_name: str
    x: float
    y: float
    z: float
    diameter: float
    timepoint: int
    flag: int = 0

    @property
    def position(self) -> np.ndarray:
        """Return position as numpy array [x, y, z]."""
        return np.array([self.x, self.y, self.z])

    @property
    def time_minutes(self) -> float:
        """Return time in minutes since first cleavage."""
        return START_TIME_MIN + (self.timepoint - 1) * TIME_RESOLUTION_MIN

    def __repr__(self) -> str:
        return (
            f"CellPosition(name={self.cell_name!r}, "
            f"pos=({self.x:.1f}, {self.y:.1f}, {self.z:.1f}), "
            f"t={self.timepoint})"
        )


@dataclass
class Timepoint:
    """All cell positions at a single timepoint."""

    timepoint: int
    cells: List[CellPosition] = field(default_factory=list)

    @property
    def named_cells(self) -> List[CellPosition]:
        """Return only cells with names (not empty strings)."""
        return [c for c in self.cells if c.cell_name]

    @property
    def cell_names(self) -> List[str]:
        """Return list of cell names at this timepoint."""
        return [c.cell_name for c in self.named_cells]

    @property
    def position_matrix(self) -> np.ndarray:
        """Return positions as Nx3 matrix for named cells."""
        named = self.named_cells
        if not named:
            return np.zeros((0, 3))
        return np.array([[c.x, c.y, c.z] for c in named])

    def get_cell(self, name: str) -> Optional[CellPosition]:
        """Get cell by name."""
        for c in self.cells:
            if c.cell_name == name:
                return c
        return None

    @property
    def time_minutes(self) -> float:
        """Return time in minutes since first cleavage."""
        return START_TIME_MIN + (self.timepoint - 1) * TIME_RESOLUTION_MIN


class SpatialDataParser:
    """
    Parser for WormGUIDES nuclei 4D position data.

    Example usage:
        parser = SpatialDataParser("dataset/raw/wormguides/nuclei_files")

        # Get all cells at timepoint 100
        tp = parser.parse_timepoint(100)
        print(f"Cells at t=100: {len(tp.named_cells)}")

        # Get trajectory for a specific cell
        trajectory = parser.get_cell_trajectory("ABa")

        # Get all named cells across all timepoints
        for name, positions in parser.iter_cell_trajectories():
            print(f"{name}: {len(positions)} timepoints")
    """

    def __init__(self, nuclei_dir: str):
        """
        Initialize the parser.

        Args:
            nuclei_dir: Path to the directory containing nuclei files
                        (e.g., "dataset/raw/wormguides/nuclei_files")
        """
        self.nuclei_dir = Path(nuclei_dir)
        if not self.nuclei_dir.exists():
            raise FileNotFoundError(f"Nuclei directory not found: {nuclei_dir}")

        self._cache: Dict[int, Timepoint] = {}

    def get_available_timepoints(self) -> List[int]:
        """Return list of available timepoint numbers."""
        timepoints = []
        for f in self.nuclei_dir.iterdir():
            if f.name.startswith("t") and f.name.endswith("-nuclei"):
                try:
                    tp = int(f.name[1:4])
                    timepoints.append(tp)
                except ValueError:
                    continue
        return sorted(timepoints)

    def _get_nuclei_path(self, timepoint: int) -> Path:
        """Get path to nuclei file for a timepoint."""
        return self.nuclei_dir / f"t{timepoint:03d}-nuclei"

    def parse_timepoint(self, timepoint: int, use_cache: bool = True) -> Timepoint:
        """
        Parse nuclei file for a specific timepoint.

        Args:
            timepoint: Timepoint number (1-360)
            use_cache: Whether to use cached results

        Returns:
            Timepoint object containing all cell positions
        """
        if use_cache and timepoint in self._cache:
            return self._cache[timepoint]

        path = self._get_nuclei_path(timepoint)
        if not path.exists():
            raise FileNotFoundError(f"Nuclei file not found: {path}")

        cells = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 10:
                    continue

                try:
                    cell = CellPosition(
                        cell_id=int(parts[COL_ID]),
                        flag=int(parts[COL_FLAG]),
                        x=float(parts[COL_X]),
                        y=float(parts[COL_Y]),
                        z=float(parts[COL_Z]),
                        diameter=float(parts[COL_DIAMETER]),
                        cell_name=parts[COL_CELL_NAME],
                        timepoint=timepoint,
                    )
                    cells.append(cell)
                except (ValueError, IndexError):
                    continue

        tp = Timepoint(timepoint=timepoint, cells=cells)

        if use_cache:
            self._cache[timepoint] = tp

        return tp

    def parse_timepoints(
        self,
        start: int = 1,
        end: int = 360,
        step: int = 1,
    ) -> Dict[int, Timepoint]:
        """
        Parse multiple timepoints.

        Args:
            start: Starting timepoint (inclusive)
            end: Ending timepoint (inclusive)
            step: Step between timepoints

        Returns:
            Dictionary mapping timepoint -> Timepoint
        """
        available = set(self.get_available_timepoints())
        result = {}

        for tp in range(start, end + 1, step):
            if tp in available:
                result[tp] = self.parse_timepoint(tp)

        return result

    def get_cell_trajectory(
        self,
        cell_name: str,
        timepoints: Optional[List[int]] = None,
    ) -> List[CellPosition]:
        """
        Get the trajectory (positions over time) for a specific cell.

        Args:
            cell_name: Name of the cell (e.g., "ABa", "ABpla")
            timepoints: Optional list of timepoints to check.
                       If None, checks all available timepoints.

        Returns:
            List of CellPosition objects sorted by timepoint
        """
        if timepoints is None:
            timepoints = self.get_available_timepoints()

        trajectory = []
        for tp in sorted(timepoints):
            try:
                tp_data = self.parse_timepoint(tp)
                cell = tp_data.get_cell(cell_name)
                if cell is not None:
                    trajectory.append(cell)
            except FileNotFoundError:
                continue

        return trajectory

    def iter_cell_trajectories(
        self,
        timepoints: Optional[List[int]] = None,
    ) -> Iterator[Tuple[str, List[CellPosition]]]:
        """
        Iterate over all cell trajectories.

        Args:
            timepoints: Optional list of timepoints to use.
                       If None, uses all available timepoints.

        Yields:
            Tuple of (cell_name, list of positions)
        """
        if timepoints is None:
            timepoints = self.get_available_timepoints()

        # Collect all cells across timepoints
        cell_positions: Dict[str, List[CellPosition]] = {}

        for tp in timepoints:
            try:
                tp_data = self.parse_timepoint(tp)
                for cell in tp_data.named_cells:
                    if cell.cell_name not in cell_positions:
                        cell_positions[cell.cell_name] = []
                    cell_positions[cell.cell_name].append(cell)
            except FileNotFoundError:
                continue

        # Yield sorted by cell name
        for name in sorted(cell_positions.keys()):
            positions = sorted(cell_positions[name], key=lambda c: c.timepoint)
            yield name, positions

    def get_all_cell_names(
        self,
        timepoints: Optional[List[int]] = None,
    ) -> List[str]:
        """
        Get all unique cell names across timepoints.

        Args:
            timepoints: Optional list of timepoints to check.
                       If None, uses all available timepoints.

        Returns:
            Sorted list of unique cell names
        """
        if timepoints is None:
            timepoints = self.get_available_timepoints()

        names = set()
        for tp in timepoints:
            try:
                tp_data = self.parse_timepoint(tp)
                names.update(tp_data.cell_names)
            except FileNotFoundError:
                continue

        return sorted(names)

    def build_position_matrix(
        self,
        timepoint: int,
        cell_names: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Build a position matrix for specified cells at a timepoint.

        Args:
            timepoint: The timepoint to use
            cell_names: Optional list of cell names to include.
                       If None, includes all named cells at that timepoint.

        Returns:
            Tuple of (Nx3 position matrix, list of cell names in same order)
        """
        tp_data = self.parse_timepoint(timepoint)

        if cell_names is None:
            cells = tp_data.named_cells
            names = [c.cell_name for c in cells]
            positions = np.array([[c.x, c.y, c.z] for c in cells])
        else:
            positions = []
            names = []
            for name in cell_names:
                cell = tp_data.get_cell(name)
                if cell is not None:
                    positions.append([cell.x, cell.y, cell.z])
                    names.append(name)
            positions = np.array(positions) if positions else np.zeros((0, 3))

        return positions, names

    def build_4d_tensor(
        self,
        timepoints: Optional[List[int]] = None,
        cell_names: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[int], List[str]]:
        """
        Build a 4D tensor of cell positions over time.

        Args:
            timepoints: List of timepoints to include.
                       If None, uses all available timepoints.
            cell_names: List of cell names to include.
                       If None, uses all cells that appear in any timepoint.

        Returns:
            Tuple of:
                - TxNx3 tensor (timepoints x cells x xyz)
                - List of timepoints
                - List of cell names
            Missing positions are filled with NaN.
        """
        if timepoints is None:
            timepoints = self.get_available_timepoints()

        if cell_names is None:
            cell_names = self.get_all_cell_names(timepoints)

        name_to_idx = {name: i for i, name in enumerate(cell_names)}

        T = len(timepoints)
        N = len(cell_names)

        tensor = np.full((T, N, 3), np.nan)

        for t_idx, tp in enumerate(timepoints):
            try:
                tp_data = self.parse_timepoint(tp)
                for cell in tp_data.named_cells:
                    if cell.cell_name in name_to_idx:
                        n_idx = name_to_idx[cell.cell_name]
                        tensor[t_idx, n_idx, :] = [cell.x, cell.y, cell.z]
            except FileNotFoundError:
                continue

        return tensor, timepoints, cell_names

    def clear_cache(self) -> None:
        """Clear the timepoint cache."""
        self._cache.clear()


def compute_cell_distances(
    positions: np.ndarray,
    metric: str = "euclidean",
) -> np.ndarray:
    """
    Compute pairwise distances between cells.

    Args:
        positions: Nx3 array of cell positions
        metric: Distance metric ("euclidean" or "manhattan")

    Returns:
        NxN distance matrix
    """
    N = positions.shape[0]
    if N == 0:
        return np.zeros((0, 0))

    if metric == "euclidean":
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        return np.sqrt(np.sum(diff**2, axis=2))
    elif metric == "manhattan":
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        return np.sum(np.abs(diff), axis=2)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def compute_cell_neighbors(
    positions: np.ndarray,
    cell_names: List[str],
    k: int = 5,
) -> Dict[str, List[str]]:
    """
    Find k-nearest neighbors for each cell.

    Args:
        positions: Nx3 array of cell positions
        cell_names: List of cell names (length N)
        k: Number of neighbors to find

    Returns:
        Dictionary mapping cell name -> list of neighbor names
    """
    distances = compute_cell_distances(positions)
    N = len(cell_names)
    k = min(k, N - 1)  # Can't have more neighbors than cells - 1

    neighbors = {}
    for i, name in enumerate(cell_names):
        # Get indices of k smallest distances (excluding self)
        dist_from_i = distances[i].copy()
        dist_from_i[i] = np.inf  # Exclude self
        neighbor_indices = np.argsort(dist_from_i)[:k]
        neighbors[name] = [cell_names[j] for j in neighbor_indices]

    return neighbors
