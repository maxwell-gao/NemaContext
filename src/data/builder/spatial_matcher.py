"""
Spatial coordinate matcher for WormGUIDES nuclei data.

Matches cells from transcriptome datasets (Large2025/Packer2019) to
4D spatial positions from WormGUIDES based on lineage names or cell types.
"""

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Column indices in WormGUIDES nuclei files (0-based)
NUCLEI_COL_ID = 0
NUCLEI_COL_FLAG = 1
NUCLEI_COL_X = 5
NUCLEI_COL_Y = 6
NUCLEI_COL_Z = 7
NUCLEI_COL_DIAMETER = 8
NUCLEI_COL_CELL_NAME = 9

# WormGUIDES time parameters
WORMGUIDES_START_TIME_MIN = 20  # First timepoint corresponds to ~20 min
WORMGUIDES_TIME_RESOLUTION_SEC = 60  # 1 minute per timepoint


class SpatialMatcher:
    """
    Matches cells to WormGUIDES 4D spatial coordinates.

    WormGUIDES provides nuclear positions for ~360 timepoints covering
    approximately 20-380 minutes of embryo development. Each timepoint
    file contains cell names and their XYZ coordinates.

    Matching can be done by:
    1. Exact lineage name match (e.g., "ABplpapppa")
    2. Cell type to lineage mapping via WormAtlas
    3. Nearest timepoint matching based on embryo time
    """

    def __init__(
        self,
        data_dir: str = "dataset/raw",
        cache_spatial: bool = True,
    ):
        """
        Initialize the spatial matcher.

        Args:
            data_dir: Base directory containing raw data.
            cache_spatial: Whether to cache loaded spatial data in memory.
        """
        self.data_dir = Path(data_dir)
        self.wormguides_dir = self.data_dir / "wormguides" / "nuclei_files"
        self.cache_spatial = cache_spatial

        # Cache structures
        self._spatial_cache: Dict[int, Dict[str, np.ndarray]] = {}
        self._all_cell_names: Optional[set] = None
        self._cell_timepoints: Optional[Dict[str, List[int]]] = None

    @property
    def nuclei_dir(self) -> Path:
        """Path to WormGUIDES nuclei files directory."""
        return self.wormguides_dir

    def get_available_timepoints(self) -> List[int]:
        """
        Get list of available timepoint indices.

        Returns:
            List of timepoint indices (1-360 typically).
        """
        if not self.wormguides_dir.exists():
            raise FileNotFoundError(
                f"WormGUIDES nuclei directory not found: {self.wormguides_dir}\n"
                "Run: uv run python -m src.data.downloader --source wormguides"
            )

        timepoints = []
        for f in self.wormguides_dir.glob("t*-nuclei"):
            match = re.match(r"t(\d+)-nuclei", f.name)
            if match:
                timepoints.append(int(match.group(1)))

        return sorted(timepoints)

    def timepoint_to_minutes(self, timepoint: int) -> float:
        """
        Convert WormGUIDES timepoint index to minutes post-fertilization.

        Args:
            timepoint: Timepoint index (1-based).

        Returns:
            Time in minutes from first cleavage.
        """
        return WORMGUIDES_START_TIME_MIN + (timepoint - 1)

    def minutes_to_timepoint(self, minutes: float) -> int:
        """
        Convert minutes post-fertilization to nearest WormGUIDES timepoint.

        Args:
            minutes: Time in minutes from first cleavage.

        Returns:
            Nearest timepoint index (clamped to valid range).
        """
        timepoints = self.get_available_timepoints()
        if not timepoints:
            raise ValueError("No timepoints available")

        tp = int(round(minutes - WORMGUIDES_START_TIME_MIN + 1))
        return max(timepoints[0], min(timepoints[-1], tp))

    def load_timepoint(
        self,
        timepoint: int,
        use_cache: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Load spatial coordinates for a specific timepoint.

        Args:
            timepoint: Timepoint index (1-based).
            use_cache: Whether to use/update cache.

        Returns:
            Dict mapping cell names to (x, y, z, diameter) arrays.
        """
        # Check cache
        if use_cache and self.cache_spatial and timepoint in self._spatial_cache:
            return self._spatial_cache[timepoint]

        # Load from file
        nuclei_file = self.wormguides_dir / f"t{timepoint:03d}-nuclei"
        if not nuclei_file.exists():
            raise FileNotFoundError(f"Nuclei file not found: {nuclei_file}")

        cells = {}
        with open(nuclei_file, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) > NUCLEI_COL_CELL_NAME:
                    cell_name = parts[NUCLEI_COL_CELL_NAME].strip().strip('"')
                    if cell_name and cell_name != "Nuc" and cell_name.strip():
                        try:
                            x = float(parts[NUCLEI_COL_X])
                            y = float(parts[NUCLEI_COL_Y])
                            z = float(parts[NUCLEI_COL_Z])
                            diameter = float(parts[NUCLEI_COL_DIAMETER])
                            cells[cell_name] = np.array([x, y, z, diameter])
                        except (ValueError, IndexError):
                            continue

        # Update cache
        if use_cache and self.cache_spatial:
            self._spatial_cache[timepoint] = cells

        return cells

    def get_all_cell_names(self) -> set:
        """
        Get all unique cell names across all timepoints.

        Returns:
            Set of all cell names found in WormGUIDES data.
        """
        if self._all_cell_names is not None:
            return self._all_cell_names

        all_names = set()
        for tp in self.get_available_timepoints():
            cells = self.load_timepoint(tp)
            all_names.update(cells.keys())

        self._all_cell_names = all_names
        logger.info(f"Found {len(all_names)} unique cell names in WormGUIDES")
        return all_names

    def get_cell_timepoints(self) -> Dict[str, List[int]]:
        """
        Get mapping of cell names to timepoints where they appear.

        Returns:
            Dict mapping cell name to list of timepoint indices.
        """
        if self._cell_timepoints is not None:
            return self._cell_timepoints

        cell_tps: Dict[str, List[int]] = defaultdict(list)
        for tp in self.get_available_timepoints():
            cells = self.load_timepoint(tp)
            for cell_name in cells:
                cell_tps[cell_name].append(tp)

        self._cell_timepoints = dict(cell_tps)
        return self._cell_timepoints

    def match_by_lineage(
        self,
        lineage_names: List[str],
        target_time: Optional[float] = None,
        time_window: float = 30.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Match cells by lineage name to spatial coordinates.

        Args:
            lineage_names: List of lineage names to match.
            target_time: Target embryo time in minutes (optional).
                        If provided, uses closest timepoint.
                        If None, averages across all timepoints where cell exists.
            time_window: Window (±minutes) around target_time to consider.

        Returns:
            Tuple of:
            - spatial_coords: (n_cells, 3) array of XYZ coordinates (NaN if no match)
            - matched_mask: (n_cells,) boolean array of successful matches
            - matched_timepoints: (n_cells,) array of matched timepoint indices (-1 if no match)
        """
        n_cells = len(lineage_names)
        spatial_coords = np.full((n_cells, 3), np.nan)
        matched_mask = np.zeros(n_cells, dtype=bool)
        matched_timepoints = np.full(n_cells, -1, dtype=int)

        # Get all cell names in WormGUIDES
        wg_cells = self.get_all_cell_names()
        cell_tps = self.get_cell_timepoints()

        for i, lineage in enumerate(lineage_names):
            if lineage not in wg_cells:
                continue

            # Get timepoints where this cell exists
            available_tps = cell_tps.get(lineage, [])
            if not available_tps:
                continue

            # Determine which timepoint(s) to use
            if target_time is not None:
                # Find closest timepoint within window
                target_tp = self.minutes_to_timepoint(target_time)
                min_tp = self.minutes_to_timepoint(target_time - time_window)
                max_tp = self.minutes_to_timepoint(target_time + time_window)

                valid_tps = [tp for tp in available_tps if min_tp <= tp <= max_tp]
                if not valid_tps:
                    # Use closest available
                    valid_tps = available_tps

                # Pick closest to target
                best_tp = min(valid_tps, key=lambda tp: abs(tp - target_tp))
                coords = self.load_timepoint(best_tp)[lineage][:3]
                matched_timepoints[i] = best_tp
            else:
                # Average across all timepoints
                all_coords = []
                for tp in available_tps:
                    cells = self.load_timepoint(tp)
                    if lineage in cells:
                        all_coords.append(cells[lineage][:3])

                coords = np.mean(all_coords, axis=0) if all_coords else None
                matched_timepoints[i] = available_tps[len(available_tps) // 2]

            if coords is not None:
                spatial_coords[i] = coords
                matched_mask[i] = True

        n_matched = matched_mask.sum()
        logger.info(f"Matched {n_matched}/{n_cells} cells by lineage name")

        return spatial_coords, matched_mask, matched_timepoints

    def match_by_time(
        self,
        cell_times: np.ndarray,
        lineage_names: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Match cells to spatial coordinates at their specific embryo time.

        For each cell, finds the closest timepoint to its developmental time
        and looks up its spatial coordinates by lineage name.

        Args:
            cell_times: (n_cells,) array of embryo times in minutes.
            lineage_names: List of lineage names corresponding to each cell.

        Returns:
            Tuple of (spatial_coords, matched_mask, matched_timepoints)
        """
        n_cells = len(lineage_names)
        spatial_coords = np.full((n_cells, 3), np.nan)
        matched_mask = np.zeros(n_cells, dtype=bool)
        matched_timepoints = np.full(n_cells, -1, dtype=int)

        wg_cells = self.get_all_cell_names()
        timepoints = self.get_available_timepoints()
        min_tp, max_tp = timepoints[0], timepoints[-1]

        for i, (time, lineage) in enumerate(zip(cell_times, lineage_names)):
            if pd.isna(time) or lineage not in wg_cells:
                continue

            # Check if time is within WormGUIDES range
            min_time = self.timepoint_to_minutes(min_tp)
            max_time = self.timepoint_to_minutes(max_tp)

            if time < min_time or time > max_time:
                continue

            # Get closest timepoint
            tp = self.minutes_to_timepoint(time)
            cells = self.load_timepoint(tp)

            if lineage in cells:
                spatial_coords[i] = cells[lineage][:3]
                matched_mask[i] = True
                matched_timepoints[i] = tp

        n_matched = matched_mask.sum()
        logger.info(f"Matched {n_matched}/{n_cells} cells by time")

        return spatial_coords, matched_mask, matched_timepoints

    def get_spatial_statistics(self) -> pd.DataFrame:
        """
        Get statistics about spatial data coverage.

        Returns:
            DataFrame with per-timepoint statistics.
        """
        stats = []
        for tp in self.get_available_timepoints():
            cells = self.load_timepoint(tp)
            named_cells = len(cells)

            if named_cells > 0:
                coords = np.array([c[:3] for c in cells.values()])
                stats.append(
                    {
                        "timepoint": tp,
                        "time_min": self.timepoint_to_minutes(tp),
                        "n_named_cells": named_cells,
                        "x_mean": coords[:, 0].mean(),
                        "y_mean": coords[:, 1].mean(),
                        "z_mean": coords[:, 2].mean(),
                        "x_std": coords[:, 0].std(),
                        "y_std": coords[:, 1].std(),
                        "z_std": coords[:, 2].std(),
                    }
                )

        return pd.DataFrame(stats)

    def clear_cache(self):
        """Clear all cached spatial data."""
        self._spatial_cache.clear()
        self._all_cell_names = None
        self._cell_timepoints = None
        logger.info("Spatial cache cleared")
