"""
CShaper morphological data processor.

This module provides utilities to load and process CShaper 4D morphological
atlas data, including:
- Cell-cell contact matrices (ContactInterface)
- Cell volume and surface area (VolumeAndSurface)
- Standardized spatial coordinates (Standard Dataset 1)

Paper: Cao et al. 2020 "Establishment of a morphological atlas of the
Caenorhabditis elegans embryo using deep-learning-based 4D segmentation"
DOI: 10.1038/s41467-020-19863-x
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# CShaper time parameters
CSHAPER_FRAMES = 54  # Total number of time frames
CSHAPER_START_TIME_MIN = 20  # Approximate start time (minutes post-fertilization)
CSHAPER_END_TIME_MIN = 380  # Approximate end time
CSHAPER_CELL_STAGE_RANGE = (4, 350)  # Cell count range

# Sample IDs available in CShaper
CSHAPER_SAMPLE_IDS = [f"Sample{i:02d}" for i in range(4, 21)]  # Sample04-Sample20

# Founder cells for lineage parsing
FOUNDER_PREFIXES = ["AB", "MS", "EMS", "P0", "P1", "P2", "P3", "P4", "Z2", "Z3", "E", "C", "D"]


# =============================================================================
# Helper Functions
# =============================================================================

def normalize_lineage_name(name: str) -> str:
    """
    Normalize a lineage name to standard format.
    
    Standard format: Founder in uppercase, path in lowercase
    Example: "ABplpapppa", "MSapaap", "Ealaad"
    
    Args:
        name: Raw lineage name string
        
    Returns:
        Normalized lineage name
    """
    if not name or pd.isna(name):
        return ""
    
    # Remove whitespace and periods
    name = str(name).strip().replace(" ", "").replace(".", "")
    
    if not name:
        return ""
    
    # Try to match known founder prefixes (longest first)
    for founder in sorted(FOUNDER_PREFIXES, key=len, reverse=True):
        if name.upper().startswith(founder.upper()):
            path = name[len(founder):]
            return founder + path.lower()
    
    # Single letter founders (E, C, D)
    if name[0].upper() in "ECD":
        return name[0].upper() + name[1:].lower()
    
    return name


def embryo_time_to_cshaper_frame(time_min: float) -> int:
    """
    Map embryo time (minutes) to CShaper frame number.
    
    Args:
        time_min: Embryo time in minutes post-fertilization
        
    Returns:
        CShaper frame index (0-53)
    """
    if pd.isna(time_min):
        return -1
    
    if time_min < CSHAPER_START_TIME_MIN:
        return 0
    if time_min > CSHAPER_END_TIME_MIN:
        return CSHAPER_FRAMES - 1
    
    # Linear mapping
    fraction = (time_min - CSHAPER_START_TIME_MIN) / (CSHAPER_END_TIME_MIN - CSHAPER_START_TIME_MIN)
    return int(fraction * (CSHAPER_FRAMES - 1))


def cshaper_frame_to_embryo_time(frame: int) -> float:
    """
    Map CShaper frame number to embryo time (minutes).
    
    Args:
        frame: CShaper frame index (0-53)
        
    Returns:
        Estimated embryo time in minutes
    """
    if frame < 0:
        return CSHAPER_START_TIME_MIN
    if frame >= CSHAPER_FRAMES:
        return CSHAPER_END_TIME_MIN
    
    fraction = frame / (CSHAPER_FRAMES - 1)
    return CSHAPER_START_TIME_MIN + fraction * (CSHAPER_END_TIME_MIN - CSHAPER_START_TIME_MIN)


# =============================================================================
# Contact Matrix Loader
# =============================================================================

class ContactLoader:
    """
    Loader for CShaper ContactInterface data.
    
    ContactInterface CSVs contain pairwise cell-cell contact data with
    time-series information. Each column represents a contact pair with
    contact area values at different time frames.
    
    File format: Sample{04-20}_Stat.csv
    - Row 1: "cell1" followed by first cell names in each contact pair
    - Row 2: "cell2" followed by second cell names in each contact pair  
    - Rows 3+: time_frame_index followed by contact area values (μm²)
    
    The loader parses this format and constructs symmetric adjacency matrices.
    """
    
    def __init__(self, contact_dir: Union[str, Path]):
        """
        Initialize the contact loader.
        
        Args:
            contact_dir: Path to ContactInterface directory
        """
        self.contact_dir = Path(contact_dir)
        self._cache: Dict[str, Dict[int, pd.DataFrame]] = {}  # sample -> {frame -> contact_matrix}
        self._all_cells: Optional[set] = None
        
        if not self.contact_dir.exists():
            raise FileNotFoundError(f"Contact directory not found: {contact_dir}")
    
    def get_available_samples(self) -> List[str]:
        """Get list of available sample IDs."""
        samples = []
        for f in self.contact_dir.glob("Sample*_Stat.csv"):
            match = re.match(r"(Sample\d+)_Stat\.csv", f.name)
            if match:
                samples.append(match.group(1))
        return sorted(samples)
    
    def _parse_contact_csv(self, csv_path: Path) -> Dict[int, pd.DataFrame]:
        """
        Parse contact CSV file into per-frame contact matrices.
        
        Args:
            csv_path: Path to the contact CSV file
            
        Returns:
            Dictionary mapping frame index -> symmetric contact matrix DataFrame
        """
        # Read raw CSV without any index
        raw_lines = []
        with open(csv_path, 'r') as f:
            for line in f:
                raw_lines.append(line.strip().split(','))
        
        if len(raw_lines) < 3:
            raise ValueError(f"Contact CSV too short: {csv_path}")
        
        # Row 0: "cell1" + first cells in each pair
        # Row 1: "cell2" + second cells in each pair
        # Row 2+: frame_idx + contact values
        
        cell1_row = raw_lines[0]
        cell2_row = raw_lines[1]
        
        # Skip the first column (header label)
        cell1_names = [normalize_lineage_name(c) for c in cell1_row[1:]]
        cell2_names = [normalize_lineage_name(c) for c in cell2_row[1:]]
        
        # Get all unique cells
        all_cells = sorted(set(c for c in cell1_names + cell2_names if c))
        cell_to_idx = {c: i for i, c in enumerate(all_cells)}
        n_cells = len(all_cells)
        
        # Parse each time frame
        frame_matrices = {}
        
        for row in raw_lines[2:]:
            if not row or not row[0]:
                continue
            
            try:
                frame_idx = int(row[0])
            except ValueError:
                continue
            
            # Initialize symmetric matrix for this frame
            matrix = np.zeros((n_cells, n_cells), dtype=np.float32)
            
            # Fill in contact values
            for col_idx, value_str in enumerate(row[1:]):
                if col_idx >= len(cell1_names):
                    break
                
                c1 = cell1_names[col_idx]
                c2 = cell2_names[col_idx]
                
                if not c1 or not c2:
                    continue
                
                # Parse contact area value
                try:
                    if value_str and value_str.strip():
                        value = float(value_str)
                    else:
                        value = 0.0
                except ValueError:
                    value = 0.0
                
                if value > 0:
                    i = cell_to_idx[c1]
                    j = cell_to_idx[c2]
                    # Make symmetric
                    matrix[i, j] = value
                    matrix[j, i] = value
            
            frame_matrices[frame_idx] = pd.DataFrame(
                matrix, index=all_cells, columns=all_cells
            )
        
        return frame_matrices
    
    def load_sample(
        self,
        sample_id: str,
        use_cache: bool = True,
        frame: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load contact matrix for a specific sample.
        
        Args:
            sample_id: Sample identifier (e.g., "Sample04")
            use_cache: Whether to cache loaded data
            frame: Specific frame to load (None = aggregate all frames)
            
        Returns:
            DataFrame with cell names as index/columns, contact areas as values
        """
        # Check cache
        if use_cache and sample_id in self._cache:
            frame_matrices = self._cache[sample_id]
        else:
            # Find and parse the file
            csv_path = self.contact_dir / f"{sample_id}_Stat.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"Contact file not found: {csv_path}")
            
            frame_matrices = self._parse_contact_csv(csv_path)
            
            if use_cache:
                self._cache[sample_id] = frame_matrices
        
        if not frame_matrices:
            raise ValueError(f"No frames found in {sample_id}")
        
        # Return specific frame or aggregate
        if frame is not None:
            if frame in frame_matrices:
                return frame_matrices[frame]
            else:
                # Return empty matrix with same cells
                ref_df = next(iter(frame_matrices.values()))
                return pd.DataFrame(0.0, index=ref_df.index, columns=ref_df.columns)
        
        # Aggregate: take maximum contact across all frames
        ref_df = next(iter(frame_matrices.values()))
        all_cells = ref_df.index.tolist()
        n_cells = len(all_cells)
        
        agg_matrix = np.zeros((n_cells, n_cells), dtype=np.float32)
        for fm in frame_matrices.values():
            agg_matrix = np.maximum(agg_matrix, fm.values)
        
        result = pd.DataFrame(agg_matrix, index=all_cells, columns=all_cells)
        logger.debug(f"Loaded contact matrix for {sample_id}: {result.shape}, {len(frame_matrices)} frames")
        return result
    
    def get_available_frames(self, sample_id: str) -> List[int]:
        """Get list of available frame indices for a sample."""
        if sample_id not in self._cache:
            self.load_sample(sample_id)
        return sorted(self._cache[sample_id].keys())
    
    def load_all_samples(self) -> Dict[str, pd.DataFrame]:
        """Load contact matrices for all available samples."""
        samples = self.get_available_samples()
        return {s: self.load_sample(s) for s in samples}
    
    def get_all_cell_names(self) -> set:
        """Get all unique cell names across all samples."""
        if self._all_cells is not None:
            return self._all_cells
        
        all_cells = set()
        for sample_id in self.get_available_samples():
            df = self.load_sample(sample_id)
            all_cells.update(df.index.tolist())
        
        self._all_cells = all_cells
        return all_cells
    
    def get_contact_strength(
        self,
        cell1: str,
        cell2: str,
        sample_id: Optional[str] = None,
        aggregation: str = "mean",
    ) -> float:
        """
        Get contact strength between two cells.
        
        Args:
            cell1: First cell lineage name
            cell2: Second cell lineage name
            sample_id: Specific sample to query (None = aggregate across samples)
            aggregation: How to aggregate across samples ("mean", "max", "any")
            
        Returns:
            Contact surface area (μm²), or 0 if no contact
        """
        cell1 = normalize_lineage_name(cell1)
        cell2 = normalize_lineage_name(cell2)
        
        if sample_id is not None:
            df = self.load_sample(sample_id)
            if cell1 in df.index and cell2 in df.columns:
                return float(df.loc[cell1, cell2])
            return 0.0
        
        # Aggregate across all samples
        values = []
        for sid in self.get_available_samples():
            df = self.load_sample(sid)
            if cell1 in df.index and cell2 in df.columns:
                val = df.loc[cell1, cell2]
                if val > 0:
                    values.append(val)
        
        if not values:
            return 0.0
        
        if aggregation == "mean":
            return float(np.mean(values))
        elif aggregation == "max":
            return float(np.max(values))
        elif aggregation == "any":
            return float(values[0]) if values else 0.0
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
    
    def get_consensus_contacts(
        self,
        min_samples: int = 3,
        min_area: float = 0.0,
    ) -> pd.DataFrame:
        """
        Get consensus contact matrix (contacts appearing in multiple samples).
        
        Args:
            min_samples: Minimum number of samples where contact must appear
            min_area: Minimum contact area threshold
            
        Returns:
            DataFrame with mean contact areas (NaN where < min_samples)
        """
        all_dfs = self.load_all_samples()
        
        # Get union of all cells
        all_cells = sorted(self.get_all_cell_names())
        
        # Stack all matrices
        n_cells = len(all_cells)
        cell_to_idx = {c: i for i, c in enumerate(all_cells)}
        
        count_matrix = np.zeros((n_cells, n_cells), dtype=np.int32)
        sum_matrix = np.zeros((n_cells, n_cells), dtype=np.float64)
        
        for df in all_dfs.values():
            for c1 in df.index:
                if c1 not in cell_to_idx:
                    continue
                i = cell_to_idx[c1]
                for c2 in df.columns:
                    if c2 not in cell_to_idx:
                        continue
                    j = cell_to_idx[c2]
                    val = df.loc[c1, c2]
                    if val > min_area:
                        count_matrix[i, j] += 1
                        sum_matrix[i, j] += val
        
        # Compute mean where count >= min_samples
        with np.errstate(divide='ignore', invalid='ignore'):
            mean_matrix = np.where(
                count_matrix >= min_samples,
                sum_matrix / count_matrix,
                np.nan
            )
        
        result = pd.DataFrame(mean_matrix, index=all_cells, columns=all_cells)
        return result
    
    def build_adjacency_matrix(
        self,
        cell_list: List[str],
        sample_id: Optional[str] = None,
        threshold: float = 0.0,
        binary: bool = False,
    ) -> csr_matrix:
        """
        Build sparse adjacency matrix for a list of cells.
        
        Args:
            cell_list: List of cell lineage names (defines matrix order)
            sample_id: Specific sample to use (None = consensus)
            threshold: Minimum contact area to include edge
            binary: If True, return binary adjacency (1/0); else weighted
            
        Returns:
            Sparse CSR matrix of shape (n_cells, n_cells)
        """
        n_cells = len(cell_list)
        cell_to_idx = {normalize_lineage_name(c): i for i, c in enumerate(cell_list)}
        
        # Use LIL format for efficient construction
        adj = lil_matrix((n_cells, n_cells), dtype=np.float32)
        
        if sample_id is not None:
            # Single sample
            df = self.load_sample(sample_id)
            for c1 in df.index:
                c1_norm = normalize_lineage_name(c1)
                if c1_norm not in cell_to_idx:
                    continue
                i = cell_to_idx[c1_norm]
                for c2 in df.columns:
                    c2_norm = normalize_lineage_name(c2)
                    if c2_norm not in cell_to_idx:
                        continue
                    j = cell_to_idx[c2_norm]
                    val = df.loc[c1, c2]
                    if val > threshold:
                        adj[i, j] = 1.0 if binary else val
        else:
            # Consensus across samples
            consensus = self.get_consensus_contacts(min_samples=1)
            for c1 in consensus.index:
                c1_norm = normalize_lineage_name(c1)
                if c1_norm not in cell_to_idx:
                    continue
                i = cell_to_idx[c1_norm]
                for c2 in consensus.columns:
                    c2_norm = normalize_lineage_name(c2)
                    if c2_norm not in cell_to_idx:
                        continue
                    j = cell_to_idx[c2_norm]
                    val = consensus.loc[c1, c2]
                    if not pd.isna(val) and val > threshold:
                        adj[i, j] = 1.0 if binary else val
        
        return adj.tocsr()
    
    def get_contact_statistics(self) -> Dict[str, float]:
        """Get summary statistics about contact data."""
        all_contacts = []
        all_degrees = []
        
        for df in self.load_all_samples().values():
            # Flatten upper triangle (avoid double counting)
            for i, c1 in enumerate(df.index):
                degree = 0
                for j, c2 in enumerate(df.columns):
                    if j > i:
                        val = df.loc[c1, c2]
                        if val > 0:
                            all_contacts.append(val)
                            degree += 1
                all_degrees.append(degree)
        
        return {
            "n_samples": len(self.get_available_samples()),
            "n_unique_cells": len(self.get_all_cell_names()),
            "n_total_contacts": len(all_contacts),
            "mean_contact_area": float(np.mean(all_contacts)) if all_contacts else 0.0,
            "max_contact_area": float(np.max(all_contacts)) if all_contacts else 0.0,
            "mean_degree": float(np.mean(all_degrees)) if all_degrees else 0.0,
        }


# =============================================================================
# Morphology (Volume/Surface) Loader
# =============================================================================

class MorphologyLoader:
    """
    Loader for CShaper VolumeAndSurface data.
    
    VolumeAndSurface CSVs contain time-series data of cell volumes and
    surface areas across developmental time frames.
    
    File format:
    - Sample{04-20}_volume.csv: Volume data (μm³)
    - Sample{04-20}_surface.csv: Surface area data (μm²)
    
    CSV structure:
    - Row 0-1: Cell names split across two header rows
    - Row 2+: time_frame_index followed by values
    - First column in data rows: frame index
    - Empty cells indicate the cell doesn't exist at that time frame
    """
    
    def __init__(self, volume_dir: Union[str, Path]):
        """
        Initialize the morphology loader.
        
        Args:
            volume_dir: Path to VolumeAndSurface directory
        """
        self.volume_dir = Path(volume_dir)
        self._cache: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
        
        if not self.volume_dir.exists():
            raise FileNotFoundError(f"Volume directory not found: {volume_dir}")
    
    def get_available_samples(self) -> List[str]:
        """Get list of available sample IDs."""
        samples = set()
        # Look for both volume and surface files
        for f in self.volume_dir.glob("Sample*_volume.csv"):
            match = re.match(r"(Sample\d+)_volume\.csv", f.name)
            if match:
                samples.add(match.group(1))
        for f in self.volume_dir.glob("Sample*_surface.csv"):
            match = re.match(r"(Sample\d+)_surface\.csv", f.name)
            if match:
                samples.add(match.group(1))
        return sorted(samples)
    
    def _parse_morphology_csv(self, csv_path: Path) -> pd.DataFrame:
        """
        Parse a morphology CSV file with multi-row headers.
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            DataFrame with frame as index, cell names as columns
        """
        # Read raw CSV
        raw_lines = []
        with open(csv_path, 'r') as f:
            for line in f:
                raw_lines.append(line.strip().split(','))
        
        if len(raw_lines) < 3:
            raise ValueError(f"Morphology CSV too short: {csv_path}")
        
        # Combine header rows to get cell names
        # Row 0 and Row 1 together form cell name columns
        # The cell names are split across the two rows due to line length
        header_row_1 = raw_lines[0]
        header_row_2 = raw_lines[1]
        
        # Build column names by combining both header rows
        # Each column either has a name in row 0 or row 1
        cell_names = []
        for i in range(1, max(len(header_row_1), len(header_row_2))):
            name1 = header_row_1[i] if i < len(header_row_1) else ""
            name2 = header_row_2[i] if i < len(header_row_2) else ""
            # Use whichever is non-empty
            name = name1 if name1 else name2
            cell_names.append(normalize_lineage_name(name))
        
        # Parse data rows
        data_rows = []
        frame_indices = []
        
        for row in raw_lines[2:]:
            if not row or not row[0]:
                continue
            
            # First value might be empty or frame index
            try:
                frame_idx = int(row[0])
            except ValueError:
                # Try the second element (some rows have empty first cell)
                if len(row) > 1:
                    try:
                        frame_idx = int(row[1])
                    except ValueError:
                        continue
                else:
                    continue
            
            frame_indices.append(frame_idx)
            
            # Parse values
            values = []
            for i, val_str in enumerate(row[1:]):
                if i >= len(cell_names):
                    break
                try:
                    if val_str and val_str.strip():
                        values.append(float(val_str))
                    else:
                        values.append(np.nan)
                except ValueError:
                    values.append(np.nan)
            
            # Pad if needed
            while len(values) < len(cell_names):
                values.append(np.nan)
            
            data_rows.append(values)
        
        # Build DataFrame
        df = pd.DataFrame(data_rows, index=frame_indices, columns=cell_names)
        
        # Remove columns with empty names
        df = df.loc[:, df.columns != ""]
        
        return df
    
    def load_sample(
        self,
        sample_id: str,
        use_cache: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load volume and surface data for a specific sample.
        
        Args:
            sample_id: Sample identifier (e.g., "Sample04")
            use_cache: Whether to cache loaded data
            
        Returns:
            Tuple of (volume_df, surface_df)
            - volume_df: DataFrame with time as index, cells as columns (μm³)
            - surface_df: DataFrame with time as index, cells as columns (μm²)
        """
        if use_cache and sample_id in self._cache:
            return self._cache[sample_id]
        
        volume_path = self.volume_dir / f"{sample_id}_volume.csv"
        surface_path = self.volume_dir / f"{sample_id}_surface.csv"
        
        # Load volume data
        if volume_path.exists():
            volume_df = self._parse_morphology_csv(volume_path)
        else:
            volume_df = pd.DataFrame()
        
        # Load surface data
        if surface_path.exists():
            surface_df = self._parse_morphology_csv(surface_path)
        else:
            surface_df = pd.DataFrame()
        
        if volume_df.empty and surface_df.empty:
            raise FileNotFoundError(
                f"Neither volume nor surface file found for {sample_id}: "
                f"{volume_path}, {surface_path}"
            )
        
        if use_cache:
            self._cache[sample_id] = (volume_df, surface_df)
        
        logger.debug(
            f"Loaded morphology for {sample_id}: "
            f"{len(volume_df.columns)} volume cols, {len(surface_df.columns)} surface cols"
        )
        return volume_df, surface_df
    
    def get_cell_timeseries(
        self,
        cell_name: str,
        sample_id: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Get volume/surface time series for a single cell.
        
        Args:
            cell_name: Cell lineage name
            sample_id: Specific sample (None = average across samples)
            
        Returns:
            Dict with 'volume', 'surface', 'frames' arrays
        """
        cell_name = normalize_lineage_name(cell_name)
        
        if sample_id is not None:
            vol_df, surf_df = self.load_sample(sample_id)
            return {
                'frames': vol_df.index.values,
                'volume': vol_df[cell_name].values if cell_name in vol_df.columns else np.array([]),
                'surface': surf_df[cell_name].values if cell_name in surf_df.columns else np.array([]),
            }
        
        # Average across samples
        all_volumes = []
        all_surfaces = []
        
        for sid in self.get_available_samples():
            vol_df, surf_df = self.load_sample(sid)
            if cell_name in vol_df.columns:
                all_volumes.append(vol_df[cell_name].values)
            if cell_name in surf_df.columns:
                all_surfaces.append(surf_df[cell_name].values)
        
        return {
            'frames': np.arange(CSHAPER_FRAMES),
            'volume': np.nanmean(all_volumes, axis=0) if all_volumes else np.array([]),
            'surface': np.nanmean(all_surfaces, axis=0) if all_surfaces else np.array([]),
        }
    
    def get_morphology_at_frame(
        self,
        frame: int,
        sample_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get morphology features for all cells at a specific time frame.
        
        Args:
            frame: Time frame index (0-53)
            sample_id: Specific sample (None = average across samples)
            
        Returns:
            DataFrame with columns: volume, surface, sphericity
        """
        if sample_id is not None:
            vol_df, surf_df = self.load_sample(sample_id)
            
            result = pd.DataFrame(index=vol_df.columns)
            result['volume'] = vol_df.loc[frame] if frame in vol_df.index else np.nan
            result['surface'] = surf_df.loc[frame] if frame in surf_df.index else np.nan
        else:
            # Average across samples
            all_cells = set()
            for sid in self.get_available_samples():
                vol_df, _ = self.load_sample(sid)
                all_cells.update(vol_df.columns)
            
            all_cells = sorted(all_cells)
            result = pd.DataFrame(index=all_cells)
            
            vol_values = {c: [] for c in all_cells}
            surf_values = {c: [] for c in all_cells}
            
            for sid in self.get_available_samples():
                vol_df, surf_df = self.load_sample(sid)
                for cell in all_cells:
                    if cell in vol_df.columns and frame in vol_df.index:
                        val = vol_df.loc[frame, cell]
                        if not pd.isna(val):
                            vol_values[cell].append(val)
                    if cell in surf_df.columns and frame in surf_df.index:
                        val = surf_df.loc[frame, cell]
                        if not pd.isna(val):
                            surf_values[cell].append(val)
            
            # Compute means safely
            result['volume'] = [
                np.mean(vol_values[c]) if len(vol_values[c]) > 0 else np.nan 
                for c in all_cells
            ]
            result['surface'] = [
                np.mean(surf_values[c]) if len(surf_values[c]) > 0 else np.nan 
                for c in all_cells
            ]
        
        # Compute sphericity: (36π * V²)^(1/3) / S
        # Ranges from 0 to 1, where 1 = perfect sphere
        with np.errstate(divide='ignore', invalid='ignore'):
            V = result['volume'].values
            S = result['surface'].values
            sphericity = np.power(36 * np.pi * V**2, 1/3) / S
            sphericity = np.where(S > 0, sphericity, np.nan)
            sphericity = np.clip(sphericity, 0, 1)  # Numerical stability
        result['sphericity'] = sphericity
        
        return result
    
    def get_features_for_cells(
        self,
        cell_names: List[str],
        time_frames: Optional[np.ndarray] = None,
        sample_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get morphology features for a list of cells.
        
        Args:
            cell_names: List of cell lineage names
            time_frames: Array of time frames for each cell (None = use middle frame)
            sample_id: Specific sample (None = average across samples)
            
        Returns:
            DataFrame with index matching cell_names, columns: volume, surface, sphericity
        """
        n_cells = len(cell_names)
        
        # Default to middle frame
        if time_frames is None:
            time_frames = np.full(n_cells, CSHAPER_FRAMES // 2, dtype=np.int32)
        
        # Normalize names
        cell_names_norm = [normalize_lineage_name(c) for c in cell_names]
        
        # Get unique frames to minimize loading
        unique_frames = np.unique(time_frames[time_frames >= 0])
        
        # Pre-load morphology for each unique frame
        frame_data = {}
        for frame in unique_frames:
            frame_data[frame] = self.get_morphology_at_frame(int(frame), sample_id)
        
        # Build result
        result = pd.DataFrame(
            index=range(n_cells),
            columns=['volume', 'surface', 'sphericity'],
            dtype=np.float64,  # Use float64 to avoid dtype conversion warnings
        )
        result[:] = np.nan
        
        for i, (cell, frame) in enumerate(zip(cell_names_norm, time_frames)):
            if frame < 0 or not cell:
                continue
            
            frame_df = frame_data.get(frame)
            if frame_df is not None and cell in frame_df.index:
                result.iloc[i, 0] = float(frame_df.loc[cell, 'volume'])  # volume
                result.iloc[i, 1] = float(frame_df.loc[cell, 'surface'])  # surface
                result.iloc[i, 2] = float(frame_df.loc[cell, 'sphericity'])  # sphericity
        
        return result
    
    def get_all_cell_names(self) -> set:
        """Get all unique cell names across all samples."""
        all_cells = set()
        for sid in self.get_available_samples():
            vol_df, _ = self.load_sample(sid)
            all_cells.update(vol_df.columns)
        return all_cells


# =============================================================================
# Standard Dataset Spatial Loader
# =============================================================================

class StandardSpatialLoader:
    """
    Loader for CShaper Standard Dataset 1 (standardized spatial coordinates).
    
    Standard Dataset 1 contains averaged spatial coordinates from 46 embryos,
    organized by lineage tree structure. The data is stored in HDF5 (.mat) files
    with MATLAB v7.3 format.
    
    Structure:
    - Organized by founder lineage (AB, MS, E, C, D, etc.)
    - Within each founder: (generation × position) matrix
    - Position encoded as binary tree index
    """
    
    def __init__(self, standard_dir: Union[str, Path]):
        """
        Initialize the spatial loader.
        
        Args:
            standard_dir: Path to "Standard Dataset 1" directory
        """
        self.standard_dir = Path(standard_dir)
        self._cache: Dict[str, Dict] = {}
        
        if not self.standard_dir.exists():
            logger.warning(f"Standard dataset directory not found: {standard_dir}")
    
    def get_available_files(self) -> List[Path]:
        """Get list of available .mat files."""
        return sorted(self.standard_dir.glob("*.mat"))
    
    def load_mat_file(self, mat_path: Path, use_cache: bool = True) -> Dict:
        """
        Load a Standard Dataset 1 .mat file (HDF5 format).
        
        Args:
            mat_path: Path to .mat file
            use_cache: Whether to cache loaded data
            
        Returns:
            Dictionary with extracted spatial data
        """
        import h5py
        
        cache_key = str(mat_path)
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        result = {}
        
        try:
            with h5py.File(mat_path, 'r') as f:
                for key in f.keys():
                    if key.startswith('#'):
                        continue
                    
                    data = f[key]
                    if isinstance(data, h5py.Dataset):
                        result[key] = np.array(data)
                    elif isinstance(data, h5py.Group):
                        # Nested structure
                        result[key] = self._extract_group(data)
        except Exception as e:
            logger.warning(f"Failed to load {mat_path}: {e}")
            return {}
        
        if use_cache:
            self._cache[cache_key] = result
        
        return result
    
    def _extract_group(self, group) -> Dict:
        """Recursively extract HDF5 group data."""
        import h5py
        
        result = {}
        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Dataset):
                result[key] = np.array(item)
            elif isinstance(item, h5py.Group):
                result[key] = self._extract_group(item)
        return result
    
    def lineage_to_tree_index(self, lineage: str) -> Optional[Tuple[str, int, int]]:
        """
        Convert lineage name to tree index (founder, generation, position).
        
        The tree index encodes the position in the binary lineage tree:
        - generation: number of divisions from founder
        - position: binary encoding of division path (a=0, p=1)
        
        Args:
            lineage: Lineage name (e.g., "ABplpa")
            
        Returns:
            Tuple of (founder, generation, position) or None if invalid
        """
        lineage = normalize_lineage_name(lineage)
        if not lineage:
            return None
        
        # Identify founder
        founder = None
        path = ""
        
        for f in sorted(FOUNDER_PREFIXES, key=len, reverse=True):
            if lineage.startswith(f):
                founder = f
                path = lineage[len(f):]
                break
        
        if founder is None:
            return None
        
        # Calculate generation and position
        generation = len(path)
        
        # Convert path to binary position (a=0, p=1)
        position = 0
        for i, char in enumerate(path.lower()):
            if char == 'p':
                position |= (1 << (generation - 1 - i))
            elif char not in ['a', 'l', 'r', 'd', 'v']:
                # Unknown character
                pass
        
        return (founder, generation, position)
    
    def get_spatial_coords(
        self,
        lineage_names: List[str],
        time_frame: int = 27,  # Middle frame
    ) -> np.ndarray:
        """
        Get spatial coordinates for a list of lineage names.
        
        Args:
            lineage_names: List of lineage names
            time_frame: Time frame index (0-53)
            
        Returns:
            Array of shape (n_cells, 3) with XYZ coordinates (NaN if not found)
        """
        n_cells = len(lineage_names)
        coords = np.full((n_cells, 3), np.nan, dtype=np.float32)
        
        # Load relevant .mat files
        mat_files = self.get_available_files()
        if not mat_files:
            logger.warning("No Standard Dataset 1 files found")
            return coords
        
        # For now, use first available file (should be enhanced to select by time)
        # In reality, files may be organized by time frame
        mat_data = self.load_mat_file(mat_files[0])
        
        # Extract coordinates for each cell
        for i, name in enumerate(lineage_names):
            tree_idx = self.lineage_to_tree_index(name)
            if tree_idx is None:
                continue
            
            founder, gen, pos = tree_idx
            
            # Try to find data for this founder
            # The exact structure depends on how the .mat files are organized
            # This is a placeholder implementation
            founder_key = founder.lower()
            if founder_key in mat_data:
                founder_data = mat_data[founder_key]
                # Data might be organized as (gen, pos, xyz) or similar
                # Need to adapt based on actual file structure
                pass
        
        return coords


# =============================================================================
# Main CShaper Processor
# =============================================================================

class CShaperProcessor:
    """
    Main CShaper data processor combining all data loaders.
    
    Provides a unified interface to access:
    - Cell-cell contact matrices
    - Cell morphology (volume, surface, sphericity)
    - Standardized spatial coordinates
    """
    
    def __init__(self, data_dir: str = "dataset/raw"):
        """
        Initialize the CShaper processor.
        
        Args:
            data_dir: Base directory containing raw data
        """
        self.data_dir = Path(data_dir)
        self.cshaper_dir = self.data_dir / "cshaper"
        
        # Initialize sub-loaders
        self._contact_loader: Optional[ContactLoader] = None
        self._morphology_loader: Optional[MorphologyLoader] = None
        self._spatial_loader: Optional[StandardSpatialLoader] = None
        
        # Validate directories
        self._validate_directories()
    
    def _validate_directories(self):
        """Check which CShaper data directories are available."""
        self.has_contact = (self.cshaper_dir / "ContactInterface").exists()
        self.has_morphology = (self.cshaper_dir / "VolumeAndSurface").exists()
        self.has_standard_spatial = (self.cshaper_dir / "Standard Dataset 1").exists()
        
        available = []
        if self.has_contact:
            available.append("ContactInterface")
        if self.has_morphology:
            available.append("VolumeAndSurface")
        if self.has_standard_spatial:
            available.append("Standard Dataset 1")
        
        if available:
            logger.info(f"CShaper data available: {', '.join(available)}")
        else:
            logger.warning(f"No CShaper data found in {self.cshaper_dir}")
    
    @property
    def contact_loader(self) -> ContactLoader:
        """Get contact matrix loader."""
        if self._contact_loader is None:
            if not self.has_contact:
                raise FileNotFoundError("ContactInterface data not found")
            self._contact_loader = ContactLoader(self.cshaper_dir / "ContactInterface")
        return self._contact_loader
    
    @property
    def morphology_loader(self) -> MorphologyLoader:
        """Get morphology data loader."""
        if self._morphology_loader is None:
            if not self.has_morphology:
                raise FileNotFoundError("VolumeAndSurface data not found")
            self._morphology_loader = MorphologyLoader(self.cshaper_dir / "VolumeAndSurface")
        return self._morphology_loader
    
    @property
    def spatial_loader(self) -> StandardSpatialLoader:
        """Get standard spatial data loader."""
        if self._spatial_loader is None:
            if not self.has_standard_spatial:
                raise FileNotFoundError("Standard Dataset 1 not found")
            self._spatial_loader = StandardSpatialLoader(self.cshaper_dir / "Standard Dataset 1")
        return self._spatial_loader
    
    # === Contact Interface ===
    
    def get_contact_adjacency(
        self,
        lineage_names: List[str],
        sample_id: Optional[str] = None,
        threshold: float = 0.0,
        binary: bool = False,
    ) -> csr_matrix:
        """
        Build contact adjacency matrix for given cells.
        
        Args:
            lineage_names: List of cell lineage names
            sample_id: Specific sample to use (None = consensus)
            threshold: Minimum contact area threshold
            binary: If True, return binary adjacency
            
        Returns:
            Sparse CSR matrix of shape (n_cells, n_cells)
        """
        return self.contact_loader.build_adjacency_matrix(
            lineage_names,
            sample_id=sample_id,
            threshold=threshold,
            binary=binary,
        )
    
    def get_contact_statistics(self) -> Dict:
        """Get summary statistics about contact data."""
        return self.contact_loader.get_contact_statistics()
    
    # === Morphology ===
    
    def get_morphology_features(
        self,
        lineage_names: List[str],
        embryo_times: Optional[np.ndarray] = None,
        sample_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get morphology features for cells.
        
        Args:
            lineage_names: List of cell lineage names
            embryo_times: Array of embryo times in minutes (optional)
            sample_id: Specific sample to use (None = average)
            
        Returns:
            DataFrame with columns: volume, surface, sphericity
        """
        # Convert embryo times to CShaper frames
        if embryo_times is not None:
            frames = np.array([embryo_time_to_cshaper_frame(t) for t in embryo_times])
        else:
            frames = None
        
        return self.morphology_loader.get_features_for_cells(
            lineage_names,
            time_frames=frames,
            sample_id=sample_id,
        )
    
    # === Spatial Coordinates ===
    
    def get_spatial_coords(
        self,
        lineage_names: List[str],
        embryo_times: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Get standardized spatial coordinates for cells.
        
        Args:
            lineage_names: List of cell lineage names
            embryo_times: Array of embryo times in minutes (optional)
            
        Returns:
            Array of shape (n_cells, 3) with XYZ coordinates
        """
        if embryo_times is not None:
            frame = int(np.median([embryo_time_to_cshaper_frame(t) for t in embryo_times if not pd.isna(t)]))
        else:
            frame = CSHAPER_FRAMES // 2
        
        return self.spatial_loader.get_spatial_coords(lineage_names, time_frame=frame)
    
    # === Utilities ===
    
    def get_all_cell_names(self) -> set:
        """Get union of all cell names across available data."""
        all_cells = set()
        
        if self.has_contact:
            all_cells.update(self.contact_loader.get_all_cell_names())
        if self.has_morphology:
            all_cells.update(self.morphology_loader.get_all_cell_names())
        
        return all_cells
    
    def summary(self) -> str:
        """Get a summary of available CShaper data."""
        lines = [
            "=" * 50,
            "CShaper Data Summary",
            "=" * 50,
            f"Data directory: {self.cshaper_dir}",
            "",
        ]
        
        if self.has_contact:
            stats = self.get_contact_statistics()
            lines.extend([
                "ContactInterface: ✓",
                f"  Samples: {stats['n_samples']}",
                f"  Unique cells: {stats['n_unique_cells']}",
                f"  Mean contact area: {stats['mean_contact_area']:.2f} μm²",
                f"  Mean degree: {stats['mean_degree']:.1f}",
            ])
        else:
            lines.append("ContactInterface: ✗")
        
        if self.has_morphology:
            n_cells = len(self.morphology_loader.get_all_cell_names())
            n_samples = len(self.morphology_loader.get_available_samples())
            lines.extend([
                "",
                "VolumeAndSurface: ✓",
                f"  Samples: {n_samples}",
                f"  Unique cells: {n_cells}",
                f"  Time frames: {CSHAPER_FRAMES}",
            ])
        else:
            lines.append("VolumeAndSurface: ✗")
        
        if self.has_standard_spatial:
            n_files = len(self.spatial_loader.get_available_files())
            lines.extend([
                "",
                "Standard Dataset 1: ✓",
                f"  Files: {n_files}",
            ])
        else:
            lines.append("Standard Dataset 1: ✗")
        
        lines.append("=" * 50)
        return "\n".join(lines)
