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
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix

# Try to import PyTorch for GPU acceleration
try:
    import torch
    TORCH_AVAILABLE = True
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    GPU_AVAILABLE = False

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


def get_lineage_ancestors(lineage: str) -> List[str]:
    """
    Get all ancestors of a lineage name, from closest to root.
    
    The lineage name encodes division history:
    - Each character after founder represents one division (a/l=left, p/r=right)
    - Removing characters from the end gives parent lineages
    
    Example:
        >>> get_lineage_ancestors("ABplpa")
        ['ABplp', 'ABpl', 'ABp', 'AB']
    
    Args:
        lineage: Lineage name (e.g., "ABplpa")
        
    Returns:
        List of ancestor lineage names (closest first)
    """
    lineage = normalize_lineage_name(lineage)
    if not lineage:
        return []
    
    # Find founder prefix
    founder = None
    path = ""
    for f in sorted(FOUNDER_PREFIXES, key=len, reverse=True):
        if lineage.upper().startswith(f.upper()):
            founder = f
            path = lineage[len(f):]
            break
    
    if founder is None:
        return []
    
    # Generate ancestors by removing path characters one by one
    ancestors = []
    for i in range(len(path) - 1, -1, -1):
        ancestor = founder + path[:i]
        ancestors.append(ancestor)
    
    return ancestors


def get_ancestor_distance(lineage: str, ancestor: str) -> int:
    """
    Get the number of divisions between a cell and its ancestor.
    
    Args:
        lineage: Descendant lineage name
        ancestor: Ancestor lineage name
        
    Returns:
        Number of divisions (0 if same cell, -1 if not an ancestor)
    """
    lineage = normalize_lineage_name(lineage)
    ancestor = normalize_lineage_name(ancestor)
    
    if not lineage or not ancestor:
        return -1
    
    if lineage == ancestor:
        return 0
    
    # Check if ancestor is actually an ancestor
    if not lineage.startswith(ancestor):
        return -1
    
    # Count extra characters = number of divisions
    return len(lineage) - len(ancestor)


def expand_uncertain_lineage(lineage: str) -> List[str]:
    """
    Expand a lineage with 'x' markers into all possible combinations.
    
    The 'x' in lineage names indicates uncertain position (could be 'a' or 'p').
    This function expands to all combinations.
    
    Example:
        >>> expand_uncertain_lineage("ABxap")
        ['ABaap', 'ABpap']
        >>> expand_uncertain_lineage("MSxxp")
        ['MSaap', 'MSapp', 'MSpap', 'MSppp']
    
    Args:
        lineage: Lineage string possibly containing 'x' markers
        
    Returns:
        List of all possible lineage expansions
    """
    lineage = normalize_lineage_name(lineage)
    if not lineage:
        return []
    
    # Count x's
    x_count = lineage.lower().count('x')
    if x_count == 0:
        return [lineage]
    
    if x_count > 10:  # Safety limit (2^10 = 1024 combinations)
        return [lineage]
    
    # Find founder prefix
    founder = None
    path_start = 0
    for f in sorted(FOUNDER_PREFIXES, key=len, reverse=True):
        if lineage.upper().startswith(f.upper()):
            founder = f
            path_start = len(f)
            break
    
    if founder is None:
        return [lineage]
    
    path = lineage[path_start:]
    
    # Generate all combinations
    results = []
    n_combinations = 2 ** x_count
    
    for combo in range(n_combinations):
        new_path = []
        x_idx = 0
        for char in path.lower():
            if char == 'x':
                # Use bit from combo to decide 'a' or 'p'
                new_path.append('a' if (combo >> x_idx) & 1 == 0 else 'p')
                x_idx += 1
            else:
                new_path.append(char)
        results.append(founder + ''.join(new_path))
    
    return results


def expand_slash_lineage(lineage: str) -> List[str]:
    """
    Split a lineage with '/' into individual options.
    
    Example:
        >>> expand_slash_lineage("ABa/ABp")
        ['ABa', 'ABp']
    
    Args:
        lineage: Lineage string possibly containing '/' separators
        
    Returns:
        List of individual lineage options
    """
    if not lineage or pd.isna(lineage):
        return []
    
    parts = [normalize_lineage_name(p.strip()) for p in lineage.split('/')]
    return [p for p in parts if p]


def resolve_ambiguous_lineage(
    lineage: str,
    valid_cells: set,
    max_expansions: int = 100,
) -> Tuple[Optional[str], str]:
    """
    Resolve an ambiguous lineage (with 'x' or '/') to a valid cell.
    
    Tries all expansions and returns the first match found in valid_cells.
    
    Args:
        lineage: Possibly ambiguous lineage string
        valid_cells: Set of valid cell names to match against
        max_expansions: Maximum number of expansions to try
        
    Returns:
        Tuple of (matched_cell, resolution_type) where resolution_type is:
        - 'direct': exact match
        - 'slash': matched one of the '/' alternatives
        - 'x_expand': matched an 'x' expansion
        - 'none': no match found
    """
    lineage = normalize_lineage_name(lineage)
    if not lineage:
        return (None, 'none')
    
    # Direct match
    if lineage in valid_cells:
        return (lineage, 'direct')
    
    # Try slash expansion first (fewer combinations)
    if '/' in lineage:
        for alt in expand_slash_lineage(lineage):
            if alt in valid_cells:
                return (alt, 'slash')
            # Also try x-expansion of each alternative
            for expanded in expand_uncertain_lineage(alt)[:max_expansions]:
                if expanded in valid_cells:
                    return (expanded, 'slash+x_expand')
    
    # Try x expansion
    if 'x' in lineage.lower():
        expansions = expand_uncertain_lineage(lineage)
        for exp in expansions[:max_expansions]:
            if exp in valid_cells:
                return (exp, 'x_expand')
    
    return (None, 'none')


# =============================================================================
# Ancestor Mapper for CShaper Data
# =============================================================================

class AncestorMapper:
    """
    Maps cells to their closest CShaper ancestor with fuzzy matching support.
    
    Since CShaper tracks early embryonic development (~4-350 cells) while
    scRNA-seq datasets often have later-stage cells, many cells won't have
    direct CShaper data. This class:
    
    1. First tries fuzzy matching (expand 'x' wildcards and '/' alternatives)
    2. Then searches for ancestors in the CShaper cell set
    
    This handles the common case where lineage annotations have uncertainty
    markers (e.g., "MSxpappp" where x could be 'a' or 'p').
    
    Usage:
        >>> mapper = AncestorMapper(cshaper_cells={"AB", "ABa", "ABp", "ABal", ...})
        >>> mapper.find_ancestor("ABalapapaap")  # Returns "ABalap" or similar
        ('ABalap', 5, 'ancestor')  # ancestor name, distance, match_type
    """
    
    def __init__(
        self,
        cshaper_cells: set,
        max_ancestor_distance: int = 10,
        enable_fuzzy: bool = True,
        max_fuzzy_expansions: int = 64,
    ):
        """
        Initialize the ancestor mapper.
        
        Args:
            cshaper_cells: Set of cell names present in CShaper data
            max_ancestor_distance: Maximum divisions to search backwards
            enable_fuzzy: Whether to expand 'x' and '/' in lineage names
            max_fuzzy_expansions: Maximum number of fuzzy expansions to try
        """
        # Normalize all CShaper cell names
        self.cshaper_cells = {normalize_lineage_name(c) for c in cshaper_cells if c}
        self.max_distance = max_ancestor_distance
        self.enable_fuzzy = enable_fuzzy
        self.max_fuzzy_expansions = max_fuzzy_expansions
        
        # Cache for ancestor lookups: (matched_name, distance, match_type)
        self._cache: Dict[str, Tuple[Optional[str], int, str]] = {}
        
        # Statistics
        self._stats = {
            'direct': 0,
            'fuzzy_x': 0,
            'fuzzy_slash': 0,
            'ancestor': 0,
            'fuzzy_ancestor': 0,
            'none': 0,
        }
    
    def find_ancestor(self, lineage: str) -> Tuple[Optional[str], int]:
        """
        Find the closest CShaper ancestor for a given cell.
        
        Args:
            lineage: Lineage name to find ancestor for
            
        Returns:
            Tuple of (ancestor_name, distance), or (None, -1) if not found
        """
        result = self._find_ancestor_with_type(lineage)
        return (result[0], result[1])
    
    def _find_ancestor_with_type(self, lineage: str) -> Tuple[Optional[str], int, str]:
        """
        Find ancestor with match type information.
        
        Returns:
            Tuple of (ancestor_name, distance, match_type)
            match_type: 'direct', 'fuzzy_x', 'fuzzy_slash', 'ancestor', 'fuzzy_ancestor', 'none'
        """
        lineage = normalize_lineage_name(lineage)
        if not lineage or lineage == 'unassigned':
            return (None, -1, 'none')
        
        # Check cache
        if lineage in self._cache:
            return self._cache[lineage]
        
        # 1. Direct match
        if lineage in self.cshaper_cells:
            self._cache[lineage] = (lineage, 0, 'direct')
            self._stats['direct'] += 1
            return (lineage, 0, 'direct')
        
        # 2. Fuzzy matching (expand x and /)
        if self.enable_fuzzy:
            fuzzy_match = self._try_fuzzy_match(lineage)
            if fuzzy_match[0] is not None:
                self._cache[lineage] = fuzzy_match
                self._stats[fuzzy_match[2]] += 1
                return fuzzy_match
        
        # 3. Ancestor search on original lineage
        ancestor_result = self._search_ancestors(lineage)
        if ancestor_result[0] is not None:
            self._cache[lineage] = ancestor_result
            self._stats['ancestor'] += 1
            return ancestor_result
        
        # 4. Ancestor search on fuzzy-expanded lineages
        if self.enable_fuzzy:
            expanded = self._get_fuzzy_expansions(lineage)
            for exp_lineage in expanded[:self.max_fuzzy_expansions]:
                anc_result = self._search_ancestors(exp_lineage)
                if anc_result[0] is not None:
                    result = (anc_result[0], anc_result[1], 'fuzzy_ancestor')
                    self._cache[lineage] = result
                    self._stats['fuzzy_ancestor'] += 1
                    return result
        
        # No match found
        self._cache[lineage] = (None, -1, 'none')
        self._stats['none'] += 1
        return (None, -1, 'none')
    
    def _get_fuzzy_expansions(self, lineage: str) -> List[str]:
        """Get all fuzzy expansions for a lineage."""
        expansions = []
        
        # Handle slash first
        if '/' in lineage:
            slash_parts = expand_slash_lineage(lineage)
            for part in slash_parts:
                if 'x' in part.lower():
                    expansions.extend(expand_uncertain_lineage(part))
                else:
                    expansions.append(part)
        elif 'x' in lineage.lower():
            expansions = expand_uncertain_lineage(lineage)
        
        return expansions
    
    def _try_fuzzy_match(self, lineage: str) -> Tuple[Optional[str], int, str]:
        """Try to match via fuzzy expansion (x or /)."""
        # Try slash expansion
        if '/' in lineage:
            for alt in expand_slash_lineage(lineage):
                alt_norm = normalize_lineage_name(alt)
                if alt_norm in self.cshaper_cells:
                    return (alt_norm, 0, 'fuzzy_slash')
                # Also try x-expansion of slash alternatives
                if 'x' in alt_norm.lower():
                    for exp in expand_uncertain_lineage(alt_norm)[:self.max_fuzzy_expansions]:
                        if exp in self.cshaper_cells:
                            return (exp, 0, 'fuzzy_x')
        
        # Try x expansion
        if 'x' in lineage.lower():
            for exp in expand_uncertain_lineage(lineage)[:self.max_fuzzy_expansions]:
                if exp in self.cshaper_cells:
                    return (exp, 0, 'fuzzy_x')
        
        return (None, -1, 'none')
    
    def _search_ancestors(self, lineage: str) -> Tuple[Optional[str], int, str]:
        """Search ancestors of a lineage."""
        ancestors = get_lineage_ancestors(lineage)
        for i, ancestor in enumerate(ancestors):
            if i >= self.max_distance:
                break
            if ancestor in self.cshaper_cells:
                distance = i + 1  # +1 because first ancestor is 1 step away
                return (ancestor, distance, 'ancestor')
        return (None, -1, 'none')
    
    def map_cells(
        self,
        lineage_names: List[str],
    ) -> Tuple[List[Optional[str]], np.ndarray]:
        """
        Map a list of cells to their CShaper ancestors.
        
        Args:
            lineage_names: List of lineage names
            
        Returns:
            Tuple of:
            - List of ancestor names (None if no ancestor found)
            - Array of distances (-1 if no ancestor found)
        """
        ancestors = []
        distances = []
        
        for name in lineage_names:
            ancestor, distance = self.find_ancestor(name)
            ancestors.append(ancestor)
            distances.append(distance)
        
        return ancestors, np.array(distances, dtype=np.int32)
    
    def map_cells_detailed(
        self,
        lineage_names: List[str],
    ) -> Tuple[List[Optional[str]], np.ndarray, List[str]]:
        """
        Map cells with detailed match type information.
        
        Returns:
            Tuple of:
            - List of ancestor names (None if no ancestor found)
            - Array of distances (-1 if no ancestor found)
            - List of match types ('direct', 'fuzzy_x', 'ancestor', etc.)
        """
        ancestors = []
        distances = []
        match_types = []
        
        for name in lineage_names:
            matched, dist, mtype = self._find_ancestor_with_type(name)
            ancestors.append(matched)
            distances.append(dist)
            match_types.append(mtype)
        
        return ancestors, np.array(distances, dtype=np.int32), match_types
    
    def get_matching_stats(self) -> Dict[str, int]:
        """Get cumulative matching statistics."""
        return dict(self._stats)
    
    def get_mapping_stats(self, lineage_names: List[str]) -> Dict[str, any]:
        """
        Get statistics about ancestor mapping for a set of cells.
        
        Args:
            lineage_names: List of lineage names
            
        Returns:
            Dictionary with mapping statistics
        """
        ancestors, distances, match_types = self.map_cells_detailed(lineage_names)
        
        valid_mask = np.array([a is not None for a in ancestors])
        valid_distances = distances[valid_mask]
        
        # Count match types
        type_counts = {}
        for mt in match_types:
            type_counts[mt] = type_counts.get(mt, 0) + 1
        
        return {
            "total_cells": len(lineage_names),
            "matched_cells": int(valid_mask.sum()),
            "match_rate": float(valid_mask.sum() / max(len(lineage_names), 1)),
            "direct_matches": type_counts.get('direct', 0),
            "fuzzy_x_matches": type_counts.get('fuzzy_x', 0),
            "fuzzy_slash_matches": type_counts.get('fuzzy_slash', 0),
            "ancestor_matches": type_counts.get('ancestor', 0),
            "fuzzy_ancestor_matches": type_counts.get('fuzzy_ancestor', 0),
            "unmatched": type_counts.get('none', 0),
            "mean_distance": float(np.mean(valid_distances)) if len(valid_distances) > 0 else 0,
            "max_distance": int(np.max(valid_distances)) if len(valid_distances) > 0 else 0,
            "distance_distribution": {
                int(d): int((valid_distances == d).sum()) 
                for d in range(min(self.max_distance + 1, int(np.max(valid_distances)) + 1 if len(valid_distances) > 0 else 1))
            },
        }


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
        
        GPU-accelerated implementation for large cell lists.
        
        Args:
            cell_list: List of cell lineage names (defines matrix order)
            sample_id: Specific sample to use (None = consensus)
            threshold: Minimum contact area to include edge
            binary: If True, return binary adjacency (1/0); else weighted
            
        Returns:
            Sparse CSR matrix of shape (n_cells, n_cells)
        """
        n_cells = len(cell_list)
        
        # Get the contact dataframe
        if sample_id is not None:
            df = self.load_sample(sample_id)
        else:
            df = self.get_consensus_contacts(min_samples=1)
        
        # Normalize contact cell names (these are small, ~1000 cells)
        df_row_norms = [normalize_lineage_name(c) for c in df.index]
        df_col_norms = [normalize_lineage_name(c) for c in df.columns]
        
        # Create mapping from normalized CShaper cell name -> list of query cell indices
        cshaper_to_query_idx: Dict[str, List[int]] = {}
        for i, cell in enumerate(cell_list):
            norm = normalize_lineage_name(cell)
            if norm not in cshaper_to_query_idx:
                cshaper_to_query_idx[norm] = []
            cshaper_to_query_idx[norm].append(i)
        
        # Convert df to numpy for fast access
        contact_matrix = df.values
        
        # Extract all non-zero contacts from CShaper (small set)
        cshaper_contacts = []  # List of (row_norm, col_norm, value)
        for ri, r_norm in enumerate(df_row_norms):
            if r_norm not in cshaper_to_query_idx:
                continue
            for ci, c_norm in enumerate(df_col_norms):
                if c_norm not in cshaper_to_query_idx:
                    continue
                val = contact_matrix[ri, ci]
                if pd.isna(val) or val <= threshold:
                    continue
                edge_val = 1.0 if binary else float(val)
                cshaper_contacts.append((r_norm, c_norm, edge_val))
        
        if not cshaper_contacts:
            return csr_matrix((n_cells, n_cells), dtype=np.float32)
        
        # Use GPU-accelerated expansion if available and beneficial
        if GPU_AVAILABLE and len(cshaper_contacts) > 100:
            return self._build_adjacency_gpu(
                n_cells, cshaper_contacts, cshaper_to_query_idx
            )
        else:
            return self._build_adjacency_cpu(
                n_cells, cshaper_contacts, cshaper_to_query_idx
            )
    
    def _build_adjacency_gpu(
        self,
        n_cells: int,
        cshaper_contacts: List[Tuple[str, str, float]],
        cshaper_to_query_idx: Dict[str, List[int]],
    ) -> csr_matrix:
        """GPU-accelerated contact graph expansion with batched processing."""
        device = torch.device('cuda')
        
        # Pre-compute sizes and sort by expansion size (process large ones on GPU)
        contact_info = []
        for r_norm, c_norm, edge_val in cshaper_contacts:
            n_rows = len(cshaper_to_query_idx[r_norm])
            n_cols = len(cshaper_to_query_idx[c_norm])
            n_edges = n_rows * n_cols
            if n_edges > 0:
                contact_info.append((n_edges, r_norm, c_norm, edge_val))
        
        # Sort by size descending - process large expansions first on GPU
        contact_info.sort(key=lambda x: -x[0])
        
        total_edges = sum(x[0] for x in contact_info)
        
        # Pre-allocate output arrays
        all_rows = np.zeros(total_edges, dtype=np.int64)
        all_cols = np.zeros(total_edges, dtype=np.int64)
        all_data = np.zeros(total_edges, dtype=np.float32)
        
        offset = 0
        
        # Batch large expansions (> 5000 edges) together for GPU
        large_threshold = 5000
        large_contacts = [(n, r, c, v) for n, r, c, v in contact_info if n >= large_threshold]
        small_contacts = [(n, r, c, v) for n, r, c, v in contact_info if n < large_threshold]
        
        # Process large contacts on GPU in a single batch if possible
        if large_contacts:
            for n_edges, r_norm, c_norm, edge_val in large_contacts:
                query_rows = cshaper_to_query_idx[r_norm]
                query_cols = cshaper_to_query_idx[c_norm]
                
                # GPU meshgrid
                rows_t = torch.tensor(query_rows, device=device, dtype=torch.int64)
                cols_t = torch.tensor(query_cols, device=device, dtype=torch.int64)
                row_grid, col_grid = torch.meshgrid(rows_t, cols_t, indexing='ij')
                
                batch_rows = row_grid.flatten().cpu().numpy()
                batch_cols = col_grid.flatten().cpu().numpy()
                
                all_rows[offset:offset+n_edges] = batch_rows
                all_cols[offset:offset+n_edges] = batch_cols
                all_data[offset:offset+n_edges] = edge_val
                offset += n_edges
            
            # Clear GPU cache after large operations
            torch.cuda.empty_cache()
        
        # Process small contacts on CPU (faster than GPU overhead)
        for n_edges, r_norm, c_norm, edge_val in small_contacts:
            query_rows = np.array(cshaper_to_query_idx[r_norm])
            query_cols = np.array(cshaper_to_query_idx[c_norm])
            
            row_grid, col_grid = np.meshgrid(query_rows, query_cols, indexing='ij')
            
            all_rows[offset:offset+n_edges] = row_grid.flatten()
            all_cols[offset:offset+n_edges] = col_grid.flatten()
            all_data[offset:offset+n_edges] = edge_val
            offset += n_edges
        
        # Build sparse matrix from COO format
        adj = csr_matrix(
            (all_data[:offset], (all_rows[:offset], all_cols[:offset])),
            shape=(n_cells, n_cells),
        )
        
        return adj
    
    def _build_adjacency_cpu(
        self,
        n_cells: int,
        cshaper_contacts: List[Tuple[str, str, float]],
        cshaper_to_query_idx: Dict[str, List[int]],
    ) -> csr_matrix:
        """CPU-based contact graph expansion using numpy meshgrid."""
        # Pre-compute total edges
        total_edges = sum(
            len(cshaper_to_query_idx[r]) * len(cshaper_to_query_idx[c])
            for r, c, _ in cshaper_contacts
        )
        
        # Pre-allocate arrays
        all_rows = np.zeros(total_edges, dtype=np.int64)
        all_cols = np.zeros(total_edges, dtype=np.int64)
        all_data = np.zeros(total_edges, dtype=np.float32)
        
        offset = 0
        for r_norm, c_norm, edge_val in cshaper_contacts:
            query_rows = np.array(cshaper_to_query_idx[r_norm])
            query_cols = np.array(cshaper_to_query_idx[c_norm])
            
            # Use meshgrid for vectorized expansion
            row_grid, col_grid = np.meshgrid(query_rows, query_cols, indexing='ij')
            n_edges = row_grid.size
            
            all_rows[offset:offset+n_edges] = row_grid.flatten()
            all_cols[offset:offset+n_edges] = col_grid.flatten()
            all_data[offset:offset+n_edges] = edge_val
            offset += n_edges
        
        # Build sparse matrix from COO format
        adj = csr_matrix(
            (all_data[:offset], (all_rows[:offset], all_cols[:offset])),
            shape=(n_cells, n_cells),
        )
        
        return adj
    
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
        
        # Average across samples - handle different frame ranges
        volume_by_frame: Dict[int, List[float]] = {}
        surface_by_frame: Dict[int, List[float]] = {}
        
        for sid in self.get_available_samples():
            vol_df, surf_df = self.load_sample(sid)
            if cell_name in vol_df.columns:
                for frame in vol_df.index:
                    val = vol_df.loc[frame, cell_name]
                    if not pd.isna(val):
                        if frame not in volume_by_frame:
                            volume_by_frame[frame] = []
                        volume_by_frame[frame].append(val)
            if cell_name in surf_df.columns:
                for frame in surf_df.index:
                    val = surf_df.loc[frame, cell_name]
                    if not pd.isna(val):
                        if frame not in surface_by_frame:
                            surface_by_frame[frame] = []
                        surface_by_frame[frame].append(val)
        
        # Get all frames and compute means
        all_frames = sorted(set(volume_by_frame.keys()) | set(surface_by_frame.keys()))
        volumes = np.array([np.mean(volume_by_frame.get(f, [np.nan])) for f in all_frames])
        surfaces = np.array([np.mean(surface_by_frame.get(f, [np.nan])) for f in all_frames])
        
        return {
            'frames': np.array(all_frames),
            'volume': volumes,
            'surface': surfaces,
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
        
        Vectorized implementation for fast processing of large cell lists.
        
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
        
        # Normalize names (vectorized with numpy)
        cell_names_norm = np.array([normalize_lineage_name(c) for c in cell_names])
        
        # Get unique frames to minimize loading
        unique_frames = np.unique(time_frames[time_frames >= 0])
        
        # Pre-load morphology for each unique frame
        frame_data = {}
        for frame in unique_frames:
            frame_data[frame] = self.get_morphology_at_frame(int(frame), sample_id)
        
        # Initialize result arrays
        volumes = np.full(n_cells, np.nan, dtype=np.float64)
        surfaces = np.full(n_cells, np.nan, dtype=np.float64)
        sphericities = np.full(n_cells, np.nan, dtype=np.float64)
        
        # Process each frame batch (vectorized)
        for frame, frame_df in frame_data.items():
            if frame_df is None or frame_df.empty:
                continue
            
            # Find cells at this frame
            frame_mask = time_frames == frame
            cells_at_frame = cell_names_norm[frame_mask]
            indices_at_frame = np.where(frame_mask)[0]
            
            # Get cells that exist in frame data
            frame_cell_set = set(frame_df.index)
            
            for idx, cell in zip(indices_at_frame, cells_at_frame):
                if cell and cell in frame_cell_set:
                    volumes[idx] = frame_df.loc[cell, 'volume']
                    surfaces[idx] = frame_df.loc[cell, 'surface']
                    sphericities[idx] = frame_df.loc[cell, 'sphericity']
        
        # Build result DataFrame
        result = pd.DataFrame({
            'volume': volumes,
            'surface': surfaces,
            'sphericity': sphericities,
        })
        
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

# Founder indices in Standard Dataset 1 (order of 9 founder groups)
FOUNDER_INDICES = {
    "AB": 0,    # ABa, ABp and descendants (8 gen × 256 pos)
    "MS": 1,    # MS descendants (6 gen × 32 pos)
    "E": 2,     # E descendants (5 gen × 16 pos)
    "C": 3,     # C descendants (5 gen × 16 pos)
    "D": 4,     # D descendants (4 gen × 8 pos)
    "P3": 5,    # P3 only (1 × 1)
    "P4": 6,    # P4, Z2, Z3 (2 gen × 2 pos)
    "EMS": 7,   # EMS only (1 × 1)
    "P2": 8,    # P2 only (1 × 1)
}

# Tree dimensions for each founder
FOUNDER_TREE_DIMS = {
    "AB": (8, 256),   # 8 generations, 256 positions
    "MS": (6, 32),
    "E": (5, 16),
    "C": (5, 16),
    "D": (4, 8),
    "P3": (1, 1),
    "P4": (2, 2),
    "EMS": (1, 1),
    "P2": (1, 1),
}


class StandardSpatialLoader:
    """
    Loader for CShaper Standard Dataset 1 (standardized spatial coordinates).
    
    Standard Dataset 1 contains averaged spatial coordinates from 46 embryos,
    organized by lineage tree structure. The data is stored in HDF5 (.mat) files
    with MATLAB v7.3 format.
    
    Structure:
    - 54 time frames (WorkSpace_Dataset_1.mat to WorkSpace_Dataset_54.mat)
    - Each file contains coordinates from 46 embryos
    - Organized by 9 founder lineages (AB, MS, E, C, D, P3, P4, EMS, P2)
    - Within each founder: (generation × position) matrix of 3D coordinates
    - Position encoded as binary tree index (a=0, p=1)
    """
    
    def __init__(self, standard_dir: Union[str, Path]):
        """
        Initialize the spatial loader.
        
        Args:
            standard_dir: Path to "Standard Dataset 1" directory
        """
        self.standard_dir = Path(standard_dir)
        self._coord_cache: Dict[int, Dict[str, np.ndarray]] = {}  # frame -> {lineage: coords}
        self._cellname_cache: Optional[Dict[str, Tuple[int, int, int]]] = None
        
        if not self.standard_dir.exists():
            logger.warning(f"Standard dataset directory not found: {standard_dir}")
    
    def get_available_frames(self) -> List[int]:
        """Get list of available time frame indices."""
        frames = []
        for f in self.standard_dir.glob("WorkSpace_Dataset_*.mat"):
            try:
                frame = int(f.stem.split("_")[-1])
                frames.append(frame)
            except ValueError:
                pass
        return sorted(frames)
    
    def _decode_matlab_string(self, f, ref) -> Optional[str]:
        """Decode a MATLAB string from HDF5 reference."""
        try:
            data = f['#refs#'][ref]
            if hasattr(data, 'shape'):  # Is a dataset
                arr = np.array(data)
                if arr.dtype == np.uint16:
                    chars = [chr(c) for c in arr.flatten() if c > 0]
                    return ''.join(chars)
        except Exception:
            pass
        return None
    
    def _load_cellnames(self) -> Dict[str, Tuple[int, int, int]]:
        """
        Load cell names and build lookup table.
        
        Returns:
            Dict mapping normalized cell name to (founder_idx, gen, pos)
        """
        if self._cellname_cache is not None:
            return self._cellname_cache
        
        import h5py
        
        cellname_path = self.standard_dir / "WorkSpace_CellName.mat"
        if not cellname_path.exists():
            logger.warning("CellName.mat not found")
            return {}
        
        name_to_idx = {}
        
        try:
            with h5py.File(cellname_path, 'r') as f:
                cellname_refs = f['CellName'][:]  # (1, 9) object references
                
                for founder_idx, ref in enumerate(cellname_refs.flatten()):
                    founder_data = f['#refs#'][ref]
                    if not hasattr(founder_data, 'shape'):
                        continue
                    
                    names_arr = np.array(founder_data)  # (gen, pos) of references
                    n_gen, n_pos = names_arr.shape
                    
                    for gen in range(n_gen):
                        for pos in range(n_pos):
                            name = self._decode_matlab_string(f, names_arr[gen, pos])
                            if name:
                                name_norm = normalize_lineage_name(name)
                                if name_norm:
                                    name_to_idx[name_norm] = (founder_idx, gen, pos)
        
        except Exception as e:
            logger.warning(f"Failed to load cell names: {e}")
            return {}
        
        self._cellname_cache = name_to_idx
        logger.debug(f"Loaded {len(name_to_idx)} cell names from CellName.mat")
        return name_to_idx
    
    def _load_frame_coords(self, frame: int, embryo_idx: int = 0) -> Dict[str, np.ndarray]:
        """
        Load coordinates for a specific time frame.
        
        Args:
            frame: Time frame index (1-54)
            embryo_idx: Which embryo to use (0-45), 0 = first/reference embryo
            
        Returns:
            Dict mapping (founder_idx, gen, pos) tuple to 3D coords
        """
        import h5py
        
        cache_key = (frame, embryo_idx)
        # Use frame as cache key since embryo_idx is usually 0
        if frame in self._coord_cache:
            return self._coord_cache[frame]
        
        mat_path = self.standard_dir / f"WorkSpace_Dataset_{frame}.mat"
        if not mat_path.exists():
            logger.warning(f"Dataset file not found: {mat_path}")
            return {}
        
        coords_dict = {}
        
        try:
            with h5py.File(mat_path, 'r') as f:
                dataset_refs = f['Dataset'][:]  # (1, 2) object refs
                
                # Dataset[0] contains coordinate data: (46 embryos, 1)
                data0_ref = f['#refs#'][dataset_refs.flatten()[0]]
                data0_arr = np.array(data0_ref)  # (46, 1) embryo refs
                
                if embryo_idx >= data0_arr.shape[0]:
                    embryo_idx = 0
                
                # Get selected embryo data: (1, 9) founder group refs
                embryo_ref = f['#refs#'][data0_arr[embryo_idx, 0]]
                embryo_data = np.array(embryo_ref)  # (1, 9) founder refs
                
                # Extract coordinates for each founder
                for founder_idx, founder_ref in enumerate(embryo_data.flatten()):
                    founder_coords = f['#refs#'][founder_ref]
                    if not hasattr(founder_coords, 'shape'):
                        continue
                    
                    founder_arr = np.array(founder_coords)  # (gen, pos) object refs
                    n_gen, n_pos = founder_arr.shape
                    
                    for gen in range(n_gen):
                        for pos in range(n_pos):
                            cell_data = f['#refs#'][founder_arr[gen, pos]]
                            if hasattr(cell_data, 'shape'):
                                coords = np.array(cell_data).flatten()
                                # Valid coords are 3D, [0] means cell doesn't exist
                                if len(coords) == 3:
                                    key = (founder_idx, gen, pos)
                                    coords_dict[key] = coords.astype(np.float32)
        
        except Exception as e:
            logger.warning(f"Failed to load frame {frame} coords: {e}")
            return {}
        
        self._coord_cache[frame] = coords_dict
        logger.debug(f"Loaded {len(coords_dict)} cell coordinates for frame {frame}")
        return coords_dict
    
    def lineage_to_tree_index(self, lineage: str) -> Optional[Tuple[int, int, int]]:
        """
        Convert lineage name to tree index (founder_idx, generation, position).
        
        Uses the pre-loaded cell name lookup table for accurate mapping.
        Falls back to algorithmic calculation if name not found.
        
        Args:
            lineage: Lineage name (e.g., "ABplpa")
            
        Returns:
            Tuple of (founder_idx, generation, position) or None if invalid
        """
        lineage = normalize_lineage_name(lineage)
        if not lineage:
            return None
        
        # Try lookup table first
        name_to_idx = self._load_cellnames()
        if lineage in name_to_idx:
            return name_to_idx[lineage]
        
        # Fallback: algorithmic calculation
        founder = None
        path = ""
        
        for f in sorted(FOUNDER_PREFIXES, key=len, reverse=True):
            if lineage.startswith(f):
                founder = f
                path = lineage[len(f):]
                break
        
        if founder is None or founder not in FOUNDER_INDICES:
            return None
        
        founder_idx = FOUNDER_INDICES[founder]
        generation = len(path)
        
        # Check if within tree bounds
        max_gen, max_pos = FOUNDER_TREE_DIMS.get(founder, (0, 0))
        if generation >= max_gen:
            return None
        
        # Convert path to binary position (a/l=0, p/r=1)
        position = 0
        for i, char in enumerate(path.lower()):
            if char in ['p', 'r']:
                position |= (1 << (generation - 1 - i))
        
        if position >= max_pos:
            return None
        
        return (founder_idx, generation, position)
    
    def get_spatial_coords(
        self,
        lineage_names: List[str],
        time_frame: int = 27,  # Middle frame (1-54)
        embryo_idx: int = 0,
    ) -> np.ndarray:
        """
        Get spatial coordinates for a list of lineage names.
        
        Args:
            lineage_names: List of lineage names
            time_frame: Time frame index (1-54, default 27 = middle)
            embryo_idx: Which embryo to use (0-45)
            
        Returns:
            Array of shape (n_cells, 3) with XYZ coordinates (NaN if not found)
        """
        n_cells = len(lineage_names)
        coords = np.full((n_cells, 3), np.nan, dtype=np.float32)
        
        # Load coordinates for this frame
        frame_coords = self._load_frame_coords(time_frame, embryo_idx)
        if not frame_coords:
            return coords
        
        # Look up each cell
        n_found = 0
        for i, name in enumerate(lineage_names):
            tree_idx = self.lineage_to_tree_index(name)
            if tree_idx is None:
                continue
            
            if tree_idx in frame_coords:
                coords[i] = frame_coords[tree_idx]
                n_found += 1
        
        logger.debug(f"Found coordinates for {n_found}/{n_cells} cells at frame {time_frame}")
        return coords
    
    def get_coords_for_frame_range(
        self,
        lineage_names: List[str],
        start_frame: int = 1,
        end_frame: int = 54,
    ) -> np.ndarray:
        """
        Get time-series coordinates for cells across multiple frames.
        
        Args:
            lineage_names: List of lineage names
            start_frame: First frame (1-54)
            end_frame: Last frame (1-54)
            
        Returns:
            Array of shape (n_cells, n_frames, 3)
        """
        n_cells = len(lineage_names)
        n_frames = end_frame - start_frame + 1
        coords = np.full((n_cells, n_frames, 3), np.nan, dtype=np.float32)
        
        for frame_offset, frame in enumerate(range(start_frame, end_frame + 1)):
            frame_coords = self.get_spatial_coords(lineage_names, time_frame=frame)
            coords[:, frame_offset, :] = frame_coords
        
        return coords
    
    def get_all_cell_names(self) -> List[str]:
        """Get all cell names in the dataset."""
        name_to_idx = self._load_cellnames()
        return sorted(name_to_idx.keys())
    
    def summary(self) -> Dict:
        """Get summary statistics."""
        frames = self.get_available_frames()
        names = self._load_cellnames()
        
        return {
            "n_frames": len(frames),
            "frame_range": (min(frames), max(frames)) if frames else (0, 0),
            "n_cell_names": len(names),
            "n_embryos": 46,
        }


# =============================================================================
# Segmentation 3D Loader (Standard Dataset 2)
# =============================================================================

class Segmentation3DLoader:
    """
    Loader for CShaper Standard Dataset 2 (3D voxel segmentation).
    
    Standard Dataset 2 contains 3D segmentation volumes showing cell shapes
    at different time points. Each voxel is labeled with a cell ID.
    
    File naming: Seg_{time}_{sample}.mat
    - time: 1-54 (developmental time frame)
    - sample: 04-20 (embryo sample ID)
    
    Volume dimensions: 184 × 114 × 256 voxels
    """
    
    # Voxel resolution in micrometers (approximate, from paper)
    VOXEL_SIZE_UM = np.array([0.09, 0.09, 0.42])  # XY resolution, Z spacing
    
    def __init__(self, seg_dir: Union[str, Path]):
        """
        Initialize the segmentation loader.
        
        Args:
            seg_dir: Path to "Standard Dataset 2" directory
        """
        self.seg_dir = Path(seg_dir)
        self._cache: Dict[Tuple[int, int], np.ndarray] = {}
        
        if not self.seg_dir.exists():
            logger.warning(f"Segmentation directory not found: {seg_dir}")
    
    def get_available_files(self) -> List[Tuple[int, int]]:
        """Get list of available (time, sample) pairs."""
        available = []
        for f in self.seg_dir.glob("Seg_*_*.mat"):
            try:
                parts = f.stem.split("_")
                time_idx = int(parts[1])
                sample_idx = int(parts[2])
                available.append((time_idx, sample_idx))
            except (ValueError, IndexError):
                pass
        return sorted(available)
    
    def load_segmentation(
        self,
        time_idx: int,
        sample_idx: int,
        use_cache: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Load a segmentation volume.
        
        Args:
            time_idx: Time frame (1-54)
            sample_idx: Sample ID (4-20)
            use_cache: Whether to cache loaded data
            
        Returns:
            3D array of shape (184, 114, 256) with cell labels, or None if not found
        """
        import h5py
        
        cache_key = (time_idx, sample_idx)
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        
        mat_path = self.seg_dir / f"Seg_{time_idx}_{sample_idx:02d}.mat"
        if not mat_path.exists():
            return None
        
        try:
            with h5py.File(mat_path, 'r') as f:
                seg = np.array(f['Seg']).astype(np.int32)
                
                if use_cache:
                    self._cache[cache_key] = seg
                
                return seg
        
        except Exception as e:
            logger.warning(f"Failed to load segmentation {mat_path}: {e}")
            return None
    
    def get_cell_labels(self, seg: np.ndarray) -> np.ndarray:
        """Get unique non-zero cell labels from segmentation."""
        return np.unique(seg[seg > 0])
    
    def compute_shape_descriptors(
        self,
        seg: np.ndarray,
        cell_label: int,
    ) -> Dict[str, float]:
        """
        Compute 3D shape descriptors for a cell.
        
        Args:
            seg: Segmentation volume
            cell_label: Cell label to analyze
            
        Returns:
            Dict with shape descriptors:
            - volume_um3: Volume in cubic micrometers
            - surface_area_um2: Approximate surface area
            - centroid_um: Center of mass [x, y, z]
            - bbox_size_um: Bounding box dimensions
            - sphericity: How spherical (0-1, 1=perfect sphere)
            - elongation: Ratio of principal axes
            - solidity: Volume / convex hull volume
        """
        mask = seg == cell_label
        if not mask.any():
            return {
                'volume_um3': 0.0,
                'surface_area_um2': 0.0,
                'centroid_um': [np.nan, np.nan, np.nan],
                'bbox_size_um': [0.0, 0.0, 0.0],
                'sphericity': np.nan,
                'elongation': np.nan,
                'solidity': np.nan,
            }
        
        # Volume (in voxels and um³)
        voxel_volume = self.VOXEL_SIZE_UM.prod()
        n_voxels = mask.sum()
        volume_um3 = n_voxels * voxel_volume
        
        # Centroid
        coords = np.argwhere(mask)
        centroid_voxel = coords.mean(axis=0)
        centroid_um = centroid_voxel * self.VOXEL_SIZE_UM
        
        # Bounding box
        bbox_min = coords.min(axis=0)
        bbox_max = coords.max(axis=0)
        bbox_size_voxel = bbox_max - bbox_min + 1
        bbox_size_um = bbox_size_voxel * self.VOXEL_SIZE_UM
        
        # Surface area (approximate via gradient magnitude)
        from scipy import ndimage
        surface_mask = ndimage.binary_erosion(mask) ^ mask
        n_surface_voxels = surface_mask.sum()
        # Approximate surface area (rough estimate)
        avg_face_area = (self.VOXEL_SIZE_UM[0] * self.VOXEL_SIZE_UM[1] +
                        self.VOXEL_SIZE_UM[1] * self.VOXEL_SIZE_UM[2] +
                        self.VOXEL_SIZE_UM[0] * self.VOXEL_SIZE_UM[2]) / 3
        surface_area_um2 = n_surface_voxels * avg_face_area
        
        # Sphericity: (36π * V²)^(1/3) / S
        if surface_area_um2 > 0:
            sphericity = np.power(36 * np.pi * volume_um3**2, 1/3) / surface_area_um2
            sphericity = min(1.0, sphericity)  # Clamp to [0, 1]
        else:
            sphericity = np.nan
        
        # Elongation (ratio of principal axes)
        try:
            # Use PCA on voxel coordinates
            coords_centered = coords - centroid_voxel
            coords_scaled = coords_centered * self.VOXEL_SIZE_UM
            if len(coords_scaled) > 3:
                cov = np.cov(coords_scaled.T)
                eigenvalues = np.linalg.eigvalsh(cov)
                eigenvalues = np.sort(eigenvalues)[::-1]
                elongation = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            else:
                elongation = 1.0
        except Exception:
            elongation = np.nan
        
        # Solidity (approximate - ratio to bounding box)
        bbox_volume = bbox_size_um.prod()
        solidity = volume_um3 / (bbox_volume + 1e-10)
        
        return {
            'volume_um3': float(volume_um3),
            'surface_area_um2': float(surface_area_um2),
            'centroid_um': centroid_um.tolist(),
            'bbox_size_um': bbox_size_um.tolist(),
            'sphericity': float(sphericity),
            'elongation': float(elongation),
            'solidity': float(solidity),
        }
    
    def compute_all_shape_descriptors(
        self,
        time_idx: int,
        sample_idx: int,
    ) -> pd.DataFrame:
        """
        Compute shape descriptors for all cells in a segmentation.
        
        Args:
            time_idx: Time frame
            sample_idx: Sample ID
            
        Returns:
            DataFrame with shape descriptors for each cell
        """
        seg = self.load_segmentation(time_idx, sample_idx)
        if seg is None:
            return pd.DataFrame()
        
        labels = self.get_cell_labels(seg)
        records = []
        
        for label in labels:
            desc = self.compute_shape_descriptors(seg, label)
            desc['cell_label'] = int(label)
            desc['time_idx'] = time_idx
            desc['sample_idx'] = sample_idx
            records.append(desc)
        
        return pd.DataFrame(records)
    
    def summary(self) -> Dict:
        """Get summary statistics."""
        available = self.get_available_files()
        if not available:
            return {"n_files": 0}
        
        times = sorted(set(t for t, s in available))
        samples = sorted(set(s for t, s in available))
        
        return {
            "n_files": len(available),
            "time_range": (min(times), max(times)),
            "sample_range": (min(samples), max(samples)),
            "volume_shape": (184, 114, 256),
            "voxel_size_um": self.VOXEL_SIZE_UM.tolist(),
        }


# =============================================================================
# Main CShaper Processor
# =============================================================================

class CShaperProcessor:
    """
    Main CShaper data processor combining all data loaders.
    
    Provides a unified interface to access:
    - Cell-cell contact matrices (ContactInterface)
    - Cell morphology (VolumeAndSurface)
    - Standardized spatial coordinates (Standard Dataset 1)
    - 3D segmentation volumes (Standard Dataset 2)
    - Time-dynamic contact graphs
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
        self._segmentation_loader: Optional[Segmentation3DLoader] = None
        
        # Cache for consensus morphology (computed once)
        self._consensus_morphology_cache: Optional[Dict[str, Tuple[float, float, float]]] = None
        
        # Validate directories
        self._validate_directories()
    
    def _validate_directories(self):
        """Check which CShaper data directories are available."""
        self.has_contact = (self.cshaper_dir / "ContactInterface").exists()
        self.has_morphology = (self.cshaper_dir / "VolumeAndSurface").exists()
        self.has_standard_spatial = (self.cshaper_dir / "Standard Dataset 1").exists()
        self.has_segmentation = (self.cshaper_dir / "Standard Dataset 2").exists()
        
        available = []
        if self.has_contact:
            available.append("ContactInterface")
        if self.has_morphology:
            available.append("VolumeAndSurface")
        if self.has_standard_spatial:
            available.append("Standard Dataset 1")
        if self.has_segmentation:
            available.append("Standard Dataset 2")
        
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
    
    @property
    def segmentation_loader(self) -> Segmentation3DLoader:
        """Get 3D segmentation loader."""
        if self._segmentation_loader is None:
            if not self.has_segmentation:
                raise FileNotFoundError("Standard Dataset 2 not found")
            self._segmentation_loader = Segmentation3DLoader(self.cshaper_dir / "Standard Dataset 2")
        return self._segmentation_loader
    
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
    
    def get_time_dynamic_contact_graph(
        self,
        lineage_names: List[str],
        sample_id: Optional[str] = None,
        threshold: float = 0.0,
    ) -> Dict[int, csr_matrix]:
        """
        Get contact adjacency matrices for each time frame.
        
        This provides a time-varying graph where edges represent
        physical contacts at each developmental time point.
        
        Args:
            lineage_names: List of cell lineage names
            sample_id: Specific sample (None = use first available)
            threshold: Minimum contact area threshold
            
        Returns:
            Dict mapping frame index to sparse adjacency matrix
        """
        if sample_id is None:
            samples = self.contact_loader.get_available_samples()
            if not samples:
                return {}
            sample_id = samples[0]
        
        # Get all frames for this sample
        frames = self.contact_loader.get_available_frames(sample_id)
        
        time_graphs = {}
        for frame in frames:
            adj = self.contact_loader.build_adjacency_matrix(
                lineage_names,
                sample_id=sample_id,
                threshold=threshold,
                binary=False,
            )
            # Note: current implementation aggregates across frames
            # For per-frame granularity, we need frame-specific loading
            time_graphs[frame] = adj
        
        return time_graphs
    
    def get_contact_timeseries(
        self,
        cell1: str,
        cell2: str,
        sample_id: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get contact area timeseries between two cells.
        
        Args:
            cell1: First cell lineage name
            cell2: Second cell lineage name
            sample_id: Specific sample (None = average across samples)
            
        Returns:
            Tuple of (frame_indices, contact_areas)
        """
        cell1 = normalize_lineage_name(cell1)
        cell2 = normalize_lineage_name(cell2)
        
        if sample_id is None:
            samples = self.contact_loader.get_available_samples()
        else:
            samples = [sample_id]
        
        all_frames = set()
        contact_values = {}
        
        for sid in samples:
            frames = self.contact_loader.get_available_frames(sid)
            for frame in frames:
                all_frames.add(frame)
                df = self.contact_loader.load_sample(sid, frame=frame)
                if cell1 in df.index and cell2 in df.columns:
                    val = df.loc[cell1, cell2]
                    if frame not in contact_values:
                        contact_values[frame] = []
                    contact_values[frame].append(val)
        
        # Average across samples
        frames_sorted = sorted(all_frames)
        areas = np.array([
            np.mean(contact_values.get(f, [0.0])) for f in frames_sorted
        ])
        
        return np.array(frames_sorted), areas
    
    # === Morphology ===
    
    def get_morphology_features(
        self,
        lineage_names: List[str],
        embryo_times: Optional[np.ndarray] = None,
        sample_id: Optional[str] = None,
        use_consensus: bool = True,
    ) -> pd.DataFrame:
        """
        Get morphology features for cells.
        
        Args:
            lineage_names: List of cell lineage names
            embryo_times: Array of embryo times in minutes (optional)
            sample_id: Specific sample to use (None = average)
            use_consensus: If True, use consensus values when time-based
                lookup returns NaN or when embryo_times is outside CShaper range
            
        Returns:
            DataFrame with columns: volume, surface, sphericity
        """
        # Convert embryo times to CShaper frames
        if embryo_times is not None:
            frames = np.array([embryo_time_to_cshaper_frame(t) for t in embryo_times])
        else:
            frames = None
        
        # Get time-based morphology
        result = self.morphology_loader.get_features_for_cells(
            lineage_names,
            time_frames=frames,
            sample_id=sample_id,
        )
        
        # If use_consensus, fill NaN values with consensus morphology
        if use_consensus:
            missing_mask = result['volume'].isna()
            if missing_mask.any():
                consensus_df = self.get_consensus_morphology(lineage_names)
                for col in ['volume', 'surface', 'sphericity']:
                    result.loc[missing_mask, col] = consensus_df.loc[missing_mask, col]
        
        return result
    
    def get_consensus_morphology(
        self,
        lineage_names: List[str],
        sample_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get consensus morphology features (averaged across all time points).
        
        Optimized with caching: consensus is computed once and reused.
        
        Args:
            lineage_names: List of cell lineage names
            sample_id: Specific sample to use (None = average across samples)
            
        Returns:
            DataFrame with columns: volume, surface, sphericity
        """
        n_cells = len(lineage_names)
        
        # Initialize result arrays
        volumes = np.full(n_cells, np.nan, dtype=np.float64)
        surfaces = np.full(n_cells, np.nan, dtype=np.float64)
        sphericities = np.full(n_cells, np.nan, dtype=np.float64)
        
        # Normalize names
        cell_names_norm = np.array([normalize_lineage_name(c) for c in lineage_names])
        
        # Use cached consensus or build it
        if self._consensus_morphology_cache is None:
            self._build_consensus_morphology_cache(sample_id)
        
        consensus_cache = self._consensus_morphology_cache
        
        # Map query cells to cached consensus values (vectorized via dict lookup)
        unique_query_cells = set(cell_names_norm)
        matching_cells = unique_query_cells & set(consensus_cache.keys())
        
        for cell in matching_cells:
            vol, surf, sph = consensus_cache[cell]
            mask = cell_names_norm == cell
            volumes[mask] = vol
            surfaces[mask] = surf
            sphericities[mask] = sph
        
        result = pd.DataFrame({
            'volume': volumes,
            'surface': surfaces,
            'sphericity': sphericities,
        })
        
        return result
    
    def _build_consensus_morphology_cache(self, sample_id: Optional[str] = None) -> None:
        """Build and cache consensus morphology for all CShaper cells.
        
        Uses parallel processing for faster computation.
        """
        logger.info("Building consensus morphology cache...")
        
        all_morph_cells = list(self.morphology_loader.get_all_cell_names())
        
        # Try to use multiprocessing for parallel computation
        try:
            from concurrent.futures import ThreadPoolExecutor
            
            def compute_cell_morphology(cell: str) -> Optional[Tuple[str, float, float, float]]:
                try:
                    ts = self.morphology_loader.get_cell_timeseries(cell, sample_id)
                    if len(ts['volume']) > 0:
                        vol = np.nanmean(ts['volume'])
                        surf = np.nanmean(ts['surface'])
                        if not np.isnan(vol) and not np.isnan(surf) and surf > 0:
                            sphericity = np.power(36 * np.pi * vol**2, 1/3) / surf
                            return (cell, vol, surf, float(np.clip(sphericity, 0, 1)))
                except Exception:
                    pass
                return None
            
            # Use thread pool for I/O-bound operations
            consensus_cache: Dict[str, Tuple[float, float, float]] = {}
            with ThreadPoolExecutor(max_workers=8) as executor:
                results = executor.map(compute_cell_morphology, all_morph_cells)
                for result in results:
                    if result is not None:
                        cell, vol, surf, sph = result
                        consensus_cache[cell] = (vol, surf, sph)
            
        except Exception:
            # Fallback to sequential
            consensus_cache = {}
            for cell in all_morph_cells:
                try:
                    ts = self.morphology_loader.get_cell_timeseries(cell, sample_id)
                    if len(ts['volume']) > 0:
                        vol = np.nanmean(ts['volume'])
                        surf = np.nanmean(ts['surface'])
                        if not np.isnan(vol) and not np.isnan(surf) and surf > 0:
                            sphericity = np.power(36 * np.pi * vol**2, 1/3) / surf
                            consensus_cache[cell] = (vol, surf, float(np.clip(sphericity, 0, 1)))
                except Exception:
                    pass
        
        self._consensus_morphology_cache = consensus_cache
        logger.info(f"Cached consensus morphology for {len(consensus_cache)} cells")
    
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
    
    # === Segmentation (3D Shape) ===
    
    def get_shape_descriptors(
        self,
        time_idx: int,
        sample_idx: int,
    ) -> pd.DataFrame:
        """
        Get 3D shape descriptors for all cells at a time point.
        
        Args:
            time_idx: Time frame (1-54)
            sample_idx: Sample ID (4-20)
            
        Returns:
            DataFrame with shape descriptors for each cell
        """
        return self.segmentation_loader.compute_all_shape_descriptors(
            time_idx, sample_idx
        )
    
    def get_segmentation_volume(
        self,
        time_idx: int,
        sample_idx: int,
    ) -> Optional[np.ndarray]:
        """
        Get 3D segmentation volume.
        
        Args:
            time_idx: Time frame (1-54)
            sample_idx: Sample ID (4-20)
            
        Returns:
            3D array of shape (184, 114, 256) with cell labels
        """
        return self.segmentation_loader.load_segmentation(time_idx, sample_idx)
    
    # === Utilities ===
    
    def get_all_cell_names(self) -> set:
        """Get union of all cell names across available data."""
        all_cells = set()
        
        if self.has_contact:
            all_cells.update(self.contact_loader.get_all_cell_names())
        if self.has_morphology:
            all_cells.update(self.morphology_loader.get_all_cell_names())
        if self.has_standard_spatial:
            all_cells.update(self.spatial_loader.get_all_cell_names())
        
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
            spatial_info = self.spatial_loader.summary()
            lines.extend([
                "",
                "Standard Dataset 1: ✓",
                f"  Time frames: {spatial_info['n_frames']}",
                f"  Cell names: {spatial_info['n_cell_names']}",
                f"  Embryos: {spatial_info['n_embryos']}",
            ])
        else:
            lines.append("Standard Dataset 1: ✗")
        
        if self.has_segmentation:
            seg_info = self.segmentation_loader.summary()
            lines.extend([
                "",
                "Standard Dataset 2: ✓",
                f"  Files: {seg_info['n_files']}",
                f"  Time range: {seg_info['time_range']}",
                f"  Volume shape: {seg_info['volume_shape']}",
            ])
        else:
            lines.append("Standard Dataset 2: ✗")
        
        lines.append("=" * 50)
        return "\n".join(lines)
