"""
Lineage encoder for C. elegans cell lineage strings.

Parses lineage names (e.g., "ABplpapppa") into structured representations:
- Binary path encoding (a=0, p=1)
- Depth (number of divisions from founder)
- Founder lineage identification
- Poincaré disk embedding coordinates (optional)
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Founder cells and their standard naming
FOUNDER_CELLS = {
    "P0": {"depth": 0, "founder": "P0"},
    "AB": {"depth": 1, "founder": "AB"},
    "P1": {"depth": 1, "founder": "P1"},
    "EMS": {"depth": 2, "founder": "EMS"},
    "P2": {"depth": 2, "founder": "P2"},
    "MS": {"depth": 3, "founder": "MS"},
    "E": {"depth": 3, "founder": "E"},
    "C": {"depth": 3, "founder": "C"},
    "P3": {"depth": 3, "founder": "P3"},
    "D": {"depth": 4, "founder": "D"},
    "P4": {"depth": 4, "founder": "P4"},
    "Z2": {"depth": 5, "founder": "Z2"},
    "Z3": {"depth": 5, "founder": "Z3"},
}

# Mapping of founder prefixes to their canonical names
FOUNDER_PREFIXES = {
    "AB": "AB",
    "MS": "MS",
    "E": "E",
    "C": "C",
    "D": "D",
    "P4": "P4",
    "Z2": "Z2",
    "Z3": "Z3",
    "EMS": "EMS",
    "P0": "P0",
    "P1": "P1",
    "P2": "P2",
    "P3": "P3",
}


class LineageEncoder:
    """
    Encoder for C. elegans lineage names.

    Supports multiple encoding schemes:
    1. Binary path: Encode division history as binary (a=0, p=1)
    2. Depth encoding: Number of divisions from P0
    3. Founder encoding: Which major lineage (AB, MS, E, C, D, P4)
    4. Hierarchical encoding: Full tree path as token sequence

    Example:
        >>> encoder = LineageEncoder()
        >>> encoder.parse_lineage("ABplpapppa")
        {'founder': 'AB', 'path': 'plpapppa', 'depth': 9, 'binary': [1, 0, 1, 0, 1, 1, 1, 0]}
    """

    def __init__(
        self,
        data_dir: str = "dataset/raw",
        max_depth: int = 20,
    ):
        """
        Initialize the lineage encoder.

        Args:
            data_dir: Base directory containing lineage data files.
            max_depth: Maximum lineage depth for padding.
        """
        self.data_dir = Path(data_dir)
        self.max_depth = max_depth

        # Cache for lineage tree
        self._lineage_tree: Optional[Dict] = None
        self._cell_timing: Optional[Dict] = None

    @property
    def lineage_tree(self) -> Dict:
        """Load and cache the lineage tree."""
        if self._lineage_tree is None:
            tree_path = self.data_dir / "wormbase" / "lineage_tree.json"
            if tree_path.exists():
                with open(tree_path, "r") as f:
                    self._lineage_tree = json.load(f)
                logger.info(f"Loaded lineage tree with {len(self._lineage_tree)} cells")
            else:
                logger.warning(f"Lineage tree not found: {tree_path}")
                self._lineage_tree = {}
        return self._lineage_tree

    @property
    def cell_timing(self) -> Dict:
        """Load and cache cell timing information."""
        if self._cell_timing is None:
            timing_path = self.data_dir / "wormbase" / "cell_timing.json"
            if timing_path.exists():
                with open(timing_path, "r") as f:
                    self._cell_timing = json.load(f)
                logger.info(f"Loaded timing for {len(self._cell_timing)} cells")
            else:
                logger.warning(f"Cell timing not found: {timing_path}")
                self._cell_timing = {}
        return self._cell_timing

    def is_valid_lineage(self, lineage: str) -> bool:
        """
        Check if a lineage string is valid (no ambiguity markers).

        Args:
            lineage: Lineage string to validate.

        Returns:
            True if lineage is clean (no 'x', '/', or 'unassigned').
        """
        if not lineage or lineage == "unassigned":
            return False

        # Check for ambiguity markers
        if "x" in lineage.lower():
            return False
        if "/" in lineage:
            return False
        if "?" in lineage:
            return False

        return True

    def parse_lineage(self, lineage: str) -> Optional[Dict]:
        """
        Parse a lineage string into its components.

        Args:
            lineage: Lineage string (e.g., "ABplpapppa").

        Returns:
            Dict with keys: founder, path, depth, binary, full_path
            Returns None if lineage is invalid.
        """
        if not self.is_valid_lineage(lineage):
            return None

        # Identify founder
        founder = None
        path = ""

        # Try to match known founders (longest first)
        for prefix in sorted(FOUNDER_PREFIXES.keys(), key=len, reverse=True):
            if lineage.startswith(prefix):
                founder = FOUNDER_PREFIXES[prefix]
                path = lineage[len(prefix) :]
                break

        if founder is None:
            # Try to identify founder from first characters
            if lineage[0] in ["A", "a"]:
                if len(lineage) > 1 and lineage[1] in ["B", "b"]:
                    founder = "AB"
                    path = lineage[2:]
            elif lineage[0] in ["M", "m"]:
                if len(lineage) > 1 and lineage[1] in ["S", "s"]:
                    founder = "MS"
                    path = lineage[2:]
            elif lineage[0] in ["E", "e"]:
                if len(lineage) > 1 and lineage[1] in ["M", "m"]:
                    founder = "EMS"
                    path = lineage[3:]
                else:
                    founder = "E"
                    path = lineage[1:]
            elif lineage[0] in ["C", "c"]:
                founder = "C"
                path = lineage[1:]
            elif lineage[0] in ["D", "d"]:
                founder = "D"
                path = lineage[1:]
            elif lineage[0] in ["P", "p"]:
                # Could be P0, P1, P2, P3, P4
                match = re.match(r"P(\d+)(.*)", lineage, re.IGNORECASE)
                if match:
                    p_num = match.group(1)
                    founder = f"P{p_num}"
                    path = match.group(2)
            elif lineage[0] in ["Z", "z"]:
                if len(lineage) > 1 and lineage[1] in ["2", "3"]:
                    founder = f"Z{lineage[1]}"
                    path = lineage[2:]

        if founder is None:
            logger.debug(f"Could not identify founder for lineage: {lineage}")
            return None

        # Parse path into binary (a=0, p=1)
        path_lower = path.lower()
        binary = []
        for char in path_lower:
            if char == "a":
                binary.append(0)
            elif char == "p":
                binary.append(1)
            elif char in ["l", "r", "d", "v"]:
                # Handle left/right/dorsal/ventral (less common)
                # Map to a/p for simplicity
                if char in ["l", "d"]:
                    binary.append(0)
                else:
                    binary.append(1)
            # Skip other characters (spaces, periods, etc.)

        # Calculate depth
        founder_depth = FOUNDER_CELLS.get(founder, {}).get("depth", 1)
        total_depth = founder_depth + len(binary)

        return {
            "founder": founder,
            "path": path,
            "depth": total_depth,
            "binary": binary,
            "full_path": lineage,
        }

    def encode_binary(
        self,
        lineage: str,
        pad_to: Optional[int] = None,
        pad_value: int = -1,
    ) -> np.ndarray:
        """
        Encode lineage as binary path array.

        Args:
            lineage: Lineage string.
            pad_to: Pad/truncate to this length. If None, uses max_depth.
            pad_value: Value to use for padding.

        Returns:
            1D numpy array of binary path (0/1) with padding.
        """
        if pad_to is None:
            pad_to = self.max_depth

        parsed = self.parse_lineage(lineage)
        if parsed is None:
            return np.full(pad_to, pad_value, dtype=np.int8)

        binary = parsed["binary"]

        # Pad or truncate
        if len(binary) >= pad_to:
            return np.array(binary[:pad_to], dtype=np.int8)
        else:
            padded = np.full(pad_to, pad_value, dtype=np.int8)
            padded[: len(binary)] = binary
            return padded

    def encode_batch(
        self,
        lineages: List[str],
        pad_to: Optional[int] = None,
    ) -> np.ndarray:
        """
        Encode a batch of lineages to binary arrays.

        Args:
            lineages: List of lineage strings.
            pad_to: Pad length (uses max_depth if None).

        Returns:
            2D numpy array of shape (n_cells, pad_to).
        """
        return np.stack([self.encode_binary(lin, pad_to) for lin in lineages])

    def encode_founder_onehot(
        self,
        lineage: str,
        founders: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Encode founder lineage as one-hot vector.

        Args:
            lineage: Lineage string.
            founders: List of founder names for encoding order.
                     Defaults to ['AB', 'MS', 'E', 'C', 'D', 'P4', 'other'].

        Returns:
            One-hot encoded founder vector.
        """
        if founders is None:
            founders = ["AB", "MS", "E", "C", "D", "P4", "other"]

        parsed = self.parse_lineage(lineage)
        if parsed is None:
            # Return zeros or 'other'
            onehot = np.zeros(len(founders), dtype=np.float32)
            if "other" in founders:
                onehot[founders.index("other")] = 1.0
            return onehot

        founder = parsed["founder"]

        # Map to canonical founders
        if founder in ["P0", "P1", "P2", "P3"]:
            founder = "other"
        if founder in ["Z2", "Z3"]:
            founder = "P4"  # Germline
        if founder == "EMS":
            founder = "other"  # Or could split

        onehot = np.zeros(len(founders), dtype=np.float32)
        if founder in founders:
            onehot[founders.index(founder)] = 1.0
        elif "other" in founders:
            onehot[founders.index("other")] = 1.0

        return onehot

    def get_depth(self, lineage: str) -> int:
        """
        Get the depth (number of divisions from P0) for a lineage.

        Args:
            lineage: Lineage string.

        Returns:
            Depth value, or -1 if invalid.
        """
        parsed = self.parse_lineage(lineage)
        if parsed is None:
            return -1
        return parsed["depth"]

    def get_founder(self, lineage: str) -> str:
        """
        Get the founder lineage for a cell.

        Args:
            lineage: Lineage string.

        Returns:
            Founder name (e.g., 'AB', 'MS') or 'unknown'.
        """
        parsed = self.parse_lineage(lineage)
        if parsed is None:
            return "unknown"
        return parsed["founder"]

    def encode_for_dataframe(
        self,
        lineages: pd.Series,
    ) -> pd.DataFrame:
        """
        Encode lineages and return as a DataFrame with multiple columns.

        Args:
            lineages: Series of lineage strings.

        Returns:
            DataFrame with columns:
            - lineage_valid: bool
            - lineage_founder: str
            - lineage_depth: int
            - lineage_binary: str (e.g., "01010110")
        """
        results = {
            "lineage_valid": [],
            "lineage_founder": [],
            "lineage_depth": [],
            "lineage_binary": [],
        }

        for lin in lineages:
            parsed = self.parse_lineage(lin)
            if parsed is None:
                results["lineage_valid"].append(False)
                results["lineage_founder"].append("unknown")
                results["lineage_depth"].append(-1)
                results["lineage_binary"].append("")
            else:
                results["lineage_valid"].append(True)
                results["lineage_founder"].append(parsed["founder"])
                results["lineage_depth"].append(parsed["depth"])
                results["lineage_binary"].append(
                    "".join(str(b) for b in parsed["binary"])
                )

        return pd.DataFrame(results)

    def get_parent(self, lineage: str) -> Optional[str]:
        """
        Get the parent cell name for a given lineage.

        Args:
            lineage: Lineage string.

        Returns:
            Parent lineage name, or None if at root.
        """
        # First check lineage tree
        if lineage in self.lineage_tree:
            parent = self.lineage_tree[lineage].get("parent")
            if parent:
                return parent

        # Otherwise, parse and remove last division
        parsed = self.parse_lineage(lineage)
        if parsed is None or not parsed["path"]:
            return None

        # Remove last character from path
        founder = parsed["founder"]
        path = parsed["path"]

        if len(path) > 0:
            return founder + path[:-1]
        else:
            # Return founder's parent
            if founder in self.lineage_tree:
                return self.lineage_tree[founder].get("parent")
            return None

    def get_children(self, lineage: str) -> List[str]:
        """
        Get the two daughter cells for a given lineage.

        Args:
            lineage: Lineage string.

        Returns:
            List of two child lineage names.
        """
        if lineage in self.lineage_tree:
            children = self.lineage_tree[lineage].get("children", [])
            if children:
                return children

        # Otherwise, append 'a' and 'p'
        return [lineage + "a", lineage + "p"]

    def build_adjacency_matrix(
        self,
        lineages: List[str],
        include_parent: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Build an adjacency matrix for lineage relationships.

        Args:
            lineages: List of lineage names to include.
            include_parent: If True, edge from child to parent.

        Returns:
            Tuple of (adjacency_matrix, name_to_idx mapping).
        """
        n = len(lineages)
        name_to_idx = {name: i for i, name in enumerate(lineages)}
        adj = np.zeros((n, n), dtype=np.float32)

        for name in lineages:
            idx = name_to_idx[name]

            # Add parent edge
            if include_parent:
                parent = self.get_parent(name)
                if parent and parent in name_to_idx:
                    parent_idx = name_to_idx[parent]
                    adj[idx, parent_idx] = 1.0
                    adj[parent_idx, idx] = 1.0  # Undirected

        return adj, name_to_idx
