"""
Multimodal data integration module for C. elegans single-cell analysis.

This module provides utilities to integrate and align data from multiple modalities:
- Transcriptome: Single-cell RNA-seq data (Packer et al. 2019)
- Spatial: 4D cell positions (WormGUIDES nuclei data)
- Lineage: Cell lineage tree and relationships

The integrated data can be used for multi-modal embedding and analysis.
"""

import gzip
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from .spatial import CellPosition, SpatialDataParser


@dataclass
class LineageNode:
    """A node in the cell lineage tree."""

    name: str
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    birth_time: Optional[float] = None
    division_time: Optional[float] = None
    fate: Optional[str] = None
    description: Optional[str] = None

    @property
    def is_terminal(self) -> bool:
        """Return True if this is a terminal cell (no children)."""
        return len(self.children) == 0

    @property
    def depth(self) -> int:
        """Return the depth in lineage (number of characters in name for AB lineage)."""
        # Simple heuristic based on cell naming convention
        if self.name in ("P0", "AB", "P1", "EMS", "P2", "P3", "P4"):
            return {"P0": 0, "AB": 1, "P1": 1, "EMS": 2, "P2": 2, "P3": 3, "P4": 4}.get(
                self.name, 0
            )
        elif self.name.startswith("AB"):
            return len(self.name)
        elif self.name.startswith(("MS", "E", "C", "D")):
            return len(self.name) + 1
        else:
            return 0


class LineageTree:
    """
    C. elegans cell lineage tree.

    Provides methods to traverse and query the lineage relationships.
    """

    def __init__(self):
        self.nodes: Dict[str, LineageNode] = {}
        self._build_founder_cells()

    def _build_founder_cells(self):
        """Build the founder cell lineage (P0 through early divisions)."""
        founders = {
            "P0": {"children": ["AB", "P1"], "parent": None},
            "AB": {"children": ["ABa", "ABp"], "parent": "P0"},
            "P1": {"children": ["EMS", "P2"], "parent": "P0"},
            "EMS": {"children": ["MS", "E"], "parent": "P1"},
            "P2": {"children": ["C", "P3"], "parent": "P1"},
            "MS": {"children": ["MSa", "MSp"], "parent": "EMS"},
            "E": {"children": ["Ea", "Ep"], "parent": "EMS"},
            "C": {"children": ["Ca", "Cp"], "parent": "P2"},
            "P3": {"children": ["D", "P4"], "parent": "P2"},
            "D": {"children": ["Da", "Dp"], "parent": "P3"},
            "P4": {"children": ["Z2", "Z3"], "parent": "P3"},
        }

        for name, info in founders.items():
            self.nodes[name] = LineageNode(
                name=name,
                parent=info["parent"],
                children=info["children"],
            )

    def add_node(
        self,
        name: str,
        parent: Optional[str] = None,
        children: Optional[List[str]] = None,
        **kwargs,
    ) -> LineageNode:
        """Add a node to the lineage tree."""
        node = LineageNode(
            name=name,
            parent=parent,
            children=children or [],
            **kwargs,
        )
        self.nodes[name] = node
        return node

    def get_node(self, name: str) -> Optional[LineageNode]:
        """Get a node by name."""
        return self.nodes.get(name)

    def get_ancestors(self, name: str) -> List[str]:
        """Get all ancestors of a cell (from immediate parent to P0)."""
        ancestors = []
        current = name
        while current in self.nodes:
            parent = self.nodes[current].parent
            if parent is None:
                break
            ancestors.append(parent)
            current = parent
        return ancestors

    def get_descendants(self, name: str) -> List[str]:
        """Get all descendants of a cell."""
        descendants = []
        if name not in self.nodes:
            return descendants

        queue = list(self.nodes[name].children)
        while queue:
            child = queue.pop(0)
            descendants.append(child)
            if child in self.nodes:
                queue.extend(self.nodes[child].children)

        return descendants

    def get_lineage_distance(self, cell1: str, cell2: str) -> int:
        """
        Compute the lineage distance between two cells.

        Distance = number of edges in the tree between the two cells.
        """
        if cell1 == cell2:
            return 0

        # Get ancestors for both cells
        ancestors1 = set(self.get_ancestors(cell1))
        ancestors1.add(cell1)
        ancestors2 = set(self.get_ancestors(cell2))
        ancestors2.add(cell2)

        # Find common ancestors
        common = ancestors1 & ancestors2
        if not common:
            return -1  # No common ancestor found

        # Find the nearest common ancestor
        # It's the one with the maximum depth
        nca = max(common, key=lambda x: self.nodes[x].depth if x in self.nodes else 0)

        # Distance = steps from cell1 to NCA + steps from cell2 to NCA
        dist1 = 0
        current = cell1
        while current != nca and current in self.nodes:
            current = self.nodes[current].parent
            dist1 += 1
            if current is None:
                break

        dist2 = 0
        current = cell2
        while current != nca and current in self.nodes:
            current = self.nodes[current].parent
            dist2 += 1
            if current is None:
                break

        return dist1 + dist2

    def infer_lineage_from_name(self, name: str) -> List[str]:
        """
        Infer the lineage path from cell name.

        C. elegans cell names encode their lineage. For example:
        - ABpla -> AB -> ABp -> ABpl -> ABpla
        """
        lineage = []

        if name.startswith("AB"):
            lineage.append("AB")
            suffix = name[2:]
            for i in range(len(suffix)):
                lineage.append("AB" + suffix[: i + 1])
        elif name.startswith("MS"):
            lineage.extend(["EMS", "MS"])
            suffix = name[2:]
            for i in range(len(suffix)):
                lineage.append("MS" + suffix[: i + 1])
        elif name.startswith("E") and len(name) <= 4:
            lineage.extend(["EMS", "E"])
            suffix = name[1:]
            for i in range(len(suffix)):
                lineage.append("E" + suffix[: i + 1])
        elif name.startswith("C"):
            lineage.extend(["P2", "C"])
            suffix = name[1:]
            for i in range(len(suffix)):
                lineage.append("C" + suffix[: i + 1])
        elif name.startswith("D"):
            lineage.extend(["P3", "D"])
            suffix = name[1:]
            for i in range(len(suffix)):
                lineage.append("D" + suffix[: i + 1])

        return lineage

    def build_from_cell_names(self, cell_names: List[str]) -> None:
        """
        Build lineage tree from a list of cell names.

        Uses the C. elegans naming convention to infer lineage relationships.
        """
        for name in cell_names:
            if name in self.nodes:
                continue

            lineage = self.infer_lineage_from_name(name)
            if not lineage:
                continue

            # Ensure all ancestors exist
            for i, ancestor in enumerate(lineage):
                if ancestor not in self.nodes:
                    parent = lineage[i - 1] if i > 0 else None
                    self.add_node(ancestor, parent=parent)

            # Add the cell itself if not already present
            if name not in self.nodes:
                parent = lineage[-1] if lineage else None
                self.add_node(name, parent=parent)

            # Update parent's children list
            if name in self.nodes and self.nodes[name].parent:
                parent_name = self.nodes[name].parent
                if parent_name in self.nodes:
                    if name not in self.nodes[parent_name].children:
                        self.nodes[parent_name].children.append(name)

    def compute_lineage_distance_matrix(
        self, cell_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute pairwise lineage distances for a list of cells.

        Args:
            cell_names: List of cell names

        Returns:
            Tuple of (NxN distance matrix, list of cell names)
        """
        N = len(cell_names)
        distances = np.zeros((N, N), dtype=np.int32)

        for i in range(N):
            for j in range(i + 1, N):
                dist = self.get_lineage_distance(cell_names[i], cell_names[j])
                distances[i, j] = dist
                distances[j, i] = dist

        return distances, cell_names

    def to_dict(self) -> Dict[str, Any]:
        """Export lineage tree as dictionary."""
        return {
            name: {
                "parent": node.parent,
                "children": node.children,
                "birth_time": node.birth_time,
                "division_time": node.division_time,
                "fate": node.fate,
                "description": node.description,
            }
            for name, node in self.nodes.items()
        }

    def save_json(self, path: str) -> None:
        """Save lineage tree to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "LineageTree":
        """Load lineage tree from JSON file."""
        tree = cls()
        with open(path, "r") as f:
            data = json.load(f)

        for name, info in data.items():
            tree.nodes[name] = LineageNode(
                name=name,
                parent=info.get("parent"),
                children=info.get("children", []),
                birth_time=info.get("birth_time"),
                division_time=info.get("division_time"),
                fate=info.get("fate"),
                description=info.get("description"),
            )

        return tree


@dataclass
class CellData:
    """Integrated data for a single cell."""

    name: str
    lineage_name: Optional[str] = None  # Systematic lineage name
    cell_type: Optional[str] = None
    tissue: Optional[str] = None

    # Spatial data
    position: Optional[np.ndarray] = None  # XYZ coordinates
    timepoint: Optional[int] = None

    # Transcriptome data
    gene_expression: Optional[np.ndarray] = None  # Gene expression vector
    gene_names: Optional[List[str]] = None

    # Lineage data
    lineage_depth: Optional[int] = None
    parent: Optional[str] = None
    birth_time: Optional[float] = None

    # Embryo time
    embryo_time: Optional[float] = None  # Minutes post-fertilization


class MultimodalDataset:
    """
    Integrated multimodal dataset for C. elegans.

    Combines transcriptome, spatial, and lineage data into a unified structure
    for multi-modal embedding and analysis.
    """

    def __init__(self, data_dir: str = "dataset/raw"):
        self.data_dir = Path(data_dir)
        self.cells: Dict[str, CellData] = {}
        self.gene_names: List[str] = []
        self.lineage_tree: Optional[LineageTree] = None
        self.spatial_parser: Optional[SpatialDataParser] = None

        # Data matrices (computed on demand)
        self._expression_matrix: Optional[np.ndarray] = None
        self._position_matrix: Optional[np.ndarray] = None
        self._lineage_distance_matrix: Optional[np.ndarray] = None

    def load_packer_data(self, sample_frac: Optional[float] = None) -> None:
        """
        Load Packer et al. 2019 single-cell transcriptome data.

        Args:
            sample_frac: Optional fraction of cells to sample (for memory efficiency)
        """
        packer_dir = self.data_dir / "packer2019"

        # Load cell annotations
        annot_path = packer_dir / "GSE126954_cell_annotation.csv.gz"
        if not annot_path.exists():
            raise FileNotFoundError(f"Cell annotation file not found: {annot_path}")

        print(f"Loading cell annotations from {annot_path}...")
        cell_annot = pd.read_csv(annot_path, compression="gzip")

        if sample_frac is not None:
            cell_annot = cell_annot.sample(frac=sample_frac, random_state=42)

        # Extract cell information
        for _, row in cell_annot.iterrows():
            cell_name = str(row.get("cell", row.name))

            # Get lineage name if available
            lineage_name = row.get("lineage", None)
            if pd.isna(lineage_name):
                lineage_name = None

            # Get cell type if available
            cell_type = row.get("cell.type", row.get("cell_type", None))
            if pd.isna(cell_type):
                cell_type = None

            # Get tissue if available
            tissue = row.get("tissue", None)
            if pd.isna(tissue):
                tissue = None

            # Get embryo time if available
            embryo_time = row.get("embryo.time", row.get("embryo_time", None))
            if pd.notna(embryo_time):
                embryo_time = float(embryo_time)
            else:
                embryo_time = None

            self.cells[cell_name] = CellData(
                name=cell_name,
                lineage_name=str(lineage_name) if lineage_name else None,
                cell_type=str(cell_type) if cell_type else None,
                tissue=str(tissue) if tissue else None,
                embryo_time=embryo_time,
            )

        print(f"Loaded annotations for {len(self.cells)} cells")

        # Load gene annotations
        gene_path = packer_dir / "GSE126954_gene_annotation.csv.gz"
        if gene_path.exists():
            print(f"Loading gene annotations from {gene_path}...")
            gene_annot = pd.read_csv(gene_path, compression="gzip")
            if "gene" in gene_annot.columns:
                self.gene_names = gene_annot["gene"].tolist()
            elif "gene_name" in gene_annot.columns:
                self.gene_names = gene_annot["gene_name"].tolist()
            print(f"Loaded {len(self.gene_names)} genes")

    def load_spatial_data(
        self,
        timepoints: Optional[List[int]] = None,
        match_by_lineage: bool = True,
    ) -> None:
        """
        Load WormGUIDES spatial data.

        Args:
            timepoints: Optional list of timepoints to load.
            match_by_lineage: If True, match cells by lineage name.
        """
        nuclei_dir = self.data_dir / "wormguides" / "nuclei_files"
        if not nuclei_dir.exists():
            raise FileNotFoundError(f"Nuclei directory not found: {nuclei_dir}")

        self.spatial_parser = SpatialDataParser(str(nuclei_dir))

        if timepoints is None:
            timepoints = self.spatial_parser.get_available_timepoints()

        print(f"Loading spatial data from {len(timepoints)} timepoints...")

        # Get all cell names with spatial data
        spatial_cells = set(self.spatial_parser.get_all_cell_names(timepoints))
        print(f"Found {len(spatial_cells)} cells with spatial data")

        if match_by_lineage:
            # Match cells by their lineage names
            matched = 0
            for cell_name, cell_data in self.cells.items():
                if cell_data.lineage_name and cell_data.lineage_name in spatial_cells:
                    # Get position at the most relevant timepoint
                    trajectory = self.spatial_parser.get_cell_trajectory(
                        cell_data.lineage_name, timepoints
                    )
                    if trajectory:
                        # Use the first available position
                        pos = trajectory[0]
                        cell_data.position = pos.position
                        cell_data.timepoint = pos.timepoint
                        matched += 1

            print(f"Matched {matched} cells to spatial positions")

    def load_lineage_data(self) -> None:
        """Load and build lineage tree."""
        self.lineage_tree = LineageTree()

        # Get all lineage names from cells
        lineage_names = []
        for cell_data in self.cells.values():
            if cell_data.lineage_name:
                lineage_names.append(cell_data.lineage_name)

        # Build tree from cell names
        self.lineage_tree.build_from_cell_names(lineage_names)

        # Update cell data with lineage information
        for cell_name, cell_data in self.cells.items():
            if cell_data.lineage_name:
                node = self.lineage_tree.get_node(cell_data.lineage_name)
                if node:
                    cell_data.lineage_depth = node.depth
                    cell_data.parent = node.parent

        print(f"Built lineage tree with {len(self.lineage_tree.nodes)} nodes")

    def get_cells_with_all_modalities(self) -> List[str]:
        """Get cell names that have data in all modalities."""
        complete_cells = []
        for name, cell in self.cells.items():
            has_spatial = cell.position is not None
            has_lineage = cell.lineage_name is not None
            # has_expression = cell.gene_expression is not None

            if has_spatial and has_lineage:
                complete_cells.append(name)

        return complete_cells

    def build_expression_matrix(
        self, cell_names: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Build expression matrix for specified cells.

        Args:
            cell_names: List of cell names. If None, uses all cells.

        Returns:
            Tuple of (NxG expression matrix, cell names, gene names)
        """
        if cell_names is None:
            cell_names = list(self.cells.keys())

        # Filter to cells with expression data
        valid_cells = [
            n for n in cell_names if self.cells[n].gene_expression is not None
        ]

        if not valid_cells:
            return np.zeros((0, 0)), [], self.gene_names

        matrix = np.stack([self.cells[n].gene_expression for n in valid_cells])
        return matrix, valid_cells, self.gene_names

    def build_position_matrix(
        self, cell_names: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Build position matrix for specified cells.

        Args:
            cell_names: List of cell names. If None, uses all cells with positions.

        Returns:
            Tuple of (Nx3 position matrix, cell names)
        """
        if cell_names is None:
            cell_names = [
                n for n in self.cells.keys() if self.cells[n].position is not None
            ]
        else:
            cell_names = [n for n in cell_names if self.cells[n].position is not None]

        if not cell_names:
            return np.zeros((0, 3)), []

        matrix = np.stack([self.cells[n].position for n in cell_names])
        return matrix, cell_names

    def build_lineage_distance_matrix(
        self, cell_names: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Build pairwise lineage distance matrix.

        Args:
            cell_names: List of cell names. If None, uses all cells with lineage.

        Returns:
            Tuple of (NxN distance matrix, cell names)
        """
        if self.lineage_tree is None:
            raise ValueError("Lineage tree not loaded. Call load_lineage_data() first.")

        if cell_names is None:
            cell_names = [
                n for n in self.cells.keys() if self.cells[n].lineage_name is not None
            ]

        # Use lineage names for distance computation
        lineage_names = [self.cells[n].lineage_name for n in cell_names]
        lineage_names = [n for n in lineage_names if n is not None]

        return self.lineage_tree.compute_lineage_distance_matrix(lineage_names)

    def build_multimodal_features(
        self,
        cell_names: Optional[List[str]] = None,
        include_expression: bool = True,
        include_spatial: bool = True,
        include_lineage: bool = True,
        normalize: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Build feature matrices for multi-modal embedding.

        Args:
            cell_names: List of cell names.
            include_expression: Include gene expression features.
            include_spatial: Include spatial position features.
            include_lineage: Include lineage distance features.
            normalize: Normalize features.

        Returns:
            Dictionary with keys 'expression', 'spatial', 'lineage', 'cell_names'
        """
        if cell_names is None:
            cell_names = self.get_cells_with_all_modalities()

        result = {"cell_names": cell_names}

        if include_spatial:
            positions, valid_cells = self.build_position_matrix(cell_names)
            if normalize and positions.size > 0:
                positions = (positions - positions.mean(axis=0)) / (
                    positions.std(axis=0) + 1e-8
                )
            result["spatial"] = positions
            result["spatial_cells"] = valid_cells

        if include_lineage and self.lineage_tree is not None:
            distances, lineage_cells = self.build_lineage_distance_matrix(cell_names)
            if normalize and distances.size > 0:
                max_dist = distances.max()
                if max_dist > 0:
                    distances = distances / max_dist
            result["lineage"] = distances
            result["lineage_cells"] = lineage_cells

        if include_expression:
            expression, expr_cells, genes = self.build_expression_matrix(cell_names)
            if normalize and expression.size > 0:
                # Log-normalize expression
                expression = np.log1p(expression)
                expression = (expression - expression.mean(axis=0)) / (
                    expression.std(axis=0) + 1e-8
                )
            result["expression"] = expression
            result["expression_cells"] = expr_cells
            result["genes"] = genes

        return result

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the dataset."""
        n_cells = len(self.cells)
        n_with_spatial = sum(1 for c in self.cells.values() if c.position is not None)
        n_with_lineage = sum(
            1 for c in self.cells.values() if c.lineage_name is not None
        )
        n_with_expression = sum(
            1 for c in self.cells.values() if c.gene_expression is not None
        )
        n_complete = len(self.get_cells_with_all_modalities())

        return {
            "total_cells": n_cells,
            "cells_with_spatial": n_with_spatial,
            "cells_with_lineage": n_with_lineage,
            "cells_with_expression": n_with_expression,
            "cells_with_all_modalities": n_complete,
            "n_genes": len(self.gene_names),
            "n_lineage_nodes": (
                len(self.lineage_tree.nodes) if self.lineage_tree else 0
            ),
        }

    def __repr__(self) -> str:
        summary = self.get_summary()
        return (
            f"MultimodalDataset(\n"
            f"  cells={summary['total_cells']},\n"
            f"  spatial={summary['cells_with_spatial']},\n"
            f"  lineage={summary['cells_with_lineage']},\n"
            f"  expression={summary['cells_with_expression']},\n"
            f"  complete={summary['cells_with_all_modalities']}\n"
            f")"
        )


def compute_multimodal_similarity(
    spatial_dist: np.ndarray,
    lineage_dist: np.ndarray,
    expression_corr: Optional[np.ndarray] = None,
    weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """
    Compute combined similarity from multiple modalities.

    Args:
        spatial_dist: NxN spatial distance matrix
        lineage_dist: NxN lineage distance matrix
        expression_corr: Optional NxN expression correlation matrix
        weights: Tuple of (spatial_weight, lineage_weight, expression_weight)

    Returns:
        NxN combined similarity matrix
    """
    w_spatial, w_lineage, w_expr = weights

    # Normalize distances to [0, 1]
    if spatial_dist.max() > 0:
        spatial_sim = 1 - spatial_dist / spatial_dist.max()
    else:
        spatial_sim = np.ones_like(spatial_dist)

    if lineage_dist.max() > 0:
        lineage_sim = 1 - lineage_dist / lineage_dist.max()
    else:
        lineage_sim = np.ones_like(lineage_dist)

    # Combine modalities
    total_weight = w_spatial + w_lineage
    similarity = (w_spatial * spatial_sim + w_lineage * lineage_sim) / total_weight

    if expression_corr is not None:
        # Expression correlation should already be in [-1, 1], scale to [0, 1]
        expr_sim = (expression_corr + 1) / 2
        total_weight += w_expr
        similarity = (
            similarity * (w_spatial + w_lineage) + w_expr * expr_sim
        ) / total_weight

    return similarity
