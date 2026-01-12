"""
Model module for C. elegans multi-modal analysis.

This module provides components for:
- Spatial data parsing and 4D position tracking
- Spatial neighborhood graph construction
- Multi-modal data integration (transcriptome, spatial, lineage)
- Graph-based embeddings for cell representation
"""

from .multimodal import (
    CellData,
    LineageNode,
    LineageTree,
    MultimodalDataset,
    compute_multimodal_similarity,
)
from .spatial import (
    CellPosition,
    SpatialDataParser,
    Timepoint,
    compute_cell_distances,
    compute_cell_neighbors,
)
from .spatial_graph import (
    GraphEdge,
    GraphNode,
    SpatialGraph,
    SpatialGraphBuilder,
    compute_lineage_neighborhood_overlap,
    compute_neighborhood_features,
    graph_to_pyg_data,
)

__all__ = [
    # Spatial
    "CellPosition",
    "Timepoint",
    "SpatialDataParser",
    "compute_cell_distances",
    "compute_cell_neighbors",
    # Graph
    "GraphNode",
    "GraphEdge",
    "SpatialGraph",
    "SpatialGraphBuilder",
    "compute_neighborhood_features",
    "compute_lineage_neighborhood_overlap",
    "graph_to_pyg_data",
    # Multimodal
    "LineageNode",
    "LineageTree",
    "CellData",
    "MultimodalDataset",
    "compute_multimodal_similarity",
]
