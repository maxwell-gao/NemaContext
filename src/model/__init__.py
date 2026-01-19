"""
Model module for C. elegans multi-modal analysis.

This module provides components for:
- Spatial data parsing and 4D position tracking
- Spatial neighborhood graph construction
- Contact graph neural networks (CShaper-based)
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
from .contact_gnn import (
    ContactGraph,
    TimeDynamicContactGraph,
    ContactMessagePassing,
    ContactGNN,
    compare_contact_vs_knn,
    build_contact_graph_from_cshaper,
)

__all__ = [
    # Spatial
    "CellPosition",
    "Timepoint",
    "SpatialDataParser",
    "compute_cell_distances",
    "compute_cell_neighbors",
    # Spatial Graph
    "GraphNode",
    "GraphEdge",
    "SpatialGraph",
    "SpatialGraphBuilder",
    "compute_neighborhood_features",
    "compute_lineage_neighborhood_overlap",
    "graph_to_pyg_data",
    # Contact Graph GNN
    "ContactGraph",
    "TimeDynamicContactGraph",
    "ContactMessagePassing",
    "ContactGNN",
    "compare_contact_vs_knn",
    "build_contact_graph_from_cshaper",
    # Multimodal
    "LineageNode",
    "LineageTree",
    "CellData",
    "MultimodalDataset",
    "compute_multimodal_similarity",
]
