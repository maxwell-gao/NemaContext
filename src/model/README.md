# Model Module

Multi-modal analysis components for C. elegans single-cell data integration.

## Overview

This module provides tools for integrating and analyzing C. elegans single-cell data across three modalities:

1. **Spatial** - 4D cell positions during embryonic development
2. **Lineage** - Cell lineage tree and relationships
3. **Transcriptome** - Single-cell RNA-seq gene expression

The key insight is that **spatial context matters more than absolute position**. Instead of using raw XYZ coordinates, we build neighborhood graphs that capture what cells are nearby, enabling graph neural network-based embeddings.

## Modules

### `spatial.py` - Spatial Data Parser

Parses WormGUIDES nuclei 4D position data (360 timepoints covering embryonic development).

```python
from src.model import SpatialDataParser, compute_cell_distances

# Initialize parser
parser = SpatialDataParser("dataset/raw/wormguides/nuclei_files")

# Get all cells at timepoint 200
tp = parser.parse_timepoint(200)
print(f"Cells: {len(tp.named_cells)}")

# Get trajectory for a specific cell
trajectory = parser.get_cell_trajectory("ABpla")

# Build 4D tensor (timepoints x cells x xyz)
tensor, timepoints, cell_names = parser.build_4d_tensor()
```

**Key Classes:**
- `CellPosition` - Single cell position at a timepoint
- `Timepoint` - All cells at a single timepoint
- `SpatialDataParser` - Main parser for nuclei files

### `spatial_graph.py` - Spatial Neighborhood Graphs

Builds graphs where nodes are cells and edges represent spatial proximity.

```python
from src.model import SpatialDataParser, SpatialGraphBuilder, graph_to_pyg_data

parser = SpatialDataParser("dataset/raw/wormguides/nuclei_files")
builder = SpatialGraphBuilder(parser)

# Build k-nearest neighbor graph
graph = builder.build_knn_graph(timepoint=200, k=10)
print(f"Nodes: {graph.num_nodes}, Edges: {graph.num_edges}")

# Get neighbors for a cell
neighbors = graph.get_neighbors("ABpla")

# Convert to PyTorch Geometric format
pyg_data = graph_to_pyg_data(graph)
# Returns: {'x': node_features, 'edge_index': edges, 'edge_attr': edge_features}
```

**Graph Construction Methods:**

| Method | Description | Use Case |
|--------|-------------|----------|
| `build_knn_graph(k=10)` | Connect each cell to k nearest neighbors | General purpose |
| `build_radius_graph(r=50)` | Connect cells within distance r | Dense local connections |
| `build_adaptive_graph()` | Consider cell diameter for connections | Biologically informed |

**Key Classes:**
- `GraphNode` - Node in the spatial graph
- `GraphEdge` - Edge with distance and relative position
- `SpatialGraph` - Graph data structure
- `SpatialGraphBuilder` - Factory for building graphs

### `multimodal.py` - Multi-modal Integration

Integrates transcriptome, spatial, and lineage data into a unified dataset.

```python
from src.model import MultimodalDataset, LineageTree

# Create dataset
dataset = MultimodalDataset("dataset/raw")

# Load each modality
dataset.load_packer_data()      # Transcriptome (Packer et al. 2019)
dataset.load_spatial_data()      # Spatial (WormGUIDES)
dataset.load_lineage_data()      # Lineage tree

# Get cells with all modalities
complete_cells = dataset.get_cells_with_all_modalities()

# Build feature matrices
positions, names = dataset.build_position_matrix()
lineage_dists, names = dataset.build_lineage_distance_matrix()

# Get summary
print(dataset.get_summary())
```

**Key Classes:**
- `LineageNode` - Node in the lineage tree
- `LineageTree` - Cell lineage tree with distance computation
- `CellData` - Integrated data for a single cell
- `MultimodalDataset` - Main dataset class

## Data Flow

```
WormGUIDES nuclei files     Packer 2019 scRNA-seq     Cell lineage
        │                           │                      │
        ▼                           ▼                      ▼
  SpatialDataParser          load_packer_data()      LineageTree
        │                           │                      │
        ▼                           │                      │
  SpatialGraphBuilder               │                      │
        │                           │                      │
        ▼                           ▼                      ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                    MultimodalDataset                         │
  │  - Position matrix (N x 3)                                   │
  │  - Neighborhood graph (edge_index, edge_attr)                │
  │  - Lineage distance matrix (N x N)                           │
  │  - Expression matrix (N x G) [optional]                      │
  └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    Graph Neural Network
                    (e.g., PyTorch Geometric)
```

## Graph Format (PyTorch Geometric Compatible)

The `graph_to_pyg_data()` function returns data in PyG format:

```python
{
    'x': np.ndarray,          # Node features (N x F)
    'edge_index': np.ndarray, # Edge indices (2 x E) in COO format
    'edge_attr': np.ndarray,  # Edge attributes (E x 5: dist, weight, dx, dy, dz)
    'pos': np.ndarray,        # Node positions (N x 3)
    'node_names': List[str],  # Cell names
}
```

## Spatial-Lineage Analysis

Analyze the relationship between spatial neighborhoods and lineage:

```python
from src.model import (
    SpatialGraphBuilder,
    SpatialDataParser,
    LineageTree,
    compute_lineage_neighborhood_overlap,
)

# Build graph and lineage tree
parser = SpatialDataParser("dataset/raw/wormguides/nuclei_files")
builder = SpatialGraphBuilder(parser)
graph = builder.build_knn_graph(timepoint=200, k=10)

lineage_tree = LineageTree()
lineage_tree.build_from_cell_names(graph.node_names)

# Compute overlap metrics
metrics = compute_lineage_neighborhood_overlap(graph, lineage_tree)

# Each cell gets:
# - sibling_fraction: % of spatial neighbors that are lineage siblings
# - cousin_fraction: % of spatial neighbors that are lineage cousins
# - mean_lineage_distance: average lineage distance to spatial neighbors
```

## Temporal Graph Sequence

Build graphs across development time:

```python
# Build graphs at multiple timepoints
graphs = builder.build_temporal_graph_sequence(
    timepoints=[50, 100, 150, 200, 250, 300, 350],
    k=10,
    method="knn"
)

for tp, graph in graphs.items():
    print(f"t={tp}: {graph.num_nodes} cells, {graph.num_edges} edges")
```

## Example: Multi-modal Embedding

See `examples/multimodal_embedding.py` for a complete example that:

1. Loads spatial graph at a timepoint
2. Computes lineage-based features
3. Runs a simple GNN to produce embeddings
4. Analyzes correlation between embedding distance, spatial distance, and lineage distance

## API Reference

### Spatial Module

| Function | Description |
|----------|-------------|
| `compute_cell_distances(positions)` | Compute NxN distance matrix |
| `compute_cell_neighbors(positions, names, k)` | Find k-nearest neighbors |

### Graph Module

| Function | Description |
|----------|-------------|
| `compute_neighborhood_features(graph)` | Aggregate neighbor features (10-dim) |
| `compute_lineage_neighborhood_overlap(graph, tree)` | Spatial-lineage overlap metrics |
| `graph_to_pyg_data(graph)` | Convert to PyTorch Geometric format |

### Multimodal Module

| Function | Description |
|----------|-------------|
| `compute_multimodal_similarity(spatial, lineage, expr)` | Combined similarity matrix |

## Dependencies

- `numpy` - Array operations
- `pandas` - Data loading (for Packer data)
- `torch` / `torch-geometric` - Optional, for GNN training

## References

- **WormGUIDES**: Bao Lab - 4D cell tracking in C. elegans embryo
- **Packer et al. 2019**: Single-cell RNA-seq of C. elegans embryogenesis
- **Sulston et al. 1983**: Complete cell lineage of C. elegans