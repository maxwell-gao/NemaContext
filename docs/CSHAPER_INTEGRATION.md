# CShaper Data Integration Guide

This document describes how to integrate CShaper 4D morphological atlas data into the NemaContext trimodal framework.

---

## Table of Contents

1. [Overview](#overview)
2. [Current Architecture](#current-architecture)
3. [CShaper Data Resources](#cshaper-data-resources)
4. [Integration Design](#integration-design)
5. [Implementation Details](#implementation-details)
6. [Lineage Name Matching](#lineage-name-matching)
7. [Time Alignment](#time-alignment)
8. [Output Schema](#output-schema)
9. [Implementation Status](#implementation-status)
10. [Model Impact](#model-impact)
11. [Validation Plan](#validation-plan)

---

## Overview

CShaper (Cao et al. 2020) provides a 4D morphological atlas of C. elegans embryogenesis, including:
- **Cell-cell contact matrices**: True physical contact surface areas between cells
- **Cell morphology**: Volume, surface area, and derived sphericity
- **Standardized spatial coordinates**: Averaged 3D positions from 46 embryos

This data enhances NemaContext by:
1. Replacing k-NN approximated spatial graphs with true contact-based graphs
2. Adding morphological features for cell representation
3. Providing standardized spatial coordinates for improved consistency

**Paper**: Cao et al. 2020 "Establishment of a morphological atlas of the Caenorhabditis elegans embryo using deep-learning-based 4D segmentation" (DOI: 10.1038/s41467-020-19863-x)

---

## Current Architecture

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Current Data Processing Pipeline                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Large2025/Packer2019          WormGUIDES           WormBase          │
│   (scRNA-seq MTX)               (nuclei files)       (lineage JSON)    │
│         │                            │                    │            │
│         ▼                            ▼                    ▼            │
│   ┌────────────────┐         ┌────────────────┐   ┌────────────────┐   │
│   │ExpressionLoader│         │ SpatialMatcher │   │ LineageEncoder │   │
│   │                │         │                │   │                │   │
│   │ - load_large2025()       │ - match_by_time()  │ - parse_lineage()  │
│   │ - load_packer2019()      │ - match_by_lineage()│ - encode_binary() │
│   │ - CSR matrix output      │ - XYZ coords   │   │ - build_adjacency()│
│   └────────────────┘         └────────────────┘   └────────────────┘   │
│         │                            │                    │            │
│         └────────────────────────────┼────────────────────┘            │
│                                      │                                 │
│                                      ▼                                 │
│                          ┌────────────────────────┐                    │
│                          │TrimodalAnnDataBuilder  │                    │
│                          │                        │                    │
│                          │ - build()              │                    │
│                          │ - build_spatial_graph()│                    │
│                          │ - build_lineage_graph()│                    │
│                          └────────────────────────┘                    │
│                                      │                                 │
│                                      ▼                                 │
│                          ┌────────────────────────┐                    │
│                          │     AnnData Output     │                    │
│                          │                        │                    │
│                          │ X: expression matrix   │                    │
│                          │ obsm['X_spatial']      │                    │
│                          │ obsm['X_lineage_binary']│                   │
│                          │ obsp['spatial_distances']│                  │
│                          │ obsp['lineage_adjacency']│                  │
│                          └────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Core Components

| Component | File | Responsibility |
|-----------|------|----------------|
| `ExpressionLoader` | `src/data/builder/expression_loader.py` | Load Large2025/Packer2019 expression matrices |
| `SpatialMatcher` | `src/data/builder/spatial_matcher.py` | Match WormGUIDES spatial coordinates |
| `LineageEncoder` | `src/data/builder/lineage_encoder.py` | Parse/encode lineage names |
| `WormAtlasMapper` | `src/data/builder/worm_atlas.py` | Cell type ↔ lineage mappings |
| `TrimodalAnnDataBuilder` | `src/data/builder/anndata_builder.py` | Integrate and build AnnData |
| `SpatialGraphBuilder` | `src/model/spatial_graph.py` | Build spatial neighbor graph (k-NN) |

### Current Limitations

1. **Spatial data is point-only**: WormGUIDES only provides nuclear positions, no cell morphology
2. **Neighbor graph is k-NN approximation**: Not true physical contacts
3. **No morphological features**: Missing volume, surface area, irregularity, etc.
4. **Limited time coverage**: Large2025 time distribution doesn't fully align with WormGUIDES

---

## CShaper Data Resources

### Available Data

| Data File | Path | Content | Key Fields |
|-----------|------|---------|------------|
| **ContactInterface/** | `dataset/raw/cshaper/ContactInterface/Sample*_Stat.csv` | Cell-cell contact area matrices | Symmetric matrix, non-zero = contact area (μm²) |
| **VolumeAndSurface/** | `dataset/raw/cshaper/VolumeAndSurface/Sample*_Stat.csv` | Cell volume/surface time series | Row = time frame, Column = cell |
| **Standard Dataset 1** | `dataset/raw/cshaper/Standard Dataset 1/*.mat` | Standardized 3D coordinates (46 embryo average) | Organized by lineage tree structure (gen×pos) |
| **Standard Dataset 2** | `dataset/raw/cshaper/Standard Dataset 2/*.mat` | 3D voxel segmentation results | 184×114×256 matrices |

### Data Characteristics

```
CShaper time range: 4-350 cell stage (approximately 20-380 minutes)
Time frames: 54 frames
Embryos: 17 (complete membrane segmentation) + 29 (nuclear tracking only) = 46 standard dataset
Cell naming: Standard C. elegans lineage naming (ABa, ABp, MS, E, C, ...)
```

### Sample Files Available

ContactInterface and VolumeAndSurface each contain 17 sample files:
- Sample04 through Sample20 (`Sample{04-20}_Stat.csv`)

---

## Integration Design

### Enhanced Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Enhanced Data Processing Pipeline                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Large2025          WormGUIDES       WormBase        CShaper          │
│   (scRNA-seq)        (nuclei)         (lineage)       (morphology)     │
│        │                 │                │               │            │
│        ▼                 ▼                ▼               ▼            │
│   ┌──────────┐     ┌──────────┐    ┌──────────┐    ┌──────────────┐   │
│   │Expression│     │ Spatial  │    │ Lineage  │    │ CShaper      │   │
│   │ Loader   │     │ Matcher  │    │ Encoder  │    │ Processor    │   │
│   └──────────┘     └──────────┘    └──────────┘    │              │   │
│        │                 │                │        │-load_contact()│   │
│        │                 │                │        │-load_volume() │   │
│        │                 │                │        │-load_spatial()│   │
│        │                 │                │        └──────────────┘   │
│        │                 │                │               │            │
│        └─────────────────┴────────────────┴───────────────┘            │
│                                   │                                    │
│                                   ▼                                    │
│                     ┌──────────────────────────────┐                   │
│                     │  EnhancedAnnDataBuilder      │                   │
│                     │                              │                   │
│                     │  - build_with_cshaper()      │                   │
│                     │  - _add_morphology()         │                   │
│                     │  - _add_contact_graph()      │                   │
│                     │  - _enhance_spatial()        │                   │
│                     └──────────────────────────────┘                   │
│                                   │                                    │
│                                   ▼                                    │
│                     ┌──────────────────────────────┐                   │
│                     │     Enhanced AnnData         │                   │
│                     │                              │                   │
│                     │  X: expression matrix        │                   │
│                     │                              │                   │
│                     │  obsm['X_spatial']           │ (WormGUIDES)      │
│                     │  obsm['X_lineage_binary']    │                   │
│                     │  obsm['X_cshaper_spatial']   │ (optional)        │
│                     │                              │                   │
│                     │  obs['cell_volume']          │                   │
│                     │  obs['cell_surface']         │                   │
│                     │  obs['sphericity']           │                   │
│                     │                              │                   │
│                     │  obsp['contact_adjacency']   │ (true contacts)   │
│                     │  obsp['spatial_distances']   │                   │
│                     │  obsp['lineage_adjacency']   │                   │
│                     └──────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### New/Modified Files

```
src/data/builder/
├── cshaper_processor.py    # CShaper data processor (implemented)
└── enhanced_builder.py     # Enhanced AnnData builder (implemented)
```

---

## Implementation Details

### CShaperProcessor (`src/data/builder/cshaper_processor.py`)

Main class combining all CShaper data loaders:

```python
class CShaperProcessor:
    """
    CShaper data processor - load and process morphological data
    """
    
    def __init__(self, data_dir: str = "dataset/raw"):
        self.data_dir = Path(data_dir)
        self.cshaper_dir = self.data_dir / "cshaper"
        
    # === Contact Interface ===
    def get_contact_adjacency(
        self,
        lineage_names: List[str],
        sample_id: Optional[str] = None,
        threshold: float = 0.0,
        binary: bool = False,
    ) -> csr_matrix:
        """Build contact adjacency matrix for given cells"""
        
    def get_contact_statistics(self) -> Dict:
        """Get summary statistics about contact data"""
    
    # === Morphology ===    
    def get_morphology_features(
        self,
        lineage_names: List[str],
        embryo_times: Optional[np.ndarray] = None,
        sample_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get cell morphology features (volume, surface, sphericity)"""
    
    # === Standardized Spatial Coordinates ===
    def get_spatial_coords(
        self,
        lineage_names: List[str],
        embryo_times: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Get standardized 3D coordinates"""
```

### ContactLoader

Handles ContactInterface CSV files:

```python
class ContactLoader:
    """
    CShaper ContactInterface data loader
    
    File format: Sample{04-20}_Stat.csv
    - Row 1: "cell1" + first cell names in each contact pair
    - Row 2: "cell2" + second cell names in each contact pair
    - Rows 3+: frame index + contact area values (μm²)
    """
    
    def load_sample(self, sample_id: str, frame: Optional[int] = None) -> pd.DataFrame
    def get_consensus_contacts(self, min_samples: int = 3) -> pd.DataFrame
    def build_adjacency_matrix(self, cell_list: List[str], ...) -> csr_matrix
```

### MorphologyLoader

Handles VolumeAndSurface CSV files:

```python
class MorphologyLoader:
    """
    CShaper VolumeAndSurface data loader
    
    File format:
    - Sample{04-20}_volume.csv: Volume data (μm³)
    - Sample{04-20}_surface.csv: Surface area data (μm²)
    """
    
    def load_sample(self, sample_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]
    def get_morphology_at_frame(self, frame: int, sample_id: Optional[str] = None) -> pd.DataFrame
    def get_features_for_cells(self, cell_names: List[str], time_frames: np.ndarray = None) -> pd.DataFrame
```

### EnhancedAnnDataBuilder (`src/data/builder/enhanced_builder.py`)

Extends TrimodalAnnDataBuilder to add CShaper data:

```python
class EnhancedAnnDataBuilder(TrimodalAnnDataBuilder):
    """
    Enhanced AnnData builder with CShaper morphological data integration
    """
    
    def build_with_cshaper(
        self,
        variant: Literal["complete", "extended"] = "complete",
        source: Literal["large2025", "packer2019"] = "large2025",
        include_morphology: bool = True,
        include_contact_graph: bool = True,
        use_cshaper_spatial: bool = False,
        **kwargs
    ) -> ad.AnnData:
        """Build enhanced AnnData with CShaper integration"""
```

---

## Lineage Name Matching

### Naming Conventions

| Data Source | Example Name | Format Description |
|-------------|--------------|-------------------|
| Large2025 | `ABplpapppa` | Full lowercase path |
| WormGUIDES | `ABplpapppa` | Full lowercase path |
| CShaper ContactInterface | `ABplpapppa` | Full lowercase path |
| CShaper VolumeAndSurface | `ABplpapppa` | Full lowercase path |
| CShaper Standard DS1 | By tree index | gen×pos matrix |

### Normalization Function

```python
def normalize_lineage_name(name: str) -> str:
    """Normalize lineage name to standard format"""
    # Remove whitespace and periods
    name = name.replace(" ", "").replace(".", "")
    # Standardize founder prefix to uppercase
    for founder in ["AB", "MS", "EMS", "P0", "P1", "P2", "P3", "P4", "Z2", "Z3"]:
        if name.upper().startswith(founder):
            return founder + name[len(founder):].lower()
    # E, C, D special handling
    if name[0].upper() in "ECD":
        return name[0].upper() + name[1:].lower()
    return name
```

---

## Time Alignment

### Time Mapping

```python
# CShaper: 54 frames covering 4-350 cell stage
# Approximately corresponds to: 20 min - 380 min (similar to WormGUIDES)

CSHAPER_FRAMES = 54
CSHAPER_START_TIME_MIN = 20
CSHAPER_END_TIME_MIN = 380

def embryo_time_to_cshaper_frame(time_min: float) -> int:
    """Map embryo time to CShaper frame"""
    if time_min < CSHAPER_START_TIME_MIN:
        return 0
    if time_min > CSHAPER_END_TIME_MIN:
        return CSHAPER_FRAMES - 1
    
    # Linear mapping
    fraction = (time_min - CSHAPER_START_TIME_MIN) / (CSHAPER_END_TIME_MIN - CSHAPER_START_TIME_MIN)
    return int(fraction * (CSHAPER_FRAMES - 1))
```

### Handling Time Mismatches

For cells in Large2025 with missing or out-of-range embryo_time:
1. Estimate developmental stage from lineage depth
2. Infer time window from cell type
3. Use cross-time averages

---

## Output Schema

### Enhanced AnnData Structure

```python
adata = AnnData(
    # === Original ===
    X=expression_matrix,                    # (n_cells, n_genes) sparse
    
    obs={
        # Original
        'cell_type': ...,
        'lineage_complete': ...,
        'embryo_time_min': ...,
        'lineage_valid': ...,
        'lineage_founder': ...,
        'lineage_depth': ...,
        'has_spatial': ...,
        
        # CShaper additions
        'cell_volume': ...,                 # Cell volume (μm³)
        'cell_surface': ...,                # Surface area (μm²)
        'sphericity': ...,                  # Sphericity [0,1]
        'has_morphology': ...,              # bool: has morphology data
        'cshaper_frame': ...,               # Matched CShaper frame number
    },
    
    var={...},                              # Gene annotations
    
    obsm={
        # Original
        'X_spatial': ...,                   # (n_cells, 3) WormGUIDES coords
        'X_lineage_binary': ...,            # (n_cells, max_depth) lineage encoding
        
        # CShaper addition
        'X_cshaper_spatial': ...,           # (n_cells, 3) CShaper standardized coords
    },
    
    obsp={
        # Original
        'spatial_distances': ...,           # k-NN distances
        'spatial_connectivities': ...,      # k-NN connections
        'lineage_adjacency': ...,           # Lineage adjacency
        
        # CShaper additions
        'contact_adjacency': ...,           # True contact graph (weighted)
        'contact_binary': ...,              # True contact graph (binary)
    },
    
    uns={
        # Original
        'data_sources': {...},
        'build_params': {...},
        
        # CShaper additions
        'cshaper_info': {
            'source': ...,
            'n_cells_with_morphology': ...,
            'n_contact_edges': ...,
            'mean_volume': ...,
            'mean_surface': ...,
            'mean_sphericity': ...,
            'mean_contact_degree': ...,
        },
    }
)
```

---

## Implementation Status

### Phase 1 (MVP) - Contact Graph Integration ✅
- [x] `CShaperProcessor` base class
- [x] `ContactLoader` contact matrix loading
- [x] `EnhancedAnnDataBuilder._add_contact_graph()`
- [x] Graph comparison utilities (contact vs k-NN)

### Phase 2 - Morphology Features ✅
- [x] `MorphologyLoader` volume/surface loading
- [x] `EnhancedAnnDataBuilder._add_morphology_features()`
- [x] Derived feature computation (sphericity)

### Phase 3 - Spatial Coordinate Enhancement ✅
- [x] `StandardSpatialLoader` for HDF5/MAT files
- [x] Lineage → tree index conversion
- [x] Complete coordinate extraction from Standard Dataset 1
- [x] `EnhancedAnnDataBuilder._add_cshaper_spatial()`

### Phase 4 - Advanced Features ✅
- [x] `Segmentation3DLoader` for Standard Dataset 2
- [x] 3D shape descriptors (volume, surface, sphericity, elongation, solidity)
- [x] Time-dynamic contact graphs (`TimeDynamicContactGraph`)
- [x] Contact graph GNN (`ContactGraph`, `ContactMessagePassing`, `ContactGNN`)

---

## Model Impact

### Spatial GNN Enhancement

```python
# Before: k-NN graph (approximate neighbors)
edge_index = knn_graph(spatial_coords, k=10)

# After: True contact graph
edge_index = contact_adjacency.nonzero()
edge_weight = contact_adjacency.data  # Contact area as edge weight
```

### Cell Token Feature Enhancement

```python
# Before: expression + lineage encoding
cell_features = torch.cat([expression, lineage_binary], dim=-1)

# After: expression + lineage encoding + morphology features
morphology = torch.stack([volume, surface, sphericity], dim=-1)
cell_features = torch.cat([expression, lineage_binary, morphology], dim=-1)
```

### Message Passing Improvement

```python
# GNN message passing with true neighbor relationships
def forward(self, x, contact_edge_index, contact_edge_weight):
    # Neighbor aggregation weighted by contact area
    return self.conv(x, contact_edge_index, edge_weight=contact_edge_weight)
```

---

## Validation Plan

1. **Data Integrity**: Verify CShaper lineage name match rate with Large2025
2. **Contact Graph Quality**: Compare contact graph with k-NN graph structural differences
3. **Morphology Feature Distribution**: Visualize volume/surface changes over developmental time
4. **Downstream Tasks**: Evaluate enhanced data impact on cell type prediction

---

## Usage

### Building Enhanced AnnData

```bash
# Build enhanced dataset with contact graph and morphology
uv run python examples/build_enhanced_anndata.py

# Include CShaper spatial coordinates
uv run python examples/build_enhanced_anndata.py --use-cshaper-spatial

# Build extended variant (all cells)
uv run python examples/build_enhanced_anndata.py --variant extended

# Compare contact graph with k-NN graph
uv run python examples/build_enhanced_anndata.py --compare-graphs

# Print CShaper data summary
uv run python examples/build_enhanced_anndata.py --cshaper-summary
```

### Python API

```python
from src.data.builder import EnhancedAnnDataBuilder, CShaperProcessor

# Check available CShaper data
processor = CShaperProcessor("dataset/raw")
print(processor.summary())

# Build enhanced AnnData
builder = EnhancedAnnDataBuilder()
adata = builder.build_with_cshaper(
    variant="complete",
    include_morphology=True,
    include_contact_graph=True,
    use_cshaper_spatial=False,
)

# Access CShaper-added data
print(adata.obs[['cell_volume', 'cell_surface', 'sphericity']].describe())
print(f"Contact graph edges: {adata.obsp['contact_binary'].nnz // 2}")

# Compare graphs
comparison = builder.compare_graphs(adata)
print(f"Jaccard similarity: {comparison['jaccard']:.3f}")
```

---

## New Components

### Segmentation3DLoader

Loads 3D voxel segmentation data from Standard Dataset 2:

```python
from src.data.builder import CShaperProcessor

processor = CShaperProcessor("dataset/raw")
loader = processor.segmentation_loader

# Load segmentation volume
seg = loader.load_segmentation(time_idx=1, sample_idx=4)  # (184, 114, 256)

# Get cell labels
labels = loader.get_cell_labels(seg)

# Compute 3D shape descriptors for a cell
descriptors = loader.compute_shape_descriptors(seg, cell_label=labels[0])
# Returns: volume_um3, surface_area_um2, centroid_um, bbox_size_um,
#          sphericity, elongation, solidity

# Compute for all cells
all_desc = loader.compute_all_shape_descriptors(time_idx=1, sample_idx=4)
```

### Time-Dynamic Contact Graphs

Track how cell-cell contacts evolve over developmental time:

```python
from src.model import ContactGraph, TimeDynamicContactGraph

# Build per-frame graphs
graphs = {}
for frame in range(1, 55):
    adj = processor.contact_loader.load_sample("Sample04", frame=frame)
    # ... build ContactGraph
    graphs[frame] = graph

# Create time-dynamic graph
time_graph = TimeDynamicContactGraph.from_frame_graphs(graphs)

# Get contact trajectory between two cells
times, areas = time_graph.get_contact_trajectory(cell1_idx, cell2_idx)

# Get all temporal edges for spatio-temporal GNN
edge_index, edge_weight, edge_time = time_graph.get_temporal_edges()
```

### Contact Graph GNN

Graph neural network layers for contact-based cell embeddings:

```python
from src.model import ContactGNN, build_contact_graph_from_cshaper

# Build ContactGraph from CShaper data
graph = build_contact_graph_from_cshaper(
    processor,
    lineage_names=cell_list,
    include_morphology=True,
)

# Create GNN
gnn = ContactGNN(
    in_features=3,  # morphology features
    hidden_features=64,
    out_features=32,
    n_layers=2,
)

# Get cell embeddings
embeddings = gnn.get_cell_embeddings(graph)
```

---

## References

1. Cao J et al. (2020) "Establishment of a morphological atlas of the Caenorhabditis elegans embryo using deep-learning-based 4D segmentation." Nat Commun 11:6254.

2. Large CRL et al. (2025) "Lineage-resolved analysis of embryonic gene expression evolution in C. elegans and C. briggsae." Science 388:eadu8249.

3. Bao Lab WormGUIDES: https://github.com/zhirongbaolab/WormGUIDES

---

## Changelog

- **v0.3.0** (2026-01): Complete CShaper integration
  - StandardSpatialLoader: Full coordinate extraction from Standard Dataset 1
  - Segmentation3DLoader: 3D shape descriptors from Standard Dataset 2
  - TimeDynamicContactGraph: Time-varying contact graph support
  - ContactGNN: Graph neural network layers for contact graphs
  - Test suite: Comprehensive validation script
- **v0.2.0** (2024-01): CShaper integration implemented
  - ContactLoader, MorphologyLoader, StandardSpatialLoader
  - EnhancedAnnDataBuilder with contact graph and morphology
  - Graph comparison utilities
- **v0.1.0** (2024-01): Initial trimodal implementation with Large2025 support
