# NemaContext Data Integration Guide

This document describes how to integrate the three modalities of C. elegans embryo data
into a unified AnnData format for downstream modeling.

---

## Table of Contents

1. [Overview](#overview)
2. [Data Sources](#data-sources)
3. [Data Coverage Analysis](#data-coverage-analysis)
4. [AnnData Schema](#anndata-schema)
5. [Build Variants](#build-variants)
6. [Usage Guide](#usage-guide)
7. [Handling Missing Modalities](#handling-missing-modalities)
8. [Technical Details](#technical-details)

---

## Overview

NemaContext integrates three modalities of C. elegans embryo data:

| Modality | Source | Content | Coverage |
|----------|--------|---------|----------|
| **Transcriptome** | Large et al. 2025 | Gene expression (242k cells) | 0-830 min |
| **Spatial** | WormGUIDES | 4D nuclear positions | 20-380 min |
| **Lineage** | WormBase/Sulston | Cell division tree | Complete |

The integration produces AnnData (`.h5ad`) files compatible with the scverse ecosystem
(Scanpy, Squidpy, etc.) and PyTorch-based deep learning pipelines.

### Key Challenge

The three modalities have **different temporal coverage**:

```
Time (min):  0    100    200    300    400    500    600    700    800
             |------|------|------|------|------|------|------|------|
Transcriptome:      ████████████████████████████████████████████████
Spatial:              █████████████████████
Lineage:     ████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
                                     ↑
                            Quality degrades
```

**Result**: Only ~38% of cells have complete trimodal data (20-380 min window).

---

## Data Sources

### 1. Large et al. 2025 (GSE292756) - Transcriptome

**The recommended transcriptome source.**

- **Publication**: Science 2025, DOI: 10.1126/science.adu8249
- **Total cells**: 425,174 (C. elegans: 242,188; C. briggsae: 183,000)
- **Genes**: ~20,000
- **Format**: MTX (Matrix Market) + CSV annotations

Key columns in cell annotations:
```
cell_type           - Terminal cell type (e.g., "BWM_headrow1")
lineage_complete    - Full lineage path (e.g., "MSxpappp")  
species             - "C.elegans" or "C.briggsae"
smoothed.embryo.time - Developmental time in minutes
batch               - Technical batch identifier
```

**Lineage annotation quality by time:**

| Time Period | Cells | Has Lineage | Clean Lineage | Clean % |
|-------------|-------|-------------|---------------|---------|
| 0-100 min   | 446   | 322         | 17            | 3.8%    |
| 100-200 min | 15,334| 6,762       | 1,060         | 6.9%    |
| 200-300 min | 33,282| 15,034      | 1,933         | 5.8%    |
| 300-400 min | 51,544| 13,259      | 871           | 1.7%    |
| 400-500 min | 54,315| 1,262       | 221           | 0.4%    |
| >500 min    | 87,267| 166         | 25            | 0.03%   |

"Clean lineage" = no ambiguity markers (`x`, `/`, `?`).

### 2. WormGUIDES - Spatial Coordinates

**4D nuclear tracking data from the Bao Lab.**

- **Source**: https://github.com/zhirongbaolab/WormGUIDES
- **Timepoints**: 360 (t001 to t360)
- **Time range**: ~20-380 minutes post-fertilization
- **Resolution**: 1 minute per timepoint

File format (`tXXX-nuclei`):
```
ID, FLAG, ..., X, Y, Z, DIAMETER, CELL_NAME, ...
```

Example:
```
8, 1, -1, 26, -1, 228, 138, 127.0, 27, EMS, ...
9, 0, -1, -1, -1, 108, 119, 131.0, 29, ABa, ...
```

**Named cells per timepoint:**
- t001: 5 cells (4-cell stage)
- t100: 87 cells
- t200: 360 cells
- t350: 579 cells (late gastrulation)

### 3. WormBase - Lineage Tree

**Complete lineage structure from Sulston et al. 1983.**

Files in `dataset/raw/wormbase/`:
```
lineage_tree.json      - Tree structure (parent/children for 2,105 cells)
cell_timing.json       - Division timing information
cell_fates.json        - Terminal fate annotations
cell_lineage_map.json  - Name mappings
```

Tree structure example:
```json
{
  "P0": {"children": ["AB", "P1"], "parent": null},
  "AB": {"children": ["ABa", "ABp"], "parent": "P0"},
  "ABa": {"children": ["ABal", "ABar"], "parent": "AB"},
  ...
}
```

---

## Data Coverage Analysis

### Temporal Overlap

```
               0         100        200        300        400        500+
               |----------|----------|----------|----------|----------|
Transcriptome: ·····██████████████████████████████████████████████████
WormGUIDES:    ·······████████████████████████████·····················
Clean Lineage: ·····████████████████████░░░░░░░░···················· 

Legend:
  █ = Good coverage
  ░ = Sparse/degraded
  · = No data
```

### Quantified Coverage

| Subset | Cell Count | % of C. elegans |
|--------|------------|-----------------|
| All C. elegans | 242,188 | 100% |
| In WormGUIDES time (20-380 min) | 93,097 | 38.4% |
| With clean lineage | 4,127 | 1.7% |
| With spatial match | ~3,851 | 1.6% |
| **Complete trimodal** | **~3,800** | **1.6%** |

### Why Does Coverage Drop?

1. **WormGUIDES stops at ~380 min**: Nuclear tracking becomes difficult as:
   - Embryo becomes dense
   - Cells undergo morphogenesis
   - Embryo starts twitching

2. **Lineage annotation degrades after ~400 min**:
   - Cells become "terminally differentiated"
   - Identified by TYPE (BWM_middle) not LINEAGE (MSxpappp)
   - Many cells fuse (hyp7 = 100+ cells)

3. **No deterministic mapping**: Later cells identified by function, not position.

---

## AnnData Schema

### Overview

```python
adata = AnnData(X=expression_matrix)  # (n_cells, n_genes)

# Core attributes
adata.obs      # Cell metadata (DataFrame)
adata.var      # Gene metadata (DataFrame)
adata.obsm     # Embeddings (dict of arrays)
adata.obsp     # Cell-cell graphs (dict of sparse matrices)
adata.layers   # Alternative expression matrices
adata.uns      # Unstructured metadata
```

### Detailed Schema

#### `adata.obs` - Cell Metadata

| Column | Type | Source | Description |
|--------|------|--------|-------------|
| `cell_type` | Categorical | Large2025 | Terminal cell type |
| `species` | Categorical | Large2025 | C.elegans / C.briggsae |
| `embryo_time_min` | float | Large2025 | Developmental time (minutes) |
| `batch` | Categorical | Large2025 | Technical batch |
| `lineage_valid` | bool | Derived | Has clean lineage annotation |
| `lineage_founder` | Categorical | Derived | AB, MS, E, C, D, P4, unknown |
| `lineage_depth` | int | Derived | Divisions from P0 (-1 if invalid) |
| `has_spatial` | bool | Derived | Matched to WormGUIDES |
| `wormguides_timepoint` | int | Derived | Matched timepoint (-1 if none) |
| `modality_mask` | int8 | Derived | Bitmask: 1=trans, 2=spatial, 4=lineage |

#### `adata.var` - Gene Metadata

| Column | Type | Description |
|--------|------|-------------|
| `gene_id` | str | WormBase gene ID |
| `gene_name` | str | Common name |
| *(other columns from source)* | | |

#### `adata.obsm` - Embeddings

| Key | Shape | Description |
|-----|-------|-------------|
| `X_spatial` | (n_cells, 3) | XYZ coordinates from WormGUIDES (NaN if no match) |
| `X_lineage_binary` | (n_cells, 20) | Binary path encoding (-1 = pad, 0 = 'a', 1 = 'p') |
| `X_pca` | (n_cells, 50) | PCA of log-normalized expression |

#### `adata.obsp` - Graphs

| Key | Type | Description |
|-----|------|-------------|
| `spatial_distances` | sparse | Pairwise spatial distances (KNN) |
| `spatial_connectivities` | sparse | Binary spatial adjacency |
| `lineage_adjacency` | sparse | Parent-child relationships |

#### `adata.layers` - Expression Variants

| Key | Description |
|-----|-------------|
| `counts` | Raw UMI counts |
| `log1p` | Log-normalized (after total count normalization) |

#### `adata.uns` - Metadata

| Key | Type | Description |
|-----|------|-------------|
| `data_sources` | dict | Source information for each modality |
| `build_params` | dict | Parameters used during build |
| `lineage_tree` | dict | Complete lineage tree structure |
| `celltype_to_lineage` | dict | WormAtlas cell type → lineage mapping |
| `modality_stats` | dict | Coverage statistics |
| `founder_distribution` | dict | Cells per founder lineage |

---

## Build Variants

### Variant 1: `complete` (Recommended for Model Development)

**Only cells with all three modalities.**

```bash
uv run python examples/build_anndata.py --variant complete
```

Output: `dataset/processed/nema_complete_large2025.h5ad`

Properties:
- ~3,800 cells
- All cells have: expression, spatial coords, clean lineage
- Time range: 20-380 min
- Suitable for: model prototyping, validation

### Variant 2: `extended` (Full Dataset)

**All cells with modality availability flags.**

```bash
uv run python examples/build_anndata.py --variant extended
```

Output: `dataset/processed/nema_extended_large2025.h5ad`

Properties:
- ~242,000 cells (C. elegans)
- Modality availability in `obs['modality_mask']`
- Full time range: 0-830 min
- Suitable for: large-scale training, cell type analysis

### Modality Mask Usage

The `modality_mask` column uses a bitmask:
```python
TRANSCRIPTOME = 1  # bit 0
SPATIAL = 2        # bit 1  
LINEAGE = 4        # bit 2

# Examples:
# mask = 1 → transcriptome only
# mask = 5 → transcriptome + lineage (1 + 4)
# mask = 7 → all three (1 + 2 + 4)

# Filter to cells with all modalities:
complete_mask = (adata.obs['modality_mask'] & 7) == 7
```

---

## Usage Guide

### Prerequisites

1. Download required data:
```bash
uv run python -m src.data.downloader --source core
```

2. Verify downloads:
```bash
ls dataset/raw/large2025/     # Expression data
ls dataset/raw/wormguides/nuclei_files/ | head  # Spatial data
ls dataset/raw/wormbase/      # Lineage data
```

### Building AnnData

#### Basic Usage

```bash
# Complete trimodal (recommended)
uv run python examples/build_anndata.py --variant complete

# Extended with all cells
uv run python examples/build_anndata.py --variant extended
```

#### Advanced Options

```bash
# Use Packer2019 instead of Large2025
uv run python examples/build_anndata.py --source packer2019

# Custom UMI threshold
uv run python examples/build_anndata.py --min-umi 1000

# Add spatial and lineage graphs
uv run python examples/build_anndata.py \
    --add-spatial-graph \
    --add-lineage-graph \
    --n-neighbors 15

# Skip normalization and PCA (for custom preprocessing)
uv run python examples/build_anndata.py --no-normalize --no-pca
```

### Loading and Using AnnData

```python
import scanpy as sc
import anndata as ad

# Load
adata = ad.read_h5ad("dataset/processed/nema_complete_large2025.h5ad")

# Basic info
print(f"Cells: {adata.n_obs}, Genes: {adata.n_vars}")
print(adata.obs.head())

# Access spatial coordinates
spatial = adata.obsm['X_spatial']  # (n_cells, 3)

# Access lineage encoding
lineage_binary = adata.obsm['X_lineage_binary']  # (n_cells, 20)

# Filter by modality
has_all = (adata.obs['modality_mask'] & 7) == 7
adata_complete = adata[has_all]

# Filter by time
early = adata[adata.obs['embryo_time_min'] < 200]

# Standard Scanpy workflow
sc.pp.neighbors(adata, use_rep='X_pca')
sc.tl.umap(adata)
sc.pl.umap(adata, color=['lineage_founder', 'cell_type'])
```

### PyTorch DataLoader

```python
from anndata.experimental import AnnLoader

# Create DataLoader
loader = AnnLoader(adata, batch_size=64, shuffle=True)

for batch in loader:
    # batch.X → expression tensor
    # batch.obs → metadata DataFrame
    # batch.obsm['X_spatial'] → spatial coords
    break
```

---

## Handling Missing Modalities

### Strategy 1: Use Complete Subset Only

Best for: Model development, validation

```python
adata = ad.read_h5ad("nema_complete_large2025.h5ad")
# All cells guaranteed to have all modalities
```

### Strategy 2: Conditional Masking

Best for: Training with partial data

```python
# During training, mask unavailable modalities
spatial_available = adata.obs['has_spatial'].values
lineage_available = adata.obs['lineage_valid'].values

# Example: masked loss
if spatial_available[i]:
    loss += spatial_reconstruction_loss(...)
```

### Strategy 3: Cell Type → Lineage Imputation

Best for: Extending lineage coverage to late-stage cells

The `celltype_to_lineage` mapping in `adata.uns` allows inferring lineage from cell type:

```python
from src.data.builder import WormAtlasMapper

mapper = WormAtlasMapper()

# Get lineage for a cell type
lineages = mapper.celltype_to_lineage("ADAL")
# Returns: ['ABplapaaaapp']

# Infer lineage for cells missing it
for i, row in adata.obs.iterrows():
    if not row['lineage_valid'] and row['cell_type'] != 'unassigned':
        inferred = mapper.celltype_to_lineage(row['cell_type'])
        if inferred:
            # Use inferred lineage (mark as imputed)
            ...
```

### Strategy 4: Spatial Position Inference

For cells outside WormGUIDES time range, spatial positions can be estimated from:

1. **Cell type anatomy**: BWM_middle has known anatomical location
2. **Neighbor averaging**: Average position of nearby cells in expression space
3. **Terminal positions**: Use adult atlas positions for late-stage cells

---

## Technical Details

### File Structure

```
NemaContext/
├── src/data/builder/
│   ├── __init__.py
│   ├── anndata_builder.py    # Main builder class
│   ├── expression_loader.py  # MTX loading
│   ├── spatial_matcher.py    # WormGUIDES matching
│   ├── lineage_encoder.py    # Lineage parsing
│   └── worm_atlas.py         # Cell type mappings
├── examples/
│   └── build_anndata.py      # CLI script
├── dataset/
│   ├── raw/                  # Downloaded data
│   └── processed/            # Built .h5ad files
└── docs/
    └── DATA_INTEGRATION.md   # This file
```

### Lineage Encoding Details

Lineage strings (e.g., "ABplpapppa") are encoded as:

1. **Founder identification**: AB, MS, E, C, D, P4
2. **Binary path**: Each division encoded as 0 (anterior/left) or 1 (posterior/right)
3. **Depth**: Total divisions from P0

Example:
```
"ABplpapppa" → {
    founder: "AB",
    path: "plpapppa",
    binary: [1, 0, 1, 0, 1, 1, 1, 0],
    depth: 9
}
```

### Spatial Matching Algorithm

1. For each cell, get `lineage_complete` from Large2025
2. Get `smoothed_embryo_time` and find nearest WormGUIDES timepoint
3. Look up cell name in that timepoint's nuclei file
4. If found, extract (X, Y, Z) coordinates

Match rate for clean lineages: **~98.4%**

### Performance Notes

- **Large2025 MTX loading**: ~2-3 min (1.1 GB compressed)
- **WormGUIDES parsing** (all timepoints): ~1-2 min
- **Complete build**: ~5-10 min total
- **Extended build**: ~10-15 min total

Memory requirements:
- Complete variant: ~2 GB RAM
- Extended variant: ~8-10 GB RAM

---

## References

1. Large CRL et al. (2025) "Lineage-resolved analysis of embryonic gene expression 
   evolution in C. elegans and C. briggsae." Science 388:eadu8249.
   
2. Bao Lab WormGUIDES: https://github.com/zhirongbaolab/WormGUIDES

3. Sulston JE et al. (1983) "The embryonic cell lineage of the nematode 
   Caenorhabditis elegans." Dev Biol 100:64-119.

4. Packer JS et al. (2019) "A lineage-resolved molecular atlas of C. elegans
   embryogenesis at single-cell resolution." Science 365:eaax1971.

---

## Changelog

- **v0.1.0** (2024-01): Initial implementation with Large2025 support