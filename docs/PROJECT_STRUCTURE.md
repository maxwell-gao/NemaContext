# NemaContext Project Structure

This document provides an overview of the NemaContext project organization and the purpose of each component.

---

## Directory Overview

```
NemaContext/
├── dataset/                    # Data storage
│   ├── raw/                    # Downloaded raw data
│   └── processed/              # Processed AnnData files
├── docs/                       # Documentation
├── examples/                   # Example scripts and notebooks
├── src/                        # Source code
│   ├── data/                   # Data handling modules
│   │   ├── builder/            # AnnData construction
│   │   ├── downloader/         # Data downloading
│   │   └── processors/         # Data processing utilities
│   └── model/                  # Model implementations
├── README.md                   # Project overview
├── pyproject.toml              # Project configuration
└── uv.lock                     # Dependency lock file
```

---

## Source Code (`src/`)

### `src/data/` - Data Handling

The data module handles all aspects of data acquisition, processing, and integration.

#### `src/data/downloader/` - Data Downloading

Downloads raw data from various sources.

| File | Description |
|------|-------------|
| `base.py` | Abstract base class for all downloaders |
| `constants.py` | URLs, file definitions, and configuration constants |
| `large2025.py` | Large et al. 2025 (GSE292756) downloader |
| `packer.py` | Packer et al. 2019 (GSE126954) downloader |
| `wormguides.py` | WormGUIDES 4D nuclei data downloader |
| `wormbase.py` | WormBase lineage tree downloader |
| `openworm.py` | OpenWorm connectome data downloader |
| `main.py` | Orchestrator class combining all downloaders |
| `__main__.py` | CLI entry point for `python -m src.data.downloader` |

**Usage:**
```bash
# Download core datasets (recommended)
uv run python -m src.data.downloader --source core

# Download specific dataset
uv run python -m src.data.downloader --source large2025
```

#### `src/data/builder/` - AnnData Construction

Builds integrated trimodal AnnData objects.

| File | Description |
|------|-------------|
| `anndata_builder.py` | Main `TrimodalAnnDataBuilder` class |
| `expression_loader.py` | Loads MTX expression matrices and annotations |
| `spatial_matcher.py` | Matches cells to WormGUIDES spatial coordinates |
| `lineage_encoder.py` | Parses and encodes lineage information |
| `worm_atlas.py` | Cell type to lineage mapping from WormAtlas |

**Key Class: `TrimodalAnnDataBuilder`**

```python
from src.data.builder import TrimodalAnnDataBuilder

builder = TrimodalAnnDataBuilder()
adata = builder.build(variant="complete")  # or "extended"
```

#### `src/data/processors/` - Data Processing Utilities

Convenience wrappers for data processing pipelines.

| File | Description |
|------|-------------|
| `celltype_mapper.py` | Cell type to lineage mapping utilities |
| `expression_loader.py` | Re-exports ExpressionLoader |
| `spatial_matcher.py` | Re-exports SpatialMatcher |
| `lineage_encoder.py` | Re-exports LineageEncoder |

---

### `src/model/` - Model Implementations

Neural network models for trimodal embedding and generation.

| File | Description |
|------|-------------|
| `multimodal.py` | Multimodal encoder/decoder architectures |
| `spatial.py` | Spatial embedding models |
| `spatial_graph.py` | GNN-based spatial processing |

*Note: Model implementations are under active development.*

---

## Data Directory (`dataset/`)

### `dataset/raw/` - Raw Downloaded Data

| Subdirectory | Content | Size (approx) |
|--------------|---------|---------------|
| `large2025/` | Expression matrix + annotations | ~1.2 GB |
| `packer2019/` | Legacy expression data | ~250 MB |
| `wormguides/nuclei_files/` | 360 timepoint nuclei files | ~25 MB |
| `wormbase/` | Lineage tree JSON files | ~600 KB |
| `openworm/` | Connectome data | ~2 MB |

### `dataset/processed/` - Processed AnnData

Output location for built `.h5ad` files.

| File | Description |
|------|-------------|
| `nema_complete_large2025.h5ad` | Complete trimodal (~3.8k cells) |
| `nema_extended_large2025.h5ad` | Extended dataset (~242k cells) |

---

## Documentation (`docs/`)

| File | Description |
|------|-------------|
| `DATA_INTEGRATION.md` | Comprehensive data integration guide |
| `PROJECT_STRUCTURE.md` | This file |

---

## Examples (`examples/`)

| Script | Description |
|--------|-------------|
| `build_anndata.py` | CLI script for building AnnData |
| `analyze_large2025.py` | Exploratory analysis of Large2025 data |
| `multimodal_embedding.py` | Example multimodal embedding pipeline |

**Running Examples:**
```bash
# Build trimodal AnnData
uv run python examples/build_anndata.py --variant complete

# Analyze dataset
uv run python examples/analyze_large2025.py
```

---

## Configuration Files

### `pyproject.toml`

Project metadata and dependencies:

```toml
[project]
name = "nemacontext"
dependencies = [
    "torch>=2.0.0",
    "scanpy",
    "anndata",
    "geoopt",           # Hyperbolic optimization
    "torch-geometric",  # GNN
    "torchsde",         # Stochastic differential equations
    "pandas",
    "scipy",
    "requests",
    "tqdm",
]
```

### `uv.lock`

Locked dependency versions for reproducibility.

---

## Module Import Structure

### From Project Root

```python
# Data downloading
from src.data.downloader import NemaContextDownloader

# AnnData building
from src.data.builder import (
    TrimodalAnnDataBuilder,
    ExpressionLoader,
    SpatialMatcher,
    LineageEncoder,
    WormAtlasMapper,
)

# Data processing
from src.data.processors import CellTypeMapper

# Models
from src.model import (
    MultiModalEmbedder,
    SpatialGraphNetwork,
)
```

### Lazy Loading

The `src.data` module uses lazy loading to avoid import overhead:

```python
from src.data import TrimodalAnnDataBuilder  # Loaded on first access
```

---

## Development Workflow

### 1. Setup Environment

```bash
# Clone and enter project
cd NemaContext

# Install dependencies with uv
uv sync
```

### 2. Download Data

```bash
# Download core datasets
uv run python -m src.data.downloader --source core

# Or download individually
uv run python -m src.data.downloader --source large2025
uv run python -m src.data.downloader --source wormguides
```

### 3. Build AnnData

```bash
# Build complete trimodal dataset
uv run python examples/build_anndata.py --variant complete

# Build extended dataset with all cells
uv run python examples/build_anndata.py --variant extended
```

### 4. Use in Code

```python
import anndata as ad

adata = ad.read_h5ad("dataset/processed/nema_complete_large2025.h5ad")
print(f"Cells: {adata.n_obs}, Genes: {adata.n_vars}")
```

---

## File Naming Conventions

| Pattern | Description | Example |
|---------|-------------|---------|
| `*.py` | Python source files | `anndata_builder.py` |
| `*.md` | Documentation | `DATA_INTEGRATION.md` |
| `*.h5ad` | AnnData files | `nema_complete_large2025.h5ad` |
| `*.json` | JSON data files | `lineage_tree.json` |
| `*.csv.gz` | Compressed CSV | `GSE292756_cell_annotations.csv.gz` |
| `*.mtx.gz` | Compressed MTX | `GSE292756_expression_matrix.mtx.gz` |

---

## Testing

```bash
# Quick import test
uv run python -c "from src.data.builder import TrimodalAnnDataBuilder; print('OK')"

# Test lineage encoder
uv run python -c "
from src.data.builder import LineageEncoder
enc = LineageEncoder()
print(enc.parse_lineage('ABplpapppa'))
"

# Test WormAtlas mapper
uv run python -c "
from src.data.builder import WormAtlasMapper
mapper = WormAtlasMapper()
print(mapper.celltype_to_lineage('ADAL'))
"
```

---

## Next Steps

See the following documents for more details:

- **[DATA_INTEGRATION.md](DATA_INTEGRATION.md)**: Detailed data integration guide
- **[README.md](../README.md)**: Project overview and research goals
- **[src/model/README.md](../src/model/README.md)**: Model architecture documentation

---

## Changelog

- **2024-01**: Initial structure with builder and processor modules
- **2024-01**: Added Large2025 downloader and integration
- **2024-01**: Created comprehensive documentation