# Data Downloader Module

Automated data acquisition for C. elegans multi-modal single-cell analysis.

## Overview

This module downloads and organizes data from multiple public sources for integrating:
- **Transcriptome**: Single-cell RNA-seq gene expression
- **Spatial**: 4D cell positions during embryonic development  
- **Lineage**: Cell lineage tree and relationships

## Quick Start

```bash
# Download core datasets (recommended)
uv run python -m src.data.downloader --source core

# Download all datasets including optional connectome
uv run python -m src.data.downloader --source all

# Download only the new Large 2025 transcriptome data
uv run python -m src.data.downloader --source large2025

# Programmatic usage
from src.data.downloader import NemaContextDownloader
downloader = NemaContextDownloader("dataset/raw")
downloader.download_core()  # Recommended
```

## Data Sources

### 1. 🌟 Large et al. 2025 (RECOMMENDED - Lineage-Resolved Atlas)

**Source**: NCBI GEO [GSE292756](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE292756)  
**Publication**: Science, June 2025 (PMID: 40536976)

| File | Size | Description |
|------|------|-------------|
| `GSE292756_expression_matrix.mtx.gz` | 1.1 GB | Expression matrix (C. elegans + C. briggsae) |
| `GSE292756_cell_annotations.csv.gz` | 11.7 MB | Lineage-resolved cell annotations |
| `GSE292756_gene_annotations.csv.gz` | 364 KB | Gene annotations |

**Key advantages over Packer 2019**:
- ✅ **>375,000 cells** (vs ~86,000 in Packer 2019)
- ✅ **Lineage-resolved annotations** - direct cell-to-lineage mapping
- ✅ **429 annotated cell types** (progenitor + terminal)
- ✅ **Cross-species comparison** with C. briggsae data
- ✅ **Gene regulatory networks** inference included
- ✅ **Solves annotation ambiguity** (no 'x' or '/' in lineage names)

**Coverage**: 120-600 minutes post-fertilization

**Additional Resources**:
- Interactive visualization: https://cello.shinyapps.io/cel_cbr_embryo_single_cell/
- Gene browser (GExplore): https://genome.science.sfu.ca/gexplore
- GitHub: https://github.com/livinlrg/C.elegans_C.briggsae_Embryo_Single_Cell
- Dryad: https://doi.org/10.5061/dryad.1rn8pk15n

---

### 2. WormGUIDES (4D Spatial Positions)

**Source**: [zhirongbaolab/WormGUIDES](https://github.com/zhirongbaolab/WormGUIDES) on GitHub

| File | Description |
|------|-------------|
| `nuclei_files/t001-nuclei` to `t360-nuclei` | 360 timepoints of 3D cell positions |
| `anatomy.csv` | Cell anatomy annotations |
| `CellDeaths.csv` | Programmed cell death events |
| `NeuronConnect.csv` | Neuron connectivity data |
| `partslist.txt` | Complete cell parts list |

**Nuclei file format** (comma-separated):
```
ID, flag, ..., X, Y, Z, diameter, cell_name, ...
```

**Coverage**: ~20-380 minutes (1-cell to ~580-cell stage)

**Cell count by timepoint**:
| Timepoint | Cells |
|-----------|-------|
| t=1 | 15 |
| t=100 | 113 |
| t=200 | 1,185 |
| t=300 | 1,890 |
| t=360 | 1,086 |

---

### 3. WormBase/Lineage Data

**Source**: Derived from WormGUIDES partslist + Sulston timing data

| File | Description |
|------|-------------|
| `lineage_tree.json` | 2,105 lineage tree nodes |
| `cell_lineage_map.json` | 722 terminal cell → lineage mappings |
| `cell_timing.json` | 1,323 cell birth/division times |
| `cell_fates.json` | 12 cell fate categories |

**Cell fate distribution**:
- Neuron: 380
- Muscle: 121
- Hypodermis: 44
- Pharynx: 23
- Other: 155

---

### 4. Packer et al. 2019 (Legacy - Single-cell Transcriptomics)

> ⚠️ **Note**: Superseded by Large et al. 2025. Use `--source packer` only if you specifically need this dataset.

**Source**: NCBI GEO [GSE126954](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE126954)

| File | Size | Description |
|------|------|-------------|
| `GSE126954_gene_by_cell_count_matrix.txt.gz` | 250 MB | Gene expression matrix |
| `GSE126954_cell_annotation.csv.gz` | 2.7 MB | Cell annotations (89,701 cells) |
| `GSE126954_gene_annotation.csv.gz` | 174 KB | Gene annotations (20,222 genes) |

**Known issues with Packer 2019**:
- Only ~10.8% of cells have "clean" lineage annotations
- 67% of annotations contain 'x' (branch uncertain)
- 32% contain '/' (multiple possibilities)
- Only ~5,143 cells directly matchable to spatial data

---

### 5. OpenWorm (Connectome) - Optional

**Source**: [openworm/c302](https://github.com/openworm/c302) on GitHub

| File | Description |
|------|-------------|
| `aconnectome_white_1986_whole.csv` | White et al. 1986 connectome |
| `herm_full_edgelist.csv` | Full hermaphrodite edge list |
| `CElegansNeuronTables.xls` | Neuron information tables |

**Note**: Connectome data is for **adult** neurons (302 cells) and may not directly match embryonic cell names.

---

## Dataset Comparison

| Feature | Large 2025 | Packer 2019 |
|---------|------------|-------------|
| Total cells | >375,000 | ~86,000 |
| C. elegans cells | >200,000 | ~86,000 |
| C. briggsae cells | >175,000 | None |
| Annotated cell types | 429 | Variable |
| Lineage quality | Direct mapping | ~10% clean |
| Time range | 120-600 min | 0-830 min |
| GRN inference | ✅ Included | ❌ No |
| Recommended | ✅ Yes | ❌ Legacy |

---

## Data Integration Strategy

### Recommended Workflow (with Large 2025)

The Large 2025 dataset provides direct cell-type-to-lineage mapping, making integration straightforward:

```
Large 2025              WormGUIDES           WormBase
(Transcriptome)         (Spatial)            (Lineage)
   429 cell types       1,341 cells          2,105 nodes
      ↓                     ↓                    ↓
   Cell type name       Cell names           Tree structure
      ↓                     ↓                    ↓
      └────────────→  DIRECT MATCHING  ←────────┘
```

### Example Code

```python
import pandas as pd
from scipy.io import mmread
import gzip

# Load Large 2025 data
with gzip.open('dataset/raw/large2025/GSE292756_cell_annotations.csv.gz', 'rt') as f:
    cell_annot = pd.read_csv(f)

# Cell types are directly usable for matching with WormGUIDES
print(f"Total cells: {len(cell_annot)}")
print(f"C. elegans cells: {(cell_annot['species'] == 'elegans').sum()}")
print(f"Unique cell types: {cell_annot['cell_type'].nunique()}")
```

---

## CLI Options

```bash
# Show help
uv run python -m src.data.downloader --help

# Download options
--source core       # Large 2025 + WormGUIDES + WormBase (default)
--source all        # All datasets including connectome
--source large2025  # Only Large 2025 transcriptome
--source packer     # Only Packer 2019 (legacy)
--source wormguides # Only WormGUIDES spatial data
--source wormbase   # Only WormBase lineage data
--source openworm   # Only OpenWorm connectome

# Custom data directory
--data-dir /path/to/data
```

---

## Module Structure

```
downloader/
├── __init__.py          # Package exports
├── __main__.py          # CLI entry point
├── base.py              # BaseDownloader class
├── constants.py         # URLs, file definitions, config
├── main.py              # NemaContextDownloader orchestrator
├── large2025.py         # Large 2025 downloader (RECOMMENDED)
├── packer.py            # Packer 2019 downloader (legacy)
├── openworm.py          # OpenWorm downloader
├── wormguides.py        # WormGUIDES downloader (incl. nuclei)
└── wormbase.py          # WormBase lineage generator
```

## Adding New Data Sources

1. Create a new downloader class inheriting from `BaseDownloader`
2. Define URLs and files in `constants.py`
3. Implement the `download()` method
4. Register in `NemaContextDownloader`

---

## References

1. **Large, C. R. L., et al. (2025)**. Lineage-resolved analysis of embryonic gene expression evolution in C. elegans and C. briggsae. *Science* 388:eadu8249. DOI: 10.1126/science.adu8249

2. **Packer, J. S., et al. (2019)**. A lineage-resolved molecular atlas of C. elegans embryogenesis at single-cell resolution. *Science*.

3. **Bao, Z., et al. (2006)**. Automated cell lineage tracing in Caenorhabditis elegans. *PNAS*.

4. **Sulston, J. E., et al. (1983)**. The embryonic cell lineage of the nematode Caenorhabditis elegans. *Developmental Biology*.

5. **WormGUIDES**: Bao Lab, Sloan Kettering Institute. https://wormguides.org/

6. **OpenWorm**: Open source project to simulate C. elegans. https://openworm.org/