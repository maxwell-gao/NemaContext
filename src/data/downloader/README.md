# Data Downloader Module

Automated data acquisition for C. elegans multi-modal single-cell analysis.

## Overview

This module downloads and organizes data from multiple public sources for integrating:
- **Transcriptome**: Single-cell RNA-seq gene expression
- **Spatial**: 4D cell positions during embryonic development  
- **Lineage**: Cell lineage tree and relationships

## Usage

```bash
# Download all data sources
uv run python -m src.data.downloader

# Or use programmatically
from src.data.downloader import NemaContextDownloader
downloader = NemaContextDownloader("dataset/raw")
downloader.download_all()
```

## Data Sources

### 1. Packer et al. 2019 (Single-cell Transcriptomics)

**Source**: NCBI GEO [GSE126954](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE126954)

| File | Size | Description |
|------|------|-------------|
| `GSE126954_gene_by_cell_count_matrix.txt.gz` | 250 MB | Gene expression matrix |
| `GSE126954_cell_annotation.csv.gz` | 2.7 MB | Cell annotations (89,701 cells) |
| `GSE126954_gene_annotation.csv.gz` | 174 KB | Gene annotations (20,222 genes) |

**Key fields in cell annotation**:
- `lineage`: Cell lineage name (e.g., `ABpla`, `MSaapp`)
- `embryo.time`: Developmental time in minutes (0-830)
- `cell.type`: Terminal cell type annotation

**Coverage**: Embryo 0-830 minutes post-fertilization

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

### 4. OpenWorm (Connectome) - Optional

**Source**: [openworm/c302](https://github.com/openworm/c302) on GitHub

| File | Description |
|------|-------------|
| `aconnectome_white_1986_whole.csv` | White et al. 1986 connectome |
| `herm_full_edgelist.csv` | Full hermaphrodite edge list |
| `CElegansNeuronTables.xls` | Neuron information tables |

**Note**: Connectome data is for **adult** neurons (302 cells) and may not directly match embryonic cell names.

---

## Data Matching Analysis

### The Challenge

The three modalities use different naming systems and have incomplete overlap:

```
Packer 2019        WormGUIDES         WormBase
(Transcriptome)    (Spatial)          (Lineage)
   89,701 cells    1,341 cells        2,105 nodes
      ↓                ↓                  ↓
   Lineage names   Cell names         Tree structure
      ↓                ↓                  ↓
      └──────────→ MATCHING ←──────────┘
```

### Lineage Name Format Issues

Packer 2019 lineage annotations have several formats:

| Format | Example | Issue | Frequency |
|--------|---------|-------|-----------|
| Clean | `ABpla` | ✅ Direct match | 10.8% |
| With 'x' | `MSxpappp` | ❌ Branch uncertain | **67%** |
| With '/' | `ABala/ABarp` | ⚠️ Multiple possibilities | 32% |
| Early | `28_cell_or_earlier` | ❌ Cannot resolve | <1% |

**Why 'x'?**: Packer's scRNA-seq was performed on dissociated embryos. Computational lineage inference cannot determine left/right branch decisions, marked as 'x'.

### Matching Statistics

| Metric | Count |
|--------|-------|
| Packer cells with lineage annotation | 47,798 |
| Packer cells with **clean** lineage | 5,143 (10.8%) |
| Unique clean lineage names | 95 |
| Matched to WormGUIDES | **95 (100%)** |

### Time Overlap

```
WormGUIDES:  |████████████████████|
             20              380 minutes

Packer:      |██████████████████████████████████████████|
             0    100   200   300   400   500   600   700   800 minutes
                        ↑ overlap ↑
```

- **Overlap region**: ~20-380 minutes
- **Packer cells in overlap**: 39,528 (83% of annotated cells)

---

## Recommended Approach

### For Prototyping: Use Clean Data

Start with the **5,143 cells** that have clean lineage names:
- 100% match rate to WormGUIDES spatial positions
- 95 unique lineage identities
- Sufficient for model development and validation

```python
# Filter to clean lineage cells
clean_cells = df[
    ~df['lineage'].str.contains('x', na=True) & 
    ~df['lineage'].str.contains('/', na=True) &
    ~df['lineage'].str.contains('earlier', case=False, na=True)
]
```

### For Production: Fuzzy Matching

Expand coverage by resolving ambiguous lineages:

```python
# MSxpappp could be MSapappp or MSpappp
def expand_x(lineage):
    if 'x' not in lineage:
        return [lineage]
    return [lineage.replace('x', 'a', 1), lineage.replace('x', 'p', 1)]
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
├── packer.py            # Packer 2019 downloader
├── openworm.py          # OpenWorm downloader
├── wormguides.py        # WormGUIDES downloader (incl. nuclei)
└── wormbase.py          # WormBase lineage generator
```

## Adding New Data Sources

1. Create a new downloader class inheriting from `BaseDownloader`
2. Define URLs and files in `constants.py`
3. Implement the `download()` method
4. Register in `NemaContextDownloader.download_all()`

---

## References

1. **Packer, J. S., et al. (2019)**. A lineage-resolved molecular atlas of C. elegans embryogenesis at single-cell resolution. *Science*.

2. **Bao, Z., et al. (2006)**. Automated cell lineage tracing in Caenorhabditis elegans. *PNAS*.

3. **Sulston, J. E., et al. (1983)**. The embryonic cell lineage of the nematode Caenorhabditis elegans. *Developmental Biology*.

4. **WormGUIDES**: Bao Lab, Sloan Kettering Institute. https://wormguides.org/

5. **OpenWorm**: Open source project to simulate C. elegans. https://openworm.org/