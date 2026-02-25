# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Management

This project uses `uv` for Python package management. Key commands:

```bash
# Sync dependencies (installs from pyproject.toml)
uv sync

# Run a Python script with dependencies
uv run python <script.py>

# Run with development dependencies (pytest, matplotlib, etc.)
uv run --group dev python <script.py>
```

## Data Pipeline

The project requires downloading and integrating three modalities of C. elegans embryo data:

```bash
# Download core datasets (Large2025 transcriptome + WormGUIDES spatial + WormBase lineage)
uv run python -m src.data.downloader --source core

# Download specific sources
uv run python -m src.data.downloader --source large2025   # ~1.2GB
uv run python -m src.data.downloader --source wormguides  # ~25MB
uv run python -m src.data.downloader --source wormbase    # ~600KB

# Build integrated AnnData (complete = cells with all 3 modalities, ~3.8k cells)
uv run python examples/build_anndata.py --variant complete

# Build extended dataset (~242k cells with modality availability flags)
uv run python examples/build_anndata.py --variant extended
```

## Training

Two main training scripts exist:

```bash
# Train BROT model on Large2025 transcriptome data (requires built AnnData)
uv run python examples/train_nema.py --epochs 30 --batch_size 3 --device cpu

# Train on WormGUIDES 4D spatial dynamics (uses real Sulston lineage tree)
uv run python examples/train_wormguides.py --epochs 30 --stride 10 --device cpu

# Key hyperparameters for both:
# --lambda_ot 0.1       # Sinkhorn OT loss weight
# --lambda_mass 0.01    # Mass matching loss weight
# --lambda_energy 0.001 # Energy regularization weight
# --grad_clip 1.0       # Gradient clipping
```

## Architecture Overview

### BranchingFlows Framework (`src/branching_flows/`)

The core generative modeling framework extends flow matching to variable-length sequences:

- **Core abstraction**: `BranchingState` - batched state container with masks for padding (`padmask`), flow evolution (`flowmask`), and branching/deletion permissions (`branchmask`)
- **Flow process**: `CoalescentFlow` wraps base processes (OUFlow, DiscreteInterpolatingFlow) with split/deletion handling
- **Bridge construction**: `branching_bridge()` creates training targets by sampling forests backward in time and interpolating
- **Loss components**: Per-element losses (continuous MSE, discrete CE, split Bregman-Poisson, deletion BCE) + distributional losses (Sinkhorn, mass matching, energy)

Key classes:
- `NemaFlowModel` - Transformer with RoPE, adaLN-Zero conditioning, four output heads (continuous, discrete, split, deletion)
- `NemaDataset` / `WormGUIDESDataset` - Load transcriptome or spatial data as `SampleState` objects

### Data Integration (`src/data/`)

- **Downloader** (`downloader/`): Modular downloaders for GEO (Large2025, Packer2019), WormGUIDES, WormBase, OpenWorm
- **AnnData Builder** (`builder/`): `TrimodalAnnDataBuilder` integrates expression (MTX), spatial (nuclei files), and lineage (JSON) into AnnData with `modality_mask` column (bitmask: 1=transcriptome, 2=spatial, 4=lineage)

### Spatial/Contact Models (`src/model/`)

- `SpatialGraphBuilder` - Build KNN/Delaunay graphs from WormGUIDES coordinates
- `ContactGNN` - GNN for cell-cell contact prediction (CShaper-based)
- `MultimodalDataset` - Combine transcriptome + spatial + lineage modalities

## Development Notes

- Python 3.12+ required
- No test suite currently (examples/ contain manual test scripts)
- Checkpoints saved to `checkpoints/` (Large2025) or `checkpoints_wg/` (WormGUIDES)
- `.gitignore` excludes `dataset/`, `checkpoints*/`, `ref/`, `uv.lock`
- Large2025 is the recommended transcriptome source; Packer2019 is legacy

## Common Issues

- Missing data files: Run downloader first, then check `dataset/raw/<source>/` exists
- AnnData not found: Build with `examples/build_anndata.py` before training
- Memory issues: Reduce `--batch_size` or `--n_hvg` (number of highly variable genes)
