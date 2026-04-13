# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Our Creed: Discover, Don't Inject

**We do not inject biological priors into the model. We discover biological priors from the data-trained model.**

This principle follows the Bitter Lesson of machine learning in biology: hardcoded assumptions about gene regulatory networks, developmental pathways, or spatial patterns often limit what our models can learn. Instead, we embrace the following approach:

1. **Tabula Rasa Architecture**: The model begins with minimal inductive biases—no hand-crafted gene modules, no predefined lineage hierarchies, no anatomical atlases. The Transformer learns what matters through attention, not through our assumptions.

2. **Let the Data Speak**: Biological structure emerges from training on raw observations. Cell type clustering appears in the latent space. Spatial gradients materialize in the continuous representations. Lineage relationships crystallize through attention patterns.

3. **Interpret After Training**: Once the model has learned from millions of cell-states across developmental time, we interrogate it: Which genes does it attend to when predicting position? How does it represent lineage divergence? The answers are discovered, not prescribed.

4. **Challenge Our Assumptions**: When the model disagrees with textbook biology, we listen. Perhaps the "established" pathway is context-dependent. Perhaps the canonical cell type boundary is more fluid than believed. Data-driven discovery often precedes experimental validation.

In practice, this means: no feature selection based on GO terms, no loss terms enforcing spatial smoothness, no architectural constraints mimicking known developmental trees. We build the capacity; training writes the program.

---

## Coding Style & Structure Constraints

### No Versioning or Descriptive Names (Clean Architecture)
- **No Versioning in Names**: Strictly prohibit the use of `v+number` (e.g., `model_v1.py`, `experiment_v2`) for naming files or directories. 
- **Underscore Restriction**: Do not use more than one underscore (`_`) in any single filename. 
- **Hierarchical Management**: Organize code and logic through directory nesting and file hierarchy rather than descriptive or versioned filenames.
- **Refactoring Rule**: If a file requires complex naming to describe its function, it should be refactored into a subdirectory with a clean `main.py` or `index.py`.
- Strictly use 'uv' for all Python dependency management. Do not use 'pip' or 'conda' commands directly. Any new dependency must be added to 'pyproject.toml' or via 'uv add'.

### No Standalone Files (Architectural Integrity)
- **No Standalone Files**: Strictly forbid the creation of isolated, standalone files that are disconnected from the existing project structure or dependency graph.
- **Structural Belonging**: Every new file MUST belong to a specific module or sub-package. It must be properly integrated into the directory hierarchy and, if applicable, referenced by an `__init__.py` or a parent orchestrator.
- **Dependency Integration**: New scripts must utilize the established internal libraries (e.g., `src.lutflow.*`) and respect the project's entry points. No "one-off" scripts that duplicate logic or bypass the common configuration system.
- **Verification**: Before creating a file, justify its position in the architecture. If it doesn't fit the current tree, refactor the tree first rather than creating a disconnected root-level file.

### Strict File Segregation (Log & Data Isolation)
- **Dedicated Storage**: All generated output files, including logs (`.log`, `.txt`) and data exports (`.csv`, `.json`, `.xlsx`), MUST be stored in dedicated, standalone directories (e.g., `logs/`, `results/`, `data/`).
- **No Mixing**: Strictly prohibit mixing log or CSV files with source code (`.py`) or configuration files in the same directory.
- **Hierarchical Data Folders**: Log and CSV storage must mirror the project's logic or experiment configuration through sub-directories. For example: `results/experiments/ptq/sensitivity/metrics.csv` instead of `src/experiments/metrics.csv`.
- **Clean Workspace**: Source directories must remain "pristine"—only containing logic and necessary assets. Any tool-generated artifacts found in source folders must be moved to their designated data directories immediately.

### Pre-execution Requirements (Ruff Enforcement)
- **Mandatory Lint & Format**: Before executing any script or launching any experiment, you MUST run:
  - `uvx ruff check --fix`
  - `uvx ruff format`
- **Zero Tolerance**: If Ruff reports unfixable errors, stop execution and fix the code before proceeding.

### File Name Uniqueness (Cognitive Optimization)
- **No Generic Filenames**: Strictly forbid generic, repetitive filenames such as `main.py`, `mod.rs`, `__init__.py`, `utils.py`, or `index.ts` in subdirectories.
- **Self-Descriptive Naming**: Every file must have a unique, descriptive name that reflects its specific responsibility within the system. 
- **Namespace-like Naming**: If a file represents a module's core, name it after the module itself. 

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

Three training scripts are available:

```bash
# Train BROT model on Large2025 transcriptome data (requires built AnnData)
uv run python examples/train_nema.py --epochs 30 --batch_size 3 --device cpu

# Train on WormGUIDES 4D spatial dynamics (uses real Sulston lineage tree)
uv run python examples/train_wormguides.py --epochs 30 --stride 10 --device cpu

# Train cross-modal trimodal model (transcriptome + spatial + lineage)
uv run python examples/train_trimodal_crossmodal.py \
    --epochs 100 --batch_size 8 --device cuda \
    --cross_modal_every 2 --augment_spatial

# Key hyperparameters for all:
# --lambda_ot 0.1       # Sinkhorn OT loss weight
# --lambda_mass 0.01    # Mass matching loss weight
# --lambda_energy 0.001 # Energy regularization weight
# --grad_clip 1.0       # Gradient clipping
```

## Discovery & Evaluation

After training, discover biological priors from the model:

```bash
# Extract learned gene-spatial relationships, cell types, trajectories
uv run python examples/discover_priors.py \
    --checkpoint checkpoints_trimodal_crossmodal/best.pt \
    --output discoveries/

# Evaluate modality completion (gene->spatial, spatial->gene)
uv run python examples/evaluate_modality_completion.py \
    --checkpoint checkpoints_trimodal_crossmodal/best.pt \
    --test_mode gene_to_spatial

# Monitor training progress
./monitor.sh
```

## Architecture Overview

### Active Path (`src/branching_flows/`)

The project goal is whole-organism developmental prediction from early embryo
state toward the full embryo.

The current main executable path is the transcriptomic `gene-context` baseline,
which now includes a newer patch-to-patch set-level path. The current active
readout should be understood as the first validated local population-update
approximation to that embryo-scale goal:

- structured local developmental patches built from real transcriptomic
  windows,
- patch-to-patch set prediction as the main active task,
- short-horizon gene-state transition prediction retained as a historical and
  diagnostic baseline,
- auxiliary split/delete heads used cautiously because event supervision is
  still weak,
- context validation, patch scaling, and readout repair first,
- later promotion into larger population updates and embryo-scale rollout.

For local transcriptomic patch video models trained on the current small-data regime:

- do not introduce early pooled frame bottlenecks before temporal prediction,
- keep token-level temporal pathways end-to-end whenever possible,
- use space only to define local patch membership unless a stronger signal is demonstrated empirically.

Key active classes:

- `GeneContextModel` - bidirectional attention model over structured
  transcriptomic context windows
- `SingleCellGeneTimeModel` - control baseline without multi-cell context
- `GeneContextDataset` - transcriptome time-window dataset with anchor, local,
  and global background context construction
- `MultiCellPatchSetModel` / `SingleCellPatchSetModel` - set-level local
  population-state predictors built on the same encoder stack

### BranchingFlows Framework (`src/branching_flows/`)

The BranchingFlows-derived stack remains in the repository as later
infrastructure for variable-cell-count developmental dynamics:

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

## Stage 1 Lineage-First Rule

For raw Large2025 stage-one pretraining, prefer whole-embryo lineage-first snapshots over strict spatial patch construction. Use lineage as a mandatory structural embedding and keep WormGUIDES-derived spatial information as a later-stage prior or refinement signal, not as a sample eligibility gate.
