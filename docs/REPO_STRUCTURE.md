# Repository Structure (Current)

## Top-Level

- `src/`: model and data pipeline source code.
- `examples/whole_organism_ar/`: active scripts for whole-organism autoregressive work.
- `examples/legacy/`: older experiments kept outside the active path.
- `tests/`: integration tests.
- `dataset/`: raw and processed data artifacts.
- `docs/`: current documentation.

## Source Code

- `src/branching_flows/`
  - `autoregressive_model.py`: core whole-organism AR model.
  - `dynamic_cell_manager.py`: split/delete dynamic cell operations.
  - `fusion.py`: active gene/spatial fusion utility.
  - `gene_context.py`: active multi-cell gene-context baseline model.
  - supporting modules from the current BranchingFlows-derived stack.
  - `legacy/`: archived trimodal, lineage-biased, and crossmodal modules kept for legacy scripts only.
- `src/legacy_model/`
  - archived graph, contact-GNN, and multimodal utilities used only by legacy scripts.

- `src/data/`
  - `trajectory_extractor.py`: whole-embryo trajectory extraction.
  - `gene_context_dataset.py`: transcriptome time-window dataset for the active gene-context baseline.
  - `legacy/trajectory_extractor.py`: archived synthetic and per-founder extraction paths.
  - `downloader/`: dataset downloaders.
  - `builder/`: AnnData and integration utilities (legacy + supporting tooling).

## Example Scripts

### Active (`examples/whole_organism_ar/`)

- `train_autoregressive_full.py`: main training script.
- `evaluate_rollout.py`: rollout and perturbation evaluation for the main AR path.
- `train_gene_context.py`: active multi-cell gene-context baseline training.
- `evaluate_gene_context.py`: evaluation for the gene-context baseline.
- `train_spatial_rollout.py`: engineering-only spatial rollout baseline.
- `evaluate_spatial_rollout.py`: evaluation for the spatial baseline.

### Legacy (`examples/legacy/`)

Trimodal/crossmodal and earlier data-integration training/evaluation scripts.
These are outside the active path for new development.

- `whole_organism_ar/`: older autoregressive scripts that depend on synthetic trajectories,
  explicit lineage supervision, founder-centric perturbation analysis, or
  founder-centric demos/visualization, or crossmodal checkpoints.
