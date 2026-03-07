# Repository Structure (Current)

## Top-Level

- `src/`: model and data pipeline source code.
- `examples/whole_organism_ar/`: active scripts for whole-organism autoregressive work.
- `examples/legacy/`: older experiments kept for reference only.
- `tests/`: integration tests.
- `dataset/`: raw and processed data artifacts.
- `docs/`: current documentation.

## Source Code

- `src/branching_flows/`
  - `autoregressive_model.py`: core whole-organism AR model.
  - `dynamic_cell_manager.py`: split/delete dynamic cell operations.
  - `cross_lineage_probe.py`: cross-lineage influence analysis.
  - supporting modules from prior BranchingFlows implementation.

- `src/data/`
  - `trajectory_extractor.py`: whole-embryo trajectory extraction.
  - `downloader/`: dataset downloaders.
  - `builder/`: AnnData and integration utilities (legacy + supporting tooling).

## Example Scripts

### Active (`examples/whole_organism_ar/`)

- `train_autoregressive_full.py`: main training script.
- `evaluate_rollout.py`: rollout and perturbation evaluation for the main AR path.
- `train_spatial_rollout.py`: engineering-only spatial rollout baseline.
- `evaluate_spatial_rollout.py`: evaluation for the spatial baseline.

### Legacy (`examples/legacy/`)

Trimodal/crossmodal and earlier data-integration training/evaluation scripts.
These are not the default path for new development.

- `whole_organism_ar/`: older autoregressive scripts that depend on synthetic trajectories,
  explicit lineage supervision, founder-centric perturbation analysis, or
  founder-centric demos/visualization, or crossmodal checkpoints.
