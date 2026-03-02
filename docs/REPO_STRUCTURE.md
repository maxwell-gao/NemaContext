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
- `train_with_division_supervision.py`: division-focused training variant.
- `autoregressive_simulation.py`: rollout simulation.
- `perturbation_cross_lineage.py`: perturbation evaluation in shared context.
- `test_cross_lineage_attention.py`: validates cross-lineage interaction effects.

### Legacy (`examples/legacy/`)

Trimodal/crossmodal and earlier data-integration training/evaluation scripts.
These are not the default path for new development.
