# Repository Structure (Current)

## Top-Level

- `src/`: model and data pipeline source code.
- `examples/whole_organism_ar/`: active scripts for the current local-update validation path and later embryo-scale rollout work.
- `examples/legacy/`: older experiments kept outside the active path.
- `tests/`: integration tests.
- `dataset/`: raw and processed data artifacts.
- `docs/`: current documentation.

Interpretation:

- the final project target is whole-organism developmental prediction,
- the current active evidence path is a smaller transcriptomic population-update
  problem plus embryo-level self-supervised state learning,
- the repo therefore contains both active local-update tooling and embryo-scale
  rollout infrastructure.

## Source Code

- `src/branching_flows/`
  - `gene_context.py`: active multi-cell gene-context baseline model.
  - `autoregressive_model.py`: whole-organism AR model intended for later embryo-scale rollout phases.
  - `dynamic_cell_manager.py`: split/delete dynamic cell operations used by downstream rollout experiments.
  - `fusion.py`: supporting gene/spatial fusion utility.
  - supporting modules from the current BranchingFlows-derived stack.
  - `legacy/`: archived trimodal, lineage-biased, and crossmodal modules kept for legacy scripts only.
- `src/legacy_model/`
  - archived graph, contact-GNN, and multimodal utilities used only by legacy scripts.

- `src/data/`
  - `gene_context_dataset.py`: transcriptome time-window dataset for the active context-validation path.
  - `trajectory_extractor.py`: whole-embryo trajectory extraction for downstream engineering and rollout diagnostics.
  - `legacy/trajectory_extractor.py`: archived synthetic and per-founder extraction paths.
  - `downloader/`: dataset downloaders.
  - `builder/`: AnnData and integration utilities (legacy + supporting tooling).

## Example Scripts

### Active (`examples/whole_organism_ar/`)

- `train_gene_context.py`: active multi-cell gene-context baseline training.
- `evaluate_gene_context.py`: evaluation and context ablation for the gene-context baseline.
- `train_gene_single_cell.py`: single-cell control baseline for context comparison.
- `evaluate_gene_single_cell.py`: evaluation for the single-cell control.
- `train_patch_set.py`: local patch-to-patch set-level pretext task.
- `evaluate_patch_set.py`: local patch-set evaluation and ablation.
- `train_state_views.py`: shared-encoder multi-view local state learning.
- `train_masked_state_views.py`: strongest current local self-supervised route via masked view, masked future, and masked gene reconstruction.
- `train_embryo_masked_views.py`: active embryo-level masked multi-view route with masked future views.
- `train_embryo_future_set.py`: strongest current embryo predictive route via end-to-end reconstruction-backed MAE-style future local-view set completion with an internal pred-x local-code bottleneck.
- `train_embryo_state.py`: direct embryo summary regression baseline retained as a scaffold and comparison point.
- `evaluate_embryo_state.py`: embryo summary-regression evaluation helper.
- `train_autoregressive_full.py`: downstream whole-organism AR training path for later embryo-scale phases.
- `evaluate_rollout.py`: rollout and perturbation evaluation for the downstream AR path.
- `train_spatial_rollout.py`: engineering-only spatial rollout baseline.
- `evaluate_spatial_rollout.py`: evaluation for the spatial baseline.

### Legacy (`examples/legacy/`)

Trimodal/crossmodal and earlier data-integration training/evaluation scripts.
These are outside the active path for new development.

- `whole_organism_ar/`: older autoregressive scripts that depend on synthetic trajectories,
  explicit lineage supervision, founder-centric perturbation analysis, or
  founder-centric demos/visualization, or crossmodal checkpoints.
