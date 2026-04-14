# Repository Structure (Current)

## Top-Level

- `src/`: active worm-mainline source plus a reduced set of raw-data utilities.
- `examples/whole_organism_ar/`: active worm-mainline scripts only.
- `examples/legacy/`: archived experiments and historical entrypoints.
- `tests/`: worm-mainline tests only.
- `dataset/`: raw and processed data artifacts.
- `docs/`: current and historical documentation.

## Active Source Surface

The active code path is intentionally small.

### `src/branching_flows/`
Only these modules are part of the current mainline:
- `gene_context.py`: narrow public export for the active worm model.
- `gene_context_patch.py`: implementation of `LineageWholeEmbryoModel` and shared transformer-based token dynamics.
- `gene_context_shared.py`: active output dataclasses.
- `emergent_loss.py`: Sinkhorn / OT-style losses used by the worm benchmark and training path.
- `autoregressive_model.py`: retained only for `TransformerBlockAutoregressive`, which is still used by the active model.

Everything else that previously lived under `src/branching_flows/` has been removed from the active path.

### `src/data/`
The active data path is also reduced:
- `gene_context_dataset.py`: narrow public export for the active Large2025 dataset.
- `gene_context_dataset_large2025.py`: whole-embryo lineage-first dataset used by the worm mainline.
- `builder/expression_loader.py`: raw Large2025 expression loading.
- `builder/lineage_encoder.py`: lineage parsing and encoding.
- `downloader/`: retained data download interfaces for raw sources and later spatial alignment stages.

Old patch/embryo dataset modules, processors, and legacy trajectory utilities are no longer part of the active source tree.

### `src/legacy_model/`
Archived graph/contact/multimodal utilities kept only as historical material.
They are not part of the current worm mainline.

## Active Example Scripts

### `examples/whole_organism_ar/`
Current active scripts:
- `train_large2025_lineage_stage1.py`
- `benchmark_worm_dynamics.py`
- `benchmark_worm_scnode.py`
- `benchmark_worm_prescient.py`

These are the only scripts that should define new work by default.

### `examples/legacy/`
Older patch, embryo, autoregressive, trimodal, and historical evaluation scripts.
These remain for reference only and should be treated as archive material unless explicitly revived.

## Interpretation Rule

The repo is currently organized around one active line:
- lineage-first whole-embryo gene dynamics on raw Large2025,
- evaluated with worm-native forecasting metrics,
- before later WormGUIDES/CShaper spatial alignment.

Anything outside that path is archive or historical context.
