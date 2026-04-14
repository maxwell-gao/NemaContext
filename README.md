# NemaContext

NemaContext is currently focused on one executable mainline:

**lineage-first multicellular gene dynamics prediction on the *C. elegans* embryo**

## Current Mainline

The active path is now:
- raw `Large2025` transcriptome data
- whole-embryo context
- lineage-conditioned token structure
- dynamics-first future prediction
- worm-native forecasting benchmark

This is the path to read and run first:
- `docs/WORM_MAINLINE.md`
- `docs/WORM_FORECASTING_RESULTS.md`
- `docs/REPO_STRUCTURE.md`
- `examples/whole_organism_ar/train_large2025_lineage_stage1.py`
- `examples/whole_organism_ar/benchmark_worm_dynamics.py`

Current active source modules are intentionally narrow:
- `src/branching_flows/gene_context.py`
- `src/branching_flows/lineage_backbone.py`
- `src/branching_flows/gene_context_shared.py`
- `src/branching_flows/emergent_loss.py`
- `src/data/gene_context_dataset.py`
- `src/data/gene_context_dataset_large2025.py`

## Current Position

The repo is not yet at whole-organism spatial structure prediction.
It is at the prior stage:
- whole-embryo gene dynamics backbone learning
- followed by worm-native forecasting evaluation
- before spatial alignment with WormGUIDES/CShaper

## Mainline Commands

```bash
# Train the current worm mainline backbone
uv run python examples/whole_organism_ar/train_large2025_lineage_stage1.py \
  --device cuda \
  --epochs 20 \
  --batch_size 4 \
  --token_budget 128 \
  --experiment_name raw_large2025_stage1_dyn_e20_b4_t128

# Run the worm-native benchmark
uv run python examples/whole_organism_ar/benchmark_worm_dynamics.py \
  --checkpoint checkpoints/large2025_lineage_stage1/raw_large2025_stage1_dyn_e20_b4_t128/best.pt \
  --split_mode time_transition
```

## Status

Current worm forecasting comparisons show that the mainline lineage-first model
outperforms:
- persistence
- a scNODE-style generic latent-dynamics baseline
- a PRESCIENT-style generic OT baseline

See `docs/WORM_FORECASTING_RESULTS.md` for the current comparison table.

## Historical Work

Older patch, embryo-future-set, autoregressive, and human benchmark side paths
may still exist in the repository, but they are no longer the canonical route.
Legacy material is kept only for reference and should not drive new work by default.
