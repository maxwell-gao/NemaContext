# NemaContext

NemaContext is focused on one goal:

**Predict whole-organism development from the earliest embryo state, using real transcriptomic context learning as the first validated step toward embryo-scale rollout.**

## Current Direction

- Final target: **whole-embryo developmental prediction from early embryo state toward the full worm embryo**.
- Active training task: **patch-to-patch set-level developmental prediction** as the current local-population pretext task.
- Immediate question: **can shared encoders learn stable developmental state representations from multiple local views of the same embryo window?**
- Near-term priority: **treat patches as training views, not biological units, and move from patch prediction toward multi-view state representation learning**.
- Later direction: **return to embryo-scale rollout and variable-cell-count generation once the update rule is credible**.

## Repository Layout

- `src/branching_flows/`: core modeling code (`autoregressive_model.py`, `dynamic_cell_manager.py`, `fusion.py`)
- `src/branching_flows/legacy/`: archived trimodal, lineage-biased, and crossmodal modules
- `src/legacy_model/`: archived graph and multimodal utilities used only by legacy workflows
- `src/data/`: transcriptome window datasets, trajectory extraction, and data pipeline
- `examples/whole_organism_ar/`: active scripts
- `examples/legacy/`: previous-generation experiments
- `docs/`: canonical architecture and roadmap

See:

- `docs/ARCHITECTURE_WHOLE_ORGANISM_AR.md`
- `docs/REPO_STRUCTURE.md`
- `docs/ROADMAP.md`

## Quick Start

```bash
# 1) Train the active multi-cell patch-set baseline
uv run python examples/whole_organism_ar/train_patch_set.py \
  --h5ad_path dataset/processed/nema_extended_large2025.h5ad \
  --model_type multi_cell \
  --sampling_strategy spatial_anchor \
  --context_size 256 \
  --global_context_size 16 \
  --spatial_input_mode relative_position \
  --pairwise_spatial_bias \
  --epochs 10

# 2) Evaluate the active patch-set baseline
uv run python examples/whole_organism_ar/evaluate_patch_set.py \
  --checkpoint checkpoints_patch_set/best.pt \
  --output result/gene_context/evaluation_patch_set.json

# 3) Compare with anchor-only ablation
uv run python examples/whole_organism_ar/evaluate_patch_set.py \
  --checkpoint checkpoints_patch_set/best.pt \
  --context_ablation anchor_only \
  --output result/gene_context/evaluation_patch_set_anchor_only.json

# 4) Historical token-level baseline remains available for diagnosis
uv run python examples/whole_organism_ar/train_gene_context.py \
  --h5ad_path dataset/processed/nema_extended_large2025.h5ad \
  --sampling_strategy spatial_anchor \
  --context_size 64 \
  --global_context_size 16 \
  --epochs 10
```

Founder-specific perturbation, visualization, and demo scripts are retained under
`examples/legacy/whole_organism_ar/`, not in the active path.
Synthetic and per-founder trajectory generation is archived under
`src/data/legacy/trajectory_extractor.py`.

Whole-organism autoregressive rollout remains in the repository as the intended
destination of the current line of work, but the active evidence path still
starts with smaller real-data update problems before embryo-scale claims.
