# NemaContext

NemaContext is focused on one goal:

**Predict whole-organism development from the earliest embryo state, using real transcriptomic context learning as the first validated step toward embryo-scale rollout.**

## Current Direction

- Final target: **whole-embryo developmental prediction from early embryo state toward the full worm embryo**.
- Active training task: **anchor-centered multi-cell gene-context prediction** as the first validated local population-update problem.
- Immediate question: **does structured multi-cell context improve short-horizon developmental updates?**
- Near-term priority: **strengthen event supervision and expand from local context toward larger population context**.
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
# 1) Train the active multi-cell gene-context baseline
uv run python examples/whole_organism_ar/train_gene_context.py \
  --h5ad_path dataset/processed/nema_extended_large2025.h5ad \
  --sampling_strategy spatial_anchor \
  --context_size 64 \
  --global_context_size 16 \
  --epochs 10

# 2) Evaluate whether context is actually used
uv run python examples/whole_organism_ar/evaluate_gene_context.py \
  --checkpoint checkpoints_gene_context/best.pt \
  --context_ablation full \
  --output result/gene_context/evaluation.json

# 3) Compare with anchor-only evaluation
uv run python examples/whole_organism_ar/evaluate_gene_context.py \
  --checkpoint checkpoints_gene_context/best.pt \
  --context_ablation anchor_only \
  --output result/gene_context/evaluation_anchor_only.json

Founder-specific perturbation, visualization, and demo scripts are retained under
`examples/legacy/whole_organism_ar/`, not in the active path.
Synthetic and per-founder trajectory generation is archived under
`src/data/legacy/trajectory_extractor.py`.
```

Whole-organism autoregressive rollout remains in the repository as the intended
destination of the current line of work, but the active evidence path still
starts with smaller real-data update problems before embryo-scale claims.
