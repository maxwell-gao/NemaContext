# NemaContext

NemaContext is focused on one goal:

**Predict whole-organism development from the earliest embryo state, using real transcriptomic context learning as the first validated step toward embryo-scale rollout.**

## Current Direction

- Final target: **whole-embryo developmental prediction from early embryo state toward the full worm embryo**.
- Active training tasks:
  - **patch-to-patch set-level developmental prediction** as the current local-population pretext task,
  - **shared-encoder multi-view state learning** as the active representation-learning stage built on top of patch views,
  - **masked self-supervised state learning with gene reconstruction** as the current strongest local self-supervised route,
  - **embryo-level masked multi-view modeling with masked future views** as the current embryo-scale self-supervised route.
- Immediate question: **can shared encoders learn stable developmental state representations from multiple local views of the same embryo window?**
- Near-term priority: **treat patches as training views, not biological units, and move from patch prediction toward multi-view state representation learning**.
- Later direction: **return to embryo-scale rollout and variable-cell-count generation once the update rule is credible**.

Current strongest biological result:

- the broad-coverage `dt = 40` multi-view multi-cell encoder learns a latent
  that aligns with developmental time and predicts future developmental
  structure better than the matched broad single-cell control.
- the strongest explicitly self-supervised route is now
  `masked view + masked future + masked gene reconstruction`, which preserves
  biological structure and improves several future developmental probes.
- the strongest embryo-level route is now `masked future views`, where the
  current embryo state is trained to recover masked views from both the
  present and the paired future window.
- a first embryo one-step latent predictor now also works on top of that
  backbone: future embryo latent cosine loss is very low, but jointly trained
  probe heads are still weak, and later diagnostics show that pure cosine
  matching does not preserve the biology-readable future-latent manifold.
- temporal discrimination, hard-negative discrimination, queue-based
  discrimination, and future-retrieval ranking have all failed to become
  effective training signals in the current setup.

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

# 4) Train the broad shared-encoder multi-view state baseline
uv run python examples/whole_organism_ar/train_state_views.py \
  --h5ad_path dataset/processed/nema_extended_large2025.h5ad \
  --model_type multi_cell \
  --sampling_strategy spatial_anchor \
  --context_size 256 \
  --global_context_size 32 \
  --dt_minutes 40 \
  --pairwise_spatial_bias \
  --views_per_state 2 \
  --future_views_per_state 1 \
  --epochs 10

# 5) Train the strongest current self-supervised state encoder
uv run python examples/whole_organism_ar/train_masked_state_views.py \
  --h5ad_path dataset/processed/nema_extended_large2025.h5ad \
  --model_type multi_cell \
  --sampling_strategy spatial_anchor \
  --context_size 256 \
  --global_context_size 32 \
  --dt_minutes 40 \
  --pairwise_spatial_bias \
  --samples_per_pair 16 \
  --event_subset none \
  --val_event_subset none \
  --epochs 10

# 6) Train the active embryo-level masked-future encoder
uv run python examples/whole_organism_ar/train_embryo_masked_views.py \
  --h5ad_path dataset/processed/nema_extended_large2025.h5ad \
  --model_type multi_cell \
  --context_size 256 \
  --global_context_size 32 \
  --dt_minutes 40 \
  --views_per_embryo 8 \
  --future_views_per_embryo 8 \
  --samples_per_pair 16 \
  --pairwise_spatial_bias \
  --epochs 10

# 7) Train embryo one-step latent prediction on top of the best embryo backbone
uv run python examples/whole_organism_ar/train_embryo_one_step.py \
  --backbone_checkpoint checkpoints_embryo_masked_views/best.pt \
  --context_size 256 \
  --global_context_size 32 \
  --dt_minutes 40 \
  --views_per_embryo 8 \
  --future_views_per_embryo 8 \
  --samples_per_pair 16 \
  --freeze_backbone \
  --epochs 10

# 8) Historical token-level baseline remains available for diagnosis
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
starts with smaller real-data update problems before rollout claims.
The current embryo-scale step is masked multi-view reconstruction with masked
future views, followed by latent-first embryo one-step prediction, not more
patch-level contrastive tuning and not direct embryo summary regression.
Current embryo one-step diagnosis is also sharper than before:
the future latent target is stable under view resampling and permutation,
but the latent geometry is too concentrated for cosine alone to enforce
biology-readable dynamics.
