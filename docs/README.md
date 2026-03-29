# NemaContext Documentation

This repo is organized around one primary objective:

**Predict whole-organism development from early embryo state, while using real-data context validation as the first executable stage toward that goal.**

## Canonical Docs

- `docs/REPO_STRUCTURE.md`: current repository layout and what each area is for.
- `docs/ROADMAP.md`: merged long-term goal and near-term execution order.
- `docs/GENE_CONTEXT_BASELINE.md`: biological interpretation of the active local developmental prediction baselines, including the patch-set pretext task and the multi-view transition.
- `docs/EXPERIMENT_HISTORY.md`: chronological record of the main context-validation, patch-set, multi-patch, and representation-learning transition experiments.
- `docs/ARCHITECTURE_WHOLE_ORGANISM_AR.md`: whole-embryo architecture as the intended destination of the current line of work.
- `docs/DESIGN_DECISIONS_GENE_CONTEXT_WORLD_MODEL.md`: detailed scientific and architectural reasoning behind the recent embryo future-set, latent world-model, alignment, gene-first, and JiT-style design decisions.

## Historical Work

Legacy trimodal/crossmodal and early experiments are kept in code under `examples/legacy/`,
but they are no longer the primary direction.
Archived supporting modules also live under `src/branching_flows/legacy/`,
`src/legacy_model/`, and `src/data/legacy/`.

## Active Reading Order

If you want the current project logic in the right order, read:

1. `docs/ROADMAP.md`
2. `docs/GENE_CONTEXT_BASELINE.md`
3. `docs/EXPERIMENT_HISTORY.md`
4. `docs/ARCHITECTURE_WHOLE_ORGANISM_AR.md`
5. `docs/DESIGN_DECISIONS_GENE_CONTEXT_WORLD_MODEL.md`
6. `docs/REPO_STRUCTURE.md`

Current active scripts now split into two levels:

- `train_patch_set.py` / `evaluate_patch_set.py`: local state pretext task.
- `train_state_views.py`: shared-encoder multi-view state representation learning.
- `train_masked_state_views.py`: strongest current self-supervised developmental
  state route via masked view, masked future, and masked gene reconstruction.
- `train_embryo_masked_views.py`: active embryo-level self-supervised route via
  masked current and masked future local views of the same embryo state.
- `train_embryo_future_set.py`: strongest current embryo predictive route via
  end-to-end reconstruction-backed MAE-style masked future local-view set
  completion with an internal pred-x local-code bottleneck.

Current biological milestone:

- the broad `dt = 40` multi-view state encoder is the first model in the repo
  whose latent can be argued to have direct biological meaning,
  because it aligns with developmental time and predicts future developmental
  structure better than the broad single-cell control.
- a masked self-supervised extension now also exists:
  `masked view + masked future + masked gene` currently yields the strongest
  self-supervised biological state representation.
- the active embryo-scale representation route is now embryo-level masked
  multi-view modeling with masked future views; this replaced direct embryo
  summary regression as the preferred embryo-state training interface.
- the active embryo-scale predictive route is now reconstruction-backed
  MAE-style future-set completion:
  current context plus visible future parts predict masked future local-view
  sets more reliably than direct global one-step latent prediction.
- later diagnostics now show that the future embryo latent target itself is
  stable under view resampling and permutation; the main remaining problem is
  that cosine-only one-step dynamics do not preserve the biology-readable
  future-latent manifold.
- temporal discrimination, hard-negative discrimination, queue-based
  discrimination, and future-retrieval ranking are all currently ineffective.
- JEPA remains useful as an embedding route, but JEPA-backed future-set and
  JEPA-backed one-step prediction are still weaker than the reconstruction-led
  embryo predictive route.
- the next architectural step is to strengthen MAE-style future-part
  completion rather than returning to direct global latent jumps; SAE remains
  an analysis-only option on frozen latents, not a main training component.
