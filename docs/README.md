# NemaContext Documentation

This repo is organized around one primary objective:

**Predict whole-organism development from early embryo state, while using real-data context validation as the first executable stage toward that goal.**

## Canonical Docs

- `docs/REPO_STRUCTURE.md`: current repository layout and what each area is for.
- `docs/ROADMAP.md`: merged long-term goal and near-term execution order.
- `docs/GENE_CONTEXT_BASELINE.md`: biological interpretation of the active local developmental prediction baselines, including the patch-set pretext task and the multi-view transition.
- `docs/EXPERIMENT_HISTORY.md`: chronological record of the main context-validation, patch-set, multi-patch, and representation-learning transition experiments.
- `docs/ARCHITECTURE_WHOLE_ORGANISM_AR.md`: whole-embryo architecture as the intended destination of the current line of work.
- `docs/REPO_STRUCTURE.md`: current repository layout and what each area is for.

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
5. `docs/REPO_STRUCTURE.md`

Current active scripts now split into two levels:

- `train_patch_set.py` / `evaluate_patch_set.py`: local state pretext task.
- `train_state_views.py`: shared-encoder multi-view state representation learning.
- `train_masked_state_views.py`: strongest current self-supervised developmental
  state route via masked view, masked future, and masked gene reconstruction.

Current biological milestone:

- the broad `dt = 40` multi-view state encoder is the first model in the repo
  whose latent can be argued to have direct biological meaning,
  because it aligns with developmental time and predicts future developmental
  structure better than the broad single-cell control.
- a masked self-supervised extension now also exists:
  `masked view + masked future + masked gene` currently yields the strongest
  self-supervised biological state representation.
- temporal discrimination, hard-negative discrimination, queue-based
  discrimination, and future-retrieval ranking are all currently ineffective.
- the next architectural step is embryo-scale state aggregation from multiple
  local views; SAE is currently an analysis-only option on frozen latents, not
  a main training component.
