# NemaContext Documentation

This repo is now organized around one primary objective:

**Validate developmental context learning on real transcriptomic data before pushing whole-embryo generative rollout.**

## Canonical Docs

- `docs/REPO_STRUCTURE.md`: current repository layout and what each area is for.
- `docs/ARCHITECTURE_WHOLE_ORGANISM_AR.md`: downstream autoregressive rollout architecture and why it is no longer the immediate main path.
- `docs/ROADMAP.md`: near-term implementation milestones.
- `docs/GENE_CONTEXT_BASELINE.md`: biological interpretation of the active gene-context baseline and related published work.

## Historical Work

Legacy trimodal/crossmodal and early experiments are kept in code under `examples/legacy/`,
but they are no longer the primary direction.
Archived supporting modules also live under `src/branching_flows/legacy/`,
`src/legacy_model/`, and `src/data/legacy/`.

## Active Reading Order

If you want the current project logic in the right order, read:

1. `docs/ROADMAP.md`
2. `docs/GENE_CONTEXT_BASELINE.md`
3. `docs/REPO_STRUCTURE.md`
4. `docs/ARCHITECTURE_WHOLE_ORGANISM_AR.md`
