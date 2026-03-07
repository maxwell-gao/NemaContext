# NemaContext Documentation

This repo is now organized around one primary objective:

**Build a whole-organism developmental autoregressive model, updated in a diffusion-LLM style.**

## Canonical Docs

- `docs/REPO_STRUCTURE.md`: current repository layout and what each area is for.
- `docs/ARCHITECTURE_WHOLE_ORGANISM_AR.md`: model objective, data contract, and AR + diffusion-style update design.
- `docs/ROADMAP.md`: near-term implementation milestones.
- `docs/GENE_CONTEXT_BASELINE.md`: biological interpretation of the active gene-context baseline and related published work.

## Historical Work

Legacy trimodal/crossmodal and early experiments are kept in code under `examples/legacy/`,
but they are no longer the primary direction.
Archived supporting modules also live under `src/branching_flows/legacy/`,
`src/legacy_model/`, and `src/data/legacy/`.
