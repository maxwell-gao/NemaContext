# Whole-Organism AR Architecture

## Status

This architecture is the intended destination of the project, not a discarded
branch.

The final target remains whole-organism developmental prediction from very
early embryo state toward the later embryo.

What changed is only the execution order:

- we do not start by claiming full rollout,
- we first learn and validate smaller update rules that can later be promoted
  into this embryo-scale architecture.

So this document should be read as:

- the long-term system contract,
- partially implemented infrastructure,
- downstream of the current validation phase,
- but still the architecture the project is trying to grow back into.

## What This Architecture Is For

Model development as a repeated embryo-state update:

- state at time `t`: all currently alive cells in one shared embryo context,
- update: predict the next embryo state from the current embryo state,
- support dynamic population changes through division and disappearance events.

Formally:

`X_{t+1} = X_t + f_theta(X_t, c_t)`

where `X_t` is the embryo-scale state and `c_t` is optional conditioning such
as developmental time, schedule, or noise level.

The key point is that context is not fixed background.
It changes together with the cells.

That is why the final project cannot stop at anchor-only prediction.

## Why The Current Active Work Is Smaller

The current `gene-context` baseline exists because full embryo dynamics are not
yet easy to supervise directly from snapshot transcriptomics.

Current blockers:

- `split/delete` targets remain weaker than the gene-state target,
- multi-step reuse amplifies supervision error,
- a biologically closed population update rule is not yet fully established,
- local validation is still easier to interpret than embryo-scale rollout.

Because of this, the present order is:

1. learn a local one-step population update proxy,
2. move from token matching to patch-to-patch set prediction,
3. repair event supervision where still useful as auxiliary diagnosis,
4. expand context radius,
4. return to this embryo-scale architecture with stronger ingredients.

## Whole-Organism State Contract

If this path is active, each timestep should contain the embryo as one shared
population state.

Core fields:

- `n_cells`
- `cell_names`
- `genes`
- optional `positions` in embryo coordinates
- event labels such as `divisions` and optional `deaths`

Optional metadata may include:

- `founders`
- `founder_ids`
- evaluation-only lineage bookkeeping

These metadata help bookkeeping and evaluation.
They are not meant to become shortcut inputs that hard-code developmental
structure.

## Relationship To The Active Gene-Context Path

The active `gene-context` work should be interpreted as the first validated
approximation to this architecture.

Mapping:

- anchor-centered window -> small local slice of embryo state,
- one-step gene update -> local population update proxy,
- patch-to-patch set prediction -> local population-state update proxy,
- split/delete heads -> crude event proxies retained mainly for diagnosis,
- structured local-plus-global context -> compressed embryo-context prototype,
- patch-set scaling -> first direct test of whether larger context improves
  local population-state prediction.

So the active path is not conceptually separate from whole-organism AR.
It is the smallest version of it that can currently be validated on real data.

The most important recent shift is that the main evidence path is no longer
best summarized as "predict one anchor cell from context".

It is now better summarized as:

- encode a local developmental patch at time `t`,
- predict the next local developmental patch at `t + dt`,
- compare single-cell and multi-cell patch encoders,
- scale the patch until larger context begins to matter.

## Current Core Modules

- `src/branching_flows/autoregressive_model.py`
  - transformer backbone for embryo-level autoregressive updates
- `src/branching_flows/dynamic_cell_manager.py`
  - event sampling and application
- `src/data/trajectory_extractor.py`
  - whole-embryo trajectory extraction for engineering diagnostics

Archived synthetic and per-founder extraction paths live in
`src/data/legacy/trajectory_extractor.py` and are not part of the current main
scientific evidence path.

## Re-entry Criteria

This architecture becomes the active main path again when:

- the one-step local population update rule is strong enough,
- event supervision is no longer dominated by label construction artifacts,
- context can be expanded beyond anchor-local windows,
- multi-step reuse can be evaluated quantitatively,
- embryo-scale rollout becomes scientifically interpretable rather than only
  visually plausible.

## Interpretation Rule

If a result only shows that a focal anchor can be predicted one step ahead, it
is evidence for a component of this architecture.

It is not yet evidence that whole-organism developmental prediction has been
solved.
