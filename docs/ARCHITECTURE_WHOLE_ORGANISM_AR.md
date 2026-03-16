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
  local population-state prediction,
- multi-patch coverage -> useful as multiple local views of one regional
  state, not as a final ontology of patch entities,
- next transition -> shared-encoder multi-view state learning rather than
  stronger patch hierarchy.

So the active path is not conceptually separate from whole-organism AR.
It is the smallest version of it that can currently be validated on real data.

The most important recent shift is that the main evidence path is no longer
best summarized as "predict one anchor cell from context".

It is now better summarized as:

- encode a local developmental patch at time `t`,
- predict the next local developmental patch at `t + dt`,
- compare single-cell and multi-cell patch encoders,
- scale the patch until larger context begins to matter.

The next shift after that is now also concrete:

- treat many local patches as views of one embryo state rather than as stable
  patch entities,
- train embryo-scale latents by reconstructing masked current and masked
  future views of that embryo state,
- only then promote the learned embryo latent into embryo one-step dynamics.

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

Current embryo-scale precursor:

- direct embryo summary regression was tested and failed,
- embryo-level masked multi-view modeling works,
- adding masked future views materially improves embryo-level biological
  probes, especially future founder composition, future cell-type composition,
  future lineage-depth statistics, future spatial extent, and split-fraction
  alignment.

So the present embryo-scale contract should be read as:

- local views are observations,
- embryo latent is the state,
- masked future-view reconstruction is the best current bridge from local
  self-supervision to embryo-scale dynamics.

A first embryo one-step latent predictor is now also in place:

- frozen masked-future backbone,
- current embryo latent predicts future embryo latent,
- future latent cosine loss becomes very small,
- but jointly trained developmental probe heads are still weak.
- further diagnostics now show that this is not mainly because the future
  target is noisy:
  repeated future-view resampling produces nearly identical future latents,
  and pure view permutation barely changes them.
- the remaining failure mode is geometric:
  cosine-only one-step matching can produce a future latent that is close in
  angle while still leaving the biology-readable future-latent manifold.

So the current architecture lesson is:

- embryo latent dynamics are now more mature than embryo probe decoding,
- future latent prediction should be treated as the primary embryo one-step
  objective,
- developmental probes should remain a secondary readout or a later frozen
  probe stage until that path becomes stable,
- and future work on embryo dynamics should target manifold-preserving latent
  transitions rather than more direct probe co-training.

A minimal embryo JEPA route now also exists on the same embryo-view interface:

- visible current views define the context,
- masked future views define the target,
- an EMA teacher provides target embeddings for a lightweight predictor.

Its first instability was traced to per-batch target whitening in a highly
concentrated latent space. After switching to layer-normalized target matching,
the minimal JEPA route now smoke-trains stably and can be treated as an
exploratory geometry-focused alternative to the active masked-future
reconstruction route.

The newest predictive bridge is no longer direct one-step embryo-latent
prediction, but embryo-scale future-part completion.

In practice this now means:

- keep the masked-future embryo backbone as the state encoder,
- treat future local views as the parts that need to be completed,
- use current views plus visible future parts to reconstruct masked future
  local-view sets,
- only then aggregate those predicted future parts into larger embryo-level
  summaries.

This route matters because the repository has now tested three embryo
predictive contracts:

- direct global one-step latent prediction,
- slot-style future-set prediction,
- MAE-style future-part completion with a stronger decoder.

The current empirical ordering is:

- reconstruction-backed MAE future-set completion is best,
- reconstruction-backed slot future-set prediction is weaker but related,
- JEPA-backed one-step and JEPA-backed future-set prediction remain useful as
  representation diagnostics, not yet as the best embryo predictive contract.

## Interpretation Rule

If a result only shows that a focal anchor can be predicted one step ahead, it
is evidence for a component of this architecture.

It is not yet evidence that whole-organism developmental prediction has been
solved.
