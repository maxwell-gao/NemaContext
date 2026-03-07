# Roadmap: Developmental Context Before Generative Rollout

## Goal

Build a developmental model that learns embryo context from real data and only
later promotes that context model into a variable-length generator.

The ordering matters:

1. first prove the model uses multi-cell context,
2. then make split/delete supervision credible,
3. only then build whole-embryo generative dynamics on top.

This is a tighter version of the original roadmap. The main correction is that
`Branching Flows` is not the current training core. It is the future generative
shell for a context model that first has to be validated on real transcriptomic
supervision.

## Current Position

What is already established:

- real transcriptome windows can be sampled from `nema_extended_large2025.h5ad`,
- the active baseline predicts short-horizon gene-state change from multi-cell
  transcriptomic context,
- anchor-centered structured context now exists,
- the model uses context rather than only anchor-local signal,
- adding global background context gives a small but real gain on validation
  loss,
- autoregressive rollout code and BranchingFlows-derived dynamic machinery
  remain available as downstream infrastructure.

What remains weak:

- split/delete supervision is still sparse and weakly constructed,
- event precision/recall are not yet informative in many validation slices,
- whole-embryo rollout would currently rest on labels that are not strong
  enough,
- Branching Flows-style generation is therefore ahead of the supervision.

## Phase 1: Context Validation

Objective: prove that multi-cell context is genuinely used and improves
prediction beyond single-cell conditioning.

This is the active main path.

Tasks:

- keep transcriptome as the primary developmental state,
- use anchor-centered context windows as the default input format,
- compare local-only and local-plus-global context construction,
- compare single-cell and multi-cell baselines under matched data construction,
- run context ablations such as `full` vs `anchor_only`,
- run perturbation tests such as token dropout, neighborhood shuffling, and
  context corruption.

Definition of done:

- multi-cell context consistently outperforms single-cell controls,
- `full` context outperforms `anchor_only` across repeated runs,
- larger or more structured context improves validation loss in a stable way,
- the model's gains can be attributed to context access rather than only to a
  larger backbone.

Status:

- `src/data/gene_context_dataset.py` supports structured anchor-centered
  context,
- `src/branching_flows/gene_context.py` uses plain bidirectional attention with
  lightweight context-role and distance labels,
- current experiments show that `full` context beats `anchor_only`,
- current experiments also show a small gain from adding global background
  tokens.

## Phase 2: Supervision Repair

Objective: make event supervision strong enough that dynamic modeling is worth
trusting.

This phase is higher priority than rollout or full generative training.

Tasks:

- measure true split/delete positive coverage under the current windowing
  scheme,
- separate unmatched cells from genuine disappearance as much as the data allow,
- improve future matching so that event targets are less dominated by weak
  fallback matches,
- construct evaluation subsets enriched for meaningful split/delete events,
- report event metrics on subsets where positive labels are actually present.

Definition of done:

- split/delete labels have enough support to produce meaningful held-out event
  metrics,
- event losses correlate with actual detection quality rather than only class
  imbalance,
- event supervision is no longer the bottleneck for dynamic modeling.

Why this phase exists:

- current context experiments are scientifically useful,
- current event labels are not yet strong enough to justify ambitious rollout
  claims,
- without this repair, generative dynamics would be technically runnable but
  scientifically under-grounded.

## Phase 3: Context-to-Dynamics Transition

Objective: turn the validated context predictor into a stable short-horizon
  dynamic model.

Tasks:

- keep the context encoder fixed in spirit: bidirectional attention over
  structured multi-cell windows,
- predict short-horizon future state under repeated application,
- add dynamic event handling only after event supervision is credible,
- evaluate multi-step degradation, cell-count drift, and event-balance drift,
- avoid heuristic inference rules as the primary way to make rollout appear
  stable.

Definition of done:

- repeated short-horizon rollout is measurably more stable than naive
  independent-cell prediction,
- event balance remains plausible over multiple steps,
- embryo-level context does not collapse immediately under autoregressive use.

## Phase 4: Branching Flows Integration

Objective: use Branching Flows as the variable-length generative framework
after the context model and event supervision are both validated.

Role of Branching Flows in this roadmap:

- not the current primary learner,
- yes as the long-horizon generative mechanism,
- yes as the formal language for split/delete/base-process coupling,
- no as the first thing to optimize before supervision is ready.

Tasks:

- define a biologically defensible `Z` construction for developmental data,
- connect the base process to the learned transcriptomic context predictor,
- move split/delete from weak auxiliary heads toward true generative events,
- compare plain autoregressive rollout to Branching Flows-style rollout only
  after both are trained on credible supervision.

Definition of done:

- Branching Flows integration improves variable-cell-count generation or
  long-horizon stability,
- gains hold on real-data evaluation rather than synthetic demonstrations,
- the method is still consistent with `Discover, Don't Inject`.

## Phase 5: Spatial Reintegration

Objective: reintroduce spatial information as auxiliary context once the
transcriptomic context path is stable.

Tasks:

- use space as optional context, readout, or evaluation axis,
- test whether transcriptomic latents recover spatial organization,
- compare local transcriptomic context alone versus transcriptomic context plus
  spatial support,
- avoid making spatial coordinates the main developmental state.

Definition of done:

- spatial information improves prediction, calibration, or interpretability,
- transcriptomic context remains the main driver,
- no explicit anatomical prior is required.

## Immediate Next Steps

1. Promote `spatial_anchor` structured context to the default experimental
   interface.
2. Run a clean context-usage matrix:
   `context_size` sweep, `global_context_size` sweep, `full` vs `anchor_only`,
   and single-cell vs multi-cell controls.
3. Quantify split/delete positive coverage and identify where event supervision
   is failing.
4. Build event-enriched validation subsets before claiming dynamic progress.
5. Delay any major Branching Flows training push until Phase 2 is in better
   shape.

## Non-Goals

- no active claims based only on visually plausible rollout,
- no promotion of Branching Flows to the main path before supervision is ready,
- no injected lineage embeddings as model input,
- no architectural shortcuts that encode the developmental tree directly,
- no regression to synthetic data as the main evidence source.
