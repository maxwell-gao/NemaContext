# Roadmap: From Zygote To Whole-Embryo Development

## Goal

The final goal of this project is not a local context benchmark.

It is:

> predict the development of a whole organism from the zygote onward,
> with context growing from one cell to the full embryo, eventually on the
> order of 1k cells.

That long-term goal stays fixed.

What changes across phases is only the scale of the state we can model and
validate.

So this roadmap combines two truths:

1. the final target is whole-embryo developmental prediction,
2. the current executable work must still proceed through smaller,
   scientifically checkable subproblems.

## Project Logic

The project should be read as a scale-expansion program.

- start from real transcriptomic snapshot windows,
- learn a local population update rule,
- expand the context radius from anchor-local to embryo-scale,
- move from one-step prediction to repeated population updates,
- only then claim whole-embryo developmental rollout.

In that framing:

- the active `gene-context` work is not the end goal,
- it is the first validated approximation to an embryo-scale update model,
- whole-organism autoregressive and Branching Flows modules remain the intended
  re-entry point once the update rule is credible.

## Current Position

What is already established:

- real transcriptome windows can be sampled from `nema_extended_large2025.h5ad`,
- structured anchor-centered context exists,
- the model can use multi-cell context rather than only anchor-local signal,
- a one-step transcriptomic update task can be trained end-to-end on real data,
- whole-organism autoregressive infrastructure and variable-cell-count code are
  already present in the repository.

What remains weak:

- current work still validates local or window-scale updates, not embryo-scale
  developmental prediction,
- `split/delete` supervision is still weaker than the gene-state target,
- multi-step dynamics are not yet biologically closed under changing context,
- the active benchmark can still absorb too much attention relative to the
  final whole-embryo objective.

## Phase 1: Local Population Update

Objective: learn the smallest useful developmental update rule from real data.

This is the current active phase.

The question is no longer:

> can we predict one cell far into the future?

It is:

> given a local developmental population at time `t`, can we improve the
> prediction of the next short-horizon update at `t + dt`?

Tasks:

- keep transcriptome as the primary developmental state,
- use structured context windows as the default interface,
- compare single-cell and multi-cell predictors under matched data
  construction,
- verify that `full` context beats `anchor_only`,
- identify which contexts help `split`, which help `gene`, and which do not
  help at all,
- treat `delete` as a weak auxiliary target unless proven otherwise.

Definition of done:

- context usage is stable and measurable,
- multi-cell gains are reproducible on at least some biologically meaningful
  subsets,
- the model can be interpreted as learning a one-step local population update,
  not just a larger regression backbone.

## Phase 2: Supervision Repair

Objective: make event supervision credible enough for dynamic population
updates.

Why this matters:

- the final system must update both state and cell count,
- weak event targets will break rollout long before architecture becomes the
  bottleneck.

Tasks:

- audit `split/delete` coverage across developmental windows,
- separate weak unmatched labels from stronger disappearance evidence,
- keep `strict` delete as the default while better targets are developed,
- isolate `split-rich` and other informative subsets,
- make event losses interpretable on held-out windows with actual positives.

Definition of done:

- event supervision is no longer dominated by label construction artifacts,
- held-out event metrics carry biological information,
- dynamic modeling can be discussed without hiding behind proxy labels.

## Phase 3: Context Expansion

Objective: move from anchor-local context to larger population context.

This is the bridge from the active baseline to the final embryo model.

Tasks:

- expand from anchor-centered prediction toward predicting updates for more of
  the local window at once,
- increase context size systematically,
- test whether local-plus-global context acts like a useful compressed embryo
  state,
- identify where performance saturates as context radius grows,
- decide when the active representation is ready to become an embryo-scale
  state representation.

Definition of done:

- larger context windows improve or stabilize prediction in a meaningful way,
- the model can operate on larger populations without collapsing into
  uninformative averaging,
- context can be interpreted as a scalable state representation, not just a
  local helper signal.

## Phase 4: Population Dynamics

Objective: turn the validated one-step predictor into a repeated population
update model.

This is the first phase that should be treated as real developmental dynamics.

Tasks:

- move from anchor-query prediction to group update prediction,
- update cell states and event propensities jointly,
- evaluate short multi-step rollout under changing context,
- measure drift in gene-state distribution, cell-count balance, and event
  balance,
- reject rollout settings that only look plausible visually.

Definition of done:

- repeated short-horizon updates remain stable for more than one step,
- context does not immediately become stale under autoregressive reuse,
- rollout metrics are scientifically interpretable rather than only qualitative.

## Phase 5: Whole-Embryo Rollout

Objective: scale the population update rule to embryo-level developmental
prediction.

At this phase, the model should be asked to represent the embryo as one shared
state, not as isolated anchor tasks.

Tasks:

- define the embryo-scale state contract,
- run full-embryo one-step updates,
- support variable population size through division and disappearance events,
- evaluate whether global developmental structure is preserved over time,
- begin testing early-stage to later-stage developmental prediction.

Definition of done:

- the model can update embryo-scale states without immediate collapse,
- whole-embryo rollout preserves plausible developmental structure,
- context has effectively expanded from one cell to the embryo as a whole.

## Phase 6: Branching Flows Integration

Objective: use Branching Flows as the formal variable-cell-count generative
framework once embryo-scale dynamics are ready.

Role of Branching Flows in this roadmap:

- not the first learner,
- yes as the eventual generative language for state + split + delete coupling,
- yes once the learned update rule is already biologically grounded,
- no as a substitute for missing supervision.

Tasks:

- define a biologically defensible `Z` construction for developmental data,
- connect the learned transcriptomic update rule to the Branching Flows base
  process,
- move event heads toward true generative events,
- compare plain autoregressive embryo rollout against Branching Flows rollout.

Definition of done:

- Branching Flows improves long-horizon stability or variable-cell-count
  generation,
- gains hold on real embryo evaluation rather than synthetic demos,
- the resulting model still follows `Discover, Don't Inject`.

## Phase 7: Spatial Reintegration

Objective: reintroduce spatial information as embryo-scale auxiliary context.

Tasks:

- use space as auxiliary context, readout, or evaluation axis,
- test whether transcriptomic latents recover spatial organization,
- compare transcriptome-only embryo context against transcriptome-plus-space
  context,
- keep transcriptome as the main developmental state rather than making space
  the primary target.

Definition of done:

- spatial information improves prediction, calibration, or interpretability,
- transcriptomic context remains central,
- embryo-scale developmental prediction becomes more biologically grounded.

## Immediate Next Steps

1. Keep `strict` delete as the default training target mode.
2. Keep the active task framed as learning a one-step local population update,
   not single-cell long-range prediction.
3. Use `split-rich` and context-ablation comparisons as the highest-value tests
   of whether context is carrying real developmental information.
4. Expand from anchor-only supervision toward larger group-update supervision
   once the local update rule is stable.
5. Re-enter whole-organism rollout only when the update rule and event targets
   are both strong enough to survive repeated application.

## Non-Goals

- no active claim that the current anchor benchmark is the final biological
  task,
- no treating weak `delete` labels as clean death supervision,
- no promotion of visually plausible rollout without quantitative support,
- no injected lineage embeddings as a shortcut to developmental structure,
- no forgetting that the end target is embryo-scale developmental prediction.
