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
- continuous relative spatial encoding helps the clean split-rich baseline,
- pairwise spatial bias slightly increases measured context use,
- a one-step transcriptomic update task can be trained end-to-end on real data,
- a patch-to-patch set-level objective now outperforms the matched
  single-cell control,
- increasing patch context size strengthens that multi-cell advantage,
- shared-encoder state-view training at broad coverage and `dt = 40` now
  produces the first biologically meaningful latent,
- broad multi-cell state latents outperform broad single-cell controls on
  future developmental probes including future founder composition,
  future cell-type composition, future lineage-depth structure, future spatial
  extent, and future split-fraction alignment,
- a masked self-supervised state-learning route now preserves strong
  biological structure and, with masked gene reconstruction, matches or
  exceeds the broad state-view baseline on several future developmental
  probes,
- whole-organism autoregressive infrastructure and variable-cell-count code are
  already present in the repository.

What remains weak:

- current work still validates local or window-scale updates, not embryo-scale
  developmental prediction,
- `split/delete` supervision is still weaker than the gene-state target,
- multi-step dynamics are not yet biologically closed under changing context,
- temporal discrimination, hard-negative discrimination, queue-based
  discrimination, and future-retrieval ranking are not effective learning
  signals in the current self-supervised route,
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

Objective: move from anchor-local context to larger population context and
replace token matching with set-level local-state prediction.

This is the bridge from the active baseline to the final embryo model.

Tasks:

- expand from anchor-centered prediction toward patch-to-patch prediction,
- keep token matching as a diagnostic tool rather than the main objective,
- use set-level patch targets where future local state is predicted as a set,
- increase context size systematically,
- test whether local-plus-global context acts like a useful compressed embryo
  state,
- identify where performance saturates as context radius grows,
- decide when the active representation is ready to become an embryo-scale
  state representation.

Current update:

- single-patch set prediction is established as the strongest local pretext
  task,
- multi-patch extrapolation works, but extra patches should be understood as
  multiple views of one underlying state rather than as stable biological
  entities,
- direct patch-level attention/selection was tested and did not improve the
  active benchmark,
- the next transition is therefore shared-encoder multi-view state learning,
  not stronger patch hierarchy.
- the first biologically meaningful result has now appeared in this phase:
  broad multi-view state latents encode future developmental structure better
  than broad no-context controls.
- a second, more self-supervised result is now also in place:
  masked view + masked future + masked gene reconstruction yields a
  biologically strong latent, while all tested contrastive/retrieval-style
  temporal objectives remain weak.
- embryo-scale work has now also crossed a first useful threshold:
  embryo-level masked multi-view modeling with masked future views produces an
  embryo latent whose future developmental probes are substantially stronger
  than current-only embryo masking and far stronger than direct embryo summary
  regression.
- a minimal embryo JEPA path is now also implemented on the same embryo-view
  interface; the first instability was traced to per-batch target whitening of
  a highly concentrated target latent, and the stabilized layer-normalized
  version now smoke-trains normally.
- a first embryo one-step latent dynamics baseline also now exists:
  future embryo latent prediction is already strong on top of the masked-future
  backbone, but joint probe heads are still weak and should not yet be treated
  as the final embryo-dynamics contract.
- additional diagnostics now show that the future embryo latent target is
  itself stable under repeated future-view resampling and almost invariant to
  view order, so the main one-step bottleneck is no longer target noise but
  latent geometry: cosine-only matching is too weak to preserve the
  biology-readable future-latent manifold.

Definition of done:

- larger context windows improve or stabilize prediction in a meaningful way,
- set-level patch prediction becomes a stable default benchmark,
- multi-cell advantage grows rather than collapses as context size increases,
- the model can operate on larger populations without collapsing into
  uninformative averaging,
- context can be interpreted as a scalable state representation, not just a
  local helper signal,
- and the latent can predict biologically interpretable future developmental
  features rather than only project-specific reconstruction losses.

Practical note for the next iteration:

- self-supervised progress should now focus on reconstruction-style signals
  that are working,
- temporal discrimination should not remain the main optimization focus,
- the next scaling step should keep embryo-scale learning on masked
  multi-view reconstruction with future-view prediction rather than direct
  summary regression or more patch-level contrastive tuning,
- embryo one-step training should now be treated as a diagnostic branch:
  first learn `Z_t -> Z_{t+dt}`, then attach frozen developmental probes,
  but do not treat that contract as the main embryo predictive route,
- the main embryo predictive route should instead move through
  MAE-style future-part completion:
  current views plus visible future parts should reconstruct masked future
  local-view sets before any return to direct global latent transition,
- do not assume that smaller cosine automatically implies biologically correct
  dynamics; true-future and predicted-future latent probe gaps must remain a
  required diagnostic,
- any SAE work should be treated as frozen-latent interpretability analysis,
  not as a replacement for the active state encoder.
- embryo JEPA should currently be treated as an exploratory geometry-focused
  alternative, not yet as the primary embryo-scale dynamics result.
- reconstruction-backed MAE future-set completion is currently the strongest
  embryo-scale predictive contract in the repository.

## Phase 4: Population Dynamics

Objective: turn the validated local-state encoder into a repeated population
update model.

This is the first phase that should be treated as real developmental dynamics.

Tasks:

- move from patch views to embryo-state representations that remain stable
  across different local observations of the same embryo window,
- learn current-to-future embryo-state prediction first through masked future
  view reconstruction before full rollout,
- validate embryo one-step latent prediction separately from embryo probe
  decoding,
- keep biological developmental probes as gate metrics rather than relying
  only on latent or reconstruction loss,
- update cell states and event propensities jointly,
- evaluate short multi-step rollout under changing context,
- measure drift in gene-state distribution, cell-count balance, and event
  balance,
- reject rollout settings that only look plausible visually.

Definition of done:

- repeated short-horizon updates remain stable for more than one step,
- context does not immediately become stale under autoregressive reuse,
- rollout metrics are scientifically interpretable rather than only
  qualitative,
- embryo-level latent probes remain biologically meaningful after moving from
  current-only masking to masked future-view learning,
- embryo one-step latent prediction is reliable before joint probe training is
  treated as solved.
- the next embryo-dynamics work should improve latent geometry or target
  metrics rather than re-introducing more probe co-training or more temporal
  discrimination variants.

## Phase 5: Whole-Embryo Rollout

Objective: scale the population update rule to embryo-level developmental
prediction.

At this phase, the model should be asked to represent the embryo as one shared
state, not as isolated anchor tasks.

Tasks:

- define the embryo-scale state contract,
- promote embryo-level masked-future representation learning into embryo
  one-step latent prediction,
- keep probe decoding secondary to latent dynamics until a frozen-latent probe
  path is validated,
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
2. Treat patch-to-patch set prediction as the default active benchmark.
3. Use normalized set metrics plus biological composition readouts rather than
   raw total loss alone.
4. Scale patch context before scaling architecture complexity.
5. Sweep `global_context_size` and then `dt` at the best patch scale.
6. Treat multi-patch coverage as multiple views of one regional state rather
   than as a new ontology of patch entities.
7. Build shared-encoder multi-view state learning on top of the validated
   patch-set pretext task.
8. Compare `dt = 20` and `dt = 40` under the shared-encoder state objective and
   verify whether multi-cell advantage appears only at the more informative
   time scale.
9. Re-enter whole-organism rollout only when the set-level update rule and
   event targets are both strong enough to survive repeated application.

## Non-Goals

- no active claim that the current anchor benchmark is the final biological
  task,
- no treating weak `delete` labels as clean death supervision,
- no promotion of visually plausible rollout without quantitative support,
- no injected lineage embeddings as a shortcut to developmental structure,
- no forgetting that the end target is embryo-scale developmental prediction.
