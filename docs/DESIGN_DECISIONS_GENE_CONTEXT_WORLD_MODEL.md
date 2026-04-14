# Design Decisions and Scientific Notes:
# From Embryo Latent World Models to Direct Gene-Patch Prediction

## Purpose

This document records the main scientific and architectural decisions made in
the current `gene-context` line of work, especially the reasoning behind
recent embryo-scale predictive experiments, their failures, and the shift
toward a simpler JiT-like gene-space modeling route.

This is intentionally more detailed and reflective than:

- `docs/ROADMAP.md`
- `docs/EXPERIMENT_HISTORY.md`
- `docs/ARCHITECTURE_WHOLE_ORGANISM_AR.md`

Those documents describe the project trajectory and key milestones.
This document is for recording:

- why certain designs were tried,
- what they were meant to solve,
- what their failures actually taught us,
- which assumptions were discarded,
- and what scientific interpretation we currently believe is defensible.

## Scope

This document covers the transition from:

- embryo masked multi-view state learning,
- embryo future-set prediction,
- latent-space world-model and JEPA-inspired ideas,

to:

- gene-first evaluation,
- rejection of pooled latent as the main state object,
- and a JiT-like direct clean gene prediction view of the problem.

It does not try to summarize every local context or anchor experiment already
covered in earlier docs.

## Executive Summary

The project originally tried to build embryo-scale predictive models through
latent state learning and future latent prediction.

That route produced a useful scientific result:

- latent representations can carry real future developmental information.

But it repeatedly failed at a stronger requirement:

- predicted future latent should live in the same readable state space as true
  future latent.

The failure mode was unusually consistent across many variants:

- the predicted representation contained information,
- but it was misaligned with the true representation by a largely global
  linear transformation,
- so self-readability could be strong while transfer from true-space probes
  remained weak.

This is not specific to biology.
It is a standard world-model state-space problem:

- target space choice,
- gauge freedom,
- dense versus pooled state,
- and the distinction between observation space, prediction space, and
  readout space.

Several attempts were made to fix this in latent space:

- frozen local-code staging,
- end-to-end pred-x future-set prediction,
- explicit split/count and spatial decoder branches,
- branch/mass-aware auxiliary heads,
- stronger pooled readout modules,
- free affine alignment heads,
- fixed canonical pooled spaces,
- strict token-space JEPA,
- and hybrid token-space plus frozen linear readout anchors.

None produced a better mainline than the best reconstruction-led
`spatialtuned` checkpoint.

At the same time, three things became increasingly clear:

1. `gene` is the biologically primary variable.
2. `pooled latent` is too crude and overloaded to serve as observation,
   prediction target, and biological readout simultaneously.
3. The most promising simplification is to move toward direct clean target
   prediction in gene space, following the spirit of JiT.

The current conclusion is therefore:

- do not continue treating pooled latent as the core world state,
- do not use `celltype` as the main biological metric,
- do not make the main problem harder than the data can support,
- and prefer direct clean gene prediction with longer historical patch
  context over increasingly elaborate latent-space repair.

## Project Goal And Why This Detour Happened

The long-term project goal remains:

> whole-organism developmental prediction from very early embryo state toward
> later embryo state, using transcriptomic state as the main biological
> variable.

The active work took a local and then embryo-latent route because direct
whole-embryo prediction was too unconstrained to validate cleanly from the
available data.

The local and embryo-scale predictive branches were therefore meant as
intermediate scientific subproblems:

- learn developmental state representations,
- test whether future structure is predictable,
- and identify which parts of the problem are representation issues versus
  data or supervision issues.

## First Principles That Survived All The Experiments

Several principles now look robust, independent of which specific model won or
failed.

### 1. Gene Expression Is Primary

Gene expression is the first-order biological state in this project.

This means:

- latent variables may still be useful as internal computation,
- but they cannot replace gene expression as the main biological object,
- and gene-level evaluation matters more than derived coarse labels such as
  `celltype`.

This became especially important after it became clear that `celltype`
composition was too coarse to serve as the main scientific success criterion.

### 2. Pooled Latent Is Too Overloaded

The project repeatedly tried to use pooled latent as:

- the prediction target,
- the state representation,
- the space where biology is probed,
- and sometimes also the compression mechanism.

This turned out to be a mistake.

Pooled latent collapses several roles that should remain separate:

- observation space,
- prediction/state space,
- readout space.

This is not just an aesthetic objection.
It likely explains why `founder`, `celltype`, and `depth` could be learned
reasonably while `spatial` and `split` remained much harder.

### 3. Pred-x Is More Natural Than Pred-noise Or Pred-delta

Influence from Kaiming He's recent generative modeling work, especially JiT
and pMF, was useful at a conceptual level.

The durable lesson was not "use the exact same algorithm".
It was:

> prediction space and loss space need not be the same, and the prediction
> target should ideally live closer to the data manifold.

For this project that translated into:

- avoid noisy velocity-like or residual-like targets when possible,
- predict a denoised or clean biological object,
- prefer direct future gene or future clean local state over latent deltas.

### 4. The Data Support Longer Context Better Than Much Larger Models

The raw dataset is large in cells, but limited in independent temporal
transitions under the current training contract.

That makes it reasonable to:

- expand patch sequence length,
- increase historical context,

before trying to:

- scale width and depth aggressively,
- or add more elaborate auxiliary structures.

## What The Data Actually Look Like

The default dataset used in the current patch and embryo work is:

- [nema_extended_large2025.h5ad](/mnt/public/max/NemaContext/dataset/processed/nema_extended_large2025.h5ad)

Real scale:

- `234,888` cells
- `27,138` genes
- `267` unique `embryo_time_min` values

Relevant loader:

- historical dataset implementation formerly in `src/data/gene_context_dataset_base.py` (removed from active `src`)

But the effective predictive dataset is much smaller after the current
training contract is applied:

- `dt_minutes = 20`
- `time_window_minutes = 10`
- `min_cells_per_window = 32`
- fixed pair construction on window centers

Under the current patch-set route:

- `PatchSetDataset`, `context_size = 32`
  - train: `25` time pairs, `100` samples
  - val: `6` time pairs, `24` samples
  - all: `31` time pairs, `124` samples

Under the current temporal history patch route:

- `TemporalPatchSetDataset`, `history_patches = 4`, `context_size = 32`
  - train: `20` time pairs, `80` samples
  - val: `4` time pairs, `16` samples
  - all: `24` time pairs, `96` samples

This distinction matters:

- the raw data are not tiny,
- but independent time-transition supervision is still only on the order of
  a few dozen pair centers under the current training interface.

Scientific implication:

- it is reasonable to lengthen sequence context,
- but not to assume we can train a large-capacity world model without quickly
  overfitting pair-specific structure.

## Chronology Of The Recent Design Decisions

## Stage 1: Embryo Multi-View State Learning Established That Future Structure
## Is Predictable

The first major embryo-scale success was not direct prediction.
It was representation learning:

- masked multi-view embryo state learning,
- followed by masked future view reconstruction,

showed that embryo latents can encode future developmental structure.

That established that embryo-scale future information is present and learnable
in principle.

This was an important scientific threshold:

- the project no longer relied only on local patch tasks,
- and embryo-level state learning was not empty.

## Stage 2: Direct Global Future Latent Prediction Failed

The next obvious step was:

- predict the future embryo latent directly from the current embryo latent.

This route looked attractive because it offered a compact world-model state.

But it repeatedly failed in the same way:

- latent cosine or regression losses could become small,
- yet developmental readouts on the predicted future latent remained weak.

The early interpretation was that maybe the future latent target was noisy.
Later diagnostics showed that was not the main problem:

- future latent targets were stable under resampling,
- nearly invariant under view order,

so the failure was more geometric than stochastic.

## Stage 3: Future-Set Prediction Was Introduced To Avoid Pure Global Latent
## Jumps

Instead of predicting one global future summary, the project moved to
future-part completion:

- current context plus visible future parts predict masked future local-view
  sets.

This was scientifically better grounded:

- more local,
- less monolithic than one global latent jump,
- and closer to reconstructing future developmental structure.

The strongest early version of this route used reconstruction-backed MAE-style
future-set completion.

This branch became the first embryo-scale predictive path that was consistently
near the right regime.

## Stage 4: pMF And Latent Forcing Motivated Pred-x And Intermediate State
## Spaces

Two external conceptual influences were important:

- Kaiming He's pMF work:
  prediction space and loss space can be separated, and the prediction target
  should remain on a lower-dimensional manifold when possible.
- Latent Forcing:
  latent or code-like intermediate objects can act as a scratchpad, and the
  ordering of information matters.

This led to a temporary design decision:

- use a local code as an internal pred-x-like target,
- but keep a loss defined in a richer structured state space.

That was the motivation for the staged local-code route.

## Stage 5: Stage A / Stage B Local-Code Design

The staged design was:

- Stage A:
  learn a local cell code that could decode into structured local cell state.
- Stage B:
  predict future local codes from embryo future-set context and decode them
  back into structured state for supervision.

The conceptual attraction was strong:

- local code as manifold-respecting target,
- richer decoded state as supervision space,
- less direct pressure to predict raw future cells.

### What Was Good About The Idea

It was principled in exactly the pMF sense:

- the network output space was chosen to be easier than the supervision space.

### Why It Was Abandoned

The frozen staged version did not improve future developmental structure.

The frozen local-code checkpoint introduced interface mismatch:

- the local code was learnable in isolation,
- but once frozen, it no longer aligned with the outer embryo future-set
  objective.

Probe results showed this clearly:

- celltype could improve,
- but founder, depth, spatial, and split could all worsen badly.

This led to the first decisive simplification:

- remove the frozen two-stage contract,
- keep only an end-to-end version if local code remained useful at all.

## Stage 6: End-to-End Pred-x Future-Set Was Better Than Frozen Staging

The next version kept the idea of an internal pred-x bottleneck but removed
frozen staging:

- the local-code encoder and decoder were integrated into the embryo
  future-set model,
- training became end-to-end.

This version materially improved over the frozen Stage A / Stage B path.

Most importantly:

- founder improved,
- celltype improved,
- depth improved,

relative to the frozen-stage design.

But it still showed the same asymmetry:

- semantic or composition-like quantities improved more,
- spatial and split remained weak.

This was the first strong sign that:

- the model could learn future biological information,
- but the representation of geometry and branching remained the real hard
  part.

## Stage 7: Spatial And Split Were Explicitly Targeted

Because `spatial` and `split` remained the most fragile probe families, the
project tried to make them more explicit.

This included:

- explicit split/count heads,
- separate spatial decoder/loss,
- embryo dataset support for `split_fraction`,
- later branch-aware and mass-aware auxiliary heads.

### Why This Was A Reasonable Attempt

It matched ideas from:

- BranchingFlows:
  branching events are not the same as continuous state change.
- DeepRUOT:
  growth/mass and spatial transport deserve separate treatment.

### What Actually Happened

A light spatial tuning of the decoder losses produced the best tradeoff
checkpoint in the repository:

- [embryo_future_set_probes_dt40_multi_recon_spatialtuned_all_cv.json](/mnt/public/max/NemaContext/result/gene_context/embryo_future_set_probes_dt40_multi_recon_spatialtuned_all_cv.json)

Representative held-out probe means:

- founder: `0.159`
- celltype: `0.450`
- depth: `0.554`
- spatial: `-1.019`
- split: `-1.211`

This checkpoint is important because it remains the best compromise:

- semantic structure is still present,
- and spatial/split are less damaged than in many later runs.

But stronger structural heads did not solve the real problem.
In particular:

- branch-aware + mass-weighted auxiliary heads made things worse overall.

Scientific lesson:

- split and spatial are not just auxiliary labels to bolt onto an already
  defined future object,
- they are part of defining what the predictive object should be.

## Stage 8: Oracle Diagnostics Changed The Interpretation Of The Failure

The next major step was not a new model.
It was diagnosis.

Several oracle analyses were added, including:

- [diagnose_embryo_future_set_oracles.py](/mnt/public/max/NemaContext/scripts/diagnose_embryo_future_set_oracles.py)
- [analyze_embryo_future_set_rep_alignment.py](/mnt/public/max/NemaContext/scripts/analyze_embryo_future_set_rep_alignment.py)

The key finding was:

- predicted future representations were not empty,
- but they were misaligned with true future representations.

In other words:

- self-readability could be high,
- but transfer from probes trained in the true future space could be poor.

This ruled out a simplistic reading such as:

- "the model learned nothing"

and replaced it with:

- "the model learned an information-bearing but differently coordinated future
  space."

### Readout Was Not The Main Culprit

Oracle readout tests showed that stronger readouts did not rescue the transfer
gap in a meaningful way.

### Object Contract Was Not The Main Short-Term Culprit Either

True-support or true-alignment oracle variants did not provide the dramatic
recovery that would be expected if the main problem were simply the wrong slot
support or matching.

### The Most Important Diagnosis

The main mismatch behaved like:

- a global linear but non-orthogonal misalignment,

not like:

- a strongly nonlinear or regime-specific distortion.

This was supported by the observation that:

- direct transfer was weak or negative,
- but ridge-aligned transfer could nearly recover performance.

This is the core world-model gauge result of the recent work.

## Stage 9: Multiple Gauge-Fixing Attempts Failed

Once the problem was recognized as a world-model state-space problem, several
repair attempts were made.

### 9.1 Shared Readout Token Pooling

Idea:

- replace static weighted pooling with a stronger shared readout token and
  cross-attention pooling, closer to a set-transformer or ViT-style readout.

Why it failed:

- it introduced more readout freedom,
- and made pooled basis drift worse, not better.

### 9.2 Free Affine Alignment Head

Idea:

- explicitly learn `A(z_pred) + b ≈ z_true`.

Why it failed:

- it did not remove gauge freedom,
- it just parameterized a new one.

This matches standard world-model practice:

- a free post-hoc adapter is not the same as fixing the state space.

### 9.3 Fixed Canonical Pooled Space

Idea:

- fit a canonical pooled latent basis from true latents,
- then force prediction into that canonical space.

Why it failed:

- the pooled state itself was still too overloaded,
- and the model did not learn a stable predictive state in that canonicalized
  pooled basis.

### 9.4 Strict Token JEPA

Idea:

- predict only token-space targets,
- remove pooled and decoded objectives almost entirely,
- imitate JEPA more directly.

Why it failed:

- token-space loss alone did not fix gauge,
- the resulting token space became even less compatible with downstream shared
  readouts.

### 9.5 Token-Space Plus Frozen Linear Readout Anchor

Idea:

- keep token-space prediction,
- add a frozen linear readout anchor from token states to pooled readout space.

Why it failed:

- it prevented total collapse relative to strict token JEPA,
- but still did not create a shared world-state space.

Net conclusion from all of these:

- the project repeatedly confirmed the diagnosis,
- but none of the pooled-latent or weak-anchor fixes produced a better main
  line than the simpler `spatialtuned` reconstruction-led baseline.

## Stage 10: Gene-First Evaluation Replaced Celltype-First Comfort

A major conceptual correction was then made:

- `celltype` is too coarse to be the main biological metric.

This matters because:

- gene expression is first-order biology,
- celltype is a derived summary.

The project therefore shifted toward:

- gene-first thinking,
- and eventually token/set-level gene evaluation.

This led to two practical changes.

### 10.1 Simple Local Region Gene Baselines

A simple baseline was implemented:

- [run_local_region_gene_baseline.py](/mnt/public/max/NemaContext/examples/legacy/whole_organism_ar/run_local_region_gene_baseline.py)

Task:

- current local mean gene predicts future local mean gene.

Important result:

- this task is not empty,
- a small MLP can beat persistence in MSE,

but:

- pair count is so small that per-gene `R^2` is unstable.

Scientific value:

- it establishes that direct gene prediction is meaningful enough to justify
  a more direct model.

### 10.2 Token/Set-Level Gene Evaluation

The project stopped pretending that pooled latent to celltype was enough.

Instead:

- gene-first token/set-level evaluation was introduced,
- and pooled-gene probe logic was treated as too crude.

This reinforced the main point:

- the project should move toward gene-space prediction directly,
- not keep repairing pooled latent abstractions.

## Why JiT Became Increasingly Relevant

Kaiming He's JiT work became influential because it offered a cleaner
alternative to the increasingly baroque latent-space repair process.

The key lesson was not the exact image architecture.
It was:

> predict clean data directly in a space that already has the right semantic
> coordinates, rather than predicting noise-like or latent quantities and then
> repairing their geometry later.

Applied here:

- image patches become gene patches or gene token sets,
- clean image target becomes clean future gene target,
- tokenizer-free pixel prediction becomes latent-free gene prediction.

This suggested a much simpler question:

> given current local gene tokens, can we directly predict future clean gene
> tokens?

That is a cleaner problem than:

- predict pooled future latent,
- align it,
- decode it,
- and hope biology survives all of that.

## The Current JiT-Like Shift

This led to the implementation of:

- [JiTGenePatchModel](/mnt/public/max/NemaContext/src/branching_flows/gene_context_patch.py)
- [train_jit_gene_patch.py](/mnt/public/max/NemaContext/examples/legacy/whole_organism_ar/train_jit_gene_patch.py)

The current design is intentionally simple:

- current patch tokens enter directly as gene tokens,
- future query tokens are learned queries,
- a bidirectional transformer processes both together,
- the model directly predicts clean future genes,
- and the main loss is a gene set loss plus a small mean-gene term.

Important negative choices:

- no pooled latent state,
- no code bottleneck,
- no decoder tether,
- no latent alignment head,
- no hand-designed anchor ontology inside the model.

This is the current cleanest expression of the JiT idea in this repository.

## Why The Model Is Now Better Thought Of As Video-Like

Once pooled latent and hand-designed state bottlenecks are removed, the right
analogy is no longer a latent world model.
It is closer to a video model:

- multiple past patches or windows act like history frames,
- future query tokens act like next-frame prediction slots,
- the model directly predicts clean future token content.

It is not a standard video model because:

- tokens are not raster image patches,
- cells are not on a fixed grid,
- object count and effective support vary,

but conceptually it is much closer to:

- next-frame token prediction,

than to:

- latent state alignment and pooled world-state repair.

## Why We Rejected Hand-Designed Anchors At The Model Level

There was a growing discomfort with manually designed anchor ontologies.

The underlying objection was:

- if bidirectional attention is strong enough,
- we should not hard-code a complicated spatial anchor logic for the model's
  main predictive object.

This does not mean data sampling cannot still use patch windows.
It means:

- the network should not depend on a brittle hand-designed anchor identity
  system to do prediction.

That is why the direct JiT-like gene patch model was attractive:

- it uses queries,
- not a manually engineered future object ontology.

## Current Temporal JiT Patch Design

The JiT-like gene patch path was then expanded to use more history.

Implemented dataset and training path:

- historical `TemporalPatchSetDataset` implementation formerly in `src/data/gene_context_dataset_patch.py` (removed from active `src`)
- [train_jit_gene_patch.py](/mnt/public/max/NemaContext/examples/legacy/whole_organism_ar/train_jit_gene_patch.py)

Current temporal design:

- `history_patches = 4`
- `context_size = 32`

Effective sequence:

- observed history tokens: `4 * 32 = 128`
- future query tokens: `32`
- total sequence length: `160`

This is the current best compromise between:

- a genuinely ViT/video-like context length,
- and the limited number of independent time pairs.

## What The Data Support And What They Do Not

The current dataset support suggests:

### Supported

- longer patch sequences,
- multiple history patches,
- moderate transformer size,
- direct gene-space prediction,
- local or patch-level future prediction.

### Not Well Supported

- very large models,
- heavy latent-space repair machinery,
- highly parameterized world-model alignment stacks,
- any approach that assumes hundreds of independent temporal transitions.

This is why current safe scaling should mean:

- longer sequence,
- not much larger width and depth.

## Current Working Interpretation Of The Best Existing Embryo Checkpoint

The current best embryo predictive checkpoint remains the `spatialtuned`
future-set model:

- [embryo_future_set_probes_dt40_multi_recon_spatialtuned_all_cv.json](/mnt/public/max/NemaContext/result/gene_context/embryo_future_set_probes_dt40_multi_recon_spatialtuned_all_cv.json)

It is important not because it solved the state-space problem.
It did not.

It is important because it achieved the best compromise between:

- preserving semantic developmental structure,
- and not completely losing spatial/split information.

Best reading of this result:

- a weak local geometry tether helps,
- but only when it is weak enough not to dominate state learning,
- and more complex readout or alignment machinery was mostly harmful.

This is useful as a baseline, not as a final answer.

## Current Design Commitments

At the time of writing, the following decisions should be treated as active:

### Keep

- gene as the primary biological variable,
- direct clean target prediction whenever possible,
- longer historical context before larger model capacity,
- simple transformer contracts over increasingly elaborate latent fixes,
- token/set-level evaluation rather than pooled latent evaluation as the main
  scientific standard.

### Reject Or Deprioritize

- pooled latent as the main world state,
- celltype as the main biological success criterion,
- frozen staged local-code prediction as the main route,
- free affine alignment heads,
- stronger pooled readout modules as a primary fix,
- decoder tether as a main stabilizer,
- branch/mass auxiliary heads as a primary solution,
- strict latent alignment projects that depend on a pooled state remaining the
  core object.

## What Remains Scientifically Open

The following questions remain unresolved:

### 1. Can Direct Gene-Patch Prediction Actually Generalize Better?

The JiT-like gene patch model is simpler and cleaner, but it has not yet
produced a dominant result.

That means the conceptual simplification is attractive, but still needs to be
validated empirically.

### 2. How Much Historical Context Is Optimal?

The data likely support more temporal context than current short patch
contracts, but only up to the limit set by sparse independent time-pair
supervision.

### 3. How Fine Should The Predictive Object Be?

There is still an unresolved tension between:

- cell-level prediction,
- patch-level prediction,
- and coarser local region summaries.

The project increasingly favors direct gene-space prediction, but the right
granularity remains an empirical question.

### 4. How Should Spatial And Branching Structure Be Evaluated Without
### Reverting To Pooled Latent?

Once pooled latent is rejected as the main state, spatial and split evaluation
must also be reformulated more naturally in token/set space.

## Recommended Reading Order After This Document

To understand the current project logic in the right order:

1. [docs/ROADMAP.md](/mnt/public/max/NemaContext/docs/ROADMAP.md)
2. [docs/EXPERIMENT_HISTORY.md](/mnt/public/max/NemaContext/docs/EXPERIMENT_HISTORY.md)
3. [docs/ARCHITECTURE_WHOLE_ORGANISM_AR.md](/mnt/public/max/NemaContext/docs/ARCHITECTURE_WHOLE_ORGANISM_AR.md)
4. this document
5. current active training scripts:
   - [train_embryo_future_set.py](/mnt/public/max/NemaContext/examples/legacy/whole_organism_ar/train_embryo_future_set.py)
   - [train_jit_gene_patch.py](/mnt/public/max/NemaContext/examples/legacy/whole_organism_ar/train_jit_gene_patch.py)

## Final Position

The strongest current belief is:

> the main recent difficulty was not "biology is too special" but "we kept
> asking pooled latent to do too many incompatible jobs."

The project now has enough evidence to say the following.

- Biology did not force the use of pooled latent.
- World-model gauge problems are real in this setting.
- Stronger latent repair machinery did not solve them.
- Direct clean gene prediction is scientifically cleaner and probably closer
  to what the data actually support.

The practical implication is simple:

> prefer a modest, longer-context, JiT-like gene-space predictor over a more
> elaborate pooled-latent world model unless new evidence clearly overturns
> this conclusion.
