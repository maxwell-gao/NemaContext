# Gene-Context Baseline: Biological Meaning and Published Context

## Purpose

This document explains the active `gene-context` baseline in biological terms
and situates it relative to published bioinformatics work on developmental
state transition modeling.

Relevant code:

- `src/data/gene_context_dataset.py`
- `src/branching_flows/gene_context.py`
- `examples/whole_organism_ar/train_gene_context.py`
- `examples/whole_organism_ar/evaluate_gene_context.py`
- `examples/whole_organism_ar/train_gene_single_cell.py`
- `examples/whole_organism_ar/evaluate_gene_single_cell.py`

## What Biological Problem This Baseline Is Trying To Solve

The final project goal is whole-organism developmental prediction from the
zygote onward.

This baseline is not that full problem.

It is the smallest real-data subproblem that is currently tractable:

- take a local developmental population observed at time `t`,
- predict a short-horizon update at `t + dt`,
- test whether nearby and broader transcriptomic context improve that update.

So this baseline should be read as:

- not a spatial simulator,
- not a lineage-conditioned classifier,
- not a solved rollout model,
- but the first validated approximation to a later population-dynamics model.

The operative biological question is therefore:

> Does multi-cell transcriptomic context improve short-horizon developmental
> update prediction beyond what is available from a single cell and time alone?

This is the correct first question for the current data regime because the
available scRNA-seq data are destructive snapshots, not tracked single-cell
movies.

It also avoids a biological mistake:

- predicting one cell many steps into the future while freezing context is not
  self-consistent,
- because the surrounding developmental population is also changing.

So the current baseline is best interpreted as a one-step local population
update model, not a long-range single-cell trajectory predictor.

## Biological Meaning of the Inputs

### `genes`

Each token's main state is real gene expression from
`dataset/processed/nema_extended_large2025.h5ad`.

After HVG selection and normalization, this vector is interpreted as an
approximation to the cell's current developmental state:

- differentiation program,
- cell-cycle state,
- tissue commitment,
- transient regulatory program,
- developmental competence.

This is the main biological state variable in the active baseline.

### `time`

The model also receives embryo time.

Biologically, this represents developmental stage:

- the same expression program can have different meaning early versus late,
- developmental transitions are not time-homogeneous,
- time provides stage context without injecting explicit lineage structure.

### `multi-cell context`

In the multi-cell baseline, one sample contains multiple cells from the same
time window.

Biologically, this represents the surrounding developmental population:

- which other cell states coexist at that stage,
- what cell-state composition is present,
- what global developmental context constrains each cell.

This is important because development is not an independent-cell process.
Cells change state in the presence of other cells undergoing coordinated
programs.

### `structured anchor-centered context`

The active input format is now more specific than a generic random time window.

In the strongest current setting, one sample contains:

- one anchor cell whose future is the main supervised query,
- a local neighborhood around that anchor,
- a smaller set of global background cells from the same developmental window.

Biologically, this means the model is asked:

- what does this focal cell do next,
- given its nearby developmental neighborhood,
- while also seeing a coarse summary of the broader embryo state.

This is the current best approximation to "context" without injecting a
lineage tree as model input.

It is also only the first scale of context.

In the long-term roadmap, this context is meant to expand:

- from anchor-local neighborhood,
- to larger local populations,
- to embryo-scale shared context.

### `relative spatial geometry`

The active baseline now also uses continuous relative spatial features rather
than only coarse anchor-distance buckets.

Each token can receive:

- `dx, dy, dz` relative to the anchor,
- `r = ||d||`,
- `has_spatial`.

In the multi-cell path, this can be used at two levels:

- token level through a learned relative-position projection,
- pairwise level through an attention bias derived from pairwise relative
  geometry.

Biologically, this is a better approximation than a pure distance bucket
because it preserves:

- direction,
- magnitude,
- which cells actually have usable spatial coordinates.

This is still anchor-relative rather than embryo-global, so it should be read
as a local geometric reference frame, not yet a full embryo coordinate system.

### What Is Not Input

The active baseline does **not** input:

- founder identity,
- lineage embeddings,
- lineage tree distance,
- lineage-biased attention,
- embryo-global spatial coordinates as the main driver.

This matters because the active direction is to learn context from data rather
than from injected lineage structure.

The current spatial input should therefore be understood as:

- a local geometric conditioning signal,
- not a hand-built global spatial scaffold.

## Biological Meaning of the Outputs

### `gene_delta`

The main output is a predicted short-horizon change in gene state.

Biologically, this is a local developmental transition vector:

- where the cell's transcriptional program is moving next,
- not just what type it currently is,
- not yet a full cell-fate mechanism,
- but a useful proxy for near-future developmental motion in expression space.

At the current stage it should be interpreted as:

- one step of developmental update,
- not a full long-horizon cell-fate forecast in isolation.

### `split_logits`

This is a predicted propensity for division in the next time window.

Biologically, it should be interpreted as:

- a near-term proliferative tendency,
- not a mechanistic cell-cycle model,
- not yet an exact mitotic timing predictor.

### `del_logits`

This is a predicted propensity for disappearance by the next window.

At the current stage, this should be interpreted cautiously.

It is closer to:

- failure to persist into the next matched window,
- possible death,
- possible state-exit from the local matching regime,

than to a clean apoptosis label.

This is a known limitation of the current target construction.

## Why This Baseline Is Biologically More Meaningful Than the Spatial Baseline

The spatial rollout baseline was useful as an engineering test of dynamic cell
management, but it is not a strong biological model of development.

This gene-context baseline is more biologically meaningful because:

- gene state is treated as the primary developmental state,
- time is treated as developmental stage,
- context comes from other cells' transcriptomic states,
- lineage is not directly injected into the model.

This makes the model closer to developmental systems biology and farther from a
point-cloud kinematics model.

It also makes it the correct first stage for a later whole-embryo model:

- transcriptome remains the main developmental state,
- context is learned rather than injected,
- the model can later be promoted from local update prediction to larger
  population update prediction.

## Current Spatial-Encoding Reading

The recent spatial ablations support a specific interpretation.

What now seems true:

- dropping spatial input hurts both multi-cell and single-cell models,
- continuous relative geometry is better than omitting space,
- pairwise spatial bias gives a further small gain for the multi-cell model,
- that pairwise term also slightly increases the measured `full > anchor_only`
  context benefit.

What does **not** yet seem true:

- spatial input is not yet a decisive reason that multi-cell beats single-cell,
- most current gains still show up in event-related terms rather than in the
  gene-regression term,
- spatial encoding is still helping a local update benchmark, not an
  embryo-scale rollout model.

So the best current interpretation is:

> spatial geometry is now a useful local inductive bias for context-aware
> developmental update prediction, but not yet the defining source of
> multi-cell advantage.

## Why The Project Is Now Moving Beyond Token Matching

The strongest recent result is not another token-level event comparison.

It is that a patch-to-patch set-level objective finally changed the direction
of the comparison:

- token-level matched objectives repeatedly showed that the multi-cell model
  uses context,
- but they did not reliably make the multi-cell model beat the matched
  single-cell control,
- patch-to-patch set prediction was the first objective where multi-cell
  became clearly better.

On the clean `context_size = 64` patch-set comparison:

- multi-cell full: `114.38`
- single-cell full: `116.92`
- multi-cell anchor-only: `117.82`

So, under the new objective:

- `multi < single`,
- `full < anchor_only`,
- and both statements hold at the same time.

Biologically, this matters because the question is no longer:

> which future token should this current token map to?

It becomes:

> what local developmental population state comes next?

That is much closer to the final project goal of whole-organism developmental
prediction.

## Biological Reading Of The Patch-Set Objective

The new patch-set objective should be understood as a transition from
cell-centric supervision to local-population supervision.

Input:

- a current local patch at time `t`,
- structured as anchor plus local neighborhood plus optional background,
- with transcriptome, time, relative geometry, and pairwise spatial bias.

Target:

- a future local patch sampled from the paired future window,
- treated as a set rather than as a collection of hard token matches.

Loss:

- OT-style set reconstruction,
- future patch latent alignment,
- future patch size prediction,
- future mean-gene prediction.

Biologically, this is better aligned with developmental reality because:

- local populations evolve together,
- not every current cell has a clean one-to-one future partner,
- snapshot data are better suited to predicting future local state structure
  than to enforcing exact token identity.

So this objective is the first one in the project that begins to ask:

> given a developmental microenvironment now, what microenvironment comes
> next?

That is still local, but it is no longer merely a focal-cell query.

## Scaling Result: Larger Patch Context Now Helps More

The patch-set scaling sweep strengthened this interpretation.

For `context_size = 64 / 128 / 256`, with fixed relative-position input and
pairwise spatial bias in the multi-cell path:

- `64`: multi `114.38`, single `116.92`, delta `-2.54`
- `128`: multi `370.78`, single `373.46`, delta `-2.68`
- `256`: multi `1336.25`, single `1349.95`, delta `-13.70`

The absolute scale of the loss grows because the patch-size regression term
grows with target cardinality, so those totals should not be compared across
different patch sizes directly.

But within each fixed scale, the comparison is meaningful:

- the multi-cell model stays ahead,
- the OT term also improves,
- and the margin becomes substantially larger by `context_size = 256`.

This is currently the strongest evidence that larger developmental context is
becoming genuinely useful once the task is framed as set-level patch
prediction.

## Current Interpretation

The active `gene-context` line should therefore now be read in two layers.

The older token-level baseline established:

- context is used,
- split is more informative than delete,
- delete needed repair,
- relative geometry and pairwise bias are useful local inductive biases.

The newer patch-set baseline established:

- a set-level patch objective is more biologically aligned than token
  matching,
- multi-cell can now beat the matched single-cell control,
- and that advantage strengthens as patch context expands.

This is the clearest current bridge between the local benchmark and the final
goal of embryo-scale developmental prediction.

## Current Boundary On Patch Modeling

The project should no longer treat manually sampled patches as if they were
the true biological units of the embryo.

The current best interpretation is:

- a patch is a local training view,
- patch-set prediction is a useful pretext task,
- but a patch is not the ontology that the final developmental model should
  represent.

This matters because a direct patch-attention stage was tested and did not
improve the multi-patch benchmark. That result matches the deeper conceptual
issue:

- patch boundaries are imposed by the training pipeline,
- patch count is chosen by the experimenter,
- different patch covers of the same embryo window are different observations
  of one underlying state, not different biological entities.

So the next stage should not be "better patch hierarchy." It should be:

- shared encoders across multiple local views,
- view-consistent state representations,
- current-to-future latent prediction,
- patch-set reconstruction kept only as an auxiliary local constraint.
> embryo-scale biological structure.

## Closest Published Work

No widely adopted published method is identical to this active baseline.
The nearest prior work falls into three adjacent families.

### 1. RNA velocity and local future-state direction

- La Manno et al., 2018, *RNA velocity of single cells*, Nature
  - https://www.nature.com/articles/s41586-018-0414-6
- Bergen et al., 2020, *Generalizing RNA velocity to transient cell states through dynamical modeling*, Nature Biotechnology
  - https://www.nature.com/articles/s41587-020-0591-3

These methods are close to our `gene_delta` output because they also estimate
short-horizon future direction in transcriptomic state space.

Key difference:

- RNA velocity is fundamentally a **single-cell** local dynamics method,
- typically uses spliced/unspliced information,
- does not model explicit multi-cell transcriptomic context,
- does not directly include split/delete outputs.

### 2. Optimal transport and cross-time developmental coupling

- Schiebinger et al., 2019, *Optimal-Transport Analysis of Single-Cell Gene Expression Identifies Developmental Trajectories in Reprogramming*, Cell
  - https://pubmed.ncbi.nlm.nih.gov/30712874/
- Tong et al., 2020, *TrajectoryNet: A Dynamic Optimal Transport Network for Modeling Cellular Dynamics*, ICML / PMLR
  - https://proceedings.mlr.press/v119/tong20a.html
- Yeo et al., 2021, *Generative modeling of single-cell time series with PRESCIENT enables prediction of cell trajectories with interventions*, Nature Communications
  - https://www.nature.com/articles/s41467-021-23518-w

These are close to our data construction problem:

- scRNA-seq provides snapshots, not tracked trajectories,
- one must infer likely future couplings or flows between time points,
- the task is developmental dynamics from cross-sectional data.

Key difference:

- these methods mostly operate at the level of transport, trajectory inference,
  or continuous-time cell-state flow,
- they usually do not represent a **time-window of many cells as the explicit
  input context** for each prediction,
- they are not formulated as a multi-cell context model with per-token
  split/delete heads.

### 3. Broader trajectory inference

- VIA, Nature Communications 2021
  - https://www.nature.com/articles/s41467-021-25773-3
- Review: *Concepts and limitations for learning developmental trajectories from single cell genomics*, Development 2019
  - https://journals.biologists.com/dev/article/146/12/dev170506/19458/Concepts-and-limitations-for-learning

These are useful conceptual references because they emphasize:

- snapshot data cannot directly observe trajectories,
- lineage, proliferation, and death complicate inference,
- many inferred trajectories are only partial proxies for underlying dynamics.

That is directly relevant to how our current `split` and especially `delete`
targets should be interpreted.

## What Is Distinct About This Baseline

The active baseline combines assumptions that are uncommon in combination:

- real transcriptome is the primary state,
- one input sample is a **structured multi-cell developmental context window**,
- lineage is excluded from model inputs,
- time is allowed as a condition,
- output includes both future gene-state change and event propensity.

That combination is closer to a whole-embryo developmental context model than
standard single-cell trajectory inference, while still remaining grounded in
available snapshot data.

The important boundary is:

- this baseline is evidence for one component of a whole-embryo developmental
  model,
- it is not yet evidence that embryo-scale rollout has been solved.

## First Internal Comparison: Multi-Cell Context vs Single-Cell Control

We ran a direct comparison under the same data construction and training setup:

- same `h5ad`,
- same time windows,
- same HVG count,
- same prediction task,
- same split/delete heads.

The only structural difference was:

- `GeneContextModel`: cells can use other cells in the same time window,
- `SingleCellGeneTimeModel`: each cell uses only its own gene state plus time.

### Current comparison outputs

Multi-cell context:

- checkpoint: `checkpoints_gene_context_compare/best.pt`
- evaluation: `result/gene_context/evaluation_compare_multi.json`
- result:
  - `total = 1.5694`
  - `gene = 1.4786`
  - `del = 0.0074`

Single-cell control:

- checkpoint: `checkpoints_gene_single_cell_compare/best.pt`
- evaluation: `result/gene_context/evaluation_compare_single.json`
- result:
  - `total = 2.0261`
  - `gene = 1.4762`
  - `del = 0.2628`

### Biological interpretation of this result

At this stage, the gene-state error is roughly similar between the two models,
but the multi-cell model is much better behaved on the event side,
especially the `delete` head.

That suggests:

- a single cell's own expression plus time is not enough to stabilize its
  near-future interpretation,
- the surrounding developmental population provides additional constraint,
- multi-cell transcriptomic context helps suppress implausible disappearance
  predictions.

Biologically, this is plausible:

- cells do not exist as independent transcriptomic particles,
- developmental state is constrained by the broader composition of the embryo,
- context helps distinguish true transition from spurious uncertainty.

## Current Context-Usage Result

We also ran a more targeted context-usage check using the structured
anchor-centered input format.

Setup:

- `context_size = 64`
- compare `global_context_size = 0` vs `16`
- evaluate each checkpoint in two modes:
  - `full`: full structured context available
  - `anchor_only`: only the anchor token remains visible at evaluation time

Observed result:

- both models perform better in `full` than in `anchor_only`,
- adding global background context gives a small additional gain,
- the gain appears mainly in the event-loss terms rather than in the gene MSE.

Interpretation:

- the model is not merely using anchor-local information,
- structured context is actually contributing,
- global background context is weakly but consistently useful,
- event supervision is still too weak for strong precision/recall claims, so
  the current signal is best interpreted as improved calibration rather than
  solved event prediction.

## Current Limits of Biological Interpretation

This baseline is meaningful, but it is not yet a full developmental model.

Important limitations:

- future pairing is still weak and local,
- split/delete labels are approximate,
- `delete` is not yet a clean death label,
- many validation subsets have too few positive split/delete events for strong
  event metrics,
- no explicit cell-cell signaling mechanism is modeled,
- no real whole-embryo rollout exists yet for this transcriptomic path.

So the correct interpretation today is:

- this baseline models **short-horizon developmental state transition under
  transcriptomic context**,
- not full cell-fate mechanism,
- not lineage reconstruction,
- not organism-scale developmental simulation.

## Practical Summary

If a reader asks what this active baseline means biologically, the shortest
correct answer is:

> It treats each cell's transcriptome as the main developmental state, treats
> embryo time as stage, uses other cells in the same time window as context,
> and predicts how each cell's gene program will move next, along with whether
> it is more likely to divide or disappear.

## Current Active Readout Layer

The project has now moved beyond treating token-level `gene/split/delete`
prediction as the only active benchmark.

The main active benchmark is now patch-to-patch set prediction:

- encode a local developmental patch at time `t`,
- predict the local developmental patch at `t + dt`,
- compare multi-cell and single-cell patch encoders,
- scale context size and ask when larger developmental context becomes useful.

The most important recent result is that this set-level objective is the first
one where the multi-cell model consistently outperforms the matched
single-cell control.

## Why Patch-Level Biological Readouts Were Added

A single aggregate loss was no longer enough once patch size changed.

Two problems appeared:

- raw `total` loss was strongly affected by the patch-size regression term,
- a lower total loss did not tell us whether the model was better at
  reconstructing composition, diversity, or only a coarse average.

So the active patch-set evaluation now includes:

- normalized set metrics:
  - `normalized_total`
  - `ot_per_token`
  - `total_wo_size`
- composition readouts:
  - `mean_gene_rmse`
  - `mean_gene_cosine`
  - `pca_mean_dist`
  - `pca_var_dist`
- diversity readouts:
  - `diversity_abs_error`
  - `entropy_abs_error`
- patch-level summary context:
  - `current_split_fraction`
  - `future_split_fraction`
  - `split_fraction_shift`

These are still research readouts rather than community-standard biological
benchmarks, but they are more interpretable than raw total loss alone.

## What The New Readouts Currently Say

The current patch-set comparison suggests a more specific biological picture.

Multi-cell is currently strongest on:

- set-level future-state alignment (`ot_per_token`, `latent`),
- recovering diversity and entropy of the future patch,
- recovering variance structure in PCA space.

Single-cell can still remain competitive on:

- some mean-centered summary metrics,
- especially `mean_gene_rmse` and `pca_mean_dist` on some scales.

So the present interpretation is not:

> multi-cell is better at everything.

It is:

> multi-cell is better at recovering the structure of the next local
> developmental population, while single-cell can still stay competitive on
> simpler mean-state summaries.

That distinction matters biologically because structure recovery is closer to
the final whole-embryo objective than mean-state recovery alone.

## Current Limitations Of The Active Readouts

These readouts are useful, but they are still not domain-standard endpoints.

Important limitations:

- `split_fraction` is currently a target-side summary, not yet a predicted
  biological summary head,
- diversity, entropy, and PCA distances are still expression-space proxies,
- they do not yet give explicit cell-type composition accuracy,
- they do not directly measure tissue organization or lineage consistency,
- they should be used as interpretation layers on top of the set-level
  objective, not as the sole optimization target.

So the right reading is:

- patch-set prediction is now the active modeling task,
- composition readouts improve biological interpretability,
- but they do not yet replace the need for later embryo-scale evaluation.

## Current Transition

The active transition is now:

- keep patch-set as the strongest validated local pretext task,
- stop treating patch-level attention/selection as the main architectural path,
- train a shared encoder on multiple views of the same embryo window,
- make the main objective representation consistency and future-state
  predictability rather than patch identity itself.

## First Shared-Encoder Multi-View Result

The first direct state-representation experiment is now in place.

Setup:

- same local patch encoder family as the patch-set path,
- two current-time views from the same embryo window,
- one future view from the paired future window,
- shared encoder across all views,
- losses:
  - same-time view consistency,
  - current-to-future latent prediction,
  - low-weight OT auxiliary reconstruction.

The first comparison was run at:

- `context_size = 256`
- `global_context_size = 32`
- `dt = 20` and `40`

What it showed:

- at `dt = 20`, the single-cell control is still slightly better,
- at `dt = 40`, the multi-cell model becomes better on the primary future-state
  prediction term,
- same-time view-consistency alone is not enough to distinguish the models,
  because the single-cell control can trivially collapse different views toward
  similar representations,
- the more meaningful discriminator is current-to-future latent
  predictability.

This is important because it is the first clear result in the new direction:

> multi-cell advantage becomes more visible when the task is to predict future
> state representations across a more informative time scale, not merely to
> align views at the same time.

So the next stage of the project should evaluate multi-cell models mainly by:

- future-state latent prediction,
- not same-time similarity alone,
- and not patch identity or patch selection heuristics.

## References

1. La Manno G, et al. *RNA velocity of single cells*. Nature (2018).
   https://www.nature.com/articles/s41586-018-0414-6
2. Bergen V, et al. *Generalizing RNA velocity to transient cell states through dynamical modeling*. Nature Biotechnology (2020).
   https://www.nature.com/articles/s41587-020-0591-3
3. Schiebinger G, et al. *Optimal-Transport Analysis of Single-Cell Gene Expression Identifies Developmental Trajectories in Reprogramming*. Cell (2019).
   https://pubmed.ncbi.nlm.nih.gov/30712874/
4. Tong A, et al. *TrajectoryNet: A Dynamic Optimal Transport Network for Modeling Cellular Dynamics*. ICML / PMLR (2020).
   https://proceedings.mlr.press/v119/tong20a.html
5. Yeo SK, et al. *Generative modeling of single-cell time series with PRESCIENT enables prediction of cell trajectories with interventions*. Nature Communications (2021).
   https://www.nature.com/articles/s41467-021-23518-w
6. Stassen SV, et al. *Generalized and scalable trajectory inference in single-cell omics data with VIA*. Nature Communications (2021).
   https://www.nature.com/articles/s41467-021-25773-3
7. Tritschler S, et al. *Concepts and limitations for learning developmental trajectories from single cell genomics*. Development (2019).
   https://journals.biologists.com/dev/article/146/12/dev170506/19458/Concepts-and-limitations-for-learning
