# Roadmap: Whole-Organism Developmental Context

## Goal

Build a whole-organism developmental model that learns biological context from data rather than from injected lineage or spatial priors.

The target system should:

1. use real developmental observations as the primary training signal,
2. treat gene state as the main developmental state,
3. model multi-cell context at each developmental time window,
4. support split/delete events without hand-coded lineage structure,
5. eventually produce stable whole-embryo rollouts.

## Current Status

What is already in place:

- whole-embryo trajectory extraction exists,
- autoregressive rollout infrastructure exists,
- split/delete event heads exist,
- real WormGUIDES spatial trajectories can be extracted,
- transcriptome-centered multimodal integration exists through the AnnData builder,
- rollout evaluation scripts exist.

What has been falsified or downgraded:

- synthetic gene or synthetic spatial patterns are not acceptable as the main path,
- spatial-only modeling is an engineering baseline, not the biological main model,
- explicit lineage embeddings as model inputs are treated as injected prior,
- inference-time top-k event rules are diagnostic tools only, not the default solution.

## Phase 1: Data Grounding

Objective: ensure the main training path is based on real data only.

Tasks:

- keep real WormGUIDES extraction available for engineering diagnostics,
- make `Large2025` the primary source for developmental gene state,
- use lineage metadata only for sample construction or evaluation, not as model input,
- keep synthetic fallback paths disabled by default,
- document which outputs are real, pseudo-assembled, or synthetic.

Definition of done:

- the default training path does not depend on synthetic gene features,
- the default training path does not depend on synthetic spatial trajectories,
- every dataset artifact has an explicit provenance label.

## Phase 2: Gene-Context Baseline

Objective: build the first biologically meaningful baseline.

Model requirements:

- primary token state is gene expression or a learned gene-state projection,
- each sample contains multiple cells from the same developmental time window,
- time is allowed as conditioning input,
- lineage is not provided as an embedding or architectural bias,
- space is optional auxiliary context, not the primary state.

Training task:

- input: multi-cell gene context at time window `t`,
- output: future gene-state transition and split/delete propensity for each cell,
- target construction may use lineage only as a weak pairing filter for future candidates.

Definition of done:

- a training script exists for multi-cell gene-context learning,
- the baseline trains on real transcriptome data,
- evaluation reports future-state prediction quality and event calibration.

## Phase 3: Context Calibration

Objective: verify that the model uses multi-cell context rather than collapsing to independent-cell averages.

Tasks:

- measure cross-cell influence within the same time window,
- compare single-cell and multi-cell baselines,
- evaluate split/delete calibration separately from continuous state prediction,
- test whether biologically coherent structure emerges in latent space without lineage inputs.

Definition of done:

- multi-cell context outperforms a single-cell baseline,
- split/delete heads are calibrated on held-out data,
- latent analyses show emergent developmental structure without explicit lineage injection.

## Phase 4: Whole-Embryo Assembly

Objective: move from local context prediction to embryo-level state evolution.

Tasks:

- construct pseudo-embryo context slices from real transcriptome time windows,
- define rollout targets that do not smuggle in lineage structure,
- integrate dynamic cell management with the gene-context model,
- report embryo-level growth, event balance, and state consistency.

Definition of done:

- the model can evolve a multi-cell embryo context across multiple steps,
- rollout does not rely on deterministic top-k event rules by default,
- embryo-level metrics are stable enough for comparison experiments.

## Phase 5: Spatial Reintegration

Objective: add spatial information back in as auxiliary signal once the gene-context path is stable.

Tasks:

- use spatial coordinates as optional context or readout,
- test whether gene-state latents predict spatial organization,
- avoid promoting spatial dynamics to the main developmental driver,
- compare gene-only and gene-plus-spatial variants.

Definition of done:

- adding spatial information improves prediction or interpretability,
- the main developmental signal remains gene-context driven,
- the model still respects the `Discover, Don't Inject` constraint.

## Phase 6: Diffusion-Style Upgrade

Objective: reintroduce diffusion-style training only after the non-synthetic gene-context baseline is working.

Tasks:

- apply denoising objectives to the validated gene-context path,
- compare plain AR and diffusion-style training on the same real-data task,
- retain rollout compatibility without inference-time heuristics.

Definition of done:

- diffusion-style training improves robustness or long-horizon stability,
- gains are demonstrated on real-data evaluation, not synthetic artifacts.

## Immediate Next Steps

1. Build a multi-cell gene-context dataset from real transcriptome data.
2. Implement a baseline model with gene state plus time, but no lineage embedding input.
3. Define evaluation for future gene-state prediction, split calibration, and delete calibration.
4. Keep spatial rollout code only as an engineering diagnostic branch.

## Non-Goals

- no default training path built on synthetic gene features,
- no default training path built on synthetic spatial patterns,
- no lineage embedding injected as model input,
- no architectural constraints that encode the developmental tree,
- no success criteria based only on spatial rollout aesthetics.
