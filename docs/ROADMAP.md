# Roadmap: Whole-Organism AR + Diffusion-LLM

## Phase 1: Repository Baseline (done)

- reorganize examples into active and legacy paths
- remove stale docs and keep one canonical architecture direction

## Phase 2: Training Objective Upgrade

- add diffusion-style noise schedule in AR training loop
- train with denoising residual objective plus event supervision
- keep rollout-compatible autoregressive inference

## Phase 3: Model Upgrade

- add explicit conditioning token/embedding for `sigma_t`
- add optional classifier-free style conditioning dropout
- calibrate split/delete event thresholds over development time

## Phase 4: Evaluation

- whole-trajectory stability over long rollouts
- cross-lineage perturbation sensitivity
- division timing and cell-count realism
- gene/spatial consistency under noisy initialization

## Definition of Done

A single reproducible workflow that:

1. extracts whole-embryo trajectory,
2. trains AR diffusion-style model,
3. runs perturbation and cross-lineage evaluations,
4. reports quantitative rollout quality.
