# Whole-Organism AR Architecture (Diffusion-LLM Style)

## Objective

Model development as a **whole-organism autoregressive process**:

- state at time `t`: all currently alive cells in one shared embryo context
- update: predict next state from current state only
- support dynamic population changes via division/deletion events

Formally:

`X_{t+1} = X_t + f_theta(X_t, c_t)`

where `X_t` is the full embryo state and `c_t` is optional conditioning (time, noise level, schedule).

## Whole-Organism Context Contract

Each timestep contains all lineages together:

- `n_cells`
- `cell_names`
- `founders`, `founder_ids`
- `positions` in global embryo coordinates
- `genes`
- event labels (`divisions`, optional `deaths`)

No per-founder isolated trajectory is used in the primary pipeline.

## Diffusion-LLM Style Update

The model remains autoregressive in time, but each step uses diffusion-style denoising behavior:

1. Sample a noise level `sigma_t` from a schedule.
2. Corrupt current token states to create `X_t_noisy`.
3. Predict denoised residual / velocity (`v`-style or `epsilon`-style target).
4. Integrate one AR step with event heads for split/delete.

This gives LLM-like iterative next-state generation with diffusion-inspired denoising stability.

## Current Core Modules

- `src/branching_flows/autoregressive_model.py`
  - transformer backbone
  - heads for gene delta, spatial velocity, discrete state, split, delete
- `src/branching_flows/dynamic_cell_manager.py`
  - event sampling and application
- `src/data/trajectory_extractor.py`
  - unified whole-embryo trajectory source

## Immediate Implementation Direction

- keep `AutoregressiveNemaModel` as the base class
- add a noise-conditioning pathway and training objective for denoising residual prediction
- preserve whole-organism event supervision (split/delete)
- evaluate cross-lineage influence under perturbation
