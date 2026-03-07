# Whole-Organism AR Architecture

## Status

This document now describes a downstream architecture, not the immediate main
research path.

The active scientific path is the transcriptomic `gene-context` baseline:

- anchor-centered multi-cell context,
- short-horizon future gene-state prediction,
- supervision analysis before long-horizon generation.

The whole-organism autoregressive stack remains important as infrastructure for
later phases, especially once event supervision is stronger.

## What This Architecture Is For

Model development as a repeated embryo-state update:

- state at time `t`: all currently alive cells in one shared embryo context,
- update: predict the next state from the current state,
- support dynamic population changes via division/deletion events.

Formally:

`X_{t+1} = X_t + f_theta(X_t, c_t)`

where `X_t` is the full embryo state and `c_t` is optional conditioning such as
time, schedule, or noise level.

## Why It Is Not The Current Main Path

The blocker is not architecture. It is supervision quality.

Current limitations:

- split/delete targets are still weakly constructed,
- event positives are sparse in many validation slices,
- long-horizon rollout would currently amplify supervision error,
- rollout quality would therefore be difficult to interpret biologically.

Because of this, the roadmap now places:

1. context validation first,
2. supervision repair second,
3. whole-organism rollout only after both are in better shape.

## Whole-Organism Context Contract

If this path is activated later, each timestep should contain all lineages
together:

- `n_cells`
- `cell_names`
- `positions` in global embryo coordinates
- `genes`
- event labels such as `divisions` and optional `deaths`

Optional metadata may include:

- `founders`
- `founder_ids`

These remain bookkeeping or evaluation metadata. They are not part of the
desired biological input contract for the active model path.

## Current Core Modules

- `src/branching_flows/autoregressive_model.py`
  - transformer backbone for embryo-level autoregressive updates
- `src/branching_flows/dynamic_cell_manager.py`
  - event sampling and application
- `src/data/trajectory_extractor.py`
  - whole-embryo trajectory extraction for engineering diagnostics

Archived synthetic and per-founder extraction paths live in
`src/data/legacy/trajectory_extractor.py` and are not part of the active
scientific route.

## Relationship To The Active Gene-Context Path

The intended order is:

1. learn a strong context model on transcriptomic windows,
2. validate that context is genuinely used,
3. strengthen split/delete supervision,
4. port the resulting predictor into embryo-level dynamic rollout.

So this architecture should currently be read as:

- important,
- retained,
- still under development,
- but downstream of the context-validation work.

## Future Re-entry Criteria

This architecture becomes the main path again only if:

- context experiments keep showing stable benefit,
- event supervision becomes informative on held-out data,
- multi-step rollout can be evaluated without relying on heuristic event rules,
- rollout metrics become scientifically interpretable rather than only visually
  plausible.
