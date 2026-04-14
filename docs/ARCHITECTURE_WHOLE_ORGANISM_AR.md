# Whole-Organism AR Architecture

## Status

This document is now historical architecture context, not a description of the current active implementation.

The long-term project target still includes whole-organism developmental prediction with dynamic population updates, but the repository is not currently executing that path as the main evidence route.

Current mainline:
- raw Large2025 transcriptome data,
- lineage-first whole-embryo tokenization,
- dynamics-first future prediction,
- worm-native forecasting benchmark.

Use `docs/WORM_MAINLINE.md` for the active contract.

## What This Document Still Means

This file should now be read as:
- a long-term architectural direction,
- a record of how the project previously framed whole-organism rollout,
- not a guarantee that the modules named here still exist in active `src/`.

Several old rollout-oriented modules referenced in earlier versions of this document have been removed from the active source tree during the worm-mainline cleanup.

## Current Interpretation

The repo is presently validating:
- whole-embryo gene dynamics,
- lineage-conditioned future prediction,
- worm-native forecasting metrics,
- generic forecasting baselines (`scNODE`-style and `PRESCIENT`-style).

It is not presently validating:
- full dynamic split/delete rollout,
- full autoregressive embryo simulation,
- synthetic trajectory extraction as an active benchmark path.

## Relationship To The Active Mainline

The current lineage-first worm mainline is the precursor stage for any future whole-organism spatial or autoregressive system.

The intended progression is:
1. validate whole-embryo gene dynamics,
2. improve founder/region-consistent prediction,
3. add limited spatial alignment using WormGUIDES/CShaper,
4. only then return to full organism-scale spatial or rollout prediction.

## Archival Note

If you need the old rollout-era scripts, models, or experiments, look in `examples/legacy/` and the remaining historical docs.
They are no longer part of the active implementation contract.
