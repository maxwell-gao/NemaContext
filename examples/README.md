# Examples Layout

This directory is now split by project status.

- `examples/whole_organism_ar/`: active worm-mainline scripts only.
- `examples/legacy/`: archived historical scripts.

## Active Entry Points

If you are doing current work, start here:
- `train_large2025_lineage_stage1.py`
- `benchmark_worm_dynamics.py`
- `benchmark_worm_scnode.py`
- `benchmark_worm_prescient.py`

These scripts define the current executable path:
- lineage-first whole-embryo dynamics on raw Large2025,
- worm-native forecasting evaluation,
- generic forecasting baseline comparison.

## Legacy Material

Everything else has been moved under `examples/legacy/`.
That includes:
- old local patch and gene-context baselines,
- embryo future-set and masked-view paths,
- autoregressive rollout experiments,
- older trimodal/crossmodal work.

Legacy scripts are retained for historical reference only. They should not be treated as the default starting point for new work.
