# Examples Layout

This directory is organized by project status and scale.

- `examples/whole_organism_ar/`: active scripts for the current path from local
  population-update validation toward later embryo-scale rollout.
- `examples/legacy/`: older scripts retained for reference.

If you are starting new work, use `examples/whole_organism_ar/`.

Current interpretation:

- the final target is whole-organism developmental prediction,
- the active executable benchmark is still the local transcriptomic
  `gene-context` update task,
- downstream rollout scripts remain important, but they are not yet the main
  evidence source.

Recommended current entry points:

- `train_gene_context.py`
- `evaluate_gene_context.py`
- `train_gene_single_cell.py`
- `evaluate_gene_single_cell.py`

Downstream but not current-first entry points:

- `train_autoregressive_full.py`
- `evaluate_rollout.py`
- `train_spatial_rollout.py`
- `evaluate_spatial_rollout.py`

Legacy note:

- `examples/legacy/whole_organism_ar/` contains older autoregressive scripts that rely on
  synthetic trajectories, explicit lineage supervision, founder-centric
  perturbation analysis, founder-centric demos/visualization, or trimodal/crossmodal checkpoints.
  They are archived and are not the current primary entry points.
