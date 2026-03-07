# Examples Layout

This directory is organized by project status:

- `examples/whole_organism_ar/`: active scripts for the context-validation program and downstream rollout infrastructure.
  The immediate main path is the multi-cell gene-context baseline.
- `examples/legacy/`: older scripts retained for reference.

If you are starting new work, use `examples/whole_organism_ar/`.

Recommended current entry points:

- `train_gene_context.py`
- `evaluate_gene_context.py`
- `train_gene_single_cell.py`
- `evaluate_gene_single_cell.py`

Legacy note:

- `examples/legacy/whole_organism_ar/` contains older autoregressive scripts that rely on
  synthetic trajectories, explicit lineage supervision, founder-centric
  perturbation analysis, founder-centric demos/visualization, or trimodal/crossmodal checkpoints.
  They are archived and are not the current entry points.
