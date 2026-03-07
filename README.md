# NemaContext

NemaContext is focused on one goal:

**Build a whole-organism developmental autoregressive model, updated in a diffusion-LLM style.**

## Current Direction

- Context unit: **entire embryo state at each timestep** (not isolated founder trajectories).
- Dynamics: **autoregressive next-state updates** with dynamic division/deletion events.
- Training upgrade: **diffusion-style denoising objective** integrated into AR learning.

## Repository Layout

- `src/branching_flows/`: core modeling code (`autoregressive_model.py`, `dynamic_cell_manager.py`, probes)
- `src/data/`: whole-embryo trajectory extraction and data pipeline
- `examples/whole_organism_ar/`: active scripts
- `examples/legacy/`: previous-generation experiments (reference only)
- `docs/`: canonical architecture and roadmap

See:

- `docs/ARCHITECTURE_WHOLE_ORGANISM_AR.md`
- `docs/REPO_STRUCTURE.md`
- `docs/ROADMAP.md`

## Quick Start

```bash
# 1) Extract whole-embryo trajectory
uv run python src/data/trajectory_extractor.py \
  --output dataset/processed/embryo_trajectory.json \
  --max_time 400 \
  --time_resolution 10

# 2) Train whole-organism AR model
uv run python examples/whole_organism_ar/train_autoregressive_full.py \
  --trajectory_file dataset/processed/embryo_trajectory.json \
  --epochs 100

# 3) Evaluate rollout behavior
uv run python examples/whole_organism_ar/evaluate_rollout.py \
  --trajectory_file dataset/processed/embryo_trajectory.json \
  --checkpoint checkpoints_autoregressive_full/best.pt \
  --output result/autoregressive_results/evaluation_rollout.json
```
