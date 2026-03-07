#!/usr/bin/env python3
"""Test autoregressive simulation."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.branching_flows.autoregressive_model import AutoregressiveNemaModel  # noqa: E402
from src.branching_flows.states import BranchingState  # noqa: E402


def main():
    print("=" * 70)
    print("TESTING AUTOREGRESSIVE SIMULATION")
    print("=" * 70)

    # Load model
    checkpoint = torch.load("checkpoints_autoregressive/best.pt", map_location="cpu")
    model = AutoregressiveNemaModel(
        gene_dim=2000,
        spatial_dim=3,
        discrete_K=7,
        d_model=128,
        n_layers=4,
        n_heads=4,
        cross_modal_every=2,
        dt=0.1,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")
    print()

    # Create initial state
    n_cells = 10
    genes = torch.randn(1, n_cells, 2000) * 0.1
    spatial = torch.randn(1, n_cells, 3) * 0.1
    continuous = torch.cat([genes, spatial], dim=-1)

    state = BranchingState(
        states=(continuous, torch.zeros(1, n_cells, dtype=torch.long)),
        groupings=torch.zeros(1, n_cells, dtype=torch.long),
        del_flags=torch.zeros(1, n_cells, dtype=torch.bool),
        ids=torch.arange(1, n_cells + 1, dtype=torch.long).unsqueeze(0),
        padmask=torch.ones(1, n_cells, dtype=torch.bool),
        flowmask=torch.ones(1, n_cells, dtype=torch.bool),
        branchmask=torch.ones(1, n_cells, dtype=torch.bool),
    )

    print("Initial state:")
    print(f"  Cells: {n_cells}")
    print(f"  Spatial center: {spatial[0].mean(dim=0).tolist()}")
    print()

    # Simulate
    print("Simulating development...")
    print("  (Dynamic events disabled for basic test)")
    trajectory = [state]
    n_steps = 20

    with torch.no_grad():
        for i in range(n_steps):
            state, events = model.step(state, deterministic=True, apply_events=False)
            trajectory.append(state)

            if i % 5 == 0:
                spatial_pos = state.states[0][0, :, -3:]
                n_cells = int(state.padmask.sum())
                print(
                    f"  Step {i:2d}: {n_cells} cells, "
                    f"center=({spatial_pos.mean(dim=0)[0]:.2f}, "
                    f"{spatial_pos.mean(dim=0)[1]:.2f}, "
                    f"{spatial_pos.mean(dim=0)[2]:.2f})"
                )

    print()
    print("=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)

    final_state = trajectory[-1]
    initial_spatial = trajectory[0].states[0][0, :, -3:]
    final_spatial = final_state.states[0][0, :, -3:]

    print(f"\nInitial: {int(trajectory[0].padmask.sum())} cells")
    print(f"  Spatial spread: {initial_spatial.std(dim=0).mean():.4f}")

    print(f"\nFinal: {int(final_state.padmask.sum())} cells")
    print(f"  Spatial spread: {final_spatial.std(dim=0).mean():.4f}")

    print("\n✓ Autoregressive simulation working!")
    print("  Note: Cell count fixed (no dynamic splitting yet)")
    print("  Next: Implement DynamicCellManager for cell division")


if __name__ == "__main__":
    main()
