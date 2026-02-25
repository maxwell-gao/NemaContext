#!/usr/bin/env python3
"""Test dynamic cell simulation with division and deletion."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.branching_flows.autoregressive_model import AutoregressiveNemaModel
from src.branching_flows.states import BranchingState


def main():
    print("=" * 70)
    print("TESTING DYNAMIC CELL SIMULATION")
    print("=" * 70)

    # Load model
    checkpoint_path = "checkpoints_autoregressive_dynamic/best.pt"
    if not Path(checkpoint_path).exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Run train_autoregressive_dynamic.py first")
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = AutoregressiveNemaModel(
        gene_dim=2000,
        spatial_dim=3,
        discrete_K=7,
        d_model=128,
        n_layers=4,
        n_heads=4,
        cross_modal_every=2,
        max_seq_len=64,
        dt=0.1,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")
    print()

    # Test 1: Start with single cell, enable division
    print("TEST 1: Single cell with dynamic events")
    print("-" * 70)

    n_cells = 1
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

    print(f"Initial: {n_cells} cell(s)")

    # Simulate with events
    with torch.no_grad():
        for i in range(20):
            state, events = model.step(state, deterministic=True, apply_events=True)
            n_cells = int(state.padmask.sum())

            if i % 4 == 0:
                print(f"  Step {i:2d}: {n_cells} cells")

            if n_cells == 0:
                print("  All cells deleted!")
                break

    print()

    # Test 2: Start with multiple cells
    print("TEST 2: 5 cells with dynamic events")
    print("-" * 70)

    n_cells = 5
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

    print(f"Initial: {n_cells} cells")

    with torch.no_grad():
        for i in range(20):
            state, events = model.step(state, deterministic=True, apply_events=True)
            n_cells = int(state.padmask.sum())

            if i % 4 == 0:
                split_probs = events.split_probs[0, :n_cells].mean().item() if events else 0
                del_probs = events.del_probs[0, :n_cells].mean().item() if events else 0
                print(f"  Step {i:2d}: {n_cells} cells (split_p={split_probs:.3f}, del_p={del_probs:.3f})")

            if n_cells == 0:
                print("  All cells deleted!")
                break

    print()
    print("=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print()
    print("Notes:")
    print("  - Cell division requires training with actual division events")
    print("  - Current model may not have learned correct division timing")
    print("  - Next: Train on real Sulston lineage trees")


if __name__ == "__main__":
    main()
