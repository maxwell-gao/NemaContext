#!/usr/bin/env python3
"""Full autoregressive training with real Sulston trajectories.

This is the production-level training script that:
1. Loads real lineage trajectories
2. Trains with proper event supervision
3. Validates with perturbation experiments
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.branching_flows.autoregressive_model import AutoregressiveNemaModel
from src.branching_flows.dynamic_cell_manager import DynamicCellManager
from src.branching_flows.states import BranchingState


class SulstonTrajectoryDataset(Dataset):
    """Dataset of real developmental trajectories."""

    def __init__(
        self,
        trajectory_file: str,
        founder: str = "AB",
        n_hvg: int = 2000,
    ):
        with open(trajectory_file) as f:
            data = json.load(f)

        self.trajectory = data.get(founder, [])
        self.n_hvg = n_hvg

        print(f"Loaded {founder} trajectory: {len(self.trajectory)} time points")
        if self.trajectory:
            n_cells = [s["n_cells"] for s in self.trajectory]
            print(f"  Cell count: {min(n_cells)} -> {max(n_cells)}")

    def __len__(self):
        return max(0, len(self.trajectory) - 1)

    def __getitem__(self, idx):
        """Get consecutive time points."""
        current = self.trajectory[idx]
        next_state = self.trajectory[idx + 1]

        # Convert to BranchingStates
        current_state = self._state_to_branching(current)
        next_state_obj = self._state_to_branching(next_state)

        # Extract event targets
        n_current = current["n_cells"]
        target_split = torch.zeros(n_current)
        target_del = torch.zeros(n_current)

        # Mark divisions
        for div_idx in current.get("divisions", []):
            if div_idx < n_current:
                target_split[div_idx] = 1.0

        # Mark deaths
        for death_idx in current.get("deaths", []):
            if death_idx < n_current:
                target_del[death_idx] = 1.0

        return {
            "current": current_state,
            "next": next_state_obj,
            "target_split": target_split,
            "target_del": target_del,
            "time": current["time"],
        }

    def _state_to_branching(self, state: dict) -> BranchingState:
        """Convert trajectory state to BranchingState."""
        n_cells = state["n_cells"]

        # Get features
        if "genes" in state:
            genes = torch.tensor(state["genes"], dtype=torch.float32)
        else:
            genes = torch.randn(n_cells, self.n_hvg) * 0.1

        if "positions" in state:
            positions = torch.tensor(state["positions"], dtype=torch.float32)
        else:
            positions = torch.randn(n_cells, 3) * 0.1

        # Ensure correct shapes
        if genes.shape[0] != n_cells:
            genes = genes[:n_cells]
        if positions.shape[0] != n_cells:
            positions = positions[:n_cells]

        continuous = torch.cat([genes, positions], dim=-1).unsqueeze(0)
        discrete = torch.zeros(1, n_cells, dtype=torch.long)

        return BranchingState(
            states=(continuous, discrete),
            groupings=torch.zeros(1, n_cells, dtype=torch.long),
            del_flags=torch.zeros(1, n_cells, dtype=torch.bool),
            ids=torch.arange(1, n_cells + 1, dtype=torch.long).unsqueeze(0),
            padmask=torch.ones(1, n_cells, dtype=torch.bool),
            flowmask=torch.ones(1, n_cells, dtype=torch.bool),
            branchmask=torch.ones(1, n_cells, dtype=torch.bool),
        )


def compute_autoregressive_loss(
    model: AutoregressiveNemaModel,
    batch: dict,
    device: str,
) -> tuple[torch.Tensor, dict]:
    """Compute full autoregressive loss with events."""
    current = batch["current"].to(device)
    next_state = batch["next"].to(device)
    target_split = batch["target_split"].to(device)
    target_del = batch["target_del"].to(device)

    # Forward pass
    output = model.forward_step(current)

    # State prediction loss (only valid positions)
    valid_mask = current.padmask.squeeze(0)  # [L]

    # Gene delta loss
    gene_delta_pred = output.gene_delta[0, valid_mask, :]
    cont_current = current.states[0][0, valid_mask, :]
    cont_next = next_state.states[0][0, valid_mask, :]

    true_gene_delta = cont_next[..., :2000] - cont_current[..., :2000]
    gene_loss = F.mse_loss(gene_delta_pred, true_gene_delta)

    # Spatial velocity loss
    spatial_vel_pred = output.spatial_vel[0, valid_mask, :]
    true_spatial_vel = cont_next[..., 2000:2003] - cont_current[..., 2000:2003]
    spatial_loss = F.mse_loss(spatial_vel_pred, true_spatial_vel)

    # Event prediction losses
    split_logits = output.split_logits[0, valid_mask, 0]
    del_logits = output.del_logits[0, valid_mask, 0]

    split_loss = F.binary_cross_entropy_with_logits(
        split_logits, target_split[valid_mask]
    )
    del_loss = F.binary_cross_entropy_with_logits(
        del_logits, target_del[valid_mask]
    )

    # Total loss
    total_loss = gene_loss + spatial_loss + 0.5 * split_loss + 0.5 * del_loss

    loss_dict = {
        "total": total_loss.item(),
        "gene": gene_loss.item(),
        "spatial": spatial_loss.item(),
        "split": split_loss.item(),
        "del": del_loss.item(),
    }

    return total_loss, loss_dict


def train_epoch(
    model: AutoregressiveNemaModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> dict:
    """Train for one epoch."""
    model.train()

    total_losses = {
        "total": 0.0,
        "gene": 0.0,
        "spatial": 0.0,
        "split": 0.0,
        "del": 0.0,
    }
    n_batches = 0

    for batch in loader:
        optimizer.zero_grad()

        loss, loss_dict = compute_autoregressive_loss(model, batch, device)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        for k in total_losses:
            total_losses[k] += loss_dict[k]
        n_batches += 1

    return {k: v / n_batches for k, v in total_losses.items()}


@torch.no_grad()
def validate(
    model: AutoregressiveNemaModel,
    loader: DataLoader,
    device: str,
) -> dict:
    """Validate model."""
    model.eval()

    total_losses = {
        "total": 0.0,
        "gene": 0.0,
        "spatial": 0.0,
        "split": 0.0,
        "del": 0.0,
    }
    n_batches = 0

    for batch in loader:
        loss, loss_dict = compute_autoregressive_loss(model, batch, device)

        for k in total_losses:
            total_losses[k] += loss_dict[k]
        n_batches += 1

    return {k: v / n_batches for k, v in total_losses.items()}


@torch.no_grad()
def test_perturbation(
    model: AutoregressiveNemaModel,
    initial_state: BranchingState,
    device: str,
    perturb_time: int = 5,
) -> dict:
    """Test causal response to perturbation."""
    model.eval()

    # Control trajectory
    control_states = [initial_state]
    state = initial_state

    for _ in range(20):
        state, _ = model.step(state, deterministic=True, apply_events=True)
        control_states.append(state)

    # Perturbed trajectory (delete cells at perturb_time)
    perturbed_states = [initial_state]
    state = initial_state

    for t in range(20):
        if t == perturb_time:
            # Remove 20% of cells
            n_cells = state.padmask.sum().item()
            n_remove = max(1, n_cells // 5)

            # Create new state with fewer cells
            mask = torch.ones(n_cells, dtype=torch.bool)
            mask[torch.randperm(n_cells)[:n_remove]] = False

            new_cont = state.states[0][:, mask]
            new_disc = state.states[1][:, mask]

            state = BranchingState(
                states=(new_cont, new_disc),
                groupings=torch.zeros(1, n_cells - n_remove, dtype=torch.long),
                del_flags=torch.zeros(1, n_cells - n_remove, dtype=torch.bool),
                ids=torch.arange(1, n_cells - n_remove + 1, dtype=torch.long).unsqueeze(0),
                padmask=torch.ones(1, n_cells - n_remove, dtype=torch.bool),
                flowmask=torch.ones(1, n_cells - n_remove, dtype=torch.bool),
                branchmask=torch.ones(1, n_cells - n_remove, dtype=torch.bool),
            )

        state, _ = model.step(state, deterministic=True, apply_events=True)
        perturbed_states.append(state)

    # Compare final states
    control_final = control_states[-1]
    perturbed_final = perturbed_states[-1]

    control_pos = control_final.states[0][0, :, -3:].mean(dim=0)
    perturbed_pos = perturbed_final.states[0][0, :, -3:].mean(dim=0)

    distance = torch.norm(control_pos - perturbed_pos).item()

    return {
        "control_cells": int(control_final.padmask.sum()),
        "perturbed_cells": int(perturbed_final.padmask.sum()),
        "spatial_distance": distance,
        "compensated": distance < 1.0,  # Arbitrary threshold
    }


def main():
    parser = argparse.ArgumentParser(description="Full autoregressive training")
    parser.add_argument("--trajectory_file", type=str, required=True)
    parser.add_argument("--founder", type=str, default="AB")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="checkpoints_autoregressive_full")
    args = parser.parse_args()

    print("=" * 70)
    print("FULL AUTOREGRESSIVE TRAINING")
    print("=" * 70)
    print(f"Trajectory file: {args.trajectory_file}")
    print(f"Founder: {args.founder}")
    print()

    # Load dataset
    print("Loading dataset...")
    dataset = SulstonTrajectoryDataset(args.trajectory_file, args.founder)

    if len(dataset) == 0:
        print("ERROR: No trajectory data found!")
        print("Run trajectory_extractor.py first:")
        print("  uv run python src/data/trajectory_extractor.py")
        return

    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    print(f"  Train: {len(train_dataset)} steps")
    print(f"  Val: {len(val_dataset)} steps")
    print()

    # Create model
    print("Creating model...")
    model = AutoregressiveNemaModel(
        gene_dim=2000,
        spatial_dim=3,
        discrete_K=7,
        d_model=256,
        n_layers=6,
        n_heads=8,
        cross_modal_every=2,
        max_seq_len=128,
        dt=0.1,
    ).to(args.device)

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # Training loop
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    best_val_loss = float("inf")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_losses = train_epoch(model, train_loader, optimizer, args.device)
        val_losses = validate(model, val_loader, args.device)
        scheduler.step()

        print(f"Epoch {epoch:3d}:")
        print(f"  Train: total={train_losses['total']:.4f}, "
              f"gene={train_losses['gene']:.4f}, "
              f"split={train_losses['split']:.4f}")
        print(f"  Val:   total={val_losses['total']:.4f}, "
              f"gene={val_losses['gene']:.4f}, "
              f"split={val_losses['split']:.4f}")

        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_losses,
                },
                save_dir / "best.pt",
            )
            print("  ✓ Saved best model")

    print()
    print("=" * 70)
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")

    # Final perturbation test
    print()
    print("Testing causal response...")
    initial = dataset[0]["current"].to(args.device)
    perturb_result = test_perturbation(model, initial, args.device)

    print(f"  Control cells: {perturb_result['control_cells']}")
    print(f"  Perturbed cells: {perturb_result['perturbed_cells']}")
    print(f"  Spatial distance: {perturb_result['spatial_distance']:.4f}")
    if perturb_result["compensated"]:
        print("  ✓ Model showed compensatory behavior")
    else:
        print("  ✗ No compensation detected")

    print("=" * 70)


if __name__ == "__main__":
    main()
