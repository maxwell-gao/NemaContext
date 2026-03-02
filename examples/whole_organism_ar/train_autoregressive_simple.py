#!/usr/bin/env python3
"""
Simple training script for autoregressive model.

Uses synthetic short trajectories to validate the architecture.
Real trajectory data requires processing WormGUIDES time-series.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.branching_flows.autoregressive_model import (
    AutoregressiveNemaModel,
    autoregressive_loss,
)
from src.branching_flows.states import BranchingState


class SyntheticTrajectoryDataset(Dataset):
    """Generate synthetic short trajectories for testing.

    In real implementation, this would load actual WormGUIDES time-series data.
    """

    def __init__(
        self,
        n_trajectories: int = 100,
        trajectory_length: int = 10,
        n_cells: int = 20,
        gene_dim: int = 2000,
    ):
        self.n_trajectories = n_trajectories
        self.trajectory_length = trajectory_length
        self.n_cells = n_cells
        self.gene_dim = gene_dim

    def __len__(self):
        return self.n_trajectories

    def __getitem__(self, idx):
        """Generate a synthetic trajectory."""
        # Start with random initial state
        genes = torch.randn(self.trajectory_length, self.n_cells, self.gene_dim) * 0.1
        spatial = torch.randn(self.trajectory_length, self.n_cells, 3) * 0.1

        # Add some smooth dynamics
        for t in range(1, self.trajectory_length):
            # Smooth evolution with momentum
            genes[t] = genes[t - 1] + torch.randn_like(genes[t]) * 0.01
            spatial[t] = spatial[t - 1] + torch.randn_like(spatial[t]) * 0.01

        # Add global drift (emulating development)
        spatial[:, :, 2] += torch.linspace(0, 1, self.trajectory_length).view(-1, 1)

        continuous = torch.cat([genes, spatial], dim=-1)
        discrete = torch.zeros(self.trajectory_length, self.n_cells, dtype=torch.long)

        # Create BranchingStates
        states = []
        for t in range(self.trajectory_length):
            states.append(
                BranchingState(
                    states=(continuous[t].unsqueeze(0), discrete[t].unsqueeze(0)),
                    groupings=torch.zeros(1, self.n_cells, dtype=torch.long),
                    del_flags=torch.zeros(1, self.n_cells, dtype=torch.bool),
                    ids=torch.arange(1, self.n_cells + 1, dtype=torch.long).unsqueeze(
                        0
                    ),
                    padmask=torch.ones(1, self.n_cells, dtype=torch.bool),
                    flowmask=torch.ones(1, self.n_cells, dtype=torch.bool),
                    branchmask=torch.ones(1, self.n_cells, dtype=torch.bool),
                )
            )

        return states


def collate_states(batch):
    """Collate list of trajectories."""
    # batch: list of trajectory_length BranchingStates
    # Return as-is for trajectory training
    return batch


def train_epoch(model, loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for trajectories in loader:
        optimizer.zero_grad()

        # trajectories: list of B trajectories
        batch_loss = 0.0
        n_steps = 0

        for trajectory in trajectories:
            # trajectory: list of T BranchingStates
            trajectory_loss = 0.0

            for t in range(len(trajectory) - 1):
                current = trajectory[t].to(device)
                next_state = trajectory[t + 1].to(device)

                # Forward
                output = model.forward_step(current)

                # Loss
                loss, _ = autoregressive_loss(output, next_state, current)
                trajectory_loss += loss
                n_steps += 1

            batch_loss += trajectory_loss / (len(trajectory) - 1)

        batch_loss = batch_loss / len(trajectories)

        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += batch_loss.item()
        n_batches += 1

    return total_loss / n_batches if n_batches > 0 else 0.0


@torch.no_grad()
def validate(model, loader, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for trajectories in loader:
        batch_loss = 0.0

        for trajectory in trajectories:
            trajectory_loss = 0.0

            for t in range(len(trajectory) - 1):
                current = trajectory[t].to(device)
                next_state = trajectory[t + 1].to(device)

                output = model.forward_step(current)
                loss, _ = autoregressive_loss(output, next_state, current)
                trajectory_loss += loss.item()

            batch_loss += trajectory_loss / (len(trajectory) - 1)

        total_loss += batch_loss / len(trajectories)
        n_batches += 1

    return total_loss / n_batches if n_batches > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Train autoregressive model")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="checkpoints_autoregressive")
    args = parser.parse_args()

    print("=" * 70)
    print("TRAINING AUTOREGRESSIVE DEVELOPMENT MODEL")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print()

    # Create datasets
    print("Creating synthetic datasets...")
    train_dataset = SyntheticTrajectoryDataset(n_trajectories=100)
    val_dataset = SyntheticTrajectoryDataset(n_trajectories=20)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_states,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_states,
    )
    print(f"  Train: {len(train_dataset)} trajectories")
    print(f"  Val: {len(val_dataset)} trajectories")
    print()

    # Create model
    print("Creating model...")
    model = AutoregressiveNemaModel(
        gene_dim=2000,
        spatial_dim=3,
        discrete_K=7,
        d_model=128,  # Smaller for testing
        n_layers=4,
        n_heads=4,
        cross_modal_every=2,
        dt=0.1,
    ).to(args.device)

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Training loop
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    best_val_loss = float("inf")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, args.device)
        val_loss = validate(model, val_loader, args.device)

        print(f"Epoch {epoch:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                save_dir / "best.pt",
            )

    print()
    print("=" * 70)
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to: {save_dir / 'best.pt'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
