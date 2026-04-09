#!/usr/bin/env python3
"""Train autoregressive model with dynamic cell management.

Uses synthetic trajectories with known cell divisions.
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

from src.branching_flows.autoregressive_model import AutoregressiveNemaModel  # noqa: E402
from src.branching_flows.states import BranchingState  # noqa: E402


class DynamicTrajectoryDataset(Dataset):
    """Generate synthetic trajectories with cell division events.

    Simulates: 1 cell -> 2 cells -> 4 cells -> ...
    """

    def __init__(
        self,
        n_trajectories: int = 100,
        max_steps: int = 20,
        gene_dim: int = 2000,
    ):
        self.n_trajectories = n_trajectories
        self.max_steps = max_steps
        self.gene_dim = gene_dim

    def __len__(self):
        return self.n_trajectories

    def __getitem__(self, idx):
        """Generate a synthetic trajectory with cell divisions."""
        # Start with single cell
        n_cells = 1
        genes = torch.randn(n_cells, self.gene_dim) * 0.1
        spatial = torch.randn(n_cells, 3) * 0.1

        trajectory = []

        for step in range(self.max_steps):
            # Store current state
            continuous = torch.cat([genes, spatial], dim=-1).unsqueeze(0)
            discrete = torch.zeros(1, n_cells, dtype=torch.long)

            state = BranchingState(
                states=(continuous, discrete),
                groupings=torch.zeros(1, n_cells, dtype=torch.long),
                del_flags=torch.zeros(1, n_cells, dtype=torch.bool),
                ids=torch.arange(1, n_cells + 1, dtype=torch.long).unsqueeze(0),
                padmask=torch.ones(1, n_cells, dtype=torch.bool),
                flowmask=torch.ones(1, n_cells, dtype=torch.bool),
                branchmask=torch.ones(1, n_cells, dtype=torch.bool),
            )

            # Store with target events
            # Division happens every 5 steps until max capacity
            should_divide = (step % 5 == 4) and (n_cells < 32)
            target_split = torch.zeros(1, n_cells)
            target_del = torch.zeros(1, n_cells)

            if should_divide:
                target_split[0, :] = 1.0  # All cells divide

            trajectory.append(
                {
                    "state": state,
                    "target_split": target_split,
                    "target_del": target_del,
                    "n_cells": n_cells,
                }
            )

            # Simulate next state (with division if scheduled)
            if should_divide:
                # Each cell divides into 2
                new_genes = []
                new_spatial = []
                for i in range(n_cells):
                    # Daughter cells with small perturbation
                    new_genes.extend(
                        [
                            genes[i] + torch.randn_like(genes[i]) * 0.01,
                            genes[i] + torch.randn_like(genes[i]) * 0.01,
                        ]
                    )
                    new_spatial.extend(
                        [
                            spatial[i] + torch.randn_like(spatial[i]) * 0.01,
                            spatial[i] + torch.randn_like(spatial[i]) * 0.01,
                        ]
                    )
                genes = torch.stack(new_genes)
                spatial = torch.stack(new_spatial)
                n_cells *= 2
            else:
                # Just add movement
                genes = genes + torch.randn_like(genes) * 0.01
                spatial = spatial + torch.randn_like(spatial) * 0.01

        return trajectory


def collate_trajectories(batch):
    """Collate list of trajectories."""
    return batch


def train_epoch(model, loader, optimizer, device):
    """Train for one epoch with dynamic cells."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for trajectories in loader:
        optimizer.zero_grad()

        batch_loss = 0.0
        n_valid = 0

        for trajectory in trajectories:
            traj_loss = 0.0

            for t in range(len(trajectory) - 1):
                current_data = trajectory[t]
                next_data = trajectory[t + 1]

                current = current_data["state"].to(device)
                target_split = current_data["target_split"].to(device)
                target_del = current_data["target_del"].to(device)

                # Forward
                output = model.forward_step(current)

                # State prediction loss
                next_state = next_data["state"].to(device)

                # For dynamic cells, we need to handle variable lengths
                # For now, compute loss only on positions that exist in both states
                min_len = min(current.states[0].shape[1], next_state.states[0].shape[1])

                # Truncate to minimum length for loss computation
                gene_delta_pred = output.gene_delta[:, :min_len, :]
                spatial_vel_pred = output.spatial_vel[:, :min_len, :]

                cont_current = current.states[0][:, :min_len, :]
                cont_next = next_state.states[0][:, :min_len, :]

                true_gene_delta = cont_next[..., :2000] - cont_current[..., :2000]
                true_spatial_vel = (
                    cont_next[..., 2000:2003] - cont_current[..., 2000:2003]
                )

                state_loss = F.mse_loss(gene_delta_pred, true_gene_delta) + F.mse_loss(
                    spatial_vel_pred, true_spatial_vel
                )

                # Event prediction loss
                if output.split_logits is not None:
                    split_probs = torch.sigmoid(output.split_logits.squeeze(-1))
                    del_probs = torch.sigmoid(output.del_logits.squeeze(-1))

                    # Mask to valid cells
                    valid_mask = current.padmask

                    split_loss = (
                        F.binary_cross_entropy(
                            split_probs[valid_mask],
                            target_split[valid_mask],
                        )
                        if valid_mask.any()
                        else torch.tensor(0.0, device=device)
                    )

                    del_loss = (
                        F.binary_cross_entropy(
                            del_probs[valid_mask],
                            target_del[valid_mask],
                        )
                        if valid_mask.any()
                        else torch.tensor(0.0, device=device)
                    )

                    event_loss = split_loss + del_loss
                else:
                    event_loss = torch.tensor(0.0, device=device)

                step_loss = state_loss + 0.1 * event_loss
                traj_loss += step_loss
                n_valid += 1

            if n_valid > 0:
                batch_loss += traj_loss / n_valid

        if len(trajectories) > 0:
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
            traj_loss = 0.0
            n_valid = 0

            for t in range(len(trajectory) - 1):
                current_data = trajectory[t]
                next_data = trajectory[t + 1]

                current = current_data["state"].to(device)
                next_state = next_data["state"].to(device)

                output = model.forward_step(current)

                # Handle variable lengths
                min_len = min(current.states[0].shape[1], next_state.states[0].shape[1])
                gene_delta_pred = output.gene_delta[:, :min_len, :]
                spatial_vel_pred = output.spatial_vel[:, :min_len, :]

                cont_current = current.states[0][:, :min_len, :]
                cont_next = next_state.states[0][:, :min_len, :]

                true_gene_delta = cont_next[..., :2000] - cont_current[..., :2000]
                true_spatial_vel = (
                    cont_next[..., 2000:2003] - cont_current[..., 2000:2003]
                )

                loss = F.mse_loss(gene_delta_pred, true_gene_delta) + F.mse_loss(
                    spatial_vel_pred, true_spatial_vel
                )
                traj_loss += loss.item()
                n_valid += 1

            if n_valid > 0:
                batch_loss += traj_loss / n_valid

        total_loss += batch_loss / len(trajectories)
        n_batches += 1

    return total_loss / n_batches if n_batches > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Train autoregressive model with dynamic cells"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--save_dir", type=str, default="checkpoints/autoregressive_dynamic"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("TRAINING AUTOREGRESSIVE MODEL WITH DYNAMIC CELLS")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print()

    # Create datasets
    print("Creating datasets with cell divisions...")
    train_dataset = DynamicTrajectoryDataset(n_trajectories=50)
    val_dataset = DynamicTrajectoryDataset(n_trajectories=10)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_trajectories,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_trajectories,
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
        d_model=128,
        n_layers=4,
        n_heads=4,
        cross_modal_every=2,
        max_seq_len=64,  # Support up to 64 cells
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
