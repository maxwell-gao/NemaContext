#!/usr/bin/env python3
"""Train autoregressive model with strong cell division supervision.

This training script focuses on teaching the model when cells divide
by providing explicit supervision from the Sulston lineage tree.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.branching_flows.autoregressive_model import AutoregressiveNemaModel
from src.branching_flows.states import BranchingState


class DivisionSupervisedDataset(Dataset):
    """Dataset with explicit cell division supervision.

    Uses Sulston lineage tree to determine which cells divide at each timestep.
    """

    def __init__(self, trajectory_file: str, n_hvg: int = 2000):
        with open(trajectory_file) as f:
            self.trajectory = json.load(f)

        self.n_hvg = n_hvg

        # Build division targets from trajectory
        self.samples = self._build_samples()

        print(f"Loaded {len(self.samples)} supervised samples")

    def _build_samples(self) -> list[dict]:
        """Build training samples with division labels."""
        samples = []

        for i in range(len(self.trajectory) - 1):
            current = self.trajectory[i]
            next_state = self.trajectory[i + 1]

            n_current = current["n_cells"]
            n_next = next_state["n_cells"]

            # Determine which cells divided
            # A cell divided if:
            # 1. Next timestep has more cells
            # 2. We can match parent to children by name

            target_split = torch.zeros(n_current)

            if n_next > n_current:
                # Some cells divided - try to identify them
                current_names = current.get("cell_names", [])
                next_names = next_state.get("cell_names", [])

                # Simple heuristic: cells that appear in both are non-dividing
                # New cells (children) have longer names (parent + suffix)
                _ = set(current_names)  # Unused for now

                for j, name in enumerate(current_names):
                    # Check if this cell's children appear in next state
                    # Child names are typically parent + 'a' or 'p'
                    child1 = name + "a"
                    child2 = name + "p"

                    if child1 in next_names or child2 in next_names:
                        target_split[j] = 1.0

            samples.append(
                {
                    "idx": i,
                    "current": current,
                    "next": next_state,
                    "target_split": target_split,
                    "time": current["time"],
                    "n_current": n_current,
                    "n_next": n_next,
                }
            )

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        current_state = self._state_to_branching(sample["current"])
        next_state = self._state_to_branching(sample["next"])

        return {
            "current": current_state,
            "next": next_state,
            "target_split": sample["target_split"],
            "time": sample["time"],
            "n_current": sample["n_current"],
            "n_next": sample["n_next"],
        }

    def _state_to_branching(self, state: dict) -> BranchingState:
        """Convert trajectory state to BranchingState."""
        n_cells = state["n_cells"]

        if "genes" in state:
            genes = torch.tensor(state["genes"], dtype=torch.float32)
        else:
            genes = torch.randn(n_cells, self.n_hvg) * 0.1

        if "positions" in state:
            positions = torch.tensor(state["positions"], dtype=torch.float32)
        else:
            positions = torch.randn(n_cells, 3) * 0.1

        if genes.shape[0] != n_cells:
            genes = genes[:n_cells]
        if positions.shape[0] != n_cells:
            positions = positions[:n_cells]

        continuous = torch.cat([genes, positions], dim=-1).unsqueeze(0)

        if "founder_ids" in state:
            founder_ids = state["founder_ids"]
            discrete = torch.tensor(founder_ids, dtype=torch.long).unsqueeze(0)
        else:
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


def compute_loss_with_division_supervision(
    model: AutoregressiveNemaModel,
    batch: dict,
    device: str,
    split_weight: float = 2.0,  # Higher weight for division prediction
) -> tuple[torch.Tensor, dict]:
    """Compute loss with strong division supervision."""

    currents = batch["current"]
    next_states = batch["next"]
    target_splits = batch["target_split"]

    if isinstance(currents, BranchingState):
        currents = [currents]
        next_states = [next_states]
        target_splits = [target_splits]

    total_loss = torch.tensor(0.0, device=device)
    all_losses = {"gene": [], "spatial": [], "split": [], "count": []}

    for current, next_state, target_split in zip(currents, next_states, target_splits):
        current = current.to(device)
        next_state = next_state.to(device)
        target_split = target_split.to(device)

        # Forward pass
        output = model.forward_step(current)

        # State prediction loss
        n_current = current.states[0].shape[1]
        n_next = next_state.states[0].shape[1]
        n_common = min(n_current, n_next)

        valid_mask = current.padmask.squeeze(0)[:n_common]

        # Gene delta loss
        gene_delta_pred = output.gene_delta[0, :n_common, :][valid_mask, :]
        cont_current = current.states[0][0, :n_common, :][valid_mask, :]
        cont_next = next_state.states[0][0, :n_common, :][valid_mask, :]

        true_gene_delta = cont_next[..., :2000] - cont_current[..., :2000]
        gene_loss = F.mse_loss(gene_delta_pred, true_gene_delta)

        # Spatial velocity loss
        spatial_vel_pred = output.spatial_vel[0, :n_common, :][valid_mask, :]
        true_spatial_vel = cont_next[..., 2000:2003] - cont_current[..., 2000:2003]
        spatial_loss = F.mse_loss(spatial_vel_pred, true_spatial_vel)

        # Division prediction loss (with high weight)
        split_logits = output.split_logits[0, :n_current, 0]
        target_split_cropped = target_split[:n_current]

        # Apply mask to only compute loss on valid cells
        if valid_mask.shape[0] == target_split_cropped.shape[0]:
            split_loss = F.binary_cross_entropy_with_logits(
                split_logits[valid_mask],
                target_split_cropped[valid_mask],
            )
        else:
            split_loss = F.binary_cross_entropy_with_logits(
                split_logits,
                target_split_cropped,
            )

        # Cell count prediction loss (auxiliary)
        # Predict whether cell count increases
        count_pred = split_logits.sigmoid().sum()
        count_target = target_split_cropped.sum()
        count_loss = F.mse_loss(count_pred, count_target)

        # Total loss with emphasis on division
        sample_loss = (
            gene_loss + spatial_loss + split_weight * split_loss + 0.5 * count_loss
        )
        total_loss = total_loss + sample_loss

        all_losses["gene"].append(gene_loss.item())
        all_losses["spatial"].append(spatial_loss.item())
        all_losses["split"].append(split_loss.item())
        all_losses["count"].append(count_loss.item())

    batch_size = len(currents)
    total_loss = total_loss / batch_size

    loss_dict = {
        "total": total_loss.item(),
        "gene": sum(all_losses["gene"]) / batch_size,
        "spatial": sum(all_losses["spatial"]) / batch_size,
        "split": sum(all_losses["split"]) / batch_size,
        "count": sum(all_losses["count"]) / batch_size,
    }

    return total_loss, loss_dict


def train_epoch(model, loader, optimizer, device, split_weight=2.0):
    """Train for one epoch."""
    model.train()

    total_losses = {
        "total": 0.0,
        "gene": 0.0,
        "spatial": 0.0,
        "split": 0.0,
        "count": 0.0,
    }
    n_batches = 0

    for batch in loader:
        optimizer.zero_grad()

        loss, loss_dict = compute_loss_with_division_supervision(
            model, batch, device, split_weight
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        for k in total_losses:
            total_losses[k] += loss_dict[k]
        n_batches += 1

    return {k: v / n_batches for k, v in total_losses.items()}


@torch.no_grad()
def test_division_accuracy(model, dataset, device):
    """Test how well the model predicts divisions."""
    model.eval()

    total_cells = 0
    correct_predictions = 0
    total_divisions_predicted = 0
    total_actual_divisions = 0

    for i in range(min(20, len(dataset))):  # Test on first 20 samples
        sample = dataset[i]
        current = sample["current"].to(device)
        target_split = sample["target_split"].to(device)

        output = model.forward_step(current)

        split_probs = torch.sigmoid(output.split_logits[0, :, 0])
        n_current = current.padmask[0].sum().item()

        valid_probs = split_probs[:n_current]
        valid_targets = target_split[:n_current]

        # Count predictions
        predicted_divisions = (valid_probs > 0.5).sum().item()
        actual_divisions = valid_targets.sum().item()

        total_divisions_predicted += predicted_divisions
        total_actual_divisions += actual_divisions

        # Accuracy (within tolerance)
        for j in range(int(n_current)):
            pred = valid_probs[j] > 0.5
            target = valid_targets[j] > 0.5
            if pred == target:
                correct_predictions += 1
            total_cells += 1

    accuracy = correct_predictions / total_cells if total_cells > 0 else 0

    return {
        "accuracy": accuracy,
        "predicted_divisions": total_divisions_predicted,
        "actual_divisions": total_actual_divisions,
    }


def main():
    parser = argparse.ArgumentParser(description="Train with division supervision")
    parser.add_argument("--trajectory_file", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--split_weight", type=float, default=2.0, help="Weight for division loss"
    )
    parser.add_argument(
        "--save_dir", type=str, default="checkpoints_autoregressive_division"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("TRAINING WITH CELL DIVISION SUPERVISION")
    print("=" * 70)
    print(f"Trajectory: {args.trajectory_file}")
    print(f"Split loss weight: {args.split_weight}")
    print()

    # Load dataset
    print("Loading dataset...")
    dataset = DivisionSupervisedDataset(args.trajectory_file)

    if len(dataset) == 0:
        print("ERROR: No samples loaded!")
        return

    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    def collate_fn(batch):
        """Custom collate for BranchingState."""
        return {
            "current": [b["current"] for b in batch],
            "next": [b["next"] for b in batch],
            "target_split": [b["target_split"] for b in batch],
            "time": torch.tensor([b["time"] for b in batch]),
            "n_current": [b["n_current"] for b in batch],
            "n_next": [b["n_next"] for b in batch],
        }

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    _ = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
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
        max_seq_len=512,
        dt=0.1,
    ).to(args.device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    best_split_loss = float("inf")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_losses = train_epoch(
            model, train_loader, optimizer, args.device, args.split_weight
        )
        scheduler.step()

        # Test division accuracy
        div_acc = test_division_accuracy(model, val_dataset, args.device)

        print(f"Epoch {epoch:3d}:")
        print(
            f"  Loss: total={train_losses['total']:.4f}, "
            f"gene={train_losses['gene']:.4f}, "
            f"split={train_losses['split']:.4f}"
        )
        print(
            f"  Division: acc={div_acc['accuracy']:.2%}, "
            f"pred={div_acc['predicted_divisions']}, "
            f"actual={div_acc['actual_divisions']}"
        )

        # Save best model based on split loss
        if train_losses["split"] < best_split_loss:
            best_split_loss = train_losses["split"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "losses": train_losses,
                    "division_accuracy": div_acc,
                },
                save_dir / "best.pt",
            )
            print("  ✓ Saved best model")

    print()
    print("=" * 70)
    print(f"Training complete. Best split loss: {best_split_loss:.4f}")
    print(f"Checkpoint saved to {save_dir}/best.pt")
    print("=" * 70)


if __name__ == "__main__":
    main()
