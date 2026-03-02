#!/usr/bin/env python3
"""Full autoregressive training with whole-embryo trajectories.

This is the production-level training script that:
1. Loads unified whole-embryo trajectories (all lineages coexist)
2. Trains with proper event supervision
3. Validates with perturbation experiments
4. Supports cross-lineage attention learning
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

from src.branching_flows.autoregressive_model import AutoregressiveNemaModel  # noqa: E402
from src.branching_flows.states import BranchingState  # noqa: E402


def collate_branching_states(batch: list) -> dict:
    """Custom collate function for BranchingState objects."""
    result = {}
    for key in batch[0].keys():
        values = [d[key] for d in batch]
        if isinstance(values[0], BranchingState):
            result[key] = values
        elif isinstance(values[0], torch.Tensor):
            # Check if all tensors have same shape
            shapes = [v.shape for v in values]
            if len(set(str(s) for s in shapes)) == 1:
                result[key] = torch.stack(values)
            else:
                # Variable length tensors (target_split, target_del)
                result[key] = values
        else:
            result[key] = values
    return result


class EmbryoTrajectoryDataset(Dataset):
    """Dataset of whole-embryo developmental trajectories.

    All cells from all lineages (AB, MS, E, C, D, etc.) coexist
    at each time point in shared global coordinates.
    """

    def __init__(
        self,
        trajectory_file: str,
        n_hvg: int = 2000,
    ):
        with open(trajectory_file) as f:
            data = json.load(f)

        # Handle both formats: list (whole-embryo) or dict (legacy per-founder)
        if isinstance(data, list):
            # New whole-embryo format
            self.trajectory = data
            self.is_whole_embryo = True
        elif isinstance(data, dict):
            # Legacy format: flatten all founders into single trajectory
            print("WARNING: Loading legacy format (separate founder trajectories)")
            print(
                "  Consider regenerating with: uv run python src/data/trajectory_extractor.py"
            )
            self.trajectory = self._flatten_legacy_trajectories(data)
            self.is_whole_embryo = False
        else:
            raise ValueError(f"Unknown trajectory format: {type(data)}")

        self.n_hvg = n_hvg

        print(f"Loaded embryo trajectory: {len(self.trajectory)} time points")
        if self.trajectory:
            n_cells = [s["n_cells"] for s in self.trajectory]
            print(f"  Cell count range: {min(n_cells)} -> {max(n_cells)}")
            if self.is_whole_embryo:
                print("  Format: Whole-embryo (all lineages coexist)")

    def _flatten_legacy_trajectories(self, data: dict) -> list:
        """Convert legacy per-founder dict to unified timeline."""
        # Collect all time points from all founders
        all_events = []
        for founder, traj in data.items():
            for state in traj:
                all_events.append(
                    {
                        "time": state["time"],
                        "state": state,
                        "founder": founder,
                    }
                )

        # Group by time
        time_groups = {}
        for event in all_events:
            t = event["time"]
            if t not in time_groups:
                time_groups[t] = []
            time_groups[t].append(event["state"])

        # Merge states at each time point
        merged = []
        for t in sorted(time_groups.keys()):
            states = time_groups[t]
            merged_state = self._merge_states(states, t)
            merged.append(merged_state)

        return merged

    def _merge_states(self, states: list[dict], time: float) -> dict:
        """Merge multiple founder states into single embryo state."""
        cell_names = []
        founders = []
        founder_ids = []
        positions = []
        genes = []
        divisions = []

        cell_offset = 0
        for state in states:
            n = state["n_cells"]
            cell_names.extend(state["cell_names"])
            founders.extend(state.get("founders", ["AB"] * n))
            founder_ids.extend(state.get("founder_ids", [0] * n))
            positions.extend(state["positions"])
            genes.extend(state["genes"])

            # Adjust division indices
            for div_idx in state.get("divisions", []):
                divisions.append(div_idx + cell_offset)

            cell_offset += n

        return {
            "time": time,
            "n_cells": len(cell_names),
            "cell_names": cell_names,
            "founders": founders,
            "founder_ids": founder_ids,
            "positions": positions,
            "genes": genes,
            "divisions": divisions,
        }

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

        # Mark deaths (if any)
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
        """Convert trajectory state to BranchingState with founder identity."""
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

        # Continuous features: genes + spatial positions
        continuous = torch.cat([genes, positions], dim=-1).unsqueeze(0)

        # Discrete features: founder identity (crucial for cross-lineage learning)
        if "founder_ids" in state:
            founder_ids = state["founder_ids"]
            discrete = torch.tensor(founder_ids, dtype=torch.long).unsqueeze(0)
        else:
            # Fallback: all cells same founder
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
    currents = batch["current"]
    next_states = batch["next"]
    target_splits = batch["target_split"]
    target_dels = batch["target_del"]

    # If single sample, wrap in list
    if isinstance(currents, BranchingState):
        currents = [currents]
        next_states = [next_states]
        target_splits = [target_splits]
        target_dels = [target_dels]

    total_loss = torch.tensor(0.0, device=device)
    all_losses = {"gene": [], "spatial": [], "split": [], "del": []}

    for current, next_state, target_split, target_del in zip(
        currents, next_states, target_splits, target_dels
    ):
        current = current.to(device)
        next_state = next_state.to(device)
        target_split = target_split.to(device)
        target_del = target_del.to(device)

        # Forward pass
        output = model.forward_step(current)

        # Handle dynamic cell counts (cells may divide between t and t+1)
        n_current = current.states[0].shape[1]
        n_next = next_state.states[0].shape[1]
        n_common = min(n_current, n_next)

        # State prediction loss
        valid_mask = current.padmask.squeeze(0)[:n_common]

        # Gene delta loss (only on cells that exist in both states)
        gene_delta_pred = output.gene_delta[0, :n_common, :][valid_mask, :]
        cont_current = current.states[0][0, :n_common, :][valid_mask, :]
        cont_next = next_state.states[0][0, :n_common, :][valid_mask, :]

        true_gene_delta = cont_next[..., :2000] - cont_current[..., :2000]
        gene_loss = F.mse_loss(gene_delta_pred, true_gene_delta)

        # Spatial velocity loss
        spatial_vel_pred = output.spatial_vel[0, :n_common, :][valid_mask, :]
        true_spatial_vel = cont_next[..., 2000:2003] - cont_current[..., 2000:2003]
        spatial_loss = F.mse_loss(spatial_vel_pred, true_spatial_vel)

        # Event prediction losses
        split_logits = output.split_logits[0, :n_common, 0][valid_mask]
        del_logits = output.del_logits[0, :n_common, 0][valid_mask]

        target_split_cropped = target_split[:n_common][valid_mask]
        target_del_cropped = target_del[:n_common][valid_mask]

        split_loss = F.binary_cross_entropy_with_logits(
            split_logits, target_split_cropped
        )
        del_loss = F.binary_cross_entropy_with_logits(del_logits, target_del_cropped)

        # Accumulate loss
        sample_loss = gene_loss + spatial_loss + 0.5 * split_loss + 0.5 * del_loss
        total_loss = total_loss + sample_loss

        all_losses["gene"].append(gene_loss.item())
        all_losses["spatial"].append(spatial_loss.item())
        all_losses["split"].append(split_loss.item())
        all_losses["del"].append(del_loss.item())

    # Average over batch
    batch_size = len(currents)
    total_loss = total_loss / batch_size

    loss_dict = {
        "total": total_loss.item(),
        "gene": sum(all_losses["gene"]) / batch_size,
        "spatial": sum(all_losses["spatial"]) / batch_size,
        "split": sum(all_losses["split"]) / batch_size,
        "del": sum(all_losses["del"]) / batch_size,
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
def test_cross_lineage_attention(
    model: AutoregressiveNemaModel,
    dataset: EmbryoTrajectoryDataset,
    device: str,
    threshold: float = 0.1,
) -> dict:
    """Test if model uses cross-lineage attention.

    Method:
    1. Normal forward pass with all cells
    2. Mask non-target founder cells, forward again
    3. Compare predictions: significant difference = cross-lineage dependency

    Returns:
        Dictionary with cross-lineage attention metrics
    """
    model.eval()

    # Find a time point with multiple founders
    multi_founder_idx = None
    for idx in range(len(dataset)):
        state = dataset.trajectory[idx]
        founders = set(state.get("founders", []))
        if len(founders) > 1:
            multi_founder_idx = idx
            break

    if multi_founder_idx is None:
        print("WARNING: No multi-founder time points found")
        return {"cross_lineage_detected": False, "mean_delta_diff": 0.0}

    # Get sample
    sample = dataset[multi_founder_idx]
    current = sample["current"].to(device)

    # Normal prediction
    output_full = model.forward_step(current)

    # Isolate each founder and measure prediction change
    founder_map_inv = {0: "P0", 1: "AB", 2: "MS", 3: "E", 4: "C", 5: "D", 6: "P4"}

    delta_diffs = []
    founder_analysis = {}

    for founder_id in range(7):
        founder_name = founder_map_inv.get(founder_id, "UNKNOWN")

        # Find cells of this founder
        founder_mask = (current.states[1][0] == founder_id) & current.padmask[0]
        n_founder_cells = founder_mask.sum().item()

        if n_founder_cells == 0:
            continue

        # Create isolated state (only this founder's cells)
        isolated_indices = torch.where(founder_mask)[0]
        new_cont = current.states[0][:, isolated_indices, :]
        new_disc = current.states[1][:, isolated_indices]
        n_kept = isolated_indices.shape[0]

        isolated_state = BranchingState(
            states=(new_cont, new_disc),
            groupings=torch.zeros(1, n_kept, dtype=torch.long, device=device),
            del_flags=torch.zeros(1, n_kept, dtype=torch.bool, device=device),
            ids=torch.arange(1, n_kept + 1, dtype=torch.long, device=device).unsqueeze(
                0
            ),
            padmask=torch.ones(1, n_kept, dtype=torch.bool, device=device),
            flowmask=torch.ones(1, n_kept, dtype=torch.bool, device=device),
            branchmask=torch.ones(1, n_kept, dtype=torch.bool, device=device),
        )

        # Isolated prediction
        output_isolated = model.forward_step(isolated_state)

        # Compare predictions for founder cells
        full_gene_delta = output_full.gene_delta[0, founder_mask, :]
        isolated_gene_delta = output_isolated.gene_delta[0, :, :]

        diff = torch.norm(full_gene_delta - isolated_gene_delta, dim=-1).mean().item()
        delta_diffs.append(diff)

        founder_analysis[founder_name] = {
            "n_cells": n_founder_cells,
            "delta_diff": diff,
            "uses_context": diff > threshold,
        }

    mean_delta_diff = sum(delta_diffs) / len(delta_diffs) if delta_diffs else 0.0

    return {
        "cross_lineage_detected": mean_delta_diff > threshold,
        "mean_delta_diff": mean_delta_diff,
        "founder_analysis": founder_analysis,
    }


@torch.no_grad()
def test_perturbation_cross_lineage(
    model: AutoregressiveNemaModel,
    initial_state: BranchingState,
    device: str,
    target_founder: str = "AB",
    perturb_time: int = 5,
) -> dict:
    """Test perturbation with cross-lineage effects.

    Deletes all cells of target_founder and observes effect on other lineages.
    """
    model.eval()

    # Founder name to ID mapping
    founder_map = {"P0": 0, "AB": 1, "MS": 2, "E": 3, "C": 4, "D": 5, "P4": 6}
    target_id = founder_map.get(target_founder, 1)

    # Control trajectory
    control_states = [initial_state]
    state = initial_state

    for _ in range(20):
        state, _ = model.step(state, deterministic=True, apply_events=True)
        control_states.append(state)

    # Perturbed trajectory (delete target founder cells at perturb_time)
    perturbed_states = [initial_state]
    state = initial_state

    for t in range(20):
        if t == perturb_time:
            # Remove all cells of target founder
            founder_ids = state.states[1][0]
            keep_mask = founder_ids != target_id

            new_cont = state.states[0][:, keep_mask, :]
            new_disc = state.states[1][:, keep_mask]
            n_kept = keep_mask.sum().item()

            if n_kept > 0:
                state = BranchingState(
                    states=(new_cont, new_disc),
                    groupings=torch.zeros(1, n_kept, dtype=torch.long, device=device),
                    del_flags=torch.zeros(1, n_kept, dtype=torch.bool, device=device),
                    ids=torch.arange(
                        1, n_kept + 1, dtype=torch.long, device=device
                    ).unsqueeze(0),
                    padmask=torch.ones(1, n_kept, dtype=torch.bool, device=device),
                    flowmask=torch.ones(1, n_kept, dtype=torch.bool, device=device),
                    branchmask=torch.ones(1, n_kept, dtype=torch.bool, device=device),
                )

        state, _ = model.step(state, deterministic=True, apply_events=True)
        perturbed_states.append(state)

    # Analyze cross-lineage effects
    control_final = control_states[-1]
    perturbed_final = perturbed_states[-1]

    control_founders = control_final.states[1][0].cpu().numpy()
    perturbed_founders = perturbed_final.states[1][0].cpu().numpy()

    # Count cells per founder in each condition
    control_counts = {
        name: (control_founders == fid).sum() for name, fid in founder_map.items()
    }
    perturbed_counts = {
        name: (perturbed_founders == fid).sum() for name, fid in founder_map.items()
    }

    # Compare non-target founders (these should show effects if cross-lineage)
    cross_lineage_effects = {}
    for founder, fid in founder_map.items():
        if founder != target_founder:
            diff = control_counts[founder] - perturbed_counts[founder]
            cross_lineage_effects[founder] = {
                "control": int(control_counts[founder]),
                "perturbed": int(perturbed_counts[founder]),
                "difference": int(diff),
            }

    return {
        "target_founder": target_founder,
        "control_total_cells": int(control_final.padmask.sum()),
        "perturbed_total_cells": int(perturbed_final.padmask.sum()),
        "cross_lineage_effects": cross_lineage_effects,
        "showed_cross_lineage_effect": any(
            e["difference"] != 0 for e in cross_lineage_effects.values()
        ),
    }


@torch.no_grad()
def test_perturbation(
    model: AutoregressiveNemaModel,
    initial_state: BranchingState,
    device: str,
    perturb_time: int = 5,
) -> dict:
    """Test causal response to random cell deletion perturbation."""
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

            seq_len = state.states[0].shape[1]
            mask = torch.ones(seq_len, dtype=torch.bool, device=device)
            valid_indices = torch.where(state.padmask[0])[0]
            remove_indices = valid_indices[
                torch.randperm(n_cells, device=device)[:n_remove]
            ]
            mask[remove_indices] = False
            mask = mask & state.padmask[0]

            new_cont = state.states[0][:, mask, :]
            new_disc = state.states[1][:, mask]
            n_kept = mask.sum().item()

            state = BranchingState(
                states=(new_cont, new_disc),
                groupings=torch.zeros(1, n_kept, dtype=torch.long, device=device),
                del_flags=torch.zeros(1, n_kept, dtype=torch.bool, device=device),
                ids=torch.arange(
                    1, n_kept + 1, dtype=torch.long, device=device
                ).unsqueeze(0),
                padmask=torch.ones(1, n_kept, dtype=torch.bool, device=device),
                flowmask=torch.ones(1, n_kept, dtype=torch.bool, device=device),
                branchmask=torch.ones(1, n_kept, dtype=torch.bool, device=device),
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
        "compensated": distance < 1.0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Full autoregressive training with whole-embryo context"
    )
    parser.add_argument("--trajectory_file", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--save_dir", type=str, default="checkpoints_autoregressive_full"
    )
    parser.add_argument(
        "--test_cross_lineage",
        action="store_true",
        help="Run cross-lineage attention test",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("WHOLE-EMBRYO AUTOREGRESSIVE TRAINING")
    print("=" * 70)
    print(f"Trajectory file: {args.trajectory_file}")
    print()

    # Load dataset
    print("Loading dataset...")
    dataset = EmbryoTrajectoryDataset(args.trajectory_file)

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_branching_states,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_branching_states,
    )

    print(f"  Train: {len(train_dataset)} steps")
    print(f"  Val: {len(val_dataset)} steps")
    print()

    # Create model
    print("Creating model...")
    model = AutoregressiveNemaModel(
        gene_dim=2000,
        spatial_dim=3,
        discrete_K=7,  # 7 founders: P0, AB, MS, E, C, D, P4
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

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
        print(
            f"  Train: total={train_losses['total']:.4f}, "
            f"gene={train_losses['gene']:.4f}, "
            f"split={train_losses['split']:.4f}"
        )
        print(
            f"  Val:   total={val_losses['total']:.4f}, "
            f"gene={val_losses['gene']:.4f}, "
            f"split={val_losses['split']:.4f}"
        )

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

    # Cross-lineage attention test
    if args.test_cross_lineage:
        print()
        print("Testing cross-lineage attention...")
        cross_lineage_result = test_cross_lineage_attention(model, dataset, args.device)

        print(
            f"  Cross-lineage detected: {cross_lineage_result['cross_lineage_detected']}"
        )
        print(f"  Mean delta difference: {cross_lineage_result['mean_delta_diff']:.4f}")

        if "founder_analysis" in cross_lineage_result:
            print("  Founder-specific analysis:")
            for founder, analysis in cross_lineage_result["founder_analysis"].items():
                status = "✓" if analysis["uses_context"] else "✗"
                print(
                    f"    {status} {founder}: n={analysis['n_cells']}, "
                    f"delta_diff={analysis['delta_diff']:.4f}"
                )

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

    # Cross-lineage perturbation test
    print()
    print("Testing cross-lineage perturbation (AB deletion)...")
    cross_perturb = test_perturbation_cross_lineage(
        model, initial, args.device, target_founder="AB"
    )

    print(f"  Target: {cross_perturb['target_founder']}")
    print(f"  Control total: {cross_perturb['control_total_cells']}")
    print(f"  Perturbed total: {cross_perturb['perturbed_total_cells']}")

    if cross_perturb["showed_cross_lineage_effect"]:
        print("  ✓ Cross-lineage effects detected:")
        for founder, effect in cross_perturb["cross_lineage_effects"].items():
            if effect["difference"] != 0:
                print(
                    f"    {founder}: {effect['control']} -> {effect['perturbed']} "
                    f"(Δ{effect['difference']:+d})"
                )
    else:
        print("  No cross-lineage effects detected")

    print("=" * 70)


if __name__ == "__main__":
    main()
