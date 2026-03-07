#!/usr/bin/env python3
"""Full autoregressive training with whole-embryo trajectories.

This is the production-level training script that:
1. Loads unified whole-embryo trajectories (all cells coexist in one context)
2. Trains with proper event supervision
3. Validates with rollout perturbation experiments
4. Avoids injecting founder identity into the model input state
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.branching_flows.autoregressive_model import AutoregressiveNemaModel  # noqa: E402
from src.branching_flows.states import BranchingState  # noqa: E402


def sample_log_uniform_sigma(
    sigma_min: float,
    sigma_max: float,
    device: str,
) -> torch.Tensor:
    """Sample sigma from log-uniform distribution."""
    u = torch.rand(1, device=device)
    log_min = torch.log(torch.tensor(sigma_min, device=device))
    log_max = torch.log(torch.tensor(sigma_max, device=device))
    return torch.exp(log_min + u * (log_max - log_min))


def add_diffusion_noise(
    state: BranchingState,
    sigma: torch.Tensor,
) -> tuple[BranchingState, torch.Tensor]:
    """Create noisy state and epsilon target for denoising."""
    cont = state.states[0]
    noise = torch.randn_like(cont)
    sigma_view = sigma.view(1, 1, 1)

    noisy_cont = cont + sigma_view * noise
    mask = state.padmask.unsqueeze(-1).to(cont.dtype)
    noisy_cont = noisy_cont * mask + cont * (1.0 - mask)

    noisy_state = BranchingState(
        states=(noisy_cont, state.states[1]),
        groupings=state.groupings,
        del_flags=state.del_flags,
        ids=state.ids,
        padmask=state.padmask,
        flowmask=state.flowmask,
        branchmask=state.branchmask,
    )
    return noisy_state, noise


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

        if isinstance(data, list):
            self.trajectory = data
            self.is_whole_embryo = True
        else:
            raise ValueError(f"Unknown trajectory format: {type(data)}")

        self.n_hvg = n_hvg

        print(f"Loaded embryo trajectory: {len(self.trajectory)} time points")
        if self.trajectory:
            n_cells = [s["n_cells"] for s in self.trajectory]
            print(f"  Cell count range: {min(n_cells)} -> {max(n_cells)}")
            if self.is_whole_embryo:
                print("  Format: Whole-embryo (all lineages coexist)")

    def __len__(self):
        return max(0, len(self.trajectory) - 1)

    @staticmethod
    def _child_names_in_next(parent_name: str, next_names: set[str]) -> list[str]:
        """Find immediate daughters of a parent cell in the next state."""
        prefix_len = len(parent_name) + 1
        return [
            name
            for name in next_names
            if name.startswith(parent_name) and len(name) == prefix_len
        ]

    def _infer_event_targets(
        self,
        current: dict,
        next_state: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Infer split/delete targets from consecutive cell-name sets."""
        n_current = current["n_cells"]
        target_split = torch.zeros(n_current)
        target_del = torch.zeros(n_current)

        next_names = set(next_state.get("cell_names", []))
        for idx, cell_name in enumerate(current.get("cell_names", [])):
            daughters = self._child_names_in_next(cell_name, next_names)
            if len(daughters) >= 2:
                target_split[idx] = 1.0
            elif cell_name not in next_names:
                target_del[idx] = 1.0

        return target_split, target_del

    def __getitem__(self, idx):
        """Get consecutive time points."""
        current = self.trajectory[idx]
        next_state = self.trajectory[idx + 1]

        # Convert to BranchingStates
        current_state = self._state_to_branching(current)
        next_state_obj = self._state_to_branching(next_state)

        # Infer events from consecutive name sets rather than trusting stored
        # division indices, which may encode birth events in older datasets.
        target_split, target_del = self._infer_event_targets(current, next_state)

        return {
            "current": current_state,
            "next": next_state_obj,
            "current_names": current.get("cell_names", []),
            "next_names": next_state.get("cell_names", []),
            "target_split": target_split,
            "target_del": target_del,
            "time": current["time"],
        }

    def _state_to_branching(self, state: dict) -> BranchingState:
        """Convert trajectory state to BranchingState without founder input."""
        n_cells = state["n_cells"]

        # Get features
        if "genes" in state:
            genes = torch.tensor(state["genes"], dtype=torch.float32)
        else:
            genes = torch.zeros(n_cells, self.n_hvg, dtype=torch.float32)

        if "positions" in state:
            positions = torch.tensor(state["positions"], dtype=torch.float32)
        else:
            positions = torch.zeros(n_cells, 3, dtype=torch.float32)

        # Ensure correct shapes
        if genes.shape[0] != n_cells:
            genes = genes[:n_cells]
        if positions.shape[0] != n_cells:
            positions = positions[:n_cells]

        # Continuous features: genes + spatial positions
        continuous = torch.cat([genes, positions], dim=-1).unsqueeze(0)

        return BranchingState(
            states=(continuous, None),
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
    sigma_min: float = 0.01,
    sigma_max: float = 0.2,
    lambda_denoise: float = 0.2,
    sigma_cond_drop_prob: float = 0.1,
    lambda_event_count: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """Compute AR next-step + diffusion-style denoising + event losses."""
    currents = batch["current"]
    next_states = batch["next"]
    target_splits = batch["target_split"]
    target_dels = batch["target_del"]
    current_names_batch = batch["current_names"]
    next_names_batch = batch["next_names"]

    # If single sample, wrap in list
    if isinstance(currents, BranchingState):
        currents = [currents]
        next_states = [next_states]
        target_splits = [target_splits]
        target_dels = [target_dels]
        current_names_batch = [current_names_batch]
        next_names_batch = [next_names_batch]

    total_loss = torch.tensor(0.0, device=device)
    all_losses = {
        "gene": [],
        "spatial": [],
        "split": [],
        "del": [],
        "denoise": [],
        "count": [],
        "discrete": [],
    }

    for current, next_state, target_split, target_del, current_names, next_names in zip(
        currents,
        next_states,
        target_splits,
        target_dels,
        current_names_batch,
        next_names_batch,
    ):
        current = current.to(device)
        next_state = next_state.to(device)
        target_split = target_split.to(device)
        target_del = target_del.to(device)

        # Diffusion-style noising on current state.
        sigma = sample_log_uniform_sigma(sigma_min, sigma_max, device)
        noisy_current, eps_target = add_diffusion_noise(current, sigma)
        sigma_for_model = sigma
        if torch.rand(1, device=device).item() < sigma_cond_drop_prob:
            sigma_for_model = torch.zeros_like(sigma)

        # Forward pass from noisy current, conditioned on sigma.
        output = model.forward_step(noisy_current, sigma=sigma_for_model)

        current_name_to_idx = {name: idx for idx, name in enumerate(current_names)}
        next_name_to_idx = {name: idx for idx, name in enumerate(next_names)}
        matched_names = [name for name in current_names if name in next_name_to_idx]

        if matched_names:
            current_indices = torch.tensor(
                [current_name_to_idx[name] for name in matched_names],
                device=device,
                dtype=torch.long,
            )
            next_indices = torch.tensor(
                [next_name_to_idx[name] for name in matched_names],
                device=device,
                dtype=torch.long,
            )

            cont_current = current.states[0][0, current_indices, :]
            cont_next = next_state.states[0][0, next_indices, :]
            gene_delta_pred = output.gene_delta[0, current_indices, :]
            true_gene_delta = cont_next[..., :2000] - cont_current[..., :2000]
            gene_loss = F.mse_loss(gene_delta_pred, true_gene_delta)

            spatial_vel_pred = output.spatial_vel[0, current_indices, :]
            true_spatial_vel = cont_next[..., 2000:2003] - cont_current[..., 2000:2003]
            spatial_loss = F.mse_loss(spatial_vel_pred, true_spatial_vel)

            noise_pred = output.noise_pred[0, current_indices, :]
            eps_target_common = eps_target[0, current_indices, :]
            denoise_loss = F.mse_loss(noise_pred, eps_target_common)
        else:
            gene_loss = torch.tensor(0.0, device=device)
            spatial_loss = torch.tensor(0.0, device=device)
            denoise_loss = torch.tensor(0.0, device=device)

        n_current = current.states[0].shape[1]
        valid_mask = current.padmask.squeeze(0)[:n_current]

        # Event prediction losses
        split_logits = output.split_logits[0, :n_current, 0][valid_mask]
        del_logits = output.del_logits[0, :n_current, 0][valid_mask]

        target_split_cropped = target_split[:n_current][valid_mask]
        target_del_cropped = target_del[:n_current][valid_mask]

        split_pos = target_split_cropped.sum().item()
        split_neg = max(1.0, float(target_split_cropped.numel() - split_pos))
        split_pos_weight = torch.tensor(
            [split_neg / max(1.0, float(split_pos))],
            device=device,
        )
        split_loss = F.binary_cross_entropy_with_logits(
            split_logits,
            target_split_cropped,
            pos_weight=split_pos_weight,
        )

        del_pos = target_del_cropped.sum().item()
        del_neg = max(1.0, float(target_del_cropped.numel() - del_pos))
        del_pos_weight = torch.tensor(
            [del_neg / max(1.0, float(del_pos))],
            device=device,
        )
        del_loss = F.binary_cross_entropy_with_logits(
            del_logits,
            target_del_cropped,
            pos_weight=del_pos_weight,
        )

        split_count_loss = F.mse_loss(
            torch.sigmoid(split_logits).sum(),
            target_split_cropped.sum(),
        )
        del_count_loss = F.mse_loss(
            torch.sigmoid(del_logits).sum(),
            target_del_cropped.sum(),
        )
        count_loss = split_count_loss + del_count_loss

        # Accumulate loss
        sample_loss = (
            gene_loss
            + spatial_loss
            + 0.5 * split_loss
            + 0.5 * del_loss
            + lambda_event_count * count_loss
            + lambda_denoise * denoise_loss
        )
        total_loss = total_loss + sample_loss

        all_losses["gene"].append(gene_loss.item())
        all_losses["spatial"].append(spatial_loss.item())
        all_losses["split"].append(split_loss.item())
        all_losses["del"].append(del_loss.item())
        all_losses["denoise"].append(denoise_loss.item())
        all_losses["count"].append(count_loss.item())
        all_losses["discrete"].append(0.0)

    # Average over batch
    batch_size = max(1, len(all_losses["gene"]))
    total_loss = total_loss / batch_size

    loss_dict = {
        "total": total_loss.item(),
        "gene": sum(all_losses["gene"]) / batch_size,
        "spatial": sum(all_losses["spatial"]) / batch_size,
        "split": sum(all_losses["split"]) / batch_size,
        "del": sum(all_losses["del"]) / batch_size,
        "denoise": sum(all_losses["denoise"]) / batch_size,
        "count": sum(all_losses["count"]) / batch_size,
        "discrete": sum(all_losses["discrete"]) / batch_size,
    }

    return total_loss, loss_dict


def train_epoch(
    model: AutoregressiveNemaModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    sigma_min: float,
    sigma_max: float,
    lambda_denoise: float,
    sigma_cond_drop_prob: float,
    lambda_event_count: float,
) -> dict:
    """Train for one epoch."""
    model.train()

    total_losses = {
        "total": 0.0,
        "gene": 0.0,
        "spatial": 0.0,
        "split": 0.0,
        "del": 0.0,
        "denoise": 0.0,
        "count": 0.0,
        "discrete": 0.0,
    }
    n_batches = 0

    for batch in loader:
        optimizer.zero_grad()

        loss, loss_dict = compute_autoregressive_loss(
            model,
            batch,
            device,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            lambda_denoise=lambda_denoise,
            sigma_cond_drop_prob=sigma_cond_drop_prob,
            lambda_event_count=lambda_event_count,
        )

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
    sigma_min: float,
    sigma_max: float,
    lambda_denoise: float,
    sigma_cond_drop_prob: float,
    lambda_event_count: float,
) -> dict:
    """Validate model."""
    model.eval()

    total_losses = {
        "total": 0.0,
        "gene": 0.0,
        "spatial": 0.0,
        "split": 0.0,
        "del": 0.0,
        "denoise": 0.0,
        "count": 0.0,
        "discrete": 0.0,
    }
    n_batches = 0

    for batch in loader:
        loss, loss_dict = compute_autoregressive_loss(
            model,
            batch,
            device,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            lambda_denoise=lambda_denoise,
            sigma_cond_drop_prob=sigma_cond_drop_prob,
            lambda_event_count=lambda_event_count,
        )

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
            new_disc = state.states[1][:, mask] if state.states[1] is not None else None
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
    parser.add_argument("--gene_dim", type=int, default=2000)
    parser.add_argument("--spatial_dim", type=int, default=3)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--cross_modal_every", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--sigma_min", type=float, default=0.01)
    parser.add_argument("--sigma_max", type=float, default=0.2)
    parser.add_argument("--lambda_denoise", type=float, default=0.2)
    parser.add_argument("--lambda_event_count", type=float, default=0.0)
    parser.add_argument(
        "--sigma_cond_drop_prob",
        type=float,
        default=0.1,
        help="Probability of dropping sigma conditioning (CFG-style).",
    )
    parser.add_argument(
        "--save_dir", type=str, default="checkpoints_autoregressive_full"
    )
    parser.add_argument(
        "--init_checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint to warm-start from.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("WHOLE-EMBRYO AUTOREGRESSIVE TRAINING")
    print("=" * 70)
    print(f"Trajectory file: {args.trajectory_file}")
    print(
        f"Diffusion config: sigma=[{args.sigma_min}, {args.sigma_max}], "
        f"lambda_denoise={args.lambda_denoise}, "
        f"sigma_drop={args.sigma_cond_drop_prob}"
    )
    print(
        f"Event config: lambda_event_count={args.lambda_event_count}"
    )
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
        gene_dim=args.gene_dim,
        spatial_dim=args.spatial_dim,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        cross_modal_every=args.cross_modal_every,
        max_seq_len=args.max_seq_len,
        dt=args.dt,
    ).to(args.device)

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location=args.device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        load_result = model.load_state_dict(state_dict, strict=False)
        print(f"  Warm start: {args.init_checkpoint}")
        print(f"    Missing keys: {len(load_result.missing_keys)}")
        print(f"    Unexpected keys: {len(load_result.unexpected_keys)}")
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
        train_losses = train_epoch(
            model,
            train_loader,
            optimizer,
            args.device,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            lambda_denoise=args.lambda_denoise,
            sigma_cond_drop_prob=args.sigma_cond_drop_prob,
            lambda_event_count=args.lambda_event_count,
        )
        val_losses = validate(
            model,
            val_loader,
            args.device,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            lambda_denoise=args.lambda_denoise,
            sigma_cond_drop_prob=args.sigma_cond_drop_prob,
            lambda_event_count=args.lambda_event_count,
        )
        scheduler.step()

        print(f"Epoch {epoch:3d}:")
        print(
            f"  Train: total={train_losses['total']:.4f}, "
            f"gene={train_losses['gene']:.4f}, "
            f"split={train_losses['split']:.4f}, "
            f"count={train_losses['count']:.4f}, "
            f"denoise={train_losses['denoise']:.4f}"
        )
        print(
            f"  Val:   total={val_losses['total']:.4f}, "
            f"gene={val_losses['gene']:.4f}, "
            f"split={val_losses['split']:.4f}, "
            f"count={val_losses['count']:.4f}, "
            f"denoise={val_losses['denoise']:.4f}"
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
