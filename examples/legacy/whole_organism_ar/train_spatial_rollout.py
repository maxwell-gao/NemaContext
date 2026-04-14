#!/usr/bin/env python3
"""Train a spatial-only engineering rollout baseline on real trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.branching_flows.spatial_ar import SpatialAutoregressiveModel  # noqa: E402
from src.branching_flows.states import BranchingState  # noqa: E402


def collate_branching_states(batch: list[dict]) -> dict:
    result = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        if isinstance(values[0], BranchingState):
            result[key] = values
        else:
            result[key] = values
    return result


class SpatialTrajectoryDataset(Dataset):
    """Real-trajectory dataset for the spatial-only rollout baseline."""

    def __init__(self, trajectory_file: str, include_velocity: bool = True):
        with open(trajectory_file) as f:
            self.trajectory = json.load(f)
        self.include_velocity = include_velocity

    def __len__(self):
        return max(0, len(self.trajectory) - 1)

    @staticmethod
    def _child_names_in_next(parent_name: str, next_names: set[str]) -> list[str]:
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
        n_current = current["n_cells"]
        target_split = torch.zeros(n_current)
        target_del = torch.zeros(n_current)
        next_names = set(next_state["cell_names"])

        for idx, name in enumerate(current["cell_names"]):
            daughters = self._child_names_in_next(name, next_names)
            if len(daughters) >= 2:
                target_split[idx] = 1.0
            elif name not in next_names:
                target_del[idx] = 1.0
        return target_split, target_del

    def _build_features(self, state: dict, prev_state: dict | None) -> torch.Tensor:
        positions = torch.tensor(state["positions"], dtype=torch.float32)
        if not self.include_velocity:
            return positions

        velocity = torch.zeros_like(positions)
        if prev_state is not None:
            prev_map = {
                name: torch.tensor(pos, dtype=torch.float32)
                for name, pos in zip(prev_state["cell_names"], prev_state["positions"])
            }
            for idx, name in enumerate(state["cell_names"]):
                if name in prev_map:
                    velocity[idx] = positions[idx] - prev_map[name]
        return torch.cat([positions, velocity], dim=-1)

    def _state_to_branching(
        self, state: dict, prev_state: dict | None
    ) -> BranchingState:
        continuous = self._build_features(state, prev_state).unsqueeze(0)
        n_cells = state["n_cells"]
        return BranchingState(
            states=(continuous, None),
            groupings=torch.zeros(1, n_cells, dtype=torch.long),
            del_flags=torch.zeros(1, n_cells, dtype=torch.bool),
            ids=torch.arange(1, n_cells + 1, dtype=torch.long).unsqueeze(0),
            padmask=torch.ones(1, n_cells, dtype=torch.bool),
            flowmask=torch.ones(1, n_cells, dtype=torch.bool),
            branchmask=torch.ones(1, n_cells, dtype=torch.bool),
        )

    def __getitem__(self, idx):
        prev_state = self.trajectory[idx - 1] if idx > 0 else None
        current = self.trajectory[idx]
        next_state = self.trajectory[idx + 1]

        current_state = self._state_to_branching(current, prev_state)
        next_state_obj = self._state_to_branching(next_state, current)
        target_split, target_del = self._infer_event_targets(current, next_state)

        return {
            "current": current_state,
            "next": next_state_obj,
            "current_names": current["cell_names"],
            "next_names": next_state["cell_names"],
            "target_split": target_split,
            "target_del": target_del,
        }


def compute_loss(
    model: SpatialAutoregressiveModel,
    batch: dict,
    device: str,
    split_weight: float,
    del_weight: float,
    lambda_split_count: float,
    lambda_del_count: float,
) -> tuple[torch.Tensor, dict]:
    currents = batch["current"]
    next_states = batch["next"]
    current_names_batch = batch["current_names"]
    next_names_batch = batch["next_names"]
    target_splits = batch["target_split"]
    target_dels = batch["target_del"]

    total_loss = torch.tensor(0.0, device=device)
    losses = {"cont": [], "split": [], "del": [], "split_count": [], "del_count": []}

    for current, next_state, current_names, next_names, target_split, target_del in zip(
        currents,
        next_states,
        current_names_batch,
        next_names_batch,
        target_splits,
        target_dels,
    ):
        current = current.to(device)
        next_state = next_state.to(device)
        target_split = target_split.to(device)
        target_del = target_del.to(device)

        output = model.forward_step(current)

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
            cont_current = current.states[0][0, current_indices]
            cont_next = next_state.states[0][0, next_indices]
            cont_pred = output.continuous_delta[0, current_indices]
            cont_target = cont_next - cont_current
            cont_loss = F.mse_loss(cont_pred, cont_target)
        else:
            cont_loss = torch.tensor(0.0, device=device)

        valid_mask = current.padmask[0]
        split_logits = output.split_logits[0, : valid_mask.shape[0], 0][valid_mask]
        del_logits = output.del_logits[0, : valid_mask.shape[0], 0][valid_mask]
        target_split_valid = target_split[: valid_mask.shape[0]][valid_mask]
        target_del_valid = target_del[: valid_mask.shape[0]][valid_mask]

        split_pos = target_split_valid.sum().item()
        split_neg = max(1.0, float(target_split_valid.numel() - split_pos))
        split_pos_weight = torch.tensor(
            [split_neg / max(1.0, float(split_pos))],
            device=device,
        )
        split_loss = F.binary_cross_entropy_with_logits(
            split_logits,
            target_split_valid,
            pos_weight=split_pos_weight,
        )

        # Keep delete calibration conservative: false-positive deletions are
        # more damaging to rollout context than missing a rare death event.
        del_pos_weight = torch.tensor([1.0], device=device)
        del_loss = F.binary_cross_entropy_with_logits(
            del_logits,
            target_del_valid,
            pos_weight=del_pos_weight,
        )

        split_count_loss = F.mse_loss(
            torch.sigmoid(split_logits).sum(),
            target_split_valid.sum(),
        )
        del_count_loss = F.mse_loss(
            torch.sigmoid(del_logits).sum(),
            target_del_valid.sum(),
        )

        sample_loss = (
            cont_loss
            + split_weight * split_loss
            + del_weight * del_loss
            + lambda_split_count * split_count_loss
            + lambda_del_count * del_count_loss
        )
        total_loss = total_loss + sample_loss
        losses["cont"].append(cont_loss.item())
        losses["split"].append(split_loss.item())
        losses["del"].append(del_loss.item())
        losses["split_count"].append(split_count_loss.item())
        losses["del_count"].append(del_count_loss.item())

    batch_size = max(1, len(losses["cont"]))
    total_loss = total_loss / batch_size
    return total_loss, {
        "total": total_loss.item(),
        "cont": sum(losses["cont"]) / batch_size,
        "split": sum(losses["split"]) / batch_size,
        "del": sum(losses["del"]) / batch_size,
        "split_count": sum(losses["split_count"]) / batch_size,
        "del_count": sum(losses["del_count"]) / batch_size,
    }


def run_epoch(
    model: SpatialAutoregressiveModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    split_weight: float,
    del_weight: float,
    lambda_split_count: float,
    lambda_del_count: float,
) -> dict:
    train = optimizer is not None
    model.train(train)
    total = {
        "total": 0.0,
        "cont": 0.0,
        "split": 0.0,
        "del": 0.0,
        "split_count": 0.0,
        "del_count": 0.0,
    }
    n_batches = 0

    for batch in loader:
        if train:
            optimizer.zero_grad()
        loss, loss_dict = compute_loss(
            model,
            batch,
            device,
            split_weight=split_weight,
            del_weight=del_weight,
            lambda_split_count=lambda_split_count,
            lambda_del_count=lambda_del_count,
        )
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        for key in total:
            total[key] += loss_dict[key]
        n_batches += 1

    return {key: value / max(1, n_batches) for key, value in total.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Train the spatial-only engineering rollout baseline."
    )
    parser.add_argument("--trajectory_file", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--include_velocity", action="store_true")
    parser.add_argument("--split_weight", type=float, default=0.5)
    parser.add_argument("--del_weight", type=float, default=1.0)
    parser.add_argument("--lambda_split_count", type=float, default=0.1)
    parser.add_argument("--lambda_del_count", type=float, default=1.0)
    parser.add_argument("--split_threshold", type=float, default=0.5)
    parser.add_argument("--del_threshold", type=float, default=0.7)
    parser.add_argument("--save_dir", type=str, default="checkpoints/spatial_rollout")
    args = parser.parse_args()

    dataset = SpatialTrajectoryDataset(
        args.trajectory_file,
        include_velocity=args.include_velocity,
    )
    train_size = max(1, int(0.8 * len(dataset)))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
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

    continuous_dim = 6 if args.include_velocity else 3
    model = SpatialAutoregressiveModel(
        continuous_dim=continuous_dim,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_len=args.max_seq_len,
    ).to(args.device)
    model.cell_manager.split_threshold = args.split_threshold
    model.cell_manager.del_threshold = args.del_threshold

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SPATIAL ROLLOUT BASELINE")
    print("=" * 70)
    print(f"Trajectory file: {args.trajectory_file}")
    print(f"Continuous dim: {continuous_dim}")
    print(f"Train steps: {len(train_dataset)}, Val steps: {len(val_dataset)}")
    print()

    for epoch in range(1, args.epochs + 1):
        train_losses = run_epoch(
            model,
            train_loader,
            optimizer,
            args.device,
            split_weight=args.split_weight,
            del_weight=args.del_weight,
            lambda_split_count=args.lambda_split_count,
            lambda_del_count=args.lambda_del_count,
        )
        val_losses = run_epoch(
            model,
            val_loader,
            None,
            args.device,
            split_weight=args.split_weight,
            del_weight=args.del_weight,
            lambda_split_count=args.lambda_split_count,
            lambda_del_count=args.lambda_del_count,
        )
        scheduler.step()

        print(
            f"Epoch {epoch:3d}: "
            f"train total={train_losses['total']:.4f} "
            f"cont={train_losses['cont']:.4f} "
            f"split={train_losses['split']:.4f} "
            f"del={train_losses['del']:.4f} "
            f"del_count={train_losses['del_count']:.4f} | "
            f"val total={val_losses['total']:.4f}"
        )

        if val_losses["total"] < best_val:
            best_val = val_losses["total"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "losses": val_losses,
                    "config": vars(args),
                },
                save_dir / "best.pt",
            )


if __name__ == "__main__":
    main()
