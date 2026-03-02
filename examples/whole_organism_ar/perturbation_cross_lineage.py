#!/usr/bin/env python3
"""Cross-lineage perturbation experiments.

Tests how deleting cells from one lineage affects the development
of other lineages. This validates that the model learns true
embryonic context rather than isolated lineage trajectories.

Biological principle: Cell fate decisions are influenced by signals
from neighboring cells, regardless of lineage origin. Removing AB
cells should affect the development of EMS, MS, etc.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.branching_flows.autoregressive_model import AutoregressiveNemaModel  # noqa: E402
from src.branching_flows.states import BranchingState  # noqa: E402
from examples.whole_organism_ar.train_autoregressive_full import EmbryoTrajectoryDataset  # noqa: E402


def filter_cells_by_founder(
    state: BranchingState,
    criterion: Callable[[int], bool],
    device: str,
) -> BranchingState:
    """Filter cells based on founder ID criterion."""
    founder_ids = state.states[1][0]
    keep_mask = (
        torch.tensor(
            [criterion(fid.item()) for fid in founder_ids],
            dtype=torch.bool,
            device=device,
        )
        & state.padmask[0]
    )

    if keep_mask.sum() == 0:
        return None

    new_cont = state.states[0][:, keep_mask, :]
    new_disc = state.states[1][:, keep_mask]
    n_kept = keep_mask.sum().item()

    return BranchingState(
        states=(new_cont, new_disc),
        groupings=torch.zeros(1, n_kept, dtype=torch.long, device=device),
        del_flags=torch.zeros(1, n_kept, dtype=torch.bool, device=device),
        ids=torch.arange(1, n_kept + 1, dtype=torch.long, device=device).unsqueeze(0),
        padmask=torch.ones(1, n_kept, dtype=torch.bool, device=device),
        flowmask=torch.ones(1, n_kept, dtype=torch.bool, device=device),
        branchmask=torch.ones(1, n_kept, dtype=torch.bool, device=device),
    )


@torch.no_grad()
def simulate_trajectory(
    model: AutoregressiveNemaModel,
    initial_state: BranchingState,
    device: str,
    n_steps: int = 30,
    perturbation_fn: Callable[[BranchingState, int], BranchingState] | None = None,
) -> list[BranchingState]:
    """Simulate developmental trajectory with optional perturbations."""
    states = [initial_state]
    state = initial_state

    for t in range(n_steps):
        # Apply perturbation if provided
        if perturbation_fn is not None:
            state = perturbation_fn(state, t)
            if state is None:
                break

        # Step forward
        state, _ = model.step(state, deterministic=True, apply_events=True)
        states.append(state)

        # Stop if all cells dead
        if state.padmask.sum() == 0:
            break

    return states


def count_cells_by_founder(state: BranchingState) -> dict[str, int]:
    """Count cells per founder in a state."""
    founder_map_inv = {0: "P0", 1: "AB", 2: "MS", 3: "E", 4: "C", 5: "D", 6: "P4"}

    counts = {name: 0 for name in founder_map_inv.values()}
    founder_ids = state.states[1][0][state.padmask[0]]

    for fid in founder_ids.tolist():
        name = founder_map_inv.get(fid, "UNKNOWN")
        counts[name] = counts.get(name, 0) + 1

    return counts


def compute_trajectory_metrics(states: list[BranchingState]) -> dict:
    """Compute metrics from a simulated trajectory."""
    if not states:
        return {}

    # Cell count over time
    cell_counts = [int(s.padmask.sum()) for s in states]

    # Founder composition over time
    founder_composition = [count_cells_by_founder(s) for s in states]

    # Spatial spread (std dev of positions)
    spatial_spreads = []
    for s in states:
        if s.padmask.sum() > 0:
            positions = s.states[0][0, s.padmask[0], -3:]
            spread = positions.std(dim=0).mean().item()
            spatial_spreads.append(spread)
        else:
            spatial_spreads.append(0.0)

    return {
        "final_cell_count": cell_counts[-1],
        "max_cell_count": max(cell_counts),
        "cell_count_trajectory": cell_counts,
        "founder_composition": founder_composition,
        "final_founder_counts": founder_composition[-1],
        "spatial_spreads": spatial_spreads,
        "n_steps": len(states) - 1,
    }


@torch.no_grad()
def run_deletion_experiment(
    model: AutoregressiveNemaModel,
    initial_state: BranchingState,
    device: str,
    target_founder: str,
    deletion_time: int,
    n_steps: int = 30,
) -> dict:
    """Run experiment deleting target founder at specified time.

    Returns comparison between control and perturbed trajectories.
    """
    founder_map = {"P0": 0, "AB": 1, "MS": 2, "E": 3, "C": 4, "D": 5, "P4": 6}
    target_id = founder_map.get(target_founder)

    if target_id is None:
        raise ValueError(f"Unknown founder: {target_founder}")

    # Control simulation
    control_states = simulate_trajectory(model, initial_state, device, n_steps)
    control_metrics = compute_trajectory_metrics(control_states)

    # Perturbation: delete target founder at deletion_time
    def perturbation_fn(state: BranchingState, t: int) -> BranchingState:
        if t != deletion_time:
            return state

        # Remove target founder cells
        keep_mask = state.states[1][0] != target_id
        keep_mask = keep_mask & state.padmask[0]

        if keep_mask.sum() == 0:
            return None

        new_cont = state.states[0][:, keep_mask, :]
        new_disc = state.states[1][:, keep_mask]
        n_kept = keep_mask.sum().item()

        return BranchingState(
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

    # Perturbed simulation
    perturbed_states = simulate_trajectory(
        model, initial_state, device, n_steps, perturbation_fn
    )
    perturbed_metrics = compute_trajectory_metrics(perturbed_states)

    # Compute cross-lineage effects
    cross_lineage_effects = {}
    for founder_name in founder_map.keys():
        if founder_name == target_founder:
            continue

        control_count = control_metrics["final_founder_counts"].get(founder_name, 0)
        perturbed_count = perturbed_metrics["final_founder_counts"].get(founder_name, 0)

        cross_lineage_effects[founder_name] = {
            "control": control_count,
            "perturbed": perturbed_count,
            "difference": perturbed_count - control_count,
            "relative_change": (
                (perturbed_count - control_count) / max(control_count, 1)
            ),
        }

    return {
        "target_founder": target_founder,
        "deletion_time": deletion_time,
        "control": control_metrics,
        "perturbed": perturbed_metrics,
        "cross_lineage_effects": cross_lineage_effects,
        "total_cell_difference": (
            perturbed_metrics["final_cell_count"] - control_metrics["final_cell_count"]
        ),
        "showed_cross_lineage_effect": any(
            abs(e["difference"]) > 0 for e in cross_lineage_effects.values()
        ),
    }


@torch.no_grad()
def run_timing_perturbation_experiment(
    model: AutoregressiveNemaModel,
    initial_state: BranchingState,
    device: str,
    delay_factor: float = 0.5,
    n_steps: int = 30,
) -> dict:
    """Test effect of delaying divisions in one lineage on others.

    Delays divisions by scaling the branch probability logits.
    """
    # Control
    control_states = simulate_trajectory(model, initial_state, device, n_steps)
    control_metrics = compute_trajectory_metrics(control_states)

    # Perturbation: delay divisions (would need model modification for full implementation)
    # For now, this is a placeholder showing the experiment structure

    return {
        "experiment": "timing_perturbation",
        "delay_factor": delay_factor,
        "control": control_metrics,
        "note": "Full implementation requires model modification for division timing control",
    }


def print_experiment_results(result: dict):
    """Print experiment results in readable format."""
    print("\n" + "=" * 70)
    print(
        f"DELETION EXPERIMENT: {result['target_founder']} at t={result['deletion_time']}"
    )
    print("=" * 70)

    print("\nControl trajectory:")
    print(f"  Final cells: {result['control']['final_cell_count']}")
    print(f"  Founder composition: {result['control']['final_founder_counts']}")

    print("\nPerturbed trajectory (after deletion):")
    print(f"  Final cells: {result['perturbed']['final_cell_count']}")
    print(f"  Founder composition: {result['perturbed']['final_founder_counts']}")

    print("\nCross-lineage effects:")
    for founder, effect in result["cross_lineage_effects"].items():
        symbol = (
            "↑"
            if effect["difference"] > 0
            else "↓"
            if effect["difference"] < 0
            else "="
        )
        print(
            f"  {founder}: {effect['control']} -> {effect['perturbed']} "
            f"({symbol}{effect['difference']:+d}, {effect['relative_change']:+.1%})"
        )

    status = "✓ DETECTED" if result["showed_cross_lineage_effect"] else "✗ NOT DETECTED"
    print(f"\nCross-lineage influence: {status}")
    print(f"  Total cell difference: {result['total_cell_difference']:+d}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Cross-lineage perturbation experiments"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--trajectory_file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--target_founder", type=str, default="AB")
    parser.add_argument("--deletion_time", type=int, default=5)
    parser.add_argument("--n_steps", type=int, default=30)
    parser.add_argument("--all_founders", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("CROSS-LINEAGE PERTURBATION EXPERIMENTS")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Trajectory: {args.trajectory_file}")
    print()

    # Load dataset
    print("Loading dataset...")
    dataset = EmbryoTrajectoryDataset(args.trajectory_file)

    if len(dataset) == 0:
        print("ERROR: No trajectory data!")
        return

    # Load model
    print("Loading model...")
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

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    print()

    # Get initial state
    initial = dataset[0]["current"].to(args.device)

    # Run experiments
    all_results = []

    if args.all_founders:
        founders = ["AB", "MS", "E", "C", "D"]
    else:
        founders = [args.target_founder]

    for founder in founders:
        print(f"\nRunning deletion experiment: {founder}...")
        result = run_deletion_experiment(
            model,
            initial,
            args.device,
            target_founder=founder,
            deletion_time=args.deletion_time,
            n_steps=args.n_steps,
        )
        all_results.append(result)
        print_experiment_results(result)

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    n_with_effects = sum(1 for r in all_results if r["showed_cross_lineage_effect"])
    print(
        f"\nExperiments showing cross-lineage effects: {n_with_effects}/{len(all_results)}"
    )

    for result in all_results:
        status = "✓" if result["showed_cross_lineage_effect"] else "✗"
        print(
            f"  {status} {result['target_founder']}: "
            f"Δtotal={result['total_cell_difference']:+d}"
        )

    print("=" * 70)

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
