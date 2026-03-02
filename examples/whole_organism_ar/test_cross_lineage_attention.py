#!/usr/bin/env python3
"""Cross-lineage attention validation test.

Validates that the model learns cross-lineage interactions by measuring
prediction differences when cells from other lineages are masked.

Biological principle: Cells from different lineages (AB, MS, E, etc.) coexist
in the embryo and influence each other through physical contact and signaling.
A model with true whole-embryo context should show different predictions
when other lineages are present vs. masked.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.branching_flows.autoregressive_model import AutoregressiveNemaModel  # noqa: E402
from src.branching_flows.states import BranchingState  # noqa: E402
from examples.whole_organism_ar.train_autoregressive_full import EmbryoTrajectoryDataset  # noqa: E402


def isolate_founder(
    state: BranchingState, founder_id: int, device: str
) -> BranchingState:
    """Create isolated state with only cells of specified founder."""
    founder_mask = (state.states[1][0] == founder_id) & state.padmask[0]

    if founder_mask.sum() == 0:
        return None

    isolated_indices = torch.where(founder_mask)[0]
    new_cont = state.states[0][:, isolated_indices, :]
    new_disc = state.states[1][:, isolated_indices]
    n_kept = isolated_indices.shape[0]

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
def test_cross_lineage_attention(
    model: AutoregressiveNemaModel,
    dataset: EmbryoTrajectoryDataset,
    device: str,
    threshold: float = 0.1,
) -> dict:
    """Test if model uses cross-lineage attention.

    Method:
    1. For each multi-founder time point, do normal forward pass
    2. Isolate each founder (mask all others) and forward again
    3. Compare gene delta predictions

    Significant difference indicates the model uses cross-lineage context.
    """
    model.eval()

    founder_map_inv = {0: "P0", 1: "AB", 2: "MS", 3: "E", 4: "C", 5: "D", 6: "P4"}

    results = {
        "time_points_tested": 0,
        "founder_pairs": [],
        "per_founder_summary": {
            name: {"tests": 0, "total_diff": 0.0} for name in founder_map_inv.values()
        },
    }

    for idx in range(len(dataset)):
        sample = dataset[idx]
        current = sample["current"].to(device)

        # Check if this time point has multiple founders
        founder_ids = current.states[1][0][current.padmask[0]]
        unique_founders = torch.unique(founder_ids).tolist()

        if len(unique_founders) < 2:
            continue

        results["time_points_tested"] += 1

        # Full prediction
        output_full = model.forward_step(current)

        # Test each founder in isolation
        for founder_id in unique_founders:
            founder_name = founder_map_inv.get(founder_id, f"ID_{founder_id}")

            # Create isolated state
            isolated_state = isolate_founder(current, founder_id, device)
            if isolated_state is None:
                continue

            # Isolated prediction
            output_isolated = model.forward_step(isolated_state)

            # Find this founder's cells in full prediction
            founder_mask_full = (current.states[1][0] == founder_id) & current.padmask[
                0
            ]
            full_gene_delta = output_full.gene_delta[0, founder_mask_full, :]
            isolated_gene_delta = output_isolated.gene_delta[0, :, :]

            # Compute difference
            diff = (
                torch.norm(full_gene_delta - isolated_gene_delta, dim=-1).mean().item()
            )

            results["founder_pairs"].append(
                {
                    "time_idx": idx,
                    "time": sample["time"],
                    "founder": founder_name,
                    "n_cells": int(founder_mask_full.sum()),
                    "delta_diff": diff,
                    "uses_context": diff > threshold,
                }
            )

            results["per_founder_summary"][founder_name]["tests"] += 1
            results["per_founder_summary"][founder_name]["total_diff"] += diff

    # Compute averages
    all_diffs = [p["delta_diff"] for p in results["founder_pairs"]]
    results["mean_delta_diff"] = sum(all_diffs) / len(all_diffs) if all_diffs else 0.0
    results["max_delta_diff"] = max(all_diffs) if all_diffs else 0.0
    results["cross_lineage_detected"] = results["mean_delta_diff"] > threshold
    results["n_context_users"] = sum(
        1 for p in results["founder_pairs"] if p["uses_context"]
    )

    for founder_name in results["per_founder_summary"]:
        summary = results["per_founder_summary"][founder_name]
        if summary["tests"] > 0:
            summary["mean_diff"] = summary["total_diff"] / summary["tests"]
        else:
            summary["mean_diff"] = 0.0

    return results


def print_results(results: dict):
    """Print test results in readable format."""
    print("\n" + "=" * 70)
    print("CROSS-LINEAGE ATTENTION TEST RESULTS")
    print("=" * 70)

    print(f"\nTime points tested: {results['time_points_tested']}")
    print(f"Founder instances tested: {len(results['founder_pairs'])}")

    print("\nDelta Difference Statistics:")
    print(f"  Mean: {results['mean_delta_diff']:.4f}")
    print(f"  Max:  {results['max_delta_diff']:.4f}")

    status = "✓ DETECTED" if results["cross_lineage_detected"] else "✗ NOT DETECTED"
    print(f"\nCross-lineage attention: {status}")
    print(
        f"  ({results['n_context_users']}/{len(results['founder_pairs'])} instances use context)"
    )

    print("\nPer-founder analysis:")
    for founder, summary in results["per_founder_summary"].items():
        if summary["tests"] > 0:
            indicator = "✓" if summary["mean_diff"] > 0.1 else "✗"
            print(
                f"  {indicator} {founder}: {summary['tests']} tests, "
                f"mean_diff={summary['mean_diff']:.4f}"
            )

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Test cross-lineage attention in trained model"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--trajectory_file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("CROSS-LINEAGE ATTENTION VALIDATION")
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

    # Run test
    print("Running cross-lineage attention test...")
    results = test_cross_lineage_attention(model, dataset, args.device, args.threshold)

    print_results(results)

    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
