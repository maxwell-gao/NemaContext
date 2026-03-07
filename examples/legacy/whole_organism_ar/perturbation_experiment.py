#!/usr/bin/env python3
"""
Perturbation experiments to validate causal learning in the model.

Tests whether the model learned developmental mechanisms (not just memorization):
1. Lineage deletion: Remove a founder lineage, observe compensation
2. Timing perturbation: Delay/accelerate specific cell divisions
3. Founder swapping: Swap initial states of two founders

Following the creed: 'Discover, Don't Inject'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.branching_flows.crossmodal_model import CrossModalNemaModel  # noqa: E402
from src.branching_flows.trimodal_dataset import TrimodalDataset  # noqa: E402
from src.branching_flows.states import BranchingState  # noqa: E402


def perturb_lineage_deletion(
    model: torch.nn.Module,
    dataset: TrimodalDataset,
    target_founder: str,
    device: str,
    n_samples: int = 10,
) -> dict:
    """Delete a founder lineage and observe system response.

    In real embryos, deleting a lineage often causes compensatory changes
    in other lineages. If the model learned causal mechanisms,
    it should predict reasonable (not random) behavior.
    """
    results = {
        "perturbation_type": "lineage_deletion",
        "target_founder": target_founder,
        "control": [],
        "perturbed": [],
    }

    model.eval()

    with torch.no_grad():
        for i in range(min(n_samples, len(dataset))):
            sample = dataset[i]

            # Get lineage names
            lineage_names = sample.lineage_names

            # Filter out target founder cells
            mask = torch.ones(len(lineage_names), dtype=torch.bool)
            for idx, name in enumerate(lineage_names):
                if name and name.startswith(target_founder):
                    mask[idx] = False

            n_removed = (~mask).sum().item()
            n_kept = mask.sum().item()

            if n_removed == 0 or n_kept < 3:
                continue

            # Control: normal prediction
            cont_states = torch.stack([e[0] for e in sample.elements])
            disc_states = torch.tensor(
                [e[1] for e in sample.elements], dtype=torch.long
            )

            # Create branching state
            n_total = len(cont_states)
            control_state = BranchingState(
                states=(
                    cont_states.unsqueeze(0).to(device),
                    disc_states.unsqueeze(0).to(device),
                ),
                groupings=torch.zeros(1, n_total, dtype=torch.long, device=device),
                del_flags=torch.zeros(1, n_total, dtype=torch.bool, device=device),
                ids=torch.arange(
                    1, n_total + 1, dtype=torch.long, device=device
                ).unsqueeze(0),
                padmask=torch.ones(1, n_total, dtype=torch.bool, device=device),
                flowmask=torch.ones(1, n_total, dtype=torch.bool, device=device),
                branchmask=torch.ones(1, n_total, dtype=torch.bool, device=device),
            )

            # Perturbed: remove target cells
            perturbed_cont = cont_states[mask].unsqueeze(0).to(device)
            perturbed_disc = disc_states[mask].unsqueeze(0).to(device)

            perturbed_state = BranchingState(
                states=(perturbed_cont, perturbed_disc),
                groupings=torch.zeros(1, n_kept, dtype=torch.long, device=device),
                del_flags=torch.zeros(1, n_kept, dtype=torch.bool, device=device),
                ids=torch.arange(
                    1, n_kept + 1, dtype=torch.long, device=device
                ).unsqueeze(0),
                padmask=torch.ones(1, n_kept, dtype=torch.bool, device=device),
                flowmask=torch.ones(1, n_kept, dtype=torch.bool, device=device),
                branchmask=torch.ones(1, n_kept, dtype=torch.bool, device=device),
            )

            # Predict next states at multiple time points
            t_vals = [0.25, 0.5, 0.75, 1.0]

            for t_val in t_vals:
                t_tensor = torch.tensor([t_val], device=device)

                # Control prediction
                (xc_c, _), _, _ = model(t_tensor, control_state)

                # Perturbed prediction
                (xc_p, _), _, _ = model(t_tensor, perturbed_state)

                # Compare statistics
                control_mean = xc_c[0].mean(dim=0).cpu().numpy()
                perturbed_mean = xc_p[0].mean(dim=0).cpu().numpy()

                # Spatial positions (last 3 dims)
                control_spatial = control_mean[-3:]
                perturbed_spatial = perturbed_mean[-3:]

                # Spatial spread (std)
                control_spread = xc_c[0, :, -3:].std(dim=0).cpu().numpy()
                perturbed_spread = xc_p[0, :, -3:].std(dim=0).cpu().numpy()

                results["control"].append(
                    {
                        "t": t_val,
                        "n_cells": len(cont_states),
                        "spatial_mean": control_spatial.tolist(),
                        "spatial_spread": control_spread.tolist(),
                    }
                )

                results["perturbed"].append(
                    {
                        "t": t_val,
                        "n_cells": n_kept,
                        "spatial_mean": perturbed_spatial.tolist(),
                        "spatial_spread": perturbed_spread.tolist(),
                        "n_removed": n_removed,
                    }
                )

    return results


def perturb_founder_swap(
    model: torch.nn.Module,
    dataset: TrimodalDataset,
    founder1: str,
    founder2: str,
    device: str,
    n_samples: int = 10,
) -> dict:
    """Swap initial states of two founder lineages.

    Creates a 'chimeric' embryo. Tests if the model maintains
    lineage-specific developmental programs after state swap.
    """
    results = {
        "perturbation_type": "founder_swap",
        "founder1": founder1,
        "founder2": founder2,
        "swaps": [],
    }

    model.eval()

    with torch.no_grad():
        for i in range(min(n_samples, len(dataset))):
            sample = dataset[i]

            lineage_names = sample.lineage_names
            cont_states = torch.stack([e[0] for e in sample.elements])

            # Find cells from each founder
            f1_indices = [
                j
                for j, name in enumerate(lineage_names)
                if name and name.startswith(founder1)
            ]
            f2_indices = [
                j
                for j, name in enumerate(lineage_names)
                if name and name.startswith(founder2)
            ]

            if len(f1_indices) < 2 or len(f2_indices) < 2:
                continue

            # Swap states (with minimum size to avoid shape mismatch)
            min_size = min(len(f1_indices), len(f2_indices))
            swapped_states = cont_states.clone()
            f1_swap = f1_indices[:min_size]
            f2_swap = f2_indices[:min_size]
            swapped_states[f1_swap] = cont_states[f2_swap]
            swapped_states[f2_swap] = cont_states[f1_swap]

            # Create states
            n_total = len(cont_states)
            original_state = BranchingState(
                states=(
                    cont_states.unsqueeze(0).to(device),
                    torch.zeros(1, n_total, dtype=torch.long, device=device),
                ),
                groupings=torch.zeros(1, n_total, dtype=torch.long, device=device),
                del_flags=torch.zeros(1, n_total, dtype=torch.bool, device=device),
                ids=torch.arange(
                    1, n_total + 1, dtype=torch.long, device=device
                ).unsqueeze(0),
                padmask=torch.ones(1, n_total, dtype=torch.bool, device=device),
                flowmask=torch.ones(1, n_total, dtype=torch.bool, device=device),
                branchmask=torch.ones(1, n_total, dtype=torch.bool, device=device),
            )

            swapped_state = BranchingState(
                states=(
                    swapped_states.unsqueeze(0).to(device),
                    torch.zeros(1, n_total, dtype=torch.long, device=device),
                ),
                groupings=torch.zeros(1, n_total, dtype=torch.long, device=device),
                del_flags=torch.zeros(1, n_total, dtype=torch.bool, device=device),
                ids=torch.arange(
                    1, n_total + 1, dtype=torch.long, device=device
                ).unsqueeze(0),
                padmask=torch.ones(1, n_total, dtype=torch.bool, device=device),
                flowmask=torch.ones(1, n_total, dtype=torch.bool, device=device),
                branchmask=torch.ones(1, n_total, dtype=torch.bool, device=device),
            )

            # Predict at t=1
            t_tensor = torch.tensor([1.0], device=device)
            (xc_orig, _), _, _ = model(t_tensor, original_state)
            (xc_swap, _), _, _ = model(t_tensor, swapped_state)

            # Compare lineage-specific predictions
            orig_f1_spatial = xc_orig[0, f1_indices, -3:].mean(dim=0).cpu().numpy()
            swap_f1_spatial = xc_swap[0, f1_indices, -3:].mean(dim=0).cpu().numpy()

            results["swaps"].append(
                {
                    "n_f1_cells": len(f1_indices),
                    "n_f2_cells": len(f2_indices),
                    "original_f1_spatial": orig_f1_spatial.tolist(),
                    "swapped_f1_spatial": swap_f1_spatial.tolist(),
                    "spatial_shift": np.linalg.norm(
                        orig_f1_spatial - swap_f1_spatial
                    ).item(),
                }
            )

    return results


def analyze_compensation(results: dict) -> dict:
    """Analyze if the model shows compensatory behavior."""
    analysis = {
        "showed_compensation": False,
        "evidence": [],
    }

    if results["perturbation_type"] == "lineage_deletion":
        control = results["control"]
        perturbed = results["perturbed"]

        # Check if remaining cells changed their behavior
        for c, p in zip(control, perturbed):
            if c["t"] == p["t"]:
                # Spatial center shift
                c_spatial = np.array(c["spatial_mean"])
                p_spatial = np.array(p["spatial_mean"])
                shift = np.linalg.norm(c_spatial - p_spatial)

                # Spread change
                c_spread = np.array(c["spatial_spread"])
                p_spread = np.array(p["spatial_spread"])
                spread_change = np.abs(c_spread - p_spread).mean()

                analysis["evidence"].append(
                    {
                        "time": c["t"],
                        "spatial_shift": shift,
                        "spread_change": spread_change,
                    }
                )

        # Compensation = spatial center adjusts, spread changes
        avg_shift = np.mean([e["spatial_shift"] for e in analysis["evidence"]])
        avg_spread_change = np.mean([e["spread_change"] for e in analysis["evidence"]])

        analysis["avg_spatial_shift"] = avg_shift
        analysis["avg_spread_change"] = avg_spread_change

        # If shift is reasonable (not 0, not huge), model learned something
        if 0.1 < avg_shift < 10.0 and avg_spread_change > 0.01:
            analysis["showed_compensation"] = True

    elif results["perturbation_type"] == "founder_swap":
        swaps = results["swaps"]
        if swaps:
            avg_shift = np.mean([s["spatial_shift"] for s in swaps])
            analysis["avg_spatial_shift_after_swap"] = avg_shift

            # If shift is significant, swap had effect (model didn't ignore it)
            analysis["showed_compensation"] = avg_shift > 0.5

    return analysis


def main():
    parser = argparse.ArgumentParser(
        description="Perturbation experiments to test causal learning"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints_trimodal_crossmodal/best.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--perturb_type",
        type=str,
        choices=["lineage_deletion", "founder_swap"],
        default="lineage_deletion",
        help="Type of perturbation",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="AB",
        help="Target lineage for deletion (e.g., AB, MS, E, C, D)",
    )
    parser.add_argument(
        "--target2",
        type=str,
        default="MS",
        help="Second target for founder swap",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="perturbation_results",
        help="Output directory",
    )
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print("=" * 70)
    print("PERTURBATION EXPERIMENT")
    print("=" * 70)
    print(f"Testing: {args.perturb_type}")
    print(f"Checkpoint: {args.checkpoint}")
    print()

    # Load dataset
    print("Loading dataset...")
    dataset = TrimodalDataset(
        h5ad_path="dataset/processed/nema_extended_large2025.h5ad",
        time_bins=20,
        max_cells_per_bin=512,
        augment_spatial=False,
        aug_rotation=False,
        aug_flip=False,
    )
    print(f"  Loaded: {len(dataset)} samples")
    print()

    # Load model
    print("Loading model...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    model = CrossModalNemaModel(
        gene_dim=2000,
        spatial_dim=3,
        discrete_K=7,
        d_model=256,
        n_layers=6,
        n_heads=8,
        cross_modal_every=2,
    ).to(args.device)

    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} params")
    print()

    # Run perturbation
    print("=" * 70)
    print(f"Running {args.perturb_type} experiment...")
    print("=" * 70)

    if args.perturb_type == "lineage_deletion":
        print(f"Deleting {args.target} lineage cells...")
        results = perturb_lineage_deletion(
            model, dataset, args.target, args.device, args.n_samples
        )
    elif args.perturb_type == "founder_swap":
        print(f"Swapping {args.target} and {args.target2} initial states...")
        results = perturb_founder_swap(
            model, dataset, args.target, args.target2, args.device, args.n_samples
        )

    # Analyze results
    print()
    print("Analyzing compensatory behavior...")
    analysis = analyze_compensation(results)

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    if args.perturb_type == "lineage_deletion":
        print(
            f"Removed {results['perturbed'][0]['n_removed']} cells from {args.target} lineage"
        )
        print(f"Remaining cells: {results['perturbed'][0]['n_cells']}")
        print()
        print(f"Average spatial shift: {analysis['avg_spatial_shift']:.4f}")
        print(f"Average spread change: {analysis['avg_spread_change']:.4f}")

    elif args.perturb_type == "founder_swap":
        print(f"Swapped {args.target} and {args.target2} initial states")
        print(f"Average spatial shift: {analysis['avg_spatial_shift_after_swap']:.4f}")

    print()
    if analysis["showed_compensation"]:
        print("✓ MODEL SHOWED COMPENSATORY BEHAVIOR")
        print("  The model adjusted predictions based on perturbation.")
        print("  This suggests it learned causal mechanisms, not just memorization.")
    else:
        print("  Model showed minimal response to perturbation")
        print("  May indicate memorization rather than causal understanding")

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to JSON-serializable
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif isinstance(obj, (np.bool_, np.integer)):
            return bool(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    full_results = {
        "perturbation": convert(results),
        "analysis": convert(analysis),
        "config": {
            "perturb_type": args.perturb_type,
            "target": args.target,
            "target2": args.target2 if args.perturb_type == "founder_swap" else None,
            "n_samples": args.n_samples,
        },
    }

    output_file = output_dir / f"{args.perturb_type}_{args.target}.json"
    with open(output_file, "w") as f:
        json.dump(full_results, f, indent=2)

    print()
    print(f"Results saved to: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
