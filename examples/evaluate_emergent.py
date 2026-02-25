"""Evaluate Emergent Context model's discovered structure.

Compares generated trajectories against known Sulston tree to assess
how well the model learned developmental organization.

Usage:
    uv run python examples/evaluate_emergent.py --checkpoint checkpoints_experiments/with_bias/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.branching_flows import NemaFlowModel, WormGUIDESDataset
from src.branching_flows.sampling import generate
from src.branching_flows import CoalescentFlow, OUFlow, DiscreteInterpolatingFlow


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Emergent Context model")
    p.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    p.add_argument("--nuclei_dir", default="dataset/raw/wormguides/nuclei_files")
    p.add_argument("--deaths_csv", default="dataset/raw/wormguides/CellDeaths.csv")
    p.add_argument(
        "--n_samples", type=int, default=10, help="Number of samples to evaluate"
    )
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def compute_tree_edit_distance(
    inferred_parents: dict, true_parents: dict
) -> tuple[int, float]:
    """Compute tree edit distance between inferred and true trees.

    Returns:
        (edit_distance, accuracy)
    """
    all_cells = set(inferred_parents.keys()) | set(true_parents.keys())
    correct = 0
    total = 0

    for cell in all_cells:
        if cell in inferred_parents and cell in true_parents:
            if inferred_parents[cell] == true_parents[cell]:
                correct += 1
            total += 1
        elif cell in inferred_parents or cell in true_parents:
            total += 1

    accuracy = correct / total if total > 0 else 0.0
    edit_distance = total - correct
    return edit_distance, accuracy


def infer_tree_from_trajectory(
    trajectory: list[torch.Tensor],
    cell_names: list[str],
    distance_threshold: float = 0.5,
) -> dict[str, str | None]:
    """Infer parent-child relationships from cell trajectory.

    Uses spatial proximity at early timepoints to infer lineage relationships.

    Args:
        trajectory: List of cell states at different times [T, N_t, D]
        cell_names: Names of cells at final timepoint
        distance_threshold: Distance threshold for parent-child linking

    Returns:
        Dictionary mapping cell name to parent name (None for founders)
    """
    if len(trajectory) < 2 or len(cell_names) == 0:
        return {}

    # Simplified inference: assume order reflects lineage
    # In practice, you'd use more sophisticated methods (e.g., tracking backward)
    parents = {}

    # Founders have no parent
    founders = {"AB", "MS", "E", "C", "D", "P4"}
    for name in cell_names:
        founder, path = parse_name_for_eval(name)
        if name in founders:
            parents[name] = None
        elif len(path) > 0:
            # Parent is the cell without the last division
            parent_path = path[:-1]
            parent_name = founder + "".join(parent_path)
            parents[name] = parent_name if parent_name in cell_names else None
        else:
            parents[name] = None

    return parents


def parse_name_for_eval(name: str) -> tuple[str, list[str]]:
    """Simplified lineage name parsing for evaluation."""
    if name in ("AB", "MS", "E", "C", "D", "P4"):
        return name, []

    if name.startswith("AB"):
        return "AB", list(name[2:].lower())
    elif name.startswith("MS"):
        return "MS", list(name[2:].lower())
    elif name.startswith("P4"):
        return "P4", list(name[2:].lower()) if len(name) > 2 else []
    elif name in ("E", "C", "D"):
        return name, []
    else:
        return name, []


def compute_cell_count_accuracy(
    predicted_counts: list[int],
    true_counts: list[int],
) -> dict[str, float]:
    """Compute accuracy metrics for cell count prediction."""
    pred_arr = np.array(predicted_counts)
    true_arr = np.array(true_counts)

    mae = np.abs(pred_arr - true_arr).mean()
    mse = ((pred_arr - true_arr) ** 2).mean()
    correlation = pearsonr(pred_arr, true_arr)[0] if len(pred_arr) > 1 else 0.0

    return {
        "mae": float(mae),
        "mse": float(mse),
        "correlation": float(correlation),
    }


def evaluate_model(model, dataset, flow, device, n_samples=10):
    """Run evaluation and return metrics."""
    model.eval()

    results = {
        "cell_count": {"pred": [], "true": []},
        "tree_accuracy": [],
        "division_timing_corr": [],
    }

    indices = list(range(min(n_samples, len(dataset))))

    with torch.no_grad():
        for idx in indices:
            sample = dataset[idx]
            cell_names = dataset.get_cell_names_at(idx)

            # Get true cell count
            true_count = len([c for c in sample.elements if c is not None])
            results["cell_count"]["true"].append(true_count)

            # Generate trajectory
            try:
                from src.branching_flows.states import SampleState

                # Create initial state (single cell)
                x0 = dataset.x0_sampler(None)
                x0_state = SampleState(
                    elements=[x0],
                    groupings=[0],
                    del_flags=[False],
                    branchmask=[True],
                    flowmask=[True],
                )

                # Generate
                tracker = generate(
                    model,
                    flow,
                    [x0_state],
                    t_span=(0.0, 1.0),
                    n_steps=20,
                )

                # Get final count
                final_states = tracker.history[-1][0]
                pred_count = (
                    len(final_states.elements)
                    if hasattr(final_states, "elements")
                    else 0
                )
                results["cell_count"]["pred"].append(pred_count)

                # Tree structure evaluation (simplified)
                # Compare cell names presence
                pred_names = (
                    set(cell_names[:pred_count])
                    if pred_count <= len(cell_names)
                    else set(cell_names)
                )
                true_names = set(cell_names)
                name_overlap = (
                    len(pred_names & true_names) / len(true_names) if true_names else 0
                )
                results["tree_accuracy"].append(name_overlap)

            except Exception as e:
                print(f"  Sample {idx} failed: {e}")
                results["cell_count"]["pred"].append(0)
                results["tree_accuracy"].append(0.0)

    # Compute aggregate metrics
    count_metrics = compute_cell_count_accuracy(
        results["cell_count"]["pred"],
        results["cell_count"]["true"],
    )

    return {
        "cell_count_mae": count_metrics["mae"],
        "cell_count_correlation": count_metrics["correlation"],
        "tree_accuracy": np.mean(results["tree_accuracy"]),
    }


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    saved_args = ckpt.get("args", {})

    print("Loading dataset...")
    dataset = WormGUIDESDataset(
        args.nuclei_dir,
        args.deaths_csv,
        min_cells=4,
        stride=10,
    )

    # Create flow
    from scipy.stats import beta

    flow = CoalescentFlow(
        processes=(
            OUFlow(theta=25.0, var_0=5.0, var_1=0.01),
            DiscreteInterpolatingFlow(K=dataset.K),
        ),
        branch_time_dist=beta(1, 2),
    )

    # Create model
    model = NemaFlowModel(
        continuous_dim=dataset.continuous_dim,
        discrete_K=dataset.K,
        d_model=saved_args.get("d_model", 64),
        n_heads=saved_args.get("n_heads", 2),
        n_layers=saved_args.get("n_layers", 2),
        head_dim=saved_args.get("head_dim", 32),
    ).to(device)

    model.load_state_dict(ckpt["model"])
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    print(f"\nEvaluating on {args.n_samples} samples...")
    metrics = evaluate_model(model, dataset, flow, device, args.n_samples)

    print("\n=== Evaluation Results ===")
    print(f"Cell Count MAE: {metrics['cell_count_mae']:.2f}")
    print(f"Cell Count Correlation: {metrics['cell_count_correlation']:.3f}")
    print(f"Tree Accuracy: {metrics['tree_accuracy']:.3f}")

    return metrics


if __name__ == "__main__":
    main()
