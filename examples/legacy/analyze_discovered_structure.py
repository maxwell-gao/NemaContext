"""Analyze discovered developmental structure from Emergent Context model.

Compares model-generated trajectories against the known Sulston tree to assess
how well the model learned developmental organization without explicit tree
supervision.

Usage:
    uv run python examples/analyze_discovered_structure.py \
        --checkpoint checkpoints/experiments/with_bias/best.pt \
        --output_dir analysis_results/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.branching_flows import (
    CoalescentFlow,
    DiscreteInterpolatingFlow,
    OUFlow,
    generate,
)
from src.branching_flows.legacy.nema_model import NemaFlowModel
from src.branching_flows.wormguides_dataset import WormGUIDESDataset
from src.branching_flows.legacy.lineage import parse_lineage_name
from src.branching_flows.states import BranchingState


def parse_args():
    p = argparse.ArgumentParser(description="Analyze discovered structure")
    p.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    p.add_argument("--nuclei_dir", default="dataset/raw/wormguides/nuclei_files")
    p.add_argument("--deaths_csv", default="dataset/raw/wormguides/CellDeaths.csv")
    p.add_argument("--output_dir", default="analysis_results")
    p.add_argument(
        "--n_samples", type=int, default=20, help="Number of samples to analyze"
    )
    p.add_argument("--n_generation_steps", type=int, default=20)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def infer_tree_from_trajectory(
    trajectory: list[BranchingState],
    cell_names: list[str] | None = None,
) -> dict[str, str | None]:
    """Infer parent-child relationships from a generated trajectory.

    Strategy:
    1. Track cell count changes over time
    2. When count increases, a division occurred
    3. Map new cells to their most likely parent based on:
       - Position continuity (spatial proximity)
       - Lineage naming convention (if names available)

    Returns:
        Dict mapping cell index/name to parent index/name (None for founders)
    """
    parents = {}
    cell_positions = {}  # Track cell positions over time

    for t_idx, state in enumerate(trajectory):
        # BranchingState uses 'states' tuple
        cont = state.states[0][0]  # [L, D] - first batch item
        n_cells = cont.shape[0]

        # Store positions for this timestep
        if t_idx == 0:
            # Initialize founders
            for i in range(n_cells):
                cell_id = f"cell_{t_idx}_{i}"
                cell_positions[cell_id] = cont[i, :3].detach().cpu().numpy()
                parents[cell_id] = None  # Founders have no parent
        else:
            # Track which cells are new
            prev_positions = [
                cell_positions.get(f"cell_{t_idx - 1}_{i}")
                for i in range(trajectory[t_idx - 1].states[0][0].shape[0])
            ]

            for i in range(n_cells):
                curr_pos = cont[i, :3].detach().cpu().numpy()
                cell_id = f"cell_{t_idx}_{i}"

                # Find closest previous cell as potential parent
                if prev_positions:
                    distances = [
                        np.linalg.norm(curr_pos - pp)
                        if pp is not None
                        else float("inf")
                        for pp in prev_positions
                    ]
                    closest_idx = np.argmin(distances)
                    min_dist = distances[closest_idx]

                    # Threshold for "same cell" vs "new cell"
                    if min_dist < 2.0:  # Same cell moved
                        parent_id = f"cell_{t_idx - 1}_{closest_idx}"
                        parents[cell_id] = parent_id
                    else:  # New cell from division
                        parent_id = f"cell_{t_idx - 1}_{closest_idx}"
                        parents[cell_id] = parent_id

                cell_positions[cell_id] = curr_pos

    return parents


def compute_tree_edit_distance(
    inferred_parents: dict[str, str | None],
    true_parents: dict[str, str | None],
) -> tuple[int, float]:
    """Compute simplified tree edit distance.

    Returns:
        (edit_distance, accuracy)
    """
    all_cells = set(inferred_parents.keys()) | set(true_parents.keys())

    correct = 0
    total = 0

    for cell in all_cells:
        inf_parent = inferred_parents.get(cell)
        true_parent = true_parents.get(cell)

        if inf_parent == true_parent:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    edit_distance = total - correct

    return edit_distance, accuracy


def build_sulston_tree_reference(
    cell_names: list[str],
    division_events: list,
) -> dict[str, str | None]:
    """Build reference parent map from known Sulston tree."""
    parents = {}

    # Known founders
    founders = {"AB", "MS", "E", "C", "D", "P4"}

    for name in cell_names:
        if name in founders:
            parents[name] = None
        else:
            # Parse lineage name to find parent
            founder, path = parse_lineage_name(name)
            if len(path) == 0:
                parents[name] = None
            else:
                # Parent is name without last character
                parent_name = founder + "".join(path[:-1])
                if parent_name in cell_names:
                    parents[name] = parent_name
                else:
                    parents[name] = None

    return parents


def compute_cell_count_profile(
    trajectory: list[BranchingState],
) -> np.ndarray:
    """Extract cell count over time from trajectory."""
    return np.array([state.states[0].shape[1] for state in trajectory])


def compare_cell_count_dynamics(
    generated_counts: np.ndarray,
    true_counts: np.ndarray,
) -> dict[str, float]:
    """Compare cell count dynamics between generated and true."""
    # Interpolate to same length
    t_gen = np.linspace(0, 1, len(generated_counts))
    t_true = np.linspace(0, 1, len(true_counts))

    gen_interp = np.interp(t_true, t_gen, generated_counts)

    # Metrics
    mae = np.abs(gen_interp - true_counts).mean()
    correlation, _ = pearsonr(gen_interp, true_counts)
    max_diff = np.abs(gen_interp - true_counts).max()

    return {
        "mae": float(mae),
        "correlation": float(correlation),
        "max_diff": float(max_diff),
    }


def analyze_model(args):
    """Main analysis function."""
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        d_model=saved_args.get("d_model", 128),
        n_heads=saved_args.get("n_heads", 4),
        n_layers=saved_args.get("n_layers", 4),
        head_dim=saved_args.get("head_dim", 32),
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # Analysis results
    results = {
        "n_samples": 0,
        "tree_accuracy": [],
        "tree_edit_distance": [],
        "cell_count_metrics": [],
        "division_events_detected": [],
    }

    print(f"\nAnalyzing {args.n_samples} samples...")

    with torch.no_grad():
        for idx in range(min(args.n_samples, len(dataset))):
            print(f"\nSample {idx + 1}/{args.n_samples}")

            sample = dataset[idx]
            cell_names = dataset.get_cell_names_at(idx)

            # Get true Sulston tree
            true_parents = build_sulston_tree_reference(
                cell_names,
                dataset.get_division_events(),
            )

            # Get true cell count profile
            true_counts = len(sample.elements)

            # Generate trajectory
            try:
                x0 = dataset.x0_sampler(None)
                # Convert to BranchingState (batch=1)
                cont = x0[0].unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, D]
                disc = torch.tensor([[x0[1]]], device=device)
                x0_state = BranchingState(
                    states=(cont, disc),
                    groupings=torch.zeros(1, 1, dtype=torch.long, device=device),
                    del_flags=torch.zeros(1, 1, dtype=torch.bool, device=device),
                    ids=torch.zeros(1, 1, dtype=torch.long, device=device),
                    branchmask=torch.ones(1, 1, dtype=torch.bool, device=device),
                    flowmask=torch.ones(1, 1, dtype=torch.bool, device=device),
                    padmask=torch.ones(1, 1, dtype=torch.bool, device=device),
                )

                # Create timesteps
                timesteps = torch.linspace(0, 1, args.n_generation_steps)

                # Create model wrapper
                def model_fn(t, state):
                    t_tensor = torch.tensor([t], device=device)
                    return model(t_tensor, state)

                # Generate trajectory
                final_state = generate(flow, x0_state, model_fn, timesteps)

                # For now, just use start and end state
                trajectory = [x0_state, final_state]

                # Infer tree from trajectory
                inferred_parents = infer_tree_from_trajectory(trajectory, cell_names)

                # Compare trees
                ted, accuracy = compute_tree_edit_distance(
                    inferred_parents, true_parents
                )

                # Cell count analysis
                gen_counts = compute_cell_count_profile(trajectory)

                print(f"  True cells: {true_counts}, Generated: {gen_counts[-1]}")
                print(f"  Tree edit distance: {ted}")
                print(f"  Tree accuracy: {accuracy:.3f}")

                results["n_samples"] += 1
                results["tree_accuracy"].append(accuracy)
                results["tree_edit_distance"].append(ted)
                results["division_events_detected"].append(len(trajectory) - 1)

            except Exception as e:
                print(f"  Error: {e}")
                continue

    # Aggregate results
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if results["tree_accuracy"]:
        mean_acc = np.mean(results["tree_accuracy"])
        mean_ted = np.mean(results["tree_edit_distance"])

        print(f"Samples analyzed: {results['n_samples']}")
        print(f"Mean tree accuracy: {mean_acc:.3f}")
        print(f"Mean tree edit distance: {mean_ted:.1f}")
        print(
            f"Mean divisions detected: {np.mean(results['division_events_detected']):.1f}"
        )

        # Save results
        summary = {
            "mean_tree_accuracy": float(mean_acc),
            "mean_tree_edit_distance": float(mean_ted),
            "n_samples": results["n_samples"],
            "individual_results": {
                "tree_accuracy": results["tree_accuracy"],
                "tree_edit_distance": results["tree_edit_distance"],
            },
        }

        output_file = output_dir / "structure_analysis.json"
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to: {output_file}")

        # Interpretation
        print("\nInterpretation:")
        if mean_acc > 0.8:
            print(
                "  ✅ EXCELLENT: Model learned structure highly consistent with Sulston tree"
            )
        elif mean_acc > 0.6:
            print("  ✅ GOOD: Model captured major developmental organization")
        elif mean_acc > 0.4:
            print("  ⚠️  MODERATE: Partial structure discovery")
        else:
            print("  ❌ WEAK: Structure not well learned")

    return results


if __name__ == "__main__":
    args = parse_args()
    analyze_model(args)
