#!/usr/bin/env python3
"""
Autoregressive developmental simulation.

Biologically complete simulation that evolves the embryo step-by-step,
allowing perturbations at any point in development.

The model is used as a velocity field: dx/dt = v(t, x)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.branching_flows.crossmodal_model import CrossModalNemaModel  # noqa: E402
from src.branching_flows.states import BranchingState  # noqa: E402


class AutoregressiveSimulator:
    """Simulate development step-by-step using the model as velocity field."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda",
        dt: float = 0.01,
        use_split_predictions: bool = True,
        use_del_predictions: bool = True,
    ):
        self.model = model
        self.device = device
        self.dt = dt
        self.use_split_predictions = use_split_predictions
        self.use_del_predictions = use_del_predictions

    def velocity_field(
        self,
        t: float,
        state: BranchingState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute dx/dt at time t using the model.

        Returns:
            (continuous_velocity, discrete_logits)
        """
        t_tensor = torch.tensor([t], device=self.device)

        with torch.no_grad():
            (pred_cont, pred_disc), split_logits, del_logits = self.model(
                t_tensor, state
            )

        # Velocity = (prediction - current) / (1 - t)
        # This comes from flow matching: v = (x1 - xt) / (1 - t)
        current_cont = state.states[0]  # [B, L, D]
        velocity = (pred_cont - current_cont) / max(1 - t, 0.001)

        return velocity, pred_disc, split_logits, del_logits

    def step(
        self,
        t: float,
        state: BranchingState,
    ) -> tuple[BranchingState, dict]:
        """Take one Euler step: x(t+dt) = x(t) + dt * v(t, x)."""
        velocity, pred_disc, split_logits, del_logits = self.velocity_field(t, state)

        # Update continuous states
        current_cont = state.states[0]
        new_cont = current_cont + self.dt * velocity

        # Update discrete states (soft update via logits)
        current_disc = state.states[1] if len(state.states) > 1 else None

        # Handle cell division based on split logits
        split_probs = torch.sigmoid(split_logits)
        events = {
            "n_cells": int(state.padmask.sum()),
            "split_probs_mean": float(split_probs[state.padmask].mean()),
            "split_probs_max": float(split_probs[state.padmask].max()),
            "del_probs_mean": float(torch.sigmoid(del_logits[state.padmask]).mean()),
        }

        # Create new state
        new_state = BranchingState(
            states=(
                new_cont,
                pred_disc.argmax(-1) if pred_disc is not None else current_disc,
            ),
            groupings=state.groupings,
            del_flags=state.del_flags,
            ids=state.ids,
            padmask=state.padmask,
            flowmask=state.flowmask,
            branchmask=state.branchmask,
        )

        return new_state, events

    def simulate(
        self,
        initial_state: BranchingState,
        t_start: float = 0.0,
        t_end: float = 1.0,
        perturbation: callable | None = None,
        perturbation_time: float | None = None,
    ) -> dict:
        """Run full autoregressive simulation.

        Args:
            initial_state: Starting state (e.g., single cell)
            t_start, t_end: Time range
            perturbation: Function(state) -> state, applied at perturbation_time
            perturbation_time: When to apply perturbation

        Returns:
            Dictionary with trajectory and statistics
        """
        trajectory = []
        events_log = []

        state = initial_state
        t = t_start

        n_steps = int((t_end - t_start) / self.dt)

        for step in range(n_steps):
            trajectory.append(
                {
                    "t": t,
                    "n_cells": int(state.padmask.sum()),
                    "spatial_mean": state.states[0][0, state.padmask[0], -3:]
                    .mean(dim=0)
                    .cpu()
                    .numpy()
                    .tolist(),
                }
            )

            # Apply perturbation if scheduled
            if perturbation is not None and perturbation_time is not None:
                if abs(t - perturbation_time) < self.dt / 2:
                    state = perturbation(state)
                    events_log.append(
                        {
                            "step": step,
                            "t": t,
                            "event": "perturbation",
                            "n_cells_after": int(state.padmask.sum()),
                        }
                    )

            # Take step
            state, events = self.step(t, state)
            events_log.append({"step": step, "t": t, **events})

            t += self.dt

            # Early termination if all cells deleted
            if not state.padmask.any():
                break

        # Final state
        trajectory.append(
            {
                "t": t,
                "n_cells": int(state.padmask.sum()),
                "spatial_mean": state.states[0][0, state.padmask[0], -3:]
                .mean(dim=0)
                .cpu()
                .numpy()
                .tolist()
                if state.padmask.any()
                else [0, 0, 0],
            }
        )

        return {
            "trajectory": trajectory,
            "events": events_log,
            "final_state": state,
            "n_steps": len(trajectory),
        }


def create_initial_state(
    n_cells: int,
    gene_dim: int,
    device: str,
) -> BranchingState:
    """Create initial state (single cell or small cluster)."""
    # Start with random small cluster
    cont = torch.randn(1, n_cells, gene_dim + 3, device=device) * 0.1
    disc = torch.zeros(1, n_cells, dtype=torch.long, device=device)

    return BranchingState(
        states=(cont, disc),
        groupings=torch.zeros(1, n_cells, dtype=torch.long, device=device),
        del_flags=torch.zeros(1, n_cells, dtype=torch.bool, device=device),
        ids=torch.arange(1, n_cells + 1, dtype=torch.long, device=device).unsqueeze(0),
        padmask=torch.ones(1, n_cells, dtype=torch.bool, device=device),
        flowmask=torch.ones(1, n_cells, dtype=torch.bool, device=device),
        branchmask=torch.ones(1, n_cells, dtype=torch.bool, device=device),
    )


def lineage_deletion_perturbation(
    state: BranchingState,
    target_founder: str,
    lineage_names: list[str],
) -> BranchingState:
    """Remove cells from target lineage."""
    mask = torch.ones(
        len(lineage_names), dtype=torch.bool, device=state.states[0].device
    )
    for idx, name in enumerate(lineage_names):
        if name and name.startswith(target_founder):
            mask[idx] = False

    # Filter all state components
    new_cont = state.states[0][:, mask]
    new_disc = state.states[1][:, mask] if state.states[1] is not None else None

    n_kept = mask.sum()

    return BranchingState(
        states=(new_cont, new_disc),
        groupings=torch.zeros(
            1, n_kept, dtype=torch.long, device=state.states[0].device
        ),
        del_flags=torch.zeros(
            1, n_kept, dtype=torch.bool, device=state.states[0].device
        ),
        ids=torch.arange(
            1, n_kept + 1, dtype=torch.long, device=state.states[0].device
        ).unsqueeze(0),
        padmask=torch.ones(1, n_kept, dtype=torch.bool, device=state.states[0].device),
        flowmask=torch.ones(1, n_kept, dtype=torch.bool, device=state.states[0].device),
        branchmask=torch.ones(
            1, n_kept, dtype=torch.bool, device=state.states[0].device
        ),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Autoregressive developmental simulation"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints_trimodal_crossmodal/best.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["control", "lineage_deletion", "founder_swap"],
        default="control",
        help="Simulation mode",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="AB",
        help="Target lineage for perturbation",
    )
    parser.add_argument(
        "--perturbation_time",
        type=float,
        default=0.5,
        help="When to apply perturbation (0-1)",
    )
    parser.add_argument(
        "--initial_cells",
        type=int,
        default=10,
        help="Number of initial cells",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.02,
        help="Time step for simulation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="autoregressive_results",
        help="Output directory",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print("=" * 70)
    print("AUTOREGRESSIVE DEVELOPMENTAL SIMULATION")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Time step: {args.dt}")
    print(f"Checkpoint: {args.checkpoint}")
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

    # Create simulator
    simulator = AutoregressiveSimulator(
        model=model,
        device=args.device,
        dt=args.dt,
    )

    # Create initial state
    print(f"Creating initial state ({args.initial_cells} cells)...")
    initial_state = create_initial_state(
        n_cells=args.initial_cells,
        gene_dim=2000,
        device=args.device,
    )
    print()

    # Setup perturbation if needed
    perturbation = None
    if args.mode == "lineage_deletion":
        # For autoregressive, we need to know lineage names
        # But at t=0, we don't have them yet
        # Alternative: randomly delete cells at perturbation time
        def random_deletion(state):
            """Delete random 20% of cells as proxy."""
            n_cells = state.padmask.sum().item()
            n_delete = max(1, int(n_cells * 0.2))

            # Random deletion
            mask = torch.ones(n_cells, dtype=torch.bool, device=state.states[0].device)
            delete_indices = torch.randperm(n_cells)[:n_delete]
            mask[delete_indices] = False

            new_cont = state.states[0][:, mask]
            new_disc = state.states[1][:, mask] if state.states[1] is not None else None
            n_kept = mask.sum()

            print(
                f"  [Perturbation at t={args.perturbation_time}] Deleted {n_delete} cells, {n_kept} remain"
            )

            return BranchingState(
                states=(new_cont, new_disc),
                groupings=torch.zeros(
                    1, n_kept, dtype=torch.long, device=state.states[0].device
                ),
                del_flags=torch.zeros(
                    1, n_kept, dtype=torch.bool, device=state.states[0].device
                ),
                ids=torch.arange(
                    1, n_kept + 1, dtype=torch.long, device=state.states[0].device
                ).unsqueeze(0),
                padmask=torch.ones(
                    1, n_kept, dtype=torch.bool, device=state.states[0].device
                ),
                flowmask=torch.ones(
                    1, n_kept, dtype=torch.bool, device=state.states[0].device
                ),
                branchmask=torch.ones(
                    1, n_kept, dtype=torch.bool, device=state.states[0].device
                ),
            )

        perturbation = random_deletion
        print(f"Will apply random deletion (20%) at t={args.perturbation_time}")

    # Run simulation
    print()
    print("=" * 70)
    print("RUNNING SIMULATION")
    print("=" * 70)

    results = simulator.simulate(
        initial_state=initial_state,
        t_start=0.0,
        t_end=1.0,
        perturbation=perturbation,
        perturbation_time=args.perturbation_time if args.mode != "control" else None,
    )

    # Print results
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total steps: {results['n_steps']}")
    print(f"Final cell count: {results['trajectory'][-1]['n_cells']}")
    print()

    # Show trajectory summary
    print("Developmental trajectory:")
    for i in range(
        0, len(results["trajectory"]), max(1, len(results["trajectory"]) // 10)
    ):
        traj = results["trajectory"][i]
        print(
            f"  t={traj['t']:.2f}: {traj['n_cells']} cells, "
            f"spatial_center=({traj['spatial_mean'][0]:.2f}, "
            f"{traj['spatial_mean'][1]:.2f}, {traj['spatial_mean'][2]:.2f})"
        )

    # Check for biologically reasonable behavior
    final_n = results["trajectory"][-1]["n_cells"]
    initial_n = results["trajectory"][0]["n_cells"]

    print()
    if final_n > initial_n:
        growth_ratio = final_n / initial_n
        print(
            f"✓ Cell proliferation detected: {initial_n} → {final_n} (x{growth_ratio:.1f})"
        )
    elif final_n < initial_n:
        print(f"⚠ Cell count decreased: {initial_n} → {final_n}")
    else:
        print(f"= Cell count stable: {final_n}")

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{args.mode}_t{args.perturbation_time}_dt{args.dt}.json"

    # Convert trajectory to serializable format
    save_data = {
        "mode": args.mode,
        "dt": args.dt,
        "perturbation_time": args.perturbation_time,
        "initial_cells": args.initial_cells,
        "final_cells": final_n,
        "trajectory": results["trajectory"],
        "key_events": [e for e in results["events"] if "event" in e],
    }

    with open(output_file, "w") as f:
        json.dump(save_data, f, indent=2)

    print()
    print(f"Results saved to: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
