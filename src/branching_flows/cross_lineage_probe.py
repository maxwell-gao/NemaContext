"""Cross-lineage probing tools for whole-embryo context analysis.

Discovers biological priors related to cross-lineage interactions
from trained models with whole-embryo context.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


class CrossLineageProbe:
    """Probe cross-lineage interactions learned by the model.

    Analyzes how cells from different founder lineages (AB, MS, E, C, D, P4)
    influence each other in the model's predictions.
    """

    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device
        self.model.eval()

        self.founder_map = {"P0": 0, "AB": 1, "MS": 2, "E": 3, "C": 4, "D": 5, "P4": 6}
        self.founder_names = {v: k for k, v in self.founder_map.items()}

    def measure_cross_lineage_influence(
        self,
        state: Any,  # BranchingState
        target_founder: str,
    ) -> dict[str, Any]:
        """Measure how much other lineages influence predictions for target founder.

        Method:
        1. Full forward pass (all cells)
        2. Mask out each non-target founder one at a time
        3. Compare prediction differences

        Returns:
            Dictionary with influence scores per founder
        """
        target_id = self.founder_map.get(target_founder, 1)

        # Find target founder cells
        founder_ids = state.states[1][0]
        target_mask = founder_ids == target_id

        if not target_mask.any():
            return {"error": f"No cells of founder {target_founder} found"}

        # Full prediction
        with torch.no_grad():
            output_full = self.model.forward_step(state.to(self.device))

        # Measure influence of each other founder
        influences = {}

        for founder_name, founder_id in self.founder_map.items():
            if founder_name == target_founder:
                continue

            # Check if this founder exists in the state
            if not (founder_ids == founder_id).any():
                continue

            # Create masked state (remove this founder)
            keep_mask = founder_ids != founder_id
            new_cont = state.states[0][:, keep_mask, :]
            new_disc = state.states[1][:, keep_mask]
            n_kept = keep_mask.sum().item()

            from src.branching_flows.states import BranchingState

            masked_state = BranchingState(
                states=(new_cont, new_disc),
                groupings=torch.zeros(1, n_kept, dtype=torch.long, device=self.device),
                del_flags=torch.zeros(1, n_kept, dtype=torch.bool, device=self.device),
                ids=torch.arange(
                    1, n_kept + 1, dtype=torch.long, device=self.device
                ).unsqueeze(0),
                padmask=torch.ones(1, n_kept, dtype=torch.bool, device=self.device),
                flowmask=torch.ones(1, n_kept, dtype=torch.bool, device=self.device),
                branchmask=torch.ones(1, n_kept, dtype=torch.bool, device=self.device),
            )

            # Masked prediction
            with torch.no_grad():
                output_masked = self.model.forward_step(masked_state)

            # Find where target cells are in the masked output
            # (they shifted due to removal, need to track indices)
            target_indices_full = torch.where(target_mask)[0]
            target_indices_masked = []
            for idx in target_indices_full:
                # Count how many kept cells were before this one
                n_before = keep_mask[:idx].sum().item()
                if keep_mask[idx]:
                    target_indices_masked.append(n_before)

            if not target_indices_masked:
                continue

            # Compare predictions for target cells
            target_indices_masked_tensor = torch.tensor(
                target_indices_masked, device=self.device
            )
            full_pred = output_full.gene_delta[0, target_mask, :]
            masked_pred = output_masked.gene_delta[0, target_indices_masked_tensor, :]

            diff = torch.norm(full_pred - masked_pred, dim=-1).mean().item()

            influences[founder_name] = {
                "delta_diff": diff,
                "n_target_cells": len(target_indices_masked),
                "influences_prediction": diff > 0.01,
            }

        return {
            "target_founder": target_founder,
            "n_target_cells": target_mask.sum().item(),
            "influences": influences,
            "total_cross_lineage_influence": sum(
                inf["delta_diff"] for inf in influences.values()
            ),
        }

    def analyze_founder_specific_markers(
        self,
        dataset: Any,
        n_samples: int = 50,
    ) -> dict[str, Any]:
        """Analyze which features are specific to each founder lineage.

        Discovers: "What makes AB cells different from MS cells?"
        """
        founder_features: dict[str, list[np.ndarray]] = {
            name: [] for name in self.founder_map.keys()
        }

        with torch.no_grad():
            for i in range(min(n_samples, len(dataset))):
                sample = dataset[i]
                current = sample["current"]

                # Get latent representations
                h = self.model.encode_state(current.to(self.device))

                # Group by founder
                founder_ids = current.states[1][0]
                for founder_name, founder_id in self.founder_map.items():
                    mask = founder_ids == founder_id
                    if mask.any():
                        founder_features[founder_name].append(
                            h[0, mask, :].cpu().numpy()
                        )

        # Compute statistics per founder
        founder_stats = {}
        for founder_name, features_list in founder_features.items():
            if not features_list:
                continue

            all_features = np.concatenate(features_list, axis=0)

            founder_stats[founder_name] = {
                "n_cells": len(all_features),
                "mean_activation": np.mean(all_features, axis=0).tolist(),
                "activation_variance": np.var(all_features, axis=0).tolist(),
            }

        # Compute cross-founder distances
        cross_distances = {}
        founders_with_stats = list(founder_stats.keys())

        for i, f1 in enumerate(founders_with_stats):
            for f2 in founders_with_stats[i + 1 :]:
                mean1 = np.array(founder_stats[f1]["mean_activation"])
                mean2 = np.array(founder_stats[f2]["mean_activation"])

                distance = np.linalg.norm(mean1 - mean2)
                cross_distances[f"{f1}-{f2}"] = float(distance)

        return {
            "founder_stats": founder_stats,
            "cross_founder_distances": cross_distances,
            "founder_separation": np.mean(list(cross_distances.values()))
            if cross_distances
            else 0.0,
        }

    def detect_lineage_boundaries(
        self,
        dataset: Any,
        n_samples: int = 30,
    ) -> dict[str, Any]:
        """Detect physical boundaries between different lineages.

        Analyzes where in 3D space different lineages meet.
        """
        boundary_contacts = []

        for i in range(min(n_samples, len(dataset))):
            sample = dataset[i]
            current = sample["current"]

            positions = current.states[0][0, :, -3:].cpu().numpy()
            founder_ids = current.states[1][0].cpu().numpy()

            # Compute pairwise distances
            for j in range(len(positions)):
                for k in range(j + 1, len(positions)):
                    if founder_ids[j] != founder_ids[k]:
                        dist = np.linalg.norm(positions[j] - positions[k])
                        if dist < 0.1:  # Threshold for "touching"
                            boundary_contacts.append(
                                {
                                    "founder1": self.founder_names.get(
                                        int(founder_ids[j]), "UNKNOWN"
                                    ),
                                    "founder2": self.founder_names.get(
                                        int(founder_ids[k]), "UNKNOWN"
                                    ),
                                    "distance": float(dist),
                                    "position": (
                                        (positions[j] + positions[k]) / 2
                                    ).tolist(),
                                }
                            )

        # Group by founder pair
        contacts_by_pair: dict[str, list] = {}
        for contact in boundary_contacts:
            pair = tuple(sorted([contact["founder1"], contact["founder2"]]))
            pair_key = f"{pair[0]}-{pair[1]}"
            if pair_key not in contacts_by_pair:
                contacts_by_pair[pair_key] = []
            contacts_by_pair[pair_key].append(contact)

        return {
            "total_boundary_contacts": len(boundary_contacts),
            "contacts_by_pair": {
                pair: len(contacts) for pair, contacts in contacts_by_pair.items()
            },
            "common_boundaries": sorted(
                contacts_by_pair.keys(),
                key=lambda p: len(contacts_by_pair[p]),
                reverse=True,
            )[:5],
        }

    def generate_discovery_report(
        self,
        dataset: Any,
        output_path: str | Path,
        n_samples: int = 50,
    ):
        """Generate comprehensive cross-lineage discovery report."""
        print("Generating cross-lineage discovery report...")

        report = {
            "cross_lineage_influences": {},
            "founder_markers": self.analyze_founder_specific_markers(
                dataset, n_samples
            ),
            "lineage_boundaries": self.detect_lineage_boundaries(dataset, n_samples),
        }

        # Measure influences for each founder
        for founder_name in ["AB", "MS", "E", "C", "D"]:
            # Find a sample with this founder present
            for i in range(min(10, len(dataset))):
                sample = dataset[i]
                current = sample["current"]
                founder_ids = current.states[1][0]

                if (founder_ids == self.founder_map.get(founder_name, -1)).any():
                    influence = self.measure_cross_lineage_influence(
                        current, founder_name
                    )
                    if "error" not in influence:
                        report["cross_lineage_influences"][founder_name] = influence
                        break

        # Save report
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Report saved to: {output_path}")
        return report


def run_cross_lineage_discovery(
    checkpoint_path: str,
    trajectory_file: str,
    output_dir: str = "discoveries/cross_lineage",
    device: str = "cpu",
):
    """Run full cross-lineage discovery pipeline."""
    from src.branching_flows.autoregressive_model import AutoregressiveNemaModel
    from examples.whole_organism_ar.train_autoregressive_full import EmbryoTrajectoryDataset

    print("=" * 70)
    print("CROSS-LINEAGE DISCOVERY PIPELINE")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    model = AutoregressiveNemaModel(
        gene_dim=2000,
        spatial_dim=3,
        discrete_K=7,
        d_model=256,
        n_layers=6,
        n_heads=8,
        max_seq_len=128,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Loaded from epoch {checkpoint.get('epoch', 'unknown')}")

    # Load dataset
    print("\nLoading dataset...")
    dataset = EmbryoTrajectoryDataset(trajectory_file)

    # Run discovery
    print("\nRunning discovery analyses...")
    probe = CrossLineageProbe(model, device)

    output_path = Path(output_dir) / "cross_lineage_report.json"
    report = probe.generate_discovery_report(dataset, output_path, n_samples=50)

    # Print summary
    print("\n" + "=" * 70)
    print("DISCOVERY SUMMARY")
    print("=" * 70)

    influences = report.get("cross_lineage_influences", {})
    if influences:
        print("\nCross-lineage influences detected:")
        for founder, data in influences.items():
            total_influence = data.get("total_cross_lineage_influence", 0)
            print(f"  {founder}: total_influence={total_influence:.4f}")

    boundaries = report.get("lineage_boundaries", {})
    if boundaries:
        print("\nLineage boundary contacts:")
        for pair, count in list(boundaries.get("contacts_by_pair", {}).items())[:5]:
            print(f"  {pair}: {count} contacts")

    markers = report.get("founder_markers", {})
    if markers:
        sep = markers.get("founder_separation", 0)
        print(f"\nFounder separation score: {sep:.4f}")

    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cross-lineage discovery")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--trajectory_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="discoveries/cross_lineage")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    run_cross_lineage_discovery(
        args.checkpoint,
        args.trajectory_file,
        args.output_dir,
        args.device,
    )
