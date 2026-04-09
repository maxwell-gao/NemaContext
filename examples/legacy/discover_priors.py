"""Discover biological priors from trained cross-modal model.

Usage:
    uv run python examples/discover_priors.py \
        --checkpoint checkpoints/trimodal_crossmodal/best.pt \
        --output discoveries/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.branching_flows.legacy.crossmodal_model import CrossModalNemaModel
from src.branching_flows.model_probe import (
    CrossModalProbe,
    LatentSpaceExplorer,
    LineageProbe,
    save_discovery_report,
)
from src.branching_flows.trimodal_dataset import TrimodalDataset


def parse_args():
    p = argparse.ArgumentParser(
        description="Discover biological priors from trained model"
    )
    p.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    p.add_argument(
        "--h5ad_path", default="dataset/processed/nema_extended_large2025.h5ad"
    )
    p.add_argument(
        "--output", default="discoveries", help="Output directory for discoveries"
    )
    p.add_argument(
        "--n_samples", type=int, default=50, help="Number of samples to analyze"
    )
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 70)
    print("DISCOVERING BIOLOGICAL PRIORS FROM TRAINED MODEL")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print("Following the creed: 'Discover, Don't Inject'")
    print()

    # Load dataset
    print("Loading dataset...")
    dataset = TrimodalDataset(
        args.h5ad_path,
        n_hvg=2000,
        time_bins=10,
        ordering="random",
        max_cells_per_bin=256,
        augment_spatial=False,  # No augmentation for analysis
    )
    print(f"  Loaded: {len(dataset)} samples")

    # Load model
    print("\nLoading model...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_args = checkpoint.get("args", {})

    model = CrossModalNemaModel(
        gene_dim=dataset._gene_dim,
        spatial_dim=dataset._spatial_dim,
        discrete_K=dataset.K,
        d_model=model_args.get("d_model", 256),
        n_heads=model_args.get("n_heads", 8),
        n_layers=model_args.get("n_layers", 6),
        head_dim=model_args.get("head_dim", 32),
        cross_modal_every=model_args.get("cross_modal_every", 2),
    ).to(device)

    model.load_state_dict(checkpoint["model"])
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"  Cross-modal fusion every {model_args.get('cross_modal_every', 2)} layers")

    # Create probes
    print("\n" + "=" * 70)
    print("PROBE 1: Cross-Modal Gene-Spatial Relationships")
    print("=" * 70)
    print("Discovering: Which genes does the model use to predict position?")
    print()

    cross_modal_probe = CrossModalProbe(model, device=device)
    correlation_results = cross_modal_probe.analyze_gene_spatial_correlation(
        dataset, n_samples=args.n_samples
    )

    print(
        f"Top genes predictive of X position (indices): {correlation_results['top_genes_for_x'][:10]}"
    )
    print(
        f"Top genes predictive of Y position (indices): {correlation_results['top_genes_for_y'][:10]}"
    )
    print(
        f"Top genes predictive of Z position (indices): {correlation_results['top_genes_for_z'][:10]}"
    )

    print("\n" + "=" * 70)
    print("PROBE 2: Cell Type Discovery in Latent Space")
    print("=" * 70)
    print("Discovering: What cell types naturally emerge from the model?")
    print()

    cell_type_results = cross_modal_probe.discover_cell_type_markers(
        dataset, n_clusters=10
    )

    print(f"Discovered {len(cell_type_results['markers'])} cell clusters:")
    for cluster_name, info in cell_type_results["markers"].items():
        print(f"  {cluster_name}: {info['size']} cells")
        print(f"    Top marker gene indices: {info['top_markers'][:5]}")

    print("\n" + "=" * 70)
    print("PROBE 3: Developmental Trajectory Manifold")
    print("=" * 70)
    print("Discovering: What are the continuous paths of development?")
    print()

    explorer = LatentSpaceExplorer(model, device=device)
    manifold_results = explorer.discover_trajectory_manifold(
        dataset, n_samples=args.n_samples
    )

    print(f"PCA explained variance: {manifold_results['explained_variance']}")
    print(f"Total cells in manifold: {len(manifold_results['time_labels'])}")

    print("\n" + "=" * 70)
    print("PROBE 4: Lineage Relationship Patterns")
    print("=" * 70)
    print("Discovering: How does the model represent developmental lineage?")
    print()

    # Get some lineage names from dataset
    sample = dataset[0]
    if hasattr(sample, "lineage_names") and sample.lineage_names:
        lineage_names = [name for name in sample.lineage_names[:20] if name]
        if lineage_names:
            lineage_probe = LineageProbe(model)
            lineage_sim = lineage_probe.extract_lineage_attention_patterns(
                lineage_names
            )
            print(f"Analyzed {len(lineage_names)} cells with lineage information")
            print(f"Lineage similarity matrix shape: {lineage_sim.shape}")
        else:
            lineage_sim = None
            print("No lineage names available in sample")
    else:
        lineage_sim = None
        print("Sample does not have lineage_names attribute")

    # Compile discoveries
    print("\n" + "=" * 70)
    print("COMPILING DISCOVERY REPORT")
    print("=" * 70)

    discoveries = {
        "model_checkpoint": args.checkpoint,
        "model_params": sum(p.numel() for p in model.parameters()),
        "dataset_stats": dataset.stats,
        "gene_spatial_correlation": {
            "top_genes_x": correlation_results["top_genes_for_x"][:20].tolist(),
            "top_genes_y": correlation_results["top_genes_for_y"][:20].tolist(),
            "top_genes_z": correlation_results["top_genes_for_z"][:20].tolist(),
        },
        "discovered_cell_types": cell_type_results["markers"],
        "trajectory_manifold": {
            "explained_variance": manifold_results["explained_variance"],
            "n_cells_analyzed": len(manifold_results["time_labels"]),
        },
    }

    if lineage_sim is not None:
        discoveries["lineage_similarity_sample"] = lineage_sim[:5, :5].tolist()

    # Save report
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "discovery_report.json"
    save_discovery_report(discoveries, report_path)

    # Save visualizable data
    print("\nSaving visualizable data...")

    # Save PCA coordinates for plotting
    pca_data = {
        "coords": manifold_results["pca_coords"],
        "time_labels": manifold_results["time_labels"],
    }
    with open(output_dir / "trajectory_pca.json", "w") as f:
        json.dump(pca_data, f)
    print(f"  PCA trajectory data: {output_dir / 'trajectory_pca.json'}")

    # Save marker genes per cluster
    with open(output_dir / "cell_type_markers.json", "w") as f:
        json.dump(cell_type_results["markers"], f, indent=2)
    print(f"  Cell type markers: {output_dir / 'cell_type_markers.json'}")

    print("\n" + "=" * 70)
    print("DISCOVERY COMPLETE")
    print("=" * 70)
    print(f"\nAll discoveries saved to: {output_dir}/")
    print("\nKey Findings:")
    print(
        f"  - Identified {len(cell_type_results['markers'])} cell clusters from latent space"
    )
    print("  - Ranked genes by spatial predictive power")
    print("  - Mapped developmental trajectory manifold")
    print("\nThese are DATA-DRIVEN discoveries, not injected priors.")
    print("The model learned these patterns from raw observations.")


if __name__ == "__main__":
    main()
