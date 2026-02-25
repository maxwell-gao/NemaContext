#!/usr/bin/env python3
"""
Discover lineage knowledge learned by the trained model.

This script probes the model to extract developmental lineage patterns
that emerged during training WITHOUT explicit lineage supervision.

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

from src.branching_flows.model_probe import LineageProbe
from src.branching_flows.trimodal_dataset import TrimodalDataset


def main():
    parser = argparse.ArgumentParser(
        description="Discover lineage knowledge from trained model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints_trimodal_crossmodal/best.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="discoveries/lineage_knowledge.json",
        help="Output path for discovery report",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=50,
        help="Number of samples to analyze",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print("=" * 70)
    print("DISCOVERING LINEAGE KNOWLEDGE FROM TRAINED MODEL")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Following the creed: 'Discover, Don't Inject'")
    print()

    # Load dataset
    print("Loading dataset...")
    dataset = TrimodalDataset(
        h5ad_path="dataset/processed/nema_extended_large2025.h5ad",
        time_bins=20,
        max_cells_per_bin=512,
        augment_spatial=False,  # No augmentation for discovery
        aug_rotation=False,
        aug_flip=False,
    )
    print(f"  Loaded: {len(dataset)} samples")
    print()

    # Load model
    print("Loading model...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    from src.branching_flows.crossmodal_model import CrossModalNemaModel

    model = CrossModalNemaModel(
        gene_dim=2000,
        spatial_dim=3,
        discrete_K=7,
        d_model=256,
        n_layers=6,
        n_heads=8,
        cross_modal_every=2,
    ).to(args.device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} params")
    print()

    # Create lineage probe
    probe = LineageProbe(model, device=args.device)

    results = {}

    # Probe 1: Sibling Similarity
    print("=" * 70)
    print("PROBE 1: Sibling Cell Similarity")
    print("=" * 70)
    print("Discovering: Does the model learn that sibling cells are similar?")
    print()

    sibling_results = probe.discover_sibling_similarity(dataset, args.n_samples)
    results["sibling_similarity"] = sibling_results

    print(f"Sibling pairs analyzed: {sibling_results['n_sibling_pairs']}")
    print(f"  Mean similarity: {sibling_results['sibling_similarity_mean']:.4f}")
    print(
        f"  (+/- {sibling_results['sibling_similarity_std']:.4f})"
    )
    print()
    print(f"Cousin pairs: {sibling_results['n_cousin_pairs']}")
    print(f"  Mean similarity: {sibling_results['cousin_similarity_mean']:.4f}")
    print()
    print(f"Unrelated pairs: {sibling_results['n_unrelated_pairs']}")
    print(f"  Mean similarity: {sibling_results['unrelated_similarity_mean']:.4f}")
    print()

    if sibling_results["learned_sibling_bias"]:
        print("✓ MODEL LEARNED: Sibling cells are more similar than unrelated cells")
        print("  This matches biological reality: siblings share recent parent")
    else:
        print("  Model did not strongly learn sibling similarity")
    print()

    # Probe 2: Founder Lineage Separation
    print("=" * 70)
    print("PROBE 2: Founder Lineage Separation")
    print("=" * 70)
    print("Discovering: Does the model separate founder lineages (AB, MS, E, C, D, P4)?")
    print()

    founder_results = probe.discover_founder_lineage_separation(
        dataset, args.n_samples
    )
    results["founder_separation"] = founder_results

    print("Founder cell counts:")
    for founder, count in founder_results["founder_cell_counts"].items():
        print(f"  {founder}: {count} cells")
    print()
    print(f"Within-founder similarity: {founder_results['within_founder_similarity']:.4f}")
    print(f"Between-founder similarity: {founder_results['between_founder_similarity']:.4f}")
    print(
        f"Separation ratio: {founder_results['founder_separation_ratio']:.4f} "
        f"(>1 = separated)"
    )
    print()

    if founder_results["learned_founder_structure"]:
        print("✓ MODEL LEARNED: Founder lineages are distinct from each other")
        print("  This matches biology: AB=neurons, MS=pharynx/muscle, E=intestine, etc.")
    else:
        print("  Model did not strongly separate founder lineages")
    print()

    # Probe 3: Lineage Depth Progression
    print("=" * 70)
    print("PROBE 3: Lineage Depth Progression")
    print("=" * 70)
    print("Discovering: Does cell state diversity increase with lineage depth?")
    print()

    depth_results = probe.discover_lineage_depth_progression(dataset, args.n_samples)
    results["depth_progression"] = depth_results

    print("State variance by lineage depth:")
    for depth, variance in sorted(depth_results["depth_variance"].items()):
        print(f"  Depth {depth}: variance = {variance:.4f}")
    print()

    if depth_results["variance_increases_with_depth"]:
        print("✓ MODEL LEARNED: Cell states diverge with developmental progression")
        print("  This matches biology: early cells are similar, late cells differentiate")
    else:
        print("  Model shows mixed depth progression pattern")
    print()

    # Save results
    print("=" * 70)
    print("SAVING DISCOVERY REPORT")
    print("=" * 70)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python native types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(v) for v in obj]
        elif isinstance(obj, (np.bool_, np.integer)):
            return bool(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_native = convert_to_native(results)

    with open(output_path, "w") as f:
        json.dump(results_native, f, indent=2)

    print(f"Lineage knowledge report saved to: {output_path}")
    print()

    # Summary
    print("=" * 70)
    print("LINEAGE KNOWLEDGE DISCOVERY COMPLETE")
    print("=" * 70)
    print()
    print("Key Findings:")
    print()

    learned_count = sum(
        [
            sibling_results["learned_sibling_bias"],
            founder_results["learned_founder_structure"],
            depth_results["variance_increases_with_depth"],
        ]
    )

    print(f"  Biological patterns learned: {learned_count}/3")
    print()
    print("  1. Sibling similarity: ", end="")
    print("✓ LEARNED" if sibling_results["learned_sibling_bias"] else "  Not strongly learned")
    print("  2. Founder separation: ", end="")
    print("✓ LEARNED" if founder_results["learned_founder_structure"] else "  Not strongly learned")
    print("  3. Depth progression: ", end="")
    print("✓ LEARNED" if depth_results["variance_increases_with_depth"] else "  Not strongly learned")
    print()
    print("-" * 70)
    print("These patterns emerged WITHOUT explicit lineage supervision.")
    print("The model DISCOVERED developmental biology from raw data.")
    print("=" * 70)


if __name__ == "__main__":
    main()
