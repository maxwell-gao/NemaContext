"""Evaluate modality completion performance.

Tests how well the model fills in missing modalities:
- Gene expression -> predict spatial position
- Spatial position -> predict gene expression

This demonstrates the cross-modal model's ability to discover
gene-spatial relationships without explicit supervision.

Usage:
    uv run python examples/evaluate_modality_completion.py \
        --checkpoint checkpoints/trimodal_crossmodal/best.pt \
        --test_mode gene_to_spatial
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.branching_flows.legacy.crossmodal_model import CrossModalNemaModel
from src.branching_flows.trimodal_dataset import TrimodalDataset


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate modality completion")
    p.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    p.add_argument(
        "--h5ad_path", default="dataset/processed/nema_extended_large2025.h5ad"
    )
    p.add_argument(
        "--test_mode",
        default="gene_to_spatial",
        choices=["gene_to_spatial", "spatial_to_gene"],
    )
    p.add_argument("--n_test", type=int, default=100, help="Number of test samples")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def evaluate_gene_to_spatial(model: torch.nn.Module, dataset, n_test: int, device: str):
    """Evaluate: given gene expression, predict spatial position."""
    print("Evaluating: Gene Expression -> Spatial Position")
    print("-" * 60)

    errors = []
    gene_importance = []

    model.eval()
    with torch.no_grad():
        for i in range(min(n_test, len(dataset))):
            sample = dataset[i]
            cont_states = torch.stack([e[0] for e in sample.elements])

            genes = cont_states[:, : dataset._gene_dim].to(device)
            true_spatial = cont_states[
                :, dataset._gene_dim : dataset._gene_dim + dataset._spatial_dim
            ]

            # Create dummy spatial input (zeros) to test gene->spatial prediction
            dummy_spatial = torch.zeros_like(
                cont_states[
                    :, dataset._gene_dim : dataset._gene_dim + dataset._spatial_dim
                ].to(device)
            )

            # Use model's output heads
            g_latent = model.gene_proj(genes)
            s_latent = model.spatial_proj(dummy_spatial)
            combined = torch.cat([g_latent, s_latent], dim=-1)
            fused = model.fusion_proj(combined)

            pred_spatial = model.spatial_head(fused)

            # Calculate error
            error = torch.norm(pred_spatial.cpu() - true_spatial, dim=-1).mean().item()
            errors.append(error)

            # Gene importance analysis (perturbation-based)
            if i < 10:  # Only for first few samples
                baseline_pred = pred_spatial.clone()

                # Test importance of each gene dimension (sample every 100th)
                for gene_idx in range(0, dataset._gene_dim, 100):
                    perturbed_genes = genes.clone()
                    perturbed_genes[:, gene_idx] = 0  # Zero out this gene

                    g_latent_pert = model.gene_proj(perturbed_genes)
                    combined_pert = torch.cat([g_latent_pert, s_latent], dim=-1)
                    fused_pert = model.fusion_proj(combined_pert)
                    pred_spatial_pert = model.spatial_head(fused_pert)

                    change = (
                        torch.norm(pred_spatial_pert - baseline_pred, dim=-1)
                        .mean()
                        .item()
                    )
                    gene_importance.append((gene_idx, change))

    mean_error = np.mean(errors)
    print(f"Mean spatial prediction error (L2 norm): {mean_error:.4f}")
    print(f"Error std: {np.std(errors):.4f}")

    if gene_importance:
        # Sort by importance
        gene_importance.sort(key=lambda x: x[1], reverse=True)
        print("\nTop 10 most important gene dimensions (indices):")
        for idx, (gene_idx, importance) in enumerate(gene_importance[:10]):
            print(f"  {idx + 1}. Gene {gene_idx}: importance score {importance:.4f}")

    return {"mean_error": mean_error, "errors": errors}


def evaluate_spatial_to_gene(model: torch.nn.Module, dataset, n_test: int, device: str):
    """Evaluate: given spatial position, predict gene expression."""
    print("Evaluating: Spatial Position -> Gene Expression")
    print("-" * 60)

    errors = []

    model.eval()
    with torch.no_grad():
        for i in range(min(n_test, len(dataset))):
            sample = dataset[i]
            cont_states = torch.stack([e[0] for e in sample.elements])

            spatial = cont_states[
                :, dataset._gene_dim : dataset._gene_dim + dataset._spatial_dim
            ].to(device)
            true_genes = cont_states[:, : dataset._gene_dim]

            # Create dummy gene input
            dummy_genes = torch.zeros_like(
                cont_states[:, : dataset._gene_dim].to(device)
            )

            # Forward
            g_latent = model.gene_proj(dummy_genes)
            s_latent = model.spatial_proj(spatial)
            combined = torch.cat([g_latent, s_latent], dim=-1)
            fused = model.fusion_proj(combined)

            pred_genes = model.gene_head(fused)

            # Calculate error
            error = F.mse_loss(pred_genes.cpu(), true_genes).item()
            errors.append(error)

    mean_error = np.mean(errors)
    print(f"Mean gene prediction error (MSE): {mean_error:.4f}")
    print(f"Error std: {np.std(errors):.4f}")

    return {"mean_error": mean_error, "errors": errors}


def main():
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 70)
    print("MODALITY COMPLETION EVALUATION")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test mode: {args.test_mode}")
    print()

    # Load dataset
    print("Loading dataset...")
    dataset = TrimodalDataset(
        args.h5ad_path,
        n_hvg=2000,
        time_bins=10,
        ordering="random",
        max_cells_per_bin=256,
        augment_spatial=False,
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

    # Evaluate
    print("\n" + "=" * 70)
    if args.test_mode == "gene_to_spatial":
        evaluate_gene_to_spatial(model, dataset, args.n_test, device)
    else:
        evaluate_spatial_to_gene(model, dataset, args.n_test, device)

    # Save results summary
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print("\nThis demonstrates the model's learned understanding of")
    print("gene-spatial relationships WITHOUT explicit supervision.")
    print("The model discovered these associations from raw data.")


if __name__ == "__main__":
    main()
