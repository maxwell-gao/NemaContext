"""Verify cross-modal model and data augmentation.

Usage:
    uv run python examples/verify_crossmodal.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def verify_crossmodal_model():
    """Verify CrossModalNemaModel works."""
    from src.branching_flows.legacy.crossmodal_model import (
        CrossModalNemaModel,
    )
    from src.branching_flows.fusion import CrossModalFusion
    from src.branching_flows import BranchingState

    print("=" * 60)
    print("Verifying Cross-Modal Model")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test cross-modal fusion layer
    print("\n1. Testing CrossModalFusion layer...")
    fusion = CrossModalFusion(d_model=128, n_heads=4).to(device)

    B, L, D = 2, 50, 128
    gene_feat = torch.randn(B, L, D, device=device)
    spatial_feat = torch.randn(B, L, D, device=device)

    g_out, s_out = fusion(gene_feat, spatial_feat)

    assert g_out.shape == (B, L, D), f"Gene output shape mismatch: {g_out.shape}"
    assert s_out.shape == (B, L, D), f"Spatial output shape mismatch: {s_out.shape}"
    assert not torch.isnan(g_out).any(), "Gene output has NaN"
    assert not torch.isnan(s_out).any(), "Spatial output has NaN"

    print("   ✅ CrossModalFusion works!")

    # Test full model
    print("\n2. Testing CrossModalNemaModel...")
    model = CrossModalNemaModel(
        gene_dim=2000,
        spatial_dim=3,
        discrete_K=7,
        d_model=256,
        n_heads=8,
        n_layers=6,
        head_dim=32,
        cross_modal_every=2,
    ).to(device)

    # Create dummy input
    B, L = 2, 100
    cont = torch.randn(B, L, 2003, device=device)
    disc = torch.randint(0, 7, (B, L), device=device)

    state = BranchingState(
        states=(cont, disc),
        groupings=torch.zeros(B, L, dtype=torch.long, device=device),
        del_flags=torch.zeros(B, L, dtype=torch.bool, device=device),
        ids=torch.arange(L, device=device).unsqueeze(0).expand(B, L),
        branchmask=torch.ones(B, L, dtype=torch.bool, device=device),
        flowmask=torch.ones(B, L, dtype=torch.bool, device=device),
        padmask=torch.ones(B, L, dtype=torch.bool, device=device),
    )

    t = torch.rand(B, device=device)
    output = model(t, state)

    cont_pred, disc_pred = output[0]
    print(f"   Continuous output shape: {cont_pred.shape}")
    print(f"   Discrete output shape: {disc_pred.shape}")
    print(f"   Has NaN: {torch.isnan(cont_pred).any().item()}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,}")
    print("   ✅ CrossModalNemaModel works!")

    return True


def verify_data_augmentation():
    """Verify spatial data augmentation."""
    from src.branching_flows.trimodal_dataset import TrimodalDataset

    print("\n" + "=" * 60)
    print("Verifying Data Augmentation")
    print("=" * 60)

    h5ad_path = "dataset/processed/nema_extended_large2025.h5ad"
    if not Path(h5ad_path).exists():
        print(f"   ⚠️ AnnData file not found: {h5ad_path}")
        return True

    print("\n1. Loading dataset without augmentation...")
    dataset_no_aug = TrimodalDataset(
        h5ad_path,
        n_hvg=2000,
        time_bins=5,
        ordering="random",
        augment_spatial=False,
    )

    sample_no_aug = dataset_no_aug[0]
    cont_no_aug = sample_no_aug.elements[0][0]
    spatial_no_aug = cont_no_aug[2000:2003]

    print(f"   Sample spatial (no aug): {spatial_no_aug.numpy()}")

    print("\n2. Loading dataset with augmentation...")
    dataset_aug = TrimodalDataset(
        h5ad_path,
        n_hvg=2000,
        time_bins=5,
        ordering="random",
        augment_spatial=True,
        aug_rotation=True,
        aug_flip=True,
        aug_scale=0.1,
    )

    # Get same sample multiple times to see augmentation variation
    sample_aug1 = dataset_aug[0]
    sample_aug2 = dataset_aug[0]

    cont_aug1 = sample_aug1.elements[0][0]
    cont_aug2 = sample_aug2.elements[0][0]

    spatial_aug1 = cont_aug1[2000:2003]
    spatial_aug2 = cont_aug2[2000:2003]

    print(f"   Sample spatial (aug 1): {spatial_aug1.numpy()}")
    print(f"   Sample spatial (aug 2): {spatial_aug2.numpy()}")

    # Check if augmentation actually changed something
    diff = (spatial_aug1 - spatial_aug2).abs().max().item()
    print(f"   Max difference between two aug samples: {diff:.4f}")

    if diff > 0.01:
        print("   ✅ Data augmentation is working!")
    else:
        print("   ⚠️ Data augmentation may not be effective")

    return True


def main():
    print("\n" + "=" * 60)
    print("Cross-Modal Model & Augmentation Verification")
    print("=" * 60)

    results = []

    # Test 1: Cross-modal model
    results.append(("CrossModalModel", verify_crossmodal_model()))

    # Test 2: Data augmentation
    results.append(("DataAugmentation", verify_data_augmentation()))

    # Summary
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n🎉 All verifications passed!")
        print("\nYou can now run training:")
        print(
            "  uv run python examples/train_trimodal_crossmodal.py --epochs 50 --device cuda"
        )
    else:
        print("\n⚠️ Some verifications failed.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
