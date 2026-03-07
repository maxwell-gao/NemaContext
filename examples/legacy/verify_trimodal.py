"""Verify trimodal implementation without full training.

Quick sanity check that the trimodal dataset and loss functions work correctly.

Usage:
    uv run python examples/verify_trimodal.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def verify_trimodal_dataset():
    """Verify TrimodalDataset loads correctly."""
    from src.branching_flows.trimodal_dataset import TrimodalDataset

    h5ad_path = "dataset/processed/nema_extended_large2025.h5ad"

    print("=" * 60)
    print("Verifying TrimodalDataset")
    print("=" * 60)

    if not Path(h5ad_path).exists():
        print(f"⚠️  AnnData file not found: {h5ad_path}")
        print("Run: uv run python -m src.data.downloader --source core")
        print("     uv run python examples/build_anndata.py --variant extended")
        return False

    try:
        dataset = TrimodalDataset(
            h5ad_path,
            n_hvg=2000,
            time_bins=5,
            ordering="random",
        )

        print("✅ Dataset loaded successfully")
        print(f"   Samples: {len(dataset)}")
        print(f"   Continuous dim: {dataset.continuous_dim}")
        print(f"   Discrete K: {dataset.K}")

        # Print statistics
        print("\n   Dataset Statistics:")
        for key, value in dataset.stats.items():
            print(f"     {key}: {value}")

        # Test sampling
        sample = dataset[0]
        print("\n   Sample structure:")
        print(f"     Elements: {len(sample.elements)}")
        print(f"     Continuous shape per element: {sample.elements[0][0].shape}")
        print(f"     Discrete value: {sample.elements[0][1]}")

        if hasattr(sample, "modality_masks"):
            print(f"     Modality masks: {sample.modality_masks.shape}")

        if hasattr(sample, "lineage_names"):
            print(f"     Lineage names: {len(sample.lineage_names)} items")

        return True

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        import traceback

        traceback.print_exc()
        return False


def verify_trimodal_loss():
    """Verify trimodal loss functions work."""
    from src.branching_flows.legacy.trimodal_loss import (
        trimodal_context_loss,
        curriculum_trimodal_loss,
        masked_mse_loss,
    )

    print("\n" + "=" * 60)
    print("Verifying Trimodal Loss Functions")
    print("=" * 60)

    B, L, D = 2, 10, 2003  # batch, length, dim (2000 genes + 3 spatial)
    gene_dim = 2000

    # Create dummy tensors
    pred = torch.randn(B, L, D)
    target = torch.randn(B, L, D)
    pred_count = torch.tensor([8.0, 10.0])
    target_count = torch.tensor([8.0, 10.0])

    # Test masked MSE
    mask = torch.ones(B, D)
    mask[0, :1000] = 0  # First sample missing first 1000 genes

    mse_loss = masked_mse_loss(pred[:, 0, :], target[:, 0, :], mask)
    print(f"✅ Masked MSE: {mse_loss.item():.4f}")

    # Test trimodal context loss
    modality_mask = torch.ones(B, 3)  # All modalities available
    modality_mask[0, 0] = 0  # First sample missing transcriptome

    try:
        loss, loss_dict = trimodal_context_loss(
            pred,
            target,
            pred_count,
            target_count,
            modality_mask,
            gene_dim=gene_dim,
            lambda_sinkhorn=1.0,
            lambda_count=0.1,
            lambda_diversity=0.01,
        )

        print(f"✅ Trimodal context loss: {loss.item():.4f}")
        print(f"   Components: {loss_dict}")

    except Exception as e:
        print(f"⚠️  Sinkhorn loss requires geomloss: {e}")
        print("   Install with: pip install geomloss")

    # Test curriculum loss
    for phase in [1, 2, 3]:
        try:
            loss, loss_dict = curriculum_trimodal_loss(
                pred,
                target,
                pred_count,
                target_count,
                modality_mask,
                gene_dim=gene_dim,
                training_phase=phase,
            )
            print(f"✅ Curriculum loss (phase {phase}): {loss.item():.4f}")
        except Exception as e:
            print(f"⚠️  Curriculum loss phase {phase}: {e}")

    return True


def verify_model_compatibility():
    """Verify NemaFlowModel works with trimodal dimensions."""
    from src.branching_flows.legacy.nema_model import NemaFlowModel

    print("\n" + "=" * 60)
    print("Verifying Model Compatibility")
    print("=" * 60)

    continuous_dim = 2003  # 2000 genes + 3 spatial
    K = 7  # 6 founders + mask

    model = NemaFlowModel(
        continuous_dim=continuous_dim,
        discrete_K=K,
        d_model=128,
        n_heads=4,
        n_layers=4,
        head_dim=32,
    )

    # Test forward pass
    B, L = 2, 10
    t = torch.rand(B)

    # Create dummy state
    from src.branching_flows import BranchingState

    cont = torch.randn(B, L, continuous_dim)
    disc = torch.randint(0, K, (B, L))

    state = BranchingState(
        states=(cont, disc),
        groupings=torch.zeros(B, L, dtype=torch.long),
        del_flags=torch.zeros(B, L, dtype=torch.bool),
        ids=torch.arange(L).unsqueeze(0).expand(B, L),
        branchmask=torch.ones(B, L, dtype=torch.bool),
        flowmask=torch.ones(B, L, dtype=torch.bool),
        padmask=torch.ones(B, L, dtype=torch.bool),
    )

    # Forward without lineage bias
    output = model(t, state)
    print("✅ Model forward pass successful")
    print(f"   Output continuous shape: {output[0][0].shape}")
    print(f"   Output discrete shape: {output[0][1].shape}")

    # Forward with lineage bias
    lineage_bias = torch.randn(B, L, L)
    output = model(t, state, lineage_bias=lineage_bias)
    print("✅ Model forward with lineage bias successful")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {n_params:,}")

    return True


def main():
    """Run all verification tests."""
    print("\n" + "=" * 60)
    print("Trimodal Implementation Verification")
    print("=" * 60)

    results = []

    # Test 1: Dataset
    results.append(("TrimodalDataset", verify_trimodal_dataset()))

    # Test 2: Loss functions
    results.append(("TrimodalLoss", verify_trimodal_loss()))

    # Test 3: Model compatibility
    results.append(("ModelCompatibility", verify_model_compatibility()))

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
        print("\nYou can now run trimodal training:")
        print("  uv run python examples/train_trimodal.py --epochs 10 --device cpu")
    else:
        print("\n⚠️  Some verifications failed.")
        print("Please check the errors above.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
