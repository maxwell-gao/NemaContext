"""Focused tests for active embryo/local-code model paths."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.branching_flows.gene_context import (  # noqa: E402
    EmbryoFutureSetModel,
    EmbryoMaskedViewModel,
    EmbryoStateModel,
    FrozenLinearTokenReadout,
    LocalCellCodeModel,
    PooledLatentCanonicalizer,
)
from src.data.gene_context_dataset import (  # noqa: E402
    EmbryoViewDataset,
    PatchSetDataset,
    collate_embryo_view,
    collate_patch_set,
)


def test_embryo_view_dataset_and_model_forward():
    """Embryo-state path should collate multi-view local observations and predict embryo probes."""
    h5ad_path = Path("dataset/processed/nema_extended_large2025.h5ad")
    if not h5ad_path.exists():
        pytest.skip("Processed gene-context dataset not available")

    dataset = EmbryoViewDataset(
        h5ad_path=h5ad_path,
        n_hvg=32,
        context_size=8,
        global_context_size=2,
        dt_minutes=40.0,
        samples_per_pair=2,
        split="train",
        sampling_strategy="spatial_anchor",
        random_seed=0,
        views_per_embryo=3,
        top_cell_types=4,
    )
    if len(dataset) < 2:
        pytest.skip("Insufficient embryo-view samples")

    batch = collate_embryo_view([dataset[0], dataset[1]])
    assert batch["view_0_genes"].shape[0] == 2
    assert batch["view_0_genes"].shape[2] == dataset.gene_dim
    assert batch["future_founder_composition"].shape[1] == 8
    assert batch["future_celltype_composition"].shape[1] == 4
    assert batch["future_lineage_depth_stats"].shape[1] == 3
    assert batch["future_spatial_extent"].shape[1] == 4
    assert batch["future_split_fraction"].shape[1] == 1
    assert int(batch["future_views_per_embryo"][0].item()) == 3

    n_views = int(batch["views_per_embryo"][0].item())
    genes = torch.stack([batch[f"view_{i}_genes"] for i in range(n_views)], dim=1)
    context_role = torch.stack([batch[f"view_{i}_context_role"] for i in range(n_views)], dim=1)
    relative_position = torch.stack([batch[f"view_{i}_relative_position"] for i in range(n_views)], dim=1)
    token_times = torch.stack([batch[f"view_{i}_token_times"] for i in range(n_views)], dim=1)
    valid_mask = torch.stack([batch[f"view_{i}_valid_mask"] for i in range(n_views)], dim=1)
    anchor_mask = torch.stack([batch[f"view_{i}_anchor_mask"] for i in range(n_views)], dim=1)
    time = torch.stack([batch[f"view_{i}_time"] for i in range(n_views)], dim=1)

    model = EmbryoStateModel(
        gene_dim=dataset.gene_dim,
        context_size=8,
        celltype_dim=4,
        model_type="multi_cell",
        d_model=64,
        n_heads=4,
        n_layers=2,
        head_dim=16,
        use_pairwise_spatial_bias=True,
    )
    out = model(
        genes=genes,
        time=time,
        token_times=token_times,
        valid_mask=valid_mask,
        anchor_mask=anchor_mask,
        context_role=context_role,
        relative_position=relative_position,
    )

    assert out.embryo_latent.shape == (2, 64)
    assert out.local_latents.shape[:2] == (2, n_views)
    assert out.future_founder_composition.shape == (2, 8)
    assert out.future_celltype_composition.shape == (2, 4)
    assert out.future_lineage_depth_stats.shape == (2, 3)
    assert out.future_spatial_extent.shape == (2, 4)
    assert out.future_split_fraction.shape == (2, 1)


def test_embryo_masked_view_model_forward():
    """Embryo masked multi-view model should reconstruct masked view content from visible views."""
    h5ad_path = Path("dataset/processed/nema_extended_large2025.h5ad")
    if not h5ad_path.exists():
        pytest.skip("Processed gene-context dataset not available")

    dataset = EmbryoViewDataset(
        h5ad_path=h5ad_path,
        n_hvg=32,
        context_size=8,
        global_context_size=2,
        dt_minutes=40.0,
        samples_per_pair=2,
        split="train",
        sampling_strategy="spatial_anchor",
        random_seed=0,
        views_per_embryo=4,
        top_cell_types=4,
    )
    if len(dataset) < 2:
        pytest.skip("Insufficient embryo-view samples")

    batch = collate_embryo_view([dataset[0], dataset[1]])
    n_views = int(batch["views_per_embryo"][0].item())
    genes = torch.stack([batch[f"view_{i}_genes"] for i in range(n_views)], dim=1)
    context_role = torch.stack([batch[f"view_{i}_context_role"] for i in range(n_views)], dim=1)
    relative_position = torch.stack([batch[f"view_{i}_relative_position"] for i in range(n_views)], dim=1)
    token_times = torch.stack([batch[f"view_{i}_token_times"] for i in range(n_views)], dim=1)
    valid_mask = torch.stack([batch[f"view_{i}_valid_mask"] for i in range(n_views)], dim=1)
    anchor_mask = torch.stack([batch[f"view_{i}_anchor_mask"] for i in range(n_views)], dim=1)
    time = torch.stack([batch[f"view_{i}_time"] for i in range(n_views)], dim=1)
    n_future_views = int(batch["future_views_per_embryo"][0].item())
    future_genes = torch.stack([batch[f"future_view_{i}_genes"] for i in range(n_future_views)], dim=1)
    future_context_role = torch.stack([batch[f"future_view_{i}_context_role"] for i in range(n_future_views)], dim=1)
    future_relative_position = torch.stack([batch[f"future_view_{i}_relative_position"] for i in range(n_future_views)], dim=1)
    future_token_times = torch.stack([batch[f"future_view_{i}_token_times"] for i in range(n_future_views)], dim=1)
    future_valid_mask = torch.stack([batch[f"future_view_{i}_valid_mask"] for i in range(n_future_views)], dim=1)
    future_anchor_mask = torch.stack([batch[f"future_view_{i}_anchor_mask"] for i in range(n_future_views)], dim=1)
    future_time = torch.stack([batch[f"future_view_{i}_time"] for i in range(n_future_views)], dim=1)
    masked_view_mask = torch.tensor(
        [[False, True, False, False], [True, False, False, False]],
        dtype=torch.bool,
    )
    masked_future_view_mask = torch.tensor(
        [[True, False, False, False], [False, True, False, False]],
        dtype=torch.bool,
    )

    model = EmbryoMaskedViewModel(
        gene_dim=dataset.gene_dim,
        context_size=8,
        model_type="multi_cell",
        d_model=64,
        n_heads=4,
        n_layers=2,
        head_dim=16,
        use_pairwise_spatial_bias=True,
    )
    out = model(
        genes=genes,
        time=time,
        token_times=token_times,
        valid_mask=valid_mask,
        anchor_mask=anchor_mask,
        masked_view_mask=masked_view_mask,
        future_genes=future_genes,
        future_time=future_time,
        future_token_times=future_token_times,
        future_valid_mask=future_valid_mask,
        future_anchor_mask=future_anchor_mask,
        masked_future_view_mask=masked_future_view_mask,
        context_role=context_role,
        relative_position=relative_position,
        future_context_role=future_context_role,
        future_relative_position=future_relative_position,
    )
    assert out.embryo_latent.shape == (2, 64)
    assert out.visible_embryo_latent.shape == (2, 64)
    assert out.local_latents.shape == (2, n_views, 64)
    assert out.future_local_latents.shape == (2, n_future_views, 64)
    assert out.pred_masked_view_latents.shape == (2, 64)
    assert out.pred_masked_view_genes.shape == (2, dataset.gene_dim)
    assert out.pred_masked_future_view_latents.shape == (2, 64)
    assert out.pred_masked_future_view_genes.shape == (2, dataset.gene_dim)


def test_local_cell_code_model_forward():
    """LocalCellCodeModel should reconstruct structured local cell state from one patch view."""
    h5ad_path = Path("dataset/processed/nema_extended_large2025.h5ad")
    if not h5ad_path.exists():
        pytest.skip("Processed gene-context dataset not available")

    dataset = PatchSetDataset(
        h5ad_path=h5ad_path,
        n_hvg=32,
        context_size=8,
        global_context_size=2,
        dt_minutes=40.0,
        samples_per_pair=2,
        split="train",
        sampling_strategy="spatial_anchor",
        random_seed=0,
    )
    if len(dataset) < 2:
        pytest.skip("Insufficient patch-set samples")

    batch = collate_patch_set([dataset[0], dataset[1]])
    model = LocalCellCodeModel(
        gene_dim=dataset.gene_dim,
        context_size=8,
        model_type="multi_cell",
        d_model=64,
        n_heads=4,
        n_layers=2,
        head_dim=16,
        use_pairwise_spatial_bias=True,
        code_tokens=4,
    )
    patch_latent, local_code_tokens = model.encode_local_code(
        genes=batch["current_genes"],
        time=batch["current_time"],
        token_times=batch["current_token_times"],
        valid_mask=batch["current_valid_mask"],
        anchor_mask=batch["current_anchor_mask"],
        context_role=batch["current_context_role"],
        relative_position=batch["current_relative_position"],
    )
    decoded = model.decode_local_code(local_code_tokens)
    out = model(
        genes=batch["current_genes"],
        time=batch["current_time"],
        token_times=batch["current_token_times"],
        valid_mask=batch["current_valid_mask"],
        anchor_mask=batch["current_anchor_mask"],
        context_role=batch["current_context_role"],
        relative_position=batch["current_relative_position"],
    )

    assert patch_latent.shape == (2, 64)
    assert local_code_tokens.shape == (2, 4, 64)
    assert decoded.pred_cell_genes.shape == (2, 8, dataset.gene_dim)
    assert decoded.pred_cell_positions.shape == (2, 8, 3)
    assert decoded.pred_cell_count.shape == (2,)
    assert out.patch_latent.shape == (2, 64)
    assert out.local_code_tokens.shape == (2, 4, 64)
    assert out.pred_cell_genes.shape == (2, 8, dataset.gene_dim)
    assert out.pred_cell_positions.shape == (2, 8, 3)
    assert out.pred_cell_valid_logits.shape == (2, 8)
    assert out.pred_cell_spatial_logits.shape == (2, 8)
    assert out.pred_cell_count.shape == (2,)
    assert out.pred_mean_gene.shape == (2, dataset.gene_dim)
    assert out.pred_patch_latent.shape == (2, 64)


def test_embryo_future_set_model_forward():
    """Embryo future-set model should predict masked future local-view sets."""
    h5ad_path = Path("dataset/processed/nema_extended_large2025.h5ad")
    if not h5ad_path.exists():
        pytest.skip("Processed gene-context dataset not available")

    dataset = EmbryoViewDataset(
        h5ad_path=h5ad_path,
        n_hvg=32,
        context_size=8,
        global_context_size=2,
        dt_minutes=40.0,
        samples_per_pair=2,
        split="train",
        sampling_strategy="spatial_anchor",
        random_seed=0,
        views_per_embryo=3,
        future_views_per_embryo=3,
        top_cell_types=4,
    )
    if len(dataset) < 2:
        pytest.skip("Insufficient embryo-view samples")

    batch = collate_embryo_view([dataset[0], dataset[1]])
    n_views = int(batch["views_per_embryo"][0].item())
    n_future_views = int(batch["future_views_per_embryo"][0].item())
    genes = torch.stack([batch[f"view_{i}_genes"] for i in range(n_views)], dim=1)
    context_role = torch.stack([batch[f"view_{i}_context_role"] for i in range(n_views)], dim=1)
    relative_position = torch.stack([batch[f"view_{i}_relative_position"] for i in range(n_views)], dim=1)
    token_times = torch.stack([batch[f"view_{i}_token_times"] for i in range(n_views)], dim=1)
    valid_mask = torch.stack([batch[f"view_{i}_valid_mask"] for i in range(n_views)], dim=1)
    anchor_mask = torch.stack([batch[f"view_{i}_anchor_mask"] for i in range(n_views)], dim=1)
    time = torch.stack([batch[f"view_{i}_time"] for i in range(n_views)], dim=1)
    future_genes = torch.stack([batch[f"future_view_{i}_genes"] for i in range(n_future_views)], dim=1)
    future_context_role = torch.stack([batch[f"future_view_{i}_context_role"] for i in range(n_future_views)], dim=1)
    future_relative_position = torch.stack([batch[f"future_view_{i}_relative_position"] for i in range(n_future_views)], dim=1)
    future_split_fraction = torch.stack([batch[f"future_view_{i}_split_fraction"] for i in range(n_future_views)], dim=1)
    future_token_times = torch.stack([batch[f"future_view_{i}_token_times"] for i in range(n_future_views)], dim=1)
    future_valid_mask = torch.stack([batch[f"future_view_{i}_valid_mask"] for i in range(n_future_views)], dim=1)
    future_anchor_mask = torch.stack([batch[f"future_view_{i}_anchor_mask"] for i in range(n_future_views)], dim=1)
    future_time = torch.stack([batch[f"future_view_{i}_time"] for i in range(n_future_views)], dim=1)

    backbone = EmbryoMaskedViewModel(
        gene_dim=dataset.gene_dim,
        context_size=8,
        model_type="multi_cell",
        d_model=64,
        n_heads=4,
        n_layers=2,
        head_dim=16,
        use_pairwise_spatial_bias=True,
    )
    model = EmbryoFutureSetModel(
        backbone=backbone,
        future_slots=1,
        d_model=64,
        gene_dim=dataset.gene_dim,
        use_current_local_tokens=True,
        learn_current_token_gate=True,
        current_token_gate_init=0.5,
        current_conditioning_mode="cross_attention_memory",
        code_tokens=4,
    )
    masked_view_mask = torch.tensor(
        [[False, True, False], [False, False, False]],
        dtype=torch.bool,
    )
    masked_future_view_mask = torch.tensor(
        [[False, True, False], [True, False, False]],
        dtype=torch.bool,
    )
    out = model(
        genes=genes,
        time=time,
        token_times=token_times,
        valid_mask=valid_mask,
        anchor_mask=anchor_mask,
        future_genes=future_genes,
        future_time=future_time,
        future_token_times=future_token_times,
        future_valid_mask=future_valid_mask,
        future_anchor_mask=future_anchor_mask,
        masked_future_view_mask=masked_future_view_mask,
        future_split_fraction=future_split_fraction,
        masked_view_mask=masked_view_mask,
        context_role=context_role,
        relative_position=relative_position,
        future_context_role=future_context_role,
        future_relative_position=future_relative_position,
    )
    assert out.context_embryo_latent.shape == (2, 64)
    assert out.future_local_latents.shape == (2, n_future_views, 64)
    assert out.pred_future_set_latents.shape == (2, 1, 64)
    assert out.pred_future_set_raw_pooled_latent.shape == (2, 64)
    assert out.pred_future_set_pooled_latent.shape == (2, 64)
    assert torch.allclose(
        out.pred_future_set_raw_pooled_latent,
        out.pred_future_set_pooled_latent,
        atol=1e-5,
    )
    assert out.pred_future_set_genes.shape == (2, 1, dataset.gene_dim)
    assert out.pred_future_mass.shape == (2, 1)
    assert out.pred_future_split_logits.shape == (2, 1)
    assert out.pred_future_survival_logits.shape == (2, 1)
    assert out.pred_future_split_count.shape == (2, 1)
    assert out.pred_future_local_codes.shape == (2, 1, 4, 64)
    assert out.target_future_set_latents.shape == (2, 1, 64)
    assert out.target_future_set_raw_pooled_latent.shape == (2, 64)
    assert out.target_future_set_pooled_latent.shape == (2, 64)
    assert torch.allclose(
        out.target_future_set_raw_pooled_latent,
        out.target_future_set_pooled_latent,
        atol=1e-5,
    )
    assert out.target_future_set_genes.shape == (2, 1, dataset.gene_dim)
    assert out.target_future_mass.shape == (2, 1)
    assert out.target_future_split_fraction.shape == (2, 1)
    assert out.target_future_survival.shape == (2, 1)
    assert out.target_future_split_count.shape == (2, 1)
    assert out.target_future_local_codes.shape == (2, 1, 4, 64)
    zero_weight_pooled = model.pool_future_set(out.pred_future_set_latents.detach(), torch.zeros(2, 1))
    assert zero_weight_pooled.shape == (2, 64)
    assert torch.isfinite(zero_weight_pooled).all()
    decoded = model.decode_future_local_codes(out.pred_future_local_codes)
    assert decoded.pred_cell_genes.shape == (2, 1, 8, dataset.gene_dim)
    assert decoded.pred_cell_positions.shape == (2, 1, 8, 3)
    assert out.current_local_token_gate is not None
    assert 0.0 < float(out.current_local_token_gate.item()) < 1.0


def test_pooled_latent_canonicalizer_applies_fixed_linear_basis():
    canonicalizer = PooledLatentCanonicalizer(
        dim=2,
        mean=torch.tensor([1.0, -1.0]),
        transform=torch.tensor([[2.0, 0.0], [0.0, 0.5]]),
        mode="diag_standardize",
    )
    x = torch.tensor([[3.0, 3.0], [1.0, -1.0]])
    y = canonicalizer(x)
    expected = torch.tensor([[4.0, 2.0], [0.0, 0.0]])
    assert torch.allclose(y, expected)


def test_embryo_future_set_model_backward_compatible_without_current_local_tokens():
    """Future-set model should still support the older pooled-current contract."""
    h5ad_path = Path("dataset/processed/nema_extended_large2025.h5ad")
    if not h5ad_path.exists():
        pytest.skip("Processed gene-context dataset not available")

    dataset = EmbryoViewDataset(
        h5ad_path=h5ad_path,
        n_hvg=32,
        context_size=8,
        global_context_size=2,
        dt_minutes=40.0,
        samples_per_pair=2,
        split="train",
        sampling_strategy="spatial_anchor",
        random_seed=1,
        views_per_embryo=3,
        future_views_per_embryo=3,
        top_cell_types=4,
    )
    if len(dataset) < 2:
        pytest.skip("Insufficient embryo-view samples")

    batch = collate_embryo_view([dataset[0], dataset[1]])
    n_views = int(batch["views_per_embryo"][0].item())
    n_future_views = int(batch["future_views_per_embryo"][0].item())
    genes = torch.stack([batch[f"view_{i}_genes"] for i in range(n_views)], dim=1)
    context_role = torch.stack([batch[f"view_{i}_context_role"] for i in range(n_views)], dim=1)
    relative_position = torch.stack([batch[f"view_{i}_relative_position"] for i in range(n_views)], dim=1)
    token_times = torch.stack([batch[f"view_{i}_token_times"] for i in range(n_views)], dim=1)
    valid_mask = torch.stack([batch[f"view_{i}_valid_mask"] for i in range(n_views)], dim=1)
    anchor_mask = torch.stack([batch[f"view_{i}_anchor_mask"] for i in range(n_views)], dim=1)
    time = torch.stack([batch[f"view_{i}_time"] for i in range(n_views)], dim=1)
    future_genes = torch.stack([batch[f"future_view_{i}_genes"] for i in range(n_future_views)], dim=1)
    future_context_role = torch.stack([batch[f"future_view_{i}_context_role"] for i in range(n_future_views)], dim=1)
    future_relative_position = torch.stack([batch[f"future_view_{i}_relative_position"] for i in range(n_future_views)], dim=1)
    future_split_fraction = torch.stack([batch[f"future_view_{i}_split_fraction"] for i in range(n_future_views)], dim=1)
    future_token_times = torch.stack([batch[f"future_view_{i}_token_times"] for i in range(n_future_views)], dim=1)
    future_valid_mask = torch.stack([batch[f"future_view_{i}_valid_mask"] for i in range(n_future_views)], dim=1)
    future_anchor_mask = torch.stack([batch[f"future_view_{i}_anchor_mask"] for i in range(n_future_views)], dim=1)
    future_time = torch.stack([batch[f"future_view_{i}_time"] for i in range(n_future_views)], dim=1)

    backbone = EmbryoMaskedViewModel(
        gene_dim=dataset.gene_dim,
        context_size=8,
        model_type="multi_cell",
        d_model=64,
        n_heads=4,
        n_layers=2,
        head_dim=16,
        use_pairwise_spatial_bias=True,
    )
    model = EmbryoFutureSetModel(
        backbone=backbone,
        future_slots=1,
        d_model=64,
        gene_dim=dataset.gene_dim,
        use_current_local_tokens=False,
    )
    masked_view_mask = torch.tensor(
        [[False, True, False], [False, False, False]],
        dtype=torch.bool,
    )
    masked_future_view_mask = torch.tensor(
        [[False, True, False], [True, False, False]],
        dtype=torch.bool,
    )
    out = model(
        genes=genes,
        time=time,
        token_times=token_times,
        valid_mask=valid_mask,
        anchor_mask=anchor_mask,
        future_genes=future_genes,
        future_time=future_time,
        future_token_times=future_token_times,
        future_valid_mask=future_valid_mask,
        future_anchor_mask=future_anchor_mask,
        masked_future_view_mask=masked_future_view_mask,
        future_split_fraction=future_split_fraction,
        masked_view_mask=masked_view_mask,
        context_role=context_role,
        relative_position=relative_position,
        future_context_role=future_context_role,
        future_relative_position=future_relative_position,
    )
    assert out.pred_future_set_latents.shape == (2, 1, 64)
    assert out.pred_future_set_pooled_latent.shape == (2, 64)
    assert out.current_local_token_gate is None


def test_embryo_future_set_model_supports_dense_future_token_prediction():
    """Future-set model should support strict token-JEPA prediction without a single-slot token bottleneck."""
    h5ad_path = Path("dataset/processed/nema_extended_large2025.h5ad")
    if not h5ad_path.exists():
        pytest.skip("Processed gene-context dataset not available")

    dataset = EmbryoViewDataset(
        h5ad_path=h5ad_path,
        n_hvg=32,
        context_size=8,
        global_context_size=2,
        dt_minutes=40.0,
        samples_per_pair=2,
        split="train",
        sampling_strategy="spatial_anchor",
        random_seed=2,
        views_per_embryo=3,
        future_views_per_embryo=3,
        top_cell_types=4,
    )
    if len(dataset) < 2:
        pytest.skip("Insufficient embryo-view samples")

    batch = collate_embryo_view([dataset[0], dataset[1]])
    n_views = int(batch["views_per_embryo"][0].item())
    n_future_views = int(batch["future_views_per_embryo"][0].item())
    genes = torch.stack([batch[f"view_{i}_genes"] for i in range(n_views)], dim=1)
    context_role = torch.stack([batch[f"view_{i}_context_role"] for i in range(n_views)], dim=1)
    relative_position = torch.stack([batch[f"view_{i}_relative_position"] for i in range(n_views)], dim=1)
    token_times = torch.stack([batch[f"view_{i}_token_times"] for i in range(n_views)], dim=1)
    valid_mask = torch.stack([batch[f"view_{i}_valid_mask"] for i in range(n_views)], dim=1)
    anchor_mask = torch.stack([batch[f"view_{i}_anchor_mask"] for i in range(n_views)], dim=1)
    time = torch.stack([batch[f"view_{i}_time"] for i in range(n_views)], dim=1)
    future_genes = torch.stack([batch[f"future_view_{i}_genes"] for i in range(n_future_views)], dim=1)
    future_context_role = torch.stack([batch[f"future_view_{i}_context_role"] for i in range(n_future_views)], dim=1)
    future_relative_position = torch.stack([batch[f"future_view_{i}_relative_position"] for i in range(n_future_views)], dim=1)
    future_split_fraction = torch.stack([batch[f"future_view_{i}_split_fraction"] for i in range(n_future_views)], dim=1)
    future_token_times = torch.stack([batch[f"future_view_{i}_token_times"] for i in range(n_future_views)], dim=1)
    future_valid_mask = torch.stack([batch[f"future_view_{i}_valid_mask"] for i in range(n_future_views)], dim=1)
    future_anchor_mask = torch.stack([batch[f"future_view_{i}_anchor_mask"] for i in range(n_future_views)], dim=1)
    future_time = torch.stack([batch[f"future_view_{i}_time"] for i in range(n_future_views)], dim=1)

    backbone = EmbryoMaskedViewModel(
        gene_dim=dataset.gene_dim,
        context_size=8,
        model_type="multi_cell",
        d_model=64,
        n_heads=4,
        n_layers=2,
        head_dim=16,
        use_pairwise_spatial_bias=True,
    )
    model = EmbryoFutureSetModel(
        backbone=backbone,
        future_slots=1,
        d_model=64,
        gene_dim=dataset.gene_dim,
        use_current_local_tokens=True,
        learn_current_token_gate=True,
        current_token_gate_init=0.5,
        current_conditioning_mode="flat_tokens",
        code_tokens=4,
        predict_dense_future_tokens=True,
        strict_token_jepa=True,
        token_readout_anchor=FrozenLinearTokenReadout(64),
    )
    masked_future_view_mask = torch.tensor(
        [[False, True, False], [True, False, False]],
        dtype=torch.bool,
    )
    out = model(
        genes=genes,
        time=time,
        token_times=token_times,
        valid_mask=valid_mask,
        anchor_mask=anchor_mask,
        future_genes=future_genes,
        future_time=future_time,
        future_token_times=future_token_times,
        future_valid_mask=future_valid_mask,
        future_anchor_mask=future_anchor_mask,
        masked_future_view_mask=masked_future_view_mask,
        future_split_fraction=future_split_fraction,
        masked_view_mask=None,
        context_role=context_role,
        relative_position=relative_position,
        future_context_role=future_context_role,
        future_relative_position=future_relative_position,
    )
    assert model.future_token_queries.shape == (1, 8, 64)
    assert out.pred_future_local_codes.shape == (2, 1, 8, 64)
    assert out.target_future_local_codes.shape == (2, 1, 8, 64)
    assert out.pred_future_set_latents.shape == (2, 1, 64)
    assert out.pred_future_set_pooled_latent.shape == (2, 64)
    assert torch.isfinite(out.pred_future_set_pooled_latent).all()
    decoded = model.decode_future_local_codes(out.pred_future_local_codes)
    assert decoded.pred_cell_genes.shape == (2, 1, 8, dataset.gene_dim)
    assert decoded.pred_cell_positions.shape == (2, 1, 8, 3)


def test_embryo_future_set_model_flat_current_tokens_still_supported():
    """Future-set model should also support the older flat-token current conditioning path."""
    h5ad_path = Path("dataset/processed/nema_extended_large2025.h5ad")
    if not h5ad_path.exists():
        pytest.skip("Processed gene-context dataset not available")

    dataset = EmbryoViewDataset(
        h5ad_path=h5ad_path,
        n_hvg=32,
        context_size=8,
        global_context_size=2,
        dt_minutes=40.0,
        samples_per_pair=2,
        split="train",
        sampling_strategy="spatial_anchor",
        random_seed=2,
        views_per_embryo=3,
        future_views_per_embryo=3,
        top_cell_types=4,
    )
    if len(dataset) < 2:
        pytest.skip("Insufficient embryo-view samples")

    batch = collate_embryo_view([dataset[0], dataset[1]])
    n_views = int(batch["views_per_embryo"][0].item())
    n_future_views = int(batch["future_views_per_embryo"][0].item())
    genes = torch.stack([batch[f"view_{i}_genes"] for i in range(n_views)], dim=1)
    context_role = torch.stack([batch[f"view_{i}_context_role"] for i in range(n_views)], dim=1)
    relative_position = torch.stack([batch[f"view_{i}_relative_position"] for i in range(n_views)], dim=1)
    token_times = torch.stack([batch[f"view_{i}_token_times"] for i in range(n_views)], dim=1)
    valid_mask = torch.stack([batch[f"view_{i}_valid_mask"] for i in range(n_views)], dim=1)
    anchor_mask = torch.stack([batch[f"view_{i}_anchor_mask"] for i in range(n_views)], dim=1)
    time = torch.stack([batch[f"view_{i}_time"] for i in range(n_views)], dim=1)
    future_genes = torch.stack([batch[f"future_view_{i}_genes"] for i in range(n_future_views)], dim=1)
    future_context_role = torch.stack([batch[f"future_view_{i}_context_role"] for i in range(n_future_views)], dim=1)
    future_relative_position = torch.stack([batch[f"future_view_{i}_relative_position"] for i in range(n_future_views)], dim=1)
    future_split_fraction = torch.stack([batch[f"future_view_{i}_split_fraction"] for i in range(n_future_views)], dim=1)
    future_token_times = torch.stack([batch[f"future_view_{i}_token_times"] for i in range(n_future_views)], dim=1)
    future_valid_mask = torch.stack([batch[f"future_view_{i}_valid_mask"] for i in range(n_future_views)], dim=1)
    future_anchor_mask = torch.stack([batch[f"future_view_{i}_anchor_mask"] for i in range(n_future_views)], dim=1)
    future_time = torch.stack([batch[f"future_view_{i}_time"] for i in range(n_future_views)], dim=1)

    backbone = EmbryoMaskedViewModel(
        gene_dim=dataset.gene_dim,
        context_size=8,
        model_type="multi_cell",
        d_model=64,
        n_heads=4,
        n_layers=2,
        head_dim=16,
        use_pairwise_spatial_bias=True,
    )
    model = EmbryoFutureSetModel(
        backbone=backbone,
        future_slots=1,
        d_model=64,
        gene_dim=dataset.gene_dim,
        use_current_local_tokens=True,
        current_conditioning_mode="flat_tokens",
    )
    masked_view_mask = torch.tensor(
        [[False, True, False], [False, False, False]],
        dtype=torch.bool,
    )
    masked_future_view_mask = torch.tensor(
        [[False, True, False], [True, False, False]],
        dtype=torch.bool,
    )
    out = model(
        genes=genes,
        time=time,
        token_times=token_times,
        valid_mask=valid_mask,
        anchor_mask=anchor_mask,
        future_genes=future_genes,
        future_time=future_time,
        future_token_times=future_token_times,
        future_valid_mask=future_valid_mask,
        future_anchor_mask=future_anchor_mask,
        masked_future_view_mask=masked_future_view_mask,
        future_split_fraction=future_split_fraction,
        masked_view_mask=masked_view_mask,
        context_role=context_role,
        relative_position=relative_position,
        future_context_role=future_context_role,
        future_relative_position=future_relative_position,
    )
    assert out.pred_future_set_latents.shape == (2, 1, 64)
    assert out.pred_future_set_pooled_latent.shape == (2, 64)
    assert out.current_local_token_gate is not None
