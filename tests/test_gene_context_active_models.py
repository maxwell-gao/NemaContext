"""Focused tests for active local-state, patch-set, and masked-view training paths."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.branching_flows.gene_context import (  # noqa: E402
    GeneContextModel,
    MultiPatchSetModel,
    MultiCellPatchSetModel,
    SingleCellGeneTimeModel,
)
from src.data.gene_context_dataset import (  # noqa: E402
    GeneContextDataset,
    MultiViewPatchStateDataset,
    MultiPatchSetDataset,
    PatchSetDataset,
    collate_gene_context,
    collate_multi_patch_set,
    collate_multi_view_patch_state,
    collate_patch_set,
)
from examples.whole_organism_ar.train_masked_state_views import (  # noqa: E402
    MaskedStateViewModel,
    TemporalQueue,
    compute_masked_metrics,
)
from examples.whole_organism_ar.train_patch_set import compute_patch_set_metrics  # noqa: E402
from examples.whole_organism_ar.train_state_views import (  # noqa: E402
    StateViewModel,
    compute_metrics as compute_state_view_metrics,
)


def test_single_cell_model_has_no_cross_token_information_flow():
    """Single-cell baseline should be invariant to changes in other tokens."""
    h5ad_path = Path("dataset/processed/nema_extended_large2025.h5ad")
    if not h5ad_path.exists():
        pytest.skip("Processed gene-context dataset not available")

    dataset = GeneContextDataset(
        h5ad_path=h5ad_path,
        n_hvg=32,
        context_size=8,
        global_context_size=2,
        samples_per_pair=2,
        split="train",
        sampling_strategy="spatial_anchor",
        supervision_mode="anchor_only",
        random_seed=0,
    )
    if len(dataset) < 2:
        pytest.skip("Insufficient dataset samples for cross-token sanity check")

    item0 = dataset[0]
    item1 = dataset[1]
    batch = collate_gene_context([item0, item1])
    perturbed = {k: v.clone() for k, v in batch.items()}

    mask0 = (~batch["anchor_mask"][0]) & batch["valid_mask"][0]
    mask1 = (~batch["anchor_mask"][1]) & batch["valid_mask"][1]
    n_replace = min(int(mask0.sum().item()), int(mask1.sum().item()))
    if n_replace == 0:
        pytest.skip("No non-anchor tokens available for perturbation")

    idx0 = torch.nonzero(mask0, as_tuple=False).squeeze(-1)[:n_replace]
    idx1 = torch.nonzero(mask1, as_tuple=False).squeeze(-1)[:n_replace]
    for key in [
        "genes",
        "token_times",
        "valid_mask",
        "context_role",
        "relative_position",
    ]:
        perturbed[key][0, idx0] = batch[key][1, idx1]

    single = SingleCellGeneTimeModel(gene_dim=dataset.gene_dim, d_model=64, n_layers=2)
    multi = GeneContextModel(
        gene_dim=dataset.gene_dim,
        d_model=64,
        n_heads=4,
        n_layers=2,
        head_dim=16,
    )
    single.eval()
    multi.eval()

    with torch.no_grad():
        single_before = single(
            genes=batch["genes"],
            time=batch["time"],
            future_time=batch["future_time"],
            token_times=batch["token_times"],
            valid_mask=batch["valid_mask"],
            relative_position=batch["relative_position"],
        )
        single_after = single(
            genes=perturbed["genes"],
            time=perturbed["time"],
            future_time=perturbed["future_time"],
            token_times=perturbed["token_times"],
            valid_mask=perturbed["valid_mask"],
            relative_position=perturbed["relative_position"],
        )
        multi_before = multi(
            genes=batch["genes"],
            time=batch["time"],
            future_time=batch["future_time"],
            token_times=batch["token_times"],
            valid_mask=batch["valid_mask"],
            context_role=batch["context_role"],
            relative_position=batch["relative_position"],
        )
        multi_after = multi(
            genes=perturbed["genes"],
            time=perturbed["time"],
            future_time=perturbed["future_time"],
            token_times=perturbed["token_times"],
            valid_mask=perturbed["valid_mask"],
            context_role=perturbed["context_role"],
            relative_position=perturbed["relative_position"],
        )

    anchor_idx = torch.nonzero(batch["anchor_mask"][0], as_tuple=False).squeeze(-1)
    assert anchor_idx.numel() == 1
    anchor_idx = int(anchor_idx.item())

    assert torch.equal(
        single_before.gene_delta[0, anchor_idx],
        single_after.gene_delta[0, anchor_idx],
    )
    assert torch.equal(
        single_before.split_logits[0, anchor_idx],
        single_after.split_logits[0, anchor_idx],
    )
    assert torch.equal(
        single_before.del_logits[0, anchor_idx],
        single_after.del_logits[0, anchor_idx],
    )

    multi_gene_changed = not torch.equal(
        multi_before.gene_delta[0, anchor_idx],
        multi_after.gene_delta[0, anchor_idx],
    )
    multi_split_changed = not torch.equal(
        multi_before.split_logits[0, anchor_idx],
        multi_after.split_logits[0, anchor_idx],
    )
    multi_del_changed = not torch.equal(
        multi_before.del_logits[0, anchor_idx],
        multi_after.del_logits[0, anchor_idx],
    )
    assert multi_gene_changed or multi_split_changed or multi_del_changed


def test_matched_local_patch_supervision_masks_only_matched_local_tokens():
    """Matched local patch supervision should exclude unmatched locals and globals."""
    h5ad_path = Path("dataset/processed/nema_extended_large2025.h5ad")
    if not h5ad_path.exists():
        pytest.skip("Processed gene-context dataset not available")

    dataset = GeneContextDataset(
        h5ad_path=h5ad_path,
        n_hvg=32,
        context_size=16,
        global_context_size=4,
        samples_per_pair=2,
        split="train",
        sampling_strategy="spatial_anchor",
        supervision_mode="matched_local_patch",
        random_seed=0,
    )
    if len(dataset) == 0:
        pytest.skip("No dataset samples available")

    item = None
    for i in range(min(len(dataset), 32)):
        candidate = dataset[i]
        context_role = candidate["context_role"]
        match_type = candidate["match_type"]
        local_positions = torch.nonzero(context_role == 2, as_tuple=False).squeeze(-1)
        if local_positions.numel() == 0:
            continue
        matched_local = local_positions[
            (match_type[local_positions] != GeneContextDataset.MATCH_UNMATCHED)
            & (match_type[local_positions] != GeneContextDataset.MATCH_UNSUPERVISED)
        ]
        if matched_local.numel() > 0:
            item = candidate
            break

    if item is None:
        pytest.skip("No matched local patch sample found")

    supervision_mask = item["supervision_mask"]
    context_role = item["context_role"]
    match_type = item["match_type"]
    anchor_mask = item["anchor_mask"]

    assert anchor_mask.sum().item() == 1
    assert supervision_mask[anchor_mask].all()

    global_positions = torch.nonzero(context_role == 3, as_tuple=False).squeeze(-1)
    if global_positions.numel() > 0:
        assert not supervision_mask[global_positions].any()

    local_positions = torch.nonzero(context_role == 2, as_tuple=False).squeeze(-1)
    unmatched_local = local_positions[
        (match_type[local_positions] == GeneContextDataset.MATCH_UNMATCHED)
        | (match_type[local_positions] == GeneContextDataset.MATCH_UNSUPERVISED)
    ]
    if unmatched_local.numel() > 0:
        assert not supervision_mask[unmatched_local].any()

    matched_local = local_positions[
        (match_type[local_positions] != GeneContextDataset.MATCH_UNMATCHED)
        & (match_type[local_positions] != GeneContextDataset.MATCH_UNSUPERVISED)
    ]
    assert matched_local.numel() > 0
    assert supervision_mask[matched_local].all()
    assert not torch.any(match_type[matched_local] == GeneContextDataset.MATCH_UNSUPERVISED)


def test_patch_set_dataset_and_model_forward():
    """Patch-set dataset should produce aligned patch views and a valid forward pass."""
    h5ad_path = Path("dataset/processed/nema_extended_large2025.h5ad")
    if not h5ad_path.exists():
        pytest.skip("Processed gene-context dataset not available")

    dataset = PatchSetDataset(
        h5ad_path=h5ad_path,
        n_hvg=32,
        context_size=8,
        global_context_size=2,
        samples_per_pair=2,
        split="train",
        sampling_strategy="spatial_anchor",
        random_seed=0,
    )
    if len(dataset) == 0:
        pytest.skip("No patch-set samples available")

    batch = collate_patch_set([dataset[0], dataset[min(1, len(dataset) - 1)]])
    model = MultiCellPatchSetModel(
        gene_dim=32,
        context_size=8,
        d_model=64,
        n_heads=4,
        n_layers=2,
        head_dim=16,
        use_pairwise_spatial_bias=True,
    )
    model.eval()

    with torch.no_grad():
        output = model(
            genes=batch["current_genes"],
            time=batch["current_time"],
            future_time=batch["future_time"],
            token_times=batch["current_token_times"],
            valid_mask=batch["current_valid_mask"],
            anchor_mask=batch["current_anchor_mask"],
            context_role=batch["current_context_role"],
            relative_position=batch["current_relative_position"],
        )

    assert output.pred_future_genes.shape == (2, 8, 32)
    assert output.pred_patch_size.shape == (2,)
    assert output.pred_mean_gene.shape == (2, 32)
    assert output.patch_latent.shape == (2, 64)

    total, metrics = compute_patch_set_metrics(
        model,
        batch,
        latent_weight=0.2,
        size_weight=0.02,
        mean_weight=0.5,
        spatial_input_mode="relative_position",
    )
    assert torch.isfinite(total)
    expected_metric_keys = {
        "total",
        "total_wo_size",
        "normalized_total",
        "ot",
        "ot_per_token",
        "latent",
        "size",
        "size_abs_error",
        "size_rel_error",
        "mean_gene",
        "mean_gene_rmse",
        "mean_gene_cosine",
        "future_patch_size",
        "current_split_fraction",
        "future_split_fraction",
        "split_fraction_shift",
        "pred_diversity",
        "future_diversity",
        "diversity_abs_error",
        "pred_entropy",
        "future_entropy",
        "entropy_abs_error",
        "pca_mean_dist",
        "pca_var_dist",
    }
    assert expected_metric_keys.issubset(metrics)
    for value in metrics.values():
        assert isinstance(value, float)


def test_multi_patch_set_dataset_and_model_forward():
    """Multi-patch dataset should support patch-count extrapolation model inputs."""
    h5ad_path = Path("dataset/processed/nema_extended_large2025.h5ad")
    if not h5ad_path.exists():
        pytest.skip("Processed gene-context dataset not available")

    dataset = MultiPatchSetDataset(
        h5ad_path=h5ad_path,
        n_hvg=32,
        context_size=8,
        global_context_size=2,
        samples_per_pair=2,
        split="train",
        sampling_strategy="spatial_anchor",
        patches_per_state=2,
        random_seed=0,
    )
    if len(dataset) == 0:
        pytest.skip("No multi-patch samples available")

    batch = collate_multi_patch_set([dataset[0], dataset[min(1, len(dataset) - 1)]])
    model = MultiPatchSetModel(
        gene_dim=32,
        context_size=8,
        model_type="multi_cell",
        d_model=64,
        n_heads=4,
        n_layers=2,
        head_dim=16,
        use_pairwise_spatial_bias=True,
    )
    model.eval()

    with torch.no_grad():
        output = model(
            genes=batch["current_genes"],
            time=batch["current_time"],
            future_time=batch["future_time"],
            token_times=batch["current_token_times"],
            valid_mask=batch["current_valid_mask"],
            anchor_mask=batch["current_anchor_mask"],
            context_role=batch["current_context_role"],
            relative_position=batch["current_relative_position"],
        )

    assert output.pred_future_genes.shape == (2, 2, 8, 32)
    assert output.pred_patch_size.shape == (2, 2)
    assert output.pred_mean_gene.shape == (2, 2, 32)
    assert output.patch_latent.shape == (2, 2, 64)
    assert output.state_latent.shape == (2, 64)
    assert output.patch_attention_logits.shape == (2, 2)
    assert output.patch_attention_weights.shape == (2, 2)
    assert torch.allclose(
        output.patch_attention_weights.sum(dim=1),
        torch.ones(2),
        atol=1e-5,
    )


def test_multi_view_patch_state_dataset_and_state_model_forward():
    """Multi-view state dataset should support shared-encoder latent training."""
    h5ad_path = Path("dataset/processed/nema_extended_large2025.h5ad")
    if not h5ad_path.exists():
        pytest.skip("Processed gene-context dataset not available")

    dataset = MultiViewPatchStateDataset(
        h5ad_path=h5ad_path,
        n_hvg=32,
        context_size=8,
        global_context_size=2,
        samples_per_pair=2,
        split="train",
        sampling_strategy="spatial_anchor",
        views_per_state=2,
        future_views_per_state=1,
        random_seed=0,
    )
    if len(dataset) == 0:
        pytest.skip("No multi-view state samples available")

    batch = collate_multi_view_patch_state([dataset[0], dataset[min(1, len(dataset) - 1)]])
    sample = dataset[0]
    model = StateViewModel(
        gene_dim=32,
        context_size=8,
        model_type="multi_cell",
        d_model=64,
        n_heads=4,
        n_layers=2,
        head_dim=16,
        use_pairwise_spatial_bias=True,
    )
    model.eval()

    with torch.no_grad():
        z_a, _ = model.encode_view(batch, "current_view_0")
        z_b, _ = model.encode_view(batch, "current_view_1")
        z_f, _ = model.encode_view(batch, "future_view_0")

    assert z_a.shape == (2, 64)
    assert z_b.shape == (2, 64)
    assert z_f.shape == (2, 64)
    assert "current_view_0_indices" in sample
    assert "future_view_0_indices" in sample
    assert "current_center_min" in sample
    assert "future_center_min" in sample
    assert sample["current_view_0_indices"].ndim == 1
    assert sample["future_view_0_indices"].ndim == 1

    total, metrics = compute_state_view_metrics(model, batch, ot_weight=0.1)
    assert torch.isfinite(total)
    assert {"total", "view", "future", "ot"}.issubset(metrics)


def test_masked_state_view_model_forward():
    """Masked self-supervised state-view training path should run end-to-end."""
    h5ad_path = Path("dataset/processed/nema_extended_large2025.h5ad")
    if not h5ad_path.exists():
        pytest.skip("Processed gene-context dataset not available")

    dataset = MultiViewPatchStateDataset(
        h5ad_path=h5ad_path,
        n_hvg=32,
        context_size=8,
        global_context_size=2,
        samples_per_pair=2,
        split="train",
        sampling_strategy="spatial_anchor",
        views_per_state=2,
        future_views_per_state=2,
        random_seed=0,
    )
    if len(dataset) == 0:
        pytest.skip("No multi-view masked state samples available")

    batch = collate_multi_view_patch_state([dataset[0], dataset[min(1, len(dataset) - 1)]])
    model = MaskedStateViewModel(
        gene_dim=32,
        context_size=8,
        model_type="multi_cell",
        d_model=64,
        n_heads=4,
        n_layers=2,
        head_dim=16,
        use_pairwise_spatial_bias=True,
    )
    model.eval()

    total, metrics = compute_masked_metrics(
        model,
        batch,
        mask_ratio=0.25,
        disc_temperature=0.1,
        retrieval_margin=0.2,
        ot_weight=0.05,
    )
    assert torch.isfinite(total)
    assert {
        "total",
        "masked_view",
        "future_retrieval",
        "masked_future",
        "current_gene",
        "future_gene",
        "ot",
    }.issubset(metrics)

    total_queue, metrics_queue = compute_masked_metrics(
        model,
        batch,
        mask_ratio=0.25,
        disc_temperature=0.1,
        retrieval_margin=0.2,
        ot_weight=0.05,
        temporal_queue=TemporalQueue(max_size=8, feature_dim=64, device=torch.device("cpu")),
    )
    assert torch.isfinite(total_queue)
    assert torch.isfinite(torch.tensor(metrics_queue["future_retrieval"]))
