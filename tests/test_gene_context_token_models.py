"""Focused tests for token-level gene-context baselines and supervision masks."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.branching_flows.gene_context import (  # noqa: E402
    GeneContextModel,
    SingleCellGeneTimeModel,
)
from src.data.gene_context_dataset import (  # noqa: E402
    GeneContextDataset,
    collate_gene_context,
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
