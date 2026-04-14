"""Worm mainline tests for the active Large2025 lineage dynamics path."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import sparse
from torch.utils.data import DataLoader

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from examples.whole_organism_ar.train_large2025_lineage_stage1 import (  # noqa: E402
    compute_loss,
    prepare_batch,
    run_epoch,
)
from src.branching_flows.gene_context import LineageWholeEmbryoModel  # noqa: E402
from src.data.builder.expression_loader import ExpressionLoader  # noqa: E402
from src.data.gene_context_dataset import (  # noqa: E402
    Large2025WholeEmbryoDataset,
    collate_large2025_whole_embryo,
)


@pytest.fixture
def synthetic_large2025(monkeypatch):
    gene_dim = 8
    cells_per_bin = 4
    time_bins = [0.0, 40.0, 80.0]
    total_cells = cells_per_bin * len(time_bins)

    expr = np.arange(total_cells * gene_dim, dtype=np.float32).reshape(total_cells, gene_dim) + 1.0
    expr = sparse.csr_matrix(expr)
    cell_rows = []
    lineages = [
        "ABa",
        "ABp",
        "MSa",
        "E",
        "ABal",
        "ABpl",
        "MSap",
        "C",
        "ABala",
        "ABpla",
        "MSapp",
        "D",
    ]
    for idx, time_bin in enumerate(time_bins):
        for j in range(cells_per_bin):
            cell_idx = idx * cells_per_bin + j
            cell_rows.append(
                {
                    "barcode": f"cell_{cell_idx}",
                    "smoothed_embryo_time": time_bin,
                    "cell_type": f"ctype_{j % 2}",
                    "packer_cell_type": f"packer_{j % 2}",
                    "lineage_complete": lineages[cell_idx],
                    "species": "C.elegans",
                }
            )
    cell_df = pd.DataFrame(cell_rows)
    gene_df = pd.DataFrame({"gene_short_name": [f"gene_{i}" for i in range(gene_dim)]})

    def _fake_load_large2025(self, species_filter="C.elegans", min_umi=0):
        return expr.copy(), cell_df.copy(), gene_df.copy()

    monkeypatch.setattr(ExpressionLoader, "load_large2025", _fake_load_large2025)
    return {
        "gene_dim": gene_dim,
        "token_budget": cells_per_bin,
    }


def build_dataset(split: str = "all") -> Large2025WholeEmbryoDataset:
    return Large2025WholeEmbryoDataset(
        data_dir="dataset/raw",
        n_hvg=8,
        token_budget=4,
        history_frames=1,
        dt_minutes=40.0,
        time_bin_minutes=40.0,
        min_cells_per_snapshot=2,
        split=split,
        val_fraction=0.34,
        random_seed=0,
        species_filter="C.elegans",
        min_umi=0,
    )


def test_large2025_dataset_and_collate(synthetic_large2025):
    dataset = build_dataset(split="all")
    assert len(dataset) == 2

    item = dataset[0]
    required = {
        "history_frames",
        "history_frame_0_genes",
        "history_frame_0_valid_mask",
        "history_frame_0_founder_ids",
        "history_frame_0_region_ids",
        "future_genes",
        "future_valid_mask",
        "future_founder_ids",
        "future_region_ids",
        "future_mean_gene",
    }
    assert required.issubset(item.keys())
    assert item["history_frame_0_genes"].shape == (4, synthetic_large2025["gene_dim"])
    assert item["future_genes"].shape == (4, synthetic_large2025["gene_dim"])

    batch = collate_large2025_whole_embryo([dataset[0], dataset[1]])
    assert batch["history_frame_0_genes"].shape == (2, 4, synthetic_large2025["gene_dim"])
    assert batch["future_genes"].shape == (2, 4, synthetic_large2025["gene_dim"])
    assert batch["future_mean_gene"].shape == (2, synthetic_large2025["gene_dim"])
    assert torch.all(batch["history_frame_0_valid_mask"].sum(dim=1) >= 2)


def test_lineage_whole_embryo_model_forward(synthetic_large2025):
    dataset = build_dataset(split="all")
    batch = collate_large2025_whole_embryo([dataset[0], dataset[1]])
    batch = prepare_batch(batch, "cpu")

    model = LineageWholeEmbryoModel(
        gene_dim=dataset.gene_dim,
        context_size=dataset.token_budget,
        history_frames=dataset.history_frames,
        lineage_binary_dim=dataset.lineage_binary.shape[1],
        founder_vocab_size=len(dataset.FOUNDER_VOCAB),
        d_model=64,
        n_heads=4,
        n_spatial_layers=1,
        n_temporal_layers=1,
        n_decoder_layers=1,
        head_dim=16,
    )
    out = model(
        genes=batch["history_genes"],
        time=batch["history_time"],
        future_time=batch["future_time"],
        token_times=batch["history_token_times"],
        valid_mask=batch["history_valid_mask"],
        lineage_binary=batch["history_lineage_binary"],
        founder_ids=batch["history_founder_ids"],
        lineage_depth=batch["history_lineage_depth"],
        lineage_valid=batch["history_lineage_valid"],
        token_rank=batch["history_token_rank"],
    )

    assert out.pred_future_genes.shape == batch["future_genes"].shape
    assert out.pred_history_genes.shape == batch["history_genes"].shape
    assert out.pred_future_token_states.shape[:2] == batch["future_genes"].shape[:2]
    assert out.pred_mean_gene.shape == batch["future_mean_gene"].shape


def test_stage1_dynamics_training_helpers(synthetic_large2025):
    dataset = build_dataset(split="all")
    loader = DataLoader(
        [dataset[0], dataset[1]],
        batch_size=2,
        shuffle=False,
        collate_fn=collate_large2025_whole_embryo,
    )
    batch = next(iter(loader))
    batch = prepare_batch(batch, "cpu")

    model = LineageWholeEmbryoModel(
        gene_dim=dataset.gene_dim,
        context_size=dataset.token_budget,
        history_frames=dataset.history_frames,
        lineage_binary_dim=dataset.lineage_binary.shape[1],
        founder_vocab_size=len(dataset.FOUNDER_VOCAB),
        d_model=64,
        n_heads=4,
        n_spatial_layers=1,
        n_temporal_layers=1,
        n_decoder_layers=1,
        head_dim=16,
    )
    args = SimpleNamespace(
        device="cpu",
        mask_ratio=0.0,
        masked_gene_weight=0.0,
        gene_set_weight=1.0,
        mean_gene_weight=0.2,
        gene_sinkhorn_blur=0.1,
    )

    loss, metrics = compute_loss(model, batch, args, apply_mask=True)
    assert loss.numel() == 1
    assert metrics["masked_gene"] == 0.0
    assert metrics["gene_set"] > 0.0
    assert metrics["mean_gene"] >= 0.0

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_metrics = run_epoch(model, loader, args, optimizer=optimizer)
    eval_metrics = run_epoch(model, loader, args, optimizer=None)
    for result in (train_metrics, eval_metrics):
        assert set(result) == {
            "total",
            "masked_gene",
            "gene_set",
            "mean_gene",
            "persistence_mean_gene",
        }
        assert result["masked_gene"] == 0.0
        assert result["gene_set"] > 0.0
