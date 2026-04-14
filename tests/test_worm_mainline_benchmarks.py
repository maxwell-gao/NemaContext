"""Worm mainline benchmark and baseline smoke tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from examples.whole_organism_ar import benchmark_worm_dynamics as worm_bench  # noqa: E402
from examples.whole_organism_ar import benchmark_worm_prescient as prescient_bench  # noqa: E402
from examples.whole_organism_ar import benchmark_worm_scnode as scnode_bench  # noqa: E402


def make_eval_records(record_cls):
    return [
        record_cls(
            split_key="0->40",
            current_mean=np.array([0.0, 1.0, 0.5], dtype=np.float32),
            future_mean=np.array([0.3, 1.2, 0.1], dtype=np.float32),
            pred_mean=np.array([0.2, 1.1, 0.2], dtype=np.float32),
            future_set_sinkhorn=1.5,
        ),
        record_cls(
            split_key="40->80",
            current_mean=np.array([0.2, 0.7, 0.4], dtype=np.float32),
            future_mean=np.array([0.5, 0.8, 0.0], dtype=np.float32),
            pred_mean=np.array([0.4, 0.75, 0.1], dtype=np.float32),
            future_set_sinkhorn=1.0,
        ),
    ]


def test_worm_benchmark_common_and_structure_metrics():
    records = make_eval_records(worm_bench.EvalRecord)
    top_de_idx = worm_bench.compute_top_de_idx(records, top_k=2)
    metrics = worm_bench.compute_metrics(records, top_de_idx)
    assert set(metrics) == {
        "future_set_sinkhorn",
        "mean_gene_mse",
        "mean_gene_pearson",
        "delta_gene_mse",
        "delta_gene_pearson",
        "top_de_delta_pearson",
        "top_de_sign_acc",
    }

    group_records = [
        worm_bench.GroupEvalRecord(
            split_key="0->40",
            group_key="founder:1",
            current_mean=np.array([0.0, 1.0], dtype=np.float32),
            future_mean=np.array([0.2, 0.8], dtype=np.float32),
            pred_mean=np.array([0.1, 0.75], dtype=np.float32),
        ),
        worm_bench.GroupEvalRecord(
            split_key="40->80",
            group_key="founder:1",
            current_mean=np.array([0.1, 0.9], dtype=np.float32),
            future_mean=np.array([0.3, 0.7], dtype=np.float32),
            pred_mean=np.array([0.25, 0.72], dtype=np.float32),
        ),
    ]
    group_corr = worm_bench.aggregate_group_metric(group_records)
    assert isinstance(group_corr, float)


def test_scnode_style_baseline_forward_and_metrics():
    model = scnode_bench.ScNODEStyleBaseline(gene_dim=6, latent_dim=8, hidden_dim=16)
    current = torch.randn(2, 4, 6)
    current_time = torch.tensor([40.0, 80.0], dtype=torch.float32)
    future_time = torch.tensor([80.0, 120.0], dtype=torch.float32)
    pred_future, recon = model(current, current_time, future_time)
    assert pred_future.shape == current.shape
    assert recon.shape == current.shape

    records = make_eval_records(scnode_bench.EvalRecord)
    top_de_idx = scnode_bench.compute_top_de_idx(records, top_k=2)
    metrics = scnode_bench.compute_metrics(records, top_de_idx)
    assert set(metrics) == {
        "future_set_sinkhorn",
        "mean_gene_mse",
        "mean_gene_pearson",
        "delta_gene_mse",
        "delta_gene_pearson",
        "top_de_delta_pearson",
        "top_de_sign_acc",
    }


def test_prescient_style_baseline_forward_and_metrics():
    model = prescient_bench.PrescientStyleBaseline(gene_dim=6, latent_dim=8, hidden_dim=16)
    current = torch.randn(2, 4, 6)
    current_time = torch.tensor([40.0, 80.0], dtype=torch.float32)
    future_time = torch.tensor([80.0, 120.0], dtype=torch.float32)
    pred_future, recon = model(current, current_time, future_time)
    assert pred_future.shape == current.shape
    assert recon.shape == current.shape

    records = make_eval_records(prescient_bench.EvalRecord)
    top_de_idx = prescient_bench.compute_top_de_idx(records, top_k=2)
    metrics = prescient_bench.compute_metrics(records, top_de_idx)
    assert set(metrics) == {
        "future_set_sinkhorn",
        "mean_gene_mse",
        "mean_gene_pearson",
        "delta_gene_mse",
        "delta_gene_pearson",
        "top_de_delta_pearson",
        "top_de_sign_acc",
    }
