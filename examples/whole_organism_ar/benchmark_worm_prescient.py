#!/usr/bin/env python3
"""PRESCIENT-style OT baseline on the worm Large2025 forecasting benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.branching_flows.emergent_loss import sinkhorn_divergence  # noqa: E402
from src.data.gene_context_dataset import Large2025WholeEmbryoDataset, collate_large2025_whole_embryo  # noqa: E402


@dataclass(frozen=True)
class EvalRecord:
    split_key: str
    current_mean: np.ndarray
    future_mean: np.ndarray
    pred_mean: np.ndarray
    future_set_sinkhorn: float


class PrescientStyleBaseline(nn.Module):
    def __init__(self, gene_dim: int, latent_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(gene_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, gene_dim),
        )
        self.transport = nn.Sequential(
            nn.Linear(latent_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, current_genes: torch.Tensor, current_time: torch.Tensor, future_time: torch.Tensor):
        z = self.encoder(current_genes)
        dt = (future_time - current_time).unsqueeze(-1).unsqueeze(-1)
        t = current_time.unsqueeze(-1).unsqueeze(-1)
        time_feat = torch.cat([t.expand_as(dt), dt], dim=-1)
        transport_input = torch.cat([z, time_feat.expand(z.shape[0], z.shape[1], 2)], dim=-1)
        z_future = z + self.transport(transport_input)
        pred_future = self.decoder(z_future)
        recon = self.decoder(z)
        return pred_future, recon


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_dir', default='dataset/raw')
    p.add_argument('--n_hvg', type=int, default=256)
    p.add_argument('--token_budget', type=int, default=256)
    p.add_argument('--history_frames', type=int, default=1)
    p.add_argument('--dt_minutes', type=float, default=40.0)
    p.add_argument('--time_bin_minutes', type=float, default=40.0)
    p.add_argument('--min_cells_per_snapshot', type=int, default=64)
    p.add_argument('--species_filter', default='C.elegans')
    p.add_argument('--min_umi', type=int, default=0)
    p.add_argument('--eval_fraction', type=float, default=0.2)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--latent_dim', type=int, default=128)
    p.add_argument('--hidden_dim', type=int, default=256)
    p.add_argument('--recon_weight', type=float, default=0.05)
    p.add_argument('--mean_weight', type=float, default=0.2)
    p.add_argument('--gene_sinkhorn_blur', type=float, default=0.1)
    p.add_argument('--top_k_de', type=int, default=20)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output_json', default='result/gene_context/benchmark_worm_prescient_time_transition.json')
    p.add_argument('--output_csv', default='result/gene_context/benchmark_worm_prescient_time_transition_common.csv')
    return p.parse_args()


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    x = x - x.mean()
    y = y - y.mean()
    denom = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(x, y) / denom)


def sign_accuracy(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.abs(y) > 1e-8
    if not np.any(valid):
        return 0.0
    return float((np.sign(x[valid]) == np.sign(y[valid])).mean())


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.unsqueeze(-1).float()
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    return (x * mask_f).sum(dim=1) / denom


def prepare_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    n_frames = int(batch['history_frames'][0].item())
    batch['history_genes'] = torch.stack([batch[f'history_frame_{i}_genes'] for i in range(n_frames)], dim=1)
    batch['history_valid_mask'] = torch.stack([batch[f'history_frame_{i}_valid_mask'] for i in range(n_frames)], dim=1)
    batch['history_time'] = batch[f'history_frame_{n_frames - 1}_time']
    batch['history_time_bin'] = batch[f'history_frame_{n_frames - 1}_time_bin']
    return batch


def build_dataset(args):
    return Large2025WholeEmbryoDataset(
        data_dir=args.data_dir,
        n_hvg=args.n_hvg,
        token_budget=args.token_budget,
        history_frames=args.history_frames,
        dt_minutes=args.dt_minutes,
        time_bin_minutes=args.time_bin_minutes,
        min_cells_per_snapshot=args.min_cells_per_snapshot,
        split='all',
        val_fraction=0.2,
        random_seed=args.seed,
        species_filter=args.species_filter,
        min_umi=args.min_umi,
    )


def split_transition_keys(dataset, eval_fraction: float):
    keys = sorted({f"{pair.history_bins[-1]}->{pair.future_bin}" for pair in dataset.snapshot_pairs}, key=lambda x: tuple(int(p) for p in x.split('->')))
    n_eval = max(1, int(round(len(keys) * eval_fraction)))
    eval_keys = keys[-n_eval:]
    train_keys = keys[:-n_eval]
    if not train_keys:
        train_keys = keys[:-1]
        eval_keys = keys[-1:]
    return train_keys, eval_keys


def filter_indices_by_keys(dataset, keys):
    keyset = set(keys)
    keep = []
    for i, pair in enumerate(dataset.snapshot_pairs):
        key = f"{pair.history_bins[-1]}->{pair.future_bin}"
        if key in keyset:
            keep.append(i)
    return keep


def iter_loader(dataset, indices, batch_size, device, shuffle=True):
    subset = [dataset[i] for i in indices]
    loader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_large2025_whole_embryo)
    for batch in loader:
        yield prepare_batch(batch, device)


def compute_loss(model, batch, args):
    current = batch['history_genes'][:, -1]
    future = batch['future_genes']
    current_time = batch['history_time']
    future_time = batch['future_time']
    pred_future, recon = model(current, current_time, future_time)
    loss_sinkhorn = sinkhorn_divergence(pred_future, future, blur=args.gene_sinkhorn_blur)
    recon_loss = F.mse_loss(recon[batch['history_valid_mask'][:, -1]], current[batch['history_valid_mask'][:, -1]])
    future_mean = masked_mean(future, batch['future_valid_mask'])
    pred_mean = masked_mean(pred_future, batch['history_valid_mask'][:, -1])
    mean_loss = F.mse_loss(pred_mean, future_mean)
    return loss_sinkhorn + args.recon_weight * recon_loss + args.mean_weight * mean_loss


def train_model(dataset, train_idx, args):
    model = PrescientStyleBaseline(dataset.gene_dim, latent_dim=args.latent_dim, hidden_dim=args.hidden_dim).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for _ in range(args.epochs):
        model.train()
        for batch in iter_loader(dataset, train_idx, args.batch_size, args.device, shuffle=True):
            opt.zero_grad(set_to_none=True)
            loss = compute_loss(model, batch, args)
            loss.backward()
            opt.step()
    return model


def eval_records(model, dataset, indices, args):
    model.eval()
    records = []
    with torch.no_grad():
        for batch in iter_loader(dataset, indices, args.batch_size, args.device, shuffle=False):
            current = batch['history_genes'][:, -1]
            future = batch['future_genes']
            pred_future, _ = model(current, batch['history_time'], batch['future_time'])
            pred_np = pred_future.detach().cpu().numpy().astype(np.float32)
            future_np = future.detach().cpu().numpy().astype(np.float32)
            current_np = current.detach().cpu().numpy().astype(np.float32)
            cur_valid = batch['history_valid_mask'][:, -1].detach().cpu().numpy().astype(bool)
            fut_valid = batch['future_valid_mask'].detach().cpu().numpy().astype(bool)
            for i in range(pred_np.shape[0]):
                split_key = f"{int(batch['history_time_bin'][i].item())}->{int(batch['future_time_bin'][i].item())}"
                cur_mean = current_np[i][cur_valid[i]].mean(axis=0)
                fut_mean = future_np[i][fut_valid[i]].mean(axis=0)
                pred_mean = pred_np[i][cur_valid[i]].mean(axis=0)
                set_sink = float(sinkhorn_divergence(pred_future[i:i+1], future[i:i+1], blur=args.gene_sinkhorn_blur).item())
                records.append(EvalRecord(split_key, cur_mean, fut_mean, pred_mean, set_sink))
    return records


def persistence_records(dataset, indices, args):
    records = []
    for batch in iter_loader(dataset, indices, args.batch_size, args.device, shuffle=False):
        current = batch['history_genes'][:, -1].detach().cpu().numpy().astype(np.float32)
        future = batch['future_genes'].detach().cpu().numpy().astype(np.float32)
        cur_valid = batch['history_valid_mask'][:, -1].detach().cpu().numpy().astype(bool)
        fut_valid = batch['future_valid_mask'].detach().cpu().numpy().astype(bool)
        persist_current = batch['history_genes'][:, -1]
        for i in range(current.shape[0]):
            split_key = f"{int(batch['history_time_bin'][i].item())}->{int(batch['future_time_bin'][i].item())}"
            cur_mean = current[i][cur_valid[i]].mean(axis=0)
            fut_mean = future[i][fut_valid[i]].mean(axis=0)
            set_sink = float(sinkhorn_divergence(persist_current[i:i+1], batch['future_genes'][i:i+1], blur=args.gene_sinkhorn_blur).item())
            records.append(EvalRecord(split_key, cur_mean, fut_mean, cur_mean.copy(), set_sink))
    return records


def compute_top_de_idx(records, top_k: int):
    current = np.stack([r.current_mean for r in records], axis=0)
    future = np.stack([r.future_mean for r in records], axis=0)
    score = np.abs(future - current).mean(axis=0)
    top_k = max(1, min(top_k, score.shape[0]))
    return np.argsort(score)[-top_k:]


def compute_metrics(records, top_de_idx):
    current = np.stack([r.current_mean for r in records], axis=0)
    future = np.stack([r.future_mean for r in records], axis=0)
    pred = np.stack([r.pred_mean for r in records], axis=0)
    true_delta = future - current
    pred_delta = pred - current
    top_true = true_delta[:, top_de_idx]
    top_pred = pred_delta[:, top_de_idx]
    return {
        'future_set_sinkhorn': float(np.mean([r.future_set_sinkhorn for r in records])),
        'mean_gene_mse': float(np.mean((pred - future) ** 2)),
        'mean_gene_pearson': safe_pearson(pred, future),
        'delta_gene_mse': float(np.mean((pred_delta - true_delta) ** 2)),
        'delta_gene_pearson': safe_pearson(pred_delta, true_delta),
        'top_de_delta_pearson': safe_pearson(top_pred, top_true),
        'top_de_sign_acc': sign_accuracy(top_pred, top_true),
    }


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = build_dataset(args)
    train_keys, eval_keys = split_transition_keys(dataset, args.eval_fraction)
    train_idx = filter_indices_by_keys(dataset, train_keys)
    eval_idx = filter_indices_by_keys(dataset, eval_keys)
    model = train_model(dataset, train_idx, args)
    model_records = eval_records(model, dataset, eval_idx, args)
    persist_records = persistence_records(dataset, eval_idx, args)
    top_de_idx = compute_top_de_idx(model_records, args.top_k_de)
    results = {
        'split': {'train_keys': train_keys, 'eval_keys': eval_keys},
        'n_train': len(train_idx),
        'n_eval': len(eval_idx),
        'metrics': {
            'prescient_style': compute_metrics(model_records, top_de_idx),
            'persistence': compute_metrics(persist_records, top_de_idx),
        },
    }
    out_json = Path(args.output_json)
    out_csv = Path(args.output_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['method', 'future_set_sinkhorn', 'mean_gene_mse', 'mean_gene_pearson', 'delta_gene_mse', 'delta_gene_pearson', 'top_de_delta_pearson', 'top_de_sign_acc'])
        writer.writeheader()
        for method, metrics in results['metrics'].items():
            writer.writerow({'method': method, **metrics})
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
