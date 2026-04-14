#!/usr/bin/env python3
"""Worm-native benchmark for Large2025 lineage-first dynamics models."""

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
from torch.utils.data import DataLoader

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.branching_flows.emergent_loss import sinkhorn_divergence  # noqa: E402
from src.branching_flows.gene_context import LineageWholeEmbryoModel  # noqa: E402
from src.data.gene_context_dataset import Large2025WholeEmbryoDataset, collate_large2025_whole_embryo  # noqa: E402


UNKNOWN_FOUNDER_ID = 0
UNKNOWN_REGION_ID = 0


@dataclass(frozen=True)
class EvalRecord:
    split_key: str
    current_mean: np.ndarray
    future_mean: np.ndarray
    pred_mean: np.ndarray
    future_set_sinkhorn: float


@dataclass(frozen=True)
class GroupEvalRecord:
    split_key: str
    group_key: str
    current_mean: np.ndarray
    future_mean: np.ndarray
    pred_mean: np.ndarray


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default='checkpoints/large2025_lineage_stage1/raw_large2025_stage1_dyn_e20_b4_t128/best.pt')
    p.add_argument('--data_dir', default='dataset/raw')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--split_mode', choices=['time_transition', 'founder_group', 'region_group'], default='time_transition')
    p.add_argument('--eval_fraction', type=float, default=0.2)
    p.add_argument('--top_k_de', type=int, default=20)
    p.add_argument('--gene_sinkhorn_blur', type=float, default=0.1)
    p.add_argument('--output_json', default='result/gene_context/benchmark_worm_dynamics.json')
    p.add_argument('--output_csv', default='result/gene_context/benchmark_worm_dynamics.csv')
    p.add_argument('--output_common_csv', default=None)
    p.add_argument('--output_structure_csv', default=None)
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
    x = np.asarray(x)
    y = np.asarray(y)
    valid = np.abs(y) > 1e-8
    if not np.any(valid):
        return 0.0
    return float((np.sign(x[valid]) == np.sign(y[valid])).mean())


def prepare_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    n_frames = int(batch['history_frames'][0].item())
    stack_fields = ['genes', 'token_times', 'valid_mask', 'lineage_binary', 'founder_ids', 'lineage_depth', 'lineage_valid', 'region_ids', 'token_rank']
    for field in stack_fields:
        batch[f'history_{field}'] = torch.stack([batch[f'history_frame_{i}_{field}'] for i in range(n_frames)], dim=1)
    batch['history_time'] = batch[f'history_frame_{n_frames - 1}_time']
    batch['history_time_bin'] = batch[f'history_frame_{n_frames - 1}_time_bin']
    return batch


def load_backbone(checkpoint_path: Path, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    model = LineageWholeEmbryoModel(
        gene_dim=int(checkpoint['gene_dim']),
        context_size=int(config.get('token_budget', 256)),
        history_frames=int(config.get('history_frames', 1)),
        lineage_binary_dim=20,
        founder_vocab_size=15,
        d_model=int(config.get('d_model', 256)),
        n_heads=int(config.get('n_heads', 8)),
        n_spatial_layers=int(config.get('n_spatial_layers', 2)),
        n_temporal_layers=int(config.get('n_temporal_layers', 4)),
        n_decoder_layers=int(config.get('n_decoder_layers', 2)),
        head_dim=int(config.get('head_dim', 32)),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, config


def build_dataset(config: dict, data_dir: str):
    return Large2025WholeEmbryoDataset(
        data_dir=data_dir,
        n_hvg=int(config.get('n_hvg', 256)),
        token_budget=int(config.get('token_budget', 256)),
        history_frames=int(config.get('history_frames', 1)),
        dt_minutes=float(config.get('dt_minutes', 40.0)),
        time_bin_minutes=float(config.get('time_bin_minutes', 40.0)),
        min_cells_per_snapshot=int(config.get('min_cells_per_snapshot', 64)),
        split='all',
        val_fraction=float(config.get('val_fraction', 0.2)),
        random_seed=int(config.get('seed', 0)),
        species_filter=config.get('species_filter', 'C.elegans'),
        min_umi=int(config.get('min_umi', 0)),
    )


def mean_by_mask(values: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    if not np.any(mask):
        return None
    return values[mask].mean(axis=0)


def select_eval_keys(keys: list[str], eval_fraction: float) -> tuple[list[str], list[str]]:
    if not keys:
        return [], []
    n_eval = max(1, int(round(len(keys) * eval_fraction)))
    eval_keys = keys[-n_eval:]
    train_keys = keys[:-n_eval]
    if not train_keys:
        train_keys = keys[:-1]
        eval_keys = keys[-1:]
    return train_keys, eval_keys


def compute_top_de_idx(records: list[EvalRecord], top_k: int) -> np.ndarray:
    current = np.stack([r.current_mean for r in records], axis=0)
    future = np.stack([r.future_mean for r in records], axis=0)
    mean_abs_delta = np.abs(future - current).mean(axis=0)
    top_k = max(1, min(top_k, mean_abs_delta.shape[0]))
    return np.argsort(mean_abs_delta)[-top_k:]


def compute_metrics(records: list[EvalRecord], top_de_idx: np.ndarray) -> dict[str, float]:
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


def aggregate_group_metric(records: list[GroupEvalRecord]) -> float:
    if not records:
        return 0.0
    pred = np.stack([r.pred_mean for r in records], axis=0)
    true = np.stack([r.future_mean for r in records], axis=0)
    return safe_pearson(pred, true)


def collect_records(dataset, model, args):
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_large2025_whole_embryo)
    whole_records: list[EvalRecord] = []
    persistence_records: list[EvalRecord] = []
    founder_records: list[GroupEvalRecord] = []
    founder_persist_records: list[GroupEvalRecord] = []
    region_records: list[GroupEvalRecord] = []
    region_persist_records: list[GroupEvalRecord] = []

    with torch.no_grad():
        for batch in loader:
            batch = prepare_batch(batch, args.device)
            out = model(
                genes=batch['history_genes'],
                time=batch['history_time'],
                future_time=batch['future_time'],
                token_times=batch['history_token_times'],
                valid_mask=batch['history_valid_mask'],
                lineage_binary=batch['history_lineage_binary'],
                founder_ids=batch['history_founder_ids'],
                lineage_depth=batch['history_lineage_depth'],
                lineage_valid=batch['history_lineage_valid'],
                token_rank=batch['history_token_rank'],
            )
            current = batch['history_genes'][:, -1].detach().cpu().numpy().astype(np.float32)
            current_valid = batch['history_valid_mask'][:, -1].detach().cpu().numpy().astype(bool)
            current_founder = batch['history_founder_ids'][:, -1].detach().cpu().numpy().astype(np.int64)
            current_region = batch['history_region_ids'][:, -1].detach().cpu().numpy().astype(np.int64)
            future = batch['future_genes'].detach().cpu().numpy().astype(np.float32)
            future_valid = batch['future_valid_mask'].detach().cpu().numpy().astype(bool)
            future_founder = batch['future_founder_ids'].detach().cpu().numpy().astype(np.int64)
            future_region = batch['future_region_ids'].detach().cpu().numpy().astype(np.int64)
            pred = out.pred_future_genes.detach().cpu().numpy().astype(np.float32)
            batch_size = pred.shape[0]

            for i in range(batch_size):
                split_key = f"{int(batch['history_time_bin'][i].item())}->{int(batch['future_time_bin'][i].item())}"
                cur_mean = current[i][current_valid[i]].mean(axis=0)
                fut_mean = future[i][future_valid[i]].mean(axis=0)
                pred_mean = pred[i][future_valid[i]].mean(axis=0)
                model_sinkhorn = float(sinkhorn_divergence(out.pred_future_genes[i:i+1], batch['future_genes'][i:i+1], blur=args.gene_sinkhorn_blur).item())
                persist_current = batch['history_genes'][i:i+1, -1]
                persist_sinkhorn = float(sinkhorn_divergence(persist_current, batch['future_genes'][i:i+1], blur=args.gene_sinkhorn_blur).item())
                whole_records.append(EvalRecord(split_key, cur_mean, fut_mean, pred_mean, model_sinkhorn))
                persistence_records.append(EvalRecord(split_key, cur_mean, fut_mean, cur_mean.copy(), persist_sinkhorn))

                valid_founders = sorted(set(future_founder[i][future_valid[i]].tolist()))
                for founder_id in valid_founders:
                    if founder_id == UNKNOWN_FOUNDER_ID:
                        continue
                    fmask_future = future_valid[i] & (future_founder[i] == founder_id)
                    fmask_current = current_valid[i] & (current_founder[i] == founder_id)
                    cur_group = mean_by_mask(current[i], fmask_current)
                    fut_group = mean_by_mask(future[i], fmask_future)
                    pred_group = mean_by_mask(pred[i], fmask_future)
                    if cur_group is None or fut_group is None or pred_group is None:
                        continue
                    founder_records.append(GroupEvalRecord(split_key, f'founder:{founder_id}', cur_group, fut_group, pred_group))
                    founder_persist_records.append(GroupEvalRecord(split_key, f'founder:{founder_id}', cur_group, fut_group, cur_group.copy()))

                valid_regions = sorted(set(future_region[i][future_valid[i]].tolist()))
                for region_id in valid_regions:
                    if region_id == UNKNOWN_REGION_ID:
                        continue
                    rmask_future = future_valid[i] & (future_region[i] == region_id)
                    rmask_current = current_valid[i] & (current_region[i] == region_id)
                    cur_group = mean_by_mask(current[i], rmask_current)
                    fut_group = mean_by_mask(future[i], rmask_future)
                    pred_group = mean_by_mask(pred[i], rmask_future)
                    if cur_group is None or fut_group is None or pred_group is None:
                        continue
                    region_records.append(GroupEvalRecord(split_key, f'region:{region_id}', cur_group, fut_group, pred_group))
                    region_persist_records.append(GroupEvalRecord(split_key, f'region:{region_id}', cur_group, fut_group, cur_group.copy()))

    return whole_records, persistence_records, founder_records, founder_persist_records, region_records, region_persist_records


def split_records(args, whole_records, founder_records, region_records):
    if args.split_mode == 'time_transition':
        keys = sorted({r.split_key for r in whole_records}, key=lambda x: tuple(int(p) for p in x.split('->')))
        train_keys, eval_keys = select_eval_keys(keys, args.eval_fraction)
        train_whole = [r for r in whole_records if r.split_key in train_keys]
        eval_whole = [r for r in whole_records if r.split_key in eval_keys]
        eval_founder = [r for r in founder_records if r.split_key in eval_keys]
        eval_region = [r for r in region_records if r.split_key in eval_keys]
        return train_whole, eval_whole, eval_founder, eval_region, {'train_keys': train_keys, 'eval_keys': eval_keys}
    if args.split_mode == 'founder_group':
        keys = sorted({r.group_key for r in founder_records})
        train_keys, eval_keys = select_eval_keys(keys, args.eval_fraction)
        train_whole = whole_records
        eval_whole = whole_records
        eval_founder = [r for r in founder_records if r.group_key in eval_keys]
        eval_region = region_records
        return train_whole, eval_whole, eval_founder, eval_region, {'train_keys': train_keys, 'eval_keys': eval_keys}
    keys = sorted({r.group_key for r in region_records})
    train_keys, eval_keys = select_eval_keys(keys, args.eval_fraction)
    train_whole = whole_records
    eval_whole = whole_records
    eval_founder = founder_records
    eval_region = [r for r in region_records if r.group_key in eval_keys]
    return train_whole, eval_whole, eval_founder, eval_region, {'train_keys': train_keys, 'eval_keys': eval_keys}


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model, config = load_backbone(Path(args.checkpoint), args.device)
    dataset = build_dataset(config, args.data_dir)
    whole_records, persistence_records, founder_records, founder_persist_records, region_records, region_persist_records = collect_records(dataset, model, args)
    train_whole, eval_whole, eval_founder, eval_region, split_info = split_records(args, whole_records, founder_records, region_records)
    _, eval_persist_whole, eval_persist_founder, eval_persist_region, _ = split_records(args, persistence_records, founder_persist_records, region_persist_records)
    top_de_idx = compute_top_de_idx(train_whole, args.top_k_de)
    results = {
        'checkpoint': args.checkpoint,
        'split_mode': args.split_mode,
        'split': split_info,
        'n_train_whole': len(train_whole),
        'n_eval_whole': len(eval_whole),
        'n_eval_founder_groups': len(eval_founder),
        'n_eval_region_groups': len(eval_region),
        'metrics': {
            'model': {
                **compute_metrics(eval_whole, top_de_idx),
                'founder_group_pseudobulk_pearson': aggregate_group_metric(eval_founder),
                'region_group_pseudobulk_pearson': aggregate_group_metric(eval_region),
            },
            'persistence': {
                **compute_metrics(eval_persist_whole, top_de_idx),
                'founder_group_pseudobulk_pearson': aggregate_group_metric(eval_persist_founder),
                'region_group_pseudobulk_pearson': aggregate_group_metric(eval_persist_region),
            },
        },
    }
    out_json = Path(args.output_json)
    out_csv = Path(args.output_csv)
    common_csv = Path(args.output_common_csv) if args.output_common_csv else out_csv.with_name(out_csv.stem + '_common.csv')
    structure_csv = Path(args.output_structure_csv) if args.output_structure_csv else out_csv.with_name(out_csv.stem + '_structure.csv')
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    common_csv.parent.mkdir(parents=True, exist_ok=True)
    structure_csv.parent.mkdir(parents=True, exist_ok=True)

    common_metrics = {}
    structure_metrics = {}
    for method, metrics in results['metrics'].items():
        common_metrics[method] = {
            'future_set_sinkhorn': metrics['future_set_sinkhorn'],
            'mean_gene_mse': metrics['mean_gene_mse'],
            'mean_gene_pearson': metrics['mean_gene_pearson'],
            'delta_gene_mse': metrics['delta_gene_mse'],
            'delta_gene_pearson': metrics['delta_gene_pearson'],
            'top_de_delta_pearson': metrics['top_de_delta_pearson'],
            'top_de_sign_acc': metrics['top_de_sign_acc'],
        }
        structure_metrics[method] = {
            'founder_group_pseudobulk_pearson': metrics['founder_group_pseudobulk_pearson'],
            'region_group_pseudobulk_pearson': metrics['region_group_pseudobulk_pearson'],
        }

    results['tables'] = {
        'common_forecasting': common_metrics,
        'structure_preservation': structure_metrics,
    }

    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)

    with open(common_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['method', 'future_set_sinkhorn', 'mean_gene_mse', 'mean_gene_pearson', 'delta_gene_mse', 'delta_gene_pearson', 'top_de_delta_pearson', 'top_de_sign_acc'])
        writer.writeheader()
        for method, metrics in common_metrics.items():
            writer.writerow({'method': method, **metrics})

    with open(structure_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['method', 'founder_group_pseudobulk_pearson', 'region_group_pseudobulk_pearson'])
        writer.writeheader()
        for method, metrics in structure_metrics.items():
            writer.writerow({'method': method, **metrics})

    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['table', 'method', 'metric', 'value'])
        writer.writeheader()
        for method, metrics in common_metrics.items():
            for metric, value in metrics.items():
                writer.writerow({'table': 'common_forecasting', 'method': method, 'metric': metric, 'value': value})
        for method, metrics in structure_metrics.items():
            for metric, value in metrics.items():
                writer.writerow({'table': 'structure_preservation', 'method': method, 'metric': metric, 'value': value})

    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
