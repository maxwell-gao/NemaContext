"""Raw Large2025 whole-embryo lineage-first video dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from torch.utils.data import Dataset

from .builder.expression_loader import ExpressionLoader
from .builder.lineage_encoder import LineageEncoder


@dataclass(frozen=True)
class _SnapshotPair:
    history_bins: tuple[int, ...]
    future_bin: int


class Large2025WholeEmbryoDataset(Dataset):
    """Whole-embryo time-binned snapshots from raw Large2025 with lineage features."""

    FOUNDER_VOCAB = [
        "unknown",
        "AB",
        "MS",
        "E",
        "C",
        "D",
        "P4",
        "P0",
        "P1",
        "P2",
        "P3",
        "EMS",
        "Z2",
        "Z3",
        "other",
    ]

    def __init__(
        self,
        data_dir: str | Path = "dataset/raw",
        n_hvg: int = 256,
        token_budget: int = 256,
        history_frames: int = 1,
        dt_minutes: float = 40.0,
        time_bin_minutes: float = 40.0,
        min_cells_per_snapshot: int = 64,
        split: str = "train",
        val_fraction: float = 0.2,
        random_seed: int = 0,
        species_filter: str | None = "C.elegans",
        min_umi: int = 0,
    ):
        if token_budget < 1:
            raise ValueError("token_budget must be >= 1")
        if history_frames < 1:
            raise ValueError("history_frames must be >= 1")
        if split not in {"train", "val", "all"}:
            raise ValueError("split must be one of: train, val, all")
        self.data_dir = Path(data_dir)
        self.n_hvg = n_hvg
        self.token_budget = token_budget
        self.history_frames = history_frames
        self.dt_minutes = float(dt_minutes)
        self.time_bin_minutes = float(time_bin_minutes)
        self.min_cells_per_snapshot = int(min_cells_per_snapshot)
        self.split = split
        self.val_fraction = float(val_fraction)
        self.random_seed = int(random_seed)
        self.species_filter = species_filter
        self.min_umi = int(min_umi)
        self.lineage_encoder = LineageEncoder(data_dir=str(self.data_dir))
        self._founder_to_id = {
            founder: idx for idx, founder in enumerate(self.FOUNDER_VOCAB)
        }

        self._load_raw_large2025()
        self._build_snapshot_pairs()

    def _load_raw_large2025(self) -> None:
        loader = ExpressionLoader(data_dir=str(self.data_dir))
        expr_matrix, cell_df, gene_df = loader.load_large2025(
            species_filter=self.species_filter,
            min_umi=self.min_umi,
        )
        if not sparse.issparse(expr_matrix):
            expr_matrix = sparse.csr_matrix(expr_matrix)
        expr_matrix = expr_matrix.tocsr().astype(np.float32)

        library_size = np.asarray(expr_matrix.sum(axis=1)).ravel().astype(np.float32)
        library_size = np.clip(library_size, a_min=1.0, a_max=None)
        normalized = expr_matrix.multiply((1e4 / library_size)[:, None]).tocsr()
        normalized.data = np.log1p(normalized.data)

        mean = np.asarray(normalized.mean(axis=0)).ravel()
        squared = normalized.copy()
        squared.data **= 2
        mean_sq = np.asarray(squared.mean(axis=0)).ravel()
        variance = np.clip(mean_sq - mean**2, a_min=0.0, a_max=None)
        hvg_idx = np.argsort(-variance)[: min(self.n_hvg, normalized.shape[1])]

        genes = normalized[:, hvg_idx].toarray().astype(np.float32)
        gene_mean = genes.mean(axis=0).astype(np.float32)
        gene_std = np.clip(genes.std(axis=0), a_min=1e-6, a_max=None).astype(np.float32)
        self.genes = ((genes - gene_mean) / gene_std).astype(np.float32)
        self.gene_mean = gene_mean
        self.gene_std = gene_std
        self.gene_dim = self.genes.shape[1]
        self.hvg_indices = hvg_idx.astype(np.int64)
        self.gene_names = gene_df.iloc[hvg_idx, 0].astype(str).to_list()

        times = np.asarray(cell_df["smoothed_embryo_time"], dtype=np.float32)
        valid_time = np.isfinite(times) & (times >= 0.0)
        self.valid_cell_mask = valid_time

        self.genes = self.genes[valid_time]
        self.times = times[valid_time]
        self.time_bins = (
            np.floor(self.times / self.time_bin_minutes).astype(np.int32)
            * int(round(self.time_bin_minutes))
        )

        cell_df = cell_df.loc[valid_time].reset_index(drop=True)
        self.barcodes = np.asarray(cell_df["barcode"].astype(str))
        self.cell_types = np.asarray(cell_df["cell_type"].astype(str))
        self.packer_cell_types = np.asarray(cell_df["packer_cell_type"].astype(str))
        self.lineages = np.asarray(cell_df["lineage_complete"].astype(str))

        parsed = [self.lineage_encoder.parse_lineage(lin) for lin in self.lineages.tolist()]
        self.lineage_valid = np.asarray([item is not None for item in parsed], dtype=bool)
        self.lineage_binary = self.lineage_encoder.encode_batch(self.lineages.tolist()).astype(np.float32)

        founder_ids = np.zeros(len(parsed), dtype=np.int64)
        lineage_depth = np.full(len(parsed), -1.0, dtype=np.float32)
        lineage_sort_key = np.empty(len(parsed), dtype=object)
        for i, item in enumerate(parsed):
            if item is None:
                founder_ids[i] = self._founder_to_id["unknown"]
                lineage_sort_key[i] = "zzz_invalid"
                continue
            founder = item["founder"]
            founder_ids[i] = self._founder_to_id.get(founder, self._founder_to_id["other"])
            lineage_depth[i] = float(item["depth"])
            lineage_sort_key[i] = item["full_path"]
        self.founder_ids = founder_ids
        self.lineage_depth = lineage_depth
        self.lineage_sort_key = lineage_sort_key

        max_time = max(float(np.nanmax(self.times)), 1.0)
        self.normalized_times = (self.times / max_time).astype(np.float32)

        self.snapshot_bins = sorted(
            int(bin_id)
            for bin_id, count in zip(*np.unique(self.time_bins, return_counts=True))
            if count >= self.min_cells_per_snapshot
        )
        self.bin_to_indices = {
            int(bin_id): np.flatnonzero(self.time_bins == bin_id)
            for bin_id in self.snapshot_bins
        }

    def _history_bins(self, current_bin: int) -> tuple[int, ...]:
        step = int(round(self.dt_minutes))
        return tuple(current_bin - step * offset for offset in range(self.history_frames - 1, -1, -1))

    def _build_snapshot_pairs(self) -> None:
        available = set(self.snapshot_bins)
        pairs: list[_SnapshotPair] = []
        step = int(round(self.dt_minutes))
        for current_bin in self.snapshot_bins:
            future_bin = current_bin + step
            history_bins = self._history_bins(current_bin)
            if future_bin not in available:
                continue
            if not all(bin_id in available for bin_id in history_bins):
                continue
            pairs.append(_SnapshotPair(history_bins=history_bins, future_bin=future_bin))

        rng = np.random.default_rng(self.random_seed)
        if pairs:
            order = rng.permutation(len(pairs))
            pairs = [pairs[i] for i in order]

        split_idx = max(1, int(round(len(pairs) * (1.0 - self.val_fraction)))) if pairs else 0
        if self.split == "train":
            self.snapshot_pairs = pairs[:split_idx]
        elif self.split == "val":
            self.snapshot_pairs = pairs[split_idx:]
        else:
            self.snapshot_pairs = pairs

    def __len__(self) -> int:
        return len(self.snapshot_pairs)

    def _region_label(self, idx: int) -> str:
        packer = self.packer_cell_types[idx]
        if packer and packer != "unannotated":
            return packer
        cell_type = self.cell_types[idx]
        if cell_type and cell_type != "unassigned":
            return cell_type
        return "unknown"

    def _snapshot_sort_key(self, idx: int) -> tuple:
        return (
            0 if self.lineage_valid[idx] else 1,
            self.founder_ids[idx],
            -float(self.lineage_depth[idx]),
            str(self.lineage_sort_key[idx]),
            str(self.barcodes[idx]),
        )

    def _select_snapshot_tokens(self, indices: np.ndarray) -> np.ndarray:
        grouped: dict[str, list[int]] = {}
        for idx in indices.tolist():
            grouped.setdefault(self._region_label(idx), []).append(int(idx))
        group_names = sorted(grouped)
        ordered_groups = {
            name: sorted(grouped[name], key=self._snapshot_sort_key)
            for name in group_names
        }

        selected: list[int] = []
        offsets = {name: 0 for name in group_names}
        while len(selected) < self.token_budget:
            progressed = False
            for name in group_names:
                pos = offsets[name]
                bucket = ordered_groups[name]
                if pos >= len(bucket):
                    continue
                selected.append(bucket[pos])
                offsets[name] = pos + 1
                progressed = True
                if len(selected) >= self.token_budget:
                    break
            if not progressed:
                break
        return np.asarray(selected, dtype=np.int64)

    def _build_snapshot_view(self, bin_id: int) -> dict[str, torch.Tensor]:
        indices = self._select_snapshot_tokens(self.bin_to_indices[bin_id])
        n_valid = len(indices)
        genes = np.zeros((self.token_budget, self.gene_dim), dtype=np.float32)
        token_times = np.zeros((self.token_budget,), dtype=np.float32)
        valid_mask = np.zeros((self.token_budget,), dtype=bool)
        lineage_binary = np.full((self.token_budget, self.lineage_binary.shape[1]), -1.0, dtype=np.float32)
        founder_ids = np.zeros((self.token_budget,), dtype=np.int64)
        lineage_depth = np.full((self.token_budget,), -1.0, dtype=np.float32)
        lineage_valid = np.zeros((self.token_budget,), dtype=bool)
        token_rank = np.arange(self.token_budget, dtype=np.int64)

        if n_valid > 0:
            genes[:n_valid] = self.genes[indices]
            token_times[:n_valid] = self.normalized_times[indices]
            valid_mask[:n_valid] = True
            lineage_binary[:n_valid] = self.lineage_binary[indices]
            founder_ids[:n_valid] = self.founder_ids[indices]
            lineage_depth[:n_valid] = self.lineage_depth[indices]
            lineage_valid[:n_valid] = self.lineage_valid[indices]

        view = {
            "genes": torch.from_numpy(genes),
            "token_times": torch.from_numpy(token_times),
            "valid_mask": torch.from_numpy(valid_mask),
            "lineage_binary": torch.from_numpy(lineage_binary),
            "founder_ids": torch.from_numpy(founder_ids),
            "lineage_depth": torch.from_numpy(lineage_depth),
            "lineage_valid": torch.from_numpy(lineage_valid),
            "token_rank": torch.from_numpy(token_rank),
            "time": torch.tensor(float(bin_id), dtype=torch.float32),
            "time_bin": torch.tensor(int(bin_id), dtype=torch.long),
        }
        return view

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pair = self.snapshot_pairs[idx]
        output: dict[str, torch.Tensor] = {
            "history_frames": torch.tensor(self.history_frames, dtype=torch.long),
        }
        for i, bin_id in enumerate(pair.history_bins):
            view = self._build_snapshot_view(bin_id)
            prefix = f"history_frame_{i}"
            for key, value in view.items():
                output[f"{prefix}_{key}"] = value
        future_view = self._build_snapshot_view(pair.future_bin)
        output["future_genes"] = future_view["genes"]
        output["future_valid_mask"] = future_view["valid_mask"]
        output["future_mean_gene"] = future_view["genes"][future_view["valid_mask"]].mean(dim=0)
        output["future_time"] = future_view["time"]
        output["future_time_bin"] = future_view["time_bin"]
        return output


def collate_large2025_whole_embryo(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = batch[0].keys()
    out: dict[str, torch.Tensor] = {}
    for key in keys:
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            out[key] = torch.stack(values, dim=0)
        else:
            raise TypeError(f"Unsupported batch field type for key={key!r}")
    return out
