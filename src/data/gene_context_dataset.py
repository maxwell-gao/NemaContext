"""Multi-cell gene-context dataset built from real transcriptome time windows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import torch
from scipy import sparse
from torch.utils.data import Dataset


@dataclass(frozen=True)
class _TimePair:
    current_center: float
    future_center: float
    current_indices: np.ndarray
    future_indices: np.ndarray


class GeneContextDataset(Dataset):
    """Pseudo-embryo multi-cell contexts from transcriptome time windows.

    The active model path uses:
    - real transcriptome observations,
    - embryo time as a condition,
    - lineage names only for weak future-state pairing and event labels.
    """

    def __init__(
        self,
        h5ad_path: str | Path,
        n_hvg: int = 256,
        context_size: int = 64,
        dt_minutes: float = 20.0,
        time_window_minutes: float = 10.0,
        samples_per_pair: int = 4,
        min_cells_per_window: int = 32,
        split: str = "train",
        val_fraction: float = 0.2,
        random_seed: int = 0,
    ):
        self.h5ad_path = Path(h5ad_path)
        self.context_size = context_size
        self.dt_minutes = dt_minutes
        self.time_window_minutes = time_window_minutes
        self.samples_per_pair = samples_per_pair
        self.min_cells_per_window = min_cells_per_window
        self.split = split
        self.random_seed = random_seed

        adata = ad.read_h5ad(self.h5ad_path, backed="r")
        try:
            self._prepare_features(adata, n_hvg)
            self._build_pairs(val_fraction)
        finally:
            if getattr(adata, "file", None) is not None:
                adata.file.close()

    def _prepare_features(self, adata: ad.AnnData, n_hvg: int):
        hvg_indices = self._select_hvg_indices(adata, n_hvg)
        expr_view = adata[:, hvg_indices]
        expr = expr_view.layers["log1p"] if "log1p" in expr_view.layers else expr_view.X
        expr = expr.to_memory() if hasattr(expr, "to_memory") else expr
        if sparse.issparse(expr):
            expr = expr.toarray()
        expr = np.asarray(expr, dtype=np.float32)

        self.gene_mean = expr.mean(axis=0).astype(np.float32)
        self.gene_std = np.clip(expr.std(axis=0), a_min=1e-6, a_max=None).astype(
            np.float32
        )
        self.genes = ((expr - self.gene_mean) / self.gene_std).astype(np.float32)
        self.gene_dim = self.genes.shape[1]

        self.times = np.asarray(adata.obs["embryo_time_min"], dtype=np.float32)
        self.lineages = np.asarray(adata.obs["lineage_complete"].astype(str))
        self.has_spatial = np.asarray(adata.obs.get("has_spatial", False), dtype=bool)

        time_min = float(np.nanmin(self.times))
        time_max = float(np.nanmax(self.times))
        time_range = max(1e-6, time_max - time_min)
        self.time_min = time_min
        self.time_max = time_max
        self.normalized_times = ((self.times - time_min) / time_range).astype(np.float32)

    @staticmethod
    def _select_hvg_indices(adata: ad.AnnData, n_hvg: int) -> np.ndarray:
        if "highly_variable_rank" in adata.var.columns:
            ranks = np.asarray(adata.var["highly_variable_rank"], dtype=float)
            ranks[np.isnan(ranks)] = np.inf
            return np.argsort(ranks)[:n_hvg]
        if "highly_variable" in adata.var.columns:
            hv_idx = np.flatnonzero(np.asarray(adata.var["highly_variable"], dtype=bool))
            if len(hv_idx) >= n_hvg:
                return hv_idx[:n_hvg]

        fallback = np.arange(min(n_hvg, adata.n_vars))
        return fallback

    def _build_pairs(self, val_fraction: float):
        start = self.time_min + self.time_window_minutes / 2.0
        stop = self.time_max - self.dt_minutes - self.time_window_minutes / 2.0
        centers = np.arange(start, stop + 1e-6, self.dt_minutes, dtype=np.float32)

        pairs: list[_TimePair] = []
        half_window = self.time_window_minutes / 2.0
        for current_center in centers:
            future_center = float(current_center + self.dt_minutes)
            current_mask = np.abs(self.times - current_center) <= half_window
            future_mask = np.abs(self.times - future_center) <= half_window
            current_indices = np.flatnonzero(current_mask)
            future_indices = np.flatnonzero(future_mask)
            if (
                len(current_indices) >= self.min_cells_per_window
                and len(future_indices) >= self.min_cells_per_window
            ):
                pairs.append(
                    _TimePair(
                        current_center=float(current_center),
                        future_center=future_center,
                        current_indices=current_indices,
                        future_indices=future_indices,
                    )
                )

        split_idx = max(1, int(round(len(pairs) * (1.0 - val_fraction))))
        if self.split == "train":
            self.time_pairs = pairs[:split_idx]
        elif self.split == "val":
            self.time_pairs = pairs[split_idx:]
        else:
            self.time_pairs = pairs

    def __len__(self) -> int:
        return len(self.time_pairs) * self.samples_per_pair

    @staticmethod
    def _daughter_names(parent: str) -> tuple[str, str, str, str]:
        return (f"{parent}a", f"{parent}p", f"{parent}l", f"{parent}r")

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pair = self.time_pairs[idx % len(self.time_pairs)]
        sample_seed = self.random_seed + idx
        rng = np.random.default_rng(sample_seed)

        current_take = min(self.context_size, len(pair.current_indices))
        current_indices = rng.choice(pair.current_indices, size=current_take, replace=False)

        current_genes = self.genes[current_indices]
        current_times = self.normalized_times[current_indices]
        current_lineages = self.lineages[current_indices]

        future_lineages = self.lineages[pair.future_indices]
        future_genes = self.genes[pair.future_indices]
        future_times = self.times[pair.future_indices]

        exact_lookup: dict[str, list[int]] = {}
        daughter_lookup: dict[str, list[int]] = {}
        for local_idx, lineage in enumerate(future_lineages):
            exact_lookup.setdefault(lineage, []).append(local_idx)
            if len(lineage) > 1:
                parent = lineage[:-1]
                daughter_lookup.setdefault(parent, []).append(local_idx)

        target_genes = np.zeros_like(current_genes, dtype=np.float32)
        match_mask = np.zeros(current_take, dtype=bool)
        split_target = np.zeros(current_take, dtype=np.float32)
        del_target = np.zeros(current_take, dtype=np.float32)

        for i, lineage in enumerate(current_lineages):
            exact_matches = exact_lookup.get(lineage, [])
            if exact_matches:
                candidate_times = future_times[exact_matches]
                best_local = exact_matches[
                    int(np.argmin(np.abs(candidate_times - pair.future_center)))
                ]
                target_genes[i] = future_genes[best_local]
                match_mask[i] = True
                continue

            daughter_matches = [
                j
                for j in daughter_lookup.get(lineage, [])
                if future_lineages[j] in self._daughter_names(lineage)
            ]
            if len(daughter_matches) >= 2:
                split_target[i] = 1.0
            else:
                del_target[i] = 1.0

        return {
            "genes": torch.from_numpy(current_genes),
            "target_genes": torch.from_numpy(target_genes),
            "match_mask": torch.from_numpy(match_mask),
            "split_target": torch.from_numpy(split_target),
            "del_target": torch.from_numpy(del_target),
            "time": torch.tensor(
                (pair.current_center - self.time_min) / max(1e-6, self.time_max - self.time_min),
                dtype=torch.float32,
            ),
            "future_time": torch.tensor(
                (pair.future_center - self.time_min) / max(1e-6, self.time_max - self.time_min),
                dtype=torch.float32,
            ),
            "token_times": torch.from_numpy(current_times),
            "valid_mask": torch.ones(current_take, dtype=torch.bool),
        }


def collate_gene_context(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_len = max(item["genes"].shape[0] for item in batch)
    gene_dim = batch[0]["genes"].shape[1]

    genes = torch.zeros(len(batch), max_len, gene_dim)
    target_genes = torch.zeros(len(batch), max_len, gene_dim)
    match_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    split_target = torch.zeros(len(batch), max_len)
    del_target = torch.zeros(len(batch), max_len)
    token_times = torch.zeros(len(batch), max_len)
    valid_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    time = torch.zeros(len(batch))
    future_time = torch.zeros(len(batch))

    for i, item in enumerate(batch):
        n = item["genes"].shape[0]
        genes[i, :n] = item["genes"]
        target_genes[i, :n] = item["target_genes"]
        match_mask[i, :n] = item["match_mask"]
        split_target[i, :n] = item["split_target"]
        del_target[i, :n] = item["del_target"]
        token_times[i, :n] = item["token_times"]
        valid_mask[i, :n] = item["valid_mask"]
        time[i] = item["time"]
        future_time[i] = item["future_time"]

    return {
        "genes": genes,
        "target_genes": target_genes,
        "match_mask": match_mask,
        "split_target": split_target,
        "del_target": del_target,
        "token_times": token_times,
        "valid_mask": valid_mask,
        "time": time,
        "future_time": future_time,
    }
