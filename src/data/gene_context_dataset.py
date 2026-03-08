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

    MATCH_EXACT = 0
    MATCH_SPLIT = 1
    MATCH_DESCENDANT = 2
    MATCH_UNMATCHED = 3
    MATCH_UNSUPERVISED = -1

    def __init__(
        self,
        h5ad_path: str | Path,
        n_hvg: int = 256,
        context_size: int = 64,
        global_context_size: int | None = None,
        dt_minutes: float = 20.0,
        time_window_minutes: float = 10.0,
        samples_per_pair: int = 4,
        min_cells_per_window: int = 32,
        split: str = "train",
        val_fraction: float = 0.2,
        random_seed: int = 0,
        sampling_strategy: str = "random_window",
        min_spatial_cells_per_window: int = 8,
        spatial_neighbor_pool_size: int | None = None,
        min_event_positive: int = 0,
        min_anchor_event_positive: int = 0,
        min_split_positive: int = 0,
        min_del_positive: int = 0,
        min_anchor_split_positive: int = 0,
        min_anchor_del_positive: int = 0,
        delete_target_mode: str = "weak",
    ):
        self.h5ad_path = Path(h5ad_path)
        self.context_size = context_size
        default_global = max(0, (context_size - 1) // 4)
        self.global_context_size = min(
            max(0, global_context_size if global_context_size is not None else default_global),
            max(0, context_size - 1),
        )
        self.local_context_size = max(0, context_size - 1 - self.global_context_size)
        self.dt_minutes = dt_minutes
        self.time_window_minutes = time_window_minutes
        self.samples_per_pair = samples_per_pair
        self.min_cells_per_window = min_cells_per_window
        self.split = split
        self.random_seed = random_seed
        self.sampling_strategy = sampling_strategy
        self.min_spatial_cells_per_window = min_spatial_cells_per_window
        self.spatial_neighbor_pool_size = spatial_neighbor_pool_size or context_size
        self.min_event_positive = min_event_positive
        self.min_anchor_event_positive = min_anchor_event_positive
        self.min_split_positive = min_split_positive
        self.min_del_positive = min_del_positive
        self.min_anchor_split_positive = min_anchor_split_positive
        self.min_anchor_del_positive = min_anchor_del_positive
        if delete_target_mode not in {"weak", "strict"}:
            raise ValueError(
                f"Unsupported delete_target_mode: {delete_target_mode}"
            )
        self.delete_target_mode = delete_target_mode
        self._window_cache: dict[float, np.ndarray] = {}

        adata = ad.read_h5ad(self.h5ad_path, backed="r")
        try:
            self._prepare_features(adata, n_hvg)
            self._build_pairs(val_fraction)
            self._filter_time_pairs()
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
        spatial = np.asarray(adata.obsm.get("X_spatial"), dtype=np.float32)
        spatial_valid = np.isfinite(spatial).all(axis=1)
        self.spatial_coords = spatial
        self.valid_spatial = self.has_spatial & spatial_valid

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
            if self.sampling_strategy in {"spatial_neighbors", "spatial_anchor"}:
                current_spatial = np.intersect1d(
                    current_indices,
                    np.flatnonzero(self.valid_spatial),
                    assume_unique=False,
                )
                if len(current_spatial) < self.min_spatial_cells_per_window:
                    continue
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

        rng = np.random.default_rng(self.random_seed)
        if pairs:
            order = rng.permutation(len(pairs))
            pairs = [pairs[i] for i in order]

        split_idx = max(1, int(round(len(pairs) * (1.0 - val_fraction))))
        if self.split == "train":
            self.time_pairs = pairs[:split_idx]
        elif self.split == "val":
            self.time_pairs = pairs[split_idx:]
        else:
            self.time_pairs = pairs

    def _filter_time_pairs(self):
        needs_filter = any(
            threshold > 0
            for threshold in (
                self.min_event_positive,
                self.min_anchor_event_positive,
                self.min_split_positive,
                self.min_del_positive,
                self.min_anchor_split_positive,
                self.min_anchor_del_positive,
            )
        )
        if not needs_filter:
            return

        filtered_pairs: list[_TimePair] = []
        for i, _pair in enumerate(self.time_pairs):
            summary = self.summarize_time_pair(i)
            if summary["event_positive_count"] < self.min_event_positive:
                continue
            if summary["anchor_event_positive_count"] < self.min_anchor_event_positive:
                continue
            if summary["split_positive_count"] < self.min_split_positive:
                continue
            if summary["del_positive_count"] < self.min_del_positive:
                continue
            if summary["anchor_split_positive_count"] < self.min_anchor_split_positive:
                continue
            if summary["anchor_del_positive_count"] < self.min_anchor_del_positive:
                continue
            filtered_pairs.append(self.time_pairs[i])

        self.time_pairs = filtered_pairs

    def __len__(self) -> int:
        return len(self.time_pairs) * self.samples_per_pair

    @staticmethod
    def _daughter_names(parent: str) -> tuple[str, str, str, str]:
        return (f"{parent}a", f"{parent}p", f"{parent}l", f"{parent}r")

    def _build_future_lookups(
        self,
        future_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[int]], dict[str, list[int]]]:
        future_lineages = self.lineages[future_indices]
        future_genes = self.genes[future_indices]
        future_times = self.times[future_indices]

        exact_lookup: dict[str, list[int]] = {}
        descendant_lookup: dict[str, list[int]] = {}
        for local_idx, lineage in enumerate(future_lineages):
            exact_lookup.setdefault(lineage, []).append(local_idx)
            for prefix_len in range(1, len(lineage)):
                descendant_lookup.setdefault(lineage[:prefix_len], []).append(local_idx)

        return future_lineages, future_genes, future_times, exact_lookup, descendant_lookup

    def _get_window_indices(self, center: float) -> np.ndarray:
        center_key = float(center)
        cached = self._window_cache.get(center_key)
        if cached is not None:
            return cached
        half_window = self.time_window_minutes / 2.0
        mask = np.abs(self.times - center_key) <= half_window
        indices = np.flatnonzero(mask)
        self._window_cache[center_key] = indices
        return indices

    def _compute_targets(
        self,
        pair: _TimePair,
        current_indices: np.ndarray,
        anchor_only: bool = False,
    ) -> dict[str, np.ndarray]:
        current_take = len(current_indices)
        current_genes = self.genes[current_indices]
        current_lineages = self.lineages[current_indices]
        future_lineages, future_genes, future_times, exact_lookup, descendant_lookup = self._build_future_lookups(
            pair.future_indices
        )
        future2_indices = self._get_window_indices(pair.future_center + self.dt_minutes)
        future2_lineages, _, _future2_times, future2_exact_lookup, future2_descendant_lookup = (
            self._build_future_lookups(future2_indices)
        )

        target_genes = np.zeros_like(current_genes, dtype=np.float32)
        match_mask = np.zeros(current_take, dtype=bool)
        split_target = np.zeros(current_take, dtype=np.float32)
        weak_del_target = np.zeros(current_take, dtype=np.float32)
        strict_del_target = np.zeros(current_take, dtype=np.float32)
        used_exact = np.zeros(current_take, dtype=bool)
        used_descendant = np.zeros(current_take, dtype=bool)
        match_type = np.full(
            current_take,
            self.MATCH_UNSUPERVISED,
            dtype=np.int64,
        )

        for i, lineage in enumerate(current_lineages):
            if anchor_only and i != 0:
                continue
            daughter_matches = [
                j
                for j in descendant_lookup.get(lineage, [])
                if future_lineages[j] in self._daughter_names(lineage)
            ]
            if len(daughter_matches) >= 2:
                split_target[i] = 1.0
                target_genes[i] = future_genes[daughter_matches].mean(axis=0)
                match_mask[i] = True
                used_descendant[i] = True
                match_type[i] = self.MATCH_SPLIT
                continue

            exact_matches = exact_lookup.get(lineage, [])
            if exact_matches:
                candidate_times = future_times[exact_matches]
                best_local = exact_matches[
                    int(np.argmin(np.abs(candidate_times - pair.future_center)))
                ]
                target_genes[i] = future_genes[best_local]
                match_mask[i] = True
                used_exact[i] = True
                match_type[i] = self.MATCH_EXACT
                continue

            descendant_matches = descendant_lookup.get(lineage, [])
            if descendant_matches:
                candidate_times = future_times[descendant_matches]
                best_local = descendant_matches[
                    int(np.argmin(np.abs(candidate_times - pair.future_center)))
                ]
                target_genes[i] = future_genes[best_local]
                match_mask[i] = True
                used_descendant[i] = True
                match_type[i] = self.MATCH_DESCENDANT
            else:
                weak_del_target[i] = 1.0
                match_type[i] = self.MATCH_UNMATCHED

                future2_daughter_matches = [
                    j
                    for j in future2_descendant_lookup.get(lineage, [])
                    if future2_lineages[j] in self._daughter_names(lineage)
                ]
                future2_exact_matches = future2_exact_lookup.get(lineage, [])
                future2_descendant_matches = future2_descendant_lookup.get(lineage, [])
                if (
                    len(future2_daughter_matches) == 0
                    and len(future2_exact_matches) == 0
                    and len(future2_descendant_matches) == 0
                ):
                    strict_del_target[i] = 1.0

        del_target = (
            strict_del_target
            if self.delete_target_mode == "strict"
            else weak_del_target
        )

        return {
            "target_genes": target_genes,
            "match_mask": match_mask,
            "split_target": split_target,
            "del_target": del_target,
            "weak_del_target": weak_del_target,
            "strict_del_target": strict_del_target,
            "used_exact": used_exact,
            "used_descendant": used_descendant,
            "match_type": match_type,
        }

    def summarize_time_pair(
        self,
        pair_idx: int,
    ) -> dict[str, float | int]:
        pair = self.time_pairs[pair_idx]
        current_indices = pair.current_indices
        spatial_current = current_indices[self.valid_spatial[current_indices]]
        targets = self._compute_targets(pair, current_indices, anchor_only=False)

        split_target = targets["split_target"]
        del_target = targets["del_target"]
        weak_del_target = targets["weak_del_target"]
        strict_del_target = targets["strict_del_target"]
        match_mask = targets["match_mask"]
        used_exact = targets["used_exact"]
        used_descendant = targets["used_descendant"]

        summary: dict[str, float | int] = {
            "pair_idx": pair_idx,
            "current_center_min": float(pair.current_center),
            "future_center_min": float(pair.future_center),
            "dt_minutes": float(pair.future_center - pair.current_center),
            "current_count": int(len(pair.current_indices)),
            "future_count": int(len(pair.future_indices)),
            "spatial_current_count": int(len(spatial_current)),
            "match_count": int(match_mask.sum()),
            "exact_match_count": int(used_exact.sum()),
            "descendant_match_count": int(used_descendant.sum()),
            "split_positive_count": int(split_target.sum()),
            "del_positive_count": int(del_target.sum()),
            "weak_del_positive_count": int(weak_del_target.sum()),
            "strict_del_positive_count": int(strict_del_target.sum()),
        }
        summary["match_rate"] = summary["match_count"] / max(1, summary["current_count"])
        summary["split_positive_rate"] = summary["split_positive_count"] / max(
            1, summary["current_count"]
        )
        summary["del_positive_rate"] = summary["del_positive_count"] / max(
            1, summary["current_count"]
        )

        if len(spatial_current) > 0:
            spatial_targets = self._compute_targets(pair, spatial_current, anchor_only=False)
            spatial_split = spatial_targets["split_target"]
            spatial_del = spatial_targets["del_target"]
            spatial_weak_del = spatial_targets["weak_del_target"]
            spatial_strict_del = spatial_targets["strict_del_target"]
            summary["anchor_candidate_count"] = int(len(spatial_current))
            summary["anchor_split_positive_count"] = int(spatial_split.sum())
            summary["anchor_del_positive_count"] = int(spatial_del.sum())
            summary["anchor_weak_del_positive_count"] = int(spatial_weak_del.sum())
            summary["anchor_strict_del_positive_count"] = int(spatial_strict_del.sum())
            summary["anchor_split_positive_rate"] = summary["anchor_split_positive_count"] / max(
                1, summary["anchor_candidate_count"]
            )
            summary["anchor_del_positive_rate"] = summary["anchor_del_positive_count"] / max(
                1, summary["anchor_candidate_count"]
            )
        else:
            summary["anchor_candidate_count"] = 0
            summary["anchor_split_positive_count"] = 0
            summary["anchor_del_positive_count"] = 0
            summary["anchor_weak_del_positive_count"] = 0
            summary["anchor_strict_del_positive_count"] = 0
            summary["anchor_split_positive_rate"] = 0.0
            summary["anchor_del_positive_rate"] = 0.0

        summary["event_positive_count"] = (
            summary["split_positive_count"] + summary["del_positive_count"]
        )
        summary["anchor_event_positive_count"] = (
            summary["anchor_split_positive_count"] + summary["anchor_del_positive_count"]
        )
        return summary

    def _select_current_indices(
        self,
        pair: _TimePair,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if self.sampling_strategy == "random_window":
            take = min(self.context_size, len(pair.current_indices))
            return np.sort(rng.choice(pair.current_indices, size=take, replace=False))

        if self.sampling_strategy in {"spatial_neighbors", "spatial_anchor"}:
            spatial_candidates = pair.current_indices[
                self.valid_spatial[pair.current_indices]
            ]
            if len(spatial_candidates) == 0:
                take = min(self.context_size, len(pair.current_indices))
                return np.sort(
                    rng.choice(pair.current_indices, size=take, replace=False)
                )

            anchor = int(rng.choice(spatial_candidates))
            anchor_coord = self.spatial_coords[anchor]
            neighbor_coords = self.spatial_coords[spatial_candidates]
            distances = np.linalg.norm(neighbor_coords - anchor_coord, axis=1)
            order = np.argsort(distances)
            pool_size = min(self.spatial_neighbor_pool_size, len(spatial_candidates))
            local_pool = spatial_candidates[order[:pool_size]]
            take = min(self.context_size, len(local_pool))
            if take == len(local_pool):
                return np.sort(local_pool)
            selected = rng.choice(local_pool, size=take, replace=False)
            return np.sort(selected)

        raise ValueError(f"Unsupported sampling_strategy: {self.sampling_strategy}")

    def _select_anchor_context(
        self,
        pair: _TimePair,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        spatial_candidates = pair.current_indices[self.valid_spatial[pair.current_indices]]
        if len(spatial_candidates) == 0:
            raise ValueError("spatial_anchor requires spatially matched cells")

        anchor = int(rng.choice(spatial_candidates))
        anchor_coord = self.spatial_coords[anchor]
        neighbor_coords = self.spatial_coords[spatial_candidates]
        distances = np.linalg.norm(neighbor_coords - anchor_coord, axis=1)
        order = np.argsort(distances)
        pool_size = min(self.spatial_neighbor_pool_size, len(spatial_candidates))
        local_pool = spatial_candidates[order[:pool_size]]
        if anchor not in local_pool:
            local_pool = np.concatenate([[anchor], local_pool])
        local_pool = np.unique(local_pool)

        if len(local_pool) == 0:
            raise ValueError("No anchor context available")

        sorted_local = local_pool[local_pool != anchor]
        local_take = min(self.local_context_size, len(sorted_local))
        local_selected = sorted_local[:local_take]

        used = np.concatenate([[anchor], local_selected]).astype(np.int64)
        remaining = np.setdiff1d(pair.current_indices, used, assume_unique=False)

        global_take = min(self.global_context_size, len(remaining))
        if global_take > 0:
            global_selected = rng.choice(remaining, size=global_take, replace=False)
            global_selected = np.sort(global_selected.astype(np.int64))
        else:
            global_selected = np.empty(0, dtype=np.int64)

        selected = np.concatenate(
            [
                np.array([anchor], dtype=np.int64),
                local_selected.astype(np.int64),
                global_selected,
            ]
        )

        if len(selected) < self.context_size:
            filler_pool = np.setdiff1d(pair.current_indices, selected, assume_unique=False)
            fill_take = min(self.context_size - len(selected), len(filler_pool))
            if fill_take > 0:
                filler = rng.choice(filler_pool, size=fill_take, replace=False)
                filler = np.sort(filler.astype(np.int64))
                global_selected = np.concatenate([global_selected, filler])
                selected = np.concatenate([selected, filler])

        context_role = np.full(len(selected), 3, dtype=np.int64)
        context_role[0] = 1
        context_role[1 : 1 + len(local_selected)] = 2

        selected_coords = self.spatial_coords[selected]
        selected_distances = np.linalg.norm(selected_coords - anchor_coord, axis=1)
        finite_distances = selected_distances[np.isfinite(selected_distances)]
        max_distance = float(finite_distances.max()) if finite_distances.size else 1.0
        max_distance = max(max_distance, 1e-6)
        normalized_distance = np.clip(selected_distances / max_distance, 0.0, 1.0)
        normalized_distance = np.nan_to_num(normalized_distance, nan=1.0, posinf=1.0, neginf=0.0)
        distance_bucket = np.minimum((normalized_distance * 15.0).astype(np.int64), 15)
        distance_bucket[0] = 0

        return selected, context_role, distance_bucket

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pair = self.time_pairs[idx % len(self.time_pairs)]
        sample_seed = self.random_seed + idx
        rng = np.random.default_rng(sample_seed)

        if self.sampling_strategy == "spatial_anchor":
            current_indices, context_role, anchor_distance_bucket = self._select_anchor_context(
                pair, rng
            )
        else:
            current_indices = self._select_current_indices(pair, rng)
            context_role = np.full(len(current_indices), 3, dtype=np.int64)
            anchor_distance_bucket = np.zeros(len(current_indices), dtype=np.int64)
        current_take = len(current_indices)

        current_genes = self.genes[current_indices]
        current_times = self.normalized_times[current_indices]
        targets = self._compute_targets(
            pair,
            current_indices,
            anchor_only=self.sampling_strategy == "spatial_anchor",
        )
        target_genes = targets["target_genes"]
        match_mask = targets["match_mask"]
        split_target = targets["split_target"]
        del_target = targets["del_target"]
        weak_del_target = targets["weak_del_target"]
        strict_del_target = targets["strict_del_target"]
        match_type = targets["match_type"]
        anchor_mask = np.zeros(current_take, dtype=bool)
        if self.sampling_strategy == "spatial_anchor":
            anchor_mask[0] = True
        else:
            anchor_mask[:] = True

        return {
            "genes": torch.from_numpy(current_genes),
            "target_genes": torch.from_numpy(target_genes),
            "match_mask": torch.from_numpy(match_mask),
            "split_target": torch.from_numpy(split_target),
            "del_target": torch.from_numpy(del_target),
            "weak_del_target": torch.from_numpy(weak_del_target),
            "strict_del_target": torch.from_numpy(strict_del_target),
            "match_type": torch.from_numpy(match_type),
            "anchor_mask": torch.from_numpy(anchor_mask),
            "context_role": torch.from_numpy(context_role),
            "anchor_distance_bucket": torch.from_numpy(anchor_distance_bucket),
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
    weak_del_target = torch.zeros(len(batch), max_len)
    strict_del_target = torch.zeros(len(batch), max_len)
    match_type = torch.full(
        (len(batch), max_len),
        GeneContextDataset.MATCH_UNSUPERVISED,
        dtype=torch.long,
    )
    anchor_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    context_role = torch.zeros(len(batch), max_len, dtype=torch.long)
    anchor_distance_bucket = torch.zeros(len(batch), max_len, dtype=torch.long)
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
        weak_del_target[i, :n] = item["weak_del_target"]
        strict_del_target[i, :n] = item["strict_del_target"]
        match_type[i, :n] = item["match_type"]
        anchor_mask[i, :n] = item["anchor_mask"]
        context_role[i, :n] = item["context_role"]
        anchor_distance_bucket[i, :n] = item["anchor_distance_bucket"]
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
        "weak_del_target": weak_del_target,
        "strict_del_target": strict_del_target,
        "match_type": match_type,
        "anchor_mask": anchor_mask,
        "context_role": context_role,
        "anchor_distance_bucket": anchor_distance_bucket,
        "token_times": token_times,
        "valid_mask": valid_mask,
        "time": time,
        "future_time": future_time,
    }
