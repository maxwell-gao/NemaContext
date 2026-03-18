"""Patch-, multi-patch-, and multi-view gene-context datasets."""

from __future__ import annotations

import numpy as np
import torch

from .gene_context_dataset_base import GeneContextDataset


class PatchSetDataset(GeneContextDataset):
    """Patch-to-patch dataset with independent future patch sampling."""

    @staticmethod
    def _compute_split_fraction(lineages: np.ndarray) -> float:
        if len(lineages) == 0:
            return 0.0

        parent_counts: dict[str, int] = {}
        for lineage in lineages:
            if len(lineage) <= 1:
                continue
            parent = lineage[:-1]
            parent_counts[parent] = parent_counts.get(parent, 0) + 1

        split_like = 0
        for lineage in lineages:
            if len(lineage) <= 1:
                continue
            parent = lineage[:-1]
            if parent_counts.get(parent, 0) >= 2:
                split_like += 1
        return float(split_like / max(1, len(lineages)))

    def _select_patch_from_indices(
        self,
        window_indices: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.sampling_strategy == "spatial_anchor":
            return self._select_anchor_context_from_indices(window_indices, rng)

        selected = self._select_indices_from_window(window_indices, rng)
        context_role = np.full(len(selected), 3, dtype=np.int64)
        relative_position = np.zeros((len(selected), 5), dtype=np.float32)
        valid_spatial = self.valid_spatial[selected]
        relative_position[valid_spatial, 4] = 1.0
        anchor_mask = np.zeros(len(selected), dtype=bool)
        if len(selected) > 0:
            anchor_mask[0] = True
        return selected, context_role, relative_position

    def _build_patch_view(
        self,
        indices: np.ndarray,
        context_role: np.ndarray,
        relative_position: np.ndarray,
        center: float,
    ) -> dict[str, torch.Tensor]:
        genes = self.genes[indices]
        token_times = self.normalized_times[indices]
        anchor_mask = np.zeros(len(indices), dtype=bool)
        if len(indices) > 0:
            anchor_mask[0] = True
        return {
            "genes": torch.from_numpy(genes),
            "context_role": torch.from_numpy(context_role),
            "relative_position": torch.from_numpy(relative_position),
            "token_times": torch.from_numpy(token_times),
            "valid_mask": torch.ones(len(indices), dtype=torch.bool),
            "anchor_mask": torch.from_numpy(anchor_mask),
            "time": torch.tensor(
                (center - self.time_min) / max(1e-6, self.time_max - self.time_min),
                dtype=torch.float32,
            ),
        }

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pair = self.time_pairs[idx % len(self.time_pairs)]
        rng = np.random.default_rng(self.random_seed + idx)

        current_indices, current_roles, current_relpos = self._select_patch_from_indices(
            pair.current_indices,
            rng,
        )
        future_indices, future_roles, future_relpos = self._select_patch_from_indices(
            pair.future_indices,
            rng,
        )

        current_view = self._build_patch_view(
            current_indices, current_roles, current_relpos, pair.current_center
        )
        future_view = self._build_patch_view(
            future_indices, future_roles, future_relpos, pair.future_center
        )

        future_genes = future_view["genes"]
        future_valid = future_view["valid_mask"]
        future_mean_gene = future_genes[future_valid].mean(dim=0)
        future_patch_size = torch.tensor(float(future_valid.sum().item()), dtype=torch.float32)
        current_split_fraction = torch.tensor(
            self._compute_split_fraction(self.lineages[current_indices]),
            dtype=torch.float32,
        )
        future_split_fraction = torch.tensor(
            self._compute_split_fraction(self.lineages[future_indices]),
            dtype=torch.float32,
        )

        return {
            "current_genes": current_view["genes"],
            "current_context_role": current_view["context_role"],
            "current_relative_position": current_view["relative_position"],
            "current_token_times": current_view["token_times"],
            "current_valid_mask": current_view["valid_mask"],
            "current_anchor_mask": current_view["anchor_mask"],
            "current_time": current_view["time"],
            "future_genes": future_view["genes"],
            "future_context_role": future_view["context_role"],
            "future_relative_position": future_view["relative_position"],
            "future_token_times": future_view["token_times"],
            "future_valid_mask": future_view["valid_mask"],
            "future_anchor_mask": future_view["anchor_mask"],
            "future_time": future_view["time"],
            "future_mean_gene": future_mean_gene,
            "future_patch_size": future_patch_size,
            "current_split_fraction": current_split_fraction,
            "future_split_fraction": future_split_fraction,
        }


def collate_patch_set(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    current_max_len = max(item["current_genes"].shape[0] for item in batch)
    future_max_len = max(item["future_genes"].shape[0] for item in batch)
    gene_dim = batch[0]["current_genes"].shape[1]

    output = {
        "current_genes": torch.zeros(len(batch), current_max_len, gene_dim),
        "current_context_role": torch.zeros(len(batch), current_max_len, dtype=torch.long),
        "current_relative_position": torch.zeros(len(batch), current_max_len, 5),
        "current_token_times": torch.zeros(len(batch), current_max_len),
        "current_valid_mask": torch.zeros(len(batch), current_max_len, dtype=torch.bool),
        "current_anchor_mask": torch.zeros(len(batch), current_max_len, dtype=torch.bool),
        "current_time": torch.zeros(len(batch)),
        "future_genes": torch.zeros(len(batch), future_max_len, gene_dim),
        "future_context_role": torch.zeros(len(batch), future_max_len, dtype=torch.long),
        "future_relative_position": torch.zeros(len(batch), future_max_len, 5),
        "future_token_times": torch.zeros(len(batch), future_max_len),
        "future_valid_mask": torch.zeros(len(batch), future_max_len, dtype=torch.bool),
        "future_anchor_mask": torch.zeros(len(batch), future_max_len, dtype=torch.bool),
        "future_time": torch.zeros(len(batch)),
        "future_mean_gene": torch.zeros(len(batch), gene_dim),
        "future_patch_size": torch.zeros(len(batch)),
        "current_split_fraction": torch.zeros(len(batch)),
        "future_split_fraction": torch.zeros(len(batch)),
    }

    for i, item in enumerate(batch):
        current_len = item["current_genes"].shape[0]
        future_len = item["future_genes"].shape[0]
        output["current_genes"][i, :current_len] = item["current_genes"]
        output["current_context_role"][i, :current_len] = item["current_context_role"]
        output["current_relative_position"][i, :current_len] = item["current_relative_position"]
        output["current_token_times"][i, :current_len] = item["current_token_times"]
        output["current_valid_mask"][i, :current_len] = item["current_valid_mask"]
        output["current_anchor_mask"][i, :current_len] = item["current_anchor_mask"]
        output["current_time"][i] = item["current_time"]
        output["future_genes"][i, :future_len] = item["future_genes"]
        output["future_context_role"][i, :future_len] = item["future_context_role"]
        output["future_relative_position"][i, :future_len] = item["future_relative_position"]
        output["future_token_times"][i, :future_len] = item["future_token_times"]
        output["future_valid_mask"][i, :future_len] = item["future_valid_mask"]
        output["future_anchor_mask"][i, :future_len] = item["future_anchor_mask"]
        output["future_time"][i] = item["future_time"]
        output["future_mean_gene"][i] = item["future_mean_gene"]
        output["future_patch_size"][i] = item["future_patch_size"]
        output["current_split_fraction"][i] = item["current_split_fraction"]
        output["future_split_fraction"][i] = item["future_split_fraction"]

    return output


class MultiPatchSetDataset(PatchSetDataset):
    """Multi-patch variant for patch-count extrapolation experiments."""

    def __init__(
        self,
        *args,
        patches_per_state: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if patches_per_state < 1:
            raise ValueError("patches_per_state must be >= 1")
        self.patches_per_state = patches_per_state

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pair = self.time_pairs[idx % len(self.time_pairs)]
        rng = np.random.default_rng(self.random_seed + idx)
        patch_items = []
        for _ in range(self.patches_per_state):
            current_indices, current_roles, current_relpos = self._select_patch_from_indices(
                pair.current_indices,
                rng,
            )
            future_indices, future_roles, future_relpos = self._select_patch_from_indices(
                pair.future_indices,
                rng,
            )

            current_view = self._build_patch_view(
                current_indices, current_roles, current_relpos, pair.current_center
            )
            future_view = self._build_patch_view(
                future_indices, future_roles, future_relpos, pair.future_center
            )

            future_genes = future_view["genes"]
            future_valid = future_view["valid_mask"]
            future_mean_gene = future_genes[future_valid].mean(dim=0)
            future_patch_size = torch.tensor(float(future_valid.sum().item()), dtype=torch.float32)
            current_split_fraction = torch.tensor(
                self._compute_split_fraction(self.lineages[current_indices]),
                dtype=torch.float32,
            )
            future_split_fraction = torch.tensor(
                self._compute_split_fraction(self.lineages[future_indices]),
                dtype=torch.float32,
            )
            patch_items.append(
                {
                    "current_genes": current_view["genes"],
                    "current_context_role": current_view["context_role"],
                    "current_relative_position": current_view["relative_position"],
                    "current_token_times": current_view["token_times"],
                    "current_valid_mask": current_view["valid_mask"],
                    "current_anchor_mask": current_view["anchor_mask"],
                    "current_time": current_view["time"],
                    "future_genes": future_view["genes"],
                    "future_context_role": future_view["context_role"],
                    "future_relative_position": future_view["relative_position"],
                    "future_token_times": future_view["token_times"],
                    "future_valid_mask": future_view["valid_mask"],
                    "future_anchor_mask": future_view["anchor_mask"],
                    "future_time": future_view["time"],
                    "future_mean_gene": future_mean_gene,
                    "future_patch_size": future_patch_size,
                    "current_split_fraction": current_split_fraction,
                    "future_split_fraction": future_split_fraction,
                }
            )

        output: dict[str, torch.Tensor] = {}
        for key in patch_items[0]:
            output[key] = torch.stack([item[key] for item in patch_items], dim=0)
        return output


def collate_multi_patch_set(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    patches_per_state = batch[0]["current_genes"].shape[0]
    flat_batch: list[dict[str, torch.Tensor]] = []
    for item in batch:
        for patch_idx in range(patches_per_state):
            flat_batch.append({key: value[patch_idx] for key, value in item.items()})

    collated = collate_patch_set(flat_batch)
    batch_size = len(batch)
    current_len = collated["current_genes"].shape[1]
    future_len = collated["future_genes"].shape[1]
    gene_dim = collated["current_genes"].shape[2]

    return {
        "current_genes": collated["current_genes"].view(batch_size, patches_per_state, current_len, gene_dim),
        "current_context_role": collated["current_context_role"].view(batch_size, patches_per_state, current_len),
        "current_relative_position": collated["current_relative_position"].view(
            batch_size, patches_per_state, current_len, 5
        ),
        "current_token_times": collated["current_token_times"].view(batch_size, patches_per_state, current_len),
        "current_valid_mask": collated["current_valid_mask"].view(batch_size, patches_per_state, current_len),
        "current_anchor_mask": collated["current_anchor_mask"].view(batch_size, patches_per_state, current_len),
        "current_time": collated["current_time"].view(batch_size, patches_per_state),
        "future_genes": collated["future_genes"].view(batch_size, patches_per_state, future_len, gene_dim),
        "future_context_role": collated["future_context_role"].view(batch_size, patches_per_state, future_len),
        "future_relative_position": collated["future_relative_position"].view(
            batch_size, patches_per_state, future_len, 5
        ),
        "future_token_times": collated["future_token_times"].view(batch_size, patches_per_state, future_len),
        "future_valid_mask": collated["future_valid_mask"].view(batch_size, patches_per_state, future_len),
        "future_anchor_mask": collated["future_anchor_mask"].view(batch_size, patches_per_state, future_len),
        "future_time": collated["future_time"].view(batch_size, patches_per_state),
        "future_mean_gene": collated["future_mean_gene"].view(batch_size, patches_per_state, gene_dim),
        "future_patch_size": collated["future_patch_size"].view(batch_size, patches_per_state),
        "current_split_fraction": collated["current_split_fraction"].view(batch_size, patches_per_state),
        "future_split_fraction": collated["future_split_fraction"].view(batch_size, patches_per_state),
        "patches_per_state": torch.full((batch_size,), patches_per_state, dtype=torch.long),
    }


class MultiViewPatchStateDataset(PatchSetDataset):
    """Multiple local views of the same time-paired embryo state."""

    def __init__(
        self,
        *args,
        views_per_state: int = 2,
        future_views_per_state: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if views_per_state < 2:
            raise ValueError("views_per_state must be >= 2")
        if future_views_per_state < 1:
            raise ValueError("future_views_per_state must be >= 1")
        self.views_per_state = views_per_state
        self.future_views_per_state = future_views_per_state

    def _sample_views(
        self,
        window_indices: np.ndarray,
        center: float,
        rng: np.random.Generator,
        n_views: int,
    ) -> list[tuple[np.ndarray, dict[str, torch.Tensor]]]:
        views: list[tuple[np.ndarray, dict[str, torch.Tensor]]] = []
        for _ in range(n_views):
            indices, roles, relpos = self._select_patch_from_indices(window_indices, rng)
            views.append((indices, self._build_patch_view(indices, roles, relpos, center)))
        return views

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pair = self.time_pairs[idx % len(self.time_pairs)]
        rng = np.random.default_rng(self.random_seed + idx)

        current_views = self._sample_views(
            pair.current_indices,
            pair.current_center,
            rng,
            self.views_per_state,
        )
        future_views = self._sample_views(
            pair.future_indices,
            pair.future_center,
            rng,
            self.future_views_per_state,
        )

        output: dict[str, torch.Tensor] = {}
        for view_idx, (indices, view) in enumerate(current_views):
            prefix = f"current_view_{view_idx}"
            output[f"{prefix}_genes"] = view["genes"]
            output[f"{prefix}_context_role"] = view["context_role"]
            output[f"{prefix}_relative_position"] = view["relative_position"]
            output[f"{prefix}_token_times"] = view["token_times"]
            output[f"{prefix}_valid_mask"] = view["valid_mask"]
            output[f"{prefix}_anchor_mask"] = view["anchor_mask"]
            output[f"{prefix}_time"] = view["time"]
            output[f"{prefix}_indices"] = torch.from_numpy(indices.astype(np.int64))

        for view_idx, (indices, view) in enumerate(future_views):
            prefix = f"future_view_{view_idx}"
            output[f"{prefix}_genes"] = view["genes"]
            output[f"{prefix}_context_role"] = view["context_role"]
            output[f"{prefix}_relative_position"] = view["relative_position"]
            output[f"{prefix}_token_times"] = view["token_times"]
            output[f"{prefix}_valid_mask"] = view["valid_mask"]
            output[f"{prefix}_anchor_mask"] = view["anchor_mask"]
            output[f"{prefix}_time"] = view["time"]
            output[f"{prefix}_indices"] = torch.from_numpy(indices.astype(np.int64))

        output["views_per_state"] = torch.tensor(self.views_per_state, dtype=torch.long)
        output["future_views_per_state"] = torch.tensor(self.future_views_per_state, dtype=torch.long)
        output["current_center_min"] = torch.tensor(float(pair.current_center), dtype=torch.float32)
        output["future_center_min"] = torch.tensor(float(pair.future_center), dtype=torch.float32)
        return output


def collate_multi_view_patch_state(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    first = batch[0]
    current_prefixes = sorted(
        {
            key.rsplit("_", 1)[0]
            for key in first
            if key.startswith("current_view_") and key.endswith("_genes")
        }
    )
    future_prefixes = sorted(
        {
            key.rsplit("_", 1)[0]
            for key in first
            if key.startswith("future_view_") and key.endswith("_genes")
        }
    )

    def _collate_view(prefix: str) -> dict[str, torch.Tensor]:
        return collate_patch_set(
            [
                {
                    "current_genes": item[f"{prefix}_genes"],
                    "current_context_role": item[f"{prefix}_context_role"],
                    "current_relative_position": item[f"{prefix}_relative_position"],
                    "current_token_times": item[f"{prefix}_token_times"],
                    "current_valid_mask": item[f"{prefix}_valid_mask"],
                    "current_anchor_mask": item[f"{prefix}_anchor_mask"],
                    "current_time": item[f"{prefix}_time"],
                    "future_genes": item[f"{prefix}_genes"],
                    "future_context_role": item[f"{prefix}_context_role"],
                    "future_relative_position": item[f"{prefix}_relative_position"],
                    "future_token_times": item[f"{prefix}_token_times"],
                    "future_valid_mask": item[f"{prefix}_valid_mask"],
                    "future_anchor_mask": item[f"{prefix}_anchor_mask"],
                    "future_time": item[f"{prefix}_time"],
                    "future_mean_gene": item[f"{prefix}_genes"].mean(dim=0),
                    "future_patch_size": torch.tensor(float(item[f"{prefix}_valid_mask"].sum().item())),
                    "current_split_fraction": torch.tensor(0.0),
                    "future_split_fraction": torch.tensor(0.0),
                }
                for item in batch
            ]
        )

    output: dict[str, torch.Tensor] = {}
    for prefix in current_prefixes + future_prefixes:
        collated = _collate_view(prefix)
        base = prefix
        output[f"{base}_genes"] = collated["current_genes"]
        output[f"{base}_context_role"] = collated["current_context_role"]
        output[f"{base}_relative_position"] = collated["current_relative_position"]
        output[f"{base}_token_times"] = collated["current_token_times"]
        output[f"{base}_valid_mask"] = collated["current_valid_mask"]
        output[f"{base}_anchor_mask"] = collated["current_anchor_mask"]
        output[f"{base}_time"] = collated["current_time"]

    output["views_per_state"] = torch.stack([item["views_per_state"] for item in batch])
    output["future_views_per_state"] = torch.stack([item["future_views_per_state"] for item in batch])
    output["current_center_min"] = torch.stack([item["current_center_min"] for item in batch])
    output["future_center_min"] = torch.stack([item["future_center_min"] for item in batch])
    return output
