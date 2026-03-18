"""Embryo-view gene-context datasets."""

from __future__ import annotations

import numpy as np
import torch

from .gene_context_dataset_patch import PatchSetDataset, collate_patch_set


class EmbryoViewDataset(PatchSetDataset):
    """Embryo-scale state views built from multiple local patches per time window."""

    FOUNDERS = ("AB", "MS", "E", "C", "D", "P4", "Z", "UNK")

    def __init__(
        self,
        *args,
        views_per_embryo: int = 4,
        future_views_per_embryo: int | None = None,
        top_cell_types: int = 8,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if views_per_embryo < 1:
            raise ValueError("views_per_embryo must be >= 1")
        self.views_per_embryo = views_per_embryo
        self.future_views_per_embryo = future_views_per_embryo or views_per_embryo
        self.top_cell_types = top_cell_types
        self._top_cell_type_vocab = self._build_top_cell_type_vocab()

    def _build_top_cell_type_vocab(self) -> tuple[str, ...]:
        counts: dict[str, int] = {}
        for cell_type in self.adata_obs_cell_type:
            value = str(cell_type)
            if value == "nan":
                continue
            counts[value] = counts.get(value, 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return tuple(cell_type for cell_type, _ in ranked[: self.top_cell_types])

    @classmethod
    def _founder_from_lineage(cls, lineage: str) -> str:
        for founder in cls.FOUNDERS[:-1]:
            if lineage.startswith(founder):
                return founder
        return "UNK"

    @staticmethod
    def _composition_vector(values: list[str], vocab: tuple[str, ...]) -> np.ndarray:
        out = np.zeros(len(vocab), dtype=np.float32)
        if not values:
            return out
        counts: dict[str, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        total = float(len(values))
        for i, token in enumerate(vocab):
            out[i] = counts.get(token, 0) / total
        return out

    def _compute_embryo_future_targets(self, future_indices: np.ndarray) -> dict[str, torch.Tensor]:
        future_lineages = [str(x) for x in self.lineages[future_indices]]
        future_founders = [self._founder_from_lineage(x) for x in future_lineages]
        future_cell_types = [str(x) for x in self.adata_obs_cell_type[future_indices]]

        founder_comp = self._composition_vector(future_founders, self.FOUNDERS)
        celltype_comp = self._composition_vector(future_cell_types, self._top_cell_type_vocab)

        depths = np.asarray(self.lineage_depths[future_indices], dtype=np.float32)
        valid_depths = depths[depths >= 0]
        if len(valid_depths) == 0:
            depth_stats = np.zeros(3, dtype=np.float32)
        else:
            depth_stats = np.array(
                [
                    float(valid_depths.mean()),
                    float(valid_depths.std()),
                    float(len(valid_depths) / max(1, len(depths))),
                ],
                dtype=np.float32,
            )

        valid_future_spatial = self.valid_spatial[future_indices]
        if np.any(valid_future_spatial):
            coords = self.spatial_coords[future_indices[valid_future_spatial]]
            centered = coords - coords.mean(axis=0, keepdims=True)
            spatial_extent = np.concatenate(
                [coords.std(axis=0), [float(np.linalg.norm(centered, axis=1).mean())]],
                axis=0,
            ).astype(np.float32)
        else:
            spatial_extent = np.zeros(4, dtype=np.float32)

        split_fraction = np.array(
            [self._compute_split_fraction(self.lineages[future_indices])], dtype=np.float32
        )

        return {
            "future_founder_composition": torch.from_numpy(founder_comp),
            "future_celltype_composition": torch.from_numpy(celltype_comp),
            "future_lineage_depth_stats": torch.from_numpy(depth_stats),
            "future_spatial_extent": torch.from_numpy(spatial_extent),
            "future_split_fraction": torch.from_numpy(split_fraction),
        }

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pair = self.time_pairs[idx % len(self.time_pairs)]
        rng = np.random.default_rng(self.random_seed + idx)

        views = []
        for _ in range(self.views_per_embryo):
            indices, roles, relpos = self._select_patch_from_indices(pair.current_indices, rng)
            views.append((indices, self._build_patch_view(indices, roles, relpos, pair.current_center)))
        future_views = []
        for _ in range(self.future_views_per_embryo):
            indices, roles, relpos = self._select_patch_from_indices(pair.future_indices, rng)
            future_views.append((indices, self._build_patch_view(indices, roles, relpos, pair.future_center)))

        output: dict[str, torch.Tensor] = {}
        for view_idx, (indices, view) in enumerate(views):
            prefix = f"view_{view_idx}"
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

        output["views_per_embryo"] = torch.tensor(self.views_per_embryo, dtype=torch.long)
        output["future_views_per_embryo"] = torch.tensor(self.future_views_per_embryo, dtype=torch.long)
        output["current_center_min"] = torch.tensor(float(pair.current_center), dtype=torch.float32)
        output["future_center_min"] = torch.tensor(float(pair.future_center), dtype=torch.float32)
        output["top_cell_types"] = list(self._top_cell_type_vocab)
        output.update(self._compute_embryo_future_targets(pair.future_indices))
        return output


def collate_embryo_view(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    first = batch[0]
    view_prefixes = sorted(
        {key.rsplit("_", 1)[0] for key in first if key.startswith("view_") and key.endswith("_genes")}
    )
    future_view_prefixes = sorted(
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
    for prefix in view_prefixes + future_view_prefixes:
        collated = _collate_view(prefix)
        output[f"{prefix}_genes"] = collated["current_genes"]
        output[f"{prefix}_context_role"] = collated["current_context_role"]
        output[f"{prefix}_relative_position"] = collated["current_relative_position"]
        output[f"{prefix}_token_times"] = collated["current_token_times"]
        output[f"{prefix}_valid_mask"] = collated["current_valid_mask"]
        output[f"{prefix}_anchor_mask"] = collated["current_anchor_mask"]
        output[f"{prefix}_time"] = collated["current_time"]

    output["views_per_embryo"] = torch.stack([item["views_per_embryo"] for item in batch])
    output["future_views_per_embryo"] = torch.stack([item["future_views_per_embryo"] for item in batch])
    output["current_center_min"] = torch.stack([item["current_center_min"] for item in batch])
    output["future_center_min"] = torch.stack([item["future_center_min"] for item in batch])
    output["future_founder_composition"] = torch.stack(
        [item["future_founder_composition"] for item in batch]
    )
    output["future_celltype_composition"] = torch.stack(
        [item["future_celltype_composition"] for item in batch]
    )
    output["future_lineage_depth_stats"] = torch.stack(
        [item["future_lineage_depth_stats"] for item in batch]
    )
    output["future_spatial_extent"] = torch.stack([item["future_spatial_extent"] for item in batch])
    output["future_split_fraction"] = torch.stack([item["future_split_fraction"] for item in batch])
    output["top_cell_types"] = first["top_cell_types"]
    return output
