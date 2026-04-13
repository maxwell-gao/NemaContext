"""Compatibility exports for gene-context datasets.

The implementation is split across:
- `gene_context_dataset_base.py` for token-level supervision datasets
- `gene_context_dataset_patch.py` for patch/local-view datasets
- `gene_context_dataset_embryo.py` for embryo-view datasets
"""

from __future__ import annotations

from .gene_context_dataset_base import GeneContextDataset, _TimePair, collate_gene_context
from .gene_context_dataset_embryo import EmbryoViewDataset, collate_embryo_view
from .gene_context_dataset_large2025 import (
    Large2025WholeEmbryoDataset,
    collate_large2025_whole_embryo,
)
from .gene_context_dataset_patch import (
    MultiPatchSetDataset,
    MultiViewPatchStateDataset,
    PatchSetDataset,
    TemporalPatchSetDataset,
    collate_history_patch_set,
    collate_multi_patch_set,
    collate_multi_view_patch_state,
    collate_patch_set,
)

__all__ = [
    "EmbryoViewDataset",
    "GeneContextDataset",
    "Large2025WholeEmbryoDataset",
    "MultiPatchSetDataset",
    "MultiViewPatchStateDataset",
    "PatchSetDataset",
    "TemporalPatchSetDataset",
    "_TimePair",
    "collate_embryo_view",
    "collate_gene_context",
    "collate_large2025_whole_embryo",
    "collate_history_patch_set",
    "collate_multi_patch_set",
    "collate_multi_view_patch_state",
    "collate_patch_set",
]
