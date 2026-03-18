"""Compatibility exports for gene-context models.

The implementation is split across:
- `gene_context_shared.py` for outputs and shared codecs
- `gene_context_patch.py` for token/patch/local models
- `gene_context_embryo.py` for embryo-scale models
"""

from __future__ import annotations

from .gene_context_embryo import (
    EmbryoFutureSetModel,
    EmbryoMaskedViewModel,
    EmbryoStateModel,
)
from .gene_context_patch import (
    GeneContextModel,
    LocalCellCodeModel,
    MultiCellPatchSetModel,
    MultiPatchSetModel,
    SingleCellGeneTimeModel,
    SingleCellPatchSetModel,
)
from .gene_context_shared import (
    EmbryoFutureSetOutput,
    EmbryoMaskedOutput,
    EmbryoStateOutput,
    GeneContextOutput,
    LocalCellCodeCodec,
    LocalCellCodeOutput,
    LocalCellDecodeOutput,
    MultiPatchSetOutput,
    PatchSetOutput,
)

__all__ = [
    "EmbryoFutureSetModel",
    "EmbryoFutureSetOutput",
    "EmbryoMaskedOutput",
    "EmbryoMaskedViewModel",
    "EmbryoStateModel",
    "EmbryoStateOutput",
    "GeneContextModel",
    "GeneContextOutput",
    "LocalCellCodeCodec",
    "LocalCellCodeModel",
    "LocalCellCodeOutput",
    "LocalCellDecodeOutput",
    "MultiCellPatchSetModel",
    "MultiPatchSetModel",
    "MultiPatchSetOutput",
    "PatchSetOutput",
    "SingleCellGeneTimeModel",
    "SingleCellPatchSetModel",
]
