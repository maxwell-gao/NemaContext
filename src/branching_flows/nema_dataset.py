"""Data bridge: AnnData → BranchingFlows SampleState.

Loads the NemaContext AnnData (transcriptome + spatial + lineage) and
produces training-ready SampleState objects grouped by developmental stage.

Design principles:
- Every modality is context, not a separate prediction target.
- No biological-prior feature selection (Bitter Lesson).
- HVG expression through a data-driven variance filter, not gene-list curation.
- Configurable element ordering (random / lineage / spatial).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from .states import SampleState


class NemaDataset:
    """Bridge between AnnData and BranchingFlows training.

    Each ``__getitem__`` call returns a :class:`SampleState` representing one
    developmental-stage bin of cells, ready for ``branching_bridge``.

    Args:
        h5ad_path: Path to the processed ``.h5ad`` file.
        n_hvg: Number of highly variable genes to use.  If the AnnData already
            has ``adata.var['highly_variable']``, those are used; otherwise
            scanpy selects them.
        time_bins: Number of bins to partition cells by ``embryo_time_min``.
        ordering: How to order elements within each sample.
            ``'random'`` (default), ``'lineage'``, or ``'spatial'``.
        founder_categories: Full list of founder labels (including future ones
            not in the current dataset).  The last index is reserved as the
            mask/dummy token.
    """

    FOUNDER_CATEGORIES = ("AB", "MS", "E", "C", "D", "P4")

    def __init__(
        self,
        h5ad_path: str | Path,
        n_hvg: int = 2000,
        time_bins: int = 10,
        ordering: Literal["random", "lineage", "spatial"] = "random",
        founder_categories: tuple[str, ...] | None = None,
    ):
        import anndata as ad
        import scipy.sparse as sp

        self.ordering = ordering
        founders = founder_categories or self.FOUNDER_CATEGORIES
        self._founder_to_idx = {f: i for i, f in enumerate(founders)}
        self.K = len(founders) + 1  # +1 for mask token

        adata = ad.read_h5ad(h5ad_path)

        # --- HVG selection ---
        if "highly_variable" in adata.var.columns:
            hvg_mask = adata.var["highly_variable"].values
            if hvg_mask.sum() != n_hvg:
                hvg_mask = _select_top_hvg(adata, n_hvg)
        else:
            hvg_mask = _select_top_hvg(adata, n_hvg)

        # --- Expression matrix (log-normalized HVGs) ---
        expr = adata.layers.get("log1p", adata.X)
        if sp.issparse(expr):
            expr = expr.toarray()
        expr_hvg = expr[:, hvg_mask].astype(np.float32)  # (N, n_hvg)

        # z-score per gene
        self._expr_mean = expr_hvg.mean(axis=0)
        self._expr_std = expr_hvg.std(axis=0).clip(min=1e-6)
        expr_hvg = (expr_hvg - self._expr_mean) / self._expr_std

        # --- Spatial ---
        spatial = adata.obsm["X_spatial"].astype(np.float32)  # (N, 3)
        self._spatial_min = spatial.min(axis=0)
        self._spatial_max = spatial.max(axis=0)
        spatial_range = (self._spatial_max - self._spatial_min).clip(min=1e-6)
        spatial = (spatial - self._spatial_min) / spatial_range  # [0, 1]

        # --- Continuous state = cat(expression, spatial) ---
        continuous = np.concatenate([expr_hvg, spatial], axis=1)  # (N, n_hvg+3)
        self.continuous_dim = continuous.shape[1]

        # --- Discrete state = founder index ---
        discrete = np.array([
            self._founder_to_idx.get(f, self.K - 1)
            for f in adata.obs["lineage_founder"]
        ], dtype=np.int64)

        # --- Lineage names (for ordering) ---
        lineage_names = adata.obs["lineage_complete"].values
        times = adata.obs["embryo_time_min"].values.astype(np.float32)

        # --- Bin cells by developmental time ---
        bin_edges = np.linspace(times.min(), times.max() + 1e-6, time_bins + 1)
        bin_indices = np.digitize(times, bin_edges) - 1
        bin_indices = bin_indices.clip(0, time_bins - 1)

        self._samples: list[_BinnedSample] = []
        for b in range(time_bins):
            mask = bin_indices == b
            if mask.sum() == 0:
                continue
            self._samples.append(_BinnedSample(
                continuous=continuous[mask],
                discrete=discrete[mask],
                lineage_names=lineage_names[mask],
                spatial_raw=spatial[mask],
            ))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> SampleState:
        sample = self._samples[idx]
        n = len(sample.continuous)

        order = _compute_order(
            self.ordering, n, sample.lineage_names, sample.spatial_raw,
        )

        elements: list[tuple[torch.Tensor, int]] = []
        for i in order:
            c = torch.from_numpy(sample.continuous[i])
            d = int(sample.discrete[i])
            elements.append((c, d))

        return SampleState(
            elements=elements,
            groupings=[0] * n,
            del_flags=[False] * n,
            ids=list(range(1, n + 1)),
            branchmask=[True] * n,
            flowmask=[True] * n,
        )

    def x0_sampler(self, root: Any) -> tuple[torch.Tensor, int]:
        """Initial-distribution sampler: Gaussian noise + mask token."""
        return (torch.randn(self.continuous_dim), self.K - 1)

    # --- Denormalization utilities ---

    def denorm_continuous(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split and denormalize continuous state back to (expression, spatial)."""
        n_hvg = self.continuous_dim - 3
        expr = x[..., :n_hvg]
        spatial = x[..., n_hvg:]
        expr_denorm = expr * torch.from_numpy(self._expr_std).to(x) + torch.from_numpy(self._expr_mean).to(x)
        spatial_range = torch.from_numpy(self._spatial_max - self._spatial_min).to(x)
        spatial_denorm = spatial * spatial_range + torch.from_numpy(self._spatial_min).to(x)
        return expr_denorm, spatial_denorm


class _BinnedSample:
    __slots__ = ("continuous", "discrete", "lineage_names", "spatial_raw")

    def __init__(
        self,
        continuous: np.ndarray,
        discrete: np.ndarray,
        lineage_names: np.ndarray,
        spatial_raw: np.ndarray,
    ):
        self.continuous = continuous
        self.discrete = discrete
        self.lineage_names = lineage_names
        self.spatial_raw = spatial_raw


# ---------------------------------------------------------------------------
# Ordering strategies
# ---------------------------------------------------------------------------

def _compute_order(
    strategy: str,
    n: int,
    lineage_names: np.ndarray,
    spatial: np.ndarray,
) -> list[int]:
    if strategy == "lineage":
        return _lineage_dfs_order(lineage_names)
    if strategy == "spatial":
        return _spatial_curve_order(spatial)
    # random
    order = list(range(n))
    random.shuffle(order)
    return order


def _lineage_dfs_order(names: np.ndarray) -> list[int]:
    """Sort by lineage name -- DFS order of the Sulston tree."""
    indexed = sorted(enumerate(names), key=lambda x: x[1])
    return [i for i, _ in indexed]


def _spatial_curve_order(spatial: np.ndarray) -> list[int]:
    """Order by a simple space-filling curve (sum of coordinates)."""
    scores = spatial.sum(axis=1)
    return list(np.argsort(scores))


def _select_top_hvg(adata: Any, n_hvg: int) -> np.ndarray:
    """Select top *n_hvg* HVGs by normalized variance rank."""
    if "highly_variable_rank" in adata.var.columns:
        ranks = adata.var["highly_variable_rank"].values.copy()
        ranks[np.isnan(ranks)] = np.inf
        top_idx = np.argsort(ranks)[:n_hvg]
        mask = np.zeros(adata.shape[1], dtype=bool)
        mask[top_idx] = True
        return mask
    import scanpy as sc
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="seurat_v3")
    return adata.var["highly_variable"].values
