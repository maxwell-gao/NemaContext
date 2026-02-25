"""Trimodal dataset with modality masking for partial observations.

Extends NemaDataset to handle cells with missing modalities using
bitmask flags (1=transcriptome, 2=spatial, 4=lineage).

The design follows the Bitter Lesson: handle missing data through
learned masking rather than hardcoded imputation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from .states import SampleState


class TrimodalDataset:
    """Three-modality dataset with masking for partial observations.

    Loads from extended AnnData (nema_extended_large2025.h5ad) which contains
    234k cells with modality availability flags. Only ~1.6k cells have all
    three modalities; the rest are used with masked losses.

    Modality bitmask:
        - bit 0 (1): Transcriptome available
        - bit 1 (2): Spatial available
        - bit 2 (4): Lineage available

    Args:
        h5ad_path: Path to extended AnnData file
        n_hvg: Number of highly variable genes
        time_bins: Number of developmental time bins
        ordering: Element ordering strategy
        use_lineage_bias: Whether to compute lineage bias for attention
        max_cells_per_bin: Maximum cells per time bin (for GPU memory)
    """

    FOUNDER_CATEGORIES = ("AB", "MS", "E", "C", "D", "P4")

    def __init__(
        self,
        h5ad_path: str | Path,
        n_hvg: int = 2000,
        time_bins: int = 10,
        ordering: Literal["random", "lineage", "spatial"] = "random",
        use_lineage_bias: bool = True,
        max_cells_per_bin: int | None = None,
        augment_spatial: bool = True,
        aug_rotation: bool = True,
        aug_flip: bool = True,
        aug_scale: float = 0.1,
    ):
        import anndata as ad
        import scipy.sparse as sp

        self.ordering = ordering
        self.use_lineage_bias = use_lineage_bias
        self.n_hvg = n_hvg
        self.max_cells_per_bin = max_cells_per_bin

        # Data augmentation settings
        self.augment_spatial = augment_spatial
        self.aug_rotation = aug_rotation
        self.aug_flip = aug_flip
        self.aug_scale = aug_scale

        founders = self.FOUNDER_CATEGORIES
        self._founder_to_idx = {f: i for i, f in enumerate(founders)}
        self.K = len(founders) + 1

        adata = ad.read_h5ad(h5ad_path)

        # Check for modality mask column
        if "modality_mask" in adata.obs.columns:
            self.modality_masks = adata.obs["modality_mask"].values.astype(np.int64)
        elif "has_transcriptome" in adata.obs.columns:
            # Build mask from boolean columns
            mask = np.zeros(len(adata), dtype=np.int64)
            if "has_transcriptome" in adata.obs.columns:
                mask |= adata.obs["has_transcriptome"].values.astype(np.int64) * 1
            if "has_spatial" in adata.obs.columns:
                mask |= adata.obs["has_spatial"].values.astype(np.int64) * 2
            if "has_lineage" in adata.obs.columns:
                mask |= adata.obs["has_lineage"].values.astype(np.int64) * 4
            self.modality_masks = mask
        else:
            # Assume all modalities available
            self.modality_masks = np.full(len(adata), 7, dtype=np.int64)

        # HVG selection
        if "highly_variable" in adata.var.columns:
            hvg_mask = adata.var["highly_variable"].values
            if hvg_mask.sum() < n_hvg:
                hvg_mask = self._select_top_hvg(adata, n_hvg)
        else:
            hvg_mask = self._select_top_hvg(adata, n_hvg)

        # Expression matrix (log-normalized HVGs)
        expr = adata.layers.get("log1p", adata.X)
        if sp.issparse(expr):
            expr = expr.toarray()
        expr_hvg = expr[:, hvg_mask].astype(np.float32)

        # z-score per gene (store for denormalization)
        self._expr_mean = expr_hvg.mean(axis=0)
        self._expr_std = expr_hvg.std(axis=0).clip(min=1e-6)
        expr_hvg = (expr_hvg - self._expr_mean) / self._expr_std

        # Spatial coordinates
        if "X_spatial" in adata.obsm:
            spatial = adata.obsm["X_spatial"].astype(np.float32)
            # Fill NaN with median (for cells without spatial data)
            spatial_nan_mask = np.isnan(spatial)
            if spatial_nan_mask.any():
                spatial_median = np.nanmedian(spatial, axis=0)
                for i in range(spatial.shape[1]):
                    spatial[spatial_nan_mask[:, i], i] = spatial_median[i]
        else:
            # Create placeholder spatial
            spatial = np.zeros((len(adata), 3), dtype=np.float32)

        self._spatial_min = spatial.min(axis=0)
        self._spatial_max = spatial.max(axis=0)
        spatial_range = (self._spatial_max - self._spatial_min).clip(min=1e-6)
        spatial = (spatial - self._spatial_min) / spatial_range

        # Continuous state = cat(expression, spatial)
        self._gene_dim = expr_hvg.shape[1]
        self._spatial_dim = spatial.shape[1]
        continuous = np.concatenate([expr_hvg, spatial], axis=1)
        self.continuous_dim = continuous.shape[1]

        # Discrete state = founder index
        if "lineage_founder" in adata.obs.columns:
            discrete = np.array(
                [
                    self._founder_to_idx.get(f, self.K - 1)
                    for f in adata.obs["lineage_founder"]
                ],
                dtype=np.int64,
            )
        else:
            discrete = np.full(len(adata), self.K - 1, dtype=np.int64)

        # Lineage names for bias computation
        if "lineage_complete" in adata.obs.columns:
            lineage_names = adata.obs["lineage_complete"].values
        elif "lineage_name" in adata.obs.columns:
            lineage_names = adata.obs["lineage_name"].values
        else:
            lineage_names = np.full(len(adata), "", dtype=object)

        # Time bins
        if "embryo_time_min" in adata.obs.columns:
            times = adata.obs["embryo_time_min"].values.astype(np.float32)
        else:
            times = np.zeros(len(adata), dtype=np.float32)

        bin_edges = np.linspace(times.min(), times.max() + 1e-6, time_bins + 1)
        bin_indices = np.digitize(times, bin_edges) - 1
        bin_indices = bin_indices.clip(0, time_bins - 1)

        # Group into samples (with optional max cells per bin)
        self._samples: list[_TrimodalSample] = []
        for b in range(time_bins):
            mask = bin_indices == b
            n_cells = mask.sum()
            if n_cells == 0:
                continue

            # Subsample if too many cells
            if max_cells_per_bin is not None and n_cells > max_cells_per_bin:
                indices = np.where(mask)[0]
                selected = np.random.choice(
                    indices, size=max_cells_per_bin, replace=False
                )
                mask = np.zeros(len(mask), dtype=bool)
                mask[selected] = True

            self._samples.append(
                _TrimodalSample(
                    continuous=continuous[mask],
                    discrete=discrete[mask],
                    lineage_names=lineage_names[mask],
                    spatial_raw=spatial[mask],
                    modality_masks=self.modality_masks[mask],
                    gene_dim=self._gene_dim,
                )
            )

        # Compute modality statistics
        self._compute_stats()

    def _compute_stats(self) -> None:
        """Compute dataset statistics for logging."""
        total = len(self.modality_masks)
        has_transcriptome = (self.modality_masks & 1).sum()
        has_spatial = (self.modality_masks & 2).sum()
        has_lineage = (self.modality_masks & 4).sum()
        has_all = ((self.modality_masks & 7) == 7).sum()

        self.stats = {
            "total_cells": int(total),
            "has_transcriptome": int(has_transcriptome),
            "has_spatial": int(has_spatial),
            "has_lineage": int(has_lineage),
            "has_all_modalities": int(has_all),
            "pct_complete": 100.0 * has_all / total if total > 0 else 0.0,
        }

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> SampleState:
        sample = self._samples[idx]
        n = len(sample.continuous)

        order = self._compute_order(n, sample)

        # Get ordered continuous data
        ordered_continuous = sample.continuous[order]

        # Apply data augmentation if enabled
        if self.augment_spatial and self._gene_dim > 0:
            ordered_continuous = self._augment_continuous(ordered_continuous)

        elements: list[tuple[torch.Tensor, int]] = []
        for i, idx_in_order in enumerate(range(n)):
            c = torch.from_numpy(ordered_continuous[i])
            d = int(sample.discrete[order[i]])
            elements.append((c, d))

        state = SampleState(
            elements=elements,
            groupings=[0] * n,
            del_flags=[False] * n,
            ids=list(range(1, n + 1)),
            branchmask=[True] * n,
            flowmask=[True] * n,
        )

        # Attach modality masks and lineage names for loss computation
        state.modality_masks = torch.from_numpy(sample.modality_masks[order])
        state.lineage_names = [sample.lineage_names[i] for i in order]

        return state

    def _augment_continuous(self, continuous: np.ndarray) -> np.ndarray:
        """Apply spatial data augmentation.

        Args:
            continuous: Array of shape [N, continuous_dim] with genes first

        Returns:
            Augmented continuous array
        """
        if not self.augment_spatial:
            return continuous

        result = continuous.copy()
        spatial = result[:, self._gene_dim : self._gene_dim + self._spatial_dim]

        # Denormalize spatial to [0, 1] range
        spatial_raw = (
            spatial * (self._spatial_max - self._spatial_min) + self._spatial_min
        )

        # Random rotation around z-axis (most common in microscopy)
        if self.aug_rotation and np.random.rand() < 0.5:
            angle = np.random.uniform(0, 2 * np.pi)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            # Rotate x, y coordinates
            x, y = spatial_raw[:, 0].copy(), spatial_raw[:, 1].copy()
            spatial_raw[:, 0] = cos_a * x - sin_a * y
            spatial_raw[:, 1] = sin_a * x + cos_a * y

        # Random flip along axes
        if self.aug_flip:
            if np.random.rand() < 0.5:
                spatial_raw[:, 0] = -spatial_raw[:, 0]  # Flip x
            if np.random.rand() < 0.5:
                spatial_raw[:, 1] = -spatial_raw[:, 1]  # Flip y
            if np.random.rand() < 0.5:
                spatial_raw[:, 2] = -spatial_raw[:, 2]  # Flip z

        # Random scaling
        if self.aug_scale > 0:
            scale = np.random.uniform(1 - self.aug_scale, 1 + self.aug_scale)
            # Scale relative to centroid
            centroid = spatial_raw.mean(axis=0)
            spatial_raw = centroid + (spatial_raw - centroid) * scale

        # Renormalize
        spatial = (spatial_raw - self._spatial_min) / (
            self._spatial_max - self._spatial_min
        ).clip(min=1e-6)

        result[:, self._gene_dim : self._gene_dim + self._spatial_dim] = spatial

        return result

    def _compute_order(
        self,
        n: int,
        sample: _TrimodalSample,
    ) -> list[int]:
        if self.ordering == "lineage":
            return self._lineage_dfs_order(sample.lineage_names)
        if self.ordering == "spatial":
            return self._spatial_curve_order(sample.spatial_raw)
        # random
        order = list(range(n))
        np.random.shuffle(order)
        return order

    def _lineage_dfs_order(self, names: np.ndarray) -> list[int]:
        indexed = sorted(enumerate(names), key=lambda x: x[1])
        return [i for i, _ in indexed]

    def _spatial_curve_order(self, spatial: np.ndarray) -> list[int]:
        scores = spatial.sum(axis=1)
        return list(np.argsort(scores))

    def x0_sampler(self, root: Any) -> tuple[torch.Tensor, int]:
        """Initial distribution sampler."""
        return (torch.randn(self.continuous_dim), self.K - 1)

    def get_cell_names_at(self, idx: int) -> list[str]:
        """Get lineage names for sample (for bias computation)."""
        sample = self._samples[idx]
        return [str(name) for name in sample.lineage_names]

    def denorm_continuous(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Denormalize continuous state back to (expression, spatial, mask).

        Args:
            x: Continuous tensor [..., continuous_dim]
            mask: Optional modality mask [..., 3] (transcriptome, spatial, lineage)

        Returns:
            (expression, spatial, mask) denormalized tensors
        """
        expr = x[..., : self._gene_dim]
        spatial = x[..., self._gene_dim : self._gene_dim + self._spatial_dim]

        # Denormalize expression
        expr_denorm = expr * torch.from_numpy(self._expr_std).to(x) + torch.from_numpy(
            self._expr_mean
        ).to(x)

        # Denormalize spatial
        spatial_range = torch.from_numpy(self._spatial_max - self._spatial_min).to(x)
        spatial_denorm = spatial * spatial_range + torch.from_numpy(
            self._spatial_min
        ).to(x)

        return expr_denorm, spatial_denorm, mask

    def _select_top_hvg(self, adata: Any, n_hvg: int) -> np.ndarray:
        """Select top HVGs by variance rank."""
        if "highly_variable_rank" in adata.var.columns:
            ranks = adata.var["highly_variable_rank"].values.copy()
            ranks[np.isnan(ranks)] = np.inf
            top_idx = np.argsort(ranks)[:n_hvg]
            mask = np.zeros(adata.shape[1], dtype=bool)
            mask[top_idx] = True
            return mask
        # Fall back to variance-based selection without scanpy
        import warnings

        warnings.warn("Selecting HVGs by variance (scanpy not available)")
        expr = adata.X

        # Compute variance efficiently for sparse/dense matrices
        from scipy.sparse import issparse

        if issparse(expr):
            # Sparse matrix: compute mean and variance without densifying
            # Var(X) = E[X^2] - E[X]^2
            mean = np.asarray(expr.mean(axis=0)).flatten()
            sq_mean = np.asarray(expr.power(2).mean(axis=0)).flatten()
            variances = sq_mean - mean**2
        elif hasattr(expr, "mean"):
            variances = np.var(expr, axis=0)
        else:
            # Fallback for other array types
            variances = np.var(np.asarray(expr), axis=0)

        if hasattr(variances, "flatten"):
            variances = variances.flatten()

        top_idx = np.argsort(variances)[-n_hvg:]
        mask = np.zeros(adata.shape[1], dtype=bool)
        mask[top_idx] = True
        return mask


class _TrimodalSample:
    """Internal container for binned trimodal data."""

    __slots__ = (
        "continuous",
        "discrete",
        "lineage_names",
        "spatial_raw",
        "modality_masks",
        "gene_dim",
    )

    def __init__(
        self,
        continuous: np.ndarray,
        discrete: np.ndarray,
        lineage_names: np.ndarray,
        spatial_raw: np.ndarray,
        modality_masks: np.ndarray,
        gene_dim: int,
    ):
        self.continuous = continuous
        self.discrete = discrete
        self.lineage_names = lineage_names
        self.spatial_raw = spatial_raw
        self.modality_masks = modality_masks
        self.gene_dim = gene_dim
