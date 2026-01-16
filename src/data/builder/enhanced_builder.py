"""
Enhanced AnnData builder with CShaper integration.

This module extends TrimodalAnnDataBuilder to incorporate CShaper morphological
data, including:
- Cell-cell contact graphs (true physical neighbors)
- Cell morphology features (volume, surface area, sphericity)
- Standardized spatial coordinates (optional)

The enhanced builder creates AnnData objects with additional:
- obs columns: cell_volume, cell_surface, sphericity, has_morphology
- obsp matrices: contact_adjacency, contact_binary
- obsm arrays: X_cshaper_spatial (optional)
"""

import logging
from pathlib import Path
from typing import Dict, Literal, Optional

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .anndata_builder import TrimodalAnnDataBuilder
from .cshaper_processor import (
    CShaperProcessor,
    embryo_time_to_cshaper_frame,
    normalize_lineage_name,
)

logger = logging.getLogger(__name__)


class EnhancedAnnDataBuilder(TrimodalAnnDataBuilder):
    """
    Enhanced AnnData builder with CShaper morphological data integration.
    
    Extends TrimodalAnnDataBuilder to add:
    - Cell-cell contact adjacency from CShaper
    - Cell morphology features (volume, surface, sphericity)
    - Optionally enhanced spatial coordinates from Standard Dataset 1
    
    Usage:
        >>> builder = EnhancedAnnDataBuilder()
        >>> adata = builder.build_with_cshaper(variant="complete")
        >>> adata.write("nema_enhanced.h5ad")
    """
    
    def __init__(
        self,
        data_dir: str = "dataset/raw",
        output_dir: str = "dataset/processed",
    ):
        """
        Initialize the enhanced builder.
        
        Args:
            data_dir: Directory containing raw data files
            output_dir: Directory for output h5ad files
        """
        super().__init__(data_dir, output_dir)
        
        # Initialize CShaper processor
        try:
            self.cshaper = CShaperProcessor(data_dir)
            self.has_cshaper = True
            logger.info("CShaper processor initialized")
        except Exception as e:
            logger.warning(f"CShaper data not available: {e}")
            self.cshaper = None
            self.has_cshaper = False
    
    def build_with_cshaper(
        self,
        variant: Literal["complete", "extended"] = "complete",
        source: Literal["large2025", "packer2019"] = "large2025",
        include_morphology: bool = True,
        include_contact_graph: bool = True,
        use_cshaper_spatial: bool = False,
        contact_threshold: float = 0.0,
        min_contact_samples: int = 1,
        save: bool = True,
        **kwargs,
    ) -> ad.AnnData:
        """
        Build enhanced AnnData with CShaper integration.
        
        Args:
            variant: Which variant to build ("complete" or "extended")
            source: Transcriptome data source ("large2025" or "packer2019")
            include_morphology: Whether to add cell morphology features
            include_contact_graph: Whether to add contact adjacency graph
            use_cshaper_spatial: Whether to add CShaper spatial coordinates
            contact_threshold: Minimum contact area to include edge (μm²)
            min_contact_samples: Minimum samples for consensus contact
            save: Whether to save the result
            **kwargs: Additional arguments passed to base build()
            
        Returns:
            Enhanced AnnData object
        """
        logger.info(f"Building enhanced AnnData from {source} ({variant})")
        
        # Step 1: Build base trimodal AnnData (don't save yet)
        adata = self.build(
            variant=variant,
            source=source,
            save=False,  # We'll save after enhancement
            **kwargs,
        )
        
        if not self.has_cshaper:
            logger.warning("CShaper data not available, skipping enhancement")
            if save:
                self._save_enhanced(adata, variant, source)
            return adata
        
        # Step 2: Add CShaper enhancements
        if include_morphology and self.cshaper.has_morphology:
            logger.info("Adding morphology features...")
            adata = self._add_morphology_features(adata)
        
        if include_contact_graph and self.cshaper.has_contact:
            logger.info("Adding contact graph...")
            adata = self._add_contact_graph(
                adata,
                threshold=contact_threshold,
                min_samples=min_contact_samples,
            )
        
        if use_cshaper_spatial and self.cshaper.has_standard_spatial:
            logger.info("Adding CShaper spatial coordinates...")
            adata = self._add_cshaper_spatial(adata)
        
        # Step 3: Update metadata
        adata = self._add_cshaper_metadata(
            adata,
            include_morphology=include_morphology,
            include_contact_graph=include_contact_graph,
            use_cshaper_spatial=use_cshaper_spatial,
        )
        
        # Step 4: Save if requested
        if save:
            self._save_enhanced(adata, variant, source)
        
        return adata
    
    def _get_lineage_column(self, adata: ad.AnnData) -> str:
        """Get the lineage column name from AnnData."""
        for col in ["lineage_complete", "lineage"]:
            if col in adata.obs.columns:
                return col
        raise ValueError("No lineage column found in AnnData")
    
    def _get_time_column(self, adata: ad.AnnData) -> Optional[str]:
        """Get the embryo time column name from AnnData."""
        for col in ["embryo_time_min", "smoothed_embryo_time"]:
            if col in adata.obs.columns:
                return col
        return None
    
    def _add_morphology_features(self, adata: ad.AnnData) -> ad.AnnData:
        """
        Add cell morphology features (volume, surface, sphericity).
        
        Matches cells to CShaper morphology data by lineage name and
        developmental time.
        """
        n_cells = adata.n_obs
        
        # Get lineage names
        lineage_col = self._get_lineage_column(adata)
        lineages = adata.obs[lineage_col].fillna("").tolist()
        
        # Get embryo times
        time_col = self._get_time_column(adata)
        if time_col is not None:
            embryo_times = adata.obs[time_col].values
        else:
            embryo_times = None
        
        # Get morphology features from CShaper
        try:
            morph_df = self.cshaper.get_morphology_features(
                lineages,
                embryo_times=embryo_times,
            )
            
            # Add to obs
            adata.obs["cell_volume"] = morph_df["volume"].values.astype(np.float32)
            adata.obs["cell_surface"] = morph_df["surface"].values.astype(np.float32)
            adata.obs["sphericity"] = morph_df["sphericity"].values.astype(np.float32)
            adata.obs["has_morphology"] = ~np.isnan(morph_df["volume"].values)
            
            # Add CShaper frame mapping
            if embryo_times is not None:
                frames = np.array([
                    embryo_time_to_cshaper_frame(t) if not pd.isna(t) else -1
                    for t in embryo_times
                ], dtype=np.int32)
                adata.obs["cshaper_frame"] = frames
            
            n_matched = adata.obs["has_morphology"].sum()
            logger.info(
                f"Added morphology features: {n_matched}/{n_cells} cells matched "
                f"({100*n_matched/n_cells:.1f}%)"
            )
            
        except Exception as e:
            logger.warning(f"Failed to add morphology features: {e}")
            # Add empty columns
            adata.obs["cell_volume"] = np.nan
            adata.obs["cell_surface"] = np.nan
            adata.obs["sphericity"] = np.nan
            adata.obs["has_morphology"] = False
        
        return adata
    
    def _add_contact_graph(
        self,
        adata: ad.AnnData,
        threshold: float = 0.0,
        min_samples: int = 1,
    ) -> ad.AnnData:
        """
        Add cell-cell contact adjacency graph.
        
        Creates two graphs:
        - contact_adjacency: Weighted by contact surface area
        - contact_binary: Binary (connected/not connected)
        """
        n_cells = adata.n_obs
        
        # Get lineage names
        lineage_col = self._get_lineage_column(adata)
        lineages = adata.obs[lineage_col].fillna("").tolist()
        
        try:
            # Build weighted adjacency matrix
            contact_weighted = self.cshaper.get_contact_adjacency(
                lineages,
                threshold=threshold,
                binary=False,
            )
            
            # Build binary adjacency matrix
            contact_binary = self.cshaper.get_contact_adjacency(
                lineages,
                threshold=threshold,
                binary=True,
            )
            
            # Add to obsp
            adata.obsp["contact_adjacency"] = contact_weighted
            adata.obsp["contact_binary"] = contact_binary
            
            # Statistics
            n_edges = contact_binary.nnz // 2  # Divide by 2 for undirected
            n_cells_with_contacts = (contact_binary.sum(axis=1) > 0).sum()
            mean_degree = contact_binary.sum() / max(n_cells, 1)
            
            logger.info(
                f"Added contact graph: {n_edges} edges, "
                f"{n_cells_with_contacts}/{n_cells} cells with contacts, "
                f"mean degree {mean_degree:.1f}"
            )
            
        except Exception as e:
            logger.warning(f"Failed to add contact graph: {e}")
            # Add empty sparse matrices
            adata.obsp["contact_adjacency"] = csr_matrix((n_cells, n_cells), dtype=np.float32)
            adata.obsp["contact_binary"] = csr_matrix((n_cells, n_cells), dtype=np.float32)
        
        return adata
    
    def _add_cshaper_spatial(self, adata: ad.AnnData) -> ad.AnnData:
        """
        Add CShaper standardized spatial coordinates.
        
        These complement/replace WormGUIDES coordinates with coordinates
        averaged across 46 embryos in Standard Dataset 1.
        """
        n_cells = adata.n_obs
        
        # Get lineage names
        lineage_col = self._get_lineage_column(adata)
        lineages = adata.obs[lineage_col].fillna("").tolist()
        
        # Get embryo times
        time_col = self._get_time_column(adata)
        if time_col is not None:
            embryo_times = adata.obs[time_col].values
        else:
            embryo_times = None
        
        try:
            # Get CShaper spatial coordinates
            cshaper_coords = self.cshaper.get_spatial_coords(lineages, embryo_times)
            
            # Add to obsm
            adata.obsm["X_cshaper_spatial"] = cshaper_coords
            
            n_matched = (~np.isnan(cshaper_coords[:, 0])).sum()
            logger.info(
                f"Added CShaper spatial: {n_matched}/{n_cells} cells matched "
                f"({100*n_matched/n_cells:.1f}%)"
            )
            
        except Exception as e:
            logger.warning(f"Failed to add CShaper spatial: {e}")
            adata.obsm["X_cshaper_spatial"] = np.full((n_cells, 3), np.nan, dtype=np.float32)
        
        return adata
    
    def _add_cshaper_metadata(
        self,
        adata: ad.AnnData,
        include_morphology: bool,
        include_contact_graph: bool,
        use_cshaper_spatial: bool,
    ) -> ad.AnnData:
        """Add CShaper-related metadata to uns."""
        
        cshaper_info = {
            "source": "Cao et al. 2020 (DOI: 10.1038/s41467-020-19863-x)",
            "data_path": str(self.cshaper.cshaper_dir) if self.cshaper else None,
            "include_morphology": include_morphology,
            "include_contact_graph": include_contact_graph,
            "use_cshaper_spatial": use_cshaper_spatial,
        }
        
        # Add morphology statistics
        if include_morphology and "has_morphology" in adata.obs.columns:
            n_with_morphology = int(adata.obs["has_morphology"].sum())
            cshaper_info["n_cells_with_morphology"] = n_with_morphology
            
            if n_with_morphology > 0:
                valid_mask = adata.obs["has_morphology"].values
                cshaper_info["mean_volume"] = float(adata.obs.loc[valid_mask, "cell_volume"].mean())
                cshaper_info["mean_surface"] = float(adata.obs.loc[valid_mask, "cell_surface"].mean())
                cshaper_info["mean_sphericity"] = float(adata.obs.loc[valid_mask, "sphericity"].mean())
        
        # Add contact graph statistics
        if include_contact_graph and "contact_binary" in adata.obsp:
            contact_mat = adata.obsp["contact_binary"]
            n_edges = contact_mat.nnz // 2
            n_cells_with_contacts = int((contact_mat.sum(axis=1) > 0).sum())
            
            cshaper_info["n_contact_edges"] = n_edges
            cshaper_info["n_cells_with_contacts"] = n_cells_with_contacts
            cshaper_info["mean_contact_degree"] = float(contact_mat.sum() / max(adata.n_obs, 1))
        
        # Add CShaper spatial statistics
        if use_cshaper_spatial and "X_cshaper_spatial" in adata.obsm:
            coords = adata.obsm["X_cshaper_spatial"]
            n_with_spatial = int((~np.isnan(coords[:, 0])).sum())
            cshaper_info["n_cells_with_cshaper_spatial"] = n_with_spatial
        
        adata.uns["cshaper_info"] = cshaper_info
        
        # Update data sources
        if "data_sources" in adata.uns:
            adata.uns["data_sources"]["morphology"] = "CShaper (Cao et al. 2020)"
        
        return adata
    
    def _save_enhanced(
        self,
        adata: ad.AnnData,
        variant: str,
        source: str,
    ) -> None:
        """Save enhanced AnnData to disk."""
        output_path = self.output_dir / f"nema_enhanced_{variant}_{source}.h5ad"
        logger.info(f"Saving enhanced AnnData to {output_path}")
        adata.write(output_path, compression="gzip")
        logger.info(f"Saved: {adata.n_obs} cells, {adata.n_vars} genes")
    
    def summary(self, adata: ad.AnnData) -> str:
        """
        Generate an enhanced summary of the AnnData object.
        
        Extends base summary with CShaper-specific information.
        """
        # Get base summary
        base_summary = super().summary(adata)
        
        # Add CShaper section
        lines = base_summary.split("\n")
        
        # Find where to insert CShaper info (before "Available embeddings")
        insert_idx = next(
            (i for i, line in enumerate(lines) if "Available embeddings" in line),
            len(lines) - 2
        )
        
        cshaper_lines = ["", "CShaper Enhancement:"]
        
        if "has_morphology" in adata.obs.columns:
            n_morph = adata.obs["has_morphology"].sum()
            pct = 100 * n_morph / adata.n_obs
            cshaper_lines.append(f"  Morphology: {n_morph:,} cells ({pct:.1f}%)")
        
        if "contact_binary" in adata.obsp:
            n_edges = adata.obsp["contact_binary"].nnz // 2
            cshaper_lines.append(f"  Contact graph: {n_edges:,} edges")
        
        if "X_cshaper_spatial" in adata.obsm:
            n_spatial = (~np.isnan(adata.obsm["X_cshaper_spatial"][:, 0])).sum()
            pct = 100 * n_spatial / adata.n_obs
            cshaper_lines.append(f"  CShaper spatial: {n_spatial:,} cells ({pct:.1f}%)")
        
        # Insert CShaper lines
        lines = lines[:insert_idx] + cshaper_lines + lines[insert_idx:]
        
        return "\n".join(lines)
    
    def compare_graphs(self, adata: ad.AnnData) -> Dict[str, float]:
        """
        Compare contact graph with k-NN spatial graph.
        
        Useful for understanding how true physical contacts differ
        from proximity-based approximations.
        
        Args:
            adata: AnnData with both graphs
            
        Returns:
            Dictionary with comparison metrics
        """
        if "contact_binary" not in adata.obsp:
            raise ValueError("No contact graph in AnnData")
        if "spatial_connectivities" not in adata.obsp:
            raise ValueError("No spatial graph in AnnData")
        
        contact = adata.obsp["contact_binary"].toarray() > 0
        spatial = adata.obsp["spatial_connectivities"].toarray() > 0
        
        # Compute overlap metrics
        intersection = (contact & spatial).sum() / 2  # Divide for undirected
        contact_only = (contact & ~spatial).sum() / 2
        spatial_only = (~contact & spatial).sum() / 2
        
        contact_edges = contact.sum() / 2
        spatial_edges = spatial.sum() / 2
        
        # Jaccard similarity
        union = (contact | spatial).sum() / 2
        jaccard = intersection / union if union > 0 else 0
        
        # Precision/recall (treating contact as ground truth)
        precision = intersection / spatial_edges if spatial_edges > 0 else 0
        recall = intersection / contact_edges if contact_edges > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "contact_edges": int(contact_edges),
            "spatial_edges": int(spatial_edges),
            "intersection": int(intersection),
            "contact_only": int(contact_only),
            "spatial_only": int(spatial_only),
            "jaccard": float(jaccard),
            "precision": float(precision),  # How many spatial edges are true contacts
            "recall": float(recall),  # How many contacts are captured by spatial
            "f1": float(f1),
        }
