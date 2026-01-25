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
    AncestorMapper,
    CShaperProcessor,
    embryo_time_to_cshaper_frame,
    normalize_lineage_name,
)
from .expression_matcher import ExpressionMatcher

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
        use_ancestor_mapping: bool = True,
        use_expression_matching: bool = False,
        max_ancestor_distance: int = 5,
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
            use_ancestor_mapping: Whether to use ancestor mapping for cells
                not directly in CShaper (inherits data from closest ancestor)
            use_expression_matching: Whether to use expression-based matching
                to validate and improve lineage matches
            max_ancestor_distance: Maximum generations to search for ancestors
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
        
        # Initialize ancestor mapper if requested
        ancestor_mapper = None
        cshaper_cells = self.cshaper.get_all_cell_names()
        if use_ancestor_mapping:
            ancestor_mapper = AncestorMapper(
                cshaper_cells=cshaper_cells,
                max_ancestor_distance=max_ancestor_distance,
            )
            logger.info(
                f"Ancestor mapping enabled: {len(cshaper_cells)} CShaper cells, "
                f"max distance={max_ancestor_distance}"
            )
        
        # Initialize expression matcher if requested
        expression_matcher = None
        if use_expression_matching:
            logger.info("Building expression reference profiles...")
            expression_matcher = ExpressionMatcher(data_dir=self.data_dir)
            try:
                expression_matcher.build_reference_profiles(
                    adata,
                    cshaper_cells=cshaper_cells,
                )
                logger.info("Expression matching enabled")
            except Exception as e:
                logger.warning(f"Failed to build expression profiles: {e}")
                expression_matcher = None
        
        # Step 2: Add CShaper enhancements
        if include_morphology and self.cshaper.has_morphology:
            logger.info("Adding morphology features...")
            adata = self._add_morphology_features(
                adata, 
                ancestor_mapper=ancestor_mapper,
                expression_matcher=expression_matcher,
            )
        
        if include_contact_graph and self.cshaper.has_contact:
            logger.info("Adding contact graph...")
            adata = self._add_contact_graph(
                adata,
                threshold=contact_threshold,
                min_samples=min_contact_samples,
                ancestor_mapper=ancestor_mapper,
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
            use_ancestor_mapping=use_ancestor_mapping,
            max_ancestor_distance=max_ancestor_distance,
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
    
    def _add_morphology_features(
        self,
        adata: ad.AnnData,
        ancestor_mapper: Optional[AncestorMapper] = None,
        expression_matcher: Optional[ExpressionMatcher] = None,
    ) -> ad.AnnData:
        """
        Add cell morphology features (volume, surface, sphericity).
        
        Matches cells to CShaper morphology data by lineage name and
        developmental time. If ancestor_mapper is provided, cells without
        direct CShaper data will inherit from their closest ancestor.
        If expression_matcher is provided, matches are validated/improved
        using gene expression similarity.
        
        Args:
            adata: AnnData object to enhance
            ancestor_mapper: Optional mapper for ancestor-based matching
            expression_matcher: Optional matcher for expression-based validation
        """
        n_cells = adata.n_obs
        
        # Get lineage names
        lineage_col = self._get_lineage_column(adata)
        # Handle categorical columns properly
        lineage_series = adata.obs[lineage_col]
        if hasattr(lineage_series, 'cat'):
            lineages = lineage_series.astype(str).replace('nan', '').tolist()
        else:
            lineages = lineage_series.fillna("").tolist()
        
        # Get embryo times
        time_col = self._get_time_column(adata)
        if time_col is not None:
            embryo_times = adata.obs[time_col].values
        else:
            embryo_times = None
        
        # Apply ancestor mapping if available
        query_lineages = lineages
        ancestor_distances = None
        
        if ancestor_mapper is not None:
            # Map cells to their CShaper ancestors
            mapped_ancestors, ancestor_distances = ancestor_mapper.map_cells(lineages)
            
            # Use ancestor names for query (keep original for unmapped)
            query_lineages = [
                anc if anc is not None else orig
                for anc, orig in zip(mapped_ancestors, lineages)
            ]
            
            # Log mapping statistics
            stats = ancestor_mapper.get_mapping_stats(lineages)
            fuzzy_matches = stats.get('fuzzy_x_matches', 0) + stats.get('fuzzy_slash_matches', 0)
            ancestor_matches = stats.get('ancestor_matches', 0) + stats.get('fuzzy_ancestor_matches', 0)
            logger.info(
                f"Ancestor mapping: {stats['matched_cells']}/{stats['total_cells']} cells "
                f"({100*stats['match_rate']:.1f}%), "
                f"{stats['direct_matches']} direct, {fuzzy_matches} fuzzy, "
                f"{ancestor_matches} via ancestor, mean distance={stats['mean_distance']:.1f}"
            )
        
        # Apply expression matching validation if available
        expression_confidence = None
        expression_match_types = None
        if expression_matcher is not None:
            logger.info("Validating matches with expression profiles...")
            try:
                validated_lineages, expression_confidence, expression_match_types = \
                    expression_matcher.validate_lineage_match(
                        adata,
                        query_lineages,
                        lineage_col=lineage_col,
                    )
                
                # Count expression improvements
                n_confirmed = sum(1 for t in expression_match_types if t == 'confirmed')
                n_corrected = sum(1 for t in expression_match_types if t == 'expression_corrected')
                n_expr_only = sum(1 for t in expression_match_types if t == 'expression_only')
                
                logger.info(
                    f"Expression validation: {n_confirmed} confirmed, "
                    f"{n_corrected} corrected, {n_expr_only} expression-only matches"
                )
                
                # Use validated lineages
                query_lineages = validated_lineages
                
            except Exception as e:
                logger.warning(f"Expression validation failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Get morphology features from CShaper
        try:
            morph_df = self.cshaper.get_morphology_features(
                query_lineages,
                embryo_times=embryo_times,
            )
            
            # Add to obs
            adata.obs["cell_volume"] = morph_df["volume"].values.astype(np.float32)
            adata.obs["cell_surface"] = morph_df["surface"].values.astype(np.float32)
            adata.obs["sphericity"] = morph_df["sphericity"].values.astype(np.float32)
            adata.obs["has_morphology"] = ~np.isnan(morph_df["volume"].values)
            
            # Add ancestor distance if ancestor mapping was used
            if ancestor_distances is not None:
                adata.obs["morphology_ancestor_distance"] = ancestor_distances
            
            # Add expression confidence if expression matching was used
            if expression_confidence is not None:
                adata.obs["morphology_confidence"] = expression_confidence.astype(np.float32)
            if expression_match_types is not None:
                adata.obs["morphology_match_type"] = expression_match_types
            
            # Add CShaper frame mapping
            if embryo_times is not None:
                frames = np.array([
                    embryo_time_to_cshaper_frame(t) if not pd.isna(t) else -1
                    for t in embryo_times
                ], dtype=np.int32)
                adata.obs["cshaper_frame"] = frames
            
            n_matched = adata.obs["has_morphology"].sum()
            n_direct = 0
            n_ancestor = 0
            if ancestor_distances is not None:
                n_direct = int(((ancestor_distances == 0) & adata.obs["has_morphology"].values).sum())
                n_ancestor = int(((ancestor_distances > 0) & adata.obs["has_morphology"].values).sum())
            
            logger.info(
                f"Added morphology features: {n_matched}/{n_cells} cells matched "
                f"({100*n_matched/n_cells:.1f}%)"
                + (f" [{n_direct} direct, {n_ancestor} via ancestor]" if ancestor_mapper else "")
            )
            
        except Exception as e:
            logger.warning(f"Failed to add morphology features: {e}")
            import traceback
            traceback.print_exc()
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
        ancestor_mapper: Optional[AncestorMapper] = None,
    ) -> ad.AnnData:
        """
        Add cell-cell contact adjacency graph.
        
        Creates graphs based on biological reasoning:
        
        1. contact_adjacency/contact_binary: 
           - ONLY for cells that DIRECTLY match CShaper (same developmental stage)
           - Ancestor mapping NOT used here (biologically incorrect to assume
             all descendants of contacting ancestors also contact each other)
        
        2. lineage_proximity (new):
           - For ALL cells with lineage annotations
           - Encodes developmental proximity based on ancestor relationships
           - If ancestors contacted, descendants have higher "lineage proximity"
           - This is a prior/probability, not actual physical contact
        
        Args:
            adata: AnnData object to enhance
            threshold: Minimum contact area threshold
            min_samples: Minimum samples for consensus
            ancestor_mapper: Optional mapper (used for lineage_proximity, not contact)
        """
        n_cells = adata.n_obs
        
        # Get lineage names
        lineage_col = self._get_lineage_column(adata)
        lineages = adata.obs[lineage_col].fillna("").tolist()
        
        try:
            # ========================================
            # 1. TRUE CONTACT GRAPH (direct matches only)
            # ========================================
            # Only use original lineages - no ancestor mapping
            # This gives true physical contacts for early-stage cells
            contact_weighted = self.cshaper.get_contact_adjacency(
                lineages,  # Original lineages, NOT ancestor-mapped
                threshold=threshold,
                binary=False,
            )
            
            contact_binary = self.cshaper.get_contact_adjacency(
                lineages,
                threshold=threshold,
                binary=True,
            )
            
            # Add to obsp
            adata.obsp["contact_adjacency"] = contact_weighted
            adata.obsp["contact_binary"] = contact_binary
            
            # Statistics
            n_edges = contact_binary.nnz // 2
            has_true_contact = np.asarray(contact_binary.sum(axis=1)).flatten() > 0
            n_cells_with_contacts = int(has_true_contact.sum())
            mean_degree = float(contact_binary.sum() / max(n_cells, 1))
            
            # Add indicator for cells with true contact data (useful for training)
            # These cells can serve as supervision signal for link prediction
            adata.obs["has_true_contact"] = has_true_contact
            
            logger.info(
                f"Added contact graph: {n_edges} edges, "
                f"{n_cells_with_contacts}/{n_cells} cells with contacts, "
                f"mean degree {mean_degree:.1f} (direct matches only)"
            )
            
            # ========================================
            # 2. LINEAGE PROXIMITY GRAPH (developmental prior)
            # ========================================
            # This encodes: "cells whose ancestors contacted may share
            # developmental signals and fate decisions"
            if ancestor_mapper is not None:
                self._add_lineage_proximity_graph(
                    adata, lineages, ancestor_mapper, threshold
                )
            
        except Exception as e:
            logger.warning(f"Failed to add contact graph: {e}")
            import traceback
            traceback.print_exc()
            # Add empty sparse matrices
            adata.obsp["contact_adjacency"] = csr_matrix((n_cells, n_cells), dtype=np.float32)
            adata.obsp["contact_binary"] = csr_matrix((n_cells, n_cells), dtype=np.float32)
        
        return adata
    
    def _add_lineage_proximity_graph(
        self,
        adata: ad.AnnData,
        lineages: List[str],
        ancestor_mapper: AncestorMapper,
        threshold: float = 0.0,
    ) -> None:
        """
        Add lineage proximity graph based on ancestor contact relationships.
        
        This graph encodes developmental proximity: cells whose ancestors
        had physical contact during embryogenesis may share signaling
        environments and have related cell fates.
        
        Unlike the contact graph, this is a PRIOR over potential relationships,
        not actual physical contact in the current developmental stage.
        
        The proximity score is:
        - 1.0 if cells share the same CShaper ancestor
        - ancestor_contact_strength / (distance_i + distance_j + 1) otherwise
        
        This naturally decays with lineage distance from the contacting ancestors.
        """
        n_cells = len(lineages)
        
        # Map cells to their CShaper ancestors
        mapped_ancestors, distances = ancestor_mapper.map_cells(lineages)
        
        # Get unique ancestors that have mappings
        unique_ancestors = set(a for a in mapped_ancestors if a is not None)
        
        if len(unique_ancestors) < 2:
            logger.info("Lineage proximity graph: insufficient ancestor mappings")
            adata.obsp["lineage_proximity"] = csr_matrix((n_cells, n_cells), dtype=np.float32)
            return
        
        # Build ancestor -> cell indices mapping
        ancestor_to_cells: Dict[str, List[Tuple[int, int]]] = {}  # ancestor -> [(cell_idx, distance)]
        for i, (anc, dist) in enumerate(zip(mapped_ancestors, distances)):
            if anc is not None:
                if anc not in ancestor_to_cells:
                    ancestor_to_cells[anc] = []
                ancestor_to_cells[anc].append((i, dist))
        
        # Get ancestor contact matrix (small, ~1000x1000)
        ancestor_list = list(unique_ancestors)
        ancestor_contacts = self.cshaper.get_contact_adjacency(
            ancestor_list,
            threshold=threshold,
            binary=False,
        )
        
        # Build proximity graph with decay by lineage distance
        rows = []
        cols = []
        data = []
        
        ancestor_to_idx = {a: i for i, a in enumerate(ancestor_list)}
        
        for ai, anc_i in enumerate(ancestor_list):
            cells_i = ancestor_to_cells.get(anc_i, [])
            if not cells_i:
                continue
            
            for aj, anc_j in enumerate(ancestor_list):
                cells_j = ancestor_to_cells.get(anc_j, [])
                if not cells_j:
                    continue
                
                # Get ancestor contact strength
                if ai == aj:
                    # Same ancestor: siblings have proximity
                    base_strength = 1.0
                else:
                    # Different ancestors: use contact strength
                    base_strength = ancestor_contacts[ai, aj]
                    if base_strength <= 0:
                        continue
                
                # Add edges with distance decay
                # Only add a sample of edges to keep graph manageable
                max_edges_per_pair = 100  # Limit to prevent explosion
                
                for ci, di in cells_i[:max_edges_per_pair]:
                    for cj, dj in cells_j[:max_edges_per_pair]:
                        if ci >= cj:  # Upper triangle only
                            continue
                        # Proximity decays with lineage distance
                        proximity = base_strength / (di + dj + 1)
                        if proximity > 0.1:  # Threshold for sparsity
                            rows.extend([ci, cj])
                            cols.extend([cj, ci])
                            data.extend([proximity, proximity])
        
        if data:
            proximity_matrix = csr_matrix(
                (np.array(data, dtype=np.float32), (np.array(rows), np.array(cols))),
                shape=(n_cells, n_cells),
            )
            adata.obsp["lineage_proximity"] = proximity_matrix
            
            # Add indicator for cells with lineage proximity (candidates for prediction)
            has_proximity = np.asarray(proximity_matrix.sum(axis=1)).flatten() > 0
            adata.obs["has_lineage_proximity"] = has_proximity
            
            n_edges = len(data) // 2
            n_cells_with_prox = int(has_proximity.sum())
            
            # Log training/prediction split info
            has_contact = adata.obs.get("has_true_contact", np.zeros(n_cells, dtype=bool))
            n_train = int((has_contact & has_proximity).sum())  # Have both: can train
            n_predict = int((~has_contact & has_proximity).sum())  # Have proximity but no contact: can predict
            
            logger.info(
                f"Added lineage proximity graph: {n_edges} edges, "
                f"{n_cells_with_prox}/{n_cells} cells"
            )
            logger.info(
                f"Contact prediction split: {n_train} cells for training "
                f"(have true contacts), {n_predict} cells for prediction"
            )
        else:
            adata.obsp["lineage_proximity"] = csr_matrix((n_cells, n_cells), dtype=np.float32)
            adata.obs["has_lineage_proximity"] = False
            logger.info("Lineage proximity graph: no edges above threshold")
    
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
        use_ancestor_mapping: bool = False,
        max_ancestor_distance: int = 5,
    ) -> ad.AnnData:
        """Add CShaper-related metadata to uns."""
        
        cshaper_info = {
            "source": "Cao et al. 2020 (DOI: 10.1038/s41467-020-19863-x)",
            "data_path": str(self.cshaper.cshaper_dir) if self.cshaper else None,
            "include_morphology": include_morphology,
            "include_contact_graph": include_contact_graph,
            "use_cshaper_spatial": use_cshaper_spatial,
            "use_ancestor_mapping": use_ancestor_mapping,
            "max_ancestor_distance": max_ancestor_distance,
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
            
            # Add ancestor mapping statistics if available
            if "morphology_ancestor_distance" in adata.obs.columns:
                distances = adata.obs["morphology_ancestor_distance"].values
                valid_distances = distances[valid_mask]
                cshaper_info["n_direct_morphology_matches"] = int((valid_distances == 0).sum())
                cshaper_info["n_ancestor_morphology_matches"] = int((valid_distances > 0).sum())
                if len(valid_distances) > 0:
                    cshaper_info["mean_ancestor_distance"] = float(np.mean(valid_distances[valid_distances >= 0]))
        
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
            morph_info = f"  Morphology: {n_morph:,} cells ({pct:.1f}%)"
            
            # Add ancestor mapping breakdown if available
            if "morphology_ancestor_distance" in adata.obs.columns:
                distances = adata.obs["morphology_ancestor_distance"].values
                has_morph = adata.obs["has_morphology"].values
                n_direct = int(((distances == 0) & has_morph).sum())
                n_ancestor = int(((distances > 0) & has_morph).sum())
                morph_info += f" [{n_direct:,} direct, {n_ancestor:,} via ancestor]"
            
            cshaper_lines.append(morph_info)
        
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
