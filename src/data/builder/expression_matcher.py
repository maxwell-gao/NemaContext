"""
Expression-based cell matching for CShaper integration.

This module provides expression-based matching to improve alignment between
scRNA-seq data (Large2025) and CShaper morphological data. It uses gene
expression similarity to:

1. Validate lineage-based matches
2. Refine fuzzy/ancestor matches
3. Match cells without lineage annotations via expression profile

The approach:
1. Build reference expression profiles for cell types with known lineages
2. Compute expression similarity between query cells and references
3. Use similarity scores to weight/validate morphology assignments
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


class ExpressionMatcher:
    """
    Match cells to CShaper lineages using expression profiles.
    
    This class builds reference expression profiles for cell types/lineages
    with known CShaper data, then uses expression similarity to match
    query cells to the most likely CShaper lineage.
    
    Usage:
        >>> matcher = ExpressionMatcher(data_dir="dataset/raw")
        >>> matcher.build_reference_profiles(adata)
        >>> matches, scores = matcher.match_cells(adata)
    """
    
    def __init__(
        self,
        data_dir: str = "dataset/raw",
        n_marker_genes: int = 50,
        min_cells_per_lineage: int = 5,
    ):
        """
        Initialize the expression matcher.
        
        Args:
            data_dir: Directory containing reference data files
            n_marker_genes: Number of top marker genes to use per lineage
            min_cells_per_lineage: Minimum cells required to build a profile
        """
        self.data_dir = Path(data_dir)
        self.n_marker_genes = n_marker_genes
        self.min_cells_per_lineage = min_cells_per_lineage
        
        # Load reference data
        self._load_reference_data()
        
        # Reference profiles (built from data)
        self.reference_profiles: Optional[pd.DataFrame] = None
        self.reference_lineages: Optional[List[str]] = None
        self.marker_genes: Optional[List[str]] = None
        
    def _load_reference_data(self):
        """Load cell lineage mapping and cell fates data."""
        # Load lineage map: cell name -> lineage
        lineage_map_path = self.data_dir / "wormbase" / "cell_lineage_map.json"
        if lineage_map_path.exists():
            with open(lineage_map_path) as f:
                self._lineage_map = json.load(f)
            logger.info(f"Loaded {len(self._lineage_map)} cell-lineage mappings")
        else:
            self._lineage_map = {}
            logger.warning(f"Lineage map not found: {lineage_map_path}")
        
        # Build reverse mapping: lineage -> cell name(s)
        self._lineage_to_cellname: Dict[str, List[str]] = {}
        for cell_name, info in self._lineage_map.items():
            lineage = info['lineage'].replace(' ', '').replace('.', '')
            if lineage not in self._lineage_to_cellname:
                self._lineage_to_cellname[lineage] = []
            self._lineage_to_cellname[lineage].append(cell_name)
        
        # Load cell fates: category -> cell names
        fates_path = self.data_dir / "wormbase" / "cell_fates.json"
        if fates_path.exists():
            with open(fates_path) as f:
                self._cell_fates = json.load(f)
            logger.info(f"Loaded {len(self._cell_fates)} cell fate categories")
        else:
            self._cell_fates = {}
            logger.warning(f"Cell fates not found: {fates_path}")
        
        # Build category -> lineages mapping
        self._category_to_lineages: Dict[str, List[str]] = {}
        for category, cell_names in self._cell_fates.items():
            lineages = []
            for cn in cell_names:
                if cn in self._lineage_map:
                    lin = self._lineage_map[cn]['lineage'].replace(' ', '')
                    lineages.append(lin)
            self._category_to_lineages[category] = lineages
    
    def get_lineage_for_cellname(self, cell_name: str) -> Optional[str]:
        """Get lineage for a known cell name."""
        if cell_name in self._lineage_map:
            return self._lineage_map[cell_name]['lineage'].replace(' ', '')
        return None
    
    def get_category_for_celltype(self, cell_type: str) -> Optional[str]:
        """Map a cell type annotation to a cell fate category."""
        cell_type_lower = cell_type.lower()
        
        # Direct mapping rules based on Large2025 cell type naming
        if 'bwm' in cell_type_lower or 'muscle' in cell_type_lower:
            return 'Muscle'
        elif 'hyp' in cell_type_lower or 'hypoderm' in cell_type_lower:
            return 'Hypodermis'
        elif 'intestin' in cell_type_lower or 'int_' in cell_type_lower:
            return 'Intestine'
        elif 'pharyn' in cell_type_lower:
            return 'Pharynx'
        elif 'neuron' in cell_type_lower or cell_type in self._cell_fates.get('Neuron', []):
            return 'Neuron'
        elif 'seam' in cell_type_lower:
            return 'Hypodermis'  # Seam cells are hypodermal
        elif 'excret' in cell_type_lower:
            return 'Excretory'
        elif 'germ' in cell_type_lower:
            return 'Germline'
        elif 'sheath' in cell_type_lower:
            return 'Sheath'
        elif 'socket' in cell_type_lower:
            return 'Socket'
        
        # Check if it's a known cell name
        if cell_type in self._lineage_map:
            # Find which category this cell belongs to
            for cat, cells in self._cell_fates.items():
                if cell_type in cells:
                    return cat
        
        return None
    
    def build_reference_profiles(
        self,
        adata,
        lineage_col: str = "lineage_complete",
        celltype_col: str = "cell_type",
        cshaper_cells: Optional[Set[str]] = None,
        use_hvg: bool = True,
        n_hvg: int = 2000,
    ) -> None:
        """
        Build reference expression profiles from cells with known lineages.
        
        Args:
            adata: AnnData object with expression data
            lineage_col: Column containing lineage annotations
            celltype_col: Column containing cell type annotations
            cshaper_cells: Set of valid CShaper cell names (for filtering)
            use_hvg: Whether to filter to highly variable genes first
            n_hvg: Number of highly variable genes to use
        """
        import scanpy as sc
        
        logger.info("Building reference expression profiles...")
        
        # Get expression matrix
        X = adata.X
        if issparse(X):
            X = X.toarray()
        
        # Get gene names
        gene_names = adata.var_names.tolist()
        
        # Select genes to use
        if use_hvg and 'highly_variable' in adata.var.columns:
            hvg_mask = adata.var['highly_variable'].values
            gene_idx = np.where(hvg_mask)[0][:n_hvg]
        else:
            # Use top variable genes by variance
            gene_var = np.var(X, axis=0)
            gene_idx = np.argsort(gene_var)[-n_hvg:]
        
        selected_genes = [gene_names[i] for i in gene_idx]
        X_selected = X[:, gene_idx]
        
        # Find cells with clean lineage annotations that map to CShaper
        lineages = adata.obs[lineage_col].astype(str).values
        cell_types = adata.obs[celltype_col].astype(str).values
        
        # Group cells by their matched lineage
        lineage_to_cells: Dict[str, List[int]] = {}
        
        for i, (lin, ct) in enumerate(zip(lineages, cell_types)):
            # Skip unassigned
            if lin in ('', 'nan', 'unassigned') or 'x' in lin.lower():
                continue
            
            # Normalize lineage
            lin_clean = lin.replace(' ', '').replace('.', '')
            
            # Handle slash alternatives
            if '/' in lin_clean:
                lin_options = [l.strip() for l in lin_clean.split('/')]
            else:
                lin_options = [lin_clean]
            
            for lin_opt in lin_options:
                # Check if this lineage is in CShaper
                if cshaper_cells and lin_opt not in cshaper_cells:
                    continue
                
                if lin_opt not in lineage_to_cells:
                    lineage_to_cells[lin_opt] = []
                lineage_to_cells[lin_opt].append(i)
        
        logger.info(f"Found {len(lineage_to_cells)} lineages with expression data")
        
        # Build profiles for lineages with enough cells
        profiles = []
        profile_lineages = []
        
        for lineage, cell_indices in lineage_to_cells.items():
            if len(cell_indices) < self.min_cells_per_lineage:
                continue
            
            # Compute mean expression for this lineage
            expr = X_selected[cell_indices, :]
            mean_expr = np.mean(expr, axis=0)
            
            profiles.append(mean_expr)
            profile_lineages.append(lineage)
        
        if not profiles:
            logger.warning("No lineages with sufficient cells for profiles")
            return
        
        # Store profiles
        self.reference_profiles = pd.DataFrame(
            np.array(profiles),
            index=profile_lineages,
            columns=selected_genes,
        )
        self.reference_lineages = profile_lineages
        self.marker_genes = selected_genes
        
        logger.info(
            f"Built {len(profile_lineages)} reference profiles "
            f"using {len(selected_genes)} genes"
        )
    
    def match_cells(
        self,
        adata,
        lineage_col: str = "lineage_complete",
        method: str = "correlation",
        top_k: int = 3,
    ) -> Tuple[List[Optional[str]], np.ndarray]:
        """
        Match cells to reference lineages by expression similarity.
        
        Args:
            adata: AnnData object with expression data
            lineage_col: Column containing lineage annotations
            method: Similarity method ("correlation", "cosine", "euclidean")
            top_k: Number of top matches to consider
            
        Returns:
            Tuple of (matched_lineages, similarity_scores)
        """
        if self.reference_profiles is None:
            raise ValueError("Must call build_reference_profiles first")
        
        n_cells = adata.n_obs
        
        # Get expression matrix
        X = adata.X
        if issparse(X):
            X = X.toarray()
        
        # Select same genes as reference profiles
        gene_names = adata.var_names.tolist()
        gene_idx = [gene_names.index(g) for g in self.marker_genes if g in gene_names]
        
        if len(gene_idx) < len(self.marker_genes) * 0.5:
            logger.warning(
                f"Only {len(gene_idx)}/{len(self.marker_genes)} marker genes found"
            )
        
        X_query = X[:, gene_idx]
        ref_profiles = self.reference_profiles.iloc[:, :len(gene_idx)].values
        
        # Compute similarity
        logger.info(f"Computing expression similarity for {n_cells} cells...")
        
        if method == "correlation":
            # Pearson correlation
            similarity = self._correlation_similarity(X_query, ref_profiles)
        elif method == "cosine":
            similarity = 1 - cdist(X_query, ref_profiles, metric='cosine')
        elif method == "euclidean":
            similarity = -cdist(X_query, ref_profiles, metric='euclidean')
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Get best matches
        matched_lineages = []
        scores = np.zeros(n_cells)
        
        for i in range(n_cells):
            if np.all(np.isnan(similarity[i])):
                matched_lineages.append(None)
                scores[i] = 0
                continue
            
            best_idx = np.nanargmax(similarity[i])
            best_score = similarity[i, best_idx]
            
            if best_score > 0.3:  # Minimum correlation threshold
                matched_lineages.append(self.reference_lineages[best_idx])
                scores[i] = best_score
            else:
                matched_lineages.append(None)
                scores[i] = best_score
        
        n_matched = sum(1 for m in matched_lineages if m is not None)
        logger.info(f"Expression-matched {n_matched}/{n_cells} cells")
        
        return matched_lineages, scores
    
    def _correlation_similarity(
        self,
        X: np.ndarray,
        ref: np.ndarray,
    ) -> np.ndarray:
        """Compute Pearson correlation between each cell and each reference."""
        n_cells = X.shape[0]
        n_refs = ref.shape[0]
        
        # Standardize
        X_mean = np.mean(X, axis=1, keepdims=True)
        X_std = np.std(X, axis=1, keepdims=True) + 1e-8
        X_norm = (X - X_mean) / X_std
        
        ref_mean = np.mean(ref, axis=1, keepdims=True)
        ref_std = np.std(ref, axis=1, keepdims=True) + 1e-8
        ref_norm = (ref - ref_mean) / ref_std
        
        # Correlation = dot product of normalized vectors / n_features
        n_features = X.shape[1]
        corr = np.dot(X_norm, ref_norm.T) / n_features
        
        return corr
    
    def validate_lineage_match(
        self,
        adata,
        proposed_lineages: List[Optional[str]],
        lineage_col: str = "lineage_complete",
        threshold: float = 0.2,
    ) -> Tuple[List[Optional[str]], np.ndarray, List[str]]:
        """
        Validate proposed lineage matches using expression similarity.
        
        Args:
            adata: AnnData object
            proposed_lineages: List of proposed lineage matches
            lineage_col: Lineage column name
            threshold: Minimum similarity to validate
            
        Returns:
            Tuple of (validated_lineages, confidence_scores, match_types)
        """
        if self.reference_profiles is None:
            # Can't validate without profiles
            return proposed_lineages, np.ones(len(proposed_lineages)), ['unvalidated'] * len(proposed_lineages)
        
        # Get expression similarity scores
        expr_matches, expr_scores = self.match_cells(adata, lineage_col)
        
        validated = []
        confidences = []
        match_types = []
        
        for i, (proposed, expr_match, expr_score) in enumerate(
            zip(proposed_lineages, expr_matches, expr_scores)
        ):
            if proposed is None:
                # Use expression match if available
                if expr_match is not None and expr_score > threshold:
                    validated.append(expr_match)
                    confidences.append(float(expr_score))
                    match_types.append('expression_only')
                else:
                    validated.append(None)
                    confidences.append(0.0)
                    match_types.append('unmatched')
            else:
                # Validate proposed match
                if expr_match == proposed:
                    # Expression confirms lineage match
                    validated.append(proposed)
                    confidences.append(float(max(expr_score, 0.5)))
                    match_types.append('confirmed')
                elif expr_score > threshold and expr_match is not None:
                    # Expression suggests different lineage
                    # Use expression match if score is higher
                    if expr_score > 0.5:
                        validated.append(expr_match)
                        confidences.append(float(expr_score))
                        match_types.append('expression_corrected')
                    else:
                        validated.append(proposed)
                        confidences.append(float(0.5 - abs(expr_score - 0.5)))
                        match_types.append('lineage_preferred')
                else:
                    # Keep proposed match with lower confidence
                    validated.append(proposed)
                    confidences.append(float(0.3))
                    match_types.append('unvalidated')
        
        return validated, np.array(confidences), match_types
    
    def get_matching_stats(
        self,
        match_types: List[str],
    ) -> Dict[str, int]:
        """Get statistics about match types."""
        stats = {}
        for mt in match_types:
            stats[mt] = stats.get(mt, 0) + 1
        return stats


class CellTypeMapper:
    """
    Maps broad cell type annotations to specific lineages.
    
    Large2025 uses broad categories like "BWM_middle" (body wall muscle, middle),
    while CShaper tracks specific lineages. This class helps bridge the gap.
    """
    
    # Mapping from Large2025 cell types to cell fate categories
    CELLTYPE_TO_CATEGORY = {
        # Muscle types
        'BWM_': 'Muscle',
        'mu_': 'Muscle',
        'muscle': 'Muscle',
        # Hypodermis
        'Hyp': 'Hypodermis',
        'hyp': 'Hypodermis',
        'Seam': 'Hypodermis',
        # Intestine
        'Intestin': 'Intestine',
        'int_': 'Intestine',
        # Pharynx
        'Pharyn': 'Pharynx',
        # Neurons
        'Neuron': 'Neuron',
        # Germline
        'Germ': 'Germline',
        # Excretory
        'Excret': 'Excretory',
    }
    
    def __init__(self, data_dir: str = "dataset/raw"):
        """Initialize the cell type mapper."""
        self.data_dir = Path(data_dir)
        self._load_data()
    
    def _load_data(self):
        """Load cell fates and lineage map."""
        # Load cell fates
        fates_path = self.data_dir / "wormbase" / "cell_fates.json"
        if fates_path.exists():
            with open(fates_path) as f:
                self._cell_fates = json.load(f)
        else:
            self._cell_fates = {}
        
        # Load lineage map
        map_path = self.data_dir / "wormbase" / "cell_lineage_map.json"
        if map_path.exists():
            with open(map_path) as f:
                self._lineage_map = json.load(f)
        else:
            self._lineage_map = {}
        
        # Build category -> lineages
        self._category_lineages: Dict[str, List[str]] = {}
        for category, cell_names in self._cell_fates.items():
            lineages = []
            for cn in cell_names:
                if cn in self._lineage_map:
                    lin = self._lineage_map[cn]['lineage'].replace(' ', '')
                    lineages.append(lin)
            self._category_lineages[category] = lineages
    
    def get_category(self, cell_type: str) -> Optional[str]:
        """Get cell fate category for a cell type."""
        for prefix, category in self.CELLTYPE_TO_CATEGORY.items():
            if cell_type.startswith(prefix) or prefix.lower() in cell_type.lower():
                return category
        return None
    
    def get_possible_lineages(self, cell_type: str) -> List[str]:
        """Get possible lineages for a cell type based on category."""
        category = self.get_category(cell_type)
        if category:
            return self._category_lineages.get(category, [])
        return []
