"""
Trimodal AnnData builder for NemaContext.

This module provides the main builder class that orchestrates the construction
of AnnData objects integrating:
- Transcriptome data (Large2025/Packer2019)
- Spatial coordinates (WormGUIDES 4D nuclei positions)
- Lineage information (binary path encoding, tree structure)

Two output variants are supported:
1. Complete trimodal: Only cells with all three modalities (~3.8k cells)
2. Extended: All cells with available modality flags (~242k cells)
"""

import logging
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .expression_loader import ExpressionLoader
from .lineage_encoder import LineageEncoder
from .spatial_matcher import SpatialMatcher
from .worm_atlas import WormAtlasMapper

logger = logging.getLogger(__name__)


class TrimodalAnnDataBuilder:
    """
    Builder for trimodal AnnData objects combining transcriptome, spatial, and lineage data.

    This class orchestrates the integration of:
    - Expression matrices from Large2025 or Packer2019
    - 4D spatial coordinates from WormGUIDES
    - Lineage tree structure and encodings from WormBase

    Usage:
        >>> builder = TrimodalAnnDataBuilder()
        >>> adata = builder.build(variant="complete")
        >>> adata.write("nema_trimodal.h5ad")
    """

    # Time window for WormGUIDES data (minutes post-fertilization)
    WORMGUIDES_TIME_MIN = 20
    WORMGUIDES_TIME_MAX = 380

    # Modality flags (bitmask)
    MODALITY_TRANSCRIPTOME = 1
    MODALITY_SPATIAL = 2
    MODALITY_LINEAGE = 4

    def __init__(
        self,
        data_dir: str = "dataset/raw",
        output_dir: str = "dataset/processed",
    ):
        """
        Initialize the AnnData builder.

        Args:
            data_dir: Directory containing raw data files.
            output_dir: Directory for output h5ad files.
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize component loaders
        self.expression_loader = ExpressionLoader(data_dir)
        self.spatial_matcher = SpatialMatcher(data_dir)
        self.lineage_encoder = LineageEncoder(data_dir)
        self.worm_atlas = WormAtlasMapper(data_dir)

        logger.info(f"TrimodalAnnDataBuilder initialized with data_dir={data_dir}")

    def build(
        self,
        variant: Literal["complete", "extended"] = "complete",
        source: Literal["large2025", "packer2019"] = "large2025",
        species_filter: Optional[str] = "C.elegans",
        min_umi: int = 500,
        normalize: bool = True,
        compute_pca: bool = True,
        n_pcs: int = 50,
        save: bool = True,
    ) -> ad.AnnData:
        """
        Build the trimodal AnnData object.

        Args:
            variant: Which variant to build:
                - "complete": Only cells with all three modalities
                - "extended": All cells with modality availability flags
            source: Transcriptome data source ("large2025" or "packer2019").
            species_filter: Filter to specific species (None for all).
            min_umi: Minimum UMI count per cell.
            normalize: Whether to log-normalize expression data.
            compute_pca: Whether to compute PCA embedding.
            n_pcs: Number of principal components.
            save: Whether to save the result to disk.

        Returns:
            Constructed AnnData object.
        """
        logger.info(f"Building {variant} AnnData from {source}")

        # Step 1: Load expression data
        logger.info("Step 1/6: Loading expression data...")
        expr_matrix, cell_df, gene_df = self._load_expression(
            source=source,
            species_filter=species_filter,
            min_umi=min_umi,
        )

        # Step 2: Parse and encode lineage information
        logger.info("Step 2/6: Encoding lineage information...")
        lineage_df = self._encode_lineage(cell_df)

        # Step 3: Match cells to spatial coordinates
        logger.info("Step 3/6: Matching spatial coordinates...")
        spatial_data = self._match_spatial(cell_df, lineage_df)

        # Step 4: Filter based on variant
        logger.info("Step 4/6: Applying variant filters...")
        if variant == "complete":
            mask = self._get_complete_mask(lineage_df, spatial_data)
            expr_matrix = expr_matrix[mask]
            cell_df = cell_df[mask].reset_index(drop=True)
            lineage_df = lineage_df[mask].reset_index(drop=True)
            spatial_data = {k: v[mask] for k, v in spatial_data.items()}
            logger.info(f"Filtered to {mask.sum()} cells with complete trimodal data")

        # Step 5: Construct AnnData
        logger.info("Step 5/6: Constructing AnnData object...")
        adata = self._construct_anndata(
            expr_matrix=expr_matrix,
            cell_df=cell_df,
            gene_df=gene_df,
            lineage_df=lineage_df,
            spatial_data=spatial_data,
            normalize=normalize,
            compute_pca=compute_pca,
            n_pcs=n_pcs,
        )

        # Step 6: Add metadata
        logger.info("Step 6/6: Adding metadata...")
        adata = self._add_metadata(adata, variant=variant, source=source)

        # Save if requested
        if save:
            output_path = self.output_dir / f"nema_{variant}_{source}.h5ad"
            logger.info(f"Saving to {output_path}")
            adata.write(output_path, compression="gzip")
            logger.info(f"Saved AnnData with {adata.n_obs} cells, {adata.n_vars} genes")

        return adata

    def _load_expression(
        self,
        source: str,
        species_filter: Optional[str],
        min_umi: int,
    ) -> Tuple[csr_matrix, pd.DataFrame, pd.DataFrame]:
        """Load expression matrix and annotations."""
        if source == "large2025":
            expr_matrix, cell_df, gene_df = self.expression_loader.load_large2025(
                species_filter=species_filter,
                min_umi=min_umi,
            )
        elif source == "packer2019":
            expr_matrix, cell_df, gene_df = self.expression_loader.load_packer2019(
                min_umi=min_umi,
            )
        else:
            raise ValueError(f"Unknown source: {source}")

        logger.info(
            f"Loaded {expr_matrix.shape[0]} cells, {expr_matrix.shape[1]} genes"
        )
        return expr_matrix, cell_df, gene_df

    def _encode_lineage(self, cell_df: pd.DataFrame) -> pd.DataFrame:
        """Encode lineage information for all cells."""
        # Get lineage column (varies by dataset)
        lineage_col = None
        for col in ["lineage_complete", "lineage", "packer_lineage"]:
            if col in cell_df.columns:
                lineage_col = col
                break

        if lineage_col is None:
            logger.warning("No lineage column found, using empty lineages")
            lineages = pd.Series([""] * len(cell_df))
        else:
            lineages = cell_df[lineage_col].fillna("")

        # Encode using LineageEncoder
        lineage_df = self.lineage_encoder.encode_for_dataframe(lineages)

        # Add binary encoding matrix
        binary_encodings = self.lineage_encoder.encode_batch(lineages.tolist())
        lineage_df["lineage_binary_array"] = list(binary_encodings)

        return lineage_df

    def _match_spatial(
        self,
        cell_df: pd.DataFrame,
        lineage_df: pd.DataFrame,
    ) -> Dict[str, np.ndarray]:
        """Match cells to WormGUIDES spatial coordinates."""
        n_cells = len(cell_df)

        # Initialize output arrays
        spatial_coords = np.full((n_cells, 3), np.nan, dtype=np.float32)
        matched_mask = np.zeros(n_cells, dtype=bool)
        matched_timepoints = np.full(n_cells, -1, dtype=np.int32)

        # Get lineage column for matching
        lineage_col = None
        for col in ["lineage_complete", "lineage"]:
            if col in cell_df.columns:
                lineage_col = col
                break

        if lineage_col is None:
            logger.warning("No lineage column for spatial matching")
            return {
                "spatial_coords": spatial_coords,
                "spatial_matched": matched_mask,
                "matched_timepoints": matched_timepoints,
            }

        # Get clean lineages (valid for matching)
        clean_mask = lineage_df["lineage_valid"].values
        lineages = cell_df[lineage_col].fillna("").tolist()

        # Get embryo time if available
        time_col = None
        for col in ["smoothed_embryo_time", "smoothed.embryo.time", "embryo_time"]:
            col_clean = col.replace(".", "_")
            if col in cell_df.columns:
                time_col = col
                break
            elif col_clean in cell_df.columns:
                time_col = col_clean
                break

        try:
            if time_col is not None:
                # Match by time
                cell_times = cell_df[time_col].values
                coords, mask, tps = self.spatial_matcher.match_by_time(
                    cell_times=cell_times,
                    lineage_names=lineages,
                )
            else:
                # Match by lineage only (average across timepoints)
                coords, mask, tps = self.spatial_matcher.match_by_lineage(
                    lineage_names=lineages,
                    target_time=None,
                )

            spatial_coords = coords
            matched_mask = mask
            matched_timepoints = tps

        except FileNotFoundError as e:
            logger.warning(f"Spatial matching failed: {e}")

        n_matched = matched_mask.sum()
        logger.info(f"Matched {n_matched}/{n_cells} cells to spatial coordinates")

        return {
            "spatial_coords": spatial_coords,
            "spatial_matched": matched_mask,
            "matched_timepoints": matched_timepoints,
        }

    def _get_complete_mask(
        self,
        lineage_df: pd.DataFrame,
        spatial_data: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Get mask for cells with complete trimodal data."""
        # Has valid lineage
        has_lineage = lineage_df["lineage_valid"].values

        # Has spatial match
        has_spatial = spatial_data["spatial_matched"]

        # Combined mask
        complete_mask = has_lineage & has_spatial

        logger.info(
            f"Complete mask: {has_lineage.sum()} with lineage, "
            f"{has_spatial.sum()} with spatial, "
            f"{complete_mask.sum()} with both"
        )

        return complete_mask

    def _construct_anndata(
        self,
        expr_matrix: csr_matrix,
        cell_df: pd.DataFrame,
        gene_df: pd.DataFrame,
        lineage_df: pd.DataFrame,
        spatial_data: Dict[str, np.ndarray],
        normalize: bool,
        compute_pca: bool,
        n_pcs: int,
    ) -> ad.AnnData:
        """Construct the AnnData object."""
        # Create base AnnData
        adata = ad.AnnData(X=expr_matrix.astype(np.float32))

        # Set observation names (cell barcodes)
        if "barcode" in cell_df.columns:
            adata.obs_names = cell_df["barcode"].astype(str).values
        else:
            adata.obs_names = [f"cell_{i}" for i in range(len(cell_df))]

        # Set variable names (genes)
        if "gene_id" in gene_df.columns:
            adata.var_names = gene_df["gene_id"].astype(str).values
        elif len(gene_df.columns) > 0:
            adata.var_names = gene_df.iloc[:, 0].astype(str).values
        else:
            adata.var_names = [f"gene_{i}" for i in range(gene_df.shape[0])]

        # === obs: Cell metadata ===
        obs_columns = [
            "cell_type",
            "species",
            "batch",
            "n_umi",
            "lineage_complete",
            "lineage",
        ]
        for col in obs_columns:
            if col in cell_df.columns:
                adata.obs[col] = cell_df[col].values

        # Add time column with standardized name
        for col in ["smoothed_embryo_time", "smoothed.embryo.time", "embryo_time"]:
            col_clean = col.replace(".", "_")
            if col in cell_df.columns:
                adata.obs["embryo_time_min"] = cell_df[col].values
                break
            elif col_clean in cell_df.columns:
                adata.obs["embryo_time_min"] = cell_df[col_clean].values
                break

        # Add lineage encodings
        adata.obs["lineage_valid"] = lineage_df["lineage_valid"].values
        adata.obs["lineage_founder"] = pd.Categorical(
            lineage_df["lineage_founder"].values
        )
        adata.obs["lineage_depth"] = lineage_df["lineage_depth"].values

        # Add spatial matching info
        adata.obs["has_spatial"] = spatial_data["spatial_matched"]
        adata.obs["wormguides_timepoint"] = spatial_data["matched_timepoints"]

        # Add modality mask
        modality_mask = np.ones(len(adata), dtype=np.int8) * self.MODALITY_TRANSCRIPTOME
        modality_mask[lineage_df["lineage_valid"].values] |= self.MODALITY_LINEAGE
        modality_mask[spatial_data["spatial_matched"]] |= self.MODALITY_SPATIAL
        adata.obs["modality_mask"] = modality_mask

        # === var: Gene metadata ===
        for col in gene_df.columns:
            adata.var[col] = gene_df[col].values

        # === obsm: Embeddings ===
        # Spatial coordinates
        adata.obsm["X_spatial"] = spatial_data["spatial_coords"]

        # Lineage binary encoding
        binary_arrays = np.stack(lineage_df["lineage_binary_array"].values)
        adata.obsm["X_lineage_binary"] = binary_arrays.astype(np.float32)

        # === layers: Alternative expression representations ===
        adata.layers["counts"] = expr_matrix.copy()

        if normalize:
            # Log-normalize
            import scanpy as sc

            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            adata.layers["log1p"] = adata.X.copy()

        # === Compute PCA if requested ===
        if compute_pca and normalize:
            import scanpy as sc

            # Use highly variable genes for PCA
            sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3")
            sc.pp.pca(adata, n_comps=min(n_pcs, adata.n_obs - 1, adata.n_vars - 1))

        return adata

    def _add_metadata(
        self,
        adata: ad.AnnData,
        variant: str,
        source: str,
    ) -> ad.AnnData:
        """Add unstructured metadata to AnnData."""
        # Data sources
        adata.uns["data_sources"] = {
            "transcriptome": "Large et al. 2025 (GSE292756)"
            if source == "large2025"
            else "Packer et al. 2019 (GSE126954)",
            "spatial": "WormGUIDES (Bao Lab)",
            "lineage": "WormBase / Sulston 1983",
        }

        # Build parameters
        adata.uns["build_params"] = {
            "variant": variant,
            "source": source,
            "wormguides_time_range": [
                self.WORMGUIDES_TIME_MIN,
                self.WORMGUIDES_TIME_MAX,
            ],
        }

        # Lineage tree (if available)
        if self.lineage_encoder.lineage_tree:
            adata.uns["lineage_tree"] = self.lineage_encoder.lineage_tree

        # Cell type to lineage mappings
        adata.uns["celltype_to_lineage"] = self.worm_atlas._mappings

        # Modality statistics
        adata.uns["modality_stats"] = {
            "n_with_transcriptome": int(adata.n_obs),
            "n_with_spatial": int(adata.obs["has_spatial"].sum()),
            "n_with_lineage": int(adata.obs["lineage_valid"].sum()),
            "n_complete_trimodal": int(
                (adata.obs["has_spatial"] & adata.obs["lineage_valid"]).sum()
            ),
        }

        # Founder distribution
        founder_counts = adata.obs["lineage_founder"].value_counts().to_dict()
        adata.uns["founder_distribution"] = founder_counts

        return adata

    def build_spatial_graph(
        self,
        adata: ad.AnnData,
        n_neighbors: int = 10,
        radius: Optional[float] = None,
    ) -> ad.AnnData:
        """
        Build spatial neighborhood graph and add to AnnData.

        Args:
            adata: AnnData object with X_spatial in obsm.
            n_neighbors: Number of neighbors for KNN graph.
            radius: Radius for radius-based graph (overrides n_neighbors).

        Returns:
            AnnData with spatial graph in obsp.
        """
        from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph

        spatial_coords = adata.obsm["X_spatial"]
        valid_mask = ~np.isnan(spatial_coords).any(axis=1)

        if valid_mask.sum() < 2:
            logger.warning("Not enough valid spatial coordinates for graph")
            return adata

        # Build graph only for cells with valid coordinates
        valid_coords = spatial_coords[valid_mask]

        if radius is not None:
            # Radius-based graph
            graph = radius_neighbors_graph(
                valid_coords, radius=radius, mode="distance", include_self=False
            )
        else:
            # KNN graph
            nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(valid_coords) - 1))
            nn.fit(valid_coords)
            graph = nn.kneighbors_graph(mode="distance")

        # Expand to full size using efficient sparse matrix indexing
        n_cells = len(adata)
        valid_indices = np.where(valid_mask)[0]

        # Convert small graph to COO format for efficient remapping
        graph_coo = graph.tocoo()

        # Remap indices to full matrix indices
        new_row = valid_indices[graph_coo.row]
        new_col = valid_indices[graph_coo.col]

        # Build full-size sparse matrix directly from COO data
        full_graph = csr_matrix(
            (graph_coo.data, (new_row, new_col)),
            shape=(n_cells, n_cells),
            dtype=np.float32
        )

        adata.obsp["spatial_distances"] = full_graph
        adata.obsp["spatial_connectivities"] = (full_graph > 0).astype(np.float32)

        logger.info(f"Built spatial graph with {n_neighbors} neighbors")
        return adata

    def build_lineage_graph(
        self,
        adata: ad.AnnData,
    ) -> ad.AnnData:
        """
        Build lineage adjacency graph and add to AnnData.

        Args:
            adata: AnnData object with lineage information.

        Returns:
            AnnData with lineage graph in obsp.
        """
        # Get lineage names for cells with valid lineage
        lineage_col = None
        for col in ["lineage_complete", "lineage"]:
            if col in adata.obs.columns:
                lineage_col = col
                break

        if lineage_col is None:
            logger.warning("No lineage column found for graph building")
            return adata

        valid_mask = adata.obs["lineage_valid"].values
        lineages = adata.obs[lineage_col].values[valid_mask].tolist()

        if len(lineages) < 2:
            logger.warning("Not enough valid lineages for graph")
            return adata

        # Build adjacency for valid cells
        adj, name_to_idx = self.lineage_encoder.build_adjacency_matrix(
            lineages, include_parent=True
        )

        # Expand to full size using efficient sparse matrix indexing
        n_cells = len(adata)
        valid_indices = np.where(valid_mask)[0]

        # Convert adjacency to sparse COO format
        adj_sparse = csr_matrix(adj)
        adj_coo = adj_sparse.tocoo()

        # Remap indices to full matrix indices
        new_row = valid_indices[adj_coo.row]
        new_col = valid_indices[adj_coo.col]

        # Build full-size sparse matrix directly from COO data
        full_adj = csr_matrix(
            (adj_coo.data, (new_row, new_col)),
            shape=(n_cells, n_cells),
            dtype=np.float32
        )

        adata.obsp["lineage_adjacency"] = full_adj

        logger.info(f"Built lineage graph for {len(lineages)} cells")
        return adata

    def summary(self, adata: ad.AnnData) -> str:
        """
        Generate a summary of the AnnData object.

        Args:
            adata: AnnData object to summarize.

        Returns:
            Human-readable summary string.
        """
        lines = [
            "=" * 60,
            "NemaContext Trimodal AnnData Summary",
            "=" * 60,
            "",
            f"Cells (n_obs): {adata.n_obs:,}",
            f"Genes (n_vars): {adata.n_vars:,}",
            "",
            "Modality Coverage:",
            f"  Transcriptome: {adata.n_obs:,} (100%)",
            f"  Spatial: {adata.obs['has_spatial'].sum():,} ({100 * adata.obs['has_spatial'].mean():.1f}%)",
            f"  Lineage: {adata.obs['lineage_valid'].sum():,} ({100 * adata.obs['lineage_valid'].mean():.1f}%)",
            "",
            "Data Sources:",
        ]

        if "data_sources" in adata.uns:
            for key, val in adata.uns["data_sources"].items():
                lines.append(f"  {key}: {val}")

        lines.extend(
            [
                "",
                "Available embeddings (obsm):",
            ]
        )
        for key in adata.obsm.keys():
            shape = adata.obsm[key].shape
            lines.append(f"  {key}: {shape}")

        lines.extend(
            [
                "",
                "Available graphs (obsp):",
            ]
        )
        for key in adata.obsp.keys():
            nnz = adata.obsp[key].nnz
            lines.append(f"  {key}: {nnz:,} edges")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)
