"""
Expression matrix loader for Large2025 and Packer2019 datasets.

Handles loading of MTX format expression matrices along with
cell and gene annotations from GEO-formatted files.
"""

import gzip
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import io as scipy_io
from scipy.sparse import csr_matrix, spmatrix

logger = logging.getLogger(__name__)


class ExpressionLoader:
    """
    Loader for single-cell expression matrices from GEO datasets.

    Supports:
    - Large et al. 2025 (GSE292756): Recommended, lineage-resolved
    - Packer et al. 2019 (GSE126954): Legacy, for comparison

    The expression matrix is stored in MTX format (Matrix Market),
    with separate CSV files for cell and gene annotations.
    """

    def __init__(self, data_dir: str = "dataset/raw"):
        """
        Initialize the expression loader.

        Args:
            data_dir: Base directory containing raw data subdirectories.
        """
        self.data_dir = Path(data_dir)

    def load_large2025(
        self,
        species_filter: Optional[str] = "C.elegans",
        min_umi: int = 0,
    ) -> Tuple[csr_matrix, pd.DataFrame, pd.DataFrame]:
        """
        Load Large et al. 2025 expression data.

        Args:
            species_filter: Filter to specific species ('C.elegans', 'C.briggsae', or None for all).
            min_umi: Minimum UMI count per cell to include.

        Returns:
            Tuple of (expression_matrix, cell_annotations, gene_annotations)
            - expression_matrix: Sparse CSR matrix (cells x genes)
            - cell_annotations: DataFrame with cell metadata
            - gene_annotations: DataFrame with gene metadata
        """
        large2025_dir = self.data_dir / "large2025"

        # File paths
        mtx_path = large2025_dir / "GSE292756_expression_matrix.mtx.gz"
        cell_path = large2025_dir / "GSE292756_cell_annotations.csv.gz"
        gene_path = large2025_dir / "GSE292756_gene_annotations.csv.gz"

        # Validate files exist
        for path in [mtx_path, cell_path, gene_path]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Required file not found: {path}\n"
                    "Run: uv run python -m src.data.downloader --source large2025"
                )

        logger.info(f"Loading Large2025 data from {large2025_dir}")

        # Load gene annotations first (for matrix dimension validation)
        gene_df = self._load_gene_annotations(gene_path)
        logger.info(f"Loaded {len(gene_df)} gene annotations")

        # Load cell annotations
        cell_df = self._load_cell_annotations(cell_path)
        logger.info(f"Loaded {len(cell_df)} cell annotations")

        # Load expression matrix
        expr_matrix = self._load_mtx_matrix(mtx_path)
        logger.info(f"Loaded expression matrix: {expr_matrix.shape}")

        # Matrix is typically genes x cells, need to transpose to cells x genes
        if expr_matrix.shape[0] == len(gene_df) and expr_matrix.shape[1] == len(
            cell_df
        ):
            logger.info("Transposing matrix from (genes x cells) to (cells x genes)")
            expr_matrix = expr_matrix.T.tocsr()
        elif expr_matrix.shape[0] == len(cell_df) and expr_matrix.shape[1] == len(
            gene_df
        ):
            logger.info("Matrix already in (cells x genes) format")
            expr_matrix = expr_matrix.tocsr()
        else:
            raise ValueError(
                f"Matrix dimensions {expr_matrix.shape} don't match "
                f"annotations: {len(cell_df)} cells, {len(gene_df)} genes"
            )

        # Apply filters
        if species_filter is not None:
            mask = cell_df["species"] == species_filter
            cell_df = cell_df[mask].reset_index(drop=True)
            expr_matrix = expr_matrix[mask.values]
            logger.info(f"Filtered to {species_filter}: {len(cell_df)} cells")

        if min_umi > 0:
            umi_counts = np.array(expr_matrix.sum(axis=1)).flatten()
            mask = umi_counts >= min_umi
            cell_df = cell_df[mask].reset_index(drop=True)
            expr_matrix = expr_matrix[mask]
            logger.info(f"Filtered by min_umi >= {min_umi}: {len(cell_df)} cells")

        return expr_matrix, cell_df, gene_df

    def load_packer2019(
        self,
        min_umi: int = 0,
    ) -> Tuple[csr_matrix, pd.DataFrame, pd.DataFrame]:
        """
        Load Packer et al. 2019 expression data (legacy).

        Args:
            min_umi: Minimum UMI count per cell to include.

        Returns:
            Tuple of (expression_matrix, cell_annotations, gene_annotations)
        """
        packer_dir = self.data_dir / "packer2019"

        # File paths
        matrix_path = packer_dir / "GSE126954_gene_by_cell_count_matrix.txt.gz"
        cell_path = packer_dir / "GSE126954_cell_annotation.csv.gz"
        gene_path = packer_dir / "GSE126954_gene_annotation.csv.gz"

        # Validate files exist
        for path in [matrix_path, cell_path, gene_path]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Required file not found: {path}\n"
                    "Run: uv run python -m src.data.downloader --source packer"
                )

        logger.info(f"Loading Packer2019 data from {packer_dir}")

        # Load annotations
        gene_df = self._load_gene_annotations(gene_path)
        cell_df = self._load_cell_annotations(cell_path)
        logger.info(f"Loaded {len(cell_df)} cells, {len(gene_df)} genes")

        # Packer2019 uses a dense text matrix, not MTX
        expr_matrix = self._load_dense_matrix(matrix_path)
        logger.info(f"Loaded expression matrix: {expr_matrix.shape}")

        # Apply filters
        if min_umi > 0:
            umi_counts = np.array(expr_matrix.sum(axis=1)).flatten()
            mask = umi_counts >= min_umi
            cell_df = cell_df[mask].reset_index(drop=True)
            expr_matrix = expr_matrix[mask]
            logger.info(f"Filtered by min_umi >= {min_umi}: {len(cell_df)} cells")

        return expr_matrix, cell_df, gene_df

    def _load_mtx_matrix(self, path: Path) -> spmatrix:
        """Load a Matrix Market format file (optionally gzipped)."""
        logger.debug(f"Loading MTX file: {path}")

        if path.suffix == ".gz":
            with gzip.open(path, "rb") as f:
                matrix = scipy_io.mmread(f)
        else:
            matrix = scipy_io.mmread(path)

        return csr_matrix(matrix)

    def _load_dense_matrix(self, path: Path) -> csr_matrix:
        """Load a dense text matrix and convert to sparse."""
        logger.debug(f"Loading dense matrix: {path}")

        if path.suffix == ".gz":
            with gzip.open(path, "rt") as f:
                df = pd.read_csv(f, sep="\t", index_col=0)
        else:
            df = pd.read_csv(path, sep="\t", index_col=0)

        # Transpose if genes are rows (typical for GEO format)
        # Result should be cells x genes
        matrix = csr_matrix(df.T.values, dtype=np.float32)
        return matrix

    def _load_cell_annotations(self, path: Path) -> pd.DataFrame:
        """Load cell annotations from CSV."""
        logger.debug(f"Loading cell annotations: {path}")

        if path.suffix == ".gz":
            with gzip.open(path, "rt") as f:
                df = pd.read_csv(f)
        else:
            df = pd.read_csv(path)

        # Standardize column names (remove dots, lowercase)
        df.columns = [c.replace(".", "_") for c in df.columns]

        return df

    def _load_gene_annotations(self, path: Path) -> pd.DataFrame:
        """Load gene annotations from CSV."""
        logger.debug(f"Loading gene annotations: {path}")

        if path.suffix == ".gz":
            with gzip.open(path, "rt") as f:
                df = pd.read_csv(f)
        else:
            df = pd.read_csv(path)

        return df

    def validate_alignment(
        self,
        expr_matrix: spmatrix,
        cell_df: pd.DataFrame,
        gene_df: pd.DataFrame,
    ) -> bool:
        """
        Validate that matrix dimensions match annotations.

        Args:
            expr_matrix: Expression matrix (cells x genes)
            cell_df: Cell annotations
            gene_df: Gene annotations

        Returns:
            True if dimensions match, raises ValueError otherwise.
        """
        n_cells, n_genes = expr_matrix.shape

        if n_cells != len(cell_df):
            raise ValueError(
                f"Matrix has {n_cells} rows but cell_df has {len(cell_df)} rows"
            )

        if n_genes != len(gene_df):
            raise ValueError(
                f"Matrix has {n_genes} columns but gene_df has {len(gene_df)} rows"
            )

        logger.info(f"✓ Alignment validated: {n_cells} cells x {n_genes} genes")
        return True
