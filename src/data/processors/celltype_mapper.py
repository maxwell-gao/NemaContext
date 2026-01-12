"""
CellTypeMapper processor for NemaContext.

This module provides cell type mapping functionality,
delegating to the WormAtlasMapper in the builder module.
"""

from typing import List, Tuple

import pandas as pd

from ..builder.worm_atlas import WormAtlasMapper


class CellTypeMapper:
    """
    Maps cell types to lineages and vice versa.

    This is a convenience wrapper around WormAtlasMapper for use
    in data processing pipelines.

    Example:
        >>> mapper = CellTypeMapper()
        >>> mapper.get_lineage("ADAL")
        ['ABplapaaaapp']
    """

    def __init__(self, data_dir: str = "dataset/raw"):
        """
        Initialize the cell type mapper.

        Args:
            data_dir: Base directory containing data files.
        """
        self._mapper = WormAtlasMapper(data_dir=data_dir)

    def get_lineage(self, cell_type: str) -> List[str]:
        """
        Get lineage name(s) for a cell type.

        Args:
            cell_type: Cell type name (e.g., "ADAL", "BWM_head").

        Returns:
            List of matching lineage names.
        """
        return self._mapper.celltype_to_lineage(cell_type)

    def get_cell_type(self, lineage: str) -> List[str]:
        """
        Get cell type name(s) for a lineage.

        Args:
            lineage: Lineage name (e.g., "ABplapaaaapp").

        Returns:
            List of matching cell type names.
        """
        return self._mapper.lineage_to_celltype(lineage)

    def get_category(self, cell_type: str) -> str:
        """
        Get the category for a cell type.

        Args:
            cell_type: Cell type name.

        Returns:
            Category name (e.g., 'neuron', 'muscle') or 'unknown'.
        """
        return self._mapper.get_category(cell_type)

    def match_series(
        self,
        cell_types: pd.Series,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Match a series of cell types to lineages.

        Args:
            cell_types: Series of cell type names.

        Returns:
            Tuple of (matched_lineages, success_mask) Series.
        """
        return self._mapper.match_large2025_celltypes(cell_types)

    @property
    def all_cell_types(self) -> List[str]:
        """Get list of all known cell types."""
        return self._mapper.get_all_celltypes()

    @property
    def all_lineages(self) -> set:
        """Get set of all known lineages."""
        return self._mapper.get_all_lineages()
