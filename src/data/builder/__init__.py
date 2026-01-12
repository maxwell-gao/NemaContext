"""
AnnData builder module for NemaContext.

This module provides tools to construct trimodal AnnData objects
integrating transcriptome, spatial, and lineage information.
"""

from .anndata_builder import TrimodalAnnDataBuilder
from .expression_loader import ExpressionLoader
from .lineage_encoder import LineageEncoder
from .spatial_matcher import SpatialMatcher
from .worm_atlas import WormAtlasMapper

__all__ = [
    "TrimodalAnnDataBuilder",
    "ExpressionLoader",
    "SpatialMatcher",
    "LineageEncoder",
    "WormAtlasMapper",
]
