"""
AnnData builder module for NemaContext.

This module provides tools to construct trimodal AnnData objects
integrating transcriptome, spatial, and lineage information.

Also includes CShaper integration for morphological data.
"""

from .anndata_builder import TrimodalAnnDataBuilder
from .cshaper_processor import (
    CShaperProcessor,
    ContactLoader,
    MorphologyLoader,
    StandardSpatialLoader,
    normalize_lineage_name,
    embryo_time_to_cshaper_frame,
)
from .enhanced_builder import EnhancedAnnDataBuilder
from .expression_loader import ExpressionLoader
from .lineage_encoder import LineageEncoder
from .spatial_matcher import SpatialMatcher
from .worm_atlas import WormAtlasMapper

__all__ = [
    # Core builders
    "TrimodalAnnDataBuilder",
    "EnhancedAnnDataBuilder",
    # Loaders
    "ExpressionLoader",
    "SpatialMatcher",
    "LineageEncoder",
    "WormAtlasMapper",
    # CShaper
    "CShaperProcessor",
    "ContactLoader",
    "MorphologyLoader",
    "StandardSpatialLoader",
    # Utilities
    "normalize_lineage_name",
    "embryo_time_to_cshaper_frame",
]
