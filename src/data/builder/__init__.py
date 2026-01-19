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
    Segmentation3DLoader,
    AncestorMapper,
    normalize_lineage_name,
    embryo_time_to_cshaper_frame,
    cshaper_frame_to_embryo_time,
    get_lineage_ancestors,
    get_ancestor_distance,
    expand_uncertain_lineage,
    expand_slash_lineage,
    CSHAPER_FRAMES,
    CSHAPER_START_TIME_MIN,
    CSHAPER_END_TIME_MIN,
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
    "Segmentation3DLoader",
    "AncestorMapper",
    # Utilities
    "normalize_lineage_name",
    "embryo_time_to_cshaper_frame",
    "cshaper_frame_to_embryo_time",
    "get_lineage_ancestors",
    "get_ancestor_distance",
    "expand_uncertain_lineage",
    "expand_slash_lineage",
    # Constants
    "CSHAPER_FRAMES",
    "CSHAPER_START_TIME_MIN",
    "CSHAPER_END_TIME_MIN",
]
