"""
Data processors for NemaContext.

This module provides processors for:
- Expression matrix loading and normalization
- Spatial coordinate extraction from WormGUIDES
- Lineage parsing and encoding
- Cell type to lineage mapping via WormAtlas
"""

from .celltype_mapper import CellTypeMapper
from .expression_loader import ExpressionLoader
from .lineage_encoder import LineageEncoder
from .spatial_matcher import SpatialMatcher

__all__ = [
    "ExpressionLoader",
    "SpatialMatcher",
    "LineageEncoder",
    "CellTypeMapper",
]
