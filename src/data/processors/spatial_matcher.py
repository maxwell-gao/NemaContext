"""
Spatial matcher for WormGUIDES data.

This module re-exports the SpatialMatcher from the builder module
for backwards compatibility and convenience.

For the full implementation, see: src/data/builder/spatial_matcher.py
"""

from src.data.builder.spatial_matcher import SpatialMatcher

__all__ = ["SpatialMatcher"]
