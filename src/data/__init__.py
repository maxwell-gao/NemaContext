"""
NemaContext Data Module

Provides data downloading, processing, and AnnData building utilities
for C. elegans developmental data integration.

Submodules:
    - downloader: Data downloading from GEO, WormGUIDES, WormBase
    - builder: Trimodal AnnData construction
    - processors: Data processing utilities
"""

from .downloader import (
    DEFAULT_DATA_DIR,
    BaseDownloader,
    NemaContextDownloader,
    OpenWormDownloader,
    PackerDownloader,
    WormBaseDownloader,
    WormGUIDESDownloader,
    main,
)


# Builder exports (lazy import to avoid circular dependencies)
def __getattr__(name):
    """Lazy import for builder and processor classes."""
    if name == "TrimodalAnnDataBuilder":
        from .builder import TrimodalAnnDataBuilder

        return TrimodalAnnDataBuilder
    elif name == "ExpressionLoader":
        from .builder import ExpressionLoader

        return ExpressionLoader
    elif name == "SpatialMatcher":
        from .builder import SpatialMatcher

        return SpatialMatcher
    elif name == "LineageEncoder":
        from .builder import LineageEncoder

        return LineageEncoder
    elif name == "WormAtlasMapper":
        from .builder import WormAtlasMapper

        return WormAtlasMapper
    elif name == "CellTypeMapper":
        from .processors import CellTypeMapper

        return CellTypeMapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Main interface
    "NemaContextDownloader",
    "main",
    # Base class
    "BaseDownloader",
    # Individual downloaders
    "PackerDownloader",
    "OpenWormDownloader",
    "WormBaseDownloader",
    "WormGUIDESDownloader",
    # Constants
    "DEFAULT_DATA_DIR",
    # Builder classes (lazy loaded)
    "TrimodalAnnDataBuilder",
    "ExpressionLoader",
    "SpatialMatcher",
    "LineageEncoder",
    "WormAtlasMapper",
    # Processor classes (lazy loaded)
    "CellTypeMapper",
]
