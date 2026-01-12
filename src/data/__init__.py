"""
NemaContext Data Module

Provides data downloading and processing utilities for C. elegans developmental data.
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
]
