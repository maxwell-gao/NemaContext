"""
NemaContext Data Downloader Package.

Comprehensive downloader for C. elegans developmental data:
1. Packer et al. 2019 (GSE126954) - Single-cell transcriptomics
2. OpenWorm/c302 - Connectome and neuron tables
3. WormGUIDES - 4D embryo spatial coordinates
4. WormBase - Lineage tree data
"""

from .base import BaseDownloader
from .constants import DEFAULT_DATA_DIR
from .main import NemaContextDownloader, main
from .openworm import OpenWormDownloader
from .packer import PackerDownloader
from .wormbase import WormBaseDownloader
from .wormguides import WormGUIDESDownloader

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
