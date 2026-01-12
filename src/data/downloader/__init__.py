"""
NemaContext Data Downloader Package.

Comprehensive downloader for C. elegans developmental data:
1. Large et al. 2025 (GSE292756) - Lineage-resolved embryo atlas (RECOMMENDED)
2. Packer et al. 2019 (GSE126954) - Single-cell transcriptomics (Legacy)
3. OpenWorm/c302 - Connectome and neuron tables
4. WormGUIDES - 4D embryo spatial coordinates
5. WormBase - Lineage tree data
"""

from .base import BaseDownloader
from .constants import DEFAULT_DATA_DIR
from .large2025 import Large2025Downloader
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
    "Large2025Downloader",  # Recommended for transcriptome data
    "PackerDownloader",  # Legacy, superseded by Large2025
    "OpenWormDownloader",
    "WormBaseDownloader",
    "WormGUIDESDownloader",
    # Constants
    "DEFAULT_DATA_DIR",
]
