"""
Packer et al. 2019 (GSE126954) downloader.

Single-cell transcriptomics of C. elegans embryogenesis.
~86,024 cells from 100-650 min post-cleavage.
"""

from .base import BaseDownloader
from .constants import (
    MESSAGES,
    PACKER_FILES,
    PACKER_SUBDIR,
    PACKER_TIMEOUT,
)


class PackerDownloader(BaseDownloader):
    """
    Downloader for Packer et al. 2019 (GSE126954).
    Single-cell transcriptomics of C. elegans embryogenesis.
    """

    def download(self) -> None:
        """Download all Packer et al. 2019 files."""
        self._print_header(MESSAGES["packer_header"])

        for key, info in PACKER_FILES.items():
            self._download_file(
                url=info["url"],
                filename=info["filename"],
                subdir=PACKER_SUBDIR,
                timeout=PACKER_TIMEOUT,
            )
