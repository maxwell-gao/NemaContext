"""
OpenWorm/c302 downloader.

Includes connectome data, neuron tables, and cell information.
"""

from .base import BaseDownloader
from .constants import (
    MESSAGES,
    OPENWORM_BASE_URL,
    OPENWORM_FILES,
    OPENWORM_SUBDIR,
)


class OpenWormDownloader(BaseDownloader):
    """
    Downloader for OpenWorm/c302 data.
    Includes connectome data, neuron tables, and cell information.
    """

    def download(self) -> None:
        """Download all OpenWorm/c302 files."""
        self._print_header(MESSAGES["openworm_header"])

        for key, info in OPENWORM_FILES.items():
            url = f"{OPENWORM_BASE_URL}/{info['filename']}"
            self._download_file(
                url=url,
                filename=info["filename"],
                subdir=OPENWORM_SUBDIR,
            )
