"""
WormGUIDES downloader.

Downloads 4D embryo and cell data from WormGUIDES GitHub repository.
Includes cell anatomy, connectome, cell deaths, and parts list.
"""

from .base import BaseDownloader
from .constants import (
    MESSAGES,
    WORMGUIDES_BASE_URL,
    WORMGUIDES_FILES,
    WORMGUIDES_SUBDIR,
)


class WormGUIDESDownloader(BaseDownloader):
    """
    Downloader for WormGUIDES data from GitHub.
    Includes cell anatomy, connectome, cell deaths, and parts list.
    """

    def download(self) -> None:
        """Download all WormGUIDES data files."""
        self._print_header(MESSAGES["wormguides_header"])

        for key, info in WORMGUIDES_FILES.items():
            url = f"{WORMGUIDES_BASE_URL}/{info['url_path']}"
            self._download_file(
                url=url,
                filename=info["filename"],
                subdir=WORMGUIDES_SUBDIR,
            )
