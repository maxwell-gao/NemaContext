"""
WormGUIDES downloader.

Downloads 4D embryo and cell data from WormGUIDES GitHub repository.
Includes cell anatomy, connectome, cell deaths, parts list, and nuclei 4D positions.
"""

import os
from typing import Optional

from .base import BaseDownloader
from .constants import (
    MESSAGES,
    WORMGUIDES_BASE_URL,
    WORMGUIDES_FILES,
    WORMGUIDES_NUCLEI_BASE_URL,
    WORMGUIDES_SUBDIR,
    WORMGUIDES_TOTAL_TIMEPOINTS,
)


class WormGUIDESDownloader(BaseDownloader):
    """
    Downloader for WormGUIDES data from GitHub.
    Includes cell anatomy, connectome, cell deaths, parts list, and nuclei 4D positions.
    """

    def download(
        self,
        include_nuclei: bool = True,
        nuclei_step: int = 1,
        nuclei_start: Optional[int] = None,
        nuclei_end: Optional[int] = None,
    ) -> None:
        """
        Download all WormGUIDES data files.

        Args:
            include_nuclei: Whether to download nuclei 4D position files.
            nuclei_step: Step size for nuclei timepoints (1 = all, 10 = every 10th).
            nuclei_start: Starting timepoint for nuclei download (1-based, default 1).
            nuclei_end: Ending timepoint for nuclei download (inclusive, default 360).
        """
        self._print_header(MESSAGES["wormguides_header"])

        # Download standard files (anatomy, connectome, etc.)
        for key, info in WORMGUIDES_FILES.items():
            url = f"{WORMGUIDES_BASE_URL}/{info['url_path']}"
            self._download_file(
                url=url,
                filename=info["filename"],
                subdir=WORMGUIDES_SUBDIR,
            )

        # Download nuclei 4D position files
        if include_nuclei:
            self._download_nuclei_files(
                step=nuclei_step,
                start=nuclei_start,
                end=nuclei_end,
            )

    def _download_nuclei_files(
        self,
        step: int = 1,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> None:
        """
        Download nuclei position files for 4D spatial coordinates.

        Each file contains cell positions at a specific timepoint:
        - ID, flag, x, y, z, diameter, cell_name

        Args:
            step: Step size for timepoints (1 = all, 10 = every 10th).
            start: Starting timepoint (1-based, default 1).
            end: Ending timepoint (inclusive, default 360).
        """
        nuclei_subdir = os.path.join(WORMGUIDES_SUBDIR, "nuclei_files")

        start_tp = start if start is not None else 1
        end_tp = end if end is not None else WORMGUIDES_TOTAL_TIMEPOINTS

        # Validate range
        start_tp = max(1, min(start_tp, WORMGUIDES_TOTAL_TIMEPOINTS))
        end_tp = max(start_tp, min(end_tp, WORMGUIDES_TOTAL_TIMEPOINTS))
        step = max(1, step)

        timepoints = list(range(start_tp, end_tp + 1, step))
        total = len(timepoints)

        print("\n📍 Downloading nuclei 4D position files...")
        print(
            f"   Timepoints: {start_tp} to {end_tp} (step={step}, total={total} files)"
        )

        downloaded = 0
        skipped = 0

        for i, tp in enumerate(timepoints, 1):
            filename = f"t{tp:03d}-nuclei"
            url = f"{WORMGUIDES_NUCLEI_BASE_URL}/{filename}"

            # Show progress every 10 files or at start/end
            if i == 1 or i == total or i % 10 == 0:
                print(f"   [{i}/{total}] Downloading {filename}...")

            success = self._download_file(
                url=url,
                filename=filename,
                subdir=nuclei_subdir,
                verbose=False,  # Quiet mode for bulk download
            )

            if success:
                downloaded += 1
            else:
                skipped += 1

        print(f"   ✓ Nuclei files: {downloaded} downloaded, {skipped} skipped/failed")

    def download_nuclei_only(
        self,
        step: int = 1,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> None:
        """
        Download only nuclei 4D position files (skip other WormGUIDES files).

        Args:
            step: Step size for timepoints (1 = all, 10 = every 10th).
            start: Starting timepoint (1-based, default 1).
            end: Ending timepoint (inclusive, default 360).
        """
        self._print_header("📍 WormGUIDES - Nuclei 4D Position Data")
        self._download_nuclei_files(step=step, start=start, end=end)

    def get_nuclei_file_path(self, timepoint: int) -> str:
        """
        Get the local path for a nuclei file at a given timepoint.

        Args:
            timepoint: Timepoint number (1-360).

        Returns:
            Full path to the nuclei file.
        """
        filename = f"t{timepoint:03d}-nuclei"
        return os.path.join(
            self.data_dir,
            WORMGUIDES_SUBDIR,
            "nuclei_files",
            filename,
        )
