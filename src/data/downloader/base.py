"""
Base downloader class with common functionality.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import requests
from tqdm import tqdm

from .constants import (
    DEFAULT_DATA_DIR,
    DEFAULT_MIN_FILE_SIZE,
    DEFAULT_TIMEOUT,
    DOWNLOAD_CHUNK_SIZE,
)


class BaseDownloader(ABC):
    """Base class for all data downloaders."""

    def __init__(self, data_dir: str = DEFAULT_DATA_DIR):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _download_file(
        self,
        url: str,
        filename: str,
        subdir: str | None = None,
        timeout: int = DEFAULT_TIMEOUT,
        min_size: int = DEFAULT_MIN_FILE_SIZE,
    ) -> bool:
        """
        Download a file with progress bar.

        Args:
            url: URL to download from
            filename: Name to save the file as
            subdir: Optional subdirectory within data_dir
            timeout: Request timeout in seconds
            min_size: Minimum file size to consider valid (bytes)

        Returns:
            True if download successful, False otherwise
        """
        if subdir:
            save_dir = self.data_dir / subdir
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = self.data_dir

        save_path = save_dir / filename

        # Skip if file already exists and is valid
        if save_path.exists() and save_path.stat().st_size > min_size:
            print(f"✅ {filename} already exists. Skipping...")
            return True

        print(f"🚀 Downloading {filename}...")

        try:
            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            with (
                open(save_path, "wb") as f,
                tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc=filename,
                ) as pbar,
            ):
                for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            print(f"✅ Successfully downloaded {filename}")
            return True

        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to download {filename}: {e}")
            if save_path.exists():
                save_path.unlink()  # Clean up partial downloads
            return False

    @staticmethod
    def _print_header(message: str) -> None:
        """Print a formatted section header."""
        print("\n" + "=" * 60)
        print(message)
        print("=" * 60)

    @abstractmethod
    def download(self) -> None:
        """Download all files for this data source."""
        pass
