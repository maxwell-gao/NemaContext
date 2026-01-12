"""
Main downloader orchestrator for NemaContext.

Coordinates all data source downloaders and provides CLI interface.
"""

import argparse
from pathlib import Path

from .base import BaseDownloader
from .constants import DEFAULT_DATA_DIR, MESSAGES
from .openworm import OpenWormDownloader
from .packer import PackerDownloader
from .wormbase import WormBaseDownloader
from .wormguides import WormGUIDESDownloader


class NemaContextDownloader:
    """
    Main downloader class that orchestrates all data sources.
    """

    def __init__(self, data_dir: str = DEFAULT_DATA_DIR):
        self.data_dir = data_dir
        self.downloaders: list[BaseDownloader] = [
            PackerDownloader(data_dir),
            OpenWormDownloader(data_dir),
            WormBaseDownloader(data_dir),
            WormGUIDESDownloader(data_dir),
        ]

    def download_all(self) -> None:
        """Download data from all sources."""
        print("\n" + "=" * 60)
        print(MESSAGES["main_header"])
        print("=" * 60)
        print(f"📁 Data directory: {self.data_dir}")

        for downloader in self.downloaders:
            try:
                downloader.download()
            except Exception as e:
                print(f"⚠️ Error in {downloader.__class__.__name__}: {e}")
                continue

        self._print_summary()

    def download_packer(self) -> None:
        """Download only Packer et al. 2019 data."""
        PackerDownloader(self.data_dir).download()

    def download_openworm(self) -> None:
        """Download only OpenWorm/c302 data."""
        OpenWormDownloader(self.data_dir).download()

    def download_wormbase(self) -> None:
        """Download only WormBase lineage data."""
        WormBaseDownloader(self.data_dir).download()

    def download_wormguides(self) -> None:
        """Download WormGUIDES data."""
        WormGUIDESDownloader(self.data_dir).download()

    def _print_summary(self) -> None:
        """Print summary of downloaded data."""
        print("\n" + "=" * 60)
        print(MESSAGES["summary_header"])
        print("=" * 60)

        data_path = Path(self.data_dir)
        if data_path.exists():
            for subdir in sorted(data_path.iterdir()):
                if subdir.is_dir():
                    files = list(subdir.glob("*"))
                    total_size = sum(f.stat().st_size for f in files if f.is_file())
                    size_mb = total_size / (1024 * 1024)
                    print(f"  📂 {subdir.name}: {len(files)} files ({size_mb:.1f} MB)")

        print(f"\n{MESSAGES['download_complete']}")


def main() -> None:
    """Main entry point for the downloader."""
    parser = argparse.ArgumentParser(
        description="Download C. elegans developmental data for NemaContext"
    )
    parser.add_argument(
        "--source",
        choices=["all", "packer", "openworm", "wormbase", "wormguides"],
        default="all",
        help="Data source to download (default: all)",
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=f"Directory to store downloaded data (default: {DEFAULT_DATA_DIR})",
    )

    args = parser.parse_args()

    downloader = NemaContextDownloader(data_dir=args.data_dir)

    match args.source:
        case "all":
            downloader.download_all()
        case "packer":
            downloader.download_packer()
        case "openworm":
            downloader.download_openworm()
        case "wormbase":
            downloader.download_wormbase()
        case "wormguides":
            downloader.download_wormguides()


if __name__ == "__main__":
    main()
