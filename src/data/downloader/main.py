"""
Main downloader orchestrator for NemaContext.

Coordinates all data source downloaders and provides CLI interface.
"""

import argparse
from pathlib import Path

from .base import BaseDownloader
from .constants import DEFAULT_DATA_DIR, MESSAGES
from .large2025 import Large2025Downloader
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
        # Order: recommended first, then spatial, then legacy
        self.downloaders: list[BaseDownloader] = [
            Large2025Downloader(data_dir),  # Recommended for transcriptome
            WormGUIDESDownloader(data_dir),  # Spatial coordinates
            WormBaseDownloader(data_dir),  # Lineage tree
            OpenWormDownloader(data_dir),  # Connectome (optional)
            # PackerDownloader not included by default (superseded by Large2025)
        ]

    def download_all(self) -> None:
        """Download data from all sources."""
        print("\n" + "=" * 60)
        print(MESSAGES["main_header"])
        print("=" * 60)
        print(f"📁 Data directory: {self.data_dir}")
        print()
        print("📋 Download order:")
        print("   1. Large et al. 2025 (transcriptome + lineage annotations)")
        print("   2. WormGUIDES (4D spatial coordinates)")
        print("   3. WormBase (lineage tree data)")
        print("   4. OpenWorm (connectome)")
        print()
        print("💡 Note: Packer 2019 is superseded by Large 2025.")
        print("   Use --source packer if you specifically need the legacy data.")

        for downloader in self.downloaders:
            try:
                downloader.download()
            except Exception as e:
                print(f"⚠️ Error in {downloader.__class__.__name__}: {e}")
                continue

        self._print_summary()

    def download_large2025(self) -> None:
        """Download Large et al. 2025 lineage-resolved embryo atlas (recommended)."""
        Large2025Downloader(self.data_dir).download()

    def download_packer(self) -> None:
        """Download Packer et al. 2019 data (legacy, superseded by Large 2025)."""
        print("⚠️  Note: Packer 2019 is superseded by Large et al. 2025.")
        print("   Consider using --source large2025 for improved lineage annotations.")
        print()
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

    def download_core(self) -> None:
        """
        Download core datasets for transcriptome-spatial-lineage integration.

        This downloads:
        - Large 2025 (transcriptome + lineage annotations)
        - WormGUIDES (4D spatial coordinates)
        - WormBase (lineage tree)

        Excludes OpenWorm connectome which is optional for embryo analysis.
        """
        print("\n" + "=" * 60)
        print("🧬 NemaContext Core Data Download")
        print("=" * 60)
        print("📋 Downloading essential datasets for multimodal integration:")
        print("   - Large 2025: Transcriptome + lineage-resolved annotations")
        print("   - WormGUIDES: 4D spatial coordinates")
        print("   - WormBase: Lineage tree structure")
        print()

        Large2025Downloader(self.data_dir).download()
        WormGUIDESDownloader(self.data_dir).download()
        WormBaseDownloader(self.data_dir).download()

        self._print_summary()

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
        description="Download C. elegans developmental data for NemaContext",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all datasets (recommended)
  python -m src.data.downloader --source all

  # Download only essential datasets for multimodal integration
  python -m src.data.downloader --source core

  # Download only the new Large 2025 transcriptome data
  python -m src.data.downloader --source large2025

  # Download legacy Packer 2019 data
  python -m src.data.downloader --source packer

Dataset descriptions:
  large2025   : Large et al. 2025 (GSE292756) - Lineage-resolved embryo atlas
                >375,000 cells, direct cell-to-lineage mapping (RECOMMENDED)
  packer      : Packer et al. 2019 (GSE126954) - Legacy transcriptomics
                ~86,000 cells (superseded by large2025)
  wormguides  : WormGUIDES - 4D embryo spatial coordinates (360 timepoints)
  wormbase    : WormBase-derived lineage tree data
  openworm    : OpenWorm/c302 - Adult connectome data
  core        : Essential datasets only (large2025 + wormguides + wormbase)
  all         : All datasets including optional connectome
        """,
    )
    parser.add_argument(
        "--source",
        choices=[
            "all",
            "core",
            "large2025",
            "packer",
            "openworm",
            "wormbase",
            "wormguides",
        ],
        default="core",
        help="Data source to download (default: core)",
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
        case "core":
            downloader.download_core()
        case "large2025":
            downloader.download_large2025()
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
