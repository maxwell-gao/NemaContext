"""
CShaper morphological atlas downloader.

Downloads the CShaper segmentation results and standardized morphological atlas
from Cao et al. 2020 (Nature Communications).

Paper: "Establishment of a morphological atlas of the Caenorhabditis elegans
embryo using deep-learning-based 4D segmentation"
DOI: 10.1038/s41467-020-19863-x

Data source: https://doi.org/10.6084/m9.figshare.12839315
"""

from pathlib import Path

from .base import BaseDownloader
from .constants import DEFAULT_DATA_DIR

# =============================================================================
# CShaper Configuration
# =============================================================================

CSHAPER_SUBDIR = "cshaper"
CSHAPER_TIMEOUT = 600  # Large files, need longer timeout

# Figshare article information
CSHAPER_FIGSHARE_ARTICLE_ID = "12839315"
CSHAPER_DOI = "10.6084/m9.figshare.12839315"

# Figshare API endpoint for getting file info
CSHAPER_FIGSHARE_API_URL = (
    f"https://api.figshare.com/v2/articles/{CSHAPER_FIGSHARE_ARTICLE_ID}"
)

# Known file information from the dataset
# These are the supplementary data files from the CShaper paper
CSHAPER_FILES = {
    # Segmentation training/evaluation data
    "training_data": {
        "filename": "training_data.zip",
        "description": "Training data for DMapNet (membrane images + annotations)",
    },
    "evaluation_data": {
        "filename": "evaluation_data.zip",
        "description": "Evaluation data with manual annotations",
    },
    # Morphological atlas
    "morphological_atlas": {
        "filename": "morphological_atlas.zip",
        "description": "Standardized morphological atlas (17 embryos, 4-350 cell stages)",
    },
    # Segmentation results
    "segmentation_results": {
        "filename": "segmentation_results.zip",
        "description": "Full segmentation results for 17 embryos",
    },
    # Cell information
    "cell_morphology": {
        "filename": "cell_morphology.csv",
        "description": "Cell morphology data (volume, surface area, irregularity)",
    },
    "cell_contacts": {
        "filename": "cell_contacts.csv",
        "description": "Cell-cell contact data (479 effective contacts)",
    },
    # Supplementary data files (from paper)
    "supplementary_data_1": {
        "filename": "Supplementary_Data_1.xlsx",
        "description": "Sample information for 49 embryos",
    },
    "supplementary_data_2": {
        "filename": "Supplementary_Data_2.xlsx",
        "description": "54 developmental stages (landmarks)",
    },
    "supplementary_data_3": {
        "filename": "Supplementary_Data_3.xlsx",
        "description": "656 cells with segmentation info",
    },
    "supplementary_data_4": {
        "filename": "Supplementary_Data_4.xlsx",
        "description": "322 cells with complete lifespan",
    },
    "supplementary_data_5": {
        "filename": "Supplementary_Data_5.xlsx",
        "description": "479 effective cell-cell contacts",
    },
    "supplementary_data_6": {
        "filename": "Supplementary_Data_6.xlsx",
        "description": "Known signaling pairs comparison",
    },
    "supplementary_data_7": {
        "filename": "Supplementary_Data_7.xlsx",
        "description": "Cell irregularity scores",
    },
    "supplementary_data_8": {
        "filename": "Supplementary_Data_8.xlsx",
        "description": "C. elegans strain genotypes",
    },
}

# Key statistics from the paper for validation
CSHAPER_STATS = {
    "n_embryos_total": 49,  # Total embryos used
    "n_embryos_membrane": 17,  # Embryos with membrane segmentation
    "n_embryos_nuclei_only": 29,  # Embryos with only nuclei tracking
    "n_training_embryos": 4,  # Embryos used for training/evaluation
    "cell_stage_range": (4, 350),  # 4-cell to 350-cell stages
    "n_unique_cells": 656,  # Unique cells with segmentation
    "n_complete_lifespan_cells": 322,  # Cells with complete lifespan
    "n_effective_contacts": 479,  # Effective cell-cell contacts
    "time_resolution_min": 1.5,  # ~1.5 min imaging interval
    "wormguides_time_range": (20, 380),  # Approximate time coverage in minutes
}


class CShaperDownloader(BaseDownloader):
    """
    Downloader for CShaper morphological atlas data.

    Downloads the 4D cell shape, volume, surface area, and cell-cell contact
    data from the CShaper paper (Cao et al. 2020).
    """

    def __init__(self, data_dir: str = DEFAULT_DATA_DIR):
        super().__init__(data_dir)
        self.subdir = CSHAPER_SUBDIR
        self._file_urls: dict[str, str] = {}

    def _fetch_figshare_files(self, verbose: bool = True) -> dict[str, dict]:
        """
        Fetch file information from Figshare API.

        Returns:
            Dictionary mapping filename to file info (url, size, etc.)
        """
        import requests

        if verbose:
            print("📡 Fetching file list from Figshare...")

        try:
            response = requests.get(CSHAPER_FIGSHARE_API_URL, timeout=30)
            response.raise_for_status()
            article_data = response.json()

            files = article_data.get("files", [])
            file_map = {}

            for f in files:
                name = f.get("name", "")
                file_map[name] = {
                    "url": f.get("download_url", ""),
                    "size": f.get("size", 0),
                    "computed_md5": f.get("computed_md5", ""),
                }

            if verbose:
                print(f"✅ Found {len(file_map)} files on Figshare")
                for name, info in file_map.items():
                    size_mb = info["size"] / (1024 * 1024)
                    print(f"   - {name} ({size_mb:.2f} MB)")

            return file_map

        except requests.exceptions.RequestException as e:
            if verbose:
                print(f"❌ Failed to fetch Figshare file list: {e}")
            return {}

    def download(
        self,
        files: list[str] | None = None,
        verbose: bool = True,
    ) -> dict[str, bool]:
        """
        Download CShaper data files.

        Args:
            files: List of file keys to download (from CSHAPER_FILES).
                   If None, downloads all available files.
            verbose: Whether to print progress messages.

        Returns:
            Dictionary mapping filenames to success status.
        """
        if verbose:
            self._print_header("🔬 CShaper Morphological Atlas (Cao et al. 2020)")
            print(f"DOI: {CSHAPER_DOI}")
            print(f"Paper: Nature Communications 11, 6254 (2020)")

        # Fetch actual file URLs from Figshare
        figshare_files = self._fetch_figshare_files(verbose=verbose)

        if not figshare_files:
            if verbose:
                print("⚠️ Could not fetch file list from Figshare.")
                print("   Attempting direct download of known files...")

        results = {}

        # Determine which files to download
        if files is None:
            files_to_download = list(CSHAPER_FILES.keys())
        else:
            files_to_download = files

        for file_key in files_to_download:
            if file_key not in CSHAPER_FILES:
                if verbose:
                    print(f"⚠️ Unknown file key: {file_key}")
                continue

            file_info = CSHAPER_FILES[file_key]
            filename = file_info["filename"]

            # Check if file exists in Figshare response
            if filename in figshare_files:
                url = figshare_files[filename]["url"]
                success = self._download_file(
                    url=url,
                    filename=filename,
                    subdir=self.subdir,
                    timeout=CSHAPER_TIMEOUT,
                    verbose=verbose,
                )
                results[filename] = success
            else:
                if verbose:
                    print(f"⚠️ File not found on Figshare: {filename}")
                results[filename] = False

        return results

    def download_supplementary_data(self, verbose: bool = True) -> dict[str, bool]:
        """
        Download only the supplementary data files (Excel spreadsheets).

        These contain:
        - Sample information
        - Developmental stages
        - Cell morphology data
        - Cell-cell contact data
        """
        supp_keys = [
            k for k in CSHAPER_FILES.keys() if k.startswith("supplementary_data")
        ]
        return self.download(files=supp_keys, verbose=verbose)

    def download_morphological_atlas(self, verbose: bool = True) -> dict[str, bool]:
        """
        Download the morphological atlas data.

        This is the main dataset containing standardized cell shapes
        for 17 embryos from 4-cell to 350-cell stages.
        """
        atlas_keys = [
            "morphological_atlas",
            "segmentation_results",
            "cell_morphology",
            "cell_contacts",
        ]
        return self.download(files=atlas_keys, verbose=verbose)

    def get_data_path(self, filename: str | None = None) -> Path:
        """
        Get the path to CShaper data directory or specific file.

        Args:
            filename: Optional specific filename to get path for.

        Returns:
            Path to data directory or file.
        """
        base_path = self.data_dir / self.subdir
        if filename:
            return base_path / filename
        return base_path

    def list_downloaded_files(self) -> list[Path]:
        """List all downloaded CShaper files."""
        data_path = self.get_data_path()
        if not data_path.exists():
            return []
        return list(data_path.iterdir())

    @staticmethod
    def get_stats() -> dict:
        """Get key statistics about the CShaper dataset."""
        return CSHAPER_STATS.copy()

    @staticmethod
    def get_file_descriptions() -> dict[str, str]:
        """Get descriptions of available files."""
        return {k: v["description"] for k, v in CSHAPER_FILES.items()}
