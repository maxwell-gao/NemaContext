#!/usr/bin/env python3
"""
CShaper Morphological Atlas - Download and Evaluation Script

This script downloads and evaluates the CShaper dataset from Cao et al. 2020
(Nature Communications), which provides:
- 4D cell segmentation for 17 C. elegans embryos
- Morphological atlas from 4-cell to 350-cell stages
- Cell shape, volume, surface area data
- 479 effective cell-cell contacts

Paper: "Establishment of a morphological atlas of the Caenorhabditis elegans
embryo using deep-learning-based 4D segmentation"
DOI: 10.1038/s41467-020-19863-x
Data: https://doi.org/10.6084/m9.figshare.12839315
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import requests
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = project_root / "dataset" / "raw" / "cshaper"
FIGSHARE_ARTICLE_ID = "12839315"
FIGSHARE_API_URL = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE_ID}"

# Key statistics from the CShaper paper for validation
EXPECTED_STATS = {
    "n_embryos_total": 49,
    "n_embryos_membrane": 17,
    "n_embryos_nuclei_only": 29,
    "cell_stage_range": (4, 350),
    "n_unique_cells": 656,
    "n_complete_lifespan_cells": 322,
    "n_effective_contacts": 479,
    "time_resolution_min": 1.5,
}


# =============================================================================
# Helper Functions
# =============================================================================


def print_header(msg: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(msg)
    print("=" * 70)


def print_section(msg: str) -> None:
    """Print a formatted section header."""
    print(f"\n--- {msg} ---")


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


# =============================================================================
# Figshare API Functions
# =============================================================================


def fetch_article_info() -> dict | None:
    """Fetch article information from Figshare API."""
    print("📡 Fetching article information from Figshare API...")

    try:
        response = requests.get(FIGSHARE_API_URL, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to fetch article info: {e}")
        return None


def fetch_file_list() -> list[dict]:
    """Fetch list of files from Figshare article."""
    article = fetch_article_info()
    if not article:
        return []

    files = article.get("files", [])
    print(f"✅ Found {len(files)} files")
    return files


def download_file(url: str, save_path: Path, expected_size: int = 0) -> bool:
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=600)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", expected_size))

        save_path.parent.mkdir(parents=True, exist_ok=True)

        with (
            open(save_path, "wb") as f,
            tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=save_path.name,
            ) as pbar,
        ):
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        return True

    except Exception as e:
        print(f"❌ Download failed: {e}")
        if save_path.exists():
            save_path.unlink()
        return False


# =============================================================================
# Main Functions
# =============================================================================


def show_article_info() -> dict | None:
    """Display article information from Figshare."""
    print_header("📚 CShaper Dataset Information")

    article = fetch_article_info()
    if not article:
        return None

    print(f"\n📖 Title: {article.get('title', 'N/A')}")
    print(f"🔗 DOI: {article.get('doi', 'N/A')}")
    print(f"📅 Published: {article.get('published_date', 'N/A')}")
    print(f"📝 License: {article.get('license', {}).get('name', 'N/A')}")

    # Authors
    authors = article.get("authors", [])
    if authors:
        author_names = [a.get("full_name", "Unknown") for a in authors[:5]]
        if len(authors) > 5:
            author_names.append(f"... and {len(authors) - 5} more")
        print(f"👤 Authors: {', '.join(author_names)}")

    # Categories
    categories = article.get("categories", [])
    if categories:
        cat_names = [c.get("title", "") for c in categories[:3]]
        print(f"📂 Categories: {', '.join(cat_names)}")

    # Description excerpt
    description = article.get("description", "")
    if description:
        # Clean HTML tags
        import re

        clean_desc = re.sub(r"<[^>]+>", "", description)
        if len(clean_desc) > 200:
            clean_desc = clean_desc[:200] + "..."
        print(f"\n📄 Description: {clean_desc}")

    return article


def list_files() -> list[dict]:
    """List all available files in the dataset."""
    print_header("📁 Available Files")

    files = fetch_file_list()
    if not files:
        print("❌ No files found or API request failed")
        return []

    total_size = 0
    print(f"\n{'Filename':<50} {'Size':>12}")
    print("-" * 65)

    for f in sorted(files, key=lambda x: x.get("name", "")):
        name = f.get("name", "Unknown")
        size = f.get("size", 0)
        total_size += size
        print(f"{name:<50} {format_size(size):>12}")

    print("-" * 65)
    print(f"{'Total':<50} {format_size(total_size):>12}")

    return files


def download_all_files(force: bool = False) -> dict[str, bool]:
    """Download all files from the CShaper dataset."""
    print_header("⬇️  Downloading CShaper Dataset")

    files = fetch_file_list()
    if not files:
        return {}

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    total_size = sum(f.get("size", 0) for f in files)

    print(f"\n📊 Total files: {len(files)}")
    print(f"💾 Total size: {format_size(total_size)}")
    print(f"📂 Download directory: {DATA_DIR}")

    for f in files:
        name = f.get("name", "")
        url = f.get("download_url", "")
        size = f.get("size", 0)
        save_path = DATA_DIR / name

        # Skip if already downloaded (unless force)
        if save_path.exists() and save_path.stat().st_size > 0 and not force:
            print(f"✅ {name} already exists, skipping...")
            results[name] = True
            continue

        print(f"\n🚀 Downloading {name} ({format_size(size)})...")
        success = download_file(url, save_path, size)
        results[name] = success

    # Summary
    print_section("Download Summary")
    successful = sum(1 for v in results.values() if v)
    failed = len(results) - successful
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")

    return results


def evaluate_dataset() -> dict:
    """Evaluate the downloaded CShaper dataset."""
    print_header("📊 Evaluating CShaper Dataset")

    evaluation = {
        "files_found": [],
        "files_missing": [],
        "total_size_bytes": 0,
        "issues": [],
        "stats_validation": {},
    }

    # Check if data directory exists
    if not DATA_DIR.exists():
        print(f"❌ Data directory not found: {DATA_DIR}")
        evaluation["issues"].append("Data directory not found")
        return evaluation

    # List downloaded files
    files = list(DATA_DIR.iterdir())
    print(f"\n📁 Files in {DATA_DIR}:")

    for f in sorted(files):
        size = f.stat().st_size if f.is_file() else 0
        evaluation["total_size_bytes"] += size
        evaluation["files_found"].append(f.name)
        print(f"   {f.name} ({format_size(size)})")

    print(f"\n💾 Total size: {format_size(evaluation['total_size_bytes'])}")

    # Try to analyze specific files if they exist
    print_section("File Content Analysis")

    # Check for Excel/CSV files that we can analyze
    excel_files = [f for f in files if f.suffix in [".xlsx", ".xls"]]
    csv_files = [f for f in files if f.suffix == ".csv"]
    zip_files = [f for f in files if f.suffix == ".zip"]

    print(f"📊 Excel files: {len(excel_files)}")
    print(f"📋 CSV files: {len(csv_files)}")
    print(f"📦 ZIP archives: {len(zip_files)}")

    # Try to parse Excel files if pandas is available
    try:
        import pandas as pd

        for excel_file in excel_files[:3]:  # Analyze first 3
            print(f"\n📖 Analyzing {excel_file.name}...")
            try:
                # Try to read Excel file
                xl = pd.ExcelFile(excel_file)
                sheets = xl.sheet_names
                print(f"   Sheets: {sheets}")

                for sheet in sheets[:2]:  # First 2 sheets
                    df = pd.read_excel(excel_file, sheet_name=sheet, nrows=5)
                    print(f"   - {sheet}: {len(df.columns)} columns, sample rows:")
                    print(f"     Columns: {list(df.columns)[:5]}...")
            except Exception as e:
                print(f"   ⚠️ Could not parse: {e}")

    except ImportError:
        print("⚠️ pandas not installed, skipping Excel analysis")

    # Try to analyze CSV files
    try:
        import pandas as pd

        for csv_file in csv_files[:3]:  # Analyze first 3
            print(f"\n📋 Analyzing {csv_file.name}...")
            try:
                df = pd.read_csv(csv_file, nrows=5)
                print(f"   Columns: {list(df.columns)[:8]}...")
                print(f"   Rows (preview): {len(df)}")
            except Exception as e:
                print(f"   ⚠️ Could not parse: {e}")

    except ImportError:
        pass

    # Check ZIP contents
    import zipfile

    for zip_file in zip_files[:3]:  # Analyze first 3
        print(f"\n📦 Analyzing {zip_file.name}...")
        try:
            with zipfile.ZipFile(zip_file, "r") as zf:
                names = zf.namelist()
                print(f"   Contains {len(names)} files")
                for name in names[:5]:
                    info = zf.getinfo(name)
                    print(f"   - {name} ({format_size(info.file_size)})")
                if len(names) > 5:
                    print(f"   ... and {len(names) - 5} more files")
        except Exception as e:
            print(f"   ⚠️ Could not open: {e}")

    # Validate against expected stats
    print_section("Expected Dataset Statistics (from paper)")
    for key, value in EXPECTED_STATS.items():
        print(f"   {key}: {value}")

    return evaluation


def show_integration_info():
    """Show information about integrating CShaper with other datasets."""
    print_header("🔗 Integration with Other Datasets")

    print(
        """
The CShaper morphological atlas can be integrated with other C. elegans data:

1. 📊 SPATIAL-TRANSCRIPTOME INTEGRATION
   - CShaper provides cell shapes and positions for 4-350 cell stages
   - WormGUIDES provides 4D nuclear positions (t001-t360, ~20-380 min)
   - Large et al. 2025 provides transcriptome data

2. 🔑 KEY ALIGNMENT POINTS
   - Both use the same cell naming convention (Sulston lineage)
   - CShaper time: ~1.5 min resolution
   - WormGUIDES time: ~1 min resolution (60 sec/timepoint)
   - Temporal alignment: CShaper covers ~20-380 min post-fertilization

3. 📐 SPATIAL NORMALIZATION
   - CShaper provides normalized embryo coordinates
   - Axes: x=A-P, y=L-R, z=D-V
   - Embryo dimensions: ~27×18×18 μm (cylindroid)

4. 🎯 HIGH-VALUE DATA PRODUCTS
   - Cell volumes for all 656 unique cells
   - 479 effective cell-cell contacts
   - Cell shape irregularity scores
   - Sister cell volume ratios

5. 📋 INTEGRATION WORKFLOW
   a. Load CShaper cell morphology data
   b. Map cell names to WormGUIDES spatial data
   c. Match to Large2025 transcriptome via cell_type/lineage
   d. Build unified AnnData with spatial + shape + expression

6. ⚠️ LIMITATIONS
   - CShaper covers 17 embryos (vs WormGUIDES single reference)
   - Cell loss ratio increases after 200-cell stage
   - Not all cells have complete lifespan data
"""
    )


def generate_integration_mapping():
    """Generate a mapping file for CShaper-WormGUIDES integration."""
    print_header("🗺️  Generating Integration Mapping")

    mapping = {
        "source": "CShaper (Cao et al. 2020)",
        "doi": "10.1038/s41467-020-19863-x",
        "target_datasets": ["WormGUIDES", "Large2025"],
        "time_alignment": {
            "cshaper_resolution_min": 1.5,
            "wormguides_resolution_sec": 60,
            "overlap_range_min": [20, 380],
            "developmental_stages": {
                "4_cell": {"time_min": 0, "cell_count": 4},
                "8_cell": {"time_min": 15, "cell_count": 8},
                "16_cell": {"time_min": 30, "cell_count": 16},
                "26_cell": {"time_min": 50, "cell_count": 26},
                "100_cell": {"time_min": 150, "cell_count": 100},
                "200_cell": {"time_min": 250, "cell_count": 200},
                "350_cell": {"time_min": 350, "cell_count": 350},
            },
        },
        "spatial_alignment": {
            "coordinate_system": "cartesian",
            "axes": {
                "x": "anterior-posterior",
                "y": "left-right",
                "z": "dorsal-ventral",
            },
            "normalization": "linear_to_cylindroid",
            "embryo_dimensions_um": {"height": 18, "semimajor": 27, "semiminor": 18},
        },
        "cell_naming": {
            "convention": "Sulston_lineage",
            "founder_cells": ["AB", "P1", "EMS", "P2", "MS", "E", "C", "D", "P3", "P4"],
            "germline_precursors": ["Z2", "Z3"],
        },
        "key_morphology_fields": [
            "cell_volume",
            "surface_area",
            "irregularity_score",
            "cell_position_xyz",
            "contact_area",
            "contact_duration",
        ],
    }

    # Save mapping
    mapping_path = DATA_DIR / "integration_mapping.json"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"✅ Mapping saved to: {mapping_path}")

    # Print summary
    print("\n📋 Integration Mapping Summary:")
    print(json.dumps(mapping, indent=2))

    return mapping


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download and evaluate CShaper morphological atlas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_cshaper.py info       # Show dataset info
  python evaluate_cshaper.py list       # List available files
  python evaluate_cshaper.py download   # Download all files
  python evaluate_cshaper.py evaluate   # Evaluate downloaded data
  python evaluate_cshaper.py integrate  # Show integration info
  python evaluate_cshaper.py mapping    # Generate integration mapping
  python evaluate_cshaper.py all        # Run all steps
        """,
    )

    parser.add_argument(
        "command",
        choices=["info", "list", "download", "evaluate", "integrate", "mapping", "all"],
        default="all",
        nargs="?",
        help="Command to run (default: all)",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download of existing files",
    )

    args = parser.parse_args()

    print_header("🔬 CShaper Morphological Atlas - Evaluation Tool")
    print("Paper: Cao et al. 2020, Nature Communications")
    print("DOI: 10.1038/s41467-020-19863-x")

    if args.command == "info":
        show_article_info()

    elif args.command == "list":
        list_files()

    elif args.command == "download":
        download_all_files(force=args.force)

    elif args.command == "evaluate":
        evaluate_dataset()

    elif args.command == "integrate":
        show_integration_info()

    elif args.command == "mapping":
        generate_integration_mapping()

    elif args.command == "all":
        show_article_info()
        list_files()
        download_all_files(force=args.force)
        evaluate_dataset()
        show_integration_info()
        generate_integration_mapping()

    print("\n✨ Done!")


if __name__ == "__main__":
    main()
