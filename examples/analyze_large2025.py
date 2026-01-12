#!/usr/bin/env python3
"""
Analyze Large et al. 2025 (GSE292756) dataset and its integration potential.

This script provides:
1. Dataset overview and statistics
2. Comparison with Packer 2019 (legacy)
3. Matching analysis with WormGUIDES spatial data
4. Cell type and lineage distribution visualization

Usage:
    uv run python examples/analyze_large2025.py
"""

import gzip
from collections import Counter
from pathlib import Path

import pandas as pd


def load_large2025_annotations(data_dir: str = "dataset/raw") -> pd.DataFrame:
    """Load Large 2025 cell annotations."""
    path = Path(data_dir) / "large2025" / "GSE292756_cell_annotations.csv.gz"
    if not path.exists():
        raise FileNotFoundError(
            f"Large 2025 annotations not found at {path}.\n"
            "Run: uv run python -m src.data.downloader --source large2025"
        )

    with gzip.open(path, "rt") as f:
        return pd.read_csv(f)


def load_wormguides_cells(data_dir: str = "dataset/raw") -> set[str]:
    """Load unique cell names from WormGUIDES nuclei files."""
    nuclei_dir = Path(data_dir) / "wormguides" / "nuclei_files"
    if not nuclei_dir.exists():
        print(f"⚠️  WormGUIDES nuclei files not found at {nuclei_dir}")
        return set()

    all_cells = set()
    for nuclei_file in nuclei_dir.glob("t*-nuclei"):
        try:
            with open(nuclei_file, "r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 10:
                        cell_name = parts[9].strip().strip('"')
                        if cell_name and cell_name != "Nuc":
                            all_cells.add(cell_name)
        except Exception:
            continue

    return all_cells


def analyze_dataset_overview(df: pd.DataFrame) -> None:
    """Print dataset overview statistics."""
    print("=" * 70)
    print("📊 Large et al. 2025 (GSE292756) Dataset Overview")
    print("=" * 70)

    # Basic counts
    print(f"\n{'Total cells:':<35} {len(df):,}")
    print(f"{'Unique cell types:':<35} {df['cell_type'].nunique()}")

    # Species breakdown
    print("\n📋 Species Distribution:")
    print("-" * 40)
    for species, count in df["species"].value_counts().items():
        pct = 100 * count / len(df)
        print(f"  {species:<20} {count:>10,} ({pct:>5.1f}%)")

    # Cell type assignment
    print("\n📋 Cell Type Assignment:")
    print("-" * 40)
    assigned = df[df["cell_type"] != "unassigned"]
    unassigned = df[df["cell_type"] == "unassigned"]
    print(
        f"  {'Assigned:':<20} {len(assigned):>10,} ({100 * len(assigned) / len(df):>5.1f}%)"
    )
    print(
        f"  {'Unassigned:':<20} {len(unassigned):>10,} ({100 * len(unassigned) / len(df):>5.1f}%)"
    )


def analyze_cell_types(df: pd.DataFrame, top_n: int = 25) -> None:
    """Analyze cell type distribution."""
    print("\n" + "=" * 70)
    print("🔬 Cell Type Analysis")
    print("=" * 70)

    # Overall distribution
    print(f"\nTop {top_n} Cell Types (both species):")
    print("-" * 50)
    for i, (ct, count) in enumerate(
        df["cell_type"].value_counts().head(top_n).items(), 1
    ):
        pct = 100 * count / len(df)
        bar = "█" * int(pct / 2)
        print(f"  {i:>2}. {ct:<30} {count:>8,} ({pct:>5.1f}%) {bar}")

    # Cell type categories
    print("\n📊 Cell Type Categories:")
    print("-" * 50)

    categories = {
        "Body Wall Muscle (BWM)": df["cell_type"].str.contains("BWM", na=False),
        "Hypodermis": df["cell_type"].str.contains("hyp|Hypo", case=False, na=False),
        "Neurons": df["cell_type"].str.match(r"^[A-Z]{2,4}[LRDVlrdv]?$", na=False),
        "Seam cells": df["cell_type"].str.contains("Seam", na=False),
        "Pharynx": df["cell_type"].str.contains(
            "pharynx|pm|mc|m[1-5]", case=False, na=False
        ),
        "Intestine": df["cell_type"].str.contains("int|gut", case=False, na=False),
        "Germline": df["cell_type"].str.contains("germ|Z[23]", case=False, na=False),
    }

    for cat_name, mask in categories.items():
        count = mask.sum()
        if count > 0:
            print(f"  {cat_name:<25} {count:>10,} ({100 * count / len(df):>5.1f}%)")


def analyze_lineage_info(df: pd.DataFrame) -> dict:
    """Analyze lineage annotation quality."""
    print("\n" + "=" * 70)
    print("🌳 Lineage Annotation Analysis")
    print("=" * 70)

    stats = {}

    # Check available lineage columns
    lineage_cols = [c for c in df.columns if "lineage" in c.lower()]
    print(f"\nAvailable lineage columns: {lineage_cols}")

    # Primary lineage column
    if "lineage_complete" in df.columns:
        lin = df["lineage_complete"]

        total = len(lin)
        unassigned = (lin == "unassigned").sum()
        assigned = total - unassigned

        # Check for ambiguity markers
        has_x = lin.str.contains("x", na=False).sum()
        has_slash = lin.str.contains("/", na=False).sum()

        # Clean lineages (no x, no /)
        clean_mask = (
            (lin != "unassigned")
            & (~lin.str.contains("x", na=False))
            & (~lin.str.contains("/", na=False))
        )
        clean = clean_mask.sum()

        print("\n📊 lineage_complete column:")
        print("-" * 50)
        print(f"  {'Total cells:':<30} {total:>10,}")
        print(
            f"  {'With lineage assigned:':<30} {assigned:>10,} ({100 * assigned / total:>5.1f}%)"
        )
        print(
            f"  {'Unassigned:':<30} {unassigned:>10,} ({100 * unassigned / total:>5.1f}%)"
        )
        print()
        print(f"  {'Contains x (ambiguous):':<30} {has_x:>10,}")
        print(f"  {'Contains / (multiple):':<30} {has_slash:>10,}")
        print(
            f"  {'Clean (no x, no /):':<30} {clean:>10,} ({100 * clean / assigned:>5.1f}% of assigned)"
        )

        stats["total"] = total
        stats["assigned"] = assigned
        stats["has_x"] = has_x
        stats["clean"] = clean
        stats["unique_lineages"] = lin[lin != "unassigned"].nunique()

        print(f"\n  Unique lineage values: {stats['unique_lineages']}")

        # Sample clean lineages
        clean_lineages = lin[clean_mask].unique()[:20]
        print(f"\n  Sample clean lineages ({len(clean_lineages)} shown):")
        for i, ling in enumerate(clean_lineages, 1):
            print(f"    {i:>2}. {ling}")

    return stats


def analyze_wormguides_matching(df: pd.DataFrame, wg_cells: set[str]) -> None:
    """Analyze matching potential with WormGUIDES spatial data."""
    print("\n" + "=" * 70)
    print("🔗 WormGUIDES Spatial Data Matching Analysis")
    print("=" * 70)

    if not wg_cells:
        print("\n⚠️  WormGUIDES data not available. Skipping matching analysis.")
        print("   Run: uv run python -m src.data.downloader --source wormguides")
        return

    print(f"\n📍 WormGUIDES spatial data:")
    print(f"   Unique cell names: {len(wg_cells):,}")

    # Get cell types from Large 2025
    cell_types = df[df["cell_type"] != "unassigned"]["cell_type"].unique()

    # Direct matching by cell type name
    direct_matches = set(cell_types) & wg_cells
    print(f"\n📊 Direct cell type → WormGUIDES matching:")
    print(f"   Large 2025 cell types: {len(cell_types)}")
    print(f"   Direct matches: {len(direct_matches)}")

    if direct_matches:
        print(f"\n   Sample matched cell types:")
        for ct in sorted(direct_matches)[:15]:
            count = (df["cell_type"] == ct).sum()
            print(f"     {ct}: {count:,} cells")

    # Lineage-based matching
    if "lineage_complete" in df.columns:
        lin = df["lineage_complete"]
        clean_mask = (
            (lin != "unassigned")
            & (~lin.str.contains("x", na=False))
            & (~lin.str.contains("/", na=False))
        )
        clean_lineages = set(lin[clean_mask].unique())

        lineage_matches = clean_lineages & wg_cells

        print(f"\n📊 Clean lineage → WormGUIDES matching:")
        print(f"   Clean lineages in Large 2025: {len(clean_lineages)}")
        print(f"   Matched to WormGUIDES: {len(lineage_matches)}")
        print(f"   Match rate: {100 * len(lineage_matches) / len(clean_lineages):.1f}%")

        # Count cells with matched lineages
        matched_cells = df[lin.isin(lineage_matches)]
        print(f"\n   Cells with matched lineage: {len(matched_cells):,}")

        if lineage_matches:
            print(f"\n   Sample matched lineages:")
            for ling in sorted(lineage_matches)[:15]:
                count = (lin == ling).sum()
                print(f"     {ling}: {count:,} cells")


def analyze_time_coverage(df: pd.DataFrame) -> None:
    """Analyze temporal coverage of the dataset."""
    print("\n" + "=" * 70)
    print("⏱️  Temporal Coverage Analysis")
    print("=" * 70)

    time_col = "smoothed.embryo.time"
    if time_col not in df.columns:
        print("\n⚠️  Time column not found.")
        return

    time = df[time_col].dropna()

    print(f"\n📊 Time distribution (minutes post-fertilization):")
    print(f"   Min: {time.min():.1f}")
    print(f"   Max: {time.max():.1f}")
    print(f"   Mean: {time.mean():.1f}")
    print(f"   Median: {time.median():.1f}")

    # Time bins
    print("\n📊 Cells by time bin:")
    print("-" * 50)

    bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    df_temp = df.copy()
    df_temp["time_bin"] = pd.cut(df_temp[time_col], bins=bins)

    for bin_range, count in df_temp["time_bin"].value_counts().sort_index().items():
        if pd.notna(bin_range):
            pct = 100 * count / len(df)
            bar = "█" * int(pct)
            print(f"  {str(bin_range):<15} {count:>8,} ({pct:>5.1f}%) {bar}")

    # WormGUIDES overlap
    print("\n📊 WormGUIDES temporal overlap:")
    wg_start, wg_end = 20, 380
    overlap = df[(df[time_col] >= wg_start) & (df[time_col] <= wg_end)]
    print(f"   WormGUIDES range: {wg_start}-{wg_end} min")
    print(
        f"   Cells in overlap: {len(overlap):,} ({100 * len(overlap) / len(df):.1f}%)"
    )


def compare_with_packer(df: pd.DataFrame, data_dir: str = "dataset/raw") -> None:
    """Compare with Packer 2019 dataset if available."""
    print("\n" + "=" * 70)
    print("📈 Comparison with Packer 2019 (Legacy)")
    print("=" * 70)

    packer_path = Path(data_dir) / "packer2019" / "GSE126954_cell_annotation.csv.gz"

    if not packer_path.exists():
        print("\n⚠️  Packer 2019 data not available for comparison.")
        print("   Run: uv run python -m src.data.downloader --source packer")
        return

    with gzip.open(packer_path, "rt") as f:
        packer = pd.read_csv(f)

    print("\n📊 Dataset Size Comparison:")
    print("-" * 50)
    print(f"  {'Metric':<30} {'Packer 2019':>15} {'Large 2025':>15}")
    print("-" * 50)
    print(f"  {'Total cells':<30} {len(packer):>15,} {len(df):>15,}")
    print(
        f"  {'C. elegans cells':<30} {len(packer):>15,} {(df['species'] == 'C.elegans').sum():>15,}"
    )
    print(
        f"  {'C. briggsae cells':<30} {'N/A':>15} {(df['species'] == 'C.briggsae').sum():>15,}"
    )

    # Lineage quality comparison
    print("\n📊 Lineage Quality Comparison:")
    print("-" * 50)

    # Packer lineage stats
    if "lineage" in packer.columns:
        p_lin = packer["lineage"].dropna()
        p_total = len(p_lin)
        p_has_x = p_lin.str.contains("x", na=False).sum()
        p_has_slash = p_lin.str.contains("/", na=False).sum()
        p_clean = p_total - p_has_x - p_has_slash

        print(f"  {'Packer 2019:':<30}")
        print(f"    Cells with lineage: {p_total:,}")
        print(f"    With 'x' ambiguity: {p_has_x:,} ({100 * p_has_x / p_total:.1f}%)")
        print(f"    Clean lineages: ~{p_clean:,}")

    # Large 2025 lineage stats
    if "lineage_complete" in df.columns:
        l_lin = df["lineage_complete"]
        l_assigned = (l_lin != "unassigned").sum()
        l_has_x = l_lin.str.contains("x", na=False).sum()
        l_clean = l_assigned - l_has_x

        print(f"\n  {'Large 2025:':<30}")
        print(f"    Cells with lineage: {l_assigned:,}")
        print(
            f"    With 'x' ambiguity: {l_has_x:,} ({100 * l_has_x / l_assigned:.1f}%)"
        )
        print(f"    Clean lineages: {l_clean:,}")

    print("\n💡 Key improvements in Large 2025:")
    print("   - 2.8x more C. elegans cells")
    print("   - Additional C. briggsae data for cross-species comparison")
    print("   - Better cell type annotations (152 types vs variable)")
    print("   - Integrated lineage resolution workflow")


def print_recommendations(df: pd.DataFrame, stats: dict) -> None:
    """Print recommendations for using the dataset."""
    print("\n" + "=" * 70)
    print("💡 Recommendations for Integration")
    print("=" * 70)

    print("\n1️⃣  For PROTOTYPING (quick start):")
    print("   Use cells with clean lineage assignments:")
    print(f"   - {stats.get('clean', 0):,} cells with unambiguous lineage")
    print("   - Can directly match to WormGUIDES spatial positions")
    print()
    print("   Code:")
    print("   ```python")
    print("   clean_mask = (")
    print("       (df['lineage_complete'] != 'unassigned') &")
    print("       (~df['lineage_complete'].str.contains('x', na=False))")
    print("   )")
    print("   clean_df = df[clean_mask]")
    print("   ```")

    print("\n2️⃣  For FULL ANALYSIS:")
    print("   Use cell_type column for grouping:")
    print(f"   - {df['cell_type'].nunique()} unique cell types")
    print("   - 63.5% of cells have assigned cell type")
    print("   - Cell types map to biological function, not just lineage")

    print("\n3️⃣  For CROSS-SPECIES comparison:")
    print("   Use both C. elegans and C. briggsae data:")
    cel = (df["species"] == "C.elegans").sum()
    cbr = (df["species"] == "C.briggsae").sum()
    print(f"   - C. elegans: {cel:,} cells")
    print(f"   - C. briggsae: {cbr:,} cells")

    print("\n4️⃣  Key columns for integration:")
    print("   - cell_type: Terminal cell type annotation")
    print("   - lineage_complete: Full lineage path (use for spatial matching)")
    print("   - smoothed.embryo.time: Developmental time (for WormGUIDES alignment)")
    print("   - species: Filter for C. elegans only if needed")


def main():
    """Main analysis function."""
    print("\n" + "🧬 " * 20)
    print("\n  Large et al. 2025 (GSE292756) Dataset Analysis")
    print("  Science 2025 | DOI: 10.1126/science.adu8249")
    print("\n" + "🧬 " * 20)

    # Load data
    try:
        df = load_large2025_annotations()
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        return

    # Load WormGUIDES for matching analysis
    wg_cells = load_wormguides_cells()

    # Run analyses
    analyze_dataset_overview(df)
    analyze_cell_types(df)
    stats = analyze_lineage_info(df)
    analyze_time_coverage(df)
    analyze_wormguides_matching(df, wg_cells)
    compare_with_packer(df)
    print_recommendations(df, stats)

    print("\n" + "=" * 70)
    print("✨ Analysis complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Load expression matrix: scipy.io.mmread('...expression_matrix.mtx.gz')")
    print("  2. Match cell types to WormGUIDES spatial data")
    print("  3. Build multimodal embedding with transcriptome + spatial + lineage")
    print()


if __name__ == "__main__":
    main()
