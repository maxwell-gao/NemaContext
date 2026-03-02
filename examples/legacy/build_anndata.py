#!/usr/bin/env python3
"""
Build trimodal AnnData for NemaContext.

This script constructs AnnData objects integrating:
- Transcriptome data (Large2025 or Packer2019)
- Spatial coordinates (WormGUIDES 4D nuclei)
- Lineage information (WormBase tree structure)

Two variants are available:
- complete: Only cells with all three modalities (~3.8k cells)
- extended: All cells with modality availability flags (~242k cells)

Usage:
    # Build complete trimodal dataset (recommended for model development)
    uv run python examples/build_anndata.py --variant complete

    # Build extended dataset with all cells
    uv run python examples/build_anndata.py --variant extended

    # Use Packer2019 instead of Large2025
    uv run python examples/build_anndata.py --source packer2019

    # Custom output directory
    uv run python examples/build_anndata.py --output-dir ./my_data

    # Skip PCA computation
    uv run python examples/build_anndata.py --no-pca
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.builder import TrimodalAnnDataBuilder


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build trimodal AnnData for NemaContext",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--variant",
        type=str,
        choices=["complete", "extended"],
        default="complete",
        help="Which variant to build: 'complete' (cells with all modalities) or 'extended' (all cells). Default: complete",
    )

    parser.add_argument(
        "--source",
        type=str,
        choices=["large2025", "packer2019"],
        default="large2025",
        help="Transcriptome data source. Default: large2025",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="dataset/raw",
        help="Directory containing raw data files. Default: dataset/raw",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset/processed",
        help="Directory for output h5ad files. Default: dataset/processed",
    )

    parser.add_argument(
        "--min-umi",
        type=int,
        default=500,
        help="Minimum UMI count per cell. Default: 500",
    )

    parser.add_argument(
        "--species",
        type=str,
        default="C.elegans",
        help="Filter to specific species. Use 'all' for no filter. Default: C.elegans",
    )

    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip log-normalization of expression data",
    )

    parser.add_argument(
        "--no-pca",
        action="store_true",
        help="Skip PCA computation",
    )

    parser.add_argument(
        "--n-pcs",
        type=int,
        default=50,
        help="Number of principal components. Default: 50",
    )

    parser.add_argument(
        "--add-spatial-graph",
        action="store_true",
        help="Add spatial neighborhood graph (KNN)",
    )

    parser.add_argument(
        "--add-lineage-graph",
        action="store_true",
        help="Add lineage adjacency graph",
    )

    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=10,
        help="Number of neighbors for spatial graph. Default: 10",
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save the result (for testing)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("NemaContext Trimodal AnnData Builder")
    logger.info("=" * 60)

    # Parse species filter
    species_filter = None if args.species.lower() == "all" else args.species

    # Initialize builder
    try:
        builder = TrimodalAnnDataBuilder(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
        )
    except Exception as e:
        logger.error(f"Failed to initialize builder: {e}")
        return 1

    # Build AnnData
    try:
        logger.info(f"Building {args.variant} AnnData from {args.source}...")
        logger.info(f"  Species filter: {species_filter or 'all'}")
        logger.info(f"  Min UMI: {args.min_umi}")
        logger.info(f"  Normalize: {not args.no_normalize}")
        logger.info(f"  Compute PCA: {not args.no_pca}")

        adata = builder.build(
            variant=args.variant,
            source=args.source,
            species_filter=species_filter,
            min_umi=args.min_umi,
            normalize=not args.no_normalize,
            compute_pca=not args.no_pca,
            n_pcs=args.n_pcs,
            save=not args.no_save,
        )

        # Add optional graphs
        if args.add_spatial_graph:
            logger.info(f"Building spatial graph with {args.n_neighbors} neighbors...")
            adata = builder.build_spatial_graph(adata, n_neighbors=args.n_neighbors)

        if args.add_lineage_graph:
            logger.info("Building lineage adjacency graph...")
            adata = builder.build_lineage_graph(adata)

        # Save if graphs were added
        if (args.add_spatial_graph or args.add_lineage_graph) and not args.no_save:
            output_path = (
                Path(args.output_dir) / f"nema_{args.variant}_{args.source}.h5ad"
            )
            logger.info(f"Saving updated AnnData to {output_path}")
            adata.write(output_path, compression="gzip")

        # Print summary
        print()
        print(builder.summary(adata))

        return 0

    except FileNotFoundError as e:
        logger.error(f"Required data file not found: {e}")
        logger.error("Please run the data downloader first:")
        logger.error("  uv run python -m src.data.downloader --source core")
        return 1

    except Exception as e:
        logger.exception(f"Build failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
