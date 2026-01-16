#!/usr/bin/env python3
"""
Build enhanced AnnData with CShaper integration.

This script demonstrates how to build AnnData objects that include
CShaper morphological data:
- Cell-cell contact graphs (true physical neighbors)
- Cell morphology features (volume, surface, sphericity)
- Optionally CShaper standardized spatial coordinates

Usage:
    # Build enhanced dataset with contact graph and morphology
    uv run python examples/build_enhanced_anndata.py

    # Include CShaper spatial coordinates
    uv run python examples/build_enhanced_anndata.py --use-cshaper-spatial

    # Build extended variant (all cells)
    uv run python examples/build_enhanced_anndata.py --variant extended

    # Compare contact graph with k-NN graph
    uv run python examples/build_enhanced_anndata.py --compare-graphs
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.builder import EnhancedAnnDataBuilder, CShaperProcessor


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
        description="Build enhanced AnnData with CShaper integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--variant",
        type=str,
        choices=["complete", "extended"],
        default="complete",
        help="Which variant to build. Default: complete",
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
        help="Directory containing raw data. Default: dataset/raw",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset/processed",
        help="Directory for output files. Default: dataset/processed",
    )
    
    parser.add_argument(
        "--no-morphology",
        action="store_true",
        help="Skip adding morphology features",
    )
    
    parser.add_argument(
        "--no-contact-graph",
        action="store_true",
        help="Skip adding contact graph",
    )
    
    parser.add_argument(
        "--use-cshaper-spatial",
        action="store_true",
        help="Add CShaper standardized spatial coordinates",
    )
    
    parser.add_argument(
        "--contact-threshold",
        type=float,
        default=0.0,
        help="Minimum contact area threshold (μm²). Default: 0.0",
    )
    
    parser.add_argument(
        "--compare-graphs",
        action="store_true",
        help="Compare contact graph with k-NN spatial graph",
    )
    
    parser.add_argument(
        "--add-spatial-graph",
        action="store_true",
        help="Also build k-NN spatial graph for comparison",
    )
    
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=10,
        help="Number of neighbors for k-NN graph. Default: 10",
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save the result",
    )
    
    parser.add_argument(
        "--cshaper-summary",
        action="store_true",
        help="Print CShaper data summary and exit",
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    return parser.parse_args()


def print_cshaper_summary(data_dir: str) -> None:
    """Print summary of available CShaper data."""
    try:
        processor = CShaperProcessor(data_dir)
        print(processor.summary())
    except Exception as e:
        print(f"Error loading CShaper data: {e}")


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)
    
    logger = logging.getLogger(__name__)
    
    # If just printing summary
    if args.cshaper_summary:
        print_cshaper_summary(args.data_dir)
        return 0
    
    logger.info("=" * 60)
    logger.info("NemaContext Enhanced AnnData Builder")
    logger.info("=" * 60)
    
    # Initialize builder
    try:
        builder = EnhancedAnnDataBuilder(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
        )
    except Exception as e:
        logger.error(f"Failed to initialize builder: {e}")
        return 1
    
    # Build enhanced AnnData
    try:
        logger.info(f"Building {args.variant} AnnData from {args.source}...")
        logger.info(f"  Include morphology: {not args.no_morphology}")
        logger.info(f"  Include contact graph: {not args.no_contact_graph}")
        logger.info(f"  Include CShaper spatial: {args.use_cshaper_spatial}")
        
        adata = builder.build_with_cshaper(
            variant=args.variant,
            source=args.source,
            include_morphology=not args.no_morphology,
            include_contact_graph=not args.no_contact_graph,
            use_cshaper_spatial=args.use_cshaper_spatial,
            contact_threshold=args.contact_threshold,
            save=not args.no_save,
        )
        
        # Optionally add k-NN spatial graph
        if args.add_spatial_graph or args.compare_graphs:
            logger.info(f"Building k-NN spatial graph (k={args.n_neighbors})...")
            adata = builder.build_spatial_graph(adata, n_neighbors=args.n_neighbors)
        
        # Print summary
        print("\n" + builder.summary(adata))
        
        # Compare graphs if requested
        if args.compare_graphs:
            print("\n" + "=" * 60)
            print("Contact Graph vs k-NN Graph Comparison")
            print("=" * 60)
            
            try:
                comparison = builder.compare_graphs(adata)
                print(f"\nContact graph edges: {comparison['contact_edges']:,}")
                print(f"k-NN graph edges: {comparison['spatial_edges']:,}")
                print(f"\nOverlap analysis:")
                print(f"  Intersection: {comparison['intersection']:,} edges")
                print(f"  Contact-only: {comparison['contact_only']:,} edges")
                print(f"  k-NN-only: {comparison['spatial_only']:,} edges")
                print(f"\nMetrics:")
                print(f"  Jaccard similarity: {comparison['jaccard']:.3f}")
                print(f"  Precision (k-NN captures true contacts): {comparison['precision']:.3f}")
                print(f"  Recall (contacts captured by k-NN): {comparison['recall']:.3f}")
                print(f"  F1 score: {comparison['f1']:.3f}")
            except Exception as e:
                logger.error(f"Graph comparison failed: {e}")
        
        # Print CShaper info
        if "cshaper_info" in adata.uns:
            print("\n" + "=" * 60)
            print("CShaper Integration Details")
            print("=" * 60)
            info = adata.uns["cshaper_info"]
            for key, value in info.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")
        
        logger.info("Done!")
        return 0
        
    except Exception as e:
        logger.error(f"Build failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
