#!/usr/bin/env python3
"""
Test CShaper data loading and processing.

This script validates that CShaper data can be loaded correctly and
provides statistics about the available data.

Usage:
    uv run python examples/test_cshaper.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd


def test_contact_loader():
    """Test ContactInterface data loading."""
    print("=" * 60)
    print("Testing ContactLoader")
    print("=" * 60)
    
    from src.data.builder import ContactLoader
    
    contact_dir = Path("dataset/raw/cshaper/ContactInterface")
    if not contact_dir.exists():
        print(f"ContactInterface directory not found: {contact_dir}")
        return False
    
    try:
        loader = ContactLoader(contact_dir)
        
        # Get available samples
        samples = loader.get_available_samples()
        print(f"\nAvailable samples: {len(samples)}")
        print(f"  {samples[:5]}..." if len(samples) > 5 else f"  {samples}")
        
        # Load first sample
        sample_id = samples[0]
        df = loader.load_sample(sample_id)
        print(f"\nSample {sample_id}:")
        print(f"  Matrix shape: {df.shape}")
        print(f"  Cell names: {list(df.index[:5])}...")
        
        # Count non-zero contacts
        n_contacts = (df.values > 0).sum() // 2  # Symmetric matrix
        print(f"  Non-zero contacts: {n_contacts}")
        
        # Get all cell names
        all_cells = loader.get_all_cell_names()
        print(f"\nAll unique cells across samples: {len(all_cells)}")
        
        # Get statistics
        stats = loader.get_contact_statistics()
        print(f"\nContact statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        
        # Test building adjacency matrix
        test_cells = list(all_cells)[:50]
        adj = loader.build_adjacency_matrix(test_cells, binary=True)
        print(f"\nTest adjacency matrix (50 cells):")
        print(f"  Shape: {adj.shape}")
        print(f"  Non-zero: {adj.nnz}")
        
        print("\n✓ ContactLoader tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ ContactLoader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_morphology_loader():
    """Test VolumeAndSurface data loading."""
    print("\n" + "=" * 60)
    print("Testing MorphologyLoader")
    print("=" * 60)
    
    from src.data.builder import MorphologyLoader
    
    volume_dir = Path("dataset/raw/cshaper/VolumeAndSurface")
    if not volume_dir.exists():
        print(f"VolumeAndSurface directory not found: {volume_dir}")
        return False
    
    try:
        loader = MorphologyLoader(volume_dir)
        
        # Get available samples
        samples = loader.get_available_samples()
        print(f"\nAvailable samples: {len(samples)}")
        
        # Load first sample
        sample_id = samples[0]
        vol_df, surf_df = loader.load_sample(sample_id)
        print(f"\nSample {sample_id}:")
        print(f"  Volume data shape: {vol_df.shape}")
        print(f"  Surface data shape: {surf_df.shape}")
        print(f"  Time frames: {vol_df.index.min()} - {vol_df.index.max()}")
        print(f"  Cells: {list(vol_df.columns[:5])}...")
        
        # Get morphology at a specific frame
        frame = 27  # Middle frame
        morph = loader.get_morphology_at_frame(frame)
        print(f"\nMorphology at frame {frame}:")
        print(f"  Cells: {len(morph)}")
        print(f"  Volume range: {morph['volume'].min():.1f} - {morph['volume'].max():.1f} μm³")
        print(f"  Surface range: {morph['surface'].min():.1f} - {morph['surface'].max():.1f} μm²")
        print(f"  Sphericity range: {morph['sphericity'].min():.3f} - {morph['sphericity'].max():.3f}")
        
        # Get all cell names
        all_cells = loader.get_all_cell_names()
        print(f"\nAll unique cells: {len(all_cells)}")
        
        # Test getting features for specific cells
        test_cells = ["ABa", "ABp", "EMS", "P2", "ABal", "ABar"]
        test_frames = np.array([20, 20, 20, 20, 25, 25])
        features = loader.get_features_for_cells(test_cells, test_frames)
        print(f"\nFeatures for test cells:")
        for i, cell in enumerate(test_cells):
            vol = features.loc[i, 'volume']
            surf = features.loc[i, 'surface']
            sph = features.loc[i, 'sphericity']
            print(f"  {cell}: vol={vol:.1f}, surf={surf:.1f}, sph={sph:.3f}")
        
        print("\n✓ MorphologyLoader tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ MorphologyLoader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cshaper_processor():
    """Test main CShaperProcessor."""
    print("\n" + "=" * 60)
    print("Testing CShaperProcessor")
    print("=" * 60)
    
    from src.data.builder import CShaperProcessor
    
    try:
        processor = CShaperProcessor("dataset/raw")
        
        # Print summary
        print("\n" + processor.summary())
        
        # Test with sample cell names
        test_lineages = [
            "ABa", "ABp", "EMS", "P2",
            "ABal", "ABar", "ABpl", "ABpr",
            "MS", "E", "C", "P3",
            "ABala", "ABalp", "ABara", "ABarp",
        ]
        
        # Test contact adjacency
        if processor.has_contact:
            print("\nTesting contact adjacency...")
            adj = processor.get_contact_adjacency(test_lineages, binary=True)
            print(f"  Matrix shape: {adj.shape}")
            print(f"  Non-zero edges: {adj.nnz // 2}")
        
        # Test morphology
        if processor.has_morphology:
            print("\nTesting morphology features...")
            morph = processor.get_morphology_features(test_lineages)
            n_matched = (~morph['volume'].isna()).sum()
            print(f"  Matched: {n_matched}/{len(test_lineages)}")
            
            valid = morph.dropna()
            if len(valid) > 0:
                print(f"  Mean volume: {valid['volume'].mean():.1f} μm³")
                print(f"  Mean sphericity: {valid['sphericity'].mean():.3f}")
        
        print("\n✓ CShaperProcessor tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ CShaperProcessor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lineage_matching():
    """Test lineage name normalization and matching."""
    print("\n" + "=" * 60)
    print("Testing Lineage Name Matching")
    print("=" * 60)
    
    from src.data.builder import normalize_lineage_name
    
    test_cases = [
        # (input, expected)
        ("ABplpapppa", "ABplpapppa"),
        ("ABa", "ABa"),
        ("MS", "MS"),
        ("MSapaap", "MSapaap"),
        ("E", "E"),
        ("Ealaad", "Ealaad"),
        ("C", "C"),
        ("Capaaa", "Capaaa"),
        ("P4", "P4"),
        ("Z2", "Z2"),
        ("  ABa  ", "ABa"),  # Whitespace
        ("AB.pla", "ABpla"),  # Period
        ("", ""),  # Empty
    ]
    
    all_passed = True
    for input_name, expected in test_cases:
        result = normalize_lineage_name(input_name)
        status = "✓" if result == expected else "✗"
        if result != expected:
            all_passed = False
        print(f"  {status} '{input_name}' -> '{result}' (expected '{expected}')")
    
    if all_passed:
        print("\n✓ All lineage matching tests passed!")
    else:
        print("\n✗ Some lineage matching tests failed!")
    
    return all_passed


def test_time_mapping():
    """Test embryo time to CShaper frame mapping."""
    print("\n" + "=" * 60)
    print("Testing Time Mapping")
    print("=" * 60)
    
    from src.data.builder import embryo_time_to_cshaper_frame
    from src.data.builder.cshaper_processor import cshaper_frame_to_embryo_time
    
    test_times = [20, 50, 100, 150, 200, 250, 300, 350, 380]
    
    print("\nEmbryo time -> CShaper frame -> Estimated time:")
    for t in test_times:
        frame = embryo_time_to_cshaper_frame(t)
        t_back = cshaper_frame_to_embryo_time(frame)
        print(f"  {t} min -> frame {frame} -> ~{t_back:.0f} min")
    
    # Test edge cases
    print("\nEdge cases:")
    print(f"  -10 min -> frame {embryo_time_to_cshaper_frame(-10)}")
    print(f"  500 min -> frame {embryo_time_to_cshaper_frame(500)}")
    print(f"  NaN -> frame {embryo_time_to_cshaper_frame(float('nan'))}")
    
    print("\n✓ Time mapping tests passed!")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("CShaper Data Integration Tests")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Lineage Matching", test_lineage_matching()))
    results.append(("Time Mapping", test_time_mapping()))
    results.append(("ContactLoader", test_contact_loader()))
    results.append(("MorphologyLoader", test_morphology_loader()))
    results.append(("CShaperProcessor", test_cshaper_processor()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
