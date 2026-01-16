"""
CShaper Dataset Analysis Script

This script analyzes the CShaper morphological atlas data for C. elegans embryos.

Data Structure:
1. ContactInterface/Sample*_Stat.csv - Cell-cell contact surface areas (symmetric matrix)
2. VolumeAndSurface/Sample*_volume.csv - Cell volumes over time
3. VolumeAndSurface/Sample*_surface.csv - Cell surface areas over time  
4. Standard Dataset 1/*.mat - Normalized cell position + morphology data (MATLAB format)
5. Standard Dataset 2/Seg_*.mat - 3D matrices for embryo morphology (MATLAB format)

Samples 04-20: 17 wild-type embryos with membrane segmentation
Samples 04-49: 46 embryos with nucleus position data
54 time frames spanning the developmental period
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

# Try to import scipy for .mat file reading
try:
    from scipy.io import loadmat
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not installed. Cannot read .mat files.")


def get_cshaper_path() -> Path:
    """Get the path to the CShaper dataset."""
    base_path = Path(__file__).parent.parent / "dataset" / "raw" / "cshaper"
    if not base_path.exists():
        raise FileNotFoundError(f"CShaper data not found at {base_path}")
    return base_path


def analyze_contact_interface(data_path: Path) -> dict:
    """
    Analyze cell-cell contact interface data.
    
    Each CSV is a symmetric matrix where entry (i,j) represents the contact
    surface area between cell i and cell j.
    """
    contact_dir = data_path / "ContactInterface"
    results = {
        "samples": [],
        "timepoints_per_sample": {},
        "max_cells_per_timepoint": {},
        "total_contacts": {},
        "sample_data": {}
    }
    
    csv_files = sorted(contact_dir.glob("Sample*_Stat.csv"))
    print(f"\n{'='*60}")
    print("CONTACT INTERFACE ANALYSIS")
    print(f"{'='*60}")
    print(f"Found {len(csv_files)} sample files")
    
    for csv_file in csv_files:
        sample_id = csv_file.stem.replace("_Stat", "")
        results["samples"].append(sample_id)
        
        try:
            # Read the CSV - it's a complex format with timepoints as rows
            # First column after index contains timepoint number
            df = pd.read_csv(csv_file, header=0, index_col=0)
            
            # The first row is cell names, data starts from row 1
            # Format: Each row is a timepoint, columns are cell-cell contact values
            
            # Get cell names from header
            cell_names = [col for col in df.columns if col and not col.startswith('Unnamed')]
            
            # Count non-empty rows (timepoints)
            non_empty_rows = df.dropna(how='all')
            n_timepoints = len(non_empty_rows)
            
            results["timepoints_per_sample"][sample_id] = n_timepoints
            results["max_cells_per_timepoint"][sample_id] = len(cell_names)
            
            # Count non-zero contacts
            total_contacts = (df > 0).sum().sum()
            results["total_contacts"][sample_id] = int(total_contacts)
            
            print(f"  {sample_id}: {n_timepoints} timepoints, {len(cell_names)} cells, {total_contacts} total contact entries")
            
        except Exception as e:
            print(f"  {sample_id}: Error reading - {e}")
            
    return results


def analyze_volume_surface(data_path: Path) -> dict:
    """
    Analyze cell volume and surface area data.
    
    Each CSV has cell names as columns and timepoints as rows.
    """
    vs_dir = data_path / "VolumeAndSurface"
    results = {
        "volume_samples": [],
        "surface_samples": [],
        "cell_lineages": set(),
        "volume_stats": {},
        "surface_stats": {}
    }
    
    print(f"\n{'='*60}")
    print("VOLUME AND SURFACE ANALYSIS")
    print(f"{'='*60}")
    
    # Analyze volume files
    volume_files = sorted(vs_dir.glob("Sample*_volume.csv"))
    print(f"\nVolume files: {len(volume_files)}")
    
    for vol_file in volume_files:
        sample_id = vol_file.stem.replace("_volume", "")
        results["volume_samples"].append(sample_id)
        
        try:
            df = pd.read_csv(vol_file, header=0, index_col=0)
            
            # Get cell names (column headers)
            cell_names = [c for c in df.columns if c and not c.startswith('Unnamed')]
            results["cell_lineages"].update(cell_names)
            
            # Count non-empty data
            n_timepoints = len(df.dropna(how='all'))
            n_cells = len(cell_names)
            
            # Volume statistics
            vol_values = df.values.flatten()
            vol_values = vol_values[~np.isnan(vol_values)]
            vol_values = vol_values[vol_values > 0]
            
            if len(vol_values) > 0:
                results["volume_stats"][sample_id] = {
                    "n_timepoints": n_timepoints,
                    "n_cells": n_cells,
                    "min_volume": float(np.min(vol_values)),
                    "max_volume": float(np.max(vol_values)),
                    "mean_volume": float(np.mean(vol_values)),
                    "median_volume": float(np.median(vol_values))
                }
                print(f"  {sample_id}: {n_timepoints} timepoints, {n_cells} cells")
                print(f"    Volume range: {np.min(vol_values):.0f} - {np.max(vol_values):.0f} (mean: {np.mean(vol_values):.0f})")
                
        except Exception as e:
            print(f"  {sample_id}: Error - {e}")
    
    # Analyze surface files
    surface_files = sorted(vs_dir.glob("Sample*_surface.csv"))
    print(f"\nSurface area files: {len(surface_files)}")
    
    for surf_file in surface_files:
        sample_id = surf_file.stem.replace("_surface", "")
        results["surface_samples"].append(sample_id)
        
        try:
            df = pd.read_csv(surf_file, header=0, index_col=0)
            n_timepoints = len(df.dropna(how='all'))
            
            surf_values = df.values.flatten()
            surf_values = surf_values[~np.isnan(surf_values)]
            surf_values = surf_values[surf_values > 0]
            
            if len(surf_values) > 0:
                results["surface_stats"][sample_id] = {
                    "n_timepoints": n_timepoints,
                    "min_surface": float(np.min(surf_values)),
                    "max_surface": float(np.max(surf_values)),
                    "mean_surface": float(np.mean(surf_values))
                }
                print(f"  {sample_id}: Surface range {np.min(surf_values):.0f} - {np.max(surf_values):.0f}")
                
        except Exception as e:
            print(f"  {sample_id}: Error - {e}")
    
    results["cell_lineages"] = sorted(results["cell_lineages"])
    print(f"\nTotal unique cell lineage names: {len(results['cell_lineages'])}")
    
    return results


def analyze_standard_dataset_1(data_path: Path) -> dict:
    """
    Analyze Standard Dataset 1 (Cell-Nucleus Position + Cell-Membrane Morphology).
    
    - WorkSpace_CellName.mat: Cell naming information (9x1 table for AB2-AB256, MS1-MS32, etc.)
    - WorkSpace_Dataset_n.mat: Data for each of 54 frames
      - Row 1: Nucleus positions (46 columns for Samples 04-49)
      - Row 2: Membrane morphology (17 columns for Samples 04-20)
    """
    sd1_dir = data_path / "Standard Dataset 1"
    results = {
        "n_frames": 0,
        "files_found": [],
        "cell_name_info": None
    }
    
    print(f"\n{'='*60}")
    print("STANDARD DATASET 1 ANALYSIS")
    print(f"{'='*60}")
    
    if not sd1_dir.exists():
        print("  Directory not found!")
        return results
    
    # List all .mat files
    mat_files = sorted(sd1_dir.glob("*.mat"))
    results["files_found"] = [f.name for f in mat_files]
    
    print(f"Found {len(mat_files)} .mat files:")
    
    # Count dataset files
    dataset_files = [f for f in mat_files if f.name.startswith("WorkSpace_Dataset_")]
    results["n_frames"] = len(dataset_files)
    print(f"  - Dataset frames: {results['n_frames']} (WorkSpace_Dataset_1.mat to WorkSpace_Dataset_54.mat)")
    
    # Check for cell name file
    cellname_file = sd1_dir / "WorkSpace_CellName.mat"
    if cellname_file.exists():
        print(f"  - Cell name mapping: WorkSpace_CellName.mat")
        
    if HAS_SCIPY and cellname_file.exists():
        try:
            mat_data = loadmat(str(cellname_file))
            print(f"    Keys in CellName file: {[k for k in mat_data.keys() if not k.startswith('__')]}")
        except Exception as e:
            print(f"    Error reading: {e}")
    
    # Try to read one dataset file
    if HAS_SCIPY and dataset_files:
        try:
            mat_data = loadmat(str(dataset_files[0]))
            keys = [k for k in mat_data.keys() if not k.startswith('__')]
            print(f"  Example dataset keys: {keys}")
        except Exception as e:
            print(f"  Error reading example: {e}")
            
    return results


def analyze_standard_dataset_2(data_path: Path) -> dict:
    """
    Analyze Standard Dataset 2 (3D Matrix for Embryo Morphology).
    
    Files: Seg_n_I.mat where n=1-54 (frame), I=04-20 (embryo)
    Matrix size: 256 x 114 x 184 voxels at 0.25 μm resolution
    - 256: anterior-posterior direction
    - 114: left-right direction  
    - 184: dorsal-ventral direction
    """
    sd2_dir = data_path / "Standard Dataset 2"
    results = {
        "n_files": 0,
        "frames": set(),
        "embryos": set(),
        "matrix_shape": (256, 114, 184),
        "voxel_resolution_um": 0.25
    }
    
    print(f"\n{'='*60}")
    print("STANDARD DATASET 2 ANALYSIS")
    print(f"{'='*60}")
    
    if not sd2_dir.exists():
        print("  Directory not found!")
        return results
    
    mat_files = sorted(sd2_dir.glob("Seg_*.mat"))
    results["n_files"] = len(mat_files)
    
    for f in mat_files:
        # Parse filename: Seg_n_I.mat
        parts = f.stem.split("_")
        if len(parts) >= 3:
            results["frames"].add(int(parts[1]))
            results["embryos"].add(parts[2])
    
    results["frames"] = sorted(results["frames"])
    results["embryos"] = sorted(results["embryos"])
    
    print(f"Found {results['n_files']} segmentation files")
    print(f"  Frames: {len(results['frames'])} (range: {min(results['frames'])} - {max(results['frames'])})")
    print(f"  Embryos: {len(results['embryos'])} ({', '.join(results['embryos'][:5])}...)")
    print(f"  Expected matrix shape: {results['matrix_shape']} at {results['voxel_resolution_um']} μm/voxel")
    
    # Physical dimensions
    dims_um = tuple(d * results["voxel_resolution_um"] for d in results["matrix_shape"])
    print(f"  Physical dimensions: {dims_um[0]:.1f} x {dims_um[1]:.1f} x {dims_um[2]:.1f} μm")
    
    # Try to read one file
    if HAS_SCIPY and mat_files:
        try:
            mat_data = loadmat(str(mat_files[0]))
            keys = [k for k in mat_data.keys() if not k.startswith('__')]
            print(f"  Example file keys: {keys}")
            
            for key in keys:
                data = mat_data[key]
                if hasattr(data, 'shape'):
                    print(f"    {key}: shape={data.shape}, dtype={data.dtype}")
        except Exception as e:
            print(f"  Error reading example: {e}")
    
    return results


def extract_cell_lineage_tree(cell_names: list) -> dict:
    """
    Extract lineage tree structure from cell names.
    
    C. elegans cell naming convention:
    - P0 -> AB, P1
    - P1 -> EMS, P2  
    - EMS -> MS, E
    - P2 -> C, P3
    - P3 -> D, P4
    - Each division adds 'a' (anterior) or 'p' (posterior) suffix
    """
    tree = {
        "founder_cells": ["AB", "P1", "EMS", "P2", "MS", "E", "C", "P3", "D", "P4"],
        "lineage_counts": defaultdict(int),
        "max_divisions": 0
    }
    
    for name in cell_names:
        if not name:
            continue
            
        # Count divisions based on name length
        if name.startswith("AB"):
            n_divisions = len(name) - 2  # 'AB' + divisions
            tree["lineage_counts"]["AB"] += 1
        elif name.startswith("MS"):
            n_divisions = len(name) - 2
            tree["lineage_counts"]["MS"] += 1
        elif name.startswith("E") and not name.startswith("EMS"):
            n_divisions = len(name) - 1
            tree["lineage_counts"]["E"] += 1
        elif name.startswith("C") and len(name) > 1:
            n_divisions = len(name) - 1
            tree["lineage_counts"]["C"] += 1
        elif name.startswith("D") and len(name) > 1:
            n_divisions = len(name) - 1
            tree["lineage_counts"]["D"] += 1
        elif name.startswith("P"):
            tree["lineage_counts"]["P"] += 1
            n_divisions = 0
        elif name.startswith("Z"):
            tree["lineage_counts"]["Z"] += 1
            n_divisions = 0
        else:
            n_divisions = 0
            
        tree["max_divisions"] = max(tree["max_divisions"], n_divisions)
    
    return tree


def print_summary(contact_results, volume_results, sd1_results, sd2_results):
    """Print a comprehensive summary of the CShaper dataset."""
    
    print(f"\n{'='*60}")
    print("CSHAPER DATASET SUMMARY")
    print(f"{'='*60}")
    
    print("\n1. DATA COVERAGE:")
    print(f"   - Contact Interface: {len(contact_results['samples'])} samples")
    print(f"   - Volume/Surface: {len(volume_results['volume_samples'])} samples")
    print(f"   - Standard Dataset 1: {sd1_results['n_frames']} frames")
    print(f"   - Standard Dataset 2: {sd2_results['n_files']} 3D segmentations")
    
    print("\n2. CELL LINEAGE INFORMATION:")
    if volume_results['cell_lineages']:
        tree = extract_cell_lineage_tree(volume_results['cell_lineages'])
        print(f"   - Total unique cell names: {len(volume_results['cell_lineages'])}")
        print(f"   - Maximum division depth: {tree['max_divisions']}")
        print(f"   - Cells per founder lineage:")
        for founder, count in sorted(tree['lineage_counts'].items()):
            print(f"     {founder}: {count}")
    
    print("\n3. SPATIAL RESOLUTION:")
    print(f"   - 3D matrix: {sd2_results['matrix_shape']}")
    print(f"   - Voxel size: {sd2_results['voxel_resolution_um']} μm")
    
    print("\n4. TEMPORAL COVERAGE:")
    if sd2_results['frames']:
        print(f"   - Frames: {min(sd2_results['frames'])} - {max(sd2_results['frames'])}")
    
    print("\n5. POTENTIAL USES FOR NEMACONTEXT:")
    print("   - Cell-cell contact graphs (ContactInterface)")
    print("   - Cell volume/surface features (VolumeAndSurface)")  
    print("   - 3D spatial coordinates (Standard Dataset 1)")
    print("   - Full 3D morphology (Standard Dataset 2)")
    print("   - Cell lineage naming -> binary tree path encoding")


def main():
    """Main analysis function."""
    print("CShaper Dataset Analysis")
    print("=" * 60)
    print("Reference: Cao et al., Nature Communications 2020")
    print("'Establishment of a morphological atlas of the")
    print(" Caenorhabditis elegans embryo using deep-learning-based")
    print(" 4D segmentation'")
    print("=" * 60)
    
    try:
        data_path = get_cshaper_path()
        print(f"\nData path: {data_path}")
        
        # List contents
        print("\nDirectory contents:")
        for item in sorted(data_path.iterdir()):
            if item.is_dir():
                n_files = len(list(item.iterdir()))
                print(f"  {item.name}/ ({n_files} items)")
            else:
                print(f"  {item.name}")
        
        # Run analyses
        contact_results = analyze_contact_interface(data_path)
        volume_results = analyze_volume_surface(data_path)
        sd1_results = analyze_standard_dataset_1(data_path)
        sd2_results = analyze_standard_dataset_2(data_path)
        
        # Print summary
        print_summary(contact_results, volume_results, sd1_results, sd2_results)
        
        return {
            "contact": contact_results,
            "volume_surface": volume_results,
            "standard_dataset_1": sd1_results,
            "standard_dataset_2": sd2_results
        }
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    results = main()
