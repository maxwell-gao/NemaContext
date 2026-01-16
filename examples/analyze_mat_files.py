"""
CShaper .mat File Analysis Script

Standard Dataset 1 and 2 use MATLAB v7.3 format (HDF5-based).
This script analyzes their structure using h5py.
"""

import h5py
import numpy as np
from pathlib import Path


def get_cshaper_path() -> Path:
    """Get the path to the CShaper dataset."""
    base_path = Path(__file__).parent.parent / "dataset" / "raw" / "cshaper"
    return base_path


def read_matlab_string(f, ref):
    """Read a MATLAB string from HDF5 reference."""
    deref = f[ref]
    if isinstance(deref, h5py.Dataset):
        data = deref[:]
        if data.dtype.kind == 'u':  # uint type, likely ASCII chars
            return ''.join(chr(c) for c in data.flatten())
    return None


def analyze_standard_dataset_1():
    """Analyze Standard Dataset 1 structure."""
    print("=" * 70)
    print("STANDARD DATASET 1 ANALYSIS")
    print("=" * 70)
    
    sd1_path = get_cshaper_path() / "Standard Dataset 1"
    mat_file = sd1_path / "WorkSpace_Dataset_1.mat"
    
    with h5py.File(mat_file, 'r') as f:
        dataset = f['Dataset']
        print(f"\nDataset shape: {dataset.shape} (2 arrays: positions, morphology)")
        
        # Dataset[0,0] - 46 embryos' cell position data
        embryo_positions = f[dataset[0, 0]]
        print(f"\n[Dataset 0,0] Cell Positions: {embryo_positions.shape[0]} embryos")
        
        # Get first embryo's data structure
        emb0_ref = embryo_positions[0, 0]
        emb0 = f[emb0_ref]
        print(f"  Each embryo has {emb0.shape[1]} columns of data")
        
        # Analyze each column
        column_names = ['CellName', 'X', 'Y', 'Z', 'Time', 'Unknown1', 'Unknown2', 'Unknown3', 'Unknown4']
        
        print("\n  Column structure (Embryo 0, Frame 1):")
        for col in range(emb0.shape[1]):
            col_ref = emb0[0, col]
            col_data = f[col_ref]
            
            if col_data.dtype == 'object':
                # Reference array - contains cell names or nested data
                n_cells = col_data.shape[0]
                inner_ref = col_data[0, 0] if len(col_data.shape) > 1 else col_data[0]
                inner = f[inner_ref]
                
                # Check if it's a string
                if inner.dtype.kind == 'u' and inner.shape[0] < 100:
                    sample_str = read_matlab_string(f, inner_ref)
                    print(f"    Col {col} ({column_names[col]}): {n_cells} cells, e.g. '{sample_str}'")
                else:
                    print(f"    Col {col}: {n_cells} entries, inner shape {inner.shape}")
            else:
                data = col_data[:]
                print(f"    Col {col} ({column_names[col]}): shape {col_data.shape}, values {data.flatten()[:3]}...")
        
        # Dataset[0,1] - 46 embryos' morphology data (membrane, only 17 have data)
        embryo_morph = f[dataset[0, 1]]
        print(f"\n[Dataset 0,1] Cell Morphology: {embryo_morph.shape[0]} embryos")
        
        # Check which embryos have morphology data
        has_morph = 0
        for i in range(embryo_morph.shape[0]):
            emb_ref = embryo_morph[i, 0]
            emb_data = f[emb_ref]
            if len(emb_data.shape) >= 2 and emb_data.shape[0] > 0 and emb_data.shape[1] > 0:
                has_morph += 1
        print(f"  Embryos with morphology data: {has_morph}")
        
        # Analyze morphology structure
        for i in range(embryo_morph.shape[0]):
            emb_ref = embryo_morph[i, 0]
            emb_data = f[emb_ref]
            if len(emb_data.shape) >= 2 and emb_data.shape[0] > 0 and emb_data.shape[1] > 0:
                print(f"\n  Embryo {i} morphology: {emb_data.shape[0]} rows x {emb_data.shape[1]} cols")
                
                # Check first row structure
                row0_ref = emb_data[0, 0]
                row0 = f[row0_ref]
                print(f"    Row 0 shape: {row0.shape}")
                
                # Morphology columns: CellName, SurfaceMesh, Volume, etc.
                for col in range(min(4, emb_data.shape[1])):
                    cell_ref = emb_data[0, col]
                    cell_data = f[cell_ref]
                    if cell_data.dtype == 'object' and cell_data.shape[0] > 0:
                        inner_ref = cell_data[0, 0] if len(cell_data.shape) > 1 else cell_data[0]
                        inner = f[inner_ref]
                        if inner.dtype.kind == 'u' and np.prod(inner.shape) < 100:
                            s = read_matlab_string(f, inner_ref)
                            print(f"    Col {col}: e.g. '{s}'")
                        else:
                            print(f"    Col {col}: inner shape {inner.shape}, dtype {inner.dtype}")
                    else:
                        print(f"    Col {col}: shape {cell_data.shape}, dtype {cell_data.dtype}")
                break  # Only show first embryo with data
    
    # Analyze CellName mapping
    print("\n" + "-" * 70)
    print("CELL NAME MAPPING")
    print("-" * 70)
    
    cellname_file = sd1_path / "WorkSpace_CellName.mat"
    with h5py.File(cellname_file, 'r') as f:
        cellname = f['CellName']
        print(f"\nCellName shape: {cellname.shape}")
        
        # Each entry is a different naming scheme
        for i in range(cellname.shape[1]):
            ref = cellname[0, i]
            data = f[ref]
            if data.dtype == 'object':
                n_names = data.shape[0]
                # Get first name as sample
                name_ref = data[0, 0] if len(data.shape) > 1 else data[0]
                sample = read_matlab_string(f, name_ref)
                print(f"  Scheme {i}: {n_names} names, e.g. '{sample}'")
            else:
                print(f"  Scheme {i}: shape {data.shape}, dtype {data.dtype}")


def analyze_standard_dataset_2():
    """Analyze Standard Dataset 2 structure (3D segmentation matrices)."""
    print("\n" + "=" * 70)
    print("STANDARD DATASET 2 ANALYSIS (3D Segmentation)")
    print("=" * 70)
    
    sd2_path = get_cshaper_path() / "Standard Dataset 2"
    
    # Get list of files
    mat_files = sorted(sd2_path.glob("Seg_*.mat"))
    print(f"\nTotal segmentation files: {len(mat_files)}")
    
    # Parse filenames to understand structure
    frames = set()
    embryos = set()
    for f in mat_files:
        parts = f.stem.split('_')
        if len(parts) >= 3:
            frames.add(int(parts[1]))
            embryos.add(parts[2])
    
    print(f"Frames: {min(frames)} to {max(frames)} ({len(frames)} total)")
    print(f"Embryos: {sorted(embryos)}")
    
    # Analyze one file
    sample_file = mat_files[0]
    print(f"\nSample file: {sample_file.name}")
    
    with h5py.File(sample_file, 'r') as f:
        print(f"Top-level keys: {list(f.keys())}")
        
        for key in f.keys():
            if key == '#refs#':
                continue
            obj = f[key]
            print(f"\n'{key}':")
            print(f"  Type: {type(obj).__name__}")
            if hasattr(obj, 'shape'):
                print(f"  Shape: {obj.shape}")
            if hasattr(obj, 'dtype'):
                print(f"  Dtype: {obj.dtype}")
                
            # If it's a 3D matrix, show some stats
            if isinstance(obj, h5py.Dataset) and len(obj.shape) == 3:
                data = obj[:]
                unique_labels = np.unique(data)
                print(f"  Unique cell labels: {len(unique_labels)}")
                print(f"  Label range: {unique_labels.min()} to {unique_labels.max()}")
                print(f"  Background (0) voxels: {np.sum(data == 0)}")
                print(f"  Cell voxels: {np.sum(data > 0)}")
                
                # Physical dimensions
                voxel_size = 0.25  # μm
                physical_dim = np.array(obj.shape) * voxel_size
                print(f"  Physical dimensions: {physical_dim[0]:.1f} x {physical_dim[1]:.1f} x {physical_dim[2]:.1f} μm")


def extract_cell_positions(frame_idx=1, embryo_idx=0):
    """Extract cell positions from Standard Dataset 1."""
    print("\n" + "=" * 70)
    print(f"EXTRACTING CELL POSITIONS (Frame {frame_idx}, Embryo {embryo_idx})")
    print("=" * 70)
    
    sd1_path = get_cshaper_path() / "Standard Dataset 1"
    mat_file = sd1_path / f"WorkSpace_Dataset_{frame_idx}.mat"
    
    if not mat_file.exists():
        print(f"File not found: {mat_file}")
        return None
    
    cells = []
    
    with h5py.File(mat_file, 'r') as f:
        dataset = f['Dataset']
        embryo_positions = f[dataset[0, 0]]
        
        if embryo_idx >= embryo_positions.shape[0]:
            print(f"Embryo index {embryo_idx} out of range (max: {embryo_positions.shape[0]-1})")
            return None
        
        emb_ref = embryo_positions[embryo_idx, 0]
        emb_data = f[emb_ref]
        
        n_cols = emb_data.shape[1]
        print(f"Found {n_cols} columns of data")
        
        # Column 0 should be cell names
        names_ref = emb_data[0, 0]
        names_data = f[names_ref]
        n_cells = names_data.shape[0]
        print(f"Found {n_cells} cells")
        
        # Extract cell names
        cell_names = []
        for i in range(n_cells):
            name_ref = names_data[i, 0]
            name = read_matlab_string(f, name_ref)
            cell_names.append(name)
        
        # Try to extract X, Y, Z coordinates (columns 1, 2, 3)
        coords = []
        for col in [1, 2, 3]:
            col_ref = emb_data[0, col]
            col_data = f[col_ref][:]
            coords.append(col_data.flatten())
        
        print(f"\nFirst 10 cells:")
        print(f"{'Cell Name':<20} {'X':>10} {'Y':>10} {'Z':>10}")
        print("-" * 52)
        for i in range(min(10, n_cells)):
            print(f"{cell_names[i]:<20} {coords[0][i]:>10.2f} {coords[1][i]:>10.2f} {coords[2][i]:>10.2f}")
        
        return {
            'names': cell_names,
            'x': coords[0],
            'y': coords[1],
            'z': coords[2]
        }


def main():
    print("CShaper .mat File Analysis")
    print("Using h5py to read MATLAB v7.3 (HDF5) format\n")
    
    # Check if data exists
    cshaper_path = get_cshaper_path()
    if not cshaper_path.exists():
        print(f"CShaper data not found at {cshaper_path}")
        return
    
    # Analyze Standard Dataset 1
    analyze_standard_dataset_1()
    
    # Analyze Standard Dataset 2
    analyze_standard_dataset_2()
    
    # Print data structure explanation
    print_data_structure_explanation()


def print_data_structure_explanation():
    """Print explanation of the data structure."""
    print("\n" + "=" * 70)
    print("DATA STRUCTURE EXPLANATION")
    print("=" * 70)
    print("""
Standard Dataset 1 is organized by LINEAGE TREE structure:

CellName/Coordinates matrices are organized as (generations, positions):
  - Row i: Cell generation (0=founder, 1=first division, ...)
  - Col j: Cell index within generation (binary tree indexing)

Lineage structure:
  [0] AB lineage  (8 gen x 256 pos) -> ABa, ABal, ABala, ABalaa...
  [1] MS lineage  (6 gen x 32 pos)  -> MS, MSa, MSaa, MSaaa...
  [2] E lineage   (5 gen x 16 pos)  -> E, Ea, Eal, Eala...
  [3] C lineage   (5 gen x 16 pos)  -> C, Ca, Caa, Caaa...
  [4] D lineage   (4 gen x 8 pos)   -> D, Da, Daa, Daaa...
  [5] P3          (1 x 1)
  [6] P4/Z        (2 x 2)           -> P4, Z2, Z3
  [7] EMS         (1 x 1)
  [8] P2          (1 x 1)

Binary tree indexing:
  Gen 0: [0]              -> founder cell
  Gen 1: [0, 1]           -> anterior (a), posterior (p)
  Gen 2: [0, 1, 2, 3]     -> aa, ap, pa, pp
  Gen 3: [0..7]           -> aaa, aap, apa, app, paa, pap, ppa, ppp
  
Example: ABalaap is at lineage[0], gen=5, idx=5 (binary: 00101)
         a=0, l=0, a=1, a=0, p=1 -> idx = 0*16 + 0*8 + 1*4 + 0*2 + 1*1 = 5

This directly corresponds to NemaContext's Poincare embedding scheme!
""")
    
    print("\n" + "=" * 70)
    print("SUMMARY: How to use this data for NemaContext")
    print("=" * 70)
    print("""
1. CELL POSITIONS (Standard Dataset 1):
   - 46 embryos with nucleus positions across 54 time frames
   - Organized by lineage tree (perfect for binary path encoding)
   - Coordinates are normalized to a common reference frame
   - Use for: Spatial GNN node positions, Poincare embedding validation

2. CELL MORPHOLOGY (Standard Dataset 1, 17 embryos only):
   - Surface mesh data for cells with membrane segmentation
   - Shape: (3, N) where N is number of surface points
   - Use for: Cell shape features, contact surface computation

3. 3D SEGMENTATION (Standard Dataset 2):
   - Full 3D voxel matrices (184 x 114 x 256 at 0.25um resolution)
   - Each voxel labeled with cell ID (sparse labels, most are 0)
   - Use for: Ground truth for spatial relationships, volume computation

4. CELL NAME MAPPING (WorkSpace_CellName.mat):
   - 9 lineage trees with all cell names
   - Maps between lineage names and tree positions
   - Use for: Converting lineage names to binary paths
""")


if __name__ == "__main__":
    main()
