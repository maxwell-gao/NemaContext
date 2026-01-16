"""
Analyze Standard Dataset 2 - 3D Cell Segmentation Matrices

This dataset contains 3D voxel matrices where each voxel is labeled
with a cell ID. The data covers 54 time frames for 17 wild-type embryos.
"""

import h5py
import numpy as np
from pathlib import Path
from collections import defaultdict


def get_sd2_path() -> Path:
    """Get path to Standard Dataset 2."""
    return Path(__file__).parent.parent / "dataset" / "raw" / "cshaper" / "Standard Dataset 2"


def analyze_single_file(filepath: Path):
    """Analyze a single segmentation file."""
    with h5py.File(filepath, 'r') as f:
        seg = f['Seg'][:]
        
        # Get unique labels (cell IDs)
        unique_labels = np.unique(seg)
        non_zero = unique_labels[unique_labels != 0]
        
        result = {
            'shape': seg.shape,
            'dtype': str(seg.dtype),
            'cell_count': len(non_zero),
            'cell_labels': non_zero.tolist(),
            'total_voxels': seg.size,
            'cell_voxels': int(np.sum(seg > 0)),
            'background_voxels': int(np.sum(seg == 0)),
        }
        
        # Calculate per-cell statistics
        cell_stats = []
        for label in non_zero:
            coords = np.where(seg == label)
            centroid = tuple(np.mean(c) for c in coords)
            voxel_count = len(coords[0])
            
            # Bounding box
            bbox_min = tuple(np.min(c) for c in coords)
            bbox_max = tuple(np.max(c) for c in coords)
            
            cell_stats.append({
                'label': int(label),
                'voxels': voxel_count,
                'centroid': centroid,
                'bbox_min': bbox_min,
                'bbox_max': bbox_max,
            })
        
        result['cells'] = cell_stats
        return result


def main():
    sd2_path = get_sd2_path()
    
    print("=" * 70)
    print("STANDARD DATASET 2 - 3D Cell Segmentation Analysis")
    print("=" * 70)
    
    # Dataset overview
    print("\nFile naming convention: Seg_{frame}_{embryo}.mat")
    print("  - frame: 1-54 (developmental time points)")
    print("  - embryo: 04-20 (17 wild-type embryos with membrane data)")
    
    # Count files
    all_files = sorted(sd2_path.glob("Seg_*.mat"))
    frames = set()
    embryos = set()
    for f in all_files:
        parts = f.stem.split('_')
        frames.add(int(parts[1]))
        embryos.add(parts[2])
    
    print(f"\nDataset size:")
    print(f"  Total files: {len(all_files)}")
    print(f"  Frames: {min(frames)} to {max(frames)} ({len(frames)} frames)")
    print(f"  Embryos: {sorted(embryos)}")
    print(f"  Expected: {len(frames)} × {len(embryos)} = {len(frames) * len(embryos)} files")
    
    # Analyze sample files at different time points
    print("\n" + "=" * 70)
    print("Sample Analysis: Embryo 04 at different time points")
    print("=" * 70)
    
    sample_frames = [1, 10, 20, 30, 40, 50, 54]
    
    for frame in sample_frames:
        fpath = sd2_path / f"Seg_{frame}_04.mat"
        if not fpath.exists():
            print(f"\nFrame {frame}: NOT FOUND")
            continue
        
        result = analyze_single_file(fpath)
        
        print(f"\nFrame {frame}:")
        print(f"  Matrix shape: {result['shape']}")
        print(f"  Cell count: {result['cell_count']}")
        print(f"  Cell voxels: {result['cell_voxels']:,} / {result['total_voxels']:,} "
              f"({100*result['cell_voxels']/result['total_voxels']:.1f}%)")
        
        # Show first few cells
        if result['cells']:
            print(f"  Sample cells:")
            for cell in result['cells'][:3]:
                cx, cy, cz = cell['centroid']
                print(f"    Label {cell['label']}: {cell['voxels']:,} voxels, "
                      f"centroid=({cx:.1f}, {cy:.1f}, {cz:.1f})")
    
    # Physical dimensions
    print("\n" + "=" * 70)
    print("Physical Dimensions")
    print("=" * 70)
    
    voxel_size = 0.25  # μm per voxel
    sample_result = analyze_single_file(sd2_path / "Seg_1_04.mat")
    shape = sample_result['shape']
    physical_dim = tuple(s * voxel_size for s in shape)
    
    print(f"\nVoxel size: {voxel_size} μm")
    print(f"Matrix shape: {shape}")
    print(f"Physical size: {physical_dim[0]:.1f} × {physical_dim[1]:.1f} × {physical_dim[2]:.1f} μm")
    
    # Cell growth over time
    print("\n" + "=" * 70)
    print("Cell Count Over Time (Embryo 04)")
    print("=" * 70)
    
    print(f"\n{'Frame':<8} {'Cells':<8} {'Cell Voxels':<15} {'Occupancy %':<12}")
    print("-" * 45)
    
    for frame in range(1, 55, 5):
        fpath = sd2_path / f"Seg_{frame}_04.mat"
        if fpath.exists():
            result = analyze_single_file(fpath)
            occupancy = 100 * result['cell_voxels'] / result['total_voxels']
            print(f"{frame:<8} {result['cell_count']:<8} {result['cell_voxels']:<15,} {occupancy:<12.1f}")
    
    # Label encoding explanation
    print("\n" + "=" * 70)
    print("Cell Label Encoding")
    print("=" * 70)
    print("""
Cell labels are integers that encode the cell identity.
The label value corresponds to a cell ID that maps to cell names
in Standard Dataset 1 (WorkSpace_CellName.mat).

Example labels from Frame 30, Embryo 04:
""")
    
    fpath = sd2_path / "Seg_30_04.mat"
    if fpath.exists():
        result = analyze_single_file(fpath)
        for cell in result['cells'][:10]:
            print(f"  Label {cell['label']:>5}: {cell['voxels']:>8,} voxels")
    
    # Usage for NemaContext
    print("\n" + "=" * 70)
    print("Usage for NemaContext")
    print("=" * 70)
    print("""
Standard Dataset 2 provides:

1. GROUND TRUTH SPATIAL POSITIONS
   - Cell centroids can be computed from voxel coordinates
   - Useful for validating spatial GNN predictions

2. CELL-CELL CONTACT DETECTION
   - Neighboring voxels with different labels indicate contact
   - Can build adjacency matrix for spatial graph

3. CELL VOLUME COMPUTATION
   - Volume = voxel_count × (0.25 μm)³
   - Useful as cell feature in token representation

4. CELL SHAPE ANALYSIS
   - 3D morphology available for shape descriptors
   - Surface area, sphericity, etc.

Example code to extract adjacency matrix:
```python
import h5py
import numpy as np
from scipy import ndimage

with h5py.File('Seg_30_04.mat', 'r') as f:
    seg = f['Seg'][:]

# Find adjacent cell pairs
labels = np.unique(seg)
labels = labels[labels > 0]

adjacency = set()
for label in labels:
    mask = seg == label
    dilated = ndimage.binary_dilation(mask)
    neighbors = np.unique(seg[dilated & ~mask])
    for n in neighbors:
        if n > 0 and n != label:
            adjacency.add((min(label, n), max(label, n)))
```
""")


if __name__ == "__main__":
    main()
