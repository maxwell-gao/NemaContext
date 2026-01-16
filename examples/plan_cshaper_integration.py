"""
Analyze current processed datasets and plan CShaper integration.
"""

import anndata as ad
import numpy as np
import pandas as pd
from pathlib import Path


def analyze_current_datasets():
    """Analyze the current processed h5ad files."""
    print("=" * 70)
    print("CURRENT PROCESSED DATASETS ANALYSIS")
    print("=" * 70)
    
    # Load datasets
    complete = ad.read_h5ad('dataset/processed/nema_complete_large2025.h5ad')
    extended = ad.read_h5ad('dataset/processed/nema_extended_large2025.h5ad')
    
    print("\n1. nema_complete_large2025.h5ad (trimodal complete)")
    print(f"   Cells: {complete.n_obs:,}")
    print(f"   Genes: {complete.n_vars:,}")
    print(f"   obs columns: {list(complete.obs.columns)}")
    print(f"   obsm keys: {list(complete.obsm.keys())}")
    print(f"   uns keys: {list(complete.uns.keys())}")
    
    print("\n2. nema_extended_large2025.h5ad (all cells)")
    print(f"   Cells: {extended.n_obs:,}")
    print(f"   Genes: {extended.n_vars:,}")
    
    # Spatial coverage
    print("\n" + "=" * 70)
    print("SPATIAL DATA COVERAGE (from WormGUIDES)")
    print("=" * 70)
    
    if 'spatial_matched' in complete.obs.columns:
        print(f"\nComplete: {complete.obs['spatial_matched'].sum():,}/{complete.n_obs:,} have spatial coords")
    if 'spatial_matched' in extended.obs.columns:
        print(f"Extended: {extended.obs['spatial_matched'].sum():,}/{extended.n_obs:,} have spatial coords")
    
    # Spatial coordinates info
    if 'spatial' in complete.obsm:
        coords = complete.obsm['spatial']
        print(f"\nSpatial coords shape: {coords.shape}")
        print(f"X range: {coords[:,0].min():.2f} - {coords[:,0].max():.2f}")
        print(f"Y range: {coords[:,1].min():.2f} - {coords[:,1].max():.2f}")
        if coords.shape[1] > 2:
            print(f"Z range: {coords[:,2].min():.2f} - {coords[:,2].max():.2f}")
    
    # Lineage encoding info
    print("\n" + "=" * 70)
    print("LINEAGE ENCODING")
    print("=" * 70)
    
    if 'lineage_binary' in complete.obsm:
        print(f"\nLineage binary shape: {complete.obsm['lineage_binary'].shape}")
    if 'lineage_depth' in complete.obs.columns:
        print(f"Lineage depth range: {complete.obs['lineage_depth'].min()} - {complete.obs['lineage_depth'].max()}")
    if 'lineage_founder' in complete.obs.columns:
        print(f"Founders: {complete.obs['lineage_founder'].unique().tolist()}")
    
    # Check what's MISSING
    print("\n" + "=" * 70)
    print("CURRENT LIMITATIONS (What's MISSING)")
    print("=" * 70)
    
    print("""
1. SPATIAL DATA SOURCE: WormGUIDES nuclei positions
   - Only nucleus center positions (point coordinates)
   - No cell VOLUME information
   - No cell SURFACE AREA
   - No cell-cell CONTACT information
   - No 3D MORPHOLOGY (shape)

2. SPATIAL MATCHING RATE:""")
    
    if 'spatial_matched' in extended.obs.columns:
        match_rate = extended.obs['spatial_matched'].sum() / extended.n_obs * 100
        print(f"   Only {match_rate:.1f}% of cells matched to spatial coordinates")
    
    print("""
3. NEIGHBOR GRAPH:
   - Currently built from: lineage tree structure OR k-NN in expression space
   - MISSING: True spatial neighbor graph based on physical contact
""")
    
    return complete, extended


def plan_cshaper_integration():
    """Plan how CShaper can enhance the dataset."""
    print("\n" + "=" * 70)
    print("CSHAPER INTEGRATION PLAN")
    print("=" * 70)
    
    print("""
CShaper provides COMPLEMENTARY morphological data that can significantly
enhance our trimodal dataset:

┌─────────────────────────────────────────────────────────────────────┐
│                    CURRENT DATA FLOW                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Large2025          WormGUIDES           WormBase                   │
│  (scRNA-seq)        (nuclei positions)   (lineage tree)            │
│       │                   │                   │                     │
│       ▼                   ▼                   ▼                     │
│  ┌─────────┐        ┌─────────┐         ┌─────────┐                 │
│  │Expression│        │  X,Y,Z  │         │Binary   │                │
│  │ Matrix   │        │  point  │         │ Path    │                │
│  └─────────┘        └─────────┘         └─────────┘                 │
│       │                   │                   │                     │
│       └───────────────────┴───────────────────┘                     │
│                           │                                         │
│                           ▼                                         │
│                    AnnData (trimodal)                               │
│                    - X: expression                                  │
│                    - obsm['spatial']: point coords                  │
│                    - obsm['lineage_binary']: tree path              │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                 WITH CSHAPER ENHANCEMENT                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Large2025          WormGUIDES           WormBase      CShaper      │
│  (scRNA-seq)        (nuclei)            (lineage)     (morphology) │
│       │                   │                   │            │        │
│       │                   │                   │            │        │
│       │                   │                   │    ┌───────┴───────┐│
│       │                   │                   │    │ ContactInterface│
│       │                   │                   │    │ (adjacency)    ││
│       │                   │                   │    ├───────────────┤│
│       │                   │                   │    │ Volume/Surface ││
│       │                   │                   │    │ (cell features)││
│       │                   │                   │    ├───────────────┤│
│       │                   │                   │    │ Standard DS 1  ││
│       │                   │                   │    │ (normalized XYZ)│
│       │                   │                   │    ├───────────────┤│
│       │                   │                   │    │ Standard DS 2  ││
│       │                   │                   │    │ (3D voxels)    ││
│       │                   │                   │    └───────┬───────┘│
│       │                   │                   │            │        │
│       ▼                   ▼                   ▼            ▼        │
│                                                                     │
│                    ENHANCED AnnData                                 │
│    ┌────────────────────────────────────────────────────────┐       │
│    │ X: expression matrix (unchanged)                       │       │
│    │                                                        │       │
│    │ obsm['spatial']: 3D coordinates (from CShaper SD1)    │ NEW   │
│    │ obsm['lineage_binary']: binary path (unchanged)        │       │
│    │                                                        │       │
│    │ obs['cell_volume']: from CShaper VolumeAndSurface     │ NEW   │
│    │ obs['cell_surface']: from CShaper VolumeAndSurface    │ NEW   │
│    │ obs['sphericity']: volume/surface ratio               │ NEW   │
│    │                                                        │       │
│    │ obsp['contact_adjacency']: CShaper ContactInterface   │ NEW   │
│    │ obsp['spatial_neighbors']: from 3D proximity          │ NEW   │
│    │                                                        │       │
│    │ uns['cshaper_metadata']: embryo info, time mapping    │ NEW   │
│    └────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
""")
    
    print("\n" + "=" * 70)
    print("SPECIFIC ENHANCEMENTS")
    print("=" * 70)
    
    print("""
1. CELL-CELL CONTACT GRAPH (obsp['contact_adjacency'])
   Source: CShaper ContactInterface/*.csv
   - Symmetric sparse matrix of contact surface areas
   - TRUE physical neighbors (not k-NN approximation)
   - Critical for: Spatial GNN message passing
   
2. CELL MORPHOLOGY FEATURES (obs columns)
   Source: CShaper VolumeAndSurface/*.csv
   - cell_volume: Cell volume in μm³
   - cell_surface: Surface area in μm²
   - sphericity: 4π·V^(2/3) / S (shape descriptor)
   - Can add: elongation, flatness from SD2 voxels
   
3. IMPROVED SPATIAL COORDINATES (obsm['spatial'])
   Source: CShaper Standard Dataset 1
   - Normalized coordinates (consistent reference frame)
   - 46 embryos → more robust averaging
   - Currently using WormGUIDES (different coordinate system)
   
4. 3D SEGMENTATION FOR VALIDATION (external)
   Source: CShaper Standard Dataset 2
   - Full 3D voxel-level cell shapes
   - Can validate contact detection algorithms
   - Ground truth for spatial relationship learning

5. TIME-RESOLVED SPATIAL DYNAMICS
   - CShaper has 54 time frames
   - Can build DYNAMIC contact graphs over development
   - Enable spatiotemporal modeling
""")
    
    print("\n" + "=" * 70)
    print("IMPLEMENTATION PRIORITY")
    print("=" * 70)
    
    print("""
Priority 1: Contact Adjacency Matrix
  - Most impactful for Spatial GNN
  - Already in easy CSV format
  - Direct integration into obsp
  
Priority 2: Volume/Surface Features  
  - Easy to add as obs columns
  - Useful cell-level features
  - Can improve cell type prediction
  
Priority 3: Improved Spatial Matching
  - Replace/supplement WormGUIDES coords
  - Requires lineage name alignment
  - More complex integration

Priority 4: 3D Morphology Features
  - Compute from SD2 voxels
  - Shape descriptors, PCA of shapes
  - Computationally intensive
""")


def show_matching_strategy():
    """Show how to match CShaper to Large2025 cells."""
    print("\n" + "=" * 70)
    print("LINEAGE NAME MATCHING STRATEGY")
    print("=" * 70)
    
    print("""
CShaper and Large2025 both use standard C. elegans lineage naming:

CShaper cell names (from VolumeAndSurface):
  ABa, ABp, EMS, P2, ABal, ABar, ABpl, ABpr, MS, E, C, P3, ...

Large2025 lineage column:
  ABa, ABal, ABala, ABalaa, ... (same naming convention!)

Matching approach:
1. Parse lineage names from both sources
2. Match by exact lineage string
3. For cells at different times:
   - Use embryo_time from Large2025
   - Map to nearest CShaper frame (1-54)
   - Interpolate spatial features if needed

Code sketch:
```python
def match_cshaper_to_anndata(adata, cshaper_data):
    # Get lineage names from AnnData
    lineages = adata.obs['lineage_complete'].values
    embryo_times = adata.obs['embryo_time'].values
    
    # For each cell, find matching CShaper data
    for i, (lin, time) in enumerate(zip(lineages, embryo_times)):
        frame = time_to_cshaper_frame(time)
        if lin in cshaper_data[frame]:
            adata.obs.loc[i, 'cell_volume'] = cshaper_data[frame][lin]['volume']
            adata.obs.loc[i, 'cell_surface'] = cshaper_data[frame][lin]['surface']
```
""")


if __name__ == "__main__":
    # Analyze current state
    complete, extended = analyze_current_datasets()
    
    # Plan CShaper integration
    plan_cshaper_integration()
    
    # Show matching strategy
    show_matching_strategy()
