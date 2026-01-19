#!/usr/bin/env python3
"""
Comprehensive test script for CShaper integration.

Tests:
1. StandardSpatialLoader - coordinate extraction from Standard Dataset 1
2. Segmentation3DLoader - 3D shape descriptors from Standard Dataset 2
3. Time-dynamic contact graphs
4. Contact Graph GNN layers

Usage:
    uv run python examples/test_cshaper_complete.py
"""

import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def test_standard_spatial_loader():
    """Test StandardSpatialLoader for coordinate extraction."""
    print("\n" + "="*60)
    print("TEST 1: StandardSpatialLoader (Standard Dataset 1)")
    print("="*60)
    
    from src.data.builder import CShaperProcessor
    
    processor = CShaperProcessor("dataset/raw")
    
    if not processor.has_standard_spatial:
        print("⚠ Standard Dataset 1 not available, skipping test")
        return False
    
    loader = processor.spatial_loader
    
    # Test 1: Get available frames
    frames = loader.get_available_frames()
    print(f"✓ Available frames: {len(frames)} (range: {min(frames)}-{max(frames)})")
    
    # Test 2: Load cell names
    cell_names = loader.get_all_cell_names()
    print(f"✓ Cell names loaded: {len(cell_names)}")
    print(f"  Sample names: {cell_names[:5]}")
    
    # Test 3: Get coordinates for cells that exist at frame 27
    # Note: Early cells (ABa, ABp) have divided by frame 27, so we test with later-stage cells
    test_cells = cell_names[:10] if len(cell_names) >= 10 else cell_names
    coords = loader.get_spatial_coords(test_cells, time_frame=27)
    
    n_found = (~np.isnan(coords[:, 0])).sum()
    print(f"✓ Coordinates extracted: {n_found}/{len(test_cells)} cells")
    
    for i, cell in enumerate(test_cells[:5]):
        if not np.isnan(coords[i, 0]):
            print(f"  {cell}: [{coords[i, 0]:.2f}, {coords[i, 1]:.2f}, {coords[i, 2]:.2f}]")
    
    # Test 4: Tree index conversion
    test_lineages = ["ABplpappa", "MSapaap", "Ealad"]
    for lineage in test_lineages:
        idx = loader.lineage_to_tree_index(lineage)
        if idx:
            print(f"  {lineage} -> founder_idx={idx[0]}, gen={idx[1]}, pos={idx[2]}")
    
    # Test 5: Summary
    summary = loader.summary()
    print(f"✓ Summary: {summary}")
    
    return True


def test_segmentation_3d_loader():
    """Test Segmentation3DLoader for 3D shape descriptors."""
    print("\n" + "="*60)
    print("TEST 2: Segmentation3DLoader (Standard Dataset 2)")
    print("="*60)
    
    from src.data.builder import CShaperProcessor
    
    processor = CShaperProcessor("dataset/raw")
    
    if not processor.has_segmentation:
        print("⚠ Standard Dataset 2 not available, skipping test")
        return False
    
    loader = processor.segmentation_loader
    
    # Test 1: Get available files
    available = loader.get_available_files()
    print(f"✓ Available segmentation files: {len(available)}")
    if available:
        print(f"  Sample: {available[:3]}")
    
    # Test 2: Load a segmentation volume
    if available:
        time_idx, sample_idx = available[0]
        seg = loader.load_segmentation(time_idx, sample_idx)
        
        if seg is not None:
            print(f"✓ Loaded segmentation: shape={seg.shape}, dtype={seg.dtype}")
            print(f"  Value range: [{seg.min()}, {seg.max()}]")
            
            # Test 3: Get cell labels
            labels = loader.get_cell_labels(seg)
            print(f"✓ Cell labels found: {len(labels)}")
            
            # Test 4: Compute shape descriptors for one cell
            if len(labels) > 0:
                test_label = int(labels[0])
                descriptors = loader.compute_shape_descriptors(seg, test_label)
                print(f"✓ Shape descriptors for label {test_label}:")
                for key, val in descriptors.items():
                    if isinstance(val, float):
                        print(f"    {key}: {val:.4f}")
                    else:
                        print(f"    {key}: {val}")
            
            # Test 5: Compute all shape descriptors
            all_desc = loader.compute_all_shape_descriptors(time_idx, sample_idx)
            print(f"✓ All shape descriptors: {len(all_desc)} cells")
            if len(all_desc) > 0:
                print(f"  Columns: {list(all_desc.columns)}")
    
    # Test 6: Summary
    summary = loader.summary()
    print(f"✓ Summary: {summary}")
    
    return True


def test_time_dynamic_contact_graph():
    """Test time-dynamic contact graph functionality."""
    print("\n" + "="*60)
    print("TEST 3: Time-Dynamic Contact Graphs")
    print("="*60)
    
    from src.data.builder import CShaperProcessor
    
    processor = CShaperProcessor("dataset/raw")
    
    if not processor.has_contact:
        print("⚠ ContactInterface not available, skipping test")
        return False
    
    # Get sample cells
    all_cells = list(processor.get_all_cell_names())[:50]  # Limit for speed
    print(f"Testing with {len(all_cells)} cells")
    
    # Test 1: Get per-frame contact graphs
    sample_id = processor.contact_loader.get_available_samples()[0]
    frames = processor.contact_loader.get_available_frames(sample_id)
    print(f"✓ Sample {sample_id} has {len(frames)} frames")
    
    # Test 2: Load contact matrix for specific frame
    for frame in frames[:3]:  # First 3 frames
        df = processor.contact_loader.load_sample(sample_id, frame=frame)
        n_edges = (df.values > 0).sum() // 2  # Divide by 2 for symmetric
        print(f"  Frame {frame}: {len(df)} cells, {n_edges} contacts")
    
    # Test 3: Get contact timeseries between two cells
    if len(all_cells) >= 2:
        cell1, cell2 = all_cells[0], all_cells[1]
        times, areas = processor.get_contact_timeseries(cell1, cell2, sample_id)
        nonzero = (areas > 0).sum()
        print(f"✓ Contact timeseries {cell1}-{cell2}: {nonzero}/{len(times)} frames with contact")
    
    # Test 4: Build time-dynamic graph
    from src.model import TimeDynamicContactGraph, ContactGraph
    
    # Build graphs for first few frames
    graphs = {}
    for frame in frames[:5]:
        df = processor.contact_loader.load_sample(sample_id, frame=frame)
        # Build adjacency for cells present in data
        common_cells = [c for c in all_cells if c in df.index]
        if common_cells:
            from scipy.sparse import csr_matrix, lil_matrix
            n = len(common_cells)
            cell_to_idx = {c: i for i, c in enumerate(common_cells)}
            adj = lil_matrix((n, n), dtype=np.float32)
            
            for c1 in common_cells:
                for c2 in common_cells:
                    if c1 in df.index and c2 in df.columns:
                        val = df.loc[c1, c2]
                        if val > 0:
                            adj[cell_to_idx[c1], cell_to_idx[c2]] = val
            
            graph = ContactGraph.from_csr_matrix(
                adj.tocsr(),
                cell_names=common_cells,
                time_frame=frame,
            )
            graphs[frame] = graph
    
    if graphs:
        time_graph = TimeDynamicContactGraph.from_frame_graphs(graphs)
        print(f"✓ Created TimeDynamicContactGraph with {len(time_graph.time_frames)} frames")
        
        # Get temporal edges
        edge_idx, edge_weight, edge_time = time_graph.get_temporal_edges()
        print(f"  Total temporal edges: {edge_idx.shape[1]}")
    
    return True


def test_contact_gnn():
    """Test Contact Graph GNN layers."""
    print("\n" + "="*60)
    print("TEST 4: Contact Graph GNN")
    print("="*60)
    
    from src.data.builder import CShaperProcessor
    from src.model import (
        ContactGraph,
        ContactMessagePassing,
        ContactGNN,
        build_contact_graph_from_cshaper,
        compare_contact_vs_knn,
    )
    
    processor = CShaperProcessor("dataset/raw")
    
    if not processor.has_contact:
        print("⚠ ContactInterface not available, skipping test")
        return False
    
    # Get sample cells
    all_cells = list(processor.get_all_cell_names())[:100]
    print(f"Testing with {len(all_cells)} cells")
    
    # Test 1: Build ContactGraph from CShaperProcessor
    graph = build_contact_graph_from_cshaper(
        processor,
        all_cells,
        include_morphology=processor.has_morphology,
    )
    print(f"✓ Built ContactGraph: {graph.n_cells} cells, {graph.n_edges} edges")
    
    # Test 2: Check graph properties
    degree = graph.get_degree()
    print(f"  Degree stats: mean={degree.mean():.2f}, max={degree.max()}, min={degree.min()}")
    
    # Test 3: Test message passing layer
    in_features = 8
    out_features = 16
    
    # Create random node features
    x = np.random.randn(graph.n_cells, in_features).astype(np.float32)
    
    layer = ContactMessagePassing(
        in_features=in_features,
        out_features=out_features,
        aggregation="mean",
    )
    
    h = layer.forward(x, graph)
    print(f"✓ Message passing: input {x.shape} -> output {h.shape}")
    
    # Test 4: Test full GNN
    gnn = ContactGNN(
        in_features=in_features,
        hidden_features=32,
        out_features=16,
        n_layers=2,
    )
    
    embeddings = gnn.forward(x, graph)
    print(f"✓ Full GNN: input {x.shape} -> output {embeddings.shape}")
    
    # Test 5: Get cell embeddings with morphology
    if graph.node_features is not None:
        gnn2 = ContactGNN(
            in_features=graph.node_features.shape[1],
            hidden_features=32,
            out_features=16,
            n_layers=2,
        )
        cell_embeddings = gnn2.get_cell_embeddings(graph)
        print(f"✓ Cell embeddings with morphology: {cell_embeddings.shape}")
    
    # Test 6: Compare with k-NN graph
    # Create a fake k-NN graph for comparison
    from sklearn.neighbors import NearestNeighbors
    
    # Use degree as proxy for "position" for k-NN
    fake_positions = np.random.randn(graph.n_cells, 3)
    nn = NearestNeighbors(n_neighbors=min(10, graph.n_cells - 1))
    nn.fit(fake_positions)
    knn_graph = nn.kneighbors_graph(mode='connectivity')
    knn_coo = knn_graph.tocoo()
    knn_edge_index = np.vstack([knn_coo.row, knn_coo.col])
    
    comparison = compare_contact_vs_knn(graph, knn_edge_index)
    print(f"✓ Contact vs k-NN comparison:")
    print(f"  Contact edges: {comparison['n_contact_edges']}")
    print(f"  k-NN edges: {comparison['n_knn_edges']}")
    print(f"  Jaccard: {comparison['jaccard']:.3f}")
    print(f"  F1: {comparison['f1']:.3f}")
    
    return True


def test_enhanced_builder_integration():
    """Test full integration with EnhancedAnnDataBuilder."""
    print("\n" + "="*60)
    print("TEST 5: EnhancedAnnDataBuilder Integration")
    print("="*60)
    
    from src.data.builder import EnhancedAnnDataBuilder, CShaperProcessor
    
    processor = CShaperProcessor("dataset/raw")
    print(processor.summary())
    
    # Don't actually build (requires all raw data), just verify imports work
    print("✓ EnhancedAnnDataBuilder imported successfully")
    print("✓ CShaperProcessor summary generated")
    
    return True


def main():
    print("="*60)
    print("CShaper Integration Complete Test Suite")
    print("="*60)
    
    results = {}
    
    # Run all tests
    results["StandardSpatialLoader"] = test_standard_spatial_loader()
    results["Segmentation3DLoader"] = test_segmentation_3d_loader()
    results["TimeDynamicContactGraph"] = test_time_dynamic_contact_graph()
    results["ContactGNN"] = test_contact_gnn()
    results["Integration"] = test_enhanced_builder_integration()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL/SKIP"
        print(f"  {name}: {status}")
    
    n_passed = sum(results.values())
    n_total = len(results)
    print(f"\nTotal: {n_passed}/{n_total} tests passed")
    
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
