"""
Multimodal Embedding Example for C. elegans Single-Cell Analysis.

This example demonstrates how to create embeddings that integrate:
1. Spatial context (cell neighborhood graphs)
2. Lineage relationships (cell lineage tree distances)
3. Transcriptome data (gene expression - optional)

The approach uses graph neural networks to learn representations
that capture the spatial organization of cells, while also
incorporating lineage information as additional features.

Requirements:
    pip install torch torch-geometric numpy

Usage:
    python examples/multimodal_embedding.py
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List

import numpy as np

# =============================================================================
# Data Loading
# =============================================================================


def load_spatial_graph(timepoint: int = 200, k: int = 10):
    """Load spatial graph for a given timepoint."""
    from src.model.spatial import SpatialDataParser
    from src.model.spatial_graph import SpatialGraphBuilder, graph_to_pyg_data

    parser = SpatialDataParser("dataset/raw/wormguides/nuclei_files")
    builder = SpatialGraphBuilder(parser)

    graph = builder.build_knn_graph(timepoint=timepoint, k=k)
    pyg_data = graph_to_pyg_data(graph)

    return graph, pyg_data


def load_lineage_features(cell_names: List[str]) -> np.ndarray:
    """
    Create lineage-based features for cells.

    Features encode the lineage identity:
    - One-hot encoding of founder lineage (AB, MS, E, C, D, P4)
    - Lineage depth (normalized)
    - Binary features for common patterns
    """
    from src.model.multimodal import LineageTree

    tree = LineageTree()
    tree.build_from_cell_names(cell_names)

    # Define founder lineages
    founders = ["AB", "MS", "E", "C", "D", "P4", "Z2", "Z3"]

    features = []
    for name in cell_names:
        feat = []

        # Founder lineage one-hot (8 dims)
        founder_onehot = [0.0] * 8
        for i, founder in enumerate(founders):
            if name.startswith(founder):
                founder_onehot[i] = 1.0
                break
        feat.extend(founder_onehot)

        # Lineage depth (1 dim)
        node = tree.get_node(name)
        depth = node.depth if node else 0
        feat.append(depth / 15.0)  # Normalize by max depth

        # Pattern features (4 dims)
        # - ends with 'a' (anterior)
        # - ends with 'p' (posterior)
        # - ends with 'l' (left)
        # - ends with 'r' (right)
        if len(name) > 2:
            last_char = name[-1].lower()
            feat.append(1.0 if last_char == "a" else 0.0)
            feat.append(1.0 if last_char == "p" else 0.0)
            feat.append(1.0 if last_char == "l" else 0.0)
            feat.append(1.0 if last_char == "r" else 0.0)
        else:
            feat.extend([0.0, 0.0, 0.0, 0.0])

        features.append(feat)

    return np.array(features, dtype=np.float32)


def compute_lineage_distance_matrix(cell_names: List[str]) -> np.ndarray:
    """Compute pairwise lineage distances between cells."""
    from src.model.multimodal import LineageTree

    tree = LineageTree()
    tree.build_from_cell_names(cell_names)

    distances, _ = tree.compute_lineage_distance_matrix(cell_names)
    return distances.astype(np.float32)


# =============================================================================
# Simple Graph Neural Network (Pure NumPy Implementation)
# =============================================================================


class SimpleGNN:
    """
    A simple Graph Neural Network for learning cell embeddings.

    This is a pure NumPy implementation for demonstration.
    For production, use PyTorch Geometric.

    Architecture:
        - Input: node features (positions + lineage)
        - Hidden: 2 GCN-like layers with message passing
        - Output: embedding vectors
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 32,
        num_layers: int = 2,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Initialize weights (Xavier initialization)
        self.weights = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]

        for i in range(num_layers):
            scale = np.sqrt(2.0 / (dims[i] + dims[i + 1]))
            W = np.random.randn(dims[i], dims[i + 1]).astype(np.float32) * scale
            self.weights.append(W)

    def _normalize_adjacency(self, adj: np.ndarray) -> np.ndarray:
        """Compute normalized adjacency: D^(-1/2) A D^(-1/2)."""
        # Add self-loops
        adj = adj + np.eye(adj.shape[0], dtype=np.float32)

        # Compute degree
        degree = np.sum(adj, axis=1)
        degree_inv_sqrt = np.power(degree, -0.5, where=degree > 0)
        degree_inv_sqrt[degree == 0] = 0

        # Normalize
        D_inv_sqrt = np.diag(degree_inv_sqrt)
        return D_inv_sqrt @ adj @ D_inv_sqrt

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def forward(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
    ) -> np.ndarray:
        """
        Forward pass through the GNN.

        Args:
            node_features: (N, input_dim) node feature matrix
            edge_index: (2, E) edge index in COO format

        Returns:
            (N, output_dim) node embeddings
        """
        N = node_features.shape[0]

        # Build adjacency matrix from edge index
        adj = np.zeros((N, N), dtype=np.float32)
        for i in range(edge_index.shape[1]):
            src, tgt = edge_index[0, i], edge_index[1, i]
            adj[src, tgt] = 1.0

        # Normalize adjacency
        adj_norm = self._normalize_adjacency(adj)

        # Forward through layers
        h = node_features
        for i, W in enumerate(self.weights):
            # Message passing: aggregate neighbor features
            h = adj_norm @ h

            # Transform
            h = h @ W

            # Non-linearity (except last layer)
            if i < len(self.weights) - 1:
                h = self._relu(h)

        return h


# =============================================================================
# Multimodal Embedding
# =============================================================================


class MultimodalEmbedding:
    """
    Multimodal embedding that combines spatial and lineage information.

    The embedding learns to represent cells in a space where:
    - Spatially proximate cells are close
    - Lineage-related cells are close
    - The balance is controlled by weights
    """

    def __init__(
        self,
        spatial_dim: int = 3,
        lineage_dim: int = 13,
        hidden_dim: int = 64,
        embedding_dim: int = 32,
    ):
        self.spatial_dim = spatial_dim
        self.lineage_dim = lineage_dim
        self.embedding_dim = embedding_dim

        # Combined input: spatial (3) + lineage features (13)
        input_dim = spatial_dim + lineage_dim

        self.gnn = SimpleGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
        )

    def compute_embeddings(
        self,
        positions: np.ndarray,
        lineage_features: np.ndarray,
        edge_index: np.ndarray,
    ) -> np.ndarray:
        """
        Compute multimodal embeddings for cells.

        Args:
            positions: (N, 3) cell positions (normalized)
            lineage_features: (N, lineage_dim) lineage features
            edge_index: (2, E) spatial neighbor edges

        Returns:
            (N, embedding_dim) cell embeddings
        """
        # Normalize positions
        pos_norm = (positions - positions.mean(axis=0)) / (positions.std(axis=0) + 1e-8)

        # Concatenate features
        node_features = np.concatenate([pos_norm, lineage_features], axis=1)
        node_features = node_features.astype(np.float32)

        # Forward through GNN
        embeddings = self.gnn.forward(node_features, edge_index)

        return embeddings

    def compute_similarity_matrix(
        self,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """Compute pairwise cosine similarity between embeddings."""
        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_norm = embeddings / (norms + 1e-8)

        # Cosine similarity
        similarity = embeddings_norm @ embeddings_norm.T

        return similarity


# =============================================================================
# Analysis and Visualization
# =============================================================================


def analyze_embeddings(
    embeddings: np.ndarray,
    cell_names: List[str],
    lineage_distances: np.ndarray,
    spatial_distances: np.ndarray,
):
    """
    Analyze the learned embeddings.

    Checks if embedding distances correlate with:
    - Lineage distances
    - Spatial distances
    """
    # Compute embedding distances
    N = embeddings.shape[0]
    embedding_dists = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            embedding_dists[i, j] = np.linalg.norm(embeddings[i] - embeddings[j])

    # Flatten upper triangular (excluding diagonal)
    triu_idx = np.triu_indices(N, k=1)

    emb_flat = embedding_dists[triu_idx]
    lin_flat = lineage_distances[triu_idx]
    spa_flat = spatial_distances[triu_idx]

    # Compute correlations
    def pearson_corr(x, y):
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        num = np.sum(x_centered * y_centered)
        denom = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
        return num / (denom + 1e-8)

    corr_lineage = pearson_corr(emb_flat, lin_flat)
    corr_spatial = pearson_corr(emb_flat, spa_flat)

    print("\n=== Embedding Analysis ===")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Number of cells: {N}")
    print("\nCorrelation between embedding distance and:")
    print(f"  - Lineage distance: {corr_lineage:.3f}")
    print(f"  - Spatial distance: {corr_spatial:.3f}")

    # Find cells with most similar embeddings
    print("\n=== Most Similar Cell Pairs (by embedding) ===")
    flat_idx = np.argsort(emb_flat)[:10]
    for idx in flat_idx[:5]:
        i, j = triu_idx[0][idx], triu_idx[1][idx]
        print(f"  {cell_names[i]} <-> {cell_names[j]}")
        print(
            f"    Embedding dist: {embedding_dists[i, j]:.3f}, "
            f"Lineage dist: {lineage_distances[i, j]}, "
            f"Spatial dist: {spatial_distances[i, j]:.1f}"
        )


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 60)
    print("Multimodal Embedding: Transcriptome-Spatial-Lineage")
    print("=" * 60)

    # Load spatial graph
    print("\n[1] Loading spatial graph...")
    graph, pyg_data = load_spatial_graph(timepoint=200, k=10)
    cell_names = pyg_data["node_names"]
    positions = pyg_data["pos"]
    edge_index = pyg_data["edge_index"]

    print(f"    Cells: {len(cell_names)}")
    print(f"    Edges: {edge_index.shape[1] // 2}")  # Undirected

    # Load lineage features
    print("\n[2] Computing lineage features...")
    lineage_features = load_lineage_features(cell_names)
    print(f"    Lineage feature dim: {lineage_features.shape[1]}")

    # Compute lineage distances
    print("\n[3] Computing lineage distance matrix...")
    lineage_distances = compute_lineage_distance_matrix(cell_names)

    # Compute spatial distances
    print("\n[4] Computing spatial distance matrix...")
    from src.model.spatial import compute_cell_distances

    spatial_distances = compute_cell_distances(positions)

    # Create multimodal embedding
    print("\n[5] Computing multimodal embeddings...")
    embedder = MultimodalEmbedding(
        spatial_dim=3,
        lineage_dim=lineage_features.shape[1],
        hidden_dim=64,
        embedding_dim=32,
    )

    embeddings = embedder.compute_embeddings(
        positions=positions,
        lineage_features=lineage_features,
        edge_index=edge_index,
    )
    print(f"    Embedding shape: {embeddings.shape}")

    # Analyze embeddings
    analyze_embeddings(
        embeddings=embeddings,
        cell_names=cell_names,
        lineage_distances=lineage_distances,
        spatial_distances=spatial_distances,
    )

    # Show embedding for sample cells
    print("\n=== Sample Embeddings ===")
    sample_cells = ["ABa", "ABp", "MSa", "MSp", "Ea", "Ep", "Ca", "Cp"]
    for name in sample_cells:
        if name in cell_names:
            idx = cell_names.index(name)
            emb = embeddings[idx]
            print(f"  {name}: [{emb[0]:.2f}, {emb[1]:.2f}, {emb[2]:.2f}, ...]")

    print("\n" + "=" * 60)
    print("Done! The embeddings capture both spatial and lineage structure.")
    print("=" * 60)


if __name__ == "__main__":
    main()
