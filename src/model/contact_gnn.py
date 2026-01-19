"""
Contact Graph Neural Network module for CShaper data.

This module provides GNN layers that operate on true physical contact
graphs derived from CShaper cell-cell contact data, rather than
k-NN approximations based on spatial proximity.

Key features:
- Edge weights represent physical contact surface area
- Supports time-dynamic graphs (contacts change over development)
- Integrates morphological features (volume, surface, sphericity)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ContactGraph:
    """
    A contact graph representing physical cell-cell contacts.
    
    Unlike k-NN spatial graphs, edges here represent true physical contact
    based on membrane segmentation data from CShaper.
    
    Attributes:
        n_cells: Number of cells
        cell_names: List of cell lineage names
        edge_index: (2, n_edges) array of edge indices
        edge_weight: (n_edges,) array of contact surface areas (μm²)
        node_features: Optional (n_cells, n_features) feature matrix
        time_frame: Optional developmental time frame
    """
    n_cells: int
    cell_names: List[str]
    edge_index: np.ndarray  # (2, n_edges)
    edge_weight: np.ndarray  # (n_edges,)
    node_features: Optional[np.ndarray] = None
    time_frame: Optional[int] = None
    
    # Cached adjacency
    _adjacency: Optional[csr_matrix] = field(default=None, repr=False)
    
    @classmethod
    def from_csr_matrix(
        cls,
        adj_matrix: csr_matrix,
        cell_names: List[str],
        node_features: Optional[np.ndarray] = None,
        time_frame: Optional[int] = None,
    ) -> "ContactGraph":
        """
        Create ContactGraph from sparse adjacency matrix.
        
        Args:
            adj_matrix: Sparse CSR adjacency matrix (n_cells, n_cells)
            cell_names: List of cell names
            node_features: Optional node feature matrix
            time_frame: Optional time frame
            
        Returns:
            ContactGraph instance
        """
        # Convert to COO for edge extraction
        coo = adj_matrix.tocoo()
        
        # Only keep one direction for undirected graph
        mask = coo.row < coo.col
        row = coo.row[mask]
        col = coo.col[mask]
        data = coo.data[mask]
        
        # Create bidirectional edge_index
        edge_index = np.vstack([
            np.concatenate([row, col]),
            np.concatenate([col, row])
        ])
        edge_weight = np.concatenate([data, data])
        
        return cls(
            n_cells=adj_matrix.shape[0],
            cell_names=cell_names,
            edge_index=edge_index,
            edge_weight=edge_weight,
            node_features=node_features,
            time_frame=time_frame,
            _adjacency=adj_matrix,
        )
    
    @property
    def n_edges(self) -> int:
        """Number of edges (directed, so each undirected edge counts twice)."""
        return self.edge_index.shape[1]
    
    @property
    def adjacency(self) -> csr_matrix:
        """Get sparse adjacency matrix."""
        if self._adjacency is None:
            # Build from edge_index
            data = self.edge_weight
            row = self.edge_index[0]
            col = self.edge_index[1]
            self._adjacency = csr_matrix(
                (data, (row, col)),
                shape=(self.n_cells, self.n_cells)
            )
        return self._adjacency
    
    def get_neighbors(self, cell_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get neighbors of a cell.
        
        Args:
            cell_idx: Cell index
            
        Returns:
            Tuple of (neighbor_indices, contact_areas)
        """
        mask = self.edge_index[0] == cell_idx
        neighbors = self.edge_index[1, mask]
        weights = self.edge_weight[mask]
        return neighbors, weights
    
    def get_degree(self) -> np.ndarray:
        """Get degree (number of contacts) for each cell."""
        degree = np.zeros(self.n_cells, dtype=np.int32)
        np.add.at(degree, self.edge_index[0], 1)
        return degree // 2  # Divide by 2 since edges are bidirectional
    
    def subgraph(self, cell_indices: np.ndarray) -> "ContactGraph":
        """Extract subgraph for subset of cells."""
        index_set = set(cell_indices.tolist())
        old_to_new = {old: new for new, old in enumerate(cell_indices)}
        
        new_edges_src = []
        new_edges_dst = []
        new_weights = []
        
        for i in range(self.n_edges):
            src = self.edge_index[0, i]
            dst = self.edge_index[1, i]
            if src in index_set and dst in index_set:
                new_edges_src.append(old_to_new[src])
                new_edges_dst.append(old_to_new[dst])
                new_weights.append(self.edge_weight[i])
        
        new_edge_index = np.array([new_edges_src, new_edges_dst])
        new_edge_weight = np.array(new_weights)
        new_names = [self.cell_names[i] for i in cell_indices]
        new_features = self.node_features[cell_indices] if self.node_features is not None else None
        
        return ContactGraph(
            n_cells=len(cell_indices),
            cell_names=new_names,
            edge_index=new_edge_index,
            edge_weight=new_edge_weight,
            node_features=new_features,
            time_frame=self.time_frame,
        )


@dataclass
class TimeDynamicContactGraph:
    """
    A time-varying contact graph that tracks how contacts change over development.
    
    Useful for modeling developmental dynamics where cell-cell contacts
    form and break as cells divide and migrate.
    """
    cell_names: List[str]  # Union of all cell names across time
    time_frames: List[int]  # Sorted list of time frames
    graphs: Dict[int, ContactGraph]  # frame -> ContactGraph
    
    @classmethod
    def from_frame_graphs(
        cls,
        graphs: Dict[int, ContactGraph],
    ) -> "TimeDynamicContactGraph":
        """
        Create from dictionary of per-frame graphs.
        
        All graphs should have the same cell ordering.
        """
        if not graphs:
            return cls(cell_names=[], time_frames=[], graphs={})
        
        first_graph = next(iter(graphs.values()))
        return cls(
            cell_names=first_graph.cell_names,
            time_frames=sorted(graphs.keys()),
            graphs=graphs,
        )
    
    def get_frame(self, time_frame: int) -> Optional[ContactGraph]:
        """Get graph for a specific time frame."""
        return self.graphs.get(time_frame)
    
    def get_contact_trajectory(
        self,
        cell1_idx: int,
        cell2_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get contact area trajectory between two cells over time.
        
        Returns:
            Tuple of (time_frames, contact_areas)
        """
        times = []
        areas = []
        
        for t in self.time_frames:
            graph = self.graphs[t]
            adj = graph.adjacency
            area = adj[cell1_idx, cell2_idx]
            times.append(t)
            areas.append(area)
        
        return np.array(times), np.array(areas)
    
    def get_temporal_edges(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get all edges across all time frames.
        
        Returns:
            Tuple of (edge_index, edge_weight, edge_time)
            where edge_time indicates which frame each edge belongs to.
        """
        all_src = []
        all_dst = []
        all_weight = []
        all_time = []
        
        for t in self.time_frames:
            graph = self.graphs[t]
            n_edges = graph.n_edges
            all_src.append(graph.edge_index[0])
            all_dst.append(graph.edge_index[1])
            all_weight.append(graph.edge_weight)
            all_time.append(np.full(n_edges, t))
        
        edge_index = np.vstack([
            np.concatenate(all_src),
            np.concatenate(all_dst)
        ])
        edge_weight = np.concatenate(all_weight)
        edge_time = np.concatenate(all_time)
        
        return edge_index, edge_weight, edge_time


# =============================================================================
# GNN Layers (NumPy implementation - framework agnostic)
# =============================================================================

class ContactMessagePassing:
    """
    Message passing layer for contact graphs.
    
    Implements a simple aggregation scheme:
    h'_i = UPDATE(h_i, AGGREGATE({h_j * w_ij : j in N(i)}))
    
    where w_ij is the contact surface area (optionally normalized).
    
    This is a NumPy reference implementation. For training,
    use PyTorch/PyG version.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        aggregation: str = "mean",
        normalize_weights: bool = True,
        use_edge_features: bool = True,
    ):
        """
        Initialize the layer.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            aggregation: How to aggregate neighbor messages ("mean", "sum", "max")
            normalize_weights: Whether to normalize contact areas to sum to 1
            use_edge_features: Whether to use contact area as edge features
        """
        self.in_features = in_features
        self.out_features = out_features
        self.aggregation = aggregation
        self.normalize_weights = normalize_weights
        self.use_edge_features = use_edge_features
        
        # Initialize learnable parameters (random for testing)
        self.W_self = np.random.randn(in_features, out_features) * 0.1
        self.W_neighbor = np.random.randn(in_features, out_features) * 0.1
        self.bias = np.zeros(out_features)
    
    def forward(
        self,
        x: np.ndarray,
        graph: ContactGraph,
    ) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            x: Node features (n_cells, in_features)
            graph: ContactGraph
            
        Returns:
            Updated node features (n_cells, out_features)
        """
        n_cells = graph.n_cells
        
        # Self transformation
        h_self = x @ self.W_self  # (n_cells, out_features)
        
        # Neighbor aggregation
        h_neighbor = np.zeros((n_cells, self.out_features))
        
        for i in range(n_cells):
            neighbors, weights = graph.get_neighbors(i)
            
            if len(neighbors) == 0:
                continue
            
            # Get neighbor features
            neighbor_features = x[neighbors]  # (n_neighbors, in_features)
            
            # Transform
            neighbor_transformed = neighbor_features @ self.W_neighbor  # (n_neighbors, out_features)
            
            if self.use_edge_features:
                # Weight by contact area
                if self.normalize_weights:
                    weights = weights / (weights.sum() + 1e-8)
                # Weighted aggregation
                weighted = neighbor_transformed * weights[:, None]
            else:
                weighted = neighbor_transformed
            
            # Aggregate
            if self.aggregation == "mean":
                h_neighbor[i] = weighted.mean(axis=0)
            elif self.aggregation == "sum":
                h_neighbor[i] = weighted.sum(axis=0)
            elif self.aggregation == "max":
                h_neighbor[i] = weighted.max(axis=0)
        
        # Combine
        h_out = h_self + h_neighbor + self.bias
        
        # ReLU activation
        h_out = np.maximum(h_out, 0)
        
        return h_out


class ContactGNN:
    """
    Multi-layer Contact Graph Neural Network.
    
    A stack of ContactMessagePassing layers for learning
    cell representations from contact graph structure.
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        n_layers: int = 2,
        aggregation: str = "mean",
        dropout: float = 0.1,
    ):
        """
        Initialize the GNN.
        
        Args:
            in_features: Input feature dimension
            hidden_features: Hidden layer dimension
            out_features: Output feature dimension
            n_layers: Number of message passing layers
            aggregation: Aggregation method
            dropout: Dropout rate (not used in NumPy version)
        """
        self.n_layers = n_layers
        self.dropout = dropout
        
        # Build layers
        self.layers = []
        
        for i in range(n_layers):
            if i == 0:
                layer = ContactMessagePassing(
                    in_features, hidden_features,
                    aggregation=aggregation,
                )
            elif i == n_layers - 1:
                layer = ContactMessagePassing(
                    hidden_features, out_features,
                    aggregation=aggregation,
                )
            else:
                layer = ContactMessagePassing(
                    hidden_features, hidden_features,
                    aggregation=aggregation,
                )
            self.layers.append(layer)
    
    def forward(
        self,
        x: np.ndarray,
        graph: ContactGraph,
    ) -> np.ndarray:
        """
        Forward pass through all layers.
        
        Args:
            x: Input node features (n_cells, in_features)
            graph: ContactGraph
            
        Returns:
            Output node embeddings (n_cells, out_features)
        """
        h = x
        for layer in self.layers:
            h = layer.forward(h, graph)
        return h
    
    def get_cell_embeddings(
        self,
        graph: ContactGraph,
        morphology_features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute cell embeddings from graph structure and optional morphology.
        
        Args:
            graph: ContactGraph
            morphology_features: Optional (n_cells, n_morph) morphology features
                                (volume, surface, sphericity)
            
        Returns:
            Cell embeddings (n_cells, out_features)
        """
        # Use node features if available
        if graph.node_features is not None:
            x = graph.node_features
        else:
            # Use degree as default features
            degree = graph.get_degree().astype(np.float32)[:, None]
            x = degree
        
        # Concatenate morphology if provided
        if morphology_features is not None:
            x = np.concatenate([x, morphology_features], axis=1)
        
        # Pad to match input dimension if needed
        if x.shape[1] < self.layers[0].in_features:
            pad_width = self.layers[0].in_features - x.shape[1]
            x = np.pad(x, ((0, 0), (0, pad_width)), mode='constant')
        
        return self.forward(x, graph)


# =============================================================================
# Graph Comparison Utilities
# =============================================================================

def compare_contact_vs_knn(
    contact_graph: ContactGraph,
    knn_edge_index: np.ndarray,
) -> Dict[str, float]:
    """
    Compare contact graph with k-NN graph structure.
    
    Args:
        contact_graph: ContactGraph from CShaper
        knn_edge_index: (2, n_edges) edge index from k-NN
        
    Returns:
        Dictionary with comparison metrics
    """
    # Build edge sets
    contact_edges = set()
    for i in range(contact_graph.n_edges):
        src = contact_graph.edge_index[0, i]
        dst = contact_graph.edge_index[1, i]
        contact_edges.add((min(src, dst), max(src, dst)))
    
    knn_edges = set()
    for i in range(knn_edge_index.shape[1]):
        src = knn_edge_index[0, i]
        dst = knn_edge_index[1, i]
        knn_edges.add((min(src, dst), max(src, dst)))
    
    # Compute metrics
    intersection = contact_edges & knn_edges
    union = contact_edges | knn_edges
    
    jaccard = len(intersection) / len(union) if union else 0.0
    
    # Precision: what fraction of k-NN edges are true contacts
    precision = len(intersection) / len(knn_edges) if knn_edges else 0.0
    
    # Recall: what fraction of contacts are captured by k-NN
    recall = len(intersection) / len(contact_edges) if contact_edges else 0.0
    
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "n_contact_edges": len(contact_edges),
        "n_knn_edges": len(knn_edges),
        "n_intersection": len(intersection),
        "jaccard": jaccard,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def build_contact_graph_from_cshaper(
    cshaper_processor,
    lineage_names: List[str],
    sample_id: Optional[str] = None,
    node_features: Optional[np.ndarray] = None,
    include_morphology: bool = True,
) -> ContactGraph:
    """
    Build ContactGraph from CShaperProcessor.
    
    Args:
        cshaper_processor: CShaperProcessor instance
        lineage_names: List of cell lineage names
        sample_id: Specific sample to use (None = consensus)
        node_features: Optional additional node features
        include_morphology: Whether to include morphology features
        
    Returns:
        ContactGraph instance
    """
    # Get contact adjacency
    adj = cshaper_processor.get_contact_adjacency(
        lineage_names,
        sample_id=sample_id,
        binary=False,
    )
    
    # Get morphology features if requested
    features = node_features
    if include_morphology and cshaper_processor.has_morphology:
        morph_df = cshaper_processor.get_morphology_features(lineage_names)
        morph_features = morph_df[['volume', 'surface', 'sphericity']].values
        # Handle NaN
        morph_features = np.nan_to_num(morph_features, nan=0.0)
        
        if features is not None:
            features = np.concatenate([features, morph_features], axis=1)
        else:
            features = morph_features
    
    return ContactGraph.from_csr_matrix(
        adj,
        cell_names=lineage_names,
        node_features=features,
    )
