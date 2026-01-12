"""
Spatial graph module for cell neighborhood embedding.

Instead of using raw XYZ coordinates, this module builds graphs where:
- Nodes = cells
- Edges = spatial proximity (k-nearest neighbors or radius-based)
- Node features = cell properties (lineage, type, etc.)
- Edge features = distance, relative position

This enables graph neural network-based embeddings that capture
spatial context (what cells are nearby) rather than absolute position.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .spatial import SpatialDataParser, compute_cell_distances


@dataclass
class GraphNode:
    """A node in the spatial graph (representing a cell)."""

    name: str
    position: np.ndarray  # XYZ coordinates
    timepoint: int

    # Optional attributes
    lineage_name: Optional[str] = None
    cell_type: Optional[str] = None
    diameter: float = 0.0

    # Computed features
    features: Optional[np.ndarray] = None

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, GraphNode):
            return self.name == other.name
        return False


@dataclass
class GraphEdge:
    """An edge in the spatial graph (representing spatial proximity)."""

    source: str  # Source cell name
    target: str  # Target cell name
    distance: float  # Euclidean distance

    # Relative position (target - source)
    relative_position: Optional[np.ndarray] = None

    # Optional attributes
    weight: float = 1.0  # Edge weight (can be inverse distance, etc.)

    def __hash__(self):
        return hash((self.source, self.target))


@dataclass
class SpatialGraph:
    """
    A spatial graph representing cell neighborhoods at a specific timepoint.

    Nodes are cells, edges connect spatially proximate cells.
    """

    timepoint: int
    nodes: Dict[str, GraphNode] = field(default_factory=dict)
    edges: List[GraphEdge] = field(default_factory=list)

    # Adjacency structures (built lazily)
    _adjacency: Optional[Dict[str, List[str]]] = field(
        default=None, repr=False, compare=False
    )
    _edge_index: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    @property
    def node_names(self) -> List[str]:
        return list(self.nodes.keys())

    def add_node(self, node: GraphNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.name] = node
        self._adjacency = None  # Invalidate cache
        self._edge_index = None

    def add_edge(self, edge: GraphEdge) -> None:
        """Add an edge to the graph."""
        self.edges.append(edge)
        self._adjacency = None
        self._edge_index = None

    def get_neighbors(self, cell_name: str) -> List[str]:
        """Get neighbor cell names for a given cell."""
        if self._adjacency is None:
            self._build_adjacency()
        return self._adjacency.get(cell_name, [])

    def get_neighbor_nodes(self, cell_name: str) -> List[GraphNode]:
        """Get neighbor GraphNode objects for a given cell."""
        neighbor_names = self.get_neighbors(cell_name)
        return [self.nodes[n] for n in neighbor_names if n in self.nodes]

    def _build_adjacency(self) -> None:
        """Build adjacency list from edges."""
        self._adjacency = {name: [] for name in self.nodes}
        for edge in self.edges:
            if edge.source in self._adjacency:
                self._adjacency[edge.source].append(edge.target)
            # For undirected graph, add reverse edge too
            if edge.target in self._adjacency:
                self._adjacency[edge.target].append(edge.source)

    def get_edge_index(self) -> np.ndarray:
        """
        Get edge index in COO format (2 x num_edges).

        Returns array where:
        - row 0 = source node indices
        - row 1 = target node indices

        This format is compatible with PyTorch Geometric.
        """
        if self._edge_index is not None:
            return self._edge_index

        name_to_idx = {name: i for i, name in enumerate(self.nodes.keys())}

        sources = []
        targets = []

        for edge in self.edges:
            if edge.source in name_to_idx and edge.target in name_to_idx:
                src_idx = name_to_idx[edge.source]
                tgt_idx = name_to_idx[edge.target]
                # Add both directions for undirected graph
                sources.extend([src_idx, tgt_idx])
                targets.extend([tgt_idx, src_idx])

        self._edge_index = np.array([sources, targets], dtype=np.int64)
        return self._edge_index

    def get_node_features(self) -> Tuple[np.ndarray, List[str]]:
        """
        Get node feature matrix.

        Returns:
            Tuple of (N x F feature matrix, list of node names)
        """
        names = list(self.nodes.keys())
        features = []

        for name in names:
            node = self.nodes[name]
            if node.features is not None:
                features.append(node.features)
            else:
                # Default: use normalized position
                features.append(node.position)

        return np.array(features), names

    def get_edge_weights(self) -> np.ndarray:
        """Get edge weights as array (matches edge_index order)."""
        weights = []
        for edge in self.edges:
            # Add both directions
            weights.extend([edge.weight, edge.weight])
        return np.array(weights)

    def get_edge_distances(self) -> np.ndarray:
        """Get edge distances as array (matches edge_index order)."""
        distances = []
        for edge in self.edges:
            distances.extend([edge.distance, edge.distance])
        return np.array(distances)

    def get_position_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """Get position matrix for all nodes."""
        names = list(self.nodes.keys())
        positions = np.array([self.nodes[n].position for n in names])
        return positions, names

    def get_degree_distribution(self) -> Dict[str, int]:
        """Get degree (number of neighbors) for each cell."""
        if self._adjacency is None:
            self._build_adjacency()
        return {name: len(neighbors) for name, neighbors in self._adjacency.items()}

    def subgraph(self, cell_names: List[str]) -> "SpatialGraph":
        """Extract a subgraph containing only specified cells."""
        cell_set = set(cell_names)

        new_graph = SpatialGraph(timepoint=self.timepoint)

        # Add nodes
        for name in cell_names:
            if name in self.nodes:
                new_graph.add_node(self.nodes[name])

        # Add edges (only between cells in the subgraph)
        for edge in self.edges:
            if edge.source in cell_set and edge.target in cell_set:
                new_graph.add_edge(edge)

        return new_graph

    def to_dict(self) -> Dict[str, Any]:
        """Export graph as dictionary."""
        return {
            "timepoint": self.timepoint,
            "nodes": {
                name: {
                    "position": node.position.tolist(),
                    "lineage_name": node.lineage_name,
                    "cell_type": node.cell_type,
                    "diameter": node.diameter,
                }
                for name, node in self.nodes.items()
            },
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "distance": e.distance,
                    "weight": e.weight,
                }
                for e in self.edges
            ],
        }

    def save_json(self, path: str) -> None:
        """Save graph to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "SpatialGraph":
        """Load graph from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        graph = cls(timepoint=data["timepoint"])

        for name, info in data["nodes"].items():
            node = GraphNode(
                name=name,
                position=np.array(info["position"]),
                timepoint=data["timepoint"],
                lineage_name=info.get("lineage_name"),
                cell_type=info.get("cell_type"),
                diameter=info.get("diameter", 0.0),
            )
            graph.add_node(node)

        for e in data["edges"]:
            edge = GraphEdge(
                source=e["source"],
                target=e["target"],
                distance=e["distance"],
                weight=e.get("weight", 1.0),
            )
            graph.add_edge(edge)

        return graph


class SpatialGraphBuilder:
    """
    Builder for constructing spatial graphs from WormGUIDES nuclei data.

    Supports multiple graph construction methods:
    - k-nearest neighbors (KNN)
    - Radius-based (all cells within distance r)
    - Delaunay triangulation
    """

    def __init__(self, spatial_parser: SpatialDataParser):
        """
        Initialize the builder.

        Args:
            spatial_parser: SpatialDataParser instance for nuclei data
        """
        self.parser = spatial_parser

    def build_knn_graph(
        self,
        timepoint: int,
        k: int = 10,
        include_self_loops: bool = False,
        weight_by_distance: bool = True,
    ) -> SpatialGraph:
        """
        Build a k-nearest neighbors graph.

        Args:
            timepoint: Timepoint to build graph for
            k: Number of nearest neighbors per cell
            include_self_loops: Whether to include self-loops
            weight_by_distance: If True, edge weight = 1 / (1 + distance)

        Returns:
            SpatialGraph with KNN edges
        """
        tp_data = self.parser.parse_timepoint(timepoint)
        named_cells = tp_data.named_cells

        if not named_cells:
            return SpatialGraph(timepoint=timepoint)

        # Build graph
        graph = SpatialGraph(timepoint=timepoint)

        # Add nodes
        for cell in named_cells:
            node = GraphNode(
                name=cell.cell_name,
                position=cell.position,
                timepoint=timepoint,
                lineage_name=cell.cell_name,  # In WormGUIDES, name = lineage
                diameter=cell.diameter,
            )
            graph.add_node(node)

        # Compute distances
        positions = np.array([c.position for c in named_cells])
        names = [c.cell_name for c in named_cells]
        distances = compute_cell_distances(positions)

        # Find k-nearest neighbors for each cell
        n_cells = len(named_cells)
        k_actual = min(k, n_cells - 1)  # Can't have more neighbors than cells - 1

        for i, cell_name in enumerate(names):
            # Get distances from this cell
            dist_from_i = distances[i].copy()

            if not include_self_loops:
                dist_from_i[i] = np.inf  # Exclude self

            # Find k nearest neighbors
            neighbor_indices = np.argsort(dist_from_i)[:k_actual]

            for j in neighbor_indices:
                if i < j:  # Only add edge once (avoid duplicates)
                    dist = distances[i, j]
                    weight = 1.0 / (1.0 + dist) if weight_by_distance else 1.0

                    rel_pos = positions[j] - positions[i]

                    edge = GraphEdge(
                        source=cell_name,
                        target=names[j],
                        distance=dist,
                        relative_position=rel_pos,
                        weight=weight,
                    )
                    graph.add_edge(edge)

        return graph

    def build_radius_graph(
        self,
        timepoint: int,
        radius: float = 50.0,
        weight_by_distance: bool = True,
    ) -> SpatialGraph:
        """
        Build a radius-based graph (connect cells within distance r).

        Args:
            timepoint: Timepoint to build graph for
            radius: Maximum distance for edge connection
            weight_by_distance: If True, edge weight = 1 / (1 + distance)

        Returns:
            SpatialGraph with radius-based edges
        """
        tp_data = self.parser.parse_timepoint(timepoint)
        named_cells = tp_data.named_cells

        if not named_cells:
            return SpatialGraph(timepoint=timepoint)

        # Build graph
        graph = SpatialGraph(timepoint=timepoint)

        # Add nodes
        for cell in named_cells:
            node = GraphNode(
                name=cell.cell_name,
                position=cell.position,
                timepoint=timepoint,
                lineage_name=cell.cell_name,
                diameter=cell.diameter,
            )
            graph.add_node(node)

        # Compute distances
        positions = np.array([c.position for c in named_cells])
        names = [c.cell_name for c in named_cells]
        distances = compute_cell_distances(positions)

        # Connect cells within radius
        n_cells = len(named_cells)

        for i in range(n_cells):
            for j in range(i + 1, n_cells):
                dist = distances[i, j]
                if dist <= radius:
                    weight = 1.0 / (1.0 + dist) if weight_by_distance else 1.0

                    rel_pos = positions[j] - positions[i]

                    edge = GraphEdge(
                        source=names[i],
                        target=names[j],
                        distance=dist,
                        relative_position=rel_pos,
                        weight=weight,
                    )
                    graph.add_edge(edge)

        return graph

    def build_adaptive_graph(
        self,
        timepoint: int,
        base_k: int = 6,
        diameter_scale: float = 2.0,
    ) -> SpatialGraph:
        """
        Build an adaptive graph that considers cell size.

        Cells are connected if they are within diameter_scale * (d1 + d2) / 2
        of each other, or are in the top-k nearest neighbors.

        Args:
            timepoint: Timepoint to build graph for
            base_k: Minimum number of neighbors per cell
            diameter_scale: Scale factor for diameter-based connection

        Returns:
            SpatialGraph with adaptive edges
        """
        tp_data = self.parser.parse_timepoint(timepoint)
        named_cells = tp_data.named_cells

        if not named_cells:
            return SpatialGraph(timepoint=timepoint)

        # Build graph
        graph = SpatialGraph(timepoint=timepoint)

        # Add nodes
        for cell in named_cells:
            node = GraphNode(
                name=cell.cell_name,
                position=cell.position,
                timepoint=timepoint,
                lineage_name=cell.cell_name,
                diameter=cell.diameter,
            )
            graph.add_node(node)

        # Compute distances
        positions = np.array([c.position for c in named_cells])
        diameters = np.array([c.diameter for c in named_cells])
        names = [c.cell_name for c in named_cells]
        distances = compute_cell_distances(positions)

        n_cells = len(named_cells)
        connected = set()

        # First pass: diameter-based connections
        for i in range(n_cells):
            for j in range(i + 1, n_cells):
                avg_diameter = (diameters[i] + diameters[j]) / 2
                threshold = diameter_scale * avg_diameter

                if distances[i, j] <= threshold:
                    connected.add((i, j))

        # Second pass: ensure minimum k neighbors
        for i in range(n_cells):
            dist_from_i = distances[i].copy()
            dist_from_i[i] = np.inf

            neighbor_indices = np.argsort(dist_from_i)[:base_k]
            for j in neighbor_indices:
                if i < j:
                    connected.add((i, j))
                else:
                    connected.add((j, i))

        # Add edges
        for i, j in connected:
            dist = distances[i, j]
            weight = 1.0 / (1.0 + dist)
            rel_pos = positions[j] - positions[i]

            edge = GraphEdge(
                source=names[i],
                target=names[j],
                distance=dist,
                relative_position=rel_pos,
                weight=weight,
            )
            graph.add_edge(edge)

        return graph

    def build_temporal_graph_sequence(
        self,
        timepoints: Optional[List[int]] = None,
        k: int = 10,
        method: str = "knn",
    ) -> Dict[int, SpatialGraph]:
        """
        Build a sequence of graphs over time.

        Args:
            timepoints: List of timepoints. If None, use all available.
            k: Number of neighbors (for knn method)
            method: Graph construction method ("knn", "radius", "adaptive")

        Returns:
            Dictionary mapping timepoint -> SpatialGraph
        """
        if timepoints is None:
            timepoints = self.parser.get_available_timepoints()

        graphs = {}

        for tp in timepoints:
            if method == "knn":
                graphs[tp] = self.build_knn_graph(tp, k=k)
            elif method == "radius":
                graphs[tp] = self.build_radius_graph(tp)
            elif method == "adaptive":
                graphs[tp] = self.build_adaptive_graph(tp, base_k=k)
            else:
                raise ValueError(f"Unknown method: {method}")

        return graphs


def compute_neighborhood_features(
    graph: SpatialGraph,
    aggregation: str = "mean",
) -> Dict[str, np.ndarray]:
    """
    Compute neighborhood-based features for each cell.

    Features include:
    - Mean/sum of neighbor positions (relative)
    - Number of neighbors
    - Mean distance to neighbors
    - Neighbor cell diameters

    Args:
        graph: SpatialGraph to compute features for
        aggregation: How to aggregate neighbor features ("mean" or "sum")

    Returns:
        Dictionary mapping cell name -> feature vector
    """
    features = {}

    for name, node in graph.nodes.items():
        neighbors = graph.get_neighbor_nodes(name)

        if not neighbors:
            # No neighbors: use zeros
            feat = np.zeros(10)
        else:
            # Relative positions
            rel_positions = np.array([n.position - node.position for n in neighbors])

            if aggregation == "mean":
                mean_rel_pos = rel_positions.mean(axis=0)
                std_rel_pos = rel_positions.std(axis=0)
            else:
                mean_rel_pos = rel_positions.sum(axis=0)
                std_rel_pos = rel_positions.std(axis=0)

            # Distances
            distances = np.linalg.norm(rel_positions, axis=1)
            mean_dist = distances.mean()
            std_dist = distances.std()

            # Neighbor diameters
            neighbor_diameters = np.array([n.diameter for n in neighbors])
            mean_diameter = neighbor_diameters.mean()

            # Combine features
            feat = np.concatenate(
                [
                    mean_rel_pos,  # 3 dims
                    std_rel_pos,  # 3 dims
                    [mean_dist, std_dist],  # 2 dims
                    [len(neighbors), mean_diameter],  # 2 dims
                ]
            )

        features[name] = feat

    return features


def compute_lineage_neighborhood_overlap(
    graph: SpatialGraph,
    lineage_tree: Any,  # LineageTree from multimodal.py
) -> Dict[str, Dict[str, float]]:
    """
    Compute overlap between spatial neighborhoods and lineage relationships.

    For each cell, computes:
    - Fraction of neighbors that are lineage siblings
    - Fraction of neighbors that are lineage cousins
    - Mean lineage distance to neighbors

    Args:
        graph: SpatialGraph
        lineage_tree: LineageTree object

    Returns:
        Dictionary mapping cell name -> metrics dict
    """
    metrics = {}

    for name, node in graph.nodes.items():
        neighbors = graph.get_neighbors(name)

        if not neighbors or node.lineage_name is None:
            metrics[name] = {
                "sibling_fraction": 0.0,
                "cousin_fraction": 0.0,
                "mean_lineage_distance": 0.0,
            }
            continue

        lineage_distances = []
        sibling_count = 0
        cousin_count = 0

        for neighbor_name in neighbors:
            neighbor_node = graph.nodes.get(neighbor_name)
            if neighbor_node is None or neighbor_node.lineage_name is None:
                continue

            # Compute lineage distance
            lin_dist = lineage_tree.get_lineage_distance(
                node.lineage_name, neighbor_node.lineage_name
            )
            if lin_dist >= 0:
                lineage_distances.append(lin_dist)

                # Siblings: lineage distance = 2 (share parent)
                if lin_dist == 2:
                    sibling_count += 1
                # Cousins: lineage distance = 4 (share grandparent)
                elif lin_dist == 4:
                    cousin_count += 1

        n_valid = len(lineage_distances)
        metrics[name] = {
            "sibling_fraction": sibling_count / n_valid if n_valid > 0 else 0.0,
            "cousin_fraction": cousin_count / n_valid if n_valid > 0 else 0.0,
            "mean_lineage_distance": (
                np.mean(lineage_distances) if lineage_distances else 0.0
            ),
        }

    return metrics


def graph_to_pyg_data(
    graph: SpatialGraph,
    node_features: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Convert SpatialGraph to PyTorch Geometric compatible format.

    Args:
        graph: SpatialGraph to convert
        node_features: Optional node feature matrix (N x F).
                      If None, uses position as features.

    Returns:
        Dictionary with keys:
        - 'x': Node features (N x F)
        - 'edge_index': Edge indices (2 x E)
        - 'edge_attr': Edge attributes (E x D)
        - 'pos': Node positions (N x 3)
        - 'node_names': List of node names
    """
    # Node positions
    positions, names = graph.get_position_matrix()

    # Node features
    if node_features is None:
        # Normalize positions as default features
        x = (positions - positions.mean(axis=0)) / (positions.std(axis=0) + 1e-8)
    else:
        x = node_features

    # Edge index
    edge_index = graph.get_edge_index()

    # Edge attributes: [distance, weight, relative_x, relative_y, relative_z]
    edge_attr = []
    for edge in graph.edges:
        rel_pos = (
            edge.relative_position
            if edge.relative_position is not None
            else np.zeros(3)
        )
        # Add both directions
        edge_attr.append([edge.distance, edge.weight, *rel_pos])
        edge_attr.append([edge.distance, edge.weight, *(-rel_pos)])

    edge_attr = np.array(edge_attr) if edge_attr else np.zeros((0, 5))

    return {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "pos": positions,
        "node_names": names,
    }
