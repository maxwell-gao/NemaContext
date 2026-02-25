"""Lineage-aware attention bias computation for Emergent Context architecture.

Lineage information serves as architectural bias (attention modulation) rather than
training supervision. Closer cells in the lineage tree attend to each other more.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def parse_lineage_name(name: str) -> tuple[str, list[str]]:
    """Parse a lineage name into founder and developmental path.

    Examples:
        "ABal" -> ("AB", ["a", "l"])
        "MSaa" -> ("MS", ["a", "a"])
        "P4" -> ("P4", [])
        "E" -> ("E", [])

    Args:
        name: Cell lineage name (e.g., "ABal", "MSaa", "P4")

    Returns:
        Tuple of (founder, path) where path is list of division directions
    """
    # Founder cells have no path
    if name in ("AB", "MS", "E", "C", "D", "P4"):
        return name, []

    # Extract founder (first 1-2 characters based on known founders)
    if name.startswith("AB"):
        founder = "AB"
        path_str = name[2:]
    elif name.startswith("MS"):
        founder = "MS"
        path_str = name[2:]
    elif name.startswith("P4"):
        founder = "P4"
        path_str = name[2:] if len(name) > 2 else ""
    elif name.startswith("E") and len(name) == 1:
        return "E", []
    elif name.startswith("C") and len(name) == 1:
        return "C", []
    elif name.startswith("D") and len(name) == 1:
        return "D", []
    else:
        # Unknown format, treat entire name as founder
        return name, []

    # Parse path into individual characters (typically 'a', 'p', 'l', 'r', 'd', 'v')
    path = list(path_str.lower())
    return founder, path


def lineage_distance(name1: str, name2: str) -> int:
    """Compute tree distance between two lineage names.

    The distance is the number of edges between the two cells in the lineage tree.
    This equals: depth(cell1) + depth(cell2) - 2 * depth(LCA)
    where LCA is the lowest common ancestor.

    Examples:
        lineage_distance("ABal", "ABar") -> 2  # siblings
        lineage_distance("ABal", "ABal") -> 0  # same cell
        lineage_distance("ABal", "MSaa") -> large  # different founders
        lineage_distance("ABal", "AB") -> 1  # parent-child

    Args:
        name1: First cell lineage name
        name2: Second cell lineage name

    Returns:
        Integer tree distance
    """
    if name1 == name2:
        return 0

    founder1, path1 = parse_lineage_name(name1)
    founder2, path2 = parse_lineage_name(name2)

    # Different founders: distance through germ line (approximate)
    if founder1 != founder2:
        # Depth of each cell + distance between founders (approximated)
        depth1 = len(path1)
        depth2 = len(path2)
        # Founders are roughly equidistant in early development
        founder_distance = 10  # Approximate distance between different founder lineages
        return depth1 + depth2 + founder_distance

    # Same founder: find LCA by comparing paths
    # Cells share common prefix in their paths
    lca_depth = 0
    for p1, p2 in zip(path1, path2):
        if p1 == p2:
            lca_depth += 1
        else:
            break

    # Distance = depth1 + depth2 - 2 * lca_depth
    depth1 = len(path1)
    depth2 = len(path2)
    return depth1 + depth2 - 2 * lca_depth


def compute_lineage_bias(
    lineage_names: list[str],
    temperature: float = 1.0,
    max_distance: int | None = None,
) -> torch.Tensor:
    """Compute additive attention bias from pairwise lineage distances.

    The bias is negative for close cells (encouraging attention) and approaches
    zero for distant cells. This implements the biological intuition that cells
    from the same lineage branch should attend to each other more.

    Args:
        lineage_names: List of cell lineage names (e.g., ["ABal", "ABar", "MSaa"])
        temperature: Controls strength of bias. Lower = stronger bias.
            Typical values: 0.5 (strong), 1.0 (moderate), 2.0 (weak)
        max_distance: Optional cap on distance computation for numerical stability.
            If None, no cap is applied.

    Returns:
        Bias matrix of shape [n_cells, n_cells] where bias[i,j] represents
        the additive attention bias between cell i and cell j.
        bias[i,j] = -distance(name_i, name_j) / temperature
    """
    n = len(lineage_names)
    bias = torch.zeros(n, n)

    for i in range(n):
        for j in range(n):
            dist = lineage_distance(lineage_names[i], lineage_names[j])
            if max_distance is not None:
                dist = min(dist, max_distance)
            # Negative: closer in tree -> higher attention
            bias[i, j] = -dist / temperature

    return bias


def batch_lineage_bias(
    names_list: list[list[str]],
    device: torch.device | None = None,
    temperature: float = 1.0,
    max_seq_len: int | None = None,
) -> torch.Tensor | None:
    """Batch compute lineage bias matrices with padding.

    Args:
        names_list: List of lists, where each inner list contains lineage names
            for one sample in the batch.
        device: Target device for the tensor.
        temperature: Temperature for bias computation.
        max_seq_len: Optional maximum sequence length. If provided, pads/truncates
            to this length. If None, uses max length in batch.

    Returns:
        Batched bias tensor of shape [B, L, L] where B is batch size and L is
        max sequence length. Returns None if names_list is empty.
    """
    if not names_list:
        return None

    batch_size = len(names_list)

    # Determine max length
    if max_seq_len is None:
        max_len = max(len(names) for names in names_list)
    else:
        max_len = max_seq_len

    # Create batched bias tensor
    batch_bias = torch.zeros(batch_size, max_len, max_len)

    for b, names in enumerate(names_list):
        n = min(len(names), max_len)
        if n > 0:
            bias = compute_lineage_bias(names[:n], temperature=temperature)
            batch_bias[b, :n, :n] = bias

    if device is not None:
        batch_bias = batch_bias.to(device)

    return batch_bias


def apply_lineage_bias_to_attention(
    attn_scores: torch.Tensor,
    lineage_bias: torch.Tensor | None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply lineage bias to attention scores.

    This implements: attn_scores = (Q @ K.T) / sqrt(d_k) + lineage_bias

    Args:
        attn_scores: Attention scores of shape [B, H, L, L] or [B, L, L]
        lineage_bias: Lineage bias matrix of shape [B, L, L] or None.
            If None, returns attn_scores unchanged.
        mask: Optional attention mask. If provided, applied after bias.

    Returns:
        Modified attention scores with lineage bias applied.
    """
    if lineage_bias is None:
        return attn_scores

    # Expand bias for multi-head attention if needed
    if attn_scores.dim() == 4 and lineage_bias.dim() == 3:
        # lineage_bias: [B, L, L] -> [B, 1, L, L]
        lineage_bias = lineage_bias.unsqueeze(1)

    # Add bias to attention scores
    biased_scores = attn_scores + lineage_bias

    # Apply mask if provided
    if mask is not None:
        biased_scores = biased_scores.masked_fill(~mask, float("-inf"))

    return biased_scores


def create_relative_position_bias(
    seq_len: int,
    max_distance: int = 32,
    num_buckets: int = 32,
) -> torch.Tensor:
    """Create T5-style relative position bias as fallback/complement.

    This can be used when lineage information is not available or as a
    learned complement to lineage bias.

    Args:
        seq_len: Sequence length
        max_distance: Maximum relative distance to consider
        num_buckets: Number of buckets for bucketing distances

    Returns:
        Bias matrix of shape [seq_len, seq_len]
    """
    # Compute relative positions
    positions = torch.arange(seq_len)
    rel_pos = positions[:, None] - positions[None, :]  # [L, L]

    # Bucket the distances
    abs_pos = torch.abs(rel_pos)

    # T5 bucketing scheme
    is_small = abs_pos < max_distance // 2
    max_exact = max_distance // 2

    # Small distances get exact buckets
    small_buckets = abs_pos[is_small]

    # Large distances get logarithmic buckets
    large_mask = ~is_small
    large_pos = abs_pos[large_mask]
    large_buckets = (
        max_exact
        + torch.log(large_pos.float() / max_exact)
        / torch.log(torch.tensor((seq_len - 1) / max_exact))
        * (num_buckets - max_exact)
    ).long()
    large_buckets = torch.clamp(large_buckets, max=num_buckets - 1)

    # Combine
    buckets = torch.zeros_like(abs_pos, dtype=torch.long)
    buckets[is_small] = small_buckets
    buckets[large_mask] = large_buckets

    # Convert buckets to bias (will be learned, this is just the structure)
    # For now, return relative distances
    return -abs_pos.float()


# Pre-computed lineage distances for common C. elegans cells
# This can be used for validation and testing
LINEAGE_CACHE: dict[tuple[str, str], int] = {}


def cached_lineage_distance(name1: str, name2: str) -> int:
    """Cached version of lineage_distance for repeated queries."""
    key = (name1, name2) if name1 < name2 else (name2, name1)
    if key not in LINEAGE_CACHE:
        LINEAGE_CACHE[key] = lineage_distance(name1, name2)
    return LINEAGE_CACHE[key]


def compute_lineage_similarity_matrix(
    lineage_names: list[str],
    kernel: str = "gaussian",
    sigma: float = 2.0,
) -> torch.Tensor:
    """Compute similarity matrix based on lineage distances.

    Alternative to bias that produces positive similarities.
    Can be used for graph neural networks or as attention weights directly.

    Args:
        lineage_names: List of cell lineage names
        kernel: Kernel type ("gaussian" or "exponential")
        sigma: Bandwidth parameter for the kernel

    Returns:
        Similarity matrix of shape [n_cells, n_cells] where
        similarity[i,j] = exp(-distance(i,j)^2 / (2*sigma^2)) for gaussian
        similarity[i,j] = exp(-distance(i,j) / sigma) for exponential
    """
    n = len(lineage_names)
    distances = torch.zeros(n, n)

    for i in range(n):
        for j in range(n):
            distances[i, j] = lineage_distance(lineage_names[i], lineage_names[j])

    if kernel == "gaussian":
        similarity = torch.exp(-distances.pow(2) / (2 * sigma ** 2))
    elif kernel == "exponential":
        similarity = torch.exp(-distances / sigma)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    return similarity
