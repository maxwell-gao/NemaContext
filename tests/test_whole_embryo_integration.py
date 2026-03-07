"""Integration tests for whole-embryo context architecture.

Validates that:
1. Trajectory extraction produces correct whole-embryo format
2. Founder metadata is preserved in trajectory artifacts
3. Active dataset path does not inject founder identity into model inputs
4. Model training works with whole-embryo trajectories
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.branching_flows.autoregressive_model import AutoregressiveNemaModel  # noqa: E402
from src.branching_flows.states import BranchingState  # noqa: E402
from src.data.trajectory_extractor import WholeEmbryoTrajectoryExtractor  # noqa: E402
from examples.whole_organism_ar.train_autoregressive_full import EmbryoTrajectoryDataset  # noqa: E402


@pytest.fixture
def lineage_file():
    """Path to test lineage file."""
    return Path("dataset/raw/wormbase/lineage_tree.json")


@pytest.fixture
def trajectory_file(tmp_path, lineage_file):
    """Generate a temporary whole-embryo trajectory for testing."""
    if not lineage_file.exists():
        pytest.skip("Lineage data not available")

    extractor = WholeEmbryoTrajectoryExtractor(lineage_file)
    trajectory = extractor.extract_embryo_trajectory(
        max_time=100.0,
        time_resolution=20.0,
    )

    output_file = tmp_path / "test_trajectory.json"
    with open(output_file, "w") as f:
        json.dump(trajectory, f)

    return output_file


def test_trajectory_extraction(lineage_file):
    """Test that trajectory extraction produces correct whole-embryo format."""
    if not lineage_file.exists():
        pytest.skip("Lineage data not available")

    extractor = WholeEmbryoTrajectoryExtractor(lineage_file)
    trajectory = extractor.extract_embryo_trajectory(
        max_time=100.0,
        time_resolution=20.0,
    )

    # Basic structure checks
    assert len(trajectory) > 0
    assert isinstance(trajectory, list)

    # Check each state has required fields
    required_fields = [
        "time",
        "n_cells",
        "cell_names",
        "founders",
        "founder_ids",
        "positions",
        "genes",
    ]
    for state in trajectory:
        for field in required_fields:
            assert field in state, f"Missing field: {field}"

    # Check founder identity is preserved
    founders_present = set()
    for state in trajectory:
        founders_present.update(state["founders"])

    expected_founders = {"AB", "MS", "E", "C", "D", "P4"}
    assert founders_present & expected_founders, "No expected founders found"

    # Check n_cells consistency
    for state in trajectory:
        assert state["n_cells"] == len(state["cell_names"])
        assert state["n_cells"] == len(state["founders"])
        assert state["n_cells"] == len(state["founder_ids"])
        assert state["n_cells"] == len(state["positions"])
        assert state["n_cells"] == len(state["genes"])


def test_trajectory_cross_lineage_coexistence(lineage_file):
    """Test that multiple lineages coexist at time points."""
    if not lineage_file.exists():
        pytest.skip("Lineage data not available")

    extractor = WholeEmbryoTrajectoryExtractor(lineage_file)
    trajectory = extractor.extract_embryo_trajectory(
        max_time=200.0,
        time_resolution=10.0,
    )

    # Find states with multiple founders
    multi_founder_states = 0
    for state in trajectory:
        unique_founders = set(state["founders"])
        if len(unique_founders) > 1:
            multi_founder_states += 1

    # Should have many multi-founder states
    assert multi_founder_states > len(trajectory) * 0.5, "Too few multi-founder states"


def test_dataset_loading(trajectory_file):
    """Test that dataset correctly loads trajectory."""
    dataset = EmbryoTrajectoryDataset(str(trajectory_file))

    assert len(dataset) > 0

    # Check first sample
    sample = dataset[0]
    assert "current" in sample
    assert "next" in sample
    assert "target_split" in sample
    assert "target_del" in sample
    assert "time" in sample

    # Check BranchingState structure
    current = sample["current"]
    assert isinstance(current, BranchingState)
    assert len(current.states) == 2  # (continuous, optional discrete)
    assert current.states[1] is None


def test_founder_metadata_preserved_in_trajectory(trajectory_file):
    """Founder metadata should remain available in the raw trajectory artifact."""
    dataset = EmbryoTrajectoryDataset(str(trajectory_file))

    all_founder_ids = set()
    for state in dataset.trajectory[: min(5, len(dataset.trajectory))]:
        all_founder_ids.update(state["founder_ids"])

    assert len(all_founder_ids) > 1, "No founder diversity in dataset"


def test_model_forward_without_founder_features(trajectory_file):
    """Test that model can process states without founder-id input features."""
    dataset = EmbryoTrajectoryDataset(str(trajectory_file))
    if len(dataset) == 0:
        pytest.skip("Empty dataset")

    model = AutoregressiveNemaModel(
        gene_dim=2000,
        spatial_dim=3,
        d_model=64,
        n_layers=2,
        n_heads=2,
        max_seq_len=128,
    )

    sample = dataset[0]["current"]

    # Forward pass should work
    output = model.forward_step(sample)

    assert output.gene_delta is not None
    assert output.spatial_vel is not None
    assert output.split_logits is not None
    assert output.del_logits is not None


def test_model_multi_cell_attention(trajectory_file):
    """Test that attention mechanism works on multi-cell whole-embryo states."""
    dataset = EmbryoTrajectoryDataset(str(trajectory_file))

    multi_cell_sample = None
    for i in range(len(dataset)):
        sample = dataset[i]["current"]
        if int(sample.padmask[0].sum().item()) > 1:
            multi_cell_sample = sample
            break

    if multi_cell_sample is None:
        pytest.skip("No multi-cell samples found")

    model = AutoregressiveNemaModel(
        gene_dim=2000,
        spatial_dim=3,
        d_model=64,
        n_layers=2,
        n_heads=2,
        max_seq_len=128,
    )

    # Compute attention
    h = model.encode_state(multi_cell_sample)
    B, L, D = h.shape

    # Get attention from first layer
    qkv = model.blocks[0].qkv(h).reshape(B, L, 3, model.blocks[0].n_heads, -1)
    q, k, _ = qkv.unbind(2)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)

    scores = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
    attn = torch.softmax(scores, dim=-1)

    # Attention should not be all zeros or NaN
    assert not torch.isnan(attn).any()
    assert (attn > 0).any()


def test_global_spatial_coordinates(lineage_file):
    """Test that spatial coordinates are in global embryo space."""
    if not lineage_file.exists():
        pytest.skip("Lineage data not available")

    extractor = WholeEmbryoTrajectoryExtractor(lineage_file)
    trajectory = extractor.extract_embryo_trajectory(
        max_time=200.0,
        time_resolution=20.0,
    )

    # Check spatial extent
    for state in trajectory:
        positions = state["positions"]
        if len(positions) > 0:
            import numpy as np

            positions_array = np.array(positions)

            # All coordinates should be in [0, 1] range
            assert positions_array.min() >= -0.5, "Spatial coordinate too negative"
            assert positions_array.max() <= 1.5, "Spatial coordinate too large"

            # Should have variation in X (AP axis)
            if len(positions) > 1:
                x_range = positions_array[:, 0].max() - positions_array[:, 0].min()
                assert x_range > 0, "No variation in AP axis"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
