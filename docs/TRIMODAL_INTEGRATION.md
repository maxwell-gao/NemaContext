# Trimodal Integration: Transcriptome + Spatial + Lineage

This document describes the three-modality integration architecture for NemaContext.

## Overview

The trimodal architecture combines:
1. **Transcriptome**: Gene expression (2000 highly variable genes)
2. **Spatial**: 3D coordinates (x, y, z)
3. **Lineage**: Developmental tree structure (via attention bias)

## Key Challenge: Partial Observations

Only ~0.7% of cells have all three modalities:
- Total cells: 234,888
- With transcriptome: 100% (Large2025)
- With spatial: ~1.4% (WormGUIDES overlap)
- With lineage: ~7% (WormBase annotation)
- Complete trimodal: 1,626 cells

## Solution: Modality Masking

Instead of discarding incomplete cells, we use **learned masking**:

```python
# Modality bitmask (1=transcriptome, 2=spatial, 4=lineage)
modality_mask = torch.tensor([1, 1, 0])  # Has genes + spatial, no lineage
```

Loss functions weight each modality by availability:
```python
loss = (
    gene_mask * gene_loss +
    spatial_mask * spatial_loss +
    lineage_mask * lineage_loss
) / total_available
```

## Architecture

### Data Pipeline

```
Extended AnnData (234k cells)
    ↓
TrimodalDataset
    ├── Modality mask extraction
    ├── HVG selection (2000 genes)
    ├── Spatial normalization
    └── Time binning (10 bins)
    ↓
SampleState with:
    ├── continuous: [2000 genes + 3 spatial = 2003 dim]
    ├── discrete: founder_id
    ├── modality_masks: [N, 3]
    └── lineage_names: for bias computation
```

### Model

```
NemaFlowModel (unchanged)
    ├── continuous_proj: Linear(2003 → d_model)
    ├── discrete_embed: Embedding(7 → d_model)
    ├── RoPE positional encoding
    ├── adaLN-Zero transformer blocks
    └── Output heads for flow prediction
```

### Loss Functions

| Loss | Purpose | Weight |
|------|---------|--------|
| Sinkhorn divergence | Distribution matching | 1.0 |
| Cell count | Track cell numbers | 0.1 |
| Diversity | Prevent mode collapse | 0.01 |
| Weak anchor | Curriculum guidance | 0.1 → 0 |

## Curriculum Learning

Three-phase training strategy:

| Phase | Epochs | Focus | Data |
|-------|--------|-------|------|
| 1 | 1-15 | Spatial dynamics | All cells with spatial |
| 2 | 16-30 | Gene regulation | All cells with transcriptome |
| 3 | 31+ | Joint integration | All cells (masked) |

## Usage

### Quick Start

```bash
# Verify implementation
uv run python examples/verify_trimodal.py

# Train with curriculum
uv run python examples/train_trimodal.py \
    --epochs 50 \
    --batch_size 8 \
    --curriculum phased \
    --phase1_epochs 15 \
    --phase2_epochs 15 \
    --device cuda

# Train without curriculum
uv run python examples/train_trimodal.py \
    --epochs 50 \
    --curriculum none \
    --device cuda
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--h5ad_path` | `nema_extended_large2025.h5ad` | Extended dataset |
| `--n_hvg` | 2000 | Number of HVGs |
| `--d_model` | 256 | Model dimension |
| `--n_layers` | 6 | Transformer layers |
| `--curriculum` | `phased` | Training curriculum |
| `--use_lineage_bias` | True | Enable lineage attention |

## File Structure

```
src/branching_flows/
    ├── trimodal_dataset.py    # TrimodalDataset class
    ├── trimodal_loss.py       # Masked loss functions
    └── __init__.py            # Exports

examples/
    ├── train_trimodal.py      # Training script
    └── verify_trimodal.py     # Verification
```

## Performance Notes

- **Memory**: ~2GB for batch_size=8, d_model=256
- **Speed**: ~7s/epoch on CPU, ~1s/epoch on GPU
- **Convergence**: 50 epochs sufficient for baseline

## Future Work

1. **Full embryo scale**: 500+ cells with gradient checkpointing
2. **Cross-modal attention**: Direct gene-spatial interactions
3. **Perturbation studies**: Delete/modify lineages and observe
4. **Novel trajectory generation**: Cells not in training set

## References

- Phase 4 Findings: `docs/PHASE4_FINDINGS.md`
- Original Plan: `/home/max/.claude/plans/iridescent-hopping-kitten.md`
