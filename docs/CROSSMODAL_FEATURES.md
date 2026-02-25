# Cross-Modal Attention & Data Augmentation

This document describes the enhanced trimodal model with explicit cross-modal attention and spatial data augmentation.

## Overview

The cross-modal model improves upon the baseline trimodal architecture by:
1. **Explicit cross-modal attention**: Genes and spatial coordinates attend to each other
2. **Parallel processing streams**: Separate pathways for each modality with periodic fusion
3. **Spatial data augmentation**: Rotation, flip, and scaling to improve generalization

## Architecture

### Cross-Modal Fusion

```
Input: Genes [B, L, D_g] + Spatial [B, L, D_s]
    ↓
Separate projections:
    Gene stream: [B, L, d_model/2]
    Spatial stream: [B, L, d_model/2]
    ↓
Parallel Transformer blocks (self-attention)
    ↓ (every N layers)
Cross-Modal Fusion:
    Genes query Spatial: "Given my gene expression, where should I be?"
    Spatial query Genes: "Given my location, what should I express?"
    ↓
Concatenate and project
    ↓
Output heads: gene_pred, spatial_pred, discrete, split, del
```

### Cross-Modal Attention Mechanism

```python
# Genes attend to spatial context
attn_genes_to_spatial = softmax(Q_genes @ K_spatial^T)
gene_update = attn_genes_to_spatial @ V_spatial

# Spatial attends to gene context
attn_spatial_to_genes = softmax(Q_spatial @ K_genes^T)
spatial_update = attn_spatial_to_genes @ V_genes
```

This allows:
- **Gene→Spatial**: "Muscle genes → likely at periphery"
- **Spatial→Genes**: "Anterior position → likely neuronal genes"

## Data Augmentation

### Spatial Transformations

| Augmentation | Description | Probability |
|--------------|-------------|-------------|
| **Rotation** | Random rotation around z-axis (0-2π) | 50% |
| **Flip X** | Mirror along x-axis | 50% |
| **Flip Y** | Mirror along y-axis | 50% |
| **Flip Z** | Mirror along z-axis | 50% |
| **Scale** | Uniform scaling ±10% | Always |

### Biological Rationale

- **Rotation**: Microscope orientation is arbitrary
- **Flip**: Embryo can be imaged from different angles
- **Scale**: Slight variations in imaging magnification

All augmentations preserve:
- Relative cell positions (within the embryo)
- Gene expression (unaffected by spatial transforms)

## Usage

### Training with Cross-Modal Model

```bash
# Basic training
uv run python examples/train_trimodal_crossmodal.py \
    --device cuda \
    --epochs 50 \
    --batch_size 8

# With custom cross-modal frequency
uv run python examples/train_trimodal_crossmodal.py \
    --cross_modal_every 1  # Fuse every layer

# Disable augmentation
uv run python examples/train_trimodal_crossmodal.py \
    --no_augment

# Custom augmentation settings
uv run python examples/train_trimodal_crossmodal.py \
    --aug_rotation \
    --aug_flip \
    --aug_scale 0.15
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--cross_modal_every` | 2 | Cross-attention frequency (layers) |
| `--augment_spatial` | True | Enable data augmentation |
| `--aug_rotation` | True | Enable rotation augmentation |
| `--aug_flip` | True | Enable flip augmentation |
| `--aug_scale` | 0.1 | Scaling factor (±10%) |

## Model Comparison

| Feature | Baseline | Cross-Modal |
|---------|----------|-------------|
| Input projection | Single (2003→d_model) | Separate (genes + spatial) |
| Modality interaction | Implicit (concatenation) | Explicit (cross-attention) |
| Parameters | 8.3M | 5.1M (half-dim streams) |
| Data augmentation | ❌ | ✅ |
| Gene→Spatial prediction | Indirect | Direct |
| Spatial→Gene prediction | Indirect | Direct |

## Performance Expectations

### Advantages
1. **Better cross-modal predictions**: Gene expression more accurately predicts spatial location
2. **Improved generalization**: Data augmentation reduces overfitting
3. **Interpretability**: Can analyze cross-attention weights
4. **Efficiency**: Fewer parameters (5.1M vs 8.3M)

### Trade-offs
1. **Training time**: ~10% slower due to cross-attention
2. **Memory**: Slightly higher (parallel streams)
3. **Hyperparameters**: Additional tuning for augmentation strength

## Verification

```bash
# Test the implementation
uv run python examples/verify_crossmodal.py
```

Expected output:
```
✅ CrossModalFusion works!
✅ CrossModalNemaModel works!
✅ Data augmentation is working!
```

## Implementation Details

### CrossModalFusion Layer

Located in: `src/branching_flows/crossmodal_model.py`

Key components:
- `gene_to_spatial_q/kv`: Projections for gene→spatial attention
- `spatial_to_gene_q/kv`: Projections for spatial→gene attention
- `gene_out/spatial_out`: Output projections with residual

### Data Augmentation

Located in: `src/branching_flows/trimodal_dataset.py`

Method: `_augment_continuous()`
- Operates on normalized spatial coordinates
- Denormalizes → transforms → renormalizes
- Preserves gene expression unchanged

## Future Extensions

1. **Learned augmentation**: Train augmentation policy
2. **Multi-scale fusion**: Cross-attention at different resolutions
3. **Modality dropout**: Randomly drop modalities during training
4. **Attention visualization**: Plot cross-attention maps

## References

- Base trimodal: `docs/TRIMODAL_INTEGRATION.md`
- Training script: `examples/train_trimodal_crossmodal.py`
- Model definition: `src/branching_flows/crossmodal_model.py`
