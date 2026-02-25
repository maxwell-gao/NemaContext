# Emergent Context Implementation

This document describes the implementation of the Emergent Context architecture for NemaContext.

## Overview

The Emergent Context architecture shifts from **imposed tree structure** (hardcoded Sulston tree as training targets) to **discovered organization** (tree structure emerges from cellular interactions via lineage-aware attention bias).

**Key Principle**: Lineage information serves as **architectural bias** (attention modulation) rather than **training supervision** (what the model must replicate).

## Architecture Components

### 1. Lineage-Aware Attention Bias (`src/branching_flows/lineage.py`)

Computes additive attention bias from pairwise lineage distances:

```python
attn_scores = (Q @ K.T) / sqrt(d_k) + lineage_bias
```

- `parse_lineage_name()`: Parse "ABal" → ("AB", ["a", "l"])
- `lineage_distance()`: Compute tree distance between two cells
- `compute_lineage_bias()`: Convert distances to attention bias
- `batch_lineage_bias()`: Batch computation with padding

**Biological intuition**: Cells from the same lineage branch (close in tree) should attend to each other more, encoding developmental relatedness without forcing exact tree replication.

### 2. Distribution-Only Loss (`src/branching_flows/emergent_loss.py`)

Replaces per-element supervision with population-level constraints:

- `sinkhorn_divergence()`: Primary loss using optimal transport
- `cell_count_loss()`: Soft constraint on final cell count
- `diversity_loss()`: Prevents mode collapse (all cells same)
- `physics_constraints()`: Volume conservation, non-overlap (optional)
- `weak_anchor_loss()`: Optional curriculum learning support

**Critical difference**: No `X1_anchor` MSE, no `splits_target`, no `del_flags` as supervision targets.

### 3. Modified Model (`src/branching_flows/nema_model.py`)

`NemaFlowModel.forward()` now accepts optional `lineage_bias` parameter:

```python
(xc, xd), hs, hd = model(t, state, lineage_bias=lineage_bias)
```

The `AdaLNTransformerBlock` applies the bias additively to attention scores before softmax.

### 4. Training Script (`examples/train_emergent.py`)

Key features:
- `--use_lineage_bias` / `--no_lineage_bias`: Ablation support
- `--lineage_temp`: Control bias strength (lower = stronger)
- `--weak_anchor_weight`: Curriculum learning with gradual decay
- `--lambda_sinkhorn/--count/--diversity`: Loss component weights

## Usage

### Training with Lineage Bias
```bash
uv run python examples/train_emergent.py \
    --use_lineage_bias \
    --lineage_temp 1.0 \
    --epochs 30 \
    --batch_size 3
```

### Ablation without Lineage Bias
```bash
uv run python examples/train_emergent.py \
    --no_lineage_bias \
    --epochs 30
```

### Curriculum Learning
```bash
uv run python examples/train_emergent.py \
    --weak_anchor_weight 0.1 \
    --anchor_decay_epochs 10
```

## Comparison: Imposed vs Emergent

| Aspect | Imposed (BROT) | Emergent (CABF) |
|--------|---------------|-----------------|
| Tree role | Training target | Attention bias |
| Loss type | Per-element MSE | Distribution Sinkhorn |
| Model freedom | Low (must replicate) | High (match distribution) |
| Validation | Loss value | Discovered structure |
| Biological claim | "Follows real path" | "Discovers principles" |

## Success Criteria

1. **Training converges**: Loss decreases stably
2. **Distribution match**: Sinkhorn distance to real data < ε
3. **Structure emergence**: Inferred tree correlates with Sulston (r > 0.7)
4. **Biological plausibility**: Division timing, cell counts realistic
5. **Ablations work**: Removing lineage bias hurts performance

## Files Added/Modified

- `src/branching_flows/lineage.py` (new)
- `src/branching_flows/emergent_loss.py` (new)
- `src/branching_flows/nema_model.py` (modified)
- `src/branching_flows/__init__.py` (modified)
- `examples/train_emergent.py` (new)
