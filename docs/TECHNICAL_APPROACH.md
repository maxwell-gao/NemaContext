# NemaContext Technical Approach: Flow Matching Transformers for Contact Graph Generation

## The Bitter Lesson and Architectural Philosophy

> *"The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective, and by a large margin."*  
> — Rich Sutton, "The Bitter Lesson" (2019)

This document outlines the technical approach for NemaContext, deliberately choosing **scalable, general-purpose architectures** over domain-specific graph neural networks. We adopt a **Flow Matching Transformer** paradigm that treats contact graph prediction as a **conditional generative modeling** problem.

### Why Not GNNs?

While Graph Neural Networks (GNNs) are the intuitive choice for graph-structured data, they suffer from fundamental limitations that conflict with The Bitter Lesson:

| GNN Limitation | Consequence for NemaContext |
|----------------|----------------------------|
| **Message passing is local** | Cannot capture long-range developmental dependencies |
| **Over-smoothing** | Deep GNNs collapse node representations |
| **Fixed topology assumption** | Growing graphs (cell division) require ad-hoc extensions |
| **Inductive bias overhead** | Hand-crafted graph construction (Delaunay, k-NN) introduces researcher bias |
| **Poor scalability** | Sparse operations don't leverage modern GPU tensor cores efficiently |

**The Bitter Lesson teaches us**: Instead of encoding biological knowledge into architecture (GNN topology, lineage priors), we should let the model **learn** these relationships from data using general, scalable methods.

---

## Proposed Architecture: Flow Matching Transformer (FMT)

### Core Insight

We reformulate contact graph prediction as:

> **Given**: Cell embeddings (transcriptome + lineage encoding)  
> **Generate**: The adjacency matrix of physical contacts

This is a **conditional generative modeling** problem. Flow Matching provides a principled framework for learning to generate structured outputs.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Flow Matching Transformer                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  Cell Tokens │    │   Pairwise   │    │    Flow      │       │
│  │  (N × d)     │───▶│  Transformer │───▶│   Matching   │       │
│  │              │    │   Encoder    │    │    Head      │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ Transcriptome│    │  All-Pairs   │    │  Adjacency   │       │
│  │ + Lineage    │    │  Attention   │    │   Matrix     │       │
│  │ + Time       │    │  (N² pairs)  │    │   A ∈ {0,1}  │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. Cell Tokenization

Each cell is represented as a token with the following components:

### 1.1 Transcriptome Embedding

Instead of raw gene counts, we use **foundation model embeddings**:

```python
# Option A: scGPT embedding (768-dim)
cell_embed = scGPT.encode(gene_expression)  # [N, 768]

# Option B: scFoundation / Geneformer
cell_embed = Geneformer.encode(gene_expression)  # [N, 512]

# Option C: Lightweight - PCA + MLP
cell_embed = MLP(PCA(gene_expression, n_components=256))  # [N, 256]
```

### 1.2 Lineage Positional Encoding

The invariant C. elegans lineage provides a natural **positional encoding**:

```python
def lineage_to_binary(lineage_name: str) -> torch.Tensor:
    """
    Convert lineage name to binary path encoding.
    
    Example: "ABplp" -> [0, 1, 0, 1, 0, 0, 0, ...]
             A=0, B=1, p=0, l=1, ...
    
    Returns: Fixed-length binary vector [max_depth]
    """
    encoding = []
    for char in lineage_name:
        if char in ['A', 'a', 'l', 'p']:  # anterior/left
            encoding.append(0)
        elif char in ['B', 'E', 'r', 'd']:  # posterior/right
            encoding.append(1)
        # ... handle other characters
    return pad_to_length(encoding, max_depth=16)
```

### 1.3 Temporal Encoding

Developmental time as sinusoidal encoding (Transformer-style):

```python
def time_encoding(t: float, d_model: int) -> torch.Tensor:
    """
    Sinusoidal encoding of developmental time (0-830 minutes).
    """
    pe = torch.zeros(d_model)
    for i in range(0, d_model, 2):
        pe[i] = math.sin(t / (10000 ** (i / d_model)))
        pe[i+1] = math.cos(t / (10000 ** (i / d_model)))
    return pe
```

### 1.4 Complete Token

```python
cell_token = torch.cat([
    transcriptome_embed,   # [d_expr]
    lineage_encoding,      # [d_lin]
    time_encoding,         # [d_time]
    morphology_features,   # [d_morph] (volume, surface, sphericity)
], dim=-1)  # Total: [d_model]
```

---

## 2. Pairwise Transformer Encoder

### 2.1 The Pairwise Attention Problem

Contact prediction requires reasoning about **pairs of cells**. Standard self-attention over N cells gives N² attention weights, but we need to **output** N² contact predictions.

**Solution**: Treat the problem as sequence-to-sequence where:
- Input: N cell tokens
- Output: N² pairwise relationship tokens

### 2.2 Architecture: Axial Attention

We use **Axial Attention** (from Axial Transformers) to efficiently compute pairwise representations:

```python
class PairwiseTransformer(nn.Module):
    def __init__(self, d_model, n_heads, n_layers):
        self.cell_encoder = TransformerEncoder(d_model, n_heads, n_layers // 2)
        self.pair_encoder = AxialTransformerEncoder(d_model, n_heads, n_layers // 2)
    
    def forward(self, cell_tokens):
        # cell_tokens: [B, N, d_model]
        
        # Step 1: Encode individual cells
        cell_repr = self.cell_encoder(cell_tokens)  # [B, N, d_model]
        
        # Step 2: Create pairwise representations
        # Outer product: [B, N, d] × [B, N, d] -> [B, N, N, 2d]
        pair_repr = torch.cat([
            cell_repr.unsqueeze(2).expand(-1, -1, N, -1),  # [B, N, N, d]
            cell_repr.unsqueeze(1).expand(-1, N, -1, -1),  # [B, N, N, d]
        ], dim=-1)
        
        # Step 3: Axial attention over pairs
        # Row-wise attention: each cell attends to all its potential partners
        # Column-wise attention: each potential partner attends to all cells
        pair_repr = self.pair_encoder(pair_repr)  # [B, N, N, d_model]
        
        return pair_repr
```

### 2.3 Computational Complexity

| Method | Complexity | Notes |
|--------|------------|-------|
| Full pairwise attention | O(N⁴) | Infeasible |
| Axial attention | O(N³) | Tractable for N ≤ 2000 |
| Sparse attention (BigBird) | O(N² log N) | For larger N |
| Flash Attention | O(N²) | Memory-efficient |

For C. elegans embryo (N ≈ 1000 cells at terminal stage), **Axial Attention with Flash Attention** is computationally feasible on A100 GPUs.

---

## 3. Flow Matching for Graph Generation

### 3.1 Why Flow Matching?

Flow Matching is a modern generative modeling framework that:
- Provides **deterministic sampling** (faster than diffusion)
- Has **stable training** (no score matching issues)
- Naturally handles **structured outputs** (like adjacency matrices)
- Scales better than autoregressive models

### 3.2 Formulation

Let **A** ∈ {0,1}^{N×N} be the target contact adjacency matrix.

We learn a **conditional flow** that transforms noise **Z** ~ N(0,1) to **A**:

```
Z ───[Flow ψ_t]───▶ A
     conditioned on
     (transcriptome, lineage, time)
```

The flow is parameterized by a neural network that predicts the **velocity field**:

```python
v_θ(A_t, t, condition) = ∂A_t/∂t
```

### 3.3 Training Objective

**Optimal Transport Conditional Flow Matching (OT-CFM)**:

```python
def flow_matching_loss(model, A_target, condition):
    """
    A_target: [B, N, N] ground truth adjacency matrix
    condition: [B, N, d_model] cell embeddings
    """
    # Sample random time
    t = torch.rand(B, 1, 1)
    
    # Sample noise
    A_0 = torch.randn_like(A_target)
    
    # Interpolate (OT path)
    A_t = (1 - t) * A_0 + t * A_target
    
    # Target velocity (OT velocity)
    v_target = A_target - A_0
    
    # Predicted velocity
    v_pred = model(A_t, t, condition)
    
    # MSE loss
    loss = F.mse_loss(v_pred, v_target)
    
    return loss
```

### 3.4 Sampling (Inference)

```python
@torch.no_grad()
def sample_contact_graph(model, condition, n_steps=50):
    """
    Generate contact graph given cell embeddings.
    """
    N = condition.shape[1]
    
    # Start from noise
    A_t = torch.randn(1, N, N)
    
    # Euler integration
    dt = 1.0 / n_steps
    for t in torch.linspace(0, 1, n_steps):
        v = model(A_t, t, condition)
        A_t = A_t + v * dt
    
    # Binarize
    A_final = (A_t > 0.5).float()
    
    # Enforce symmetry
    A_final = (A_final + A_final.T) / 2
    A_final = (A_final > 0.5).float()
    
    return A_final
```

---

## 4. Complete Model Architecture

```python
class ContactFlowTransformer(nn.Module):
    """
    Flow Matching Transformer for Contact Graph Generation.
    """
    def __init__(
        self,
        d_model=512,
        n_heads=8,
        n_encoder_layers=6,
        n_flow_layers=6,
        max_cells=1024,
    ):
        super().__init__()
        
        # Cell tokenizer
        self.expr_proj = nn.Linear(768, d_model // 2)  # scGPT embedding
        self.lineage_embed = nn.Embedding(2, d_model // 8)  # Binary lineage
        self.time_embed = SinusoidalPositionalEncoding(d_model // 4)
        self.morph_proj = nn.Linear(4, d_model // 8)  # Volume, surface, sphericity, etc.
        
        # Pairwise encoder
        self.pairwise_encoder = PairwiseTransformer(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_encoder_layers,
        )
        
        # Flow model (predicts velocity)
        self.flow_net = nn.Sequential(
            nn.Linear(d_model + 1 + 1, d_model),  # +1 for A_t value, +1 for time
            *[TransformerBlock(d_model, n_heads) for _ in range(n_flow_layers)],
            nn.Linear(d_model, 1),  # Output: velocity for each pair
        )
        
        # Time embedding for flow
        self.flow_time_embed = nn.Sequential(
            SinusoidalPositionalEncoding(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
        )
    
    def encode_cells(self, transcriptome, lineage, time, morphology):
        """
        Encode all cells into tokens.
        
        Args:
            transcriptome: [B, N, 768] scGPT embeddings
            lineage: [B, N, 16] binary lineage encoding
            time: [B, N] developmental time
            morphology: [B, N, 4] morphological features
        
        Returns:
            cell_tokens: [B, N, d_model]
        """
        expr_embed = self.expr_proj(transcriptome)
        lin_embed = self.lineage_embed(lineage).sum(dim=-2)  # Sum over lineage depth
        time_embed = self.time_embed(time)
        morph_embed = self.morph_proj(morphology)
        
        cell_tokens = torch.cat([expr_embed, lin_embed, time_embed, morph_embed], dim=-1)
        return cell_tokens
    
    def forward(self, A_t, t, transcriptome, lineage, dev_time, morphology):
        """
        Predict velocity field for flow matching.
        
        Args:
            A_t: [B, N, N] noisy adjacency matrix at time t
            t: [B, 1] flow time (0 to 1)
            transcriptome, lineage, dev_time, morphology: cell features
        
        Returns:
            v: [B, N, N] predicted velocity
        """
        B, N, _ = A_t.shape
        
        # Encode cells
        cell_tokens = self.encode_cells(transcriptome, lineage, dev_time, morphology)
        
        # Get pairwise representations
        pair_repr = self.pairwise_encoder(cell_tokens)  # [B, N, N, d_model]
        
        # Embed flow time
        t_embed = self.flow_time_embed(t)  # [B, d_model]
        t_embed = t_embed.unsqueeze(1).unsqueeze(1).expand(-1, N, N, -1)
        
        # Concatenate A_t value and time
        A_t_expanded = A_t.unsqueeze(-1)  # [B, N, N, 1]
        t_scalar = t.unsqueeze(-1).unsqueeze(-1).expand(-1, N, N, 1)
        
        flow_input = torch.cat([pair_repr, A_t_expanded, t_scalar], dim=-1)
        
        # Predict velocity
        v = self.flow_net(flow_input).squeeze(-1)  # [B, N, N]
        
        return v
```

---

## 5. Training Strategy

### 5.1 Data Preparation

```python
def prepare_training_data(adata):
    """
    Prepare training batches from AnnData.
    
    Returns batches of (cells, contact_graph) for early embryo stages.
    """
    # Filter to CShaper-covered time range (20-380 min)
    early_mask = adata.obs['time_minutes'] <= 380
    early_data = adata[early_mask]
    
    # Group by developmental stage (e.g., 50-cell, 100-cell, 200-cell, etc.)
    stages = early_data.obs['stage'].unique()
    
    batches = []
    for stage in stages:
        stage_data = early_data[early_data.obs['stage'] == stage]
        
        # Get features
        transcriptome = get_scgpt_embedding(stage_data)
        lineage = encode_lineage(stage_data.obs['lineage'])
        time = stage_data.obs['time_minutes'].values
        morphology = stage_data.obsm['X_morphology']
        
        # Get ground truth contact graph
        contact = stage_data.obsp['contact_binary'].toarray()
        
        batches.append({
            'transcriptome': transcriptome,
            'lineage': lineage,
            'time': time,
            'morphology': morphology,
            'contact': contact,
        })
    
    return batches
```

### 5.2 Training Loop

```python
def train_flow_matching(model, train_data, epochs=100, lr=1e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    for epoch in range(epochs):
        for batch in train_data:
            optimizer.zero_grad()
            
            # Unpack batch
            A_target = batch['contact']
            transcriptome = batch['transcriptome']
            lineage = batch['lineage']
            time = batch['time']
            morphology = batch['morphology']
            
            # Flow matching loss
            loss = flow_matching_loss(
                model, A_target,
                transcriptome, lineage, time, morphology
            )
            
            loss.backward()
            optimizer.step()
        
        scheduler.step()
```

### 5.3 Curriculum Learning

Train progressively on increasing embryo sizes:

```
Stage 1: 4-50 cells (simple topology, quick convergence)
Stage 2: 50-200 cells (gastrulation, complex rearrangements)
Stage 3: 200-500 cells (organogenesis)
Stage 4: 500-1000 cells (terminal differentiation)
```

---

## 6. Inference: Predicting Late-Stage Contacts

### 6.1 Temporal Extrapolation

```python
def predict_late_stage_contacts(model, late_stage_adata):
    """
    Predict contact graphs for 380-830 min embryos.
    """
    # Encode late-stage cells (transcriptome is available!)
    transcriptome = get_scgpt_embedding(late_stage_adata)
    lineage = encode_lineage(late_stage_adata.obs['lineage'])
    time = late_stage_adata.obs['time_minutes'].values
    
    # Morphology is MISSING - use imputed or learned defaults
    morphology = impute_morphology(late_stage_adata)
    
    # Generate contact graph
    predicted_contact = sample_contact_graph(
        model,
        transcriptome, lineage, time, morphology,
        n_steps=100  # More steps for better quality
    )
    
    return predicted_contact
```

### 6.2 Uncertainty Quantification

Generate multiple samples and compute variance:

```python
def predict_with_uncertainty(model, adata, n_samples=10):
    """
    Generate multiple contact graph samples for uncertainty estimation.
    """
    samples = []
    for _ in range(n_samples):
        sample = sample_contact_graph(model, adata)
        samples.append(sample)
    
    samples = torch.stack(samples)
    
    # Mean prediction
    mean_contact = samples.mean(dim=0)
    
    # Uncertainty (std)
    uncertainty = samples.std(dim=0)
    
    # High-confidence edges
    confident_edges = (mean_contact > 0.9) | (mean_contact < 0.1)
    
    return mean_contact, uncertainty, confident_edges
```

---

## 7. Validation Strategy

### 7.1 Held-Out Early Stages

```python
def cross_validate(model, all_stages):
    """
    Leave-one-stage-out cross-validation on early embryo.
    """
    results = []
    for held_out_stage in all_stages:
        # Train on other stages
        train_stages = [s for s in all_stages if s != held_out_stage]
        train_model(model, train_stages)
        
        # Evaluate on held-out
        predicted = predict_contacts(model, held_out_stage)
        ground_truth = get_cshaper_contacts(held_out_stage)
        
        auc = compute_auroc(predicted, ground_truth)
        ap = compute_average_precision(predicted, ground_truth)
        
        results.append({'stage': held_out_stage, 'AUC': auc, 'AP': ap})
    
    return results
```

### 7.2 Biological Consistency (Late Stage)

```python
def validate_notch_signaling(predicted_contacts, expression_data):
    """
    Check if predicted neighbors have complementary Notch L-R pairs.
    """
    # Known Notch L-R pairs in C. elegans
    notch_pairs = [
        ('glp-1', 'apx-1'),
        ('lin-12', 'lag-2'),
        ('lin-12', 'dsl-1'),
    ]
    
    consistent = 0
    total = 0
    
    for receptor, ligand in notch_pairs:
        # Find cells expressing receptor
        receptor_cells = expression_data[:, receptor] > threshold
        
        # Find their predicted neighbors
        for cell_idx in receptor_cells.nonzero():
            neighbors = predicted_contacts[cell_idx].nonzero()
            
            # Check if any neighbor expresses ligand
            ligand_expression = expression_data[neighbors, ligand]
            if ligand_expression.max() > threshold:
                consistent += 1
            total += 1
    
    return consistent / total if total > 0 else 0
```

### 7.3 Connectome Compatibility

```python
def validate_against_connectome(predicted_neural_contacts, adult_connectome):
    """
    Check if predicted neuron contacts are compatible with adult synapses.
    """
    # Adult synapses require physical contact
    # If A synapses onto B, A must have contacted B during development
    
    synapse_pairs = get_synapse_pairs(adult_connectome)
    
    recall = 0
    for (neuron_a, neuron_b) in synapse_pairs:
        # Check if model predicts contact at any late stage
        if predicted_neural_contacts[neuron_a, neuron_b] > 0.5:
            recall += 1
    
    return recall / len(synapse_pairs)
```

---

## 8. Computational Requirements

### 8.1 Model Size

| Component | Parameters |
|-----------|------------|
| Cell Encoder | ~10M |
| Pairwise Transformer (6 layers) | ~50M |
| Flow Network (6 layers) | ~30M |
| **Total** | **~90M** |

### 8.2 Training Compute

| Batch | Cells (N) | Memory | Time/Epoch |
|-------|-----------|--------|------------|
| Early embryo | 50-200 | ~4 GB | ~1 min |
| Mid embryo | 200-500 | ~16 GB | ~5 min |
| Late embryo | 500-1000 | ~48 GB | ~15 min |

**Recommended**: A100 80GB × 1-2 GPUs, ~24 hours total training.

### 8.3 Inference

- 1000-cell embryo: ~30 seconds per sample
- With 10 samples for uncertainty: ~5 minutes

---

## 9. Advantages Over GNN Approach

| Aspect | GNN (TGN) | Flow Matching Transformer |
|--------|-----------|---------------------------|
| **Scalability** | O(E) sparse ops, poor GPU utilization | O(N²) dense ops, excellent GPU utilization |
| **Inductive bias** | Requires hand-crafted graph topology | Learns topology from data |
| **Long-range dependencies** | Limited by message passing depth | Full attention span |
| **Generative capability** | Discriminative only | Native generative model |
| **Uncertainty** | Requires MC dropout or ensembles | Natural via multiple samples |
| **The Bitter Lesson** | Encodes human knowledge | Leverages compute |

---

## 10. Summary

The **Flow Matching Transformer** approach for NemaContext:

1. **Tokenizes cells** using transcriptome embeddings, lineage encoding, and temporal position
2. **Learns pairwise relationships** via Axial Attention without hand-crafted graph structures
3. **Generates contact graphs** as structured outputs using Flow Matching
4. **Scales efficiently** with modern GPU hardware and attention optimizations
5. **Validates biologically** through Notch signaling and connectome consistency

This architecture embodies The Bitter Lesson: instead of encoding developmental biology knowledge into graph structures, we let a general-purpose Transformer learn the rules of embryonic spatial organization directly from data.

---

## References

1. Sutton, R. (2019). *The Bitter Lesson*. http://www.incompleteideas.net/IncIdeas/BitterLesson.html
2. Lipman, Y., et al. (2023). *Flow Matching for Generative Modeling*. ICLR.
3. Ho, J., et al. (2019). *Axial Attention in Multidimensional Transformers*. arXiv.
4. Cui, H., et al. (2024). *scGPT: Building the Foundation Model for Single-Cell Multi-omics*. Nature Methods.
5. Tong, A., et al. (2024). *Simulation-Free Schrödinger Bridges via Score and Flow Matching*. ICLR.
