# NemaContext: Generative Modeling of Complete Embryo Development

**NemaContext** is a generative machine learning framework that models the complete developmental trajectory of *Caenorhabditis elegans* — from a single fertilized egg to a 959-cell adult — by treating **each cell as a token** and development as a **binary tree generation process**.

> 🆕 **Update**: Now using **Large et al. 2025** (GSE292756) as the primary transcriptome dataset with lineage-resolved annotations for >425,000 cells!

---

## 🎯 Research Positioning

### What Makes This Work Unique?

Most existing work in computational developmental biology focuses on **predicting individual cell states** or **describing static embryo atlases**. NemaContext takes a fundamentally different approach:

| Existing Approaches | NemaContext |
|---------------------|-------------|
| Predict `p(cell_state_t+1 \| cell_state_t)` | Generate `p(embryo_t+1 \| embryo_t)` |
| Single-cell or population-level | **Whole-organism generation** |
| Ignore or approximate spatial structure | Explicit spatial + lineage structure |
| Fixed cell number | **Dynamic cell division** (1 → 959 cells) |
| Implicit cell-cell interactions | Explicit GNN-based communication |

### Why C. elegans?

*C. elegans* is the **only multicellular organism** where this approach is feasible:

| Property | Advantage |
|----------|-----------|
| **Invariant lineage** | 100% deterministic cell division pattern — fully verifiable |
| **Complete spatial tracking** | WormGUIDES provides 4D nuclear positions |
| **Lineage-resolved transcriptomics** | Large et al. 2025 maps cells to lineage |
| **Small cell count** | 959 cells is computationally tractable |
| **Extensive ground truth** | Decades of developmental biology research |

### The Vision: Digital Embryogenesis

```text
Input:  Zygote state at t=0 (single cell)
        ↓
Model:  Tree-structured generative model
        ↓
Output: Complete embryo state at any time t
        ├── Every cell's transcriptome
        ├── Every cell's 3D position
        ├── Cell-cell neighbor relationships
        └── Lineage tree structure
```

---

## 🧬 Core Paradigm: Cell as Token

Unlike traditional sequence models, NemaContext models development as a **token tree**, not a linear sequence:

| Concept | Analogy |
|---------|---------|
| Each cell | A Token |
| Embryonic development | Tree generation (not autoregressive sequence) |
| Cell division | Token splits into two child tokens |
| Zygote → Adult | 1 token → 959 somatic cell tokens |

```text
t=0:   [P0]                                    # 1 token (zygote)
t=1:   [AB] [P1]                               # 2 tokens
t=2:   [ABa] [ABp] [EMS] [P2]                  # 4 tokens
...
t=N:   [959 somatic cell tokens]               # Adult
```

### What the Model Learns

1. **Division Timing**: Which tokens will divide at the next timestep?
2. **Daughter Fate**: What are the states of the two child tokens after division?
3. **Cell-Cell Communication**: How do tokens influence each other (via spatial GNN)?
4. **Long-Range Extrapolation**: Predict adult tissue distribution from early embryo tokens.

---

## 🔬 Trimodal Token Representation

Each cell-token is encoded by three modalities:

| Modality | Representation | Purpose |
|----------|----------------|---------|
| **Transcriptome** | Gene expression vector | Current cell state |
| **Lineage** | Poincaré ball embedding | Historical division path from P0 |
| **Space** | GNN-aggregated neighborhood | Environmental context & cell signaling |

---

## 📚 Related Work

### Digital Embryo & Computational Development

| Work | Year | Description | Relation to NemaContext |
|------|------|-------------|-------------------------|
| **Villoutreix et al.** | 2016 | Integrated modeling framework from digital embryo cohorts (sea urchin) | Multimodal integration; static atlas |
| **Wang et al.** | 2018 | Deep reinforcement learning for C. elegans cell movement | C. elegans + DL; explains migration, not generation |
| **Kaul et al.** | 2023 | Virtual cells recapitulate early development patterns | In silico patterning; 2D, not complete organism |
| **Romeo et al.** | 2021 | Learning developmental mode dynamics from single-cell trajectories | Trajectory inference; not generative |

### Generative Models for Single-Cell Biology

| Work | Year | Description | Relation to NemaContext |
|------|------|-------------|-------------------------|
| **scGPT** | 2024 | Transformer-based foundation model for single-cell | Generates transcriptomes; no spatial/lineage |
| **scDiffusion** | 2024 | Diffusion model for scRNA-seq generation | Generates cells; not organisms |
| **cfDiffusion** | 2025 | Classifier-free diffusion for scRNA-seq | Conditional generation; single-cell level |
| **BlastDiffusion** | 2025 | Latent diffusion for embryo images (IVF) | Image generation; not molecular states |

### Developmental Dynamics & Trajectory Modeling

| Work | Year | Description | Relation to NemaContext |
|------|------|-------------|-------------------------|
| **PRESCIENT** | 2021 | Potential field + SDE for cell fate prediction | Predicts fate; population-level |
| **Waddington-OT** | 2019 | Optimal transport for developmental trajectories | Population distribution; not individual cells |
| **TrajectoryNet** | 2020 | Neural ODE for branching trajectories | Continuous branching; not explicit tree |
| **scDiffEq** | 2025 | Neural SDE for cell state dynamics | State dynamics; inspires our SDE component |
| **Weinreb et al.** | 2020 | Lineage tracing links state to fate | Lineage + transcriptome; key inspiration |

### Hierarchical & Hyperbolic Embeddings

| Work | Year | Description | Relation to NemaContext |
|------|------|-------------|-------------------------|
| **Poincaré Embeddings** | 2017 | Hyperbolic space for hierarchical data | Core method for lineage embedding |
| **Hyperbolic GNN** | 2019 | GNN in hyperbolic space | Potential extension for spatial GNN |
| **TreeVAE** | 2023 | VAE with tree-structured latent space | Tree generation; inspires architecture |

### C. elegans-Specific Computational Work

| Work | Year | Description | Relation to NemaContext |
|------|------|-------------|-------------------------|
| **WormGUIDES** | 2015 | 4D digital embryo atlas | Primary spatial data source |
| **StarryNite/AceTree** | 2006 | Automated lineage tracing | Lineage tracking methodology |
| **Large et al.** | 2025 | Lineage-resolved embryonic transcriptome | Primary transcriptome data source |
| **Packer et al.** | 2019 | Single-cell atlas of embryogenesis | Legacy transcriptome data |

### Key Methodological Gap

```text
❌ Existing work:
   - Static atlases (describe, don't generate)
   - Single-cell predictions (not whole organisms)
   - Population-level dynamics (not individual cell tracking)
   - Single modality (transcriptome OR space OR lineage)

✅ NemaContext fills:
   - Generative model (predicts complete embryo states)
   - Whole-organism generation (1 → 959 cells)
   - Individual cell tracking (via deterministic lineage)
   - Trimodal integration (transcriptome + space + lineage)
```

---

## 📊 Datasets

### Required Data Sources

#### 1. Cell Lineage Tree (Structural Backbone)

| Dataset | Description | Source |
|---------|-------------|--------|
| **Sulston Lineage** | Complete hand-traced cell lineage (959 cells) | Sulston et al., 1983 |
| **WormBase Lineage** | Digitized lineage tree with cell naming | [wormbase.org](https://wormbase.org) |
| **StarryNite/AceTree** | Automated embryo tracking lineage data | Bao et al., 2006 |

**Provides**: `parent → [child1, child2]` relationships + division timestamps

#### 2. Single-Cell Transcriptomics (Token State)

| Dataset | Cells | Time Range | Features | Status |
|---------|-------|------------|----------|--------|
| **🌟 Large et al. 2025** (GSE292756) | ~425,000 | 120-600 min | **Lineage-resolved**, C. elegans + C. briggsae | **RECOMMENDED** |
| **Packer et al. 2019** (GSE126954) | ~86,000 | 100-650 min | Lineage annotations (10% clean) | Legacy |
| **Taylor et al. 2021** (CeNGEN) | ~100,000 | Adult | Neuron-focused, fine cell types | Optional |
| **Cao et al. 2017** (sci-RNA-seq) | ~50,000 | L2 larva | Whole-body coverage | Optional |

**Provides**: Gene expression vectors + cell identity labels

**Large 2025 Key Advantages**:
- 2.8x more C. elegans cells than Packer 2019
- **Direct cell-to-lineage mapping** (vs ~10% clean annotations in Packer)
- 152 annotated cell types with 429 progenitor/terminal classifications
- Cross-species comparison with C. briggsae data
- 98.4% lineage match rate to WormGUIDES spatial data

#### 3. 3D Spatial Coordinates (Token Relationships)

| Dataset | Description | Coverage |
|---------|-------------|----------|
| **WormGUIDES** | 4D digital embryo with (x,y,z,t) for each nucleus | 1-cell to ~350-cell stage |
| **OpenWorm** | Adult 3D anatomical model | Full adult anatomy |
| **C. elegans Connectome** | Complete 302-neuron synaptic wiring | Adult nervous system |

**Provides**: Cell positions → neighbor graph construction

#### 4. Cell Fate Annotations (Supervision Signal)

| Source | Content |
|--------|---------|
| **WormBase Cell Ontology** | Terminal fate labels (neuron, muscle, hypodermis, etc.) |
| **Packer 2019 Annotations** | `cell.type`, `lineage`, `embryo.time.bin` |

### Data Coverage & Integration

#### Temporal Coverage

| Dataset | Embryo (0-800 min) | Larva | Adult |
|---------|:------------------:|:-----:|:-----:|
| WormGUIDES (spatial) | ✅ 20-380 min | ❌ | ❌ |
| Large 2025 (transcriptome) | ✅ 0-830 min | ❌ | ❌ |
| Lineage tree | ✅ Complete | ✅ | ✅ |

**Best overlap window**: 20-380 minutes (~49% of Large 2025 cells)

#### Data Integration Challenge

```text
Key Problem: Aligning heterogeneous datasets

WormBase Lineage  ──┐
                    ├──→ Cell Naming System (ABala, P2, etc.) ──→ Unified Token ID
Large 2025        ──┤         ↑
WormGUIDES        ──┘    Bridge between datasets
```

### Phased Data Strategy

| Phase | Data Sources | Coverage |
|-------|--------------|----------|
| **Phase 1 (MVP)** | Large 2025 + WormGUIDES + WormBase Lineage | 1-cell → ~350-cell embryo |
| **Phase 2** | + Cao 2017 + CeNGEN | Extend to larva & adult |
| **Phase 3** | + Connectome | Add neural circuit structure |

---

## 🗺️ Roadmap

### Phase 1: Data Engineering & Backbone (Current)

- [x] **Automated Downloader**: Fetch Large et al. (2025) from GEO (recommended)
- [x] **Legacy Support**: Packer et al. (2019) also available
- [x] **WormGUIDES Integration**: Download and parse 4D spatial coordinates (360 timepoints)
- [x] **WormBase Lineage**: Generate lineage tree, timing, and fate data
- [ ] **Lineage Parser**: Convert cell names (e.g., `ABala`) to binary tree paths
- [ ] **AnnData Construction**: Unify RNA + lineage + spatial into `.h5ad`

### Phase 2: Trimodal Token Encoder

- [ ] **Hyperbolic Lineage Embedding**: Map binary tree to Poincaré ball
- [ ] **Spatial GNN**: Aggregate neighborhood information
- [ ] **Fusion Transformer**: Create unified "Cell-Token" representation

### Phase 3: Tree Generation Dynamics

- [ ] **Division Predictor**: Binary classifier for cell division timing
- [ ] **Daughter State Predictor**: Generate child token states
- [ ] **Neural SDE**: Model stochastic cell state dynamics
- [ ] **Long-Range Extrapolation**: Train on early embryo, predict adult

### Phase 4: Validation & Application

- [ ] **Fate Divergence Analysis**: Resolve "hidden bias" at ambiguous fate points
- [ ] **Digital Embryo Projection**: Visualize predicted trajectories in 3D
- [ ] **Perturbation Prediction**: What happens if gene X is knocked out?
- [ ] **Cross-Species Comparison**: C. elegans vs C. briggsae developmental differences

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Environment** | `uv` (fast Python package management) |
| **Deep Learning** | `PyTorch`, `TorchSDE` |
| **Geometric DL** | `Geoopt` (hyperbolic optimization), `torch-geometric` (GNN) |
| **Single-Cell** | `Scanpy`, `AnnData` |

---

## 📁 Project Structure

```text
NemaContext/
├── dataset/
│   ├── raw/                # Downloaded datasets
│   │   ├── large2025/      # GSE292756 files (recommended)
│   │   ├── packer2019/     # GSE126954 files (legacy)
│   │   ├── wormguides/     # 4D spatial coordinates
│   │   └── wormbase/       # Lineage tree exports
│   └── processed/          # Integrated h5ad files
├── src/
│   ├── data/               # Download and processing scripts
│   ├── encoder/            # Trimodal token encoder
│   ├── dynamics/           # Tree generation & SDE models
│   ├── geometry/           # Poincaré ball & hyperbolic math
│   └── utils/              # Lineage parsing, GNN helpers
├── examples/               # Example analysis scripts
├── experiments/            # Training configs and scripts
├── notebooks/              # Exploration and visualization
└── pyproject.toml          # Dependencies (uv)
```

---

## 🧪 Quick Start

```bash
# Initialize environment
uv sync

# Download core datasets (Large 2025 + WormGUIDES + WormBase)
uv run python -m src.data.downloader --source core

# Or download all datasets including connectome
uv run python -m src.data.downloader --source all

# Download specific datasets
uv run python -m src.data.downloader --source large2025   # Transcriptome (recommended)
uv run python -m src.data.downloader --source wormguides  # Spatial coordinates
uv run python -m src.data.downloader --source packer      # Legacy transcriptome

# Analyze Large 2025 dataset
uv run python examples/analyze_large2025.py

# Process and integrate data
uv run python -m src.data.processor

# (Future) Train model
uv run python -m experiments.train
```

---

## 📝 References

### Core Datasets

1. **Large, C. R. L., et al. (2025).** Lineage-resolved analysis of embryonic gene expression evolution in *C. elegans* and *C. briggsae*. *Science* 388:eadu8249. DOI: 10.1126/science.adu8249
2. **Packer, J. S., et al. (2019).** A lineage-resolved molecular atlas of *C. elegans* embryogenesis at single-cell resolution. *Science*.
3. **Sulston, J. E., et al. (1983).** The embryonic cell lineage of the nematode *Caenorhabditis elegans*. *Developmental Biology*.
4. **Bao, Z., et al. (2006).** Automated cell lineage tracing in *Caenorhabditis elegans*. *PNAS*.
5. **Taylor, S. R., et al. (2021).** Molecular topography of an entire nervous system. *Cell* (CeNGEN).
6. **Cao, J., et al. (2017).** Comprehensive single-cell transcriptional profiling of a multicellular organism. *Science*.

### Related Computational Work

7. **Villoutreix, P., et al. (2016).** An integrated modelling framework from cells to organism based on a cohort of digital embryos. *Scientific Reports*.
8. **Wang, Z., et al. (2018).** Deep Reinforcement Learning of Cell Movement in the Early Stage of *C. elegans* Embryogenesis. *Bioinformatics*.
9. **Kaul, H., et al. (2023).** Virtual cells in a virtual microenvironment recapitulate early development-like patterns. *Stem Cell Reports*.
10. **Romeo, N., et al. (2021).** Learning developmental mode dynamics from single-cell trajectories. *eLife*.

### Methods & Techniques

11. **Nickel, M., & Kiela, D. (2017).** Poincaré embeddings for learning hierarchical representations. *NeurIPS*.
12. **Vinyard, M., et al. (2025).** scDiffEq: Learning the dynamics of cell state transitions with stochastic differential equations.
13. **Weinreb, C., et al. (2020).** Lineage tracing on transcriptional landscapes links state to fate during differentiation. *Science*.
14. **Schiebinger, G., et al. (2019).** Optimal-transport analysis of single-cell gene expression identifies developmental trajectories in reprogramming. *Cell*.
15. **Yeo, G. H. T., et al. (2021).** Generative modeling of single-cell time series with PRESCIENT enables prediction of cell trajectories with interventions. *Nature Communications*.

### Foundation Models & Generative Methods

16. **Cui, H., et al. (2024).** scGPT: toward building a foundation model for single-cell multi-omics using generative AI. *Nature Methods*.
17. **Theodoris, C. V., et al. (2023).** Transfer learning enables predictions in network biology. *Nature* (Geneformer).
18. **Tong, A., et al. (2020).** TrajectoryNet: A Dynamic Optimal Transport Network for Modeling Cellular Dynamics. *ICML*.

---

## 📄 License

MIT License

---