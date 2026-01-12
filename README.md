# NemaContext: Modeling *C. elegans* Development as Token Tree Generation

**NemaContext** is a machine learning framework that models the complete developmental trajectory of *Caenorhabditis elegans* — from a single fertilized egg to a 959-cell adult — by treating **each cell as a token** and development as a **binary tree generation process**.

> 🆕 **Update**: Now using **Large et al. 2025** (GSE292756) as the primary transcriptome dataset with lineage-resolved annotations for >425,000 cells!

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

### Data Integration Challenge

```text
Key Problem: Aligning heterogeneous datasets

WormBase Lineage  ──┐
                    ├──→ Cell Naming System (ABala, P2, etc.) ──→ Unified Token ID
Packer 2019       ──┤         ↑
WormGUIDES        ──┘    Bridge between datasets
```

### Phased Data Strategy

| Phase | Data Sources | Coverage |
|-------|--------------|----------|
| **Phase 1 (MVP)** | Packer 2019 + WormGUIDES + WormBase Lineage | 1-cell → ~350-cell embryo |
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
- [ ] **Germline Tracking**: Follow P-lineage from embryo to adult

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
├── data/
│   ├── raw/                # Downloaded datasets
│   │   ├── packer2019/     # GSE126954 files
│   │   ├── wormguides/     # 4D spatial coordinates
│   │   └── wormbase/       # Lineage tree exports
│   └── processed/          # Integrated h5ad files
├── src/
│   ├── data/               # Download and processing scripts
│   ├── encoder/            # Trimodal token encoder
│   ├── dynamics/           # Tree generation & SDE models
│   ├── geometry/           # Poincaré ball & hyperbolic math
│   └── utils/              # Lineage parsing, GNN helpers
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

### Methods

6. **Nickel, M., & Kiela, D. (2017).** Poincaré embeddings for learning hierarchical representations. *NeurIPS*.
7. **Vinyard, M., et al. (2025).** scDiffEq: Learning the dynamics of cell state transitions with stochastic differential equations.
8. **Weinreb, C., et al. (2020).** Lineage tracing on transcriptional landscapes links state to fate during differentiation. *Science*.

---

## 📄 License

MIT License

---