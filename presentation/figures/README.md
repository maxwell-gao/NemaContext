# Presentation Figures

This folder should contain SVG files for the NemaContext presentation.

## Required Figures

Create the following SVG files (recommended tools: Figma, Inkscape, draw.io, or Excalidraw):

### 1. `context_levels.svg`
**Description**: Three levels of developmental context
- Show three interconnected circles/boxes:
  - **Lineage Context** (purple): Tree icon, "Where did this cell come from?"
  - **Molecular Context** (blue): DNA/RNA icon, "What genes are expressed?"
  - **Spatial Context** (orange): Grid/network icon, "Who are the neighbors?"
- Arrow pointing to center: "Cell Identity"
- Style: Clean, minimal, academic

### 2. `lineage_tree.svg`
**Description**: C. elegans lineage tree (simplified)
- Show P0 (zygote) at top
- First few divisions: P0 → AB + P1, AB → ABa + ABp, P1 → EMS + P2
- Indicate "..." and "959 cells" at bottom
- Color coding: Founder cells in different colors
- Style: Tree structure, top-to-bottom

### 3. `trimodal_integration.svg`
**Description**: Data integration diagram
- Three input boxes:
  - Transcriptome (blue): "Large 2025, 234K cells, 0-830 min"
  - Spatial (green): "WormGUIDES, 1.3K cells, 20-380 min"
  - Morphology (orange): "CShaper, 1.2K cells, 20-380 min"
- Arrows converging to center
- Output: "Unified AnnData" (purple)
- Style: Flow diagram

### 4. `temporal_gap.svg`
**Description**: Timeline showing data coverage
- Horizontal timeline: 0 to 830 minutes
- Bars showing coverage:
  - CShaper (orange): 20-380 min
  - WormGUIDES (green): 20-380 min
  - Large 2025 (blue): 0-830 min (full span)
- Highlight gap region (380-830 min) with red annotation: "No contact data!"
- Style: Gantt chart-like

### 5. `gnn_problem.svg`
**Description**: GNN chicken-and-egg problem
- Left: "GNN needs graph as input"
- Middle: Question mark / circular dependency
- Right: "Graph is what we want to predict"
- Circular arrow showing the paradox
- Style: Problem illustration

### 6. `transformer_context.svg`
**Description**: Transformer attention as organism context
- Show N cells as tokens
- Attention lines from one cell to ALL other cells
- Highlight that each cell "sees" the entire embryo
- Equation: Cell_i = Attention(Cell_i, All Cells)
- Compare with GNN (limited to neighbors)
- Style: Neural network diagram

### 7. `architecture.svg`
**Description**: Flow Matching Transformer architecture
- Left: Input tokens (N × d)
  - Transcriptome embedding
  - Lineage encoding
  - Time encoding
- Middle: Pairwise Transformer
  - Axial attention blocks
- Right: Flow Matching Head
  - Velocity prediction
  - Output: N×N adjacency matrix
- Style: Neural network architecture diagram

### 8. `flow_matching.svg`
**Description**: Flow matching process illustration
- Left: Noise matrix (random)
- Middle: Flow arrows showing transformation
- Right: Contact graph (binary adjacency)
- Show time evolution: t=0 → t=1
- Style: Process diagram

### 9. `curriculum.svg`
**Description**: Curriculum learning stages
- Four stages in sequence with arrows:
  - Stage 1: 4-50 cells (simple blob)
  - Stage 2: 50-200 cells (gastrulation)
  - Stage 3: 200-500 cells (organogenesis)
  - Stage 4: 500-1000 cells (differentiation)
- Show increasing complexity
- Style: Progressive stages

### 10. `synthesis.svg`
**Description**: Philosophy to implementation flow
- Three connected elements:
  - Philosophy: "Organism as Context" (circle)
  - Architecture: "Transformer Attention" (rectangle)
  - Output: "Contact Graph" (network icon)
- Arrows showing flow
- Style: Conceptual diagram

---

## Technical Notes

### SVG Conversion for LaTeX
The presentation uses the `svg` package which requires Inkscape for conversion.

**Option A**: Use Inkscape to convert SVG to PDF
```bash
inkscape --export-type=pdf figure.svg
```

**Option B**: Convert all SVGs before compiling
```bash
for f in *.svg; do inkscape --export-type=pdf "$f"; done
```

**Option C**: Use `--shell-escape` with pdflatex
```bash
pdflatex --shell-escape nemacontext_progress.tex
```

### Fallback
If SVG support is problematic, convert to PDF and use:
```latex
\includegraphics[width=0.8\textwidth]{figures/figure_name.pdf}
```

### Recommended Dimensions
- Aspect ratio: 16:9 (match presentation)
- Width: 1600-2400 pixels
- Export at high resolution for PDF quality

### Color Palette
Use consistent colors matching the presentation:
- celegans (green): #4C9900
- contact (orange): #CC6600
- lineage (blue): #3366CC
- context (purple): #800080
