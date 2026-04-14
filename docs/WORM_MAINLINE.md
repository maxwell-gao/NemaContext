# Worm Mainline Design

## Scope

This document defines the current mainline for worm gene prediction in this repo.
The mainline is not human perturbation benchmarking.
The mainline is lineage-conditioned multicellular dynamics prediction on *C. elegans* embryo data.

## Main Objective

Predict future worm embryo gene state from current worm embryo gene state, using:
- gene expression as the primary signal,
- lineage as the required structural prior,
- time as the causal axis,
- spatial data only as a later alignment signal.

In shorthand:
- `current whole-embryo gene state -> future whole-embryo / future region gene state`

## Why This Is The Mainline

### What we have a lot of
- Large2025 raw transcriptome data
- embryo time annotations
- lineage annotations
- cell-type / packer-type annotations

### What we do not have much of
- high-confidence matched real spatial coordinates

This means the mainline should maximize use of:
- gene
- time
- lineage

and should not make real spatial coverage a hard sample eligibility condition.

## Non-Mainline Work

The following are useful but not mainline:
- human perturbation benchmark integration such as `scPerturBench`
- strict spatial patch pipelines that collapse usable sample count
- mask-heavy pretraining as the primary objective

These can still exist as side paths, but they should not define the repo's core direction.

## Mainline Data Definition

Use raw Large2025 directly.

Current dataset path:
- `src/data/gene_context_dataset_large2025.py`
- exported through `src/data/gene_context_dataset.py`
- `Large2025WholeEmbryoDataset`

### Sample Unit
A sample is a whole-embryo time-binned snapshot pair:
- history snapshot(s) at `t`
- future snapshot at `t + dt`

### Token Budget
A snapshot is converted into a fixed token budget by deterministic selection.
This is not a strict spatial patch.
It is a lineage- and region-aware whole-embryo token set.

### Required Token Structure
Each token carries:
- gene vector
- lineage binary features
- founder id
- lineage depth
- lineage validity flag
- region id derived from `packer_cell_type` or `cell_type`
- token rank

## Mainline Model Definition

Current training path:
- `examples/whole_organism_ar/train_large2025_lineage_stage1.py`

Current backbone:
- `LineageWholeEmbryoModel`
- implemented in `src/branching_flows/lineage_backbone.py`
- exported through `src/branching_flows/gene_context.py`

### Training Objective
The main training objective is dynamics-first prediction:
- `future gene_set`
- `future mean_gene`

Default behavior:
- no masking in the primary path
- dynamics prediction is the main optimization target

This keeps the model directly comparable to worm gene-state prediction tasks, instead of turning it into a reconstruction-pretraining system.

## Mainline Benchmark Definition

Current benchmark path:
- `examples/whole_organism_ar/benchmark_worm_dynamics.py`

### Primary Split
Default split is held-out late time transitions.
This tests developmental transition generalization.

### Primary Metrics
Report:
- `mean_gene_mse`
- `mean_gene_pearson`
- `delta_gene_mse`
- `delta_gene_pearson`
- `top_de_delta_pearson`
- `top_de_sign_acc`
- `founder_group_pseudobulk_pearson`
- `region_group_pseudobulk_pearson`

### Baseline
Always compare against persistence:
- current state copied forward as future prediction

## Relationship To Existing Worm Resources

### Packer / Large2025
This is the main training and evaluation source.
It defines:
- time
- lineage
- cell type
- future gene targets

### WormGUIDES
Use only as:
- coarse spatial prior
- later-stage alignment reference
- geometry consistency check on the small matched subset

Do not use it as a hard gate for Stage 1 sample construction.

### CShaper
Use only as:
- morphology/spatial validation
- optional later-stage spatial consistency supervision

Do not use it as the main source of training eligibility.

## Current Interpretation

The repo should treat the following as the active worm mainline:
1. train lineage-first whole-embryo dynamics on raw Large2025
2. evaluate on worm-native time/lineage/region metrics
3. use WormGUIDES/CShaper only for later spatial alignment and validation

## Immediate Next Steps

The next work on this line should be:
1. train longer and/or sweep model capacity on the dynamics-first path
2. add stronger worm-native splits for founder-group and region-group generalization
3. only then add limited spatial alignment using WormGUIDES/CShaper
