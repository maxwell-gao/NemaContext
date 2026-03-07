# Gene-Context Baseline: Biological Meaning and Published Context

## Purpose

This document explains the active `gene-context` baseline in biological terms
and situates it relative to published bioinformatics work on developmental
state transition modeling.

Relevant code:

- `src/data/gene_context_dataset.py`
- `src/branching_flows/gene_context.py`
- `examples/whole_organism_ar/train_gene_context.py`
- `examples/whole_organism_ar/evaluate_gene_context.py`
- `examples/whole_organism_ar/train_gene_single_cell.py`
- `examples/whole_organism_ar/evaluate_gene_single_cell.py`

## What Biological Problem This Baseline Is Trying To Solve

The active baseline is not a spatial simulator and not a lineage-conditioned
classifier.

It is a developmental transition model over real single-cell transcriptomic
snapshots:

- input: a group of cells observed in the same developmental time window,
- target: the near-future transcriptional change of each cell,
- auxiliary targets: whether a cell is likely to divide or disappear by the
  next time window.

The biological question is:

> Does multi-cell transcriptomic context improve prediction of short-horizon
> developmental state change, beyond what is available from a single cell and
> time alone?

This is the correct question for the current data regime because the available
scRNA-seq data are destructive snapshots, not true tracked single-cell movies.

## Biological Meaning of the Inputs

### `genes`

Each token's main state is real gene expression from
`dataset/processed/nema_extended_large2025.h5ad`.

After HVG selection and normalization, this vector is interpreted as an
approximation to the cell's current developmental state:

- differentiation program,
- cell-cycle state,
- tissue commitment,
- transient regulatory program,
- developmental competence.

This is the main biological state variable in the active baseline.

### `time`

The model also receives embryo time.

Biologically, this represents developmental stage:

- the same expression program can have different meaning early versus late,
- developmental transitions are not time-homogeneous,
- time provides stage context without injecting explicit lineage structure.

### `multi-cell context`

In the multi-cell baseline, one sample contains multiple cells from the same
time window.

Biologically, this represents the surrounding developmental population:

- which other cell states coexist at that stage,
- what cell-state composition is present,
- what global developmental context constrains each cell.

This is important because development is not an independent-cell process.
Cells change state in the presence of other cells undergoing coordinated
programs.

### What Is Not Input

The active baseline does **not** input:

- founder identity,
- lineage embeddings,
- lineage tree distance,
- lineage-biased attention,
- spatial coordinates as the main driver.

This matters because the active direction is to learn context from data rather
than from injected lineage structure.

## Biological Meaning of the Outputs

### `gene_delta`

The main output is a predicted short-horizon change in gene state.

Biologically, this is a local developmental transition vector:

- where the cell's transcriptional program is moving next,
- not just what type it currently is,
- not yet a full cell-fate mechanism,
- but a useful proxy for near-future developmental motion in expression space.

### `split_logits`

This is a predicted propensity for division in the next time window.

Biologically, it should be interpreted as:

- a near-term proliferative tendency,
- not a mechanistic cell-cycle model,
- not yet an exact mitotic timing predictor.

### `del_logits`

This is a predicted propensity for disappearance by the next window.

At the current stage, this should be interpreted cautiously.

It is closer to:

- failure to persist into the next matched window,
- possible death,
- possible state-exit from the local matching regime,

than to a clean apoptosis label.

This is a known limitation of the current weak target construction.

## Why This Baseline Is Biologically More Meaningful Than the Spatial Baseline

The spatial rollout baseline was useful as an engineering test of dynamic cell
management, but it is not a strong biological model of development.

This gene-context baseline is more biologically meaningful because:

- gene state is treated as the primary developmental state,
- time is treated as developmental stage,
- context comes from other cells' transcriptomic states,
- lineage is not directly injected into the model.

This makes the model closer to developmental systems biology and farther from a
point-cloud kinematics model.

## Closest Published Work

No widely adopted published method is identical to this active baseline.
The nearest prior work falls into three adjacent families.

### 1. RNA velocity and local future-state direction

- La Manno et al., 2018, *RNA velocity of single cells*, Nature
  - https://www.nature.com/articles/s41586-018-0414-6
- Bergen et al., 2020, *Generalizing RNA velocity to transient cell states through dynamical modeling*, Nature Biotechnology
  - https://www.nature.com/articles/s41587-020-0591-3

These methods are close to our `gene_delta` output because they also estimate
short-horizon future direction in transcriptomic state space.

Key difference:

- RNA velocity is fundamentally a **single-cell** local dynamics method,
- typically uses spliced/unspliced information,
- does not model explicit multi-cell transcriptomic context,
- does not directly include split/delete outputs.

### 2. Optimal transport and cross-time developmental coupling

- Schiebinger et al., 2019, *Optimal-Transport Analysis of Single-Cell Gene Expression Identifies Developmental Trajectories in Reprogramming*, Cell
  - https://pubmed.ncbi.nlm.nih.gov/30712874/
- Tong et al., 2020, *TrajectoryNet: A Dynamic Optimal Transport Network for Modeling Cellular Dynamics*, ICML / PMLR
  - https://proceedings.mlr.press/v119/tong20a.html
- Yeo et al., 2021, *Generative modeling of single-cell time series with PRESCIENT enables prediction of cell trajectories with interventions*, Nature Communications
  - https://www.nature.com/articles/s41467-021-23518-w

These are close to our data construction problem:

- scRNA-seq provides snapshots, not tracked trajectories,
- one must infer likely future couplings or flows between time points,
- the task is developmental dynamics from cross-sectional data.

Key difference:

- these methods mostly operate at the level of transport, trajectory inference,
  or continuous-time cell-state flow,
- they usually do not represent a **time-window of many cells as the explicit
  input context** for each prediction,
- they are not formulated as a multi-cell context model with per-token
  split/delete heads.

### 3. Broader trajectory inference

- VIA, Nature Communications 2021
  - https://www.nature.com/articles/s41467-021-25773-3
- Review: *Concepts and limitations for learning developmental trajectories from single cell genomics*, Development 2019
  - https://journals.biologists.com/dev/article/146/12/dev170506/19458/Concepts-and-limitations-for-learning

These are useful conceptual references because they emphasize:

- snapshot data cannot directly observe trajectories,
- lineage, proliferation, and death complicate inference,
- many inferred trajectories are only partial proxies for underlying dynamics.

That is directly relevant to how our current `split` and especially `delete`
targets should be interpreted.

## What Is Distinct About This Baseline

The active baseline combines assumptions that are uncommon in combination:

- real transcriptome is the primary state,
- one input sample is a **multi-cell developmental context window**,
- lineage is excluded from model inputs,
- time is allowed as a condition,
- output includes both future gene-state change and event propensity.

That combination is closer to a whole-embryo developmental context model than
standard single-cell trajectory inference, while still remaining grounded in
available snapshot data.

## First Internal Comparison: Multi-Cell Context vs Single-Cell Control

We ran a direct comparison under the same data construction and training setup:

- same `h5ad`,
- same time windows,
- same HVG count,
- same prediction task,
- same split/delete heads.

The only structural difference was:

- `GeneContextModel`: cells can use other cells in the same time window,
- `SingleCellGeneTimeModel`: each cell uses only its own gene state plus time.

### Current comparison outputs

Multi-cell context:

- checkpoint: `checkpoints_gene_context_compare/best.pt`
- evaluation: `result/gene_context/evaluation_compare_multi.json`
- result:
  - `total = 1.5694`
  - `gene = 1.4786`
  - `del = 0.0074`

Single-cell control:

- checkpoint: `checkpoints_gene_single_cell_compare/best.pt`
- evaluation: `result/gene_context/evaluation_compare_single.json`
- result:
  - `total = 2.0261`
  - `gene = 1.4762`
  - `del = 0.2628`

### Biological interpretation of this result

At this stage, the gene-state error is roughly similar between the two models,
but the multi-cell model is much better behaved on the event side,
especially the `delete` head.

That suggests:

- a single cell's own expression plus time is not enough to stabilize its
  near-future interpretation,
- the surrounding developmental population provides additional constraint,
- multi-cell transcriptomic context helps suppress implausible disappearance
  predictions.

Biologically, this is plausible:

- cells do not exist as independent transcriptomic particles,
- developmental state is constrained by the broader composition of the embryo,
- context helps distinguish true transition from spurious uncertainty.

## Current Limits of Biological Interpretation

This baseline is meaningful, but it is not yet a full developmental model.

Important limitations:

- future pairing is still weak and local,
- split/delete labels are approximate,
- `delete` is not yet a clean death label,
- no explicit cell-cell signaling mechanism is modeled,
- no real whole-embryo rollout exists yet for this transcriptomic path.

So the correct interpretation today is:

- this baseline models **short-horizon developmental state transition under
  transcriptomic context**,
- not full cell-fate mechanism,
- not lineage reconstruction,
- not organism-scale developmental simulation.

## Practical Summary

If a reader asks what this active baseline means biologically, the shortest
correct answer is:

> It treats each cell's transcriptome as the main developmental state, treats
> embryo time as stage, uses other cells in the same time window as context,
> and predicts how each cell's gene program will move next, along with whether
> it is more likely to divide or disappear.

## References

1. La Manno G, et al. *RNA velocity of single cells*. Nature (2018).
   https://www.nature.com/articles/s41586-018-0414-6
2. Bergen V, et al. *Generalizing RNA velocity to transient cell states through dynamical modeling*. Nature Biotechnology (2020).
   https://www.nature.com/articles/s41587-020-0591-3
3. Schiebinger G, et al. *Optimal-Transport Analysis of Single-Cell Gene Expression Identifies Developmental Trajectories in Reprogramming*. Cell (2019).
   https://pubmed.ncbi.nlm.nih.gov/30712874/
4. Tong A, et al. *TrajectoryNet: A Dynamic Optimal Transport Network for Modeling Cellular Dynamics*. ICML / PMLR (2020).
   https://proceedings.mlr.press/v119/tong20a.html
5. Yeo SK, et al. *Generative modeling of single-cell time series with PRESCIENT enables prediction of cell trajectories with interventions*. Nature Communications (2021).
   https://www.nature.com/articles/s41467-021-23518-w
6. Stassen SV, et al. *Generalized and scalable trajectory inference in single-cell omics data with VIA*. Nature Communications (2021).
   https://www.nature.com/articles/s41467-021-25773-3
7. Tritschler S, et al. *Concepts and limitations for learning developmental trajectories from single cell genomics*. Development (2019).
   https://journals.biologists.com/dev/article/146/12/dev170506/19458/Concepts-and-limitations-for-learning
