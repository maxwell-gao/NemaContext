# Worm Forecasting Results

## Scope

This document records the current worm-native forecasting comparison on the mainline task:
- raw Large2025
- lineage-first whole-embryo dynamics
- held-out late time transitions

It is not a human perturbation benchmark.
It is the current worm forecasting benchmark used to position the project on the whole-organism roadmap.

## Main Split

Primary split:
- `time_transition`
- train: `0->40` through `600->640`
- eval: `640->680`, `680->720`, `720->760`, `760->800`

Primary benchmark files:
- `result/gene_context/benchmark_worm_dynamics_time_transition_common.csv`
- `result/gene_context/benchmark_worm_scnode_time_transition_common.csv`
- `result/gene_context/benchmark_worm_prescient_time_transition_common.csv`

## Comparison Table

All numbers below are from the same worm forecasting task and the same held-out future-transition split.

| Method | future_set_sinkhorn | mean_gene_pearson | delta_gene_pearson | top_de_delta_pearson | top_de_sign_acc |
|---|---:|---:|---:|---:|---:|
| Worm mainline (lineage-first multicellular dynamics) | 71.8452 | 0.9616 | 0.8326 | 0.7295 | 0.6125 |
| PRESCIENT-style baseline | 87.0145 | 0.8157 | 0.0363 | 0.3473 | 0.6500 |
| Persistence baseline | 120.7683 | 0.8713 | 0.0000 | 0.0000 | 0.0000 |
| scNODE-style baseline | 127.3759 | 0.3460 | 0.1676 | 0.6657 | 0.3875 |

## Interpretation

Current mainline model is clearly stronger than the two generic forecasting baselines on this worm task.

Most important observations:
- The worm mainline is best on `future_set_sinkhorn`, so it is best at predicting the future population distribution.
- The worm mainline is best on `mean_gene_pearson`, so it is also best at future pseudobulk prediction.
- The worm mainline is best on `delta_gene_pearson`, which is the cleanest summary of future change prediction.
- `scNODE-style` learns some signal on top-DE genes, but is weak on the overall future state.
- `PRESCIENT-style` is stronger than `scNODE-style` on distribution-level forecasting, but still substantially behind the worm mainline.
- Persistence remains nontrivial on mean-gene correlation, which means this task still contains a strong temporal smoothness prior.


## Structure Split Diagnostics

These splits are not the headline comparison yet, but they show whether the backbone is already translating its advantages into founder- and region-structured forecasting.

### Founder-Group Split

Files:
- `result/gene_context/benchmark_worm_dynamics_founder_group_common.csv`
- `result/gene_context/benchmark_worm_dynamics_founder_group_structure.csv`

| Method | future_set_sinkhorn | mean_gene_pearson | delta_gene_pearson | top_de_delta_pearson | top_de_sign_acc | founder_group_pseudobulk_pearson | region_group_pseudobulk_pearson |
|---|---:|---:|---:|---:|---:|---:|---:|
| Worm mainline | 71.4710 | 0.9452 | 0.7731 | 0.7678 | 0.7225 | 0.1825 | 0.2246 |
| Persistence | 117.1065 | 0.8747 | 0.0000 | 0.0000 | 0.0000 | 0.4894 | 0.4394 |

### Region-Group Split

Files:
- `result/gene_context/benchmark_worm_dynamics_region_group_common.csv`
- `result/gene_context/benchmark_worm_dynamics_region_group_structure.csv`

| Method | future_set_sinkhorn | mean_gene_pearson | delta_gene_pearson | top_de_delta_pearson | top_de_sign_acc | founder_group_pseudobulk_pearson | region_group_pseudobulk_pearson |
|---|---:|---:|---:|---:|---:|---:|---:|
| Worm mainline | 71.4710 | 0.9452 | 0.7731 | 0.7678 | 0.7225 | 0.3071 | 0.2008 |
| Persistence | 117.1065 | 0.8747 | 0.0000 | 0.0000 | 0.0000 | 0.5516 | 0.4873 |

### Interpretation Of The Structure Splits

These results say two things at once:
- The current mainline clearly wins on general forecasting metrics even under founder/region split regimes.
- The current mainline does **not** yet win on the structure-preservation pseudobulk metrics.

This is why the roadmap position remains the same:
- whole-embryo dynamics is working,
- but founder/region-consistent readout is still a bottleneck,
- so the project is not yet at the whole-organism spatial structure prediction stage.

## Roadmap Position

This result does **not** mean the project is already doing whole-organism spatial structure prediction.

Current position on the roadmap:
1. Local patch experiments: completed enough to show the value of direct future gene prediction and token-wise video modeling.
2. Whole-embryo lineage-first dynamics backbone: current active mainline.
3. Region/founder-consistent future prediction: partially started, not yet strong enough.
4. Spatial alignment with WormGUIDES/CShaper: not yet the active mainline.
5. Whole-organism spatial structure prediction: not yet reached.

So the current project state is:
- we are in the **whole-embryo gene dynamics stage**,
- before the **spatial alignment** stage,
- and before the **whole-organism spatial structure prediction** stage.

## Practical Meaning

The current result is strong enough to justify the mainline design choice:
- gene-first
- lineage-first
- multicellular dynamics-first
- spatial data used later as alignment/validation, not as a hard Stage 1 gate

The current bottleneck is no longer whether whole-embryo dynamics can be learned.
The bottleneck is how to convert this backbone into:
- stronger founder/region-consistent forecasting,
- then spatially coherent whole-organism prediction.
