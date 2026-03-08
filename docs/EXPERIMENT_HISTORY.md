# Experiment History

This document records the main experiments run during the current
`gene-context` and supervision-repair phase.

It is intentionally selective:

- it keeps the high-signal milestones,
- it omits repeated smoke runs unless they changed project direction,
- it focuses on what each experiment actually established.

All result files referenced below live under `result/gene_context/` unless
otherwise noted.

## Scope

These experiments do **not** yet solve whole-embryo developmental prediction.

They belong to the current stage of the roadmap:

- learn a local population-update rule from real transcriptomic windows,
- test whether multi-cell context is genuinely used,
- repair `split/delete` supervision before re-entering embryo-scale rollout.

## 1. Structured Anchor Context Was Implemented And Smoke-Tested

Configuration:

- dataset: `dataset/processed/nema_extended_large2025.h5ad`
- structured context: `anchor + local neighborhood + global background`
- active sampling strategy: `spatial_anchor`

Representative result:

- `evaluation_anchor_smoke.json`
  - total: `2.2309`
  - gene: `0.8870`

What changed:

- the dataset was upgraded from loose random windows to structured
  anchor-centered context,
- the model was given lightweight role and anchor-distance labels,
- training and evaluation could run end-to-end on the new input contract.

What this established:

- the active path was operational,
- context could be represented as a structured local-plus-global window.

What it did **not** establish:

- that larger context helps,
- that multi-cell beats single-cell,
- that event labels are informative.

## 2. Context-Length And Global-Background Sweeps Showed Context Use

Key comparison:

- `evaluation_anchor_ctx64_g0_e10_full.json`
  - total: `1.4688`
- `evaluation_anchor_ctx64_g0_e10_anchor_only.json`
  - total: `1.5270`
- `evaluation_anchor_ctx64_g16_e10_full.json`
  - total: `1.4526`
- `evaluation_anchor_ctx64_g16_e10_anchor_only.json`
  - total: `1.5342`

Smaller sweep check:

- `evaluation_anchor_ctx32_g0_e3.json`
  - total: `2.1576`
- `evaluation_anchor_ctx32_g8_e3.json`
  - total: `2.1575`

What this established:

- `full < anchor_only`, so the multi-cell model was genuinely using context,
- adding global background tokens gave a small but consistent improvement,
- structured context was more than a cosmetic input change.

What it did **not** establish:

- that multi-cell context already beats a matched single-cell control,
- that event heads are biologically meaningful,
- that the model can support multi-step dynamics.

## 3. Event Supervision Audit Revealed The Real Bottleneck

Audit files:

- `audit_summary_ctx64_g16.json`
- `audit_pairs_ctx64_g16.csv`

Main counts:

- `num_pairs = 10`
- `total_split_positive = 2330`
- `total_del_positive = 1584`
- `total_anchor_split_positive = 50`
- `total_anchor_del_positive = 83`
- `mean_split_positive_rate = 0.0761`
- `mean_del_positive_rate = 0.0435`
- `mean_anchor_split_positive_rate = 0.0486`
- `mean_anchor_del_positive_rate = 0.0900`

Most event-rich anchor windows:

- `224 -> 244 min`: anchor event `41` (`0 split`, `41 delete`)
- `204 -> 224 min`: anchor event `35` (`20 split`, `15 delete`)
- `184 -> 204 min`: anchor event `31` (`27 split`, `4 delete`)

What this established:

- the global dataset was **not** event-empty,
- the main sparsity problem was the anchor-supervised event slice,
- event-enriched subsets were feasible and necessary.

What it changed in the project:

- event-enriched training and evaluation subsets were implemented,
- later experiments stopped treating all time windows as equally informative.

## 4. Event-Enriched Comparison Showed Context Use But Also A Problem

Files:

- `evaluation_anchor_event_enriched_full.json`
- `evaluation_anchor_event_enriched_anchor_only.json`
- `evaluation_single_cell_event_enriched_anchor.json`
- `error_compare_event_enriched_all.json`

Aggregate results:

- multi-cell full: `2.1344`
- multi-cell anchor-only: `2.1511`
- single-cell matched control: `1.8357`

Error decomposition:

- `mean_delta_total = +0.1761` (multi-cell worse)
- `mean_delta_gene = -0.0098`
- `mean_delta_split = +0.0544`
- `mean_delta_del = +0.1315`

Interpretation:

- the multi-cell model was still using context,
- but it was not winning overall against the single-cell control,
- the failure was **not** in gene regression,
- it was mainly in the event heads, especially `delete`.

What this established:

- context use alone was not enough,
- `delete` supervision had become the main scientific bottleneck.

## 5. Split-Rich And Delete-Rich Subsets Separated The Two Event Types

Files:

- `eval_anchor_split_rich_multi_full.json`
- `eval_anchor_split_rich_multi_anchor_only.json`
- `eval_anchor_split_rich_single.json`
- `error_compare_anchor_split_rich.json`
- `eval_anchor_delete_rich_multi_full.json`
- `eval_anchor_delete_rich_multi_anchor_only.json`
- `eval_anchor_delete_rich_single.json`
- `error_compare_anchor_delete_rich.json`

### Split-rich

Aggregate:

- multi-cell full: `2.0222`
- multi-cell anchor-only: `2.0389`
- single-cell: `1.8487`

Decomposition:

- `mean_delta_total = +0.0218`
- `mean_delta_gene = +0.0136`
- `mean_delta_split = -0.0022`
- `mean_delta_del = +0.0103`

Interpretation:

- split-rich was close to parity,
- multi-cell had a slight edge on the `split` term,
- but not enough to beat the single-cell control overall.

### Delete-rich

Aggregate:

- multi-cell full: `2.0459`
- multi-cell anchor-only: `2.0590`
- single-cell: `1.7360`

Decomposition:

- `mean_delta_total = +0.2266`
- `mean_delta_gene = +0.0155`
- `mean_delta_split = +0.1153`
- `mean_delta_del = +0.0958`

Interpretation:

- delete-rich was much worse for the multi-cell model,
- the problem was not limited to the delete term,
- this strongly suggested that the delete target construction itself was
  compromised.

What this established:

- `split` and `delete` should not be treated as one combined event problem,
- `split` looked like the better place to search for true context benefit,
- `delete` required target repair before further interpretation.

## 6. Delete Target Repair: Weak Delete Was Replaced By Strict Delete

Implementation change:

- old delete: one-step unmatched in the next window
- new strict delete: unmatched in `t + dt` **and** unmatched again in
  `t + 2dt`

The dataset now emits:

- `match_type`
- `weak_del_target`
- `strict_del_target`
- `del_target` selected by `delete_target_mode`

Validation summary:

- anchor weak delete count: `83`
- anchor strict delete count: `16`

This was the key sign that the old target was over-calling delete-like events.

What this established:

- a large fraction of old delete positives were probably single-step matching
  artifacts,
- strict delete was a better default for future work.

## 7. Strict Delete Reduced The Delete Penalty But Did Not Flip The Comparison

Files:

- `eval_anchor_delete_strict_multi_full.json`
- `eval_anchor_delete_strict_multi_anchor_only.json`
- `eval_anchor_delete_strict_single.json`
- `error_compare_anchor_delete_strict_multi_vs_single.json`
- `error_compare_anchor_delete_weak_vs_strict_multi.json`

Aggregate:

- strict multi-cell full: `3.2508`
- strict multi-cell anchor-only: `3.2630`
- strict single-cell: `3.1834`

Context check:

- `full < anchor_only`, so context was still being used.

Strict multi vs strict single:

- `mean_delta_total = +0.0237`
- `mean_delta_gene = +0.0136`
- `mean_delta_split = -0.0022`
- `mean_delta_del = +0.0122`

Weak multi vs strict multi:

- `mean_delta_total = +0.0198`
- `mean_delta_gene = -0.0015`
- `mean_delta_split = -0.0035`
- `mean_delta_del = +0.0247`

Interpretation:

- strict delete improved the delete term,
- the repaired label made the multi vs single comparison much closer,
- but it still did **not** make delete-rich a strong multi-cell win.

What this established:

- strict delete should be the default,
- delete-rich remains a weak mainline benchmark,
- repaired delete is useful, but still not the strongest evidence for context.

## 8. Strict Split-Rich Confirmed That Split-Rich Is Mostly Independent Of Delete Repair

Files:

- `eval_anchor_split_strict_multi_full.json`
- `eval_anchor_split_strict_multi_anchor_only.json`
- `eval_anchor_split_strict_single.json`
- `error_compare_anchor_split_strict_multi_vs_single.json`
- `error_compare_anchor_split_weak_vs_strict_multi.json`

Aggregate:

- strict multi-cell full: `3.2508`
- strict multi-cell anchor-only: `3.2630`
- strict single-cell: `3.1834`

Strict split multi vs strict split single:

- `mean_delta_total = +0.0237`
- `mean_delta_gene = +0.0136`
- `mean_delta_split = -0.0022`
- `mean_delta_del = +0.0122`

Weak split multi vs strict split multi:

- `mean_delta_total = +0.0288`
- `mean_delta_gene = 0.0`
- `mean_delta_split = 0.0`
- `mean_delta_del = +0.0288`

Interpretation:

- the split-rich conclusion did not materially change under strict delete,
- split-rich remained the cleaner place to test context,
- the slight multi-cell advantage still showed up mainly in the `split` term,
  not in the total score.

## Current Project-Level Reading

These experiments collectively support the following view:

1. The active benchmark is a real **one-step local population-update** problem,
   not a solved whole-embryo model.
2. Multi-cell context is genuinely being used.
3. Global background context gives a small but real gain.
4. The hardest scientific problem in the current phase was not model capacity.
   It was target quality, especially for `delete`.
5. `split` is the more promising event type for demonstrating biologically
   meaningful context benefit.
6. `strict delete` should remain the default.
7. None of these results yet justify claims about embryo-scale rollout from the
   zygote.

## What This History Does Not Claim

This experiment history does **not** show that we have already learned a full
developmental simulator.

It does **not** establish:

- long-horizon single-cell trajectory prediction,
- stable whole-embryo rollout,
- a biologically closed multi-step developmental system,
- zygote-to-embryo prediction.

Those remain later-phase goals in the roadmap.
