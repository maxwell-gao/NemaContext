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

- multi-cell full: `3.2508`
- multi-cell anchor-only: `3.2630`
- single-cell: `3.1834`

Interpretation:

- strict delete helped clean up the delete target,
- but it did not make token-matching become a clearly multi-cell-favoring
  task,
- the project still needed a less hand-matched objective.

## 8. Relative Spatial Encoding And Pairwise Bias Improved The Clean Baseline

Files:

- `eval_anchor_split_relpos_multi.json`
- `eval_anchor_split_nospatial_multi.json`
- `eval_anchor_split_relpos_single.json`
- `eval_anchor_split_nospatial_single.json`
- `eval_anchor_split_relpos_pairbias_full.json`
- `eval_anchor_split_relpos_pairbias_anchor_only.json`
- `error_compare_anchor_split_relpos_full_vs_anchor_only.json`
- `error_compare_anchor_split_relpos_pairbias_full_vs_anchor_only.json`

Relative position versus no spatial input:

- multi-cell relpos: `2.0558`
- multi-cell no-space: `2.0825`
- single-cell relpos: `1.8503`
- single-cell no-space: `1.9456`

Pairwise bias on top of relative position:

- relpos full: `2.0558`
- pairwise-bias full: `2.0369`
- relpos anchor-only: `2.0733`
- pairwise-bias anchor-only: `2.0577`

Context-use decomposition:

- relpos `full - anchor_only`: `-0.0175`
- pairwise-bias `full - anchor_only`: `-0.0208`

Interpretation:

- continuous relative geometry was a better spatial signal than no-space or
  coarse distance buckets,
- pairwise spatial bias gave a further small but consistent gain,
- spatial encoding was therefore retained for the next stage,
- but token-level matching still did not make multi-cell clearly win overall.

## 9. Naive Local-Group Supervision Failed, But Matched Local Patch Was Better

Files:

- `eval_anchor_split_pairbias_local_group_v2_full.json`
- `eval_anchor_split_pairbias_local_group_v2_anchor_only.json`
- `eval_anchor_split_pairbias_matched_patch_v2_full.json`
- `eval_anchor_split_pairbias_matched_patch_v2_anchor_only.json`
- `eval_anchor_split_matched_patch_v2_single.json`
- `error_compare_anchor_split_pairbias_local_group_v2_full_vs_anchor_only.json`
- `error_compare_anchor_split_pairbias_matched_patch_v2_full_vs_anchor_only.json`
- `error_compare_anchor_split_matched_patch_v2_multi_vs_single.json`

Local-group:

- full: `2.6613`
- anchor-only: `2.7430`
- mean context gain: `-0.0858`
- `match_rate = 0.8`
- `split_rate = 0.0`

Matched local patch:

- multi-cell full: `3.0887`
- multi-cell anchor-only: `3.1292`
- mean context gain: `-0.1070`
- `match_rate = 1.0`
- `split_rate = 0.1`

Single-cell matched local patch:

- single-cell full: `2.9640`
- `multi - single = +0.2844`

Interpretation:

- naive local-group supervision diluted the task,
- matched local patch was a cleaner patch-level target than local-group,
- but hard token matching still left single-cell ahead,
- this was the point where patch-level work clearly needed a set-level target.

## 10. Patch-To-Patch Set Prediction Was The First Objective To Favor Multi-Cell

Files:

- `eval_patch_set_anchor_split_relpos_pairbias_multi_full.json`
- `eval_patch_set_anchor_split_relpos_pairbias_multi_anchor_only.json`
- `eval_patch_set_anchor_split_relpos_single_full.json`

Formal `context_size = 64` comparison:

- multi-cell full: `114.38`
- multi-cell anchor-only: `117.82`
- single-cell full: `116.92`

Key deltas:

- `multi - single total = -2.54`
- `multi - single ot = -0.14`
- `multi - single latent = -0.040`
- `multi - single size = -120.03`
- `multi full - anchor_only = -3.44`

Interpretation:

- once the task became patch-to-patch set prediction, multi-cell finally
  outperformed the matched single-cell control,
- the context ablation gap also became larger than in the earlier
  token-matching experiments,
- this was the first strong sign that the project had moved onto a more
  biologically aligned objective.

## 11. Scaling Patch Size Strengthened The Multi-Cell Advantage

Files:

- `eval_patch_set_anchor_split_ctx128_relpos_pairbias_multi_full.json`
- `eval_patch_set_anchor_split_ctx128_relpos_single_full.json`
- `eval_patch_set_anchor_split_ctx256_relpos_pairbias_multi_full.json`
- `eval_patch_set_anchor_split_ctx256_relpos_single_full.json`

Patch-set scaling sweep:

- `context_size = 64`
  - multi: `114.38`
  - single: `116.92`
  - delta: `-2.54`
- `context_size = 128`
  - multi: `370.78`
  - single: `373.46`
  - delta: `-2.68`
- `context_size = 256`
  - multi: `1336.25`
  - single: `1349.95`
  - delta: `-13.70`

OT deltas:

- `64`: `-0.14`
- `128`: `-0.38`
- `256`: `-0.77`

Interpretation:

- the absolute losses grow with patch size because the patch-size regression
  term scales with the target cardinality,
- but within each fixed scale, the multi-cell model remains better,
- the advantage becomes markedly larger by `context_size = 256`,
- this is the clearest current evidence that larger developmental context is
  becoming more useful under a set-level objective.

## Current Readout

The current project status is therefore:

- token-level matched prediction was useful for diagnosing supervision issues,
  but it did not unlock clear multi-cell gains,
- set-level patch prediction is the first objective that consistently favors
  multi-cell modeling,
- scaling patch size strengthens that advantage,
- the most immediate technical issue is now loss rescaling across patch sizes,
  not whether larger context matters at all.

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

## 9. Naive Local-Group Supervision Did Not Yet Produce A Good Population Task

Files:

- `eval_anchor_split_local_group_multi_full.json`
- `eval_anchor_split_local_group_multi_anchor_only.json`
- `eval_anchor_split_local_group_single.json`
- `error_compare_anchor_split_local_group_multi_vs_single.json`
- `error_compare_anchor_split_anchor_only_vs_local_group_multi.json`

Aggregate:

- local-group multi full: `3.3798`
- local-group multi anchor-only eval: `3.5049`
- local-group single: `3.1734`

Decomposition:

- `mean_delta_total = +0.3612`
- `mean_delta_gene = +0.0114`
- `mean_delta_split = +0.1483`
- `mean_delta_del = +0.2015`

Comparison to the older anchor-only multi-cell setup:

- `mean_delta_total = -0.1163`
- `mean_delta_split = -0.0463`
- `mean_delta_del = -0.0700`

Interpretation:

- simply supervising `anchor + a few neighbors` did not become a useful local
  population-update task,
- supervision quality dropped sharply,
- most of the damage appeared in the event heads.

What this established:

- a future patch-level task should not supervise arbitrary nearby cells,
- the next group-prediction task should supervise matched local cells rather
  than every local token.

## 10. Relative Spatial Encoding Improved The Clean Split-Rich Baseline

Files:

- `eval_anchor_split_relpos_multi.json`
- `eval_anchor_split_nospatial_multi.json`
- `eval_anchor_split_relpos_single.json`
- `eval_anchor_split_nospatial_single.json`

Configuration:

- `anchor_split_rich`
- `strict delete`
- `anchor_only` supervision
- same model size and data construction,
- only `spatial_input_mode` changed:
  - `none`
  - `relative_position = (dx, dy, dz, r, has_spatial)`

Aggregate:

- multi-cell with relative position: `2.0558`
- multi-cell with no spatial input: `2.0825`
- single-cell with relative position: `1.8503`
- single-cell with no spatial input: `1.9456`

Interpretation:

- continuous relative spatial encoding was clearly better than dropping spatial
  information,
- most of the gain did not come from the gene term,
- space behaved more like a geometric prior for event prediction than like a
  direct transcriptomic-state improvement.

What this established:

- spatial information should remain in the active path,
- anchor-distance buckets were not the right long-term representation,
- continuous relative geometry was a better default.

## 11. Pairwise Spatial Bias Added A Further Multi-Cell Gain

Files:

- `eval_anchor_split_relpos_pairbias_full.json`
- `eval_anchor_split_relpos_pairbias_anchor_only.json`
- `eval_anchor_split_relpos_multi.json`
- `eval_anchor_split_relpos_anchor_only.json`

Configuration:

- same `anchor_split_rich + strict delete + relative_position` setup,
- multi-cell only,
- added learned pairwise spatial bias in attention scores.

Aggregate:

- relative-position full: `2.0558`
- relative-position + pairwise bias full: `2.0369`
- relative-position anchor-only: `2.0733`
- relative-position + pairwise bias anchor-only: `2.0577`

Interpretation:

- pairwise spatial bias gave a small but real improvement beyond token-level
  relative coordinates,
- the gain was modest, but directionally consistent,
- this fit the idea that local patch geometry matters through relationships
  between cells, not only per-token features.

What this established:

- pairwise geometry is worth keeping in the multi-cell path,
- the active spatial stack should now be thought of as:
  - token-level relative coordinates,
  - optional pairwise spatial bias in attention.

## 12. Pairwise Spatial Bias Slightly Increased Measured Context Use

Files:

- `error_compare_anchor_split_relpos_full_vs_anchor_only.json`
- `error_compare_anchor_split_relpos_pairbias_full_vs_anchor_only.json`

Window-level comparison:

- relative-position:
  - `mean_delta_total = -0.0175`
  - `mean_delta_split = -0.0084`
  - `mean_delta_del = -0.0091`
- relative-position + pairwise bias:
  - `mean_delta_total = -0.0208`
  - `mean_delta_split = -0.0099`
  - `mean_delta_del = -0.0110`

Here negative means `full` outperformed `anchor_only`.

Interpretation:

- pairwise bias did not only lower aggregate loss,
- it also slightly increased the measurable benefit of full context over
  anchor-only context,
- the effect remained concentrated in event terms rather than the gene term.

What this established:

- pairwise spatial bias is a reasonable default for the next patch-level
  population experiments,
- the project now has a stronger spatially informed local-update baseline to
  build on.

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
7. Spatial information is useful, and continuous relative geometry is a better
   default than coarse distance buckets.
8. Pairwise spatial bias gives a small but consistent additional gain and
   slightly strengthens measured context use.
9. None of these results yet justify claims about embryo-scale rollout from the
   zygote.

## 13. Patch-Set Prediction Became The Default Active Objective

Files:

- `eval_patch_set_anchor_split_relpos_pairbias_multi_full.json`
- `eval_patch_set_anchor_split_relpos_pairbias_multi_anchor_only.json`
- `eval_patch_set_anchor_split_relpos_single_full.json`
- `eval_patch_set_anchor_split_ctx128_relpos_pairbias_multi_full.json`
- `eval_patch_set_anchor_split_ctx128_relpos_single_full.json`
- `eval_patch_set_anchor_split_ctx256_relpos_pairbias_multi_full.json`
- `eval_patch_set_anchor_split_ctx256_relpos_single_full.json`

What changed:

- token-level matched supervision stopped being the main active objective,
- patch-to-patch set prediction became the default active benchmark,
- the project started comparing local population states rather than focal-cell
  token matches.

Main result:

- `context_size = 64`
  - multi-cell: `114.38`
  - single-cell: `116.92`
- `context_size = 128`
  - multi-cell: `370.78`
  - single-cell: `373.46`
- `context_size = 256`
  - multi-cell: `1336.25`
  - single-cell: `1349.95`

Interpretation:

- patch-set prediction was the first objective on which multi-cell became
  clearly better than the matched single-cell control,
- the advantage increased as patch size grew,
- this made patch-set prediction the first objective that really aligned with
  the roadmap's "expand context toward embryo-scale state" logic.

## 14. Readout Was Refactored From Raw Loss To Interpretable Patch Metrics

Files:

- `patch_set_scale_readout_summary.json`
- the patch-set evaluation JSONs listed above, now with expanded fields

What changed:

- raw `total` loss was no longer treated as sufficient across patch scales,
- evaluation added normalized metrics:
  - `normalized_total`
  - `total_wo_size`
  - `ot_per_token`
- evaluation added composition readouts:
  - `mean_gene_rmse`
  - `mean_gene_cosine`
  - `pca_mean_dist`
  - `pca_var_dist`
- evaluation added diversity readouts:
  - `diversity_abs_error`
  - `entropy_abs_error`
- evaluation added patch-level split summary:
  - `current_split_fraction`
  - `future_split_fraction`
  - `split_fraction_shift`

What this established:

- patch-size regression had been dominating absolute totals too strongly for
  cross-scale interpretation,
- within-scale comparisons remained valid, but needed more interpretable
  breakdowns,
- multi-cell currently looks strongest on:
  - future set alignment,
  - latent alignment,
  - diversity recovery,
  - entropy recovery,
  - PCA variance recovery,
- while single-cell can remain competitive on some mean-centered summaries.

Interpretation:

- the project now has a more biologically interpretable readout layer for
  future local population state prediction,
- but these are still project-specific research readouts rather than
  community-standard biological endpoints.

## What This History Does Not Claim

This experiment history does **not** show that we have already learned a full
developmental simulator.

It does **not** establish:

- long-horizon single-cell trajectory prediction,
- stable whole-embryo rollout,
- a biologically closed multi-step developmental system,
- zygote-to-embryo prediction.

Those remain later-phase goals in the roadmap.
