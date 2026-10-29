# Paper artifact and version map

Audit/documentation date: 2026-10-29. Paper: **Classification via Directly Extracted Necessary-Attribute Rules**. Paths below are repository-relative unless stated otherwise. Figure basenames are used instead of figure numbers, which may change during typesetting.

## Experiment-to-result map

Configurations and entries in this table are under `src/oeo_lab/experiments/classification/`; workbooks are under `results/classification/`. Run settings in entry-point `main` and explicit command-line arguments override configuration defaults. Checkpoint provenance is authoritative for an actual execution.

| Paper experiment | Entry / configuration | Workbook | Recorded completion and scope |
|---|---|---|---|
| Rule completeness | `completeness.py` / `completeness_config.py` | `completeness_results.xlsx` | Four successful contexts: Balloons, Lenses, COVID19, Iris. Four result rows; no rule differences. Iris bin limit 5. |
| UCI comparison, ablation, sensitivity | `uci_experiment.py` / `uci_config.py` | `uci_three_studies.xlsx` | 2,016 successful training tasks; 3,816 fold/variant rows, 774 per-seed rows, 258 summary rows. Prediction variants share fits, so row count is not task count. |
| BABD balanced | `babd_experiment.py` / `babd_config.py` | `babd_balanced_results.xlsx` | 840 successful tasks; 840 fold rows, 840 per-seed rows, 280 summary rows. Seven training sizes, five bin limits, eight models, three seeds. |
| BABD imbalanced | Same BABD entry/configuration | `babd_imbalanced_results.xlsx` | 4,200 successful tasks; 4,200 fold rows, 840 per-seed rows, 280 summary rows. Five-fold counterpart of the above grid. Approximately 688 MB; rule export spans 11 sheets. |
| Elliptic | `elliptic_experiment.py` / `elliptic_config.py` | `elliptic_results.xlsx` | 60 successful training tasks; 90 fold/variant rows, 90 per-seed rows, 30 summary rows. Full/supporting-only/excluding-only RC-OEO predictions share fits. |

Checkpoint directory: `results/classification/checkpoints/`. Classification indexes include `uci_index.json`, `babd_balanced_index.json`, `babd_imbalanced_index.json` and `elliptic_index.json`. The imbalanced index uses format version 2 and references `babd_imbalanced_index_data/`; preserve its referenced records as well as historical source checkpoints.

`uci_results.xlsx`, `bitcoin_results.xlsx`, extended WDBC binning/TabPFN outputs and separate training-size experiments are historical or exploratory outputs, not substitutes for these five paper workbooks.

## Table-to-data map

| Paper table / evidence | Source | Selection and interpretation |
|---|---|---|
| Completeness dataset/context and extraction results | Completeness `results`; checkpoint context metadata | Four contexts above; compare expanded, deduplicated single-literal rule sets under the same definitions. Timing excludes preprocessing and checking. |
| UCI performance (`tab:uci_performance`) | UCI `summary`, checked against `per_seed` | `study=comparison`; numerical `max_bins=20`, categorical unbinned. RC-OEO `rule_group=all`, `voting=uniform`. Include binary-context baselines, majority and the stated raw-feature references; distinguish their inputs. Classified accuracy, Macro-Precision, coverage and Macro-F1: mean and sample SD over seeds. |
| BABD balanced performance | Balanced `summary` / `per_seed` | Keep the manuscript's selected training-size/bin/model rows; bin limit 5 was excluded from the final performance tables, but remains in the source workbook. Do not choose a different voting mode per row. |
| BABD imbalanced performance | Imbalanced `summary` / `per_seed` | Same selection principle; metrics pooled across folds within each seed. Retain low coverage and Macro-F1 alongside classified accuracy. |
| Elliptic rule evidence and class recall | Elliptic full-rule RC-OEO records in `rules`, `folds` and checkpoints | Uniform voting, bin limits 5/10/20/30/50. Count by rule type **and consequent class**; a total A1/A2/B1/B2 count alone does not identify licit versus illicit evidence. Recall uses all objects of the true class, including undecided cases. |

The final LaTeX performance-table row selections and formatting remain in the manuscript source; an automatic table export has not been verified. These workbooks are the numerical sources, not a guarantee that every workbook row is printed in the paper.

## Figure-to-data and function map

All jobs are selected by `analysis/classification/plot_all.py`. Data functions are in `analysis/classification/result_data.py`; rendering functions are in `analysis/classification/figures.py`. Each figure's PDF, PNG and CSV share its basename.

| Basename | Workbook and selection | Data / rendering function |
|---|---|---|
| `uci_model_comparison_grid` | UCI comparison; Iris/Wine/WDBC; bins 20–100; binary DT/RF/XGBoost/RC-OEO; RC-OEO full rules, uniform voting | `comparison_grid_data` / `plot_comparison_grid` |
| `uci_rule_ablation_combined` | UCI ablation; five datasets; seven rule groups; uniform voting; numerical bin limit 20 | `ablation_grid_data` / `plot_ablation_combined` |
| `uci_voting_model_comparison` | UCI sensitivity plus comparison references; Iris/Wine/WDBC; three RC-OEO voting modes, full rules, bins 20–100 | `voting_comparison_data` / `plot_voting_comparison` |
| `uci_candidate_set_sizes` | UCI comparison; Iris/Wine/Lenses; full-rule RC-OEO, uniform voting; bin limit 20 or unbinned categorical | `candidate_size_data` / `plot_candidate_sizes` |
| `babd_behavior_grid` | Both BABD workbooks; RC-OEO, uniform voting, bin limit 20; training-size series | `babd_behavior_data` / `plot_babd_behavior` |
| `babd_runtime_comparison` | Both BABD workbooks; RC-OEO/RF/XGBoost, bin limit 20 | `babd_timing_data` / `plot_babd_timing` |

Default output is `results/figures/classification/`; a selected run also writes `manifest_selected.json` (full runs use `manifest.json`). Preserve manifests and source workbook checksums with released artifacts. See the reproduction guide for a plotting-only command.

**Packaging status:** among these six final figure sets, the mounted default output folder currently contains `uci_rule_ablation_combined.{pdf,png,csv}`. The other five final PDF/PNG/CSV sets are in the author's manuscript workspace, not yet bundled in this repository. This documentation update did not copy or regenerate them. Before release, collect those exact final files, verify their workbook/version correspondence and record checksums. Do not assume older alternative heatmap/dot/tradeoff figures are the final manuscript figures.

## Historical source and reuse audit

The inspected checkout has HEAD `38d219c` and local modifications to the three plotting modules, plus untracked launcher-related files. There is no newly frozen paper-release commit. Do not identify every workbook simply as “generated by 38d219c”.

| Result family | Source evidence | Audit boundary |
|---|---|---|
| Completeness | All four checkpoints share a source fingerprint. All 20 fingerprinted source files were located: use `4e07be0` as the base, with `classifier.py` and `preprocessing/encoding.py` matching `e89b304`. | A mixed historical snapshot, not one clean commit. Static comparison found no relevant rule-extraction change for the tested contexts. No new computation was performed. |
| BABD balanced | All 840 checkpoint source manifests match the 27 recorded source files at `1e6a8d1`. There are 99 records carrying `reuse_audit` and 741 without it. | A representative reuse chain and its original checksum were checked; a complete payload audit of all 99 balanced reuse records has **not** been established. |
| BABD imbalanced | Current index contains 4,200 tasks and its source-file manifest matches the current computation sources. Reused checkpoint records retain earlier provenance. | Complete reuse-chain audit described below; this verifies record preservation, not a fresh execution. |
| UCI / Elliptic | Representative checkpoint source manifests matched the inspected current sources in the prior read-only audit. | Do not treat representative checks as an exhaustive task-level historical audit. Preserve all checkpoint manifests in the release. |

For the completeness mixed snapshot, the two `e89b304` file checksums are:

- `src/oeo_lab/algorithms/classification/classifier.py`: `cb66e84e9c3e8247cf81292e2735a74c55fa499ea2d17ed85e593a5bb34dc504`
- `src/oeo_lab/preprocessing/encoding.py`: `5b5d2c47593194a87fd1c2f437ec68849d0f483e3d620502445985ea587e177b`

The later classifier prediction changes are not exercised by the completeness experiment, which uses fitting and rule expansion. Encoding changes added boundary safeguards; static inspection did not identify an effect on the successful historical test contexts. This is a scoped source audit, not a universal equivalence claim.

### BABD imbalanced: complete reuse-chain findings

The read-only audit traced every task in the final index to its original successful record:

- 4,200 final tasks: 1,723 reused records and 2,477 directly computed records.
- 6,592 distinct checkpoint files and 2,392 `reused_from` links; maximum chain depth two; 4,200 terminal source records.
- All visited records have status `ok`; no missing sources, cycles or files changing during the audit were found.
- The entire result payload matched byte-for-byte across every reuse link. All five recorded timing fields and computational task parameters also matched (excluding task ID and provenance, which may change for reuse).
- Seven raw source-hash cohorts occur in the chain. The current index provenance is not a replacement for those original execution records.

Thus reused rules, predictions as represented by their stored result payloads, aggregate counts and timings were not altered by checkpoint copying. This does not independently recompute per-object predictions, prove algorithm correctness, or establish identical runtime conditions across original executions. Reused timings belong to their original runs and were not remeasured after scheduler changes.

### Relevant Git history

| Commit | Role in the history |
|---|---|
| `e89b304` | Encoding/checkpoint corrections; two files in the completeness snapshot. |
| `4e07be0` | Independent BABD experiment entry and classification evaluation updates; base for the completeness source mapping. |
| `1e6a8d1` | Elliptic entry and model-adapter fixes; exact recorded balanced BABD source snapshot. |
| `dc3ce9f` | Categorical UCI experiments, scheduling and checkpoint reuse changes. |
| `0828dc3` | Empty-support construction regression expectation correction. |
| `38d219c` | Training-size and plotting additions; inspected checkout HEAD, not a complete paper-release identity. |

## Public-package completion checklist

1. Select and freeze the release source, including the intended launcher and final plotting changes; retain historical manifests rather than rewriting them to the release commit.
2. Add a dependency specification and explain the distinction between recorded historical versions and a validated installation environment.
3. Complete dataset download, license, raw-to-clean preparation and input-checksum instructions, especially BABD cleaning.
4. Package the five workbooks and selected final figures/CSVs/manifests; decide whether large checkpoints belong in a separate archive. Preserve the complete reuse graph if checkpoint auditability is claimed.
5. Record final manuscript table selections and artifact checksums. Keep old exploratory outputs out of the paper package.
6. Add the chosen code license and, when actually released, the fixed repository/archive URL. Do not describe a planned public release as already available.

No computation code, configuration, result workbook or checkpoint was changed by this documentation update; no experiments were rerun and no files were uploaded to GitHub.
