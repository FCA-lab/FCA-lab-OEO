# Reproducing the classification paper

Documentation snapshot: 2026-10-29. This guide replaces the earlier provisional experimental design. The paper is a methods study; BABD and Elliptic are application cases. Legacy Bitcoin entity-split experiments are not part of this protocol.

The [artifact and version map](paper_artifacts.md) identifies the reported results and historical code. Commands below describe future execution; none were executed during this documentation update. A current-code rerun is not automatically an exact replay of a historical source snapshot.

## 1. Environment and inputs

Install from the repository root as described in [README](../../README.md). Full BABD defaults require `catboost` in addition to the `plot,boost` extras. The historical task metadata record Python 3.13.14, NumPy 2.4.6, pandas 3.0.3, scikit-learn 1.9.0, threadpoolctl 3.6.0, openpyxl 3.1.5 and XGBoost 3.2.0; BABD also records CatBoost 1.2.10 and LightGBM 4.6.0. These are recorded environment values, not a tested lockfile or a claim that installation has been independently reproduced. Consult the individual checkpoint for its complete environment and execution settings.

All paths below are relative to `OEO_DATA_ROOT` (default: repository `data/`).

| Experiment | Input and preparation evidence |
|---|---|
| Completeness / UCI | Files under `raw/uci/`; dataset-specific files, labels and excluded ID columns are specified in `completeness_config.py` and `uci_config.py`. Preserve the checkpoint input hashes. |
| BABD | `raw/public/babd13/BABD-13-clean.csv` and `cleaning_report.json`. The report records 544,462 original rows, removal of 3,216 conflicting-label rows and 292 duplicate rows, leaving 540,954 rows and 148 features. Clean CSV SHA-256: `d380c00f6528f225c6840cb78c07fd28fa6794a98b9ed20a5c3eafcb6109f4e3`. A standalone, verified raw-to-clean preparation recipe still needs packaging. |
| Elliptic | `raw/public/elliptic/elliptic_txs_features.csv` and `elliptic_txs_classes.csv`; input acquisition checksums are recorded in `audit.json`. Downloads: [features ZIP](https://data.pyg.org/datasets/elliptic/elliptic_txs_features.csv.zip), [classes ZIP](https://data.pyg.org/datasets/elliptic/elliptic_txs_classes.csv.zip). |

Dataset download/version/license information must accompany the public package. Do not redistribute inputs merely because they were publicly downloadable.

## 2. Paper protocols

All predictive preprocessing is fitted on training objects only. Models within an experimental condition share sample splits. Numerical features use the project's dynamic binning implementation followed by one-hot encoding; categorical features use one-hot encoding. `max_bins` is an upper bound, not the actual number of intervals. The implementation uses `MIN_DATA_IN_BIN=3` (samples, not bins). Decision labels and identifiers are excluded from condition features.

| Study | Settings used for the paper |
|---|---|
| Completeness | Balloons, Lenses, COVID19 and Iris, each used in full. Iris uses `max_bins=5`. Direct extraction and extraction from all 3WOC concepts use the same rule definitions and extraction conditions. Compare distinct expanded single-literal rules, including empty-support rules. |
| UCI | Iris, Wine, WDBC, Balloons, Lenses; seeds 11/37/53; stratified five-fold CV except Lenses four-fold. Comparison uses full rules and uniform voting; sensitivity uses full rules and uniform/nonzero/support voting with numerical bin limits 20,30,...,100. Ablation uses bin limit 20 and uniform voting with seven rule configurations. Categorical datasets are not binned. |
| BABD balanced | Original class IDs `[0,1,2,3,5,6,11]`; address-level holdout, 1,000 test objects per class; training objects per class 50/100/200/500/1000/2000/4000 (total 350–28,000). |
| BABD imbalanced | Original class IDs `[0,1,2,3,4,5,6,9,10,11,12]`; address-level stratified five-fold CV; total training sizes 500/1000/2000/5000/10000/20000/30000, sampled proportionally within each training fold. |
| Both BABD studies | 148 cleaned condition features; seeds 11/37/53; bin limits 5/10/20/30/50; full rules, uniform voting. All baseline models use the same binary contexts. Address-level splitting does not establish entity-disjoint evaluation. |
| Elliptic | 93 local features (zero-based columns 2–94), excluding IDs, time and graph edges. Exclude 157,205 unknown-label objects. Steps 1–34: 29,894 training objects (26,432 licit / 3,462 illicit); steps 35–49: 16,670 test objects (15,587 / 1,083). Seeds 11/37/53 do not change this temporal split. Bin limits 5/10/20/30/50; uniform voting. Workbook also includes supporting-only and excluding-only variants; the paper evidence table uses full rules. |

UCI model specifications are in `uci_config.py`: RC-OEO, DT, RF and XGBoost have binary-context comparisons; raw-feature RF/XGBoost and standardized SVM are supplementary references. The majority baseline ignores features. BABD and Elliptic model specifications are in their own configurations; do not infer their input representation from UCI model names.

Seven UCI rule groups are `all={A1,A2,B1,B2}`, `support_only={A1,A2}`, `exclusion_only={B1,B2}`, and `without_A1`, `without_A2`, `without_B1`, `without_B2`. Rule counts refer to distinct single-literal rules, not the number of compact disjunctive expressions.

## 3. Metrics, aggregation and timing

- Classified accuracy is correct unique predictions divided by all unique predictions; undefined values remain missing when none are assigned. Coverage is unique predictions divided by all test objects.
- Macro-Precision and Macro-F1 average over the task's true classes. Undecided predictions contribute a false negative to their true class and no false positive to any class. Undefined class-wise precision/F1 is zero; undecided is not an additional class.
- For CV, pool confusion counts across test folds within each seed before computing metrics. Report the mean and sample standard deviation across the three seeds, not an average of fold percentages. Balanced BABD and Elliptic use their holdout/temporal test sets per seed. Check defined-seed counts for undefined metrics.
- BABD mean candidate-set size and true-class inclusion use **all test objects**, including uniquely classified objects. They must not be described as metrics restricted to unresolved objects. UCI candidate-size plots separately display set sizes and whether the true class is included.
- Workbooks distinguish preprocessing, fitting and prediction times. The paper's BABD runtime plot excludes preprocessing, uses `max_bins=20`, averages fitting time over folds where applicable and normalizes prediction time per 1,000 test objects. Shared RC-OEO fitting time is not added across voting variants. Reused checkpoint times are original measurements, not new measurements under later scheduling settings.
- Completeness direct timing covers construction/fit; the all-concepts route includes concept construction and rule extraction. Comparison, semantic checking, saving and input preprocessing are excluded. Equality of the tested rule sets is empirical validation on these contexts, not a replacement for the theorem.

## 4. Execution reference

Use a separate results directory to preserve the paper workbooks. Commands use one worker as a conservative execution choice; this is not a claim to reproduce original wall times. Environment, source or execution changes can change task identities and trigger computation instead of reuse.

```sh
PYTHONPATH=src python -m oeo_lab.experiments.classification.completeness \
  --datasets balloons lenses covid19 iris --max-bins 5 --n-jobs 1 \
  --results-root results/reproduction/completeness

PYTHONPATH=src python -m oeo_lab.experiments.classification.uci_experiment \
  --experiments comparison sensitivity ablation \
  --max-bins 20 30 40 50 60 70 80 90 100 --seeds 11 37 53 \
  --fixed-voting uniform --voting uniform nonzero support --ablation-bins 20 \
  --n-jobs 1 --results-root results/reproduction/uci

PYTHONPATH=src python -m oeo_lab.experiments.classification.babd_experiment \
  --experiments balanced imbalanced --max-bins 5 10 20 30 50 \
  --seeds 11 37 53 --voting uniform --n-jobs 1 \
  --results-root results/reproduction/babd

PYTHONPATH=src python -m oeo_lab.experiments.classification.elliptic_experiment \
  --feature-set local --train-end 34 --test-start 35 --test-end 49 \
  --max-bins 5 10 20 30 50 --seeds 11 37 53 --voting uniform \
  --n-jobs 1 --results-root results/reproduction/elliptic
```

These commands use each current entry point's remaining defaults, as described in the protocol table. In particular, UCI `main` overrides the legacy grid in `uci_config.py`; completeness has script-level defaults that are overridden explicitly above. Preserve the release snapshot rather than relying on defaults in a future revision. `--dry-run` still loads/validates inputs and constructs tasks; it is not a file-inventory command. Do not run concurrent writers against the same results directory.

The optional root launcher defaults to an external watchdog; use `--no-watch` if it is unavailable. The direct module commands above do not require the watchdog. The `oeo-classification` console entry now uses the unified launcher; reinstall the package after updating the entry-point configuration.

## 5. Regenerate figures without training

The shared plotting module reads existing workbooks and writes PDF, PNG and same-basename CSV files. It does not train models. Use a separate output folder for review:

```sh
python analysis/classification/plot_all.py \
  --uci results/classification/uci_three_studies.xlsx \
  --babd results/classification/babd_balanced_results.xlsx \
  --babd-imbalanced results/classification/babd_imbalanced_results.xlsx \
  --output results/figures/reproduction \
  --only uci_model_comparison_grid uci_rule_ablation_combined \
         uci_voting_model_comparison uci_candidate_set_sizes \
         babd_behavior_grid babd_runtime_comparison
```

`plot_all.py` selects jobs; `result_data.py` loads/aggregates results; `figures.py` renders them. Retain the generated manifest with the CSV/PDF/PNG files. Table values are selected from the workbooks using the filters in the artifact map; a complete automatic LaTeX-table export pipeline has not been verified.

## 6. Checkpoints and release limitations

Classification workbooks contain `folds`, `per_seed`, `summary`, `tasks` and `rules` sheets (large rule exports are split across sheets). Completeness uses `results` and `differences`. Checkpoints retain task settings, source/environment provenance, counts and rule evidence; a separate per-object prediction file is not part of the established artifact package.

Retain indexes, referenced checkpoints and every source checkpoint reachable through reuse links. Index provenance can describe the current task identity while a reused record retains earlier execution provenance; neither should be relabeled as a fresh run. See the [version audit](paper_artifacts.md#historical-source-and-reuse-audit).

Before public release: freeze the reviewed source snapshot; package dependency versions, deterministic input preparation and checksums; include the selected workbooks and final figures/CSVs; preserve historical source fingerprints and reuse records; choose a license and archival identifier. No clean-environment reproduction or new experiment was performed as part of this documentation update.
