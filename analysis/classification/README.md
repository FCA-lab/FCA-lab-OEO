# Classification experiments: from execution to the six paper figures

This guide describes the existing workflow for **Classification via Directly
Extracted Necessary-Attribute Rules**. No manual spreadsheet filtering or CSV
preparation is required. Run commands from the repository root.

For input preparation, metric definitions and historical versions, see the
[reproduction guide](../../docs/classification/reproduction.md) and
[paper artifact map](../../docs/classification/paper_artifacts.md).
The separate [3WOC construction workflow](../../src/oeo_lab/experiments/construction/README.md)
uses its own experiment and plotting entry points.

## 1. Complete workflow

```text
Prepare the documented input datasets and Python environment
    -> run UCI and BABD classification experiments
    -> generate comprehensive result workbooks
    -> plot_all.py reads each workbook's per_seed sheet
    -> select_results() selects configured experimental conditions
    -> figure-specific data functions validate and organize the selected records
    -> rendering functions create the six figures
    -> save PDF / PNG, same-basename CSV and a generation manifest
```

The CSV files are outputs of the plotting workflow, not required inputs to the
current command. The existing command reads Excel workbooks and has no generic
`--input`, `--csv` or `--figure` option. Use `--only` to select figure names.

## 2. Environment and experiment commands

Install the project in a dedicated environment, with plotting and baseline
model dependencies. The complete BABD grid additionally requires CatBoost:

```sh
python -m pip install -e '.[plot,boost]'
python -m pip install catboost
```

These installation commands do not pin the historical environment. Consult the
reproduction guide and checkpoint metadata for the recorded versions.
On D30, first enter the repository and activate the existing environment:

```sh
cd /path/to/FCA-lab-OEO
conda activate fca-oeo
```

On another machine, use its checkout path and Python environment. Inspect the
requested tasks without training if needed:

```sh
python run_experiment.py uci --dry-run --n-jobs 10
python run_experiment.py babd --dry-run --n-jobs 10
```

A dry run reads data and constructs tasks; it is not a file-inventory operation.
To generate the three workbooks needed for all six figures:

```sh
# UCI: performance comparison, rule ablation and binning/voting sensitivity.
python run_experiment.py --no-watch uci --n-jobs 10

# BABD: balanced and imbalanced studies, executed in that order by default.
python run_experiment.py --no-watch babd --n-jobs 10

# Generate all six paper figures from the completed workbooks.
python analysis/classification/plot_all.py
```

`--no-watch` avoids the external experiment-watch service. `--n-jobs 10` limits
concurrent task processes; it does not give each model ten threads. Memory guards
may reduce actual concurrency. Ten workers is an example, not a resource guarantee.
Successful compatible checkpoints can be reused. Do not start multiple writers
against the same output directory. Subset runs may replace full reports; use a
separate `--results-root` for exploratory runs and pass those workbook paths to
the plotter explicitly.

Other experiments required by the paper produce tables/evidence rather than
inputs to these six figures:

```sh
python run_experiment.py --no-watch completeness --max-bins 5 --n-jobs 1
python run_experiment.py --no-watch elliptic --n-jobs 10
```

Do not use `all` as shorthand for the exact paper configuration: it also includes
the independent training-size study and different default completeness binning.
The documented UCI grid contains 2,016 training tasks; BABD contains 840 balanced
and 4,200 imbalanced tasks. Prediction variants can share one fit and must not be
counted as additional training tasks. See the root launcher's guide for formulas.

## 3. Comprehensive result files and plotting commands

Default workbooks are under `results/classification/`:

| Workbook | Role |
|---|---|
| `uci_three_studies.xlsx` | Source of the four UCI figures. |
| `babd_balanced_results.xlsx` | Balanced BABD source for both BABD figures. |
| `babd_imbalanced_results.xlsx` | Imbalanced BABD source for both BABD figures. |
| `completeness_results.xlsx` | Rule-completeness tables/checks, not a six-figure input. |
| `elliptic_results.xlsx` | Elliptic tables/evidence, not a six-figure input. |

Classification workbooks include `folds`, `per_seed`, `summary`, `tasks` and rule
sheets. The plotter reads **per_seed**, not summary or raw test records. Fold
counts have already been pooled within each seed by the experiment pipeline;
the plotter does not recompute fold metrics.

If these workbooks are already available, skip training and run the plotter.
Use explicit paths for downloaded artifacts or separately stored rerun results:

```sh
python analysis/classification/plot_all.py \
  --uci /path/to/uci_three_studies.xlsx \
  --babd /path/to/babd_balanced_results.xlsx \
  --babd-imbalanced /path/to/babd_imbalanced_results.xlsx \
  --output /path/to/figures \
  --dpi 300
```

For example, plot only the model comparison and rule ablation:

```sh
python analysis/classification/plot_all.py \
  --only uci_model_comparison_grid uci_rule_ablation_combined
```

Default output: `results/figures/classification/`. Each figure produces `.pdf`,
`.png` and `.csv` files with the same basename. Full runs write `manifest.json`;
selected runs write `manifest_selected.json`. Existing same-name files are
overwritten. Use a separate output folder when checking new results.
Inspect manifest statuses: validation failures are reported as skipped figures,
and the command returns a nonzero exit status if any requested figure was skipped.
An old output file is not evidence of a successful current generation.

## 4. Selection logic for the six figures

Selection follows predefined experimental conditions, not the best observed
performance. Numerical bin limits are upper bounds, not actual interval counts.
Categorical data have no numerical bin setting.

| Figure basename | Selected records | Displayed information |
|---|---|---|
| `uci_model_comparison_grid` | UCI `comparison`; Iris/Wine/WDBC; binary-input DT, RF, XGBoost and RC-OEO; bin limits 20,30,...,100. RC-OEO uses full rules and uniform voting. | Classified accuracy and Macro-Precision. |
| `uci_rule_ablation_combined` | UCI `ablation`; Iris/Wine/Lenses/WDBC/Balloons; RC-OEO, uniform voting, seven rule groups; numerical bin limit 20. | Classified accuracy and coverage for each rule configuration. |
| `uci_voting_model_comparison` | Baselines from `comparison`; full-rule RC-OEO from `sensitivity`; Iris/Wine/WDBC; bin limits 20--100; uniform, nonzero-support and support-weighted voting. | Classified accuracy and coverage. |
| `uci_candidate_set_sizes` | UCI `comparison`; Iris/Wine/Lenses; full-rule RC-OEO, uniform voting; numerical bin limit 20. | Candidate sizes 1/2/3, split by whether the true class is included. |
| `babd_behavior_grid` | Both BABD workbooks; RC-OEO, uniform voting, bin limit 20; all seven training sizes in each study. | Rule count, classified accuracy, coverage, mean candidate-set size and true-class inclusion. |
| `babd_runtime_comparison` | Both BABD workbooks; RC-OEO, RF and XGBoost; bin limit 20; all seven training sizes. | Fitting time per fold and prediction time per 1,000 test objects. |

`select_results()` applies the configured filters and seeds `[11,37,53]`, rejects
empty selections and duplicate experimental-setting/seed records, and checks
complete folds and seeds. Figure-specific functions additionally check the
required bin grid, rule groups, parameter separation or count identities.

- Curves and error bars summarize seeds using equal-seed means and sample SD;
  SD is not a confidence interval. Undefined classified accuracy is not zero-filled.
- Ablation displays absolute metrics. Its exported CSV also contains paired
  differences from the full rule set, but those differences are not the plotted axes.
- The candidate-size function reconstructs exact sizes only for the three verified
  three-class datasets. It uses recorded counts, normalizes within each seed and
  checks consistency; it does not infer a size distribution from the mean size.
- BABD studies remain separate during aggregation. Mean candidate-set size and
  true-class inclusion use all test objects, including unique predictions.
- BABD timing excludes preprocessing. Per-seed summed fitting time is divided by
  completed folds; prediction time is divided by test count and multiplied by
  1,000 before seed aggregation. Shared fitting times are not added across voting
  variants. Reused checkpoint timings belong to the original executions.

The plotting workflow derives CSV data from the comprehensive workbooks
itself. Reproducers do not need to select rows, design a spreadsheet or fill an
empty CSV template. This version does not offer a separate CSV-only export mode
or a CSV-input plotting command.

## 5. Functions and parameter locations

| Figure | Data function in `result_data.py` | Renderer in `figures.py` |
|---|---|---|
| Model comparison | `comparison_grid_data` | `plot_comparison_grid` |
| Rule ablation | `ablation_grid_data` | `plot_ablation_combined` |
| Voting comparison | `voting_comparison_data` | `plot_voting_comparison` |
| Candidate sizes | `candidate_size_data` | `plot_candidate_sizes` |
| BABD behavior | `babd_behavior_data` | `plot_babd_behavior` |
| BABD runtime | `babd_timing_data` | `plot_babd_timing` |

| Setting | Where to change it |
|---|---|
| Experiment selection, parallelism and explicit overrides | `run_experiment.py` command-line arguments; dispatched by `experiments/cli.py`. |
| UCI study grid, voting modes, seeds and ablation defaults | `src/oeo_lab/experiments/classification/uci_experiment.py`, `main()`. |
| UCI dataset definitions, model specifications and rule groups | `uci_config.py` in the same directory. Lenses specifies four folds. |
| BABD sizes, bins, seeds and selected studies | `babd_experiment.py`, `main()`. |
| BABD class IDs, input path and model specifications | `babd_config.py`. |
| Elliptic / completeness settings | Their own experiment entries and configuration files; separate from plotting. |
| Plot input paths, destination, DPI and figure subset | `plot_all.py` options: `--uci`, `--babd`, `--babd-imbalanced`, `--output`, `--dpi`, `--only`. |
| Plot seeds, dataset/model filters, bin grids, metrics, labels, colors and formats | `plot_all.py`, `main()` (`settings` and `jobs`); not generic command-line options. |
| Statistical transformations and validity checks | `result_data.py`; some expected grids and supported datasets are fixed in these functions. |
| Layout, panel styles and some axis limits | `figures.py`, with some settings passed from `plot_all.py`. |

Experiment configuration is overridden by entry-point defaults and then explicit
CLI arguments. In particular, UCI `main()` uses 20--100 despite an earlier grid
remaining in `uci_config.py`. Changing experimental grids does not automatically
change plotting expectations; align both deliberately. No code changes are
needed for the documented paper configuration.

The six figures describe the classification paper's experiments, not its concept
structure illustrations. Earlier trial plotting functions were removed; their
history is preserved at Git commit `1757a97`. Regenerated runtimes may differ
from the published values when the execution environment changes.
