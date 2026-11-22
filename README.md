# FCA-lab-OEO

The project code and associated documentation use the [MIT license](LICENSE).
See [data sources](DATA_SOURCES.md) and the [reproduction entry guide](REPRODUCING.md).
See the [reviewer guide](REVIEW_GUIDE.md) for implementation locations and review commands.

Code for **Classification via Directly Extracted Necessary Attribute Rules**.
RC-OEO directly extracts necessary-attribute classification rules from OEO-concepts
and combines supporting and excluding evidence to assign a unique label or retain
candidate classes. The experiments cover rule completeness, UCI classification,
and Bitcoin application cases using BABD and Elliptic.

The repository also includes the independent experimental code for the earlier
[3WOC concept-construction paper](https://doi.org/10.3390/app15126486).
Classification is the main workflow described below.

## Quick verification (no research data required)

From the extracted repository root, copy these commands into a Conda-enabled shell:

```sh
conda create -n fca-oeo-review python=3.13 -y
conda activate fca-oeo-review
python -m pip install -r requirements-reproduce.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m pytest -q
```

Success: **116 passed** and exit code 0. If this environment already exists,
activate it rather than creating it again. Stop if any command fails.
The tests exercise direct/exhaustive rule agreement on tiny contexts, prediction,
metrics, temporary experiment reports/checkpoint reuse, and plotting with test
inputs. They require no dataset downloads and do not reproduce the full paper tables.
The extracted package passed this check on Linux/Python 3.13.14.

For full experiments, follow [data preparation and run commands](docs/classification/reproduction.md#1-environment-and-inputs),
then [plot the generated results](docs/classification/reproduction.md#5-regenerate-figures-without-training).
For the earlier 3WOC synthetic experiments, use the [two-command workflow](src/oeo_lab/experiments/construction/README.md#7-minimal-end-to-end-reproduction-of-the-six-synthetic-figures).

## 1. Environment and dependencies

The reported experiments were run on a Linux workstation (D30). Commands below
assume a shell with Python and Conda available and are run from the repository
root. The package declares Python >=3.9; this is a package requirement, not a
claim that every supported Python/platform combination has been tested.
Historical tasks record Python 3.13.14; detailed versions are in the
[reproduction guide](docs/classification/reproduction.md) and checkpoint metadata.

| Dependency group | Packages | Purpose |
|---|---|---|
| Core | NumPy, pandas, scikit-learn, SciPy, joblib, threadpoolctl, openpyxl | Algorithms, preprocessing, evaluation and Excel results. |
| `plot` extra | Matplotlib | Generate paper figures. |
| `boost` extra | XGBoost, LightGBM | Baseline classifiers. |
| Additional full-grid dependency | CatBoost | BABD baseline; included in the version snapshot. |
| `test` extra | pytest | Run implementation tests. |

Create a new environment, install the recorded dependencies, then install the project:

```sh
# From the checked-out FCA-lab-OEO repository root:
conda create -n fca-oeo python=3.13 -y
conda activate fca-oeo
python -m pip install -r requirements-reproduce.txt
python -m pip install --no-deps --no-build-isolation -e .

# Verify that the active interpreter can import the required dependencies.
python -c "import sys, numpy, pandas, sklearn, scipy, joblib, threadpoolctl, openpyxl, matplotlib, xgboost, lightgbm, catboost, pytest; print(sys.executable)"
```

[requirements-reproduce.txt](requirements-reproduce.txt) records 31 package versions
from the current D30 `fca` environment, audited on 2026-10-30 with Python 3.13.14
on Linux x86_64. It covers experiments, plotting, tests and their Python dependencies;
it is not a complete system lockfile or a record of every historical experiment
environment. Consult the reproduction guide and checkpoint metadata for historical
runs. Installation and all 116 tests were verified from an extracted review source
package in an isolated Linux Python 3.13.14 environment on 2026-10-30.
This does not establish cross-platform compatibility or rerun the paper experiments. The commands above
are intended for a new environment; the existing D30 environment can be activated
with `conda activate fca` without reinstalling packages.

## 2. Prepare the data

Input datasets are not bundled. Follow the download/preparation instructions in
the [reproduction guide](docs/classification/reproduction.md), then provide:

```text
data/
  raw/uci/                         # UCI files specified by the dataset configurations
  raw/public/babd13/
    BABD-13-clean.csv
    cleaning_report.json
  raw/public/elliptic/
    elliptic_txs_features.csv
    elliptic_txs_classes.csv
```

For classification, another data root can be selected before running:

```sh
export OEO_DATA_ROOT=/absolute/path/to/data
```

This directory must contain the same `raw/...` layout. The BABD experiment uses
the cleaned file, not the unprocessed download. Follow the
[BABD preparation guide](docs/classification/babd_preparation.md) to obtain the
recorded raw input and use `scripts/prepare_babd.py`. The tool preserves the
historical cleaning rules and refuses existing output directories. Functional
checks passed; in the same D30 environment, the packaged and original scripts
produce byte-identical CSV and cleaning-report outputs.

## 3. Run the classification experiments

```text
Prepare environment and input datasets
    -> preview the requested tasks
    -> run the selected experiments
    -> comprehensive Excel results and checkpoints
    -> plotter selects and validates the recorded data automatically
    -> six paper figures (PDF/PNG) and their same-basename CSV files
```

The supported source-checkout entry is **`run_experiment.py`**, which dispatches
to the selected experiment module. The installed command `oeo-classification` is
declared in `[project.scripts]` in `pyproject.toml` and generated automatically by
the installation tool. It calls `oeo_lab.experiments.cli.main()`, the same unified
entry used by `run_experiment.py`, with the same arguments. Examples below use only
`run_experiment.py`; no separate experiment workflow is required.

```sh
python run_experiment.py --help
python run_experiment.py uci --help
python run_experiment.py uci --dry-run --n-jobs 1
```

`--dry-run` reads inputs, checks settings and builds tasks without training.
It is not a data-free operation. To run the paper experiments:

```sh
# Rule completeness: explicit bin limit matching the paper.
python run_experiment.py --no-watch completeness --max-bins 5 --n-jobs 1

# UCI: comparison, rule ablation and binning/voting sensitivity.
python run_experiment.py --no-watch uci --n-jobs 1

# BABD: balanced and imbalanced studies, in sequence.
python run_experiment.py --no-watch babd --experiments balanced imbalanced --n-jobs 1

# Elliptic: temporal transaction classification.
python run_experiment.py --no-watch elliptic --n-jobs 1
```

`--no-watch` runs directly without the optional external experiment-watch service.
Omitting it enables supervision and requires that service. To choose interactively,
run `python run_experiment.py --no-watch` in a terminal.

### Parallel execution and task scope

`--n-jobs` is the maximum number of concurrent task **processes**, not the thread
count of one model. Classification tasks limit numerical thread pools to one;
memory guards may lower actual concurrency. Increase the worker limit only when
resources allow. For example:

```sh
python run_experiment.py --no-watch babd --experiments imbalanced --n-jobs 10
# Groups below run sequentially; each group can run up to ten tasks concurrently.
python run_experiment.py --no-watch --batch uci babd elliptic --n-jobs 10
```

| Paper configuration | Requested training/comparison tasks |
|---|---:|
| Completeness, `--max-bins 5` | 4 |
| UCI three studies | 2,016 |
| BABD balanced | 840 |
| BABD imbalanced | 4,200 |
| Elliptic | 60 |
| Total | 7,120 |

These counts apply to the documented default grids plus the explicit completeness
setting. Voting and rule-ablation variants share fits. Compatible successful
checkpoints can be reused; counts do not indicate how many new computations a
rerun will perform. The root launcher's docstring explains the parameter products.
Do not use `all` as an exact paper shortcut: it also selects the exploratory
training-size study and the completeness script's larger default grid.

Do not run concurrent writers against the same results directory. For new or
subset runs, use `--results-root /path/to/separate-results` to avoid replacing the
paper workbooks. Source/settings changes may change checkpoint identities.

## 4. Results and the six paper figures

Default experiment outputs are under `results/classification/`:

| File | Content |
|---|---|
| `completeness_results.xlsx` | Direct versus all-concepts rule extraction. |
| `uci_three_studies.xlsx` | UCI comparison, ablation and sensitivity. |
| `babd_balanced_results.xlsx` | Balanced BABD results. |
| `babd_imbalanced_results.xlsx` | Imbalanced BABD results. |
| `elliptic_results.xlsx` | Elliptic results and rule evidence. |
| `checkpoints/` | Task records, execution provenance and recovery indexes. |

After UCI and both BABD studies complete, generate all six figures:

```sh
python analysis/classification/plot_all.py
```

The plotter reads the three corresponding workbooks, selects records from their
`per_seed` sheets and performs the statistical preparation automatically.
**No manual row selection or CSV conversion is required.** Completeness and
Elliptic outputs supply tables/evidence rather than these six figures.
If the completed workbooks are already available, skip training and plot directly.
For downloaded or separately stored results:

```sh
python analysis/classification/plot_all.py \
  --uci /path/to/uci_three_studies.xlsx \
  --babd /path/to/babd_balanced_results.xlsx \
  --babd-imbalanced /path/to/babd_imbalanced_results.xlsx \
  --output /path/to/figures
```

Default figure output is `results/figures/classification/`. Each figure has PDF,
PNG and CSV files sharing its basename; the manifest records generation status.
Use `--only` followed by figure basenames to select a subset. Existing same-name
outputs are overwritten. See the [classification experiment and plotting guide](analysis/classification/README.md)
for the six figure names, exact filters, metrics and validation rules. The current
plotter accepts workbooks, not CSV input files.

## 5. Where to change parameters

| Settings | Location |
|---|---|
| Experiment selection and execution overrides | `run_experiment.py` CLI; see its opening guide and each experiment's `--help`. |
| UCI study grid, seeds, voting and ablation defaults | `src/oeo_lab/experiments/classification/uci_experiment.py`, `main()`. |
| UCI inputs, model specifications and rule groups | `src/oeo_lab/experiments/classification/uci_config.py`. |
| BABD training sizes, seeds and bin grid | `src/oeo_lab/experiments/classification/babd_experiment.py`, `main()`. |
| BABD inputs, class selection and model specifications | `src/oeo_lab/experiments/classification/babd_config.py`. |
| Elliptic and completeness | Their entry modules and corresponding `*_config.py` files in the same directory. |
| Figure paths, DPI and subset | `analysis/classification/plot_all.py` CLI. |
| Figure selection, bin grids, metrics and seeds | `plot_all.py`, `main()`; data transformations in `result_data.py`. |

Precedence is configuration -> entry-point/script defaults -> explicit CLI options.
In particular, UCI `main()` overrides an earlier grid retained in `uci_config.py`.
Changing experiment grids does not automatically change the plotter's expected grids.

## 6. Earlier work: 3WOC construction

The repository also provides source and experimental code for
**3WOC: A Fast Algorithm for Three-Way Object-Oriented Concept Acquisition**,
*Applied Sciences* 15(12), 6486 (2025),
[DOI: 10.3390/app15126486](https://doi.org/10.3390/app15126486).
Its independent workflow is retained and is not part of the classification launcher:

```sh
PYTHONPATH=src python -m oeo_lab.experiments.construction.run --suite synthetic
PYTHONPATH=src python -m oeo_lab.experiments.construction.plot
```

See the [construction README](src/oeo_lab/experiments/construction/README.md) for
all suites, parameters, plotting logic and reproduction scope. These results were
regenerated in a different environment, not taken from the original publication's
experimental records. The author-confirmed Lenses correction is **292**, not the
published typographical value 229.

## 7. Reproduction documentation

- [Detailed classification protocol](docs/classification/reproduction.md)
- [Paper artifacts, results and historical source correspondence](docs/classification/paper_artifacts.md)
- [Classification experiment-to-figure workflow](analysis/classification/README.md)
- [3WOC construction workflow](src/oeo_lab/experiments/construction/README.md)

The source includes configurations, dependency specifications, data preparation
and reproduction instructions for classification and 3WOC concept construction.
External datasets must be obtained from their providers. The plotting programs
read experiment workbooks; see REPRODUCING.md for expected filenames and commands.
Historical task provenance is retained: this maintained source tree is not the
original execution snapshot for every reported result.

## 8. License, attribution and citation

Copyright (c) 2026 Zhuo Zhang and Yuhao Jin. Project code and associated
documentation are provided under the [MIT license](LICENSE). Retain author
attribution and [third-party notices](THIRD_PARTY_NOTICES.md). External datasets
and model weights retain their providers' terms. The software license does not
relicense separately supplied experimental data.

See [citation instructions](CITATION.md) and [software metadata](RELEASE_METADATA.md).
