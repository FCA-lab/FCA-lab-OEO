# FCA-lab-OEO

Code for **Classification via Directly Extracted Necessary-Attribute Rules**.
RC-OEO directly extracts necessary-attribute classification rules from OEO-concepts
and combines supporting and excluding evidence to assign a unique label or retain
candidate classes. The experiments cover rule completeness, UCI classification,
and Bitcoin application cases using BABD and Elliptic.

The repository also includes the independent experimental code for the earlier
[3WOC concept-construction paper](https://doi.org/10.3390/app15126486).
Classification is the main workflow described below.

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
runs. Installation in a fresh environment remains unverified. The commands above
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
the cleaned file, not the unprocessed download. A complete, verified raw-to-clean
preparation recipe is still a release-preparation item; do not substitute another
file and assume it reproduces the paper's results.

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

## 7. Reproduction and release status

- [Detailed classification protocol](docs/classification/reproduction.md)
- [Paper artifacts, results and historical source correspondence](docs/classification/paper_artifacts.md)
- [Classification experiment-to-figure workflow](analysis/classification/README.md)
- [3WOC construction workflow](src/oeo_lab/experiments/construction/README.md)

Public release is planned upon acceptance of the classification paper, with the
earlier 3WOC code included. No archival release is designated yet. The planned
repository contains code, configurations and instructions; result workbooks and
figures will be provided as optional archived artifacts. Links will be added when
available. Raw data and results are ignored by Git and are not included simply by
uploading this repository. A fixed release snapshot, dependency specification,
license and completed data-preparation instructions remain to be packaged.
Historical task provenance must be retained: the current source tree is not a
single original execution version for every reported result.
