# Reproduction entry guide

## 1. Install and check the implementation (no external data)

From this source directory, follow README.md's Conda/Python 3.13 installation
commands, then run `python -m pytest -q`. Tests contain small synthetic fixtures;
they are not the full paper evaluation. The 116-pass Linux report in README is
historical (2026-10-30), not a new run made while preparing this candidate.
Use the recorded dependency snapshot; Python >=3.9 metadata is not a claim that
all snapshot packages support every such Python version.

## 2. Regenerate figures from existing results (no training or raw data)

Extract the separate lite result ZIP and run from this source directory:

```sh
python analysis/classification/plot_all.py   --uci /absolute/path/to/paper-results-lite/uci_three_studies.xlsx   --babd /absolute/path/to/paper-results-lite/babd_balanced_results.xlsx   --babd-imbalanced /absolute/path/to/paper-results-lite/babd_imbalanced_results.xlsx   --output /absolute/path/to/new-figures
```

Expected output is six PDF/PNG/CSV figure sets plus manifest.json. Use a new output
directory: same-name plots are overwritten. The completeness and Elliptic files
provide table/rule evidence rather than these six plots. Table row selection is
documented in docs/classification/paper_artifacts.md; a complete automatic export
of final manuscript tables is not claimed. Full and lite workbooks differ in
included sheets, not the retained numerical findings.

## 3. Re-run experiments from external inputs

Follow DATA_SOURCES.md, then docs/classification/babd_preparation.md as needed.
Preserve headers, row/column order and file hashes. From the source directory:

```sh
export OEO_DATA_ROOT=/absolute/path/to/data
python run_experiment.py --no-watch completeness --max-bins 5 --n-jobs 1
python run_experiment.py --no-watch uci --n-jobs 1
python run_experiment.py --no-watch babd --experiments balanced imbalanced --n-jobs 1
python run_experiment.py --no-watch elliptic --n-jobs 1
```

For new/subset runs use `--results-root` with a separate output directory.
`--dry-run` still requires inputs. `--no-watch` avoids the external watchdog.
Full settings, seeds, splits, metric populations and version boundaries are in
docs/classification/reproduction.md. Large BABD grids require substantial time
and memory; successful installation alone does not validate a full reproduction.

## 4. Version and evidence limits

Code is based on ac1f424; computational files and dependency specifications are
unchanged. Historical results were produced by multiple recorded snapshots,
not necessarily by this maintenance snapshot. See paper_artifacts.md for mappings.
Neither result archive includes raw datasets or execution checkpoints. No full
paper experiment or timing measurement was repeated for this candidate.
