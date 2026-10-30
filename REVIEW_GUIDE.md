# Reviewer guide

This repository accompanies *Classification via Directly Extracted
Necessary-Attribute Rules*. It also includes the earlier 3WOC construction code.
The current version is available only for authorized confidential review under
[LICENSE](LICENSE); third-party terms remain unchanged. Public MIT release is
planned after acceptance. Access can be arranged through the handling editor.

## 1. Locate the implementation

| Manuscript component | Source location |
|---|---|
| Direct extraction (GetBARules) | `src/oeo_lab/algorithms/classification/classifier.py`: `RC_3Woc.fit`; basic-concept helpers `get_single_att_concepts` and `get_base_OEOs` in `algorithms/construction/three_woc.py`. |
| Prediction (RC-3WOC; classifier named RC-OEO in experiments) | `RC_3Woc.predict` and `predictWithRules` in `classifier.py`. |
| Training-only binning and binary encoding | `src/oeo_lab/preprocessing/encoding.py`; categorical handling in `categorical.py`. |
| Completeness comparison | `src/oeo_lab/experiments/classification/completeness.py`; tiny-context tests in `tests/classification/test_completeness.py`. |
| Metrics and candidate sets | `src/oeo_lab/experiments/classification/evaluate.py`; aggregation in `runner.py`. |
| Paper experiments and configuration | `src/oeo_lab/experiments/classification/`: `uci_*`, `babd_*`, `elliptic_*`, `completeness_*`. |
| Six classification paper figures | `analysis/classification/plot_all.py`, `result_data.py`, `figures.py`. |
| Earlier 3WOC paper | `src/oeo_lab/algorithms/construction/` and `src/oeo_lab/experiments/construction/`. |

Paths abbreviated in the table are relative to the corresponding source folder.
The older `predictWithRules_2` method is not the main prediction procedure.

## 2. Install in a separate environment

From the extracted repository root, follow the environment instructions in
[README](README.md). `requirements-reproduce.txt` records the current D30 package
versions; it is not a lockfile for every historical experiment. Installation from an extracted review source package was verified on
2026-10-30 in an isolated Linux Python 3.13.14 environment.
Use a separate environment rather than changing an existing research environment.

## 3. Inspect interfaces and run implementation tests

After installation, these help commands require no research datasets and do not
train models:

```sh
python run_experiment.py --help
python run_experiment.py uci --help
python run_experiment.py babd --help
python run_experiment.py elliptic --help
python -m oeo_lab.experiments.construction.run --help
```

The implementation tests use small fixtures and temporary outputs; they do not
rerun the full paper experiments:

```sh
python -m pytest -q
```

All 116 tests passed in the isolated review-package environment. Workflow tests
generate temporary synthetic inputs; no downloaded research datasets are needed.
This verifies implementation tests, not reproduction of the full paper results. `--dry-run` builds task lists and can read
real input data; it is not a data-free alternative to `--help`.

## 4. Inspect results or reproduce selected work

[The artifact map](docs/classification/paper_artifacts.md) identifies the five
classification workbooks, table selections, six figure names and historical
source correspondence. Workbooks and final figure data are separate artifacts,
not automatically included with this source tree; access can be arranged through
the editor when needed. Full experimental reruns are not required for code review.

Dataset preparation and experiment commands are in
[the classification reproduction guide](docs/classification/reproduction.md).
BABD raw-input preparation is described [here](docs/classification/babd_preparation.md).
Plotting commands and data filters are in [the plotting guide](analysis/classification/README.md).
The earlier 3WOC workflow and its regenerated-result scope are documented
[separately](src/oeo_lab/experiments/construction/README.md).

## 5. Review snapshot

When supplying a package, the maintainer should record its exact commit and
archive checksum in the delivery message. Do not include `.git`, private backups,
logs, private datasets or inactive MIT document templates. No review archive or
private download URL has yet been issued. The current code is a maintained
snapshot; historical result provenance is retained separately and must not be
relabeled as a fresh run of this snapshot.
