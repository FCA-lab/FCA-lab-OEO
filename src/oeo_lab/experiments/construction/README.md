# OEO-concept construction experiments and plotting

This directory contains the independent experiment workflow for
**3WOC: A Fast Algorithm for Three-Way Object-Oriented Concept Acquisition**,
*Applied Sciences* 15(12), 6486 (2025),
[DOI: 10.3390/app15126486](https://doi.org/10.3390/app15126486).
It compares 3WOC with QM and is retained alongside the RC-OEO classification
experiments. The root classification launcher does not dispatch these suites.

This guide describes the current implementation; it does not add a new input
format or change the algorithms. For historical timing, memory and checkpoint
boundaries, see the [reproduction guide](../../../../docs/construction/reproduction.md).

## 1. Files and responsibilities

| File | Responsibility |
|---|---|
| `config.json` | Dataset definitions, synthetic grid, algorithms, execution and output settings. |
| `run.py` | Select suites, construct tasks, schedule execution and summarize results. |
| `datasets.py` | Load real/example contexts and generate seeded synthetic contexts. |
| `evaluate.py` | Run the selected construction algorithm, measure it and validate concepts. |
| `plot.py` | Read existing synthetic summaries, validate grids and generate figures. |

The algorithm implementations are
[`three_woc.py`](../../algorithms/construction/three_woc.py) and
[`juxtaposition.py`](../../algorithms/construction/juxtaposition.py).
Use `run.py` for the documented reproduction workflow rather than the older
standalone scripts in this directory.

## 2. Environment and experiment commands

Run all commands below from the repository root, with the project installed in
the selected Python environment. Plotting requires the `plot` extra:

```sh
python -m pip install -e '.[plot]'
```

On the authors' D30 workstation:

```sh
cd /home/charles/projects/FCA-lab-OEO
conda activate fca
```

On another machine, use its checkout path and environment. The experiment entry
is independent of `run_experiment.py`:

```sh
# Inspect available options.
PYTHONPATH=src python -m oeo_lab.experiments.construction.run --help

# Build the requested tasks without executing the algorithms.
PYTHONPATH=src python -m oeo_lab.experiments.construction.run --dry-run

# Run one suite.
PYTHONPATH=src python -m oeo_lab.experiments.construction.run --suite example
PYTHONPATH=src python -m oeo_lab.experiments.construction.run --suite real
PYTHONPATH=src python -m oeo_lab.experiments.construction.run --suite synthetic

# Run all suites (the default).
PYTHONPATH=src python -m oeo_lab.experiments.construction.run --suite all

# Optional experiment-level parallelism; example with two worker processes.
PYTHONPATH=src python -m oeo_lab.experiments.construction.run --suite synthetic --n-jobs 2

# Use a complete alternative JSON configuration.
PYTHONPATH=src python -m oeo_lab.experiments.construction.run --config /path/to/config.json --suite synthetic
```

`--dry-run` reads inputs and generates synthetic contexts to construct/validate
tasks; it does not run the construction algorithms. Serial execution is the
default for timing comparisons. `--n-jobs` sets concurrent task processes, not
threads within a single algorithm. `--timeout` sets a whole-task deadline,
including startup and validation; it is not the reported algorithm runtime.

With the current configuration:

| Suite | Backgrounds | Algorithm tasks |
|---|---|---|
| `example` | One matrix from the paper's Table 1 | 1 x 2 = 2 |
| `real` | Balloons, Lenses and COVID19 Surveillance | 3 x 2 = 6 |
| `synthetic` | 4 object counts x 2 attribute counts x 6 densities x 5 repeats = 240 | 240 x 2 = 480 |
| `all` | All of the above | 488 |

Each realized background is shared by 3WOC and QM. Successful compatible
checkpoints can be reused, so the number of newly executed tasks may be smaller.
Synthetic contexts are generated from recorded seeds; they are not saved as
separate matrix files. Counts above describe the configured scope, not a claim
that every suite has an existing result workbook.

## 3. Results and provenance

The default output root is `results/construction/`, with:

- `example_results.xlsx`, `real_results.xlsx`, `synthetic_results.xlsx`;
- `checkpoints/` for task records and the recovery index;
- `figures/` for plots generated in a separate postprocessing step.

Each workbook contains `runs`, `summary` and `comparisons` sheets.
`runs` records individual algorithm executions. `summary` reports means and
sample standard deviations over the configured backgrounds; incomplete groups
do not receive a partial mean. `comparisons` records paired concept-set checks.
A result count is not selected to match the paper's printed value.

The supplied construction results and figures were regenerated in a different
computing environment. They are **not the original experimental records used in
the published paper**. Runtime and memory values may differ; the memory field
measures shallow final-container overhead, not total process memory.

**Author-confirmed correction:** the published Lenses concept count of 229 is a
typographical error. The correct count is **292** for the 24 x 12 construction
context, consistent with both recorded algorithm outputs. The configuration
retains the published value as historical information, not a pass criterion.
Construction encodes the category column as an ordinary attribute; classification
uses separate condition and decision contexts, so their counts need not agree.

The planned release separates code/configuration/instructions from optional
archives of regenerated workbooks and figures. Archive links will be added when
available. No separate CSV is required to plot the supplied workbook.

## 4. Plotting commands and input selection

The plotting entry is `plot.py`. It reads the `summary` sheet of an Excel workbook;
it currently accepts **only `--config`**, not `--input` or `--csv`.

```sh
# Plot the existing default synthetic workbook; do not run experiments.
PYTHONPATH=src python -m oeo_lab.experiments.construction.plot

# Select a different workbook and plot settings through a complete JSON config.
PYTHONPATH=src python -m oeo_lab.experiments.construction.plot --config /path/to/config.json
```

The paths are resolved as follows:

```text
input workbook = repository root / output.root / output.files.results.synthetic
figure folder  = repository root / output.root / plot.directory
```

An absolute `output.root` replaces the repository-root prefix. Relative paths are
resolved against the repository root, not the directory containing the JSON file.

For an optional downloaded result archive, either place `synthetic_results.xlsx`
in the default `results/construction/` directory or copy the complete supplied
`config.json` and change these fields:

| JSON field | Default | Purpose |
|---|---|---|
| `output.root` | `results/construction` | Directory containing the workbook. |
| `output.files.results.synthetic` | `synthetic_results.xlsx` | Input workbook filename. |
| `plot.directory` | `figures` | Figure directory relative to `output.root`. |
| `plot.formats` | `["png", "pdf"]` | Export formats. |
| `plot.dpi` | `600` | Export resolution. |
| `plot.files.concepts` | `concepts_att{attributes}` | Concept-count figure basename. |
| `plot.files.time` | `time_{algorithm}_att{attributes}` | Runtime figure basename. |

Keep the other configuration fields. Its synthetic grid and repeat count must
match the workbook. Replotting overwrites same-name figures; use a different
`plot.directory` when comparing outputs. The plotter does not alter the workbook
or checkpoints. Users of the supplied workbook do not need to assemble a table
manually or convert it to CSV.

## 5. Data-selection and plotting logic

```text
Read config.json
    -> read synthetic_results.xlsx, sheet="summary"
    -> prepare_grids(summary, config)
         -> select synthetic results
         -> group by attribute count and algorithm
         -> validate parameter coverage and completed repeats
         -> pivot means into object-count x density grids
         -> verify equal concept counts between algorithms
    -> draw_3d_bar(...)
    -> save PDF / PNG
```

`prepare_grids()` uses the complete configured grid, not the best-performing
settings. Specifically:

1. Require the expected columns and retain `kind == "synthetic"` rows.
2. Reject duplicate `(objects, attributes, density, algorithm)` keys.
3. For each configured attribute count and algorithm, require exactly the
   configured object-count/density combinations. Missing or extra combinations
   within these groups cause an error. Rows for unconfigured attribute counts or
   algorithms are not plotted.
4. Require `complete == True`, with `completed_runs` and `expected_runs` equal to
   the configured repeat count for every selected row.
5. Use the existing `concept_count_mean` and `seconds_mean` values. Do not average
   runs again. Reject nonfinite or negative values.
6. Pivot by explicit keys: objects index the rows and density indexes the columns.
7. Require equal concept-count grids for both algorithms before producing a shared
   concept-count figure. Produce separate runtime figures for each algorithm.

The default grid has objects `[10,20,30,40]`, attributes `[5,10]`, densities
`[0.1,0.2,0.3,0.4,0.5,0.6]` and five repeats. Each attribute/algorithm group has
24 mean values. The resulting six figures are:

- `concepts_att5`, `concepts_att10`;
- `time_3woc_att5`, `time_3woc_att10`;
- `time_qm_att5`, `time_qm_att10`.

Bars show means, not individual runs or confidence intervals. Standard deviations
remain available in the workbook; they are not shown on these bars. `prepare_grids()`
returns in-memory grids directly to the renderer; this version does not export
intermediate CSV files.

## 6. Configurable settings versus fixed plotting rules

| Item | Location | Configurable without editing Python? |
|---|---|---|
| Objects, attributes, densities and repeat count | `config.json`: `synthetic` | Yes; must match the input summary. |
| Algorithms | `config.json`: `algorithms` | The reproduction runner requires exactly `3woc` and `qm`; changing names does not implement another algorithm. |
| Real datasets and example context | `config.json`: `real_datasets`, `example` | Yes, subject to loader validation. |
| Concurrency and timeout | `config.json`: `execution`; runner CLI overrides | Yes. |
| Workbook and output paths, filenames, formats, DPI | `config.json`: `output`, `plot` | Yes. |
| Read the `summary` worksheet | `plot.main()` | Fixed. |
| Select synthetic rows | `prepare_grids()` | Fixed. |
| Metrics: mean concept count and mean runtime | `prepare_grids()` | Fixed. |
| Facets: attribute count and algorithm; axes: objects and density | `prepare_grids()`, `draw_3d_bar()` | Fixed. |
| Completeness, duplicate-key and numerical validity checks | `prepare_grids()` | Fixed validation rules. |
| Equal concept counts before sharing the count figure | `prepare_grids()` | Fixed validation rule. |
| Bar style, view angle, labels and figure dimensions | `draw_3d_bar()` | Fixed. |

For reference, the required summary columns are `kind`, `objects`, `attributes`,
`density`, `algorithm`, `complete`, `completed_runs`, `expected_runs`,
`concept_count_mean` and `seconds_mean`. The experiment program creates this
schema automatically. A CSV cannot be passed directly or merely renamed `.xlsx`.


## 7. Minimal end-to-end reproduction of the six synthetic figures

```text
Run the synthetic construction experiment
    -> generate results/construction/synthetic_results.xlsx
    -> run the construction plotter
    -> generate six PDF/PNG figures under results/construction/figures/
```

From the repository root, in the prepared Python environment, run these two
commands in order:

```sh
# 1. Run the synthetic construction experiment and write its result workbook.
PYTHONPATH=src python -m oeo_lab.experiments.construction.run --suite synthetic

# 2. Read the completed workbook and generate the six figures.
PYTHONPATH=src python -m oeo_lab.experiments.construction.plot
```

The second command automatically reads the workbook's `summary` sheet and
performs the selection, validation and grid preparation described above. No
manual filtering or CSV conversion is needed. If the completed workbook is
already available, only the second command is necessary. These are reproduced
figures corresponding to the paper; runtime values need not equal the original
published values in a different computing environment.
