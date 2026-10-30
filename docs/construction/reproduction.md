# Zhang2025 construction reproduction

This code accompanies **3WOC: A Fast Algorithm for Three-Way Object-Oriented Concept Acquisition** (2025), DOI: https://doi.org/10.3390/app15126486. The available results were regenerated in a different environment; they are not the original publication records, and matching historical timings is not a reproduction requirement.

Prepare the environment using the root [README](../../README.md), then run from the repository root:

```sh
python -m oeo_lab.experiments.construction.run --suite example
python -m oeo_lab.experiments.construction.run --dry-run
python -m oeo_lab.experiments.construction.run
```

Configuration: `src/oeo_lab/experiments/construction/config.json`. All relative paths resolve against the project root. Outputs always use `results/construction/` with separate result workbooks and shared `checkpoints/`; there are no batch directories. Run the same command to resume. Real, synthetic and example suites use separate workbooks (real_results.xlsx, synthetic_results.xlsx, example_results.xlsx); names are configured under output.files.results. Task identity includes background inputs, algorithm, computation-code hash and recorded execution environment. Reporting-only changes do not invalidate results. Changed computation signatures are not mixed into the active workbook; old checkpoints remain recoverable. Failed/timeout tasks retry on resume. Completed checkpoints are retained. `checkpoints/index.json` stores task recovery metadata only; experiment settings are always read from the module configuration. Tests must override output.root with a temporary directory and remove it after verification.

Scope: paper Table 1; Balloons adult+stretch (all columns), Lenses (discard ID, encode remaining five columns), Surveillance (discard header, encode all columns); 48 synthetic parameter combinations with five different seeded backgrounds each, paired across 3WOC and QM (480 synthetic algorithm tasks). Real data and Table 1 run once per algorithm. Artificial matrices are generated at runtime, not persisted; seeds and matrix hashes are recorded. Duplicate realized matrices within a parameter group cause an explicit error.

Timing follows legacy `start()` boundaries, excluding initialization, encoding, validation and saving. Algorithm stdout goes to the null device; formatting overhead remains. Serial execution is the default for performance comparisons. `--n-jobs` enables experiment-level parallelism; each task runs in a fresh process with internal numerical threads limited to one. `--timeout` is a whole-task deadline (including startup and validation), not measured algorithm time; default is no timeout. Admission control checks Linux MemAvailable; it cannot guarantee protection from a single task's memory growth.

Memory (`result_container_bytes`, bytes) measures shallow `sys.getsizeof` of the final result container. After `start()` completes, 3WOC concepts are collected into a flat dictionary keyed by extent encoding (values reference original concepts); QM uses its existing `concepts` list. No recursive traversal, per-layer summation, or accumulation across construction steps is performed. Both flattening and measurement are excluded from algorithm timing. The metric includes container allocation/preallocation, but excludes referenced concept contents and intermediate construction structures. It compares implementation-specific container overhead, not total concept storage or process memory. This is consistent with the agreed reconstruction of the historical metric; the original measurement code has not been recovered. Python version and allocation strategy can change byte counts. Previous recursive-size checkpoints are incompatible and will not be reused.

Each checkpoint retains canonical concept triples as integer masks; complete sets are compared in both directions. Each concept is independently checked against necessary-attribute and extent-recovery definitions. These checks are outside timing. Agreement alone is not an independent completeness proof.

Lenses: the author confirmed that the published count 229 is a typographical error; the corrected count is 292 for the official 24x12 construction context. The configuration retains the published value as historical metadata, not a pass criterion. Classification uses a different condition context. Historical timing values are not test thresholds.

Each result workbook: runs, summary (mean and sample standard deviation across five backgrounds), comparisons. Incomplete groups have no reported mean; counts and failure states remain visible. No data or GitHub upload is performed by this entry point.

## Three-dimensional figures

Run `PYTHONPATH=src python -m oeo_lab.experiments.construction.plot` after the synthetic experiment completes. This independent postprocessing command reads the configured synthetic workbook's `summary` sheet and writes six figures (PNG and PDF) under `results/construction/figures/`. Paths, filename templates, formats and DPI are configured in the construction configuration's `plot` section. `--config` accepts an alternative configuration. Use the dependency installation procedure in the root README; no separate plotting environment is required.

The long table is pivoted using objects and density, separately for each attribute count and algorithm. Bars show means over the configured repeats; they do not show standard deviations, which remain in the workbook. Missing/duplicate grid entries, incomplete repeats and invalid metric values cause an error before plotting. Concept counts must agree between algorithms before a shared concept-count figure is produced. Timing figures retain the execution conditions of the input workbook; plotting does not rerun experiments. The bar3d design is adapted from the original `src/view/data_view.py` construction plotter, with numeric data handling replacing its CSV layout and eval calls.
