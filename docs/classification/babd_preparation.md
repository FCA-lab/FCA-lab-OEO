# Preparing the BABD experimental input

This procedure packages the recovered `data/public/babd13/prepare.py` from the
paper workspace. It preserves the historical cleaning rules; it does not add
binning, imputation, feature scaling, class selection or train/test splitting.
The original script SHA-256 is
`a102365e6c532ce33adca342d36b2439e9f7012e5c3cb1cd73b40beadb632c9c`.

## Source and input identity

Download the BABD-13 dataset from its recorded source:
<https://www.kaggle.com/datasets/lemonx/babd13>.
Use the supplied `BABD-13.csv`, not the upstream blockchain feature-extraction
pipeline. The locally retained ZIP contains this CSV. The exact Kaggle version
number and original download command were not recorded; identify the required
input using the following hashes. A newer download must not be silently substituted.

| File | Bytes | SHA-256 |
|---|---:|---|
| Retained ZIP (locally named `download.bin`) | 197599407 | `630c2d34102f7d4a268dba860e907398cb3c6d83aa219a059d0446da3e71bc84` |
| `BABD-13.csv` | 785101222 | `766b5d51af956b1a619775e5522d0c15f0c1c2ef272f87949d8bd1a685f332b6` |
| Historical `BABD-13-clean.csv` | 759605296 | `d380c00f6528f225c6840cb78c07fd28fa6794a98b9ed20a5c3eafcb6109f4e3` |

The downloaded dataset is not bundled. Check the provider's current terms before
redistribution; this document does not establish a redistribution license.

## Preserved cleaning rules

1. Read the CSV with pandas, retaining source row and column order. Define the
   148 features as all columns except `account`, `SW` and `label`.
2. Group by `account`. If an address has multiple labels, remove **all** its rows
   (historical audit: 1,414 addresses and 3,216 rows).
3. For remaining repeated addresses, compare feature vectors with the first row
   using `numpy.allclose(rtol=1e-12, atol=1e-14)`. Four addresses have small numeric
   differences within this tolerance. Larger differences cause failure.
4. Keep the first source row for each remaining address, without averaging or
   sorting. This removes 292 additional rows and retains 540,954 rows.
5. Save the CSV and the ordered feature list in `cleaning_report.json`.

All 13 labels remain in the cleaned file. The balanced seven-class and imbalanced
eleven-class selections are applied later by `babd_config.py`; in particular,
labels 7 and 8 are not deleted by this preparation tool. The CSV retains identifier
and label columns, but the experiment uses only the 148 manifest-listed features.

## Commands and handoff to the experiment

Use an already configured environment with pandas and NumPy. From the repository
root, extract the downloaded archive outside the output directory, then run:

```sh
python scripts/prepare_babd.py --input /path/to/extracted/BABD-13.csv
```

The default output is `data/raw/public/babd13/` under the repository, or
`$OEO_DATA_ROOT/raw/public/babd13/` when `OEO_DATA_ROOT` is set. The directory must
**not exist**: the tool refuses to overwrite existing experimental inputs.
For a separate preparation location, use:

```sh
python scripts/prepare_babd.py --input /path/to/extracted/BABD-13.csv --output-dir /path/to/new-data-root/raw/public/babd13
```

After reviewing the output and checksum, set `OEO_DATA_ROOT` to `/path/to/new-data-root`
if using that alternative. Follow the root README's BABD experiment command.
The experiment reads `BABD-13-clean.csv` and `cleaning_report.json`; the additional
`preparation_provenance.json` records input/output hashes, script hashes and the
Python/pandas/NumPy versions for the new preparation.

## Checks and verification limits

The tool checks the original checksum, dimensions, labels, missing/nonfinite
values, duplicate tolerance and historical cleaning counts before writing output.
These checks reject unexpected inputs rather than changing the cleaning rules.
It computes the number of near-equal duplicate addresses instead of hard-coding
that number in the report. Validation failures return nonzero; existing output
folders are refused. If an I/O failure interrupts writing, the new directory may
be incomplete; do not use it without all three output files and checksum review.

Validation on 2026-10-30 passed five CLI guard tests and a full-data comparison
with the recovered original script. In the same D30 environment, both scripts
produced byte-identical CSV and cleaning-report outputs. Validation used temporary
directories; existing experimental inputs were unchanged and no classification
or construction experiments were rerun.

```sh
python -m pytest -q tests/classification/test_babd_preparation.py
```
