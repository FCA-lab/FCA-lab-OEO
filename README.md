# FCA-lab-OEO

Private development preparation for two related studies: 3WOC concept acquisition and OEO-based classification. Public release is planned after acceptance of the classification paper.

## Status
This is a structural migration of the existing implementation, not a validated implementation of the revised classification manuscript. The classifier still uses the legacy A2, forced-decision and voting behavior. Do not use it for the new paper experiments yet.

## Setup
```bash
python -m pip install -e ".[test]"
python -m pytest
python -m oeo_lab.experiments.classification.run --help
```

`fca_core` contains shared foundations; `oeo_lab` contains algorithms, preprocessing and experiments. Construction algorithms retain their original computational logic. Only obsolete unused imports are removed.

Data are not bundled. Set `OEO_DATA_ROOT` to a data root containing `raw/bitcoin/getFeatures.csv`, or pass `--data` explicitly. See `data/README.md` and the reproduction documents before launching experiments.

No GitHub upload or public release has been performed by this migration.
