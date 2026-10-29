# FCA-lab-OEO

Research implementation for **Classification via Directly Extracted Necessary-Attribute Rules** and related 3WOC concept-construction work.

RC-OEO extracts a specified family of OEO necessary-attribute classification rules without constructing the complete concept lattice. Supporting and excluding rule evidence is used to assign a unique label or retain candidate classes. Completeness concerns the specified rule family; candidate-set inclusion on unseen objects is an empirical result, not a guarantee.

## Paper artifacts and reproduction

- [Reproduction guide](docs/classification/reproduction.md): data, protocols, metrics and commands.
- [Paper artifact and version map](docs/classification/paper_artifacts.md): experiments, tables, figures, configurations, workbooks and historical provenance.
- [Launcher reference](docs/classification/experiment_entry.md): optional interactive launcher and watchdog behavior. Its example commands are not the paper's frozen configuration.

The completed paper results cover rule completeness, UCI classification, BABD balanced/imbalanced classification, and Elliptic transaction classification. Earlier exploratory Bitcoin/entity experiments and extended grids are not part of the paper artifact set.

## Installation

From the repository root, in a dedicated Python environment:

```sh
python -m pip install -e '.[plot,boost]'
```

The full BABD model grid additionally requires `catboost`. Dependencies in `pyproject.toml` are not version-pinned; installation alone does not recreate the historical environment. Consult task provenance and the reproduction guide before claiming an exact reproduction.

Data are not bundled. Put the documented inputs under `data/`, or set `OEO_DATA_ROOT` to an equivalent data root. Do not substitute an unprocessed BABD download for the cleaned input used in the paper.

## Status and release boundary

Documentation checked on 2026-10-29. The inspected checkout has HEAD `38d219c` plus local launcher and plotting changes. It is **not** a single frozen source version for every historical result. Task-level source fingerprints and checkpoint reuse records identify the original executions; see the version map.

Public code release is planned upon acceptance. No public archival release is designated by this README. Before release, a fixed source snapshot, dependency specification, input preparation instructions and the selected result/figure artifacts still need to be packaged. The documentation audit did not rerun experiments or validate installation in a clean environment.

`results/` and raw data are ignored by Git. Uploading the source repository alone will not include the paper's workbooks, checkpoints or figures.
