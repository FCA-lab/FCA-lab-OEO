# External data sources and input identity

No external raw datasets or cleaned copies are distributed in GitHub or the
candidate Zenodo archives. Download them from their providers, follow provider
terms, and prepare them locally. The MIT license does not cover external data.
Links and hashes below are retained from the 2026-10-30 acquisition audit;
current download availability has not been rechecked during this packaging step.
Do not silently replace a historical input with a newer dataset version.

| Dataset | Provider/source | Acquisition and preparation |
|---|---|---|
| Iris | https://archive.ics.uci.edu/dataset/53/iris | Use iris.data, not bezdekIris.data. |
| Wine | https://archive.ics.uci.edu/dataset/109/wine | Use wine.data. |
| WDBC | https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic | Use wdbc.data. |
| Balloons | https://archive.ics.uci.edu/dataset/13/balloons | Use adult+stretch.data only. |
| Lenses | https://archive.ics.uci.edu/dataset/58/lenses | Use lenses.data; preserve delimiters. |
| COVID19 Surveillance | https://archive.ics.uci.edu/dataset/567/covid+19+surveillance | Completeness experiment; preserve Surveillance.csv header and order. |
| BABD-13 | https://www.kaggle.com/datasets/lemonx/babd13 | Download BABD-13.csv; run the packaged preparation script. Historical Kaggle version/date not recorded; match the input hash. |
| Elliptic | https://www.kaggle.com/datasets/ellipticco/elliptic-data-set | Recorded acquisition used the PyG mirror links in the reproduction guide; two CSVs only, no edges. Match hashes below. |

Exact archive URLs, destination paths and experiment settings are in
[the classification reproduction guide](docs/classification/reproduction.md).
[BABD preparation](docs/classification/babd_preparation.md) records raw ZIP/CSV
hashes, cleaning rules and the expected cleaned output. No upstream license
identifier or unrecorded version/date is invented here; consult provider terms.

## Required local layout and checksums

Set OEO_DATA_ROOT to your chosen data directory (default: repository data/).
The following paths are relative to that root. Verify using `sha256sum` on Linux
or `shasum -a 256` on macOS. A mismatch calls for checking source/version, not
editing values to force the published result.

| Input relative to the data root | Historical SHA-256 |
|---|---|
| `raw/uci/iris/iris.data` | `6f608b71a7317216319b4d27b4d9bc84e6abd734eda7872b71a458569e2656c0` |
| `raw/uci/wine/wine.data` | `6be6b1203f3d51df0b553a70e57b8a723cd405683958204f96d23d7cd6aea659` |
| `raw/uci/wdbc/wdbc.data` | `d606af411f3e5be8a317a5a8b652b425aaf0ff38ca683d5327ffff94c3695f4a` |
| `raw/uci/balloons/adult+stretch.data` | `56f8efed1e7f2e5e21a7f78b1cb2df75860642667b145330b3a0ea196b2e3274` |
| `raw/uci/lenses/lenses.data` | `711f22d5a4bed062bb715d39d23d54d2667206fbd97c94c7fd013ede47fb96c2` |
| `raw/uci/covid19_surveillance/Surveillance.csv` | `d9967ea62243865399d0f70e011e8c7006ead5b60a858914ac7e29c87a4f1f08` |
| `raw/public/elliptic/elliptic_txs_features.csv` | `fd7f83573443c9e302e371d3f110e3b6224160f5d1ed8a287757936127800ff0` |
| `raw/public/elliptic/elliptic_txs_classes.csv` | `93e2e7b2405c735ba752bf6ba06b947561deddd1f5a8fc91e46f6a4c0e439493` |


BABD source CSV SHA-256:
`766b5d51af956b1a619775e5522d0c15f0c1c2ef272f87949d8bd1a685f332b6`.
After preparation, `raw/public/babd13/BABD-13-clean.csv` must have SHA-256
`d380c00f6528f225c6840cb78c07fd28fa6794a98b9ed20a5c3eafcb6109f4e3`
in the historically verified environment; keep cleaning_report.json beside it.
Cross-version CSV serialization may differ; investigate rather than ignoring a mismatch.

## Preprocessing boundary

BABD preparation removes conflicting labels and duplicate addresses before the
experimental split. Predictive encoding/binning is fitted on training data only.
UCI uses seeds 11/37/53 and stratified folds (four for Lenses, five otherwise).
Elliptic uses the fixed time split 1–34 / 35–49 and 93 local features, excluding
unknown labels. Full settings and class selections remain in the reproduction
guide and unchanged code. Address-level splits do not imply entity-disjoint tests.
