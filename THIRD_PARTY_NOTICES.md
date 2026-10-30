# Third-party notices and code provenance

Copyright in the project's own code is held by the code contributors Zhuo Zhang
and Yuhao Jin, as confirmed by the project author on 2026-10-30, with no other
ownership agreements. Their code in this version is provided under the root limited review permission.
A separate MIT-licensed release is planned after acceptance; that future plan
does not expand the permission granted for the current review version.
Third-party material retains its original copyright notices and terms.
Dependencies, datasets and model weights are not relicensed by the project license.

## Project code contributions

The following assignments were confirmed by the project author on 2026-10-30.
They describe implementation contributions, not the authorship of every cited paper.

| Component | Confirmed contribution |
|---|---|
| FCA foundational structures (`src/fca_core/`) | Implemented by Zhuo Zhang. |
| OEO classification and construction algorithms | Implemented by Yuhao Jin; organized and edited by Zhuo Zhang. |
| `src/oeo_lab/preprocessing/encoding.py` | Implemented by Yuhao Jin; see the upstream binning notice below. |
| QM / object-oriented concept-construction baseline | Independently implemented by Yuhao Jin with reference to the research literature; not reported as copied third-party source code. |

Migration provenance is recorded in `docs/provenance.json`. A literature-based
implementation should cite its algorithmic source, but that citation is distinct
from licensing copied or adapted source code. The baseline source currently names
*Concept acquisition approach of object-oriented concept lattices*; complete
bibliographic details should be confirmed before release.

## LightGBM binning implementation

The binning routines in `encoding.py` were inherited through the earlier
`Features2Context` implementation. Their `GreedyFindBin` and
`FindBinWithZeroAsOneBin` logic corresponds to LightGBM's binning implementation.
This project retains the upstream MIT notice for the adapted binning logic.

- Upstream: Microsoft LightGBM.
- Reviewed source: https://github.com/microsoft/LightGBM/blob/v4.6.0/src/io/bin.cpp
- License: https://github.com/microsoft/LightGBM/blob/v4.6.0/LICENSE
- Version v4.6.0 is the source reviewed for this notice, not a confirmed original
  porting version.
- Local adaptation includes Python implementation, training-only fit/transform,
  midpoint boundaries, empty-interval handling and boundary safeguards. This is
  not a claim of byte-for-byte equivalence to LightGBM v4.6.0.

Upstream license text:

```text
The MIT License (MIT)

Copyright (c) Microsoft Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Imported dependencies

The project calls dependencies through their public APIs; it does not claim
ownership of them. The installed-version snapshot is in `requirements-reproduce.txt`.
Selected dependency metadata inspected on D30 identify BSD-family licenses for
pandas, SciPy, scikit-learn, joblib, threadpoolctl and psutil; MIT for openpyxl,
LightGBM and pytest; Apache-2.0 for XGBoost and CatBoost; and Matplotlib's own
license agreement. NumPy also includes components under additional licenses.
This summary is not a replacement for the packages' complete license files,
including bundled components. If redistributing dependencies or an environment
image, retain and review their accompanying notices separately.

Optional TabPFN weights and downloaded datasets are not covered by the root review
permission. They are not bundled and remain subject to their providers' terms.

## Items to confirm before public release

- Confirm the copyright years to retain for earlier code; 2026 in the root license
  identifies this prepared package and does not establish the creation date of
  every inherited file.
- Confirm any additional contributors to inherited plotting, experimental and
  utility code not covered by the assignments above.
- Confirm the original LightGBM porting revision if recoverable; retain this
  notice even if the exact historical revision remains unknown.
- Complete the baseline paper reference and dataset license/version records.

No public release is asserted by these preparatory files.
