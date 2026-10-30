"""Prepare the paper's BABD input without changing its historical cleaning rules.

Only explicit execution of main() reads or writes datasets. Existing output
folders are refused. See docs/classification/babd_preparation.md.
"""
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd

RAW_SHA256 = '766b5d51af956b1a619775e5522d0c15f0c1c2ef272f87949d8bd1a685f332b6'
CLEAN_SHA256 = 'd380c00f6528f225c6840cb78c07fd28fa6794a98b9ed20a5c3eafcb6109f4e3'
ORIGINAL_SCRIPT_SHA256 = 'a102365e6c532ce33adca342d36b2439e9f7012e5c3cb1cd73b40beadb632c9c'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    data_root = Path(os.environ.get('OEO_DATA_ROOT', str(root / 'data')))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True,
                        help='Unmodified BABD-13.csv from the recorded download')
    parser.add_argument('--output-dir', type=Path,
                        default=data_root / 'raw/public/babd13',
                        help='New directory; existing directories are never overwritten')
    args = parser.parse_args(argv)
    source, output = args.input.resolve(), args.output_dir.resolve()
    if output.exists():
        parser.error(f'Output directory already exists: {output}')
    if not source.is_file():
        parser.error(f'Input file not found: {source}')
    if sha256(source) != RAW_SHA256:
        parser.error('Input checksum differs from the paper input; no output written')

    # Keep the original pandas parsing and source row/column order.
    data = pd.read_csv(source)
    required = {'account', 'SW', 'label'}
    if not required.issubset(data.columns):
        raise ValueError('Missing account, SW or label column')
    features = [c for c in data if c not in ['account', 'SW', 'label']]
    if len(data) != 544462 or len(features) != 148:
        raise ValueError('Unexpected input dimensions')
    if data.isna().any().any() or data['account'].astype(str).str.strip().eq('').any():
        raise ValueError('Missing values or empty addresses; no imputation is performed')
    if not np.isfinite(data[features].to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite feature values; no replacement is performed')
    if set(data.label.unique()) != set(range(13)):
        raise ValueError('Expected original labels 0 through 12')

    # Historical rule 1: remove every row of cross-label addresses.
    conflict = data.groupby('account').label.nunique()
    bad = set(conflict[conflict > 1].index)
    bad_rows = int(data.account.isin(bad).sum())
    clean = data[~data.account.isin(bad)].copy()
    duplicates = clean[clean.account.duplicated(False)]
    differences = duplicates.groupby('account')[features].nunique(dropna=False).max(axis=1)
    different_ids = set(differences[differences > 1].index)

    # Historical rule 2: accept only identical or numerically near-equal
    # feature vectors, comparing each duplicate with its first source row.
    equivalent = all(np.allclose(group[features].to_numpy(dtype=float),
                                group[features].iloc[0].to_numpy(dtype=float),
                                rtol=1e-12, atol=1e-14)
                     for _, group in duplicates.groupby('account'))
    if different_ids and not equivalent:
        raise ValueError('Duplicate features exceed historical tolerance; no output written')

    # Historical rule 3: retain the first row, without sorting or averaging.
    before = len(clean)
    clean = clean.drop_duplicates('account')
    removed = before - len(clean)
    if (len(bad), bad_rows, removed, len(clean), len(different_ids)) != (1414, 3216, 292, 540954, 4):
        raise ValueError('Cleaning counts differ from the historical audit; no output written')
    report = dict(original_rows=len(data), excluded_label_conflict_rows=bad_rows,
                  duplicate_rows_removed=removed, clean_rows=len(clean),
                  classes={str(k): int(v) for k, v in clean.label.value_counts().sort_index().items()},
                  feature_duplicate_policy=(
                      'First source row; differences within rtol=1e-12 atol=1e-14 accepted; '
                      f'{len(different_ids)} near-equal duplicate addresses'),
                  feature_count=len(features), features=features)
    # Recheck the source before creating outputs. Never overwrite an existing run.
    if sha256(source) != RAW_SHA256:
        raise ValueError('Input changed during preparation')
    output.mkdir(parents=True, exist_ok=False)
    csv_path = output / 'BABD-13-clean.csv'
    clean.to_csv(csv_path, index=False, mode='x')
    actual_hash = sha256(csv_path)
    with (output / 'cleaning_report.json').open('x') as stream:
        stream.write(json.dumps(report, indent=2))
    provenance = dict(raw_sha256=RAW_SHA256, clean_sha256=actual_hash,
                      historical_clean_sha256=CLEAN_SHA256,
                      historical_bytes_match=(actual_hash == CLEAN_SHA256),
                      original_script_sha256=ORIGINAL_SCRIPT_SHA256,
                      preparation_script_sha256=sha256(Path(__file__)),
                      python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__)
    with (output / 'preparation_provenance.json').open('x') as stream:
        stream.write(json.dumps(provenance, indent=2))
    print(f'Prepared {len(clean)} rows and {len(features)} features in {output}')
    if actual_hash != CLEAN_SHA256:
        print('WARNING: CSV bytes differ from the historical file. Do not claim an exact replay; '
              'check parsing/serialization versions before running experiments.')
    else:
        print('CSV SHA-256 matches the historical experimental input.')


if __name__ == '__main__':
    main()
