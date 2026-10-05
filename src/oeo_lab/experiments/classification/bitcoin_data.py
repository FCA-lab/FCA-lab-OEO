"""读取 Bitcoin 数据并审核实体标识；保留样本原始顺序，不去重。"""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd


def load_bitcoin_dataset(spec):
    """返回 X、连续整数类别、原类别映射和实体；先过滤范围，再审核。"""
    path = Path(spec['path'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec['sha256']:
        raise ValueError('Dataset changed after task creation')
    frame = pd.read_csv(path, dtype={spec['group_column']: str, spec['address_column']: str})
    target, group, address = (spec[k] for k in ('decision_column','group_column','address_column'))
    features, labels = spec['feature_columns'], spec['class_ids']
    if not labels or len(set(labels)) != len(labels):
        raise ValueError('Class IDs must be nonempty and unique')
    if not features or len(set(features)) != len(features) or set(features) & {target,group,address,'label_cnt','sample_id'}:
        raise ValueError('Invalid or leaking feature selection')
    frame = frame.loc[frame[target].isin(labels)].reset_index(drop=True)
    if frame.empty or set(frame[target].unique()) != set(labels):
        raise ValueError('Missing configured Bitcoin classes')
    for col in [target,group,address]+features:
        if frame[col].isna().any() or frame[col].astype(str).str.strip().eq('').any():
            raise ValueError(f'Missing values in {col}')
    if frame[address].duplicated().any():
        raise ValueError('Duplicate addresses: review data explicitly; no automatic deduplication')
    if frame.groupby(group)[target].nunique().gt(1).any():
        raise ValueError('Entity assigned to multiple classes')
    X = frame[features].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError('Nonfinite Bitcoin features')
    mapping = {label:i for i,label in enumerate(labels)}
    return X, frame[target].map(mapping).to_numpy(dtype=int), labels, frame[group].to_numpy()
