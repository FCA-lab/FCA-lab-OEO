"""Load numerical classification inputs without sampling or deduplication."""
import numpy as np
import pandas as pd

def load_numeric_dataset(spec):
    """读取 UCI 数值数据；不采样、不去重、不分箱。类别编码映射随结果返回。"""
    import hashlib
    from pathlib import Path
    path = Path(spec['path'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec['sha256']:
        raise ValueError('Dataset changed after task creation')
    frame = pd.read_csv(path, header=None, dtype=str)
    if frame.empty or frame.isna().any().any():
        raise ValueError('Empty or missing UCI data')
    labels = sorted(frame[spec['decision_column']].unique())
    mapping = {label: i for i, label in enumerate(labels)}
    y = frame[spec['decision_column']].map(mapping).to_numpy(dtype=int)
    features = frame.drop(columns=[spec['decision_column']] + spec['id_columns'])
    X = features.apply(pd.to_numeric, errors='raise')
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError('Nonfinite numeric features')
    return X, y, labels
